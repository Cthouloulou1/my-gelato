#!/usr/bin/env python3
"""
Flow matching des positions atomiques d'un residu a partir du cube de
densite centre sur son C-alpha (connu) et du type d'acide amine (connu).

Formulation (rectified flow / interpolation lineaire) :
    y1 = positions atom14 / coord_scale        (cible, C-alpha = 0 fixe)
    y0 ~ N(0, sigma^2)                         (bruit)
    y_t = (1 - t) y0 + t y1,  t ~ U(0, 1)
    le reseau predit v(y_t, t | densite, type) ~ y1 - y0  (loss MSE)
Generation : integration d'Euler de t = 0 a 1 (--sample-steps pas).

"Loi chimique" : loss geometrique sur l'estimation de la structure finale
    x1_hat = (y_t + (1 - t) v) * coord_scale
qui penalise l'ecart (Huber, en A) entre les distances 1-2 (liaisons) et 1-3
(qui fixent les angles de liaison) de x1_hat et celles de la vraie
structure, ponderee par t (une estimation a t ~ 0 n'a pas de sens).
Desactivable avec --geom-weight 0 pour mesurer son apport. Les deux reseaux
recoivent en plus la topologie des liaisons (cf. flow_models.py).

Reseaux (--model) : cnn (CNN 3D + transformer, augmentation par rotations
conjointes densite + atomes) ou equivariant (e3nn, SE(3)-equivariant).

A chaque epoch : loss train/val (flow + geometrie) et, sur un sous-ensemble
fixe du val, generation complete -> RMSD tous atomes / backbone / chaine
laterale, fraction de residus a RMSD < 1 A, erreur sur les liaisons.
Meilleur checkpoint = RMSD moyen val le plus bas.

Usage :
    python train_flow.py --model cnn \
        --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
        --atom14 /data/eyerly/splits_atom14.npz \
        --split-file /data/eyerly/pdb_cryoem_splits/splits.json \
        --epochs 30 --out-dir runs/flow_cnn
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from flow_common import (
    FlowDataset, coord_rotation_from_grid, pair_distances, pair_tensors, predicted_mask,
    rotate_coords, sample_atoms, sample_noise, save_flow_reports, residue_metrics, aggregate,
    spherical_mask_t,
)
from flow_models import build_flow_model
from train_utils import MetricsHistory

FLOW_CURVES = [
    ("loss", "Loss totale"),
    ("loss_flow", "Loss flow matching"),
    ("loss_geom", "Loss geometrique (A)"),
    ("rmsd_mean", "RMSD moyen (A)"),
    ("rmsd_sidechain_mean", "RMSD chaine laterale (A)"),
    ("frac_rmsd_lt_1A", "Fraction RMSD < 1 A"),
    ("bond_err_mean", "Erreur longueurs de liaison (A)"),
]
LOWER_BETTER = ("loss", "loss_flow", "loss_geom", "rmsd_mean", "rmsd_sidechain_mean",
                "bond_err_mean")


def flow_losses(model, batch, args, device, augment=False, generator=None, mask_t=None):
    cube, pos, mask, restype, _ = batch
    cube, pos, mask, restype = (cube.to(device), pos.to(device), mask.to(device),
                                restype.to(device))
    if mask_t is not None:
        cube = cube * mask_t
    if augment:
        from train_cnn_baseline import random_rotation_matrices, rotate_batch_continuous
        R = random_rotation_matrices(cube.shape[0], device)
        cube = rotate_batch_continuous(cube, R)
        if mask_t is not None:
            cube = cube * mask_t
        pos = rotate_coords(pos, coord_rotation_from_grid(R))
    B = cube.shape[0]
    pm = predicted_mask(mask)
    y1 = pos / args.coord_scale * pm.unsqueeze(-1)
    y0 = sample_noise(mask, args.sigma, generator)
    t = torch.rand(B, device=device, generator=generator)
    tt = t[:, None, None]
    y_t = (1 - tt) * y0 + tt * y1
    v = model(cube, y_t, t, restype, mask)

    w = pm.float()
    loss_flow = (((v - (y1 - y0)) ** 2).sum(-1) * w).sum() / w.sum().clamp(min=1)

    x1_hat = (y_t + (1 - tt) * v) * args.coord_scale
    idx, _, valid = pair_tensors(restype, mask)
    err = pair_distances(x1_hat, idx) - pair_distances(pos, idx)
    hub = F.huber_loss(err, torch.zeros_like(err), reduction="none", delta=0.5)
    vw = valid.float() * t[:, None]
    loss_geom = (hub * vw).sum() / valid.float().sum().clamp(min=1)
    return loss_flow, loss_geom


def run_epoch(model, loader, args, device, optimizer=None, augment=False, seed=None,
              mask_t=None):
    train = optimizer is not None
    model.train(train)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    tot = {"loss": 0.0, "loss_flow": 0.0, "loss_geom": 0.0}
    n = 0
    for batch in tqdm(loader, leave=False):
        with torch.set_grad_enabled(train):
            lf, lg = flow_losses(model, batch, args, device, augment, gen, mask_t)
            loss = lf + args.geom_weight * lg
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        bs = batch[0].shape[0]
        n += bs
        tot["loss"] += loss.item() * bs
        tot["loss_flow"] += lf.item() * bs
        tot["loss_geom"] += lg.item() * bs
    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def generate(model, loader, args, device, seed=0, mask_t=None, desc=None):
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    preds, gts, masks, rts, idxs = [], [], [], [], []
    for cube, pos, mask, restype, idx in tqdm(loader, leave=False, desc=desc):
        cube, mask, restype = cube.to(device), mask.to(device), restype.to(device)
        if mask_t is not None:
            cube = cube * mask_t
        pred = sample_atoms(model, cube, restype, mask, args.sample_steps, args.sigma,
                            args.coord_scale, gen)
        preds.append(pred.cpu().numpy())
        gts.append(pos.numpy())
        masks.append(mask.cpu().numpy())
        rts.append(restype.cpu().numpy())
        idxs.append(idx.numpy())
    return (np.concatenate(preds), np.concatenate(gts), np.concatenate(masks),
            np.concatenate(rts), np.concatenate(idxs))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cnn", "equivariant"], required=True)
    p.add_argument("--lmdb-path", required=True)
    p.add_argument("--atom14", required=True, help="npz produit par extract_atom14.py")
    p.add_argument("--split-file", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--coord-scale", type=float, default=4.0,
                   help="A par unite normalisee (atomes a <= ~8 A du C-alpha)")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="ecart-type du bruit initial (unites normalisees)")
    p.add_argument("--geom-weight", type=float, default=1.0,
                   help="poids de la loss geometrique (0 = desactivee)")
    p.add_argument("--sample-steps", type=int, default=20,
                   help="pas d'Euler pour la generation pendant l'entrainement")
    p.add_argument("--val-generate", type=int, default=2048,
                   help="nb de residus val generes a chaque epoch (0 = aucun)")
    p.add_argument("--no-augment", action="store_true",
                   help="(cnn) pas de rotations aleatoires conjointes")
    p.add_argument("--no-spherical-mask", action="store_true")
    # cnn
    p.add_argument("--channels", type=int, nargs=3, default=[32, 64, 128])
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    # equivariant
    p.add_argument("--n-density", type=int, default=256)
    p.add_argument("--k-neighbors", type=int, default=16)
    p.add_argument("--max-radius", type=float, default=8.0)
    p.add_argument("--num-basis", type=int, default=10)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--eq-channels", type=int, nargs=3, default=[32, 32, 32])
    p.add_argument("--activation", default="silu", choices=["sigmoid", "tanh", "silu"])
    # divers
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help="defaut : runs/flow_<model>")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out_dir = args.out_dir or f"runs/flow_{args.model}"
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device : {device}")

    import json
    with open(args.split_file) as f:
        split = json.load(f)
    ds = FlowDataset(args.lmdb_path, args.atom14)
    tr_idx = ds.indices_for(split["train"])
    va_idx = ds.indices_for(split["val"])
    print(f"Residus train : {len(tr_idx)}  |  val : {len(va_idx)}")
    rng = np.random.default_rng(0)
    va_gen_idx = (sorted(rng.choice(va_idx, min(args.val_generate, len(va_idx)), replace=False))
                  if args.val_generate > 0 else [])

    mk = dict(num_workers=args.num_workers, batch_size=args.batch_size)
    train_loader = DataLoader(Subset(ds, tr_idx), shuffle=True, drop_last=True, **mk)
    val_loader = DataLoader(Subset(ds, va_idx), shuffle=False, **mk)
    gen_loader = DataLoader(Subset(ds, va_gen_idx), shuffle=False, **mk) if va_gen_idx else None

    model = build_flow_model(args.model, vars(args)).to(device)
    print(f"Parametres entrainables : {sum(q.numel() for q in model.parameters()):,}")
    mask_t = None if args.no_spherical_mask else spherical_mask_t(20, device)
    augment = args.model == "cnn" and not args.no_augment
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = MetricsHistory(args.out_dir, title=f"Flow matching ({args.model})",
                             curves=FLOW_CURVES, lower_better=LOWER_BETTER)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, args, device, optimizer, augment, mask_t=mask_t)
        va = run_epoch(model, val_loader, args, device, seed=1234, mask_t=mask_t)
        if gen_loader is not None:
            pred, gt, m, rt, _ = generate(model, gen_loader, args, device, mask_t=mask_t)
            va.update(aggregate(residue_metrics(pred, gt, m, rt)))
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.log(epoch, tr, va, lr=lr)
        msg = (f"Epoch {epoch:03d} | train loss={tr['loss']:.4f} (flow {tr['loss_flow']:.4f}, "
               f"geom {tr['loss_geom']:.4f}) | val loss={va['loss']:.4f}")
        if "rmsd_mean" in va:
            msg += (f" RMSD={va['rmsd_mean']:.3f} A (sc {va['rmsd_sidechain_mean']:.3f}) "
                    f"<1A={va['frac_rmsd_lt_1A']:.3f} liaisons={va['bond_err_mean']:.3f} A")
        print(msg)
        score = va.get("rmsd_mean", va["loss"])
        if score < best:
            best = score
            torch.save({"model_type": args.model, "model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_metrics": va, "args": vars(args),
                        "train_pdb_ids": sorted(set(split["train"])),
                        "val_pdb_ids": sorted(set(split["val"]))}, ckpt_path)
            print(f"  -> nouveau meilleur modele ({ckpt_path})")

    print("\n=== Entrainement termine ===")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    pred, gt, m, rt, _ = generate(model, val_loader, args, device, mask_t=mask_t,
                                  desc="generation val complete")
    agg = save_flow_reports(pred, gt, m, rt, args.out_dir, "val_best",
                            title=f"Flow {args.model} - val (epoch {ck['epoch']})",
                            extra={"best_epoch": ck["epoch"], "sample_steps": args.sample_steps})
    print(f"Val (meilleur checkpoint, epoch {ck['epoch']}) : RMSD moyen {agg['rmsd_mean']:.3f} A, "
          f"median {agg['rmsd_median']:.3f} A, chaine laterale {agg['rmsd_sidechain_mean']:.3f} A, "
          f"<1A {agg['frac_rmsd_lt_1A']:.3f}, liaisons {agg['bond_err_mean']:.3f} A")


if __name__ == "__main__":
    main()
