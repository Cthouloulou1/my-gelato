#!/usr/bin/env python3
"""
Best-of-K : le bon rotamere est-il parmi les echantillons du flow ?

Pour chaque residu, genere K predictions (graines differentes) avec un modele
entraine, puis compare plusieurs facons d'en garder une :
    premier      l'echantillon 0 (= ce que donne evaluate_flow.py)
    oracle       le meilleur des K selon la verite (borne haute : le bon
                 rotamere a-t-il ete propose ?)
    densite      celui dont la chaine laterale (atomes au-dela du CB) est dans
                 la densite la plus forte du cube (score z moyen)
    consensus    le medoide : chaine laterale la plus proche des K-1 autres
Pour chaque strategie : chi1 juste, rotamere complet juste, RMSD moyen, < 1 A.
Plus la diversite : part des residus dont les K echantillons proposent au
moins deux cases de chi1 differentes.

Lecture : oracle >> premier -> le modele propose le bon rotamere mais choisit
mal : une selection (densite, consensus) peut aider. oracle ~ premier -> le bon
rotamere n'est pas propose : il faut changer le modele.

Ecrit dans --out-dir :
    best_of_k.csv                     tableau ci-dessus
    <ensemble>_samples.npz            les K echantillons, scores de densite, idx
    <nom>_densite/, <nom>_consensus/  predictions retenues au format de
                                      evaluate_flow.py (<ensemble>_predictions.npz),
                                      utilisables par analyze_flow_geometry.py etc.

Usage :
    python sample_rotamers.py --lmdb-path /data/eyerly/splits_preprocessed.lmdb \\
        --atom14 /data/eyerly/splits_atom14.npz \\
        --splits /data/eyerly/pdb_cryoem_splits/splits.json \\
        --model hybrid_chain_pep=runs/flow_hybrid_chain_pep_e100/best_model.pt \\
        --ca-context-file /data/eyerly/splits_ca_context.npz --k 8
"""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import analyze_flow_geometry as geo
from flow_common import (CA_SLOT, PERM_SYM, FlowDataset, model_context, residue_metrics,
                         sample_atoms, sample_atoms_ar, save_flow_reports, spherical_mask_t)
from flow_models import build_flow_model, sample_features
from train_flow import ar_groups, ar_sigma, get_context

SC_FIRST = 5            # atomes au-dela du CB
STRATEGIES = ["premier", "oracle", "densite", "consensus"]


def density_score(cube, pred, mask, sphere):
    """Score z moyen du cube aux atomes predits au-dela du CB. cube (B,1,20,20,20),
    pred (B,14,3) A, mask (B,14), sphere (20,20,20) voxels pris pour la
    moyenne et l'ecart-type -> (B,)."""
    B = cube.shape[0]
    v = cube[:, 0][:, sphere]
    z = (cube - v.mean(1).view(B, 1, 1, 1, 1)) / v.std(1).clamp(min=1e-6).view(B, 1, 1, 1, 1)
    sc = sample_features(z, pred, (cube.shape[-1] - 1) / 2)[..., 0]   # (B,14)
    m = mask.clone()
    m[:, :SC_FIRST] = False
    m = m.float()
    n = m.sum(1)
    out = (sc * m).sum(1) / n.clamp(min=1)
    return torch.where(n > 0, out, torch.full_like(out, float("nan")))


def best_gt(pred, gt, restype, mask):
    """Numerotation symetrique de la verite la plus proche de pred (comme le RMSD)."""
    m = mask.copy()
    m[:, CA_SLOT] = False
    gsw = np.take_along_axis(gt, PERM_SYM[restype][..., None], 1)
    use = (((pred - gsw) ** 2).sum(-1) * m).sum(1) < (((pred - gt) ** 2).sum(-1) * m).sum(1)
    return np.where(use[:, None, None], gsw, gt)


def evaluate(pred, gt, mask, rt, chi_tol):
    g = best_gt(pred, gt, rt, mask)
    rmsd = residue_metrics(pred, gt, mask, rt)["rmsd"]
    o = geo.residue_analysis(pred.astype(np.float64), g.astype(np.float64), mask, rt, rmsd,
                             SimpleNamespace(chi_tol=chi_tol, frame_tol=30.0, fail_rmsd=1.0))
    return o["dchi"][:, 0] <= chi_tol, o["rotamer_ok"], rmsd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmdb-path", required=True)
    p.add_argument("--atom14", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--model", required=True, help="nom=chemin/best_model.pt")
    p.add_argument("--sets", nargs="*", default=["val", "test_1_2A", "test_2_3A", "test_3_4A"])
    p.add_argument("--k", type=int, default=8, help="echantillons par residu")
    p.add_argument("--max-residues", type=int, default=0,
                   help="sous-ensemble aleatoire par ensemble (0 = tous)")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chi-tol", type=float, default=40.0)
    p.add_argument("--ca-context-file", default=None,
                   help="(modeles --ca-context) remplace le chemin enregistre a l'entrainement")
    p.add_argument("--out-dir", default="runs/rotamer_best_of_k")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    name, path = args.model.split("=", 1)
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    model = build_flow_model(ck["model_type"], a).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    a = dict(a, sample_steps=args.sample_steps, atom14=args.atom14)
    if args.ca_context_file:
        a["ca_context_file"] = args.ca_context_file
    ns = argparse.Namespace(**a)
    ctxdata = get_context(ns) if (getattr(ns, "ca_context", None) or "none") != "none" else None
    mask_t = None if a.get("no_spherical_mask") else spherical_mask_t(20, device)
    sphere = spherical_mask_t(20, device).bool()

    with open(args.splits) as f:
        splits = json.load(f)
    ds = FlowDataset(args.lmdb_path, args.atom14)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    print(f"{name} : {ck['model_type']}, epoch {ck['epoch']}, K = {args.k}, "
          f"{args.sample_steps} pas")

    for s in args.sets:
        idx = ds.indices_for(splits[s])
        if args.max_residues and len(idx) > args.max_residues:
            rng = np.random.default_rng(args.seed)
            idx = sorted(rng.choice(idx, args.max_residues, replace=False).tolist())
        if not idx:
            continue
        loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        samples, scores, gts, masks, rts, iis = [], [], [], [], [], []
        gens = []
        for k in range(args.k):
            g = torch.Generator(device=device)
            g.manual_seed(args.seed + 1000 * k)
            gens.append(g)
        with torch.no_grad():
            for cube, pos, mask, restype, ii in tqdm(loader, leave=False, desc=s):
                cube, mask, restype = cube.to(device), mask.to(device), restype.to(device)
                if mask_t is not None:
                    cube = cube * mask_t
                cc = model_context(ctxdata.batch(ii, device)) if ctxdata is not None else None
                ks, ksc = [], []
                for k in range(args.k):
                    if getattr(ns, "autoregressive", False):
                        pr = sample_atoms_ar(model, cube, restype, mask, args.sample_steps,
                                             ar_sigma(ns), ns.coord_scale, ns.ar_noise_center,
                                             gens[k], ar_groups(ns), ca_ctx=cc)
                    else:
                        pr = sample_atoms(model, cube, restype, mask, args.sample_steps, ns.sigma,
                                          ns.coord_scale, gens[k], ca_ctx=cc)
                    ks.append(pr.cpu().numpy())
                    ksc.append(density_score(cube, pr, mask, sphere).cpu().numpy())
                samples.append(np.stack(ks))
                scores.append(np.stack(ksc))
                gts.append(pos.numpy().copy())
                masks.append(mask.cpu().numpy().copy())
                rts.append(restype.cpu().numpy().copy())
                iis.append(ii.numpy().copy())
        X = np.concatenate(samples, 1)                       # (K,N,14,3)
        D = np.concatenate(scores, 1)                        # (K,N)
        gt, mask, rt, ii = (np.concatenate(gts), np.concatenate(masks), np.concatenate(rts),
                            np.concatenate(iis))
        K, N = X.shape[:2]
        np.savez_compressed(os.path.join(args.out_dir, f"{s}_samples.npz"), samples=X,
                            density=D, gt=gt, mask=mask, restype=rt, idx=ii)

        # evaluation de chaque echantillon
        c1 = np.zeros((K, N), bool)
        rot = np.zeros((K, N), bool)
        rm = np.zeros((K, N))
        bins = np.zeros((K, N), np.int64)
        pi1 = geo.CHI_PI[rt][:, 0]
        for k in range(K):
            c1[k], rot[k], rm[k] = evaluate(X[k], gt, mask, rt, args.chi_tol)
            chi = geo.chis(X[k].astype(np.float64), rt)[:, 0]
            c = np.where(pi1, np.mod(chi, 180.0), np.mod(chi, 360.0))
            bins[k] = np.floor(c / np.where(pi1, 60.0, 120.0)).astype(np.int64).clip(0, 2)
        has = geo.CHI_MASK[rt][:, 0]

        # choix
        sc = mask.copy()
        sc[:, :4] = False                                    # CB et au-dela
        pair = np.zeros((K, K, N))
        for k in range(K):                                   # RMSD chaine laterale entre K
            d2 = ((X[k][None] - X) ** 2).sum(-1)             # (K,N,14)
            pair[k] = np.sqrt((d2 * sc).sum(-1) / np.maximum(sc.sum(-1), 1))
        pick = {"premier": np.zeros(N, np.int64),
                "densite": np.nanargmax(np.where(np.isfinite(D), D, -np.inf), 0),
                "consensus": pair.sum(1).argmin(0)}
        ar = np.arange(N)
        res = {}
        for st in ["premier", "densite", "consensus"]:
            j = pick[st]
            res[st] = (c1[j, ar], rot[j, ar], rm[j, ar])
        res["oracle"] = (c1.any(0), rot.any(0), rm.min(0))
        diverse = np.array([len(set(bins[:, i])) > 1 for i in range(N)])

        print(f"\n=== {s} : {N} residus ({has.sum()} avec chi1), K = {K} ===")
        print(f"{'':>11s}   chi1    rotamere   RMSD moyen   < 1 A")
        for st in STRATEGIES:
            a1, ar_, r = res[st]
            print(f"{st:>11s}  {100 * a1[has].mean():5.1f} %  {100 * ar_[has].mean():5.1f} %"
                  f"     {np.nanmean(r):.3f}     {100 * np.nanmean(r < 1):5.1f} %"
                  + ("   (RMSD : le meilleur des K)" if st == "oracle" else ""))
            rows.append({"modele": name, "ensemble": s, "k": K, "strategie": st, "n": N,
                         "n_chi1": int(has.sum()), "chi1": f"{a1[has].mean():.4f}",
                         "rotamere": f"{ar_[has].mean():.4f}", "rmsd": f"{np.nanmean(r):.4f}",
                         "frac_lt_1A": f"{np.nanmean(r < 1):.4f}"})
        print(f"  diversite : {100 * diverse[has].mean():.1f} % des residus ont >= 2 cases de "
              f"chi1 parmi les K ; chi1 juste dans k/K echantillons en moyenne : "
              f"{100 * c1[:, has].mean():.1f} %")
        rows.append({"modele": name, "ensemble": s, "k": K, "strategie": "diversite_chi1",
                     "n": N, "n_chi1": int(has.sum()), "chi1": f"{diverse[has].mean():.4f}"})

        for st in ("densite", "consensus"):
            j = pick[st]
            save_flow_reports(X[j, ar], gt, mask, rt, os.path.join(args.out_dir, f"{name}_{st}"),
                              s, title=f"{name} ({st}, K={K}) - {s}", idx=ii)
    ds.close()

    if rows:
        out = os.path.join(args.out_dir, "best_of_k.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        print(f"\nEcrit : {out}, {args.out_dir}/<ensemble>_samples.npz, "
              f"{args.out_dir}/{name}_densite|consensus/")


if __name__ == "__main__":
    main()
