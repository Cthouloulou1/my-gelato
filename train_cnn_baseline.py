#!/usr/bin/env python3
"""
Baseline : CNN 3D standard (non equivariant) sur les cubes de densite
pretraites (20^3 voxels a 1 A, z-score local), pour comparaison avec le
modele equivariant SO(3) (train_equivariant_cnn.py).

Memes donnees, meme split par pdb_id, memes poids de classes, meme
optimiseur / scheduler / critere de selection (macro-F1 val) et memes
sorties (historique par epoch, matrice de confusion, ROC) que le modele
equivariant, pour que la comparaison ne porte que sur l'architecture.

Augmentation (--augment) :
  none        : aucune.
  octahedral  : une des 24 rotations propres du cube (groupe octaedrique,
                det=+1, pas de reflexion), exacte : permutation d'axes +
                retournements, sans interpolation.
  continuous  : rotation SO(3) uniforme quelconque, appliquee sur GPU par
                interpolation trilineaire (affine_grid / grid_sample).
Dans tous les cas, aucune reflexion n'est utilisee (chiralite L).

Masque spherique (--spherical-mask, active par defaut) : les voxels a plus
de (cube_size-1)/2 voxels du centre sont mis a 0 (= moyenne du cube apres
z-score), en entrainement ET en evaluation. Sans ce masque, une rotation
continue ferait entrer dans le cube des coins vides (remplis de 0) dont
la position trahirait l'orientation ; avec le masque, le contenu visible
est le meme quelle que soit la rotation. Desactivable avec
--no-spherical-mask (l'option est sauvegardee dans le checkpoint).

Dependances :
    pip install torch numpy scikit-learn lmdb tqdm matplotlib

Usage :
    python train_cnn_baseline.py \
        --lmdb-path ca_density_dataset_preprocessed.lmdb \
        --split-file splits.json --augment continuous \
        --epochs 50 --out-dir runs/cnn_continuous

Sorties : identiques a train_equivariant_cnn.py (history.csv/json,
training_curves.png, best_model.pt, val_best_*).
"""

import argparse
import itertools
import os
import pickle
import random

import lmdb
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_utils import (
    RESIDUE_CLASSES, N_CLASSES, MetricsHistory, compute_class_weights,
    epoch_metrics, save_final_reports, split_by_pdb_id,
)


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def _perm_sign(p):
    sign = 1
    p = list(p)
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            sign = -sign
    return sign


# 24 rotations propres du cube : (permutation d'axes, axes retournes) tels
# que det = signe(permutation) * (-1)^(nb de retournements) = +1.
OCTAHEDRAL_OPS = [
    (perm, flips)
    for perm in itertools.permutations(range(3))
    for k in range(4)
    for flips in itertools.combinations(range(3), k)
    if _perm_sign(perm) * (-1) ** len(flips) == 1
]
assert len(OCTAHEDRAL_OPS) == 24


def apply_octahedral(cube, op):
    perm, flips = op
    out = np.transpose(cube, perm)
    if flips:
        out = np.flip(out, axis=flips)
    return np.ascontiguousarray(out)


def random_rotation_matrices(n, device, generator=None):
    """n matrices de rotation uniformes (Haar), det=+1, via QR."""
    A = torch.randn(n, 3, 3, device=device, generator=generator)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R, dim1=1, dim2=2))
    d[d == 0] = 1
    Q = Q * d.unsqueeze(1)
    neg = torch.linalg.det(Q) < 0
    Q[neg, :, 0] = -Q[neg, :, 0]
    return Q


def rotate_batch_continuous(x, R):
    """x : (B,1,D,H,W), R : (B,3,3). Rotation autour du centre du cube par
    interpolation trilineaire, hors-cube rempli a 0. Une permutation fixe
    des axes (conjugaison) garde det=+1 : aucune reflexion introduite."""
    B = x.shape[0]
    theta = torch.zeros(B, 3, 4, device=x.device, dtype=x.dtype)
    theta[:, :, :3] = R.to(x.dtype)
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=True)


def spherical_mask(size):
    c = (size - 1) / 2.0
    g = np.arange(size) - c
    r2 = g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2
    return (r2 <= c ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CubeDataset(Dataset):
    """Lit le LMDB pretraite et renvoie (cube (1,D,H,W), label)."""

    def __init__(self, lmdb_path, mask=True):
        self.lmdb_path = lmdb_path
        self.env = None
        self.mask_enabled = mask
        self.mask = None
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.keys, self.pdb_ids, self.labels = [], [], []
        cube_size = None
        with env.begin() as txn:
            for key, value in txn.cursor():
                if key in (b"__len__", b"__meta__", b"__processed_pdb_ids__"):
                    continue
                sample = pickle.loads(value)
                self.keys.append(key)
                self.pdb_ids.append(sample["pdb_id"])
                self.labels.append(sample["class_idx"])
                if cube_size is None:
                    cube_size = sample["cube"].shape[0]
        env.close()
        self.cube_size = cube_size
        if mask and cube_size:
            self.mask = spherical_mask(cube_size)

    def _ensure_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                 readahead=False)

    def __len__(self):
        return len(self.keys)

    def get_item(self, idx, augment="none"):
        self._ensure_env()
        with self.env.begin() as txn:
            sample = pickle.loads(txn.get(self.keys[idx]))
        cube = np.nan_to_num(sample["cube"].astype(np.float32))
        if self.mask is not None:
            cube = cube * self.mask
        if augment == "octahedral":
            cube = apply_octahedral(cube, OCTAHEDRAL_OPS[np.random.randint(24)])
        return torch.from_numpy(cube).unsqueeze(0), sample["class_idx"]

    def __getitem__(self, idx):
        return self.get_item(idx)


class CubeView(Dataset):
    def __init__(self, base, indices, augment="none"):
        self.base, self.indices, self.augment = base, indices, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # la rotation continue est faite sur GPU, par batch (voir run_epoch)
        aug = "octahedral" if self.augment == "octahedral" else "none"
        return self.base.get_item(self.indices[i], aug)


# ---------------------------------------------------------------------------
# Modele
# ---------------------------------------------------------------------------

def conv_block(cin, cout):
    return tnn.Sequential(
        tnn.Conv3d(cin, cout, 3, padding=1, bias=False),
        tnn.BatchNorm3d(cout), tnn.ReLU(inplace=True),
        tnn.Conv3d(cout, cout, 3, padding=1, bias=False),
        tnn.BatchNorm3d(cout), tnn.ReLU(inplace=True),
    )


class CNN3DClassifier(tnn.Module):
    """CNN 3D type VGG : 3 blocs (2 conv 3x3x3 + BN + ReLU), max-pool entre
    les blocs (20 -> 10 -> 5), puis global average + max pooling et tete
    MLP. Avec channels=(16, 32, 64) : ~0.2 M parametres."""

    def __init__(self, n_classes=N_CLASSES, channels=(16, 32, 64), dropout=0.3):
        super().__init__()
        c1, c2, c3 = channels
        self.features = tnn.Sequential(
            conv_block(1, c1), tnn.MaxPool3d(2),
            conv_block(c1, c2), tnn.MaxPool3d(2),
            conv_block(c2, c3),
        )
        self.head = tnn.Sequential(
            tnn.Dropout(dropout),
            tnn.Linear(2 * c3, 128), tnn.ReLU(inplace=True),
            tnn.Dropout(dropout),
            tnn.Linear(128, n_classes),
        )

    def forward(self, x):
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=(2, 3, 4)), h.amax(dim=(2, 3, 4))], dim=1)
        return self.head(pooled)


def build_cnn_from_args(a):
    return CNN3DClassifier(n_classes=N_CLASSES, channels=tuple(a["channels"]),
                           dropout=a["dropout"])


# ---------------------------------------------------------------------------
# Entrainement
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, criterion, optimizer=None,
              continuous_aug=False, mask=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if continuous_aug:
            R = random_rotation_matrices(x.shape[0], device)
            x = rotate_batch_continuous(x, R)
            if mask is not None:
                x = x * mask
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        all_probs.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
        all_targets.append(y.cpu().numpy())
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, N_CLASSES))
    targets = np.concatenate(all_targets) if all_targets else np.zeros(0, dtype=int)
    return epoch_metrics(targets, probs, total_loss / max(n, 1)), targets, probs


@torch.no_grad()
def rotation_sensitivity(model, x, device, mask=None):
    """Diagnostic : difference relative des logits apres une rotation
    continue aleatoire de l'entree. Pour un CNN standard elle n'est PAS
    nulle (contrairement au modele equivariant) ; l'augmentation la reduit."""
    model.eval()
    x = x.to(device)
    out = model(x)
    xr = rotate_batch_continuous(x, random_rotation_matrices(x.shape[0], device))
    if mask is not None:
        xr = xr * mask
    out_r = model(xr)
    return ((out - out_r).norm() / (out.norm() + 1e-12)).item()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmdb-path", required=True)
    p.add_argument("--split-file", default=None)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--augment", default="continuous",
                   choices=["none", "octahedral", "continuous"])
    p.add_argument("--no-spherical-mask", action="store_true")
    p.add_argument("--channels", type=int, nargs=3, default=[16, 32, 64])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None,
                   help="defaut : runs/cnn_<augment>")
    p.add_argument("--checkpoint", default=None,
                   help="defaut : <out-dir>/best_model.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out_dir = args.out_dir or f"runs/cnn_{args.augment}"
    args.checkpoint = args.checkpoint or os.path.join(args.out_dir, "best_model.pt")
    args.spherical_mask = not args.no_spherical_mask
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("Chargement du dataset...")
    ds = CubeDataset(args.lmdb_path, mask=args.spherical_mask)
    train_idx, val_idx, train_ids, val_ids = split_by_pdb_id(
        ds, args.val_fraction, args.seed, args.split_file)
    print(f"Entrees train: {len(train_ids)}  |  Entrees val: {len(val_ids)}")
    print(f"Echantillons train: {len(train_idx)}  |  Echantillons val: {len(val_idx)}")
    print(f"Cube : {ds.cube_size}^3  |  augmentation : {args.augment}  |  "
          f"masque spherique : {args.spherical_mask}")

    train_loader = DataLoader(CubeView(ds, train_idx, args.augment),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(CubeView(ds, val_idx, "none"),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    class_weights = compute_class_weights([ds.labels[i] for i in train_idx], N_CLASSES)
    criterion = tnn.CrossEntropyLoss(weight=class_weights.to(device))

    model = CNN3DClassifier(N_CLASSES, tuple(args.channels), args.dropout).to(device)
    print(f"Parametres entrainables : "
          f"{sum(q.numel() for q in model.parameters() if q.requires_grad):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    mask_t = (torch.from_numpy(ds.mask).to(device) if ds.mask is not None else None)
    continuous = args.augment == "continuous"
    history = MetricsHistory(args.out_dir, title=f"CNN 3D baseline (augment={args.augment})")
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        tr, _, _ = run_epoch(model, train_loader, device, criterion, optimizer,
                             continuous_aug=continuous, mask=mask_t)
        va, _, _ = run_epoch(model, val_loader, device, criterion)
        history.log(epoch, tr, va, lr=optimizer.param_groups[0]["lr"])
        scheduler.step(va["f1_macro"])
        print(f"Epoch {epoch:03d} | train loss={tr['loss']:.4f} acc={tr['accuracy']:.3f} | "
              f"val loss={va['loss']:.4f} acc={va['accuracy']:.3f} top3={va['top3_acc']:.3f} "
              f"P={va['precision_macro']:.3f} F1={va['f1_macro']:.3f} "
              f"AUC={va.get('auc_macro') or float('nan'):.3f}")
        if va["f1_macro"] > best:
            best = va["f1_macro"]
            torch.save({
                "model_type": "cnn3d",
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": va,
                "residue_classes": RESIDUE_CLASSES,
                "train_pdb_ids": train_ids,
                "val_pdb_ids": val_ids,
                "train_idx": train_idx,
                "args": vars(args),
            }, args.checkpoint)
            print(f"  -> nouveau meilleur modele sauvegarde ({args.checkpoint})")

    print("\n=== Entrainement termine ===")
    print(f"Meilleure macro-F1 de validation : {best:.4f}")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    x0, _ = next(iter(val_loader))
    print(f"Sensibilite a la rotation (diff. relative des logits) : "
          f"{rotation_sensitivity(model, x0, device, mask_t):.3f}")
    _, vt, vp = run_epoch(model, val_loader, device, criterion)
    final = save_final_reports(
        vt, vp, args.out_dir, "val_best",
        title=f"CNN 3D ({args.augment}) - val (meilleure epoch {ck['epoch']})",
        extra={"best_epoch": ck["epoch"]})
    print(f"Rapports val ecrits dans {args.out_dir} : acc={final['accuracy']:.3f} "
          f"F1={final['f1_macro']:.3f} AUC macro={final['auc_macro']}")


if __name__ == "__main__":
    main()
