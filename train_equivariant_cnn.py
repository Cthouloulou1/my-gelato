#!/usr/bin/env python3
"""
Train an amino acid classifier (cryo-EM density cubes centered on C-alpha).

Equivariance group : SO(3) (proper rotations only, no reflection), because all amino acids are of L chirality.

Dependances :
    pip install torch e3nn numpy scipy scikit-learn lmdb tqdm matplotlib

Usage :
    python train_equivariant_cnn.py \
        --lmdb-path ca_density_dataset_preprocessed.lmdb \
        --split-file splits.json --epochs 50 --batch-size 32 \
        --out-dir runs/equivariant

Sorties (dans --out-dir) :
    history.csv / history.json   metriques train+val par epoch (loss, accuracy,
                                 top3, precision/recall/F1 macro, AUC macro/micro)
    training_curves.png          courbes correspondantes (mises a jour a chaque epoch)
    best_model.pt                meilleur checkpoint (macro-F1 val)
    val_best_*                   matrice de confusion, ROC, AUC par classe,
                                 rapport par classe du meilleur checkpoint sur le val
"""

import argparse
import os
import pickle
import random

import lmdb
import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from e3nn import o3
from e3nn.nn import NormActivation, BatchNorm, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace

from train_utils import (
    RESIDUE_CLASSES, N_CLASSES, MetricsHistory, compute_class_weights,
    epoch_metrics, save_final_reports, split_by_pdb_id,
)



# ---------------------------------------------------------------------------
# Cube -> point cloud (real physical coordinates, in Angstrom)
# ---------------------------------------------------------------------------

def cube_to_point_cloud(cube, voxel_spacing, n_points, jitter_std=0.01, rng=None):
    """
    Select the n most dense voxels and return their real position (Angstrom, relative to the cube center) and their density value.

    A continuous noise (jitter_std, in Angstrom, default 0.01 is added to the positions. 
    This is NOT for regularization: 
    The positions from a regular voxel grid contain many pairs of points at EXACTLY equal distance (axial neighbors, diagonals, etc.), 
    The choice of the k nearest neighbors must then break ties, which restores equivariance to machine precision (~1e-7), verified empirically.
    
    """

    D, H, W = cube.shape
    assert n_points <= D * H * W, "n_points should be less than the number of voxels in the cube"
    rng = rng or np.random

    center = np.array([(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0])
    flat = cube.reshape(-1)

    top_idx = np.argpartition(-flat, n_points - 1)[:n_points]
    coords_flat = np.stack(np.unravel_index(top_idx, cube.shape), axis=1).astype(np.float64)

    rel_pos = (coords_flat - center) * np.asarray(voxel_spacing, dtype=np.float64)
    if jitter_std > 0:
        rel_pos = rel_pos + rng.normal(0.0, jitter_std, size=rel_pos.shape)
    values = flat[top_idx]

    return rel_pos.astype(np.float32), values.astype(np.float32)


def random_rotation_matrix(rng=None):
    """
    Uniform 3D rotation matrix (proper rotation, no reflection, consistent with the chirality of amino acids), 
    via QR decomposition of a Gaussian matrix,
    with sign correction to ensure det=+1.
    """
    rng = rng or np.random
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset LMDB -> point cloud
# ---------------------------------------------------------------------------

class CAPointCloudDataset(Dataset):
    """
    Dataset reading the preprocessed LMDB (see preprocess_lmdb.py) and converting each cube into a point cloud on the fly.
    Each item: pos (n_points,3) in Angstrom, feat (n_points,1), label.
"""

    def __init__(self, lmdb_path, n_points=128, augment=False, jitter_std=0.01):
        self.lmdb_path = lmdb_path
        self.n_points = n_points
        self.augment = augment
        self.jitter_std = jitter_std
        self.env = None  # lazily open (multiprocessing DataLoader)

        # First pass (keys + pdb_id + label) for the split.
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.keys = []
        self.pdb_ids = []
        self.labels = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if key in (b"__len__", b"__meta__", b"__processed_pdb_ids__"):
                    continue
                sample = pickle.loads(value)
                self.keys.append(key)
                self.pdb_ids.append(sample["pdb_id"])
                self.labels.append(sample["class_idx"])
        env.close()

    def _ensure_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                  readahead=False)

    def __len__(self):
        return len(self.keys)

    def get_item(self, idx, augment):
        self._ensure_env()
        key = self.keys[idx]
        with self.env.begin() as txn:
            sample = pickle.loads(txn.get(key))

        cube = sample["cube"].astype(np.float32)
        spacing = sample["voxel_spacing"]
        label = sample["class_idx"]

        pos, feat = cube_to_point_cloud(cube, spacing, self.n_points, jitter_std=self.jitter_std)

        r = np.linalg.norm(pos, axis=1, keepdims=True).astype(np.float32)   # (n,1)
        feat = np.concatenate([feat[:, None], r], axis=1)                   # (n,2)

        if augment:
            R = random_rotation_matrix()
            pos = pos @ R.T

        pos_t = torch.from_numpy(np.ascontiguousarray(pos))
        feat_t = torch.from_numpy(np.ascontiguousarray(feat))
        return pos_t, feat_t, label

    def __getitem__(self, idx):
        return self.get_item(idx, self.augment)


class AugmentedView(Dataset):
    """View of a shared CAPointCloudDataset (same LMDB environment), restricted to a subset of indices with its own augmentation flag.
    Allows having a train split (augmented) and a val split (non-augmented) without opening the same LMDB twice."""

    def __init__(self, base_dataset, indices, augment):
        self.base = base_dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base.get_item(self.indices[i], self.augment)


# ---------------------------------------------------------------------------
# K-NN graph by batch (disjoint batch, no leak between samples)
# ---------------------------------------------------------------------------

def knn_graph_batched(pos, k):
    """pos: (B,N,3) -> (src, dst), chacun (B*N*k,), index dans le tenseur
    aplati (B*N,...). Le graphe est recalcule depuis les positions REELLES
    passees en argument : si pos a ete tourne, le graphe (distances comme
    directions) en tient compte automatiquement."""
    B, N, _ = pos.shape
    dist = torch.cdist(pos, pos)
    dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    k_eff = min(k, N - 1)
    knn_idx = dist.topk(k_eff, largest=False).indices  # (B,N,k)

    device = pos.device
    batch_offset = (torch.arange(B, device=device) * N).view(B, 1, 1)
    src = (knn_idx + batch_offset).reshape(-1)
    dst = (torch.arange(N, device=device).view(1, N, 1) + batch_offset).expand(B, N, k_eff).reshape(-1)
    return src, dst, k_eff


# ---------------------------------------------------------------------------
# Couche de convolution equivariante (message passing + produit tensoriel)
# ---------------------------------------------------------------------------

class PointConvLayer(tnn.Module):
    """Une couche de convolution equivariante SO(3) sur un graphe de
    points :
      1. Pour chaque arete (voisin -> point), combine la feature du
         voisin avec l'harmonique spherique de la direction de l'arete,
         via un produit tensoriel dont le poids depend UNIQUEMENT de la
         distance (jamais de la direction absolue) -- c'est cette
         separation direction/distance qui garantit l'equivariance.
      2. Agrege (somme) les messages de tous les voisins vers chaque
         point, normalise par sqrt(k).
      3. BatchNorm equivariante : renormalise la norme de chaque irrep
         (jamais sa direction), empeche le signal de s'effondrer ou
         d'exploser a travers les couches.
    """

    def __init__(self, irreps_in, irreps_out, irreps_sh, num_basis=8, radial_hidden=16):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = irreps_sh

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False
        )
        self.radial = FullyConnectedNet(
            [num_basis, radial_hidden, self.tp.weight_numel],
            act=torch.nn.functional.silu,
        )
        self.bn = BatchNorm(self.irreps_out)

    def forward(self, node_feat, src, dst, edge_sh, edge_embed, num_nodes, k):
        weights = self.radial(edge_embed)
        messages = self.tp(node_feat[src], edge_sh, weights)

        out = torch.zeros(num_nodes, self.irreps_out.dim,
                           device=messages.device, dtype=messages.dtype)
        out.index_add_(0, dst, messages)
        out = out / (k ** 0.5)
        return self.bn(out)


# ---------------------------------------------------------------------------
# Modele complet
# ---------------------------------------------------------------------------

class EquivariantPointClassifier(tnn.Module):
    def __init__(self, n_classes=N_CLASSES, channels=(16, 16, 32), lmax=2,
                 k_neighbors=16, num_basis=8, max_radius=6.0, dropout=0.3,
                 activation="sigmoid"):
        super().__init__()
        act_fn = {"sigmoid": torch.sigmoid, "tanh": torch.tanh,
                  "silu": torch.nn.functional.silu}[activation]
        self.k_neighbors = k_neighbors
        self.num_basis = num_basis
        self.max_radius = max_radius
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
 
        irreps_in = o3.Irreps("2x0e")  # densite scalaire par point
        irreps1 = o3.Irreps(f"{channels[0]}x0e + {channels[0]//2}x1o + {channels[0]//4}x2e")
        irreps2 = o3.Irreps(f"{channels[1]}x0e + {channels[1]//2}x1o + {channels[1]//4}x2e")
        irreps3 = o3.Irreps(f"{channels[2]}x0e + {channels[2]//2}x1o + {channels[2]//4}x2e")

        self.conv1 = PointConvLayer(irreps_in, irreps1, self.irreps_sh, num_basis)
        self.conv2 = PointConvLayer(irreps1, irreps2, self.irreps_sh, num_basis)
        self.conv3 = PointConvLayer(irreps2, irreps3, self.irreps_sh, num_basis)

        # NormActivation : non-linearite appliquee a la NORME de chaque
        # irrep (invariante par rotation), la direction est preservee.
        self.act1 = NormActivation(irreps1, act_fn)
        self.act2 = NormActivation(irreps2, act_fn)
        self.act3 = NormActivation(irreps3, act_fn)

        self.irreps3 = irreps3
        n_scalars = sum(mul for mul, ir in irreps3 if ir.l == 0)

        self.head = tnn.Sequential(
            tnn.Dropout(dropout),
            tnn.Linear(2 * n_scalars, 64), tnn.ReLU(),
            tnn.Dropout(dropout),
            tnn.Linear(64, n_classes),
        )

    def _extract_scalars(self, h):
        offset = 0
        parts = []
        for mul, ir in self.irreps3:
            width = mul * ir.dim
            if ir.l == 0:
                parts.append(h[..., offset:offset + width])
            offset += width
        return torch.cat(parts, dim=-1)

    def forward(self, pos, feat):
        # pos: (B,N,3) positions reelles (Angstrom) ; feat: (B,N,1) densite
        B, N, _ = pos.shape

        # Graphe + harmoniques spheriques RECALCULES a chaque appel, a
        # partir des positions reelles passees en argument (potentiellement
        # tournees par l'augmentation) -- c'est le point cle qui garantit
        # l'equivariance SO(3) reelle (cf. docstring du module).
        src, dst, k_eff = knn_graph_batched(pos, self.k_neighbors)
        pos_flat = pos.reshape(B * N, 3)
        feat_flat = feat.reshape(B * N, -1)

        edge_vec = pos_flat[dst] - pos_flat[src]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec,
                                          normalize=True, normalization="component")
        edge_len = edge_vec.norm(dim=-1)
        edge_embed = soft_one_hot_linspace(
            edge_len, start=0.0, end=self.max_radius,
            number=self.num_basis, basis="smooth_finite", cutoff=True,
        )

        num_nodes = B * N
        h1 = self.act1(self.conv1(feat_flat, src, dst, edge_sh, edge_embed, num_nodes, k_eff))
        h2 = self.act2(self.conv2(h1, src, dst, edge_sh, edge_embed, num_nodes, k_eff))
        h3 = self.act3(self.conv3(h2, src, dst, edge_sh, edge_embed, num_nodes, k_eff))

        scalars = self._extract_scalars(h3).reshape(B, N, -1)
        # Moyenne + max sur les points : invariant par rotation (herite des
        # scalaires l=0) ET par permutation (nuage de points non ordonne).
        mean = scalars.mean(dim=1)
        maximum = scalars.max(dim=1).values
        return self.head(torch.cat([mean, maximum], dim=-1))


# ---------------------------------------------------------------------------
# Verification empirique de l'equivariance (sanity check rapide)
# ---------------------------------------------------------------------------

def check_equivariance(model, pos, feat, device):
    """Compare la sortie du modele avant/apres une rotation continue
    quelconque appliquee aux positions. Pour un modele correctement
    equivariant (et avec une tete de classification qui ne regarde que
    des features invariantes), la difference doit etre de l'ordre de la
    precision numerique (~1e-5 a 1e-7), peu importe l'angle."""
    model.eval()
    with torch.no_grad():
        pos = pos.to(device)
        feat = feat.to(device)
        out_ref = model(pos, feat)

        R = torch.from_numpy(random_rotation_matrix()).to(device)
        pos_rot = pos @ R.T
        out_rot = model(pos_rot, feat)

        rel_diff = (out_ref - out_rot).norm().item() / (out_ref.norm().item() + 1e-12)
    model.train()
    return rel_diff


# ---------------------------------------------------------------------------
# Boucle d'entrainement
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, criterion, optimizer=None):
    """Une passe sur `loader`. Retourne (metriques, targets, probs) ; les
    metriques incluent loss, accuracy, top3, precision/recall/F1 macro et
    AUC macro/micro (cf. train_utils.epoch_metrics)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_samples = 0.0, 0
    all_probs, all_targets = [], []

    for pos, feat, labels in tqdm(loader, leave=False):
        pos = pos.to(device)
        feat = feat.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(pos, feat)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        all_probs.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, N_CLASSES))
    targets = np.concatenate(all_targets) if all_targets else np.zeros(0, dtype=int)
    metrics = epoch_metrics(targets, probs, total_loss / max(n_samples, 1))
    return metrics, targets, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-path", required=True,
                         help="LMDB pretraite (sortie de preprocess_lmdb.py)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-file", default=None,
                         help="JSON {\"train\": [pdb_ids], \"val\": [pdb_ids], ...} "
                              "(ex. splits.json de download_datasets.py) pour "
                              "imposer un split explicite plutot qu'un split "
                              "aleatoire par pdb_id")
    parser.add_argument("--n-points", type=int, default=128,
                         help="Nombre de voxels (les plus denses) gardes par cube")
    parser.add_argument("--jitter-std", type=float, default=0.01,
                         help="Bruit continu (Angstrom) ajoute aux positions pour "
                              "casser les ex-aequo de distance issus de la grille "
                              "de voxels (cf. docstring de cube_to_point_cloud) ; "
                              "garder tres petit devant l'espacement de voxel")
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--lmax", type=int, default=2,
                         help="Ordre maximal des harmoniques spheriques")
    parser.add_argument("--num-basis", type=int, default=8)
    parser.add_argument("--max-radius", type=float, default=15.0,
                         help="Portee (Angstrom) de l'embedding radial ; a "
                              "adapter a la taille du cube et a n_points")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="runs/equivariant",
                         help="Dossier des sorties : historique par epoch "
                              "(CSV/JSON/PNG), matrice de confusion, ROC")
    parser.add_argument("--checkpoint", default=None,
                         help="Chemin du meilleur checkpoint (defaut : "
                              "<out-dir>/best_model.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-equivariance-check", action="store_true",
                         help="Desactive la verification empirique d'equivariance "
                              "effectuee avant l'entrainement")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.out_dir, "best_model.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Chargement du dataset...")
    full_dataset = CAPointCloudDataset(args.lmdb_path, n_points=args.n_points,
                                        augment=False, jitter_std=args.jitter_std)
    train_idx, val_idx, train_ids, val_ids = split_by_pdb_id(
        full_dataset, args.val_fraction, args.seed, args.split_file
    )
    print(f"Entrees train: {len(train_ids)}  |  Entrees val: {len(val_ids)}")
    print(f"Echantillons train: {len(train_idx)}  |  Echantillons val: {len(val_idx)}")

    # Meme environnement LMDB sous-jacent pour les deux vues (train augmente,
    # val non augmente) : evite l'erreur lmdb "already open in this process".
    train_set = AugmentedView(full_dataset, train_idx, augment=True)
    val_set = AugmentedView(full_dataset, val_idx, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    train_labels = [full_dataset.labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels, N_CLASSES)
    print("Poids de classes (train) :",
          {RESIDUE_CLASSES[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())})

    device = torch.device(args.device)
    model = EquivariantPointClassifier(
        n_classes=N_CLASSES, lmax=args.lmax, k_neighbors=args.k_neighbors,
        num_basis=args.num_basis, max_radius=args.max_radius,
    ).to(device)
    print(f"Parametres entrainables : "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if not args.skip_equivariance_check:
        print("\nVerification empirique de l'equivariance SO(3)...")
        sample_pos, sample_feat, _ = next(iter(val_loader))
        rel_diff = check_equivariance(model, sample_pos, sample_feat, device)
        print(f"Difference relative avant/apres rotation continue : {rel_diff:.2e}")
        print("(doit etre proche de 0, typiquement < 1e-4, quel que soit l'angle)\n")

    criterion = tnn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = MetricsHistory(args.out_dir, title="Equivariant SO(3) point-cloud classifier")
    best_val_metric = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics, _, _ = run_epoch(model, val_loader, device, criterion)
        history.log(epoch, train_metrics, val_metrics,
                    lr=optimizer.param_groups[0]["lr"])

        monitor = val_metrics["f1_macro"]
        scheduler.step(monitor)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} "
            f"top3={val_metrics['top3_acc']:.3f} P={val_metrics['precision_macro']:.3f} "
            f"F1={val_metrics['f1_macro']:.3f} AUC={val_metrics.get('auc_macro') or float('nan'):.3f}"
        )

        if monitor > best_val_metric:
            best_val_metric = monitor
            torch.save({
                "model_type": "equivariant",
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "residue_classes": RESIDUE_CLASSES,
                "train_pdb_ids": train_ids,
                "val_pdb_ids": val_ids,
                "train_idx": train_idx,
                "args": vars(args),
            }, args.checkpoint)
            print(f"  -> nouveau meilleur modele sauvegarde ({args.checkpoint})")

    print("\n=== Entrainement termine ===")
    print(f"Meilleure macro-F1 de validation : {best_val_metric:.4f}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_targets, val_probs = run_epoch(model, val_loader, device, criterion)
    final = save_final_reports(
        val_targets, val_probs, args.out_dir, "val_best",
        title=f"Equivariant - val (meilleure epoch {checkpoint['epoch']})",
        extra={"best_epoch": checkpoint["epoch"]},
    )
    print(f"Rapports val (meilleur checkpoint) ecrits dans {args.out_dir} : "
          f"acc={final['accuracy']:.3f} F1={final['f1_macro']:.3f} "
          f"AUC macro={final['auc_macro']}")


if __name__ == "__main__":
    main()
