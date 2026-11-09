#!/usr/bin/env python3
"""
Briques communes au flow matching de positions atomiques (un residu par
cube de densite, C-alpha connu au centre du cube, type d'acide amine connu).

  - representation atom14 (14 emplacements par residu, masque) ;
  - topologie des liaisons (paires liees 1-2 et 1-3, distances de graphe) ;
  - atomes symetriques (ASP, GLU, PHE, TYR) pour un RMSD juste ;
  - convention d'axes cube <-> coordonnees et rotation conjointe ;
  - dataset (cube depuis le LMDB + atomes depuis atom14.npz) ;
  - echantillonnage (Euler) et metriques par residu ;
  - rapports (CSV / JSON / figures).

Convention des coordonnees : en Angstrom, relatives au C-alpha, dans les
axes cartesiens du .cif. Le cube (extract_ca_density_lmdb.py) verifie
cube[i, j, k] <-> decalage (x_i, y_j, z_k), avec x_i = (i - 9.5) * 1 A pour un
cube de 20 a 1 A : l'axe 0 du cube est x, l'axe 1 est y, l'axe 2 est z.
"""

import csv
import json
import os
import pickle
from collections import deque

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from train_utils import (
    RESIDUE_CLASSES, SERIES_ALT, TEXT_SECONDARY, SURFACE, _new_fig, _style_axes, plt,
)

STANDARD = RESIDUE_CLASSES[:20]
N_SLOTS = 14
CA_SLOT = 1

# Ordre atom14 (identique a AlphaFold 2)
ATOM14 = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3",
            "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

# Liaisons covalentes intra-residu (atomes lourds)
_SC_BONDS = {
    "ALA": [],
    "ARG": [("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("CB", "SG")],
    "GLN": [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [],
    "HIS": [("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2")],
    "ILE": [("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ")],
    "PRO": [("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    "SER": [("CB", "OG")],
    "THR": [("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("NE1", "CE2"),
            ("CD2", "CE2"), ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"),
            ("CZ3", "CH2")],
    "TYR": [("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")],
    "VAL": [("CB", "CG1"), ("CB", "CG2")],
}

# Atomes equivalents par symetrie (convention AlphaFold 2)
_SYM_SWAPS = {
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}


def _bonds(res):
    b = [("N", "CA"), ("CA", "C"), ("C", "O")]
    if res != "GLY":
        b.append(("CA", "CB"))
    return b + _SC_BONDS[res]


def _build_tables():
    n_res = len(STANDARD)
    atom_mask = np.zeros((n_res, N_SLOTS), dtype=bool)
    topo = np.full((n_res, N_SLOTS, N_SLOTS), 4, dtype=np.int64)  # 0 soi,1 lie,2 1-3,3 1-4,4 loin
    perm_sym = np.tile(np.arange(N_SLOTS), (n_res, 1))
    pair_lists = []
    for r, res in enumerate(STANDARD):
        names = ATOM14[res]
        idx = {a: i for i, a in enumerate(names)}
        atom_mask[r, :len(names)] = True
        adj = {i: [] for i in range(len(names))}
        for a, b in _bonds(res):
            adj[idx[a]].append(idx[b])
            adj[idx[b]].append(idx[a])
        pairs = []
        for s in range(len(names)):
            dist = {s: 0}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for t, d in dist.items():
                topo[r, s, t] = min(d, 4)
                if s < t and d in (1, 2):
                    pairs.append((s, t, d))
        pair_lists.append(pairs)
        for a, b in _SYM_SWAPS.get(res, []):
            perm_sym[r, idx[a]], perm_sym[r, idx[b]] = idx[b], idx[a]
    n_pairs = max(len(p) for p in pair_lists)
    pairs = np.zeros((n_res, n_pairs, 2), dtype=np.int64)
    pair_kind = np.zeros((n_res, n_pairs), dtype=np.int64)    # 1 = liaison, 2 = 1-3
    pair_valid = np.zeros((n_res, n_pairs), dtype=bool)
    for r, pl in enumerate(pair_lists):
        for p, (s, t, d) in enumerate(pl):
            pairs[r, p] = (s, t)
            pair_kind[r, p] = d
            pair_valid[r, p] = True
    return atom_mask, topo, perm_sym, pairs, pair_kind, pair_valid


ATOM_MASK, TOPO, PERM_SYM, PAIRS, PAIR_KIND, PAIR_VALID = _build_tables()


def _build_order():
    """Ordre de placement du mode autoregressif : parcours en largeur du
    graphe des liaisons depuis le C-alpha (voisins par numero d'emplacement
    croissant). ORDER[r, k] = emplacement place en k-ieme (k=0 : C-alpha),
    RANK = inverse, PARENT[r, s] = atome deja place auquel s est lie
    (le C-alpha est son propre parent). Emplacements absents : rang 99."""
    n_res = len(STANDARD)
    order = np.full((n_res, N_SLOTS), CA_SLOT, dtype=np.int64)
    rank = np.full((n_res, N_SLOTS), 99, dtype=np.int64)
    parent = np.full((n_res, N_SLOTS), CA_SLOT, dtype=np.int64)
    for r, res in enumerate(STANDARD):
        names = ATOM14[res]
        idx = {a: i for i, a in enumerate(names)}
        adj = {i: set() for i in range(len(names))}
        for a, b in _bonds(res):
            adj[idx[a]].add(idx[b])
            adj[idx[b]].add(idx[a])
        seq, seen, q = [CA_SLOT], {CA_SLOT}, deque([CA_SLOT])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    parent[r, v] = u
                    seq.append(v)
                    q.append(v)
        order[r, :len(seq)] = seq
        rank[r, seq] = np.arange(len(seq))
    return order, rank, parent


ORDER, RANK, PARENT = _build_order()
N_ATOMS = ATOM_MASK.sum(1)                      # atomes par type (C-alpha compris)

BACKBONE_SLOTS = [0, 2, 3]            # N, C, O (CA fixe)
SIDECHAIN_SLOTS = list(range(4, N_SLOTS))


# ---------------------------------------------------------------------------
# Rotations : cube et coordonnees de facon coherente
# ---------------------------------------------------------------------------

_P = torch.tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])


def coord_rotation_from_grid(R):
    """R (B,3,3) est la matrice passee a train_cnn_baseline.rotate_batch_continuous
    (repere de grid_sample, ordre (W,H,D) = (z,y,x)). Retourne M (B,3,3) telle
    que les coordonnees physiques (x,y,z) des atomes tournent comme le cube :
    x' = M x. M = P R^T P (P inverse l'ordre des axes), det(M) = +1."""
    P = _P.to(R.device, R.dtype)
    return P @ R.transpose(1, 2) @ P


def rotate_coords(pos, M):
    """pos (B,14,3), M (B,3,3) -> pos @ M^T."""
    return torch.einsum("bij,baj->bai", M, pos)


def spherical_mask_t(size, device):
    c = (size - 1) / 2.0
    g = torch.arange(size, device=device, dtype=torch.float32) - c
    r2 = g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2
    return (r2 <= c ** 2).float()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FlowDataset(Dataset):
    """Cube de densite (LMDB pretraite) + atomes du residu (atom14.npz produit
    par extract_atom14.py). Seuls les residus standard COMPLETS (tous les
    atomes lourds atom14 presents) sont dans atom14.npz."""

    META_FIELDS = ("pdb_ids", "chains", "seqids", "resnames", "ca_xyz")

    def __init__(self, lmdb_path, atom14_path):
        self.lmdb_path = lmdb_path
        self.env = None
        z = np.load(atom14_path, allow_pickle=False)
        self.keys = [k.encode("ascii") for k in z["keys"]]
        self.pos = z["pos"].astype(np.float32)
        self.mask = z["mask"].astype(bool)
        self.restype = z["restype"].astype(np.int64)
        self.pdb_ids = [str(p) for p in z["pdb_ids"]]
        self.meta = {k: z[k] for k in self.META_FIELDS if k in z}

    def _ensure_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        self._ensure_env()
        with self.env.begin() as txn:
            sample = pickle.loads(txn.get(self.keys[i]))
        cube = np.nan_to_num(sample["cube"].astype(np.float32))
        return (torch.from_numpy(cube).unsqueeze(0), torch.from_numpy(self.pos[i]),
                torch.from_numpy(self.mask[i]), int(self.restype[i]), i)

    def indices_for(self, pdb_ids):
        ids = set(pdb_ids)
        return [i for i, p in enumerate(self.pdb_ids) if p in ids]


# ---------------------------------------------------------------------------
# Flow matching (interpolation lineaire / rectified flow)
# ---------------------------------------------------------------------------

def predicted_mask(mask):
    """Atomes dont la position est predite : presents, sauf le C-alpha (fixe)."""
    m = mask.clone()
    m[:, CA_SLOT] = False
    return m


def sample_noise(mask, sigma, generator=None):
    y0 = torch.randn(mask.shape + (3,), device=mask.device, generator=generator) * sigma
    return y0 * predicted_mask(mask).unsqueeze(-1)


def pair_tensors(restype, mask):
    """Paires (1-2, 1-3) du residu : idx (B,P,2), kind (B,P), valid (B,P)."""
    dev = restype.device
    idx = torch.as_tensor(PAIRS, device=dev)[restype]
    kind = torch.as_tensor(PAIR_KIND, device=dev)[restype]
    valid = torch.as_tensor(PAIR_VALID, device=dev)[restype]
    valid = valid & mask.gather(1, idx[..., 0]) & mask.gather(1, idx[..., 1])
    return idx, kind, valid


def pair_distances(pos, idx):
    a = torch.gather(pos, 1, idx[..., 0:1].expand(-1, -1, 3))
    b = torch.gather(pos, 1, idx[..., 1:2].expand(-1, -1, 3))
    return (a - b).norm(dim=-1)


def _encode(model, cube):
    """Features de densite calculees une seule fois par cube si le modele le
    permet (CNN) : le cube ne change pas pendant l'integration de l'ODE."""
    return model.encode(cube) if hasattr(model, "encode") else None


@torch.no_grad()
def sample_atoms(model, cube, restype, mask, n_steps, sigma, scale, generator=None):
    """Mode joint : integre l'ODE de t=0 (bruit) a t=1 (Euler) pour tous les
    atomes a la fois. Retourne (B,14,3) en A."""
    pm = predicted_mask(mask).unsqueeze(-1)
    y = sample_noise(mask, sigma, generator)
    B = y.shape[0]
    feats = _encode(model, cube)
    kw = {"feats": feats} if feats is not None else {}
    for s in range(n_steps):
        t = torch.full((B,), s / n_steps, device=y.device)
        v = model(cube, y, t, restype, mask, **kw)
        y = (y + v / n_steps) * pm
    return y * scale


def ar_start(ctx, restype, mask, center, sigma_ar, generator=None):
    """Point de depart du bruit, par atome, en mode autoregressif (unites
    normalisees) : centre sur le parent (deja place) ou sur le C-alpha."""
    noise = torch.randn(ctx.shape, device=ctx.device, generator=generator) * sigma_ar
    if center == "parent":
        par = torch.as_tensor(PARENT, device=ctx.device)[restype]
        noise = noise + torch.gather(ctx, 1, par[..., None].expand(-1, -1, 3))
    return noise * predicted_mask(mask).unsqueeze(-1)


@torch.no_grad()
def sample_atoms_ar(model, cube, restype, mask, n_steps, sigma_ar, scale, center="parent",
                    generator=None):
    """Mode autoregressif : les atomes sont places un par un dans l'ordre
    ORDER (parcours des liaisons depuis le C-alpha) ; chaque atome a son
    propre flow de t=0 a 1, conditionne sur les atomes deja places."""
    dev = cube.device
    B = cube.shape[0]
    ctx = torch.zeros(B, N_SLOTS, 3, device=dev)        # C-alpha = 0, deja place
    order = torch.as_tensor(ORDER, device=dev)[restype]
    n_atoms = torch.as_tensor(N_ATOMS, device=dev)[restype]
    feats = _encode(model, cube)
    ar = torch.arange(B, device=dev)
    for k in range(1, N_SLOTS):
        active = (n_atoms > k) & mask[ar, order[:, k]]
        if not active.any():
            break
        slot = order[:, k]
        y = ctx.clone()
        y[ar, slot] = ar_start(ctx, restype, mask, center, sigma_ar, generator)[ar, slot]
        for s in range(n_steps):
            t = torch.full((B, N_SLOTS), s / n_steps, device=dev)
            v = model(cube, y, t, restype, mask, ctx=ctx, feats=feats)
            y[ar, slot] = y[ar, slot] + v[ar, slot] / n_steps
        ctx[ar[active], slot[active]] = y[ar[active], slot[active]]
    return ctx * scale


# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------

def _rmsd(diff2, m):
    n = m.sum(-1)
    out = np.sqrt((diff2 * m).sum(-1) / np.maximum(n, 1))
    return np.where(n > 0, out, np.nan)


def residue_metrics(pred, gt, mask, restype):
    """pred, gt (N,14,3) en A ; mask (N,14) ; restype (N,).
    RMSD par residu (atomes lourds hors C-alpha), avec la meilleure des deux
    numerotations pour les atomes symetriques (ASP, GLU, PHE, TYR).
    Retourne un dict de tableaux (N,)."""
    pred, gt, mask = np.asarray(pred), np.asarray(gt), np.asarray(mask, bool).copy()
    restype = np.asarray(restype)
    mask[:, CA_SLOT] = False
    perm = PERM_SYM[restype]                                     # (N,14)
    gt_sw = np.take_along_axis(gt, perm[..., None], axis=1)
    d2 = ((pred - gt) ** 2).sum(-1)
    d2_sw = ((pred - gt_sw) ** 2).sum(-1)
    use_sw = _rmsd(d2_sw, mask) < _rmsd(d2, mask)
    d2 = np.where(use_sw[:, None], d2_sw, d2)
    gt_best = np.where(use_sw[:, None, None], gt_sw, gt)

    bb = np.zeros(N_SLOTS, bool); bb[BACKBONE_SLOTS] = True
    sc = np.zeros(N_SLOTS, bool); sc[SIDECHAIN_SLOTS] = True
    out = {
        "rmsd": _rmsd(d2, mask),
        "rmsd_backbone": _rmsd(d2, mask & bb),
        "rmsd_sidechain": _rmsd(d2, mask & sc),
    }
    # geometrie : ecart des longueurs de liaison (1-2) et des distances 1-3
    idx = PAIRS[restype]
    valid = PAIR_VALID[restype] & np.take_along_axis(mask | (np.arange(N_SLOTS) == CA_SLOT),
                                                     idx[..., 0], 1) \
        & np.take_along_axis(mask | (np.arange(N_SLOTS) == CA_SLOT), idx[..., 1], 1)
    kind = PAIR_KIND[restype]

    def dists(x):
        a = np.take_along_axis(x, idx[..., 0:1].repeat(3, -1), 1)
        b = np.take_along_axis(x, idx[..., 1:2].repeat(3, -1), 1)
        return np.linalg.norm(a - b, axis=-1)

    err = np.abs(dists(pred) - dists(gt_best))
    for k, name in ((1, "bond_err"), (2, "angle13_err")):
        m = valid & (kind == k)
        n = m.sum(-1)
        out[name] = np.where(n > 0, (err * m).sum(-1) / np.maximum(n, 1), np.nan)
    return out


def _nan_stat(fn, x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else float("nan")


def aggregate(per_res):
    r = per_res["rmsd"]
    agg = {
        "n_residues": int(np.isfinite(r).sum()),
        "rmsd_mean": _nan_stat(np.mean, r),
        "rmsd_median": _nan_stat(np.median, r),
        "rmsd_backbone_mean": _nan_stat(np.mean, per_res["rmsd_backbone"]),
        "rmsd_sidechain_mean": _nan_stat(np.mean, per_res["rmsd_sidechain"]),
        "frac_rmsd_lt_1A": _nan_stat(np.mean, r[np.isfinite(r)] < 1.0),
        "frac_rmsd_lt_2A": _nan_stat(np.mean, r[np.isfinite(r)] < 2.0),
        "bond_err_mean": _nan_stat(np.mean, per_res["bond_err"]),
        "angle13_err_mean": _nan_stat(np.mean, per_res["angle13_err"]),
    }
    return agg


# ---------------------------------------------------------------------------
# Rapports
# ---------------------------------------------------------------------------

def plot_rmsd_per_restype(per_res, restype, path, title=""):
    rows = []
    for r, name in enumerate(STANDARD):
        m = restype == r
        if m.sum() == 0:
            continue
        rows.append((name, int(m.sum()), np.nanmean(per_res["rmsd"][m]),
                     np.nanmean(per_res["rmsd_sidechain"][m]) if name != "GLY" else np.nan))
    rows.sort(key=lambda x: x[2])
    fig, ax = _new_fig(figsize=(6.6, 0.3 * len(rows) + 1.4))
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))
    h = 0.38
    ax.barh(y + h / 2, [r[2] for r in rows], height=h * 0.92, color=SERIES_ALT[0],
            label="tous atomes (hors C-alpha)")
    ax.barh(y - h / 2, [0 if np.isnan(r[3]) else r[3] for r in rows], height=h * 0.92,
            color=SERIES_ALT[1], label="chaine laterale")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]} (n={r[1]})" for r in rows], fontsize=8)
    xmax = np.nanmax([r[2] for r in rows] + [r[3] for r in rows if not np.isnan(r[3])])
    ax.set_xlim(0, xmax * 1.15)
    for yi, r in zip(y, rows):
        ax.text(r[2] + xmax * 0.01, yi + h / 2, f"{r[2]:.2f}", va="center", fontsize=6.5,
                color=TEXT_SECONDARY)
    ax.set_xlabel("RMSD moyen (A)")
    ax.legend(loc="lower right", frameon=False, fontsize=8, reverse=True)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_rmsd_cdf(curves, path, title=""):
    """curves : liste de (label, tableau de RMSD). <= 3 series (palette)."""
    fig, ax = _new_fig(figsize=(6.2, 4.4))
    _style_axes(ax)
    xmax = 0
    for i, (label, r) in enumerate(curves):
        r = np.sort(r[np.isfinite(r)])
        if r.size == 0:
            continue
        xmax = max(xmax, np.percentile(r, 99))
        ax.plot(r, np.arange(1, r.size + 1) / r.size, linewidth=2,
                color=SERIES_ALT[i % len(SERIES_ALT)],
                label=f"{label} (mediane {np.median(r):.2f} A)")
    ax.axvline(1.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xlim(0, max(xmax, 1.5))
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("RMSD par residu (A)  (tirets : 1 A)")
    ax.set_ylabel("fraction des residus")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def save_flow_reports(pred, gt, mask, restype, out_dir, prefix, title="", extra=None):
    """Ecrit <prefix>_metrics.json, _per_restype.csv, _rmsd_per_restype.png,
    _rmsd_cdf.png, _predictions.npz. Retourne les metriques agregees."""
    os.makedirs(out_dir, exist_ok=True)
    restype = np.asarray(restype)
    per_res = residue_metrics(pred, gt, mask, restype)
    agg = aggregate(per_res)
    if extra:
        agg.update(extra)
    base = os.path.join(out_dir, prefix)
    with open(base + "_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    with open(base + "_per_restype.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["restype", "n", "rmsd_mean", "rmsd_median", "rmsd_sidechain_mean",
                    "rmsd_backbone_mean", "bond_err_mean"])
        def stat(fn, x):
            x = x[np.isfinite(x)]
            return f"{fn(x):.4f}" if x.size else ""

        for r, name in enumerate(STANDARD):
            m = restype == r
            if m.sum():
                w.writerow([name, int(m.sum()), stat(np.mean, per_res["rmsd"][m]),
                            stat(np.median, per_res["rmsd"][m]),
                            stat(np.mean, per_res["rmsd_sidechain"][m]),
                            stat(np.mean, per_res["rmsd_backbone"][m]),
                            stat(np.mean, per_res["bond_err"][m])])
    plot_rmsd_per_restype(per_res, restype, base + "_rmsd_per_restype.png", title)
    plot_rmsd_cdf([(prefix, per_res["rmsd"])], base + "_rmsd_cdf.png", title)
    np.savez_compressed(base + "_predictions.npz", pred=np.asarray(pred, np.float32),
                        gt=np.asarray(gt, np.float32), mask=np.asarray(mask),
                        restype=restype, **{k: v for k, v in per_res.items()})
    return agg


# ---------------------------------------------------------------------------
# Export PDB (visualisation dans ChimeraX / PyMOL)
# ---------------------------------------------------------------------------

def write_pdb(path, records):
    """records : liste de (chain, seqid, resname, ca_xyz (3,), atoms_rel (14,3), mask (14,)).
    Les positions sont absolues = ca_xyz + atoms_rel."""
    lines, serial = [], 1
    for chain, seqid, resname, ca, rel, m in records:
        names = ATOM14[resname]
        for s, name in enumerate(names):
            if not m[s]:
                continue
            x, y, z = (np.asarray(ca) + rel[s]).tolist()
            el = name[0]
            nm = f" {name:<3s}" if len(name) < 4 else name
            lines.append(f"ATOM  {serial:5d} {nm:4s} {resname:3s} {str(chain)[:1]:1s}"
                         f"{int(seqid):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                         f"          {el:>2s}")
            serial += 1
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
