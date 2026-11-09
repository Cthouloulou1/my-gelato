#!/usr/bin/env python3
"""
Reseaux de champ de vitesse pour le flow matching des positions atomiques.
Interface commune :

    v = model(cube, y_t, t, restype, mask)
        cube    (B,1,S,S,S)  densite (z-score, axes x,y,z, 1 A/voxel)
        y_t     (B,14,3)     positions courantes, en unites normalisees
                             (A / coord_scale), relatives au C-alpha
        t       (B,)         temps dans [0, 1]
        restype (B,)         type d'acide amine (0..19)
        mask    (B,14)       emplacements atom14 presents
        v       (B,14,3)     vitesse (unites normalisees), nulle pour le
                             C-alpha (fixe) et les emplacements absents

Deux reseaux :
  CNNFlowNet          non equivariant : encodeur CNN 3D de la densite,
                      features lues a la position courante de chaque atome
                      (interpolation trilineaire), puis transformer sur les
                      14 atomes avec un biais d'attention selon la topologie
                      des liaisons. Rotations apprises par augmentation.
  EquivariantFlowNet  SE(3)-equivariant (e3nn) : graphe commun voxels denses
                      + atomes (k plus proches voisins + aretes de liaison
                      1-2 et 1-3 typees), sortie = un vecteur 1o par atome.
"""

import math

import torch
import torch.nn as tnn
import torch.nn.functional as F

from flow_common import CA_SLOT, N_SLOTS, TOPO, pair_tensors, predicted_mask


class TimeEmbedding(tnn.Module):
    def __init__(self, dim, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi)
        self.mlp = tnn.Sequential(tnn.Linear(2 * n_freq, dim), tnn.SiLU(), tnn.Linear(dim, dim))

    def forward(self, t):
        x = t[:, None] * self.freqs
        return self.mlp(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))


def atom_ids(restype):
    return restype[:, None] * N_SLOTS + torch.arange(N_SLOTS, device=restype.device)


# ---------------------------------------------------------------------------
# CNN + transformer
# ---------------------------------------------------------------------------

class BiasedTransformerLayer(tnn.Module):
    """Couche transformer pre-norm avec biais d'attention additif par tete
    (B,H,L,L). Ecrite explicitement : le chemin "rapide" de
    nn.TransformerEncoder en mode eval produisait des NaN avec ce masque."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.norm1 = tnn.LayerNorm(d_model)
        self.qkv = tnn.Linear(d_model, 3 * d_model)
        self.proj = tnn.Linear(d_model, d_model)
        self.norm2 = tnn.LayerNorm(d_model)
        self.ff = tnn.Sequential(tnn.Linear(d_model, 4 * d_model), tnn.GELU(),
                                 tnn.Linear(4 * d_model, d_model))
        self.drop = tnn.Dropout(dropout)

    def forward(self, x, bias):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.norm2(x)))


def _conv_block(cin, cout):
    return tnn.Sequential(
        tnn.Conv3d(cin, cout, 3, padding=1, bias=False), tnn.BatchNorm3d(cout), tnn.SiLU(),
        tnn.Conv3d(cout, cout, 3, padding=1, bias=False), tnn.BatchNorm3d(cout), tnn.SiLU(),
    )


def sample_features(feat, pos_A, half_extent):
    """Lit feat (B,C,D,H,W) (axes D,H,W = x,y,z) aux positions pos_A (B,A,3)
    en A relatives au centre. Coordonnees normalisees de grid_sample : ordre
    (W,H,D) = (z,y,x), +-1 = centre des voxels extremes (align_corners)."""
    B, A, _ = pos_A.shape
    g = (pos_A.flip(-1) / half_extent).view(B, A, 1, 1, 3)
    out = F.grid_sample(feat, g, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(B, feat.shape[1], A).transpose(1, 2)


class CNNFlowNet(tnn.Module):
    def __init__(self, coord_scale=4.0, cube_size=20, voxel=1.0, channels=(32, 64, 128),
                 d_model=128, n_layers=4, n_heads=4, dropout=0.0):
        super().__init__()
        assert cube_size % 4 == 0
        self.coord_scale = coord_scale
        self.n_heads = n_heads
        c1, c2, c3 = channels
        self.enc1 = _conv_block(1, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        # demi-etendue (A) entre centres des voxels extremes, a chaque niveau
        self.half = [(cube_size - 2 ** l) / 2 * voxel for l in range(3)]
        self.global_proj = tnn.Linear(2 * c3, d_model)
        self.in_proj = tnn.Linear(c1 + c2 + 3, d_model)
        self.atom_emb = tnn.Embedding(20 * N_SLOTS, d_model)
        self.time_emb = TimeEmbedding(d_model)
        self.topo_bias = tnn.Embedding(5, n_heads)
        self.register_buffer("topo", torch.as_tensor(TOPO))
        self.layers = tnn.ModuleList(BiasedTransformerLayer(d_model, n_heads, dropout)
                                     for _ in range(n_layers))
        self.out = tnn.Sequential(tnn.LayerNorm(d_model), tnn.Linear(d_model, 3))

    def forward(self, cube, y_t, t, restype, mask):
        B = cube.shape[0]
        f1 = self.enc1(cube)
        f2 = self.enc2(F.max_pool3d(f1, 2))
        f3 = self.enc3(F.max_pool3d(f2, 2))
        glob = self.global_proj(torch.cat([f3.mean((2, 3, 4)), f3.amax((2, 3, 4))], 1))

        pos_A = y_t * self.coord_scale
        local = torch.cat([sample_features(f1, pos_A, self.half[0]),
                           sample_features(f2, pos_A, self.half[1])], -1)
        h = (self.in_proj(torch.cat([local, y_t], -1)) + self.atom_emb(atom_ids(restype))
             + (self.time_emb(t) + glob)[:, None, :])

        bias = self.topo_bias(self.topo[restype]).permute(0, 3, 1, 2)    # (B,H,14,14)
        eye = torch.eye(N_SLOTS, dtype=torch.bool, device=cube.device)
        blocked = (~mask)[:, None, None, :] & ~eye                        # cles absentes
        bias = bias.masked_fill(blocked, -1e4)
        for layer in self.layers:
            h = layer(h, bias)
        return self.out(h) * predicted_mask(mask).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Equivariant (e3nn)
# ---------------------------------------------------------------------------

class EquivariantFlowNet(tnn.Module):
    def __init__(self, coord_scale=4.0, n_density=256, k_neighbors=16, max_radius=8.0,
                 num_basis=10, lmax=2, channels=(32, 32, 32), activation="silu",
                 d_emb=16, voxel=1.0, jitter_std=0.01):
        super().__init__()
        from e3nn import o3
        from e3nn.nn import NormActivation
        from train_equivariant_cnn import PointConvLayer

        self.coord_scale = coord_scale
        self.n_density = n_density
        self.k = k_neighbors
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.voxel = voxel
        self.jitter_std = jitter_std
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        act = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "silu": F.silu}[activation]

        self.atom_emb = tnn.Embedding(20 * N_SLOTS, d_emb)
        self.time_emb = TimeEmbedding(d_emb)
        n_in = 3 + 2 * d_emb
        irreps_in = o3.Irreps(f"{n_in}x0e")
        irreps = [o3.Irreps(f"{c}x0e + {c // 2}x1o + {c // 4}x2e") for c in channels]
        edge_dim = num_basis + 2          # base radiale + type d'arete (1-2, 1-3)
        self.convs = tnn.ModuleList()
        self.acts = tnn.ModuleList()
        prev = irreps_in
        for ir in irreps:
            self.convs.append(PointConvLayer(prev, ir, self.irreps_sh, edge_dim))
            self.acts.append(NormActivation(ir, act))
            prev = ir
        self.out = o3.Linear(prev, o3.Irreps("1x1o"))

    def density_points(self, cube):
        B, _, S, _, _ = cube.shape
        vals, idx = cube.view(B, -1).topk(self.n_density, dim=1)
        ijk = torch.stack([idx // (S * S), (idx // S) % S, idx % S], -1).float()
        pos = (ijk - (S - 1) / 2.0) * self.voxel
        if self.jitter_std > 0:   # casse les ex-aequo de distance de la grille
            pos = pos + torch.randn_like(pos) * self.jitter_std
        return pos, vals

    def forward(self, cube, y_t, t, restype, mask):
        pos_d, val_d = self.density_points(cube)
        return self.forward_points(pos_d, val_d, y_t, t, restype, mask)

    def forward_points(self, pos_d, val_d, y_t, t, restype, mask):
        from e3nn import o3
        from e3nn.math import soft_one_hot_linspace
        from train_equivariant_cnn import knn_graph_batched

        B, nd, _ = pos_d.shape
        dev = pos_d.device
        N = nd + N_SLOTS
        # atomes absents envoyes tres loin (au-dela de max_radius : aucun message)
        far = (1e3 + 50.0 * torch.arange(N_SLOTS, device=dev, dtype=pos_d.dtype))[None, :, None]
        pos_a = torch.where(mask[..., None], y_t * self.coord_scale, far.expand(B, -1, 3))
        pos = torch.cat([pos_d, pos_a], 1)

        temb = self.time_emb(t)[:, None, :]
        aemb = self.atom_emb(atom_ids(restype))
        d_emb = aemb.shape[-1]
        dens = torch.cat([val_d[..., None], torch.ones_like(val_d)[..., None],
                          torch.zeros_like(val_d)[..., None],
                          torch.zeros(B, nd, d_emb, device=dev), temb.expand(-1, nd, -1)], -1)
        atoms = torch.cat([torch.zeros(B, N_SLOTS, 2, device=dev),
                           torch.ones(B, N_SLOTS, 1, device=dev), aemb,
                           temb.expand(-1, N_SLOTS, -1)], -1)
        x = torch.cat([dens, atoms], 1).reshape(B * N, -1)

        src, dst, k_eff = knn_graph_batched(pos, self.k)
        etype = torch.zeros(src.shape[0], 2, device=dev)
        idx, kind, valid = pair_tensors(restype, mask)
        off = (torch.arange(B, device=dev) * N + nd)[:, None]
        a = (idx[..., 0] + off)[valid]
        b = (idx[..., 1] + off)[valid]
        kd = kind[valid]
        src = torch.cat([src, a, b])
        dst = torch.cat([dst, b, a])
        oh = F.one_hot(kd - 1, 2).float()
        etype = torch.cat([etype, oh, oh])

        pf = pos.reshape(B * N, 3)
        vec = pf[dst] - pf[src]
        sh = o3.spherical_harmonics(self.irreps_sh, vec, normalize=True, normalization="component")
        radial = soft_one_hot_linspace(vec.norm(dim=-1), 0.0, self.max_radius, self.num_basis,
                                       basis="smooth_finite", cutoff=True)
        edge = torch.cat([radial, etype], -1)

        h = x
        for conv, act in zip(self.convs, self.acts):
            h = act(conv(h, src, dst, sh, edge, B * N, k_eff))
        v = self.out(h).reshape(B, N, 3)[:, nd:]
        return v * predicted_mask(mask).unsqueeze(-1)


# ---------------------------------------------------------------------------

MODEL_KWARGS = {
    "cnn": ("coord_scale", "channels", "d_model", "n_layers", "n_heads", "dropout"),
    "equivariant": ("coord_scale", "n_density", "k_neighbors", "max_radius", "num_basis",
                    "lmax", "eq_channels", "activation"),
}


def build_flow_model(kind, a):
    """a : dict des arguments d'entrainement (sauves dans le checkpoint)."""
    if kind == "cnn":
        return CNNFlowNet(coord_scale=a["coord_scale"], channels=tuple(a["channels"]),
                          d_model=a["d_model"], n_layers=a["n_layers"],
                          n_heads=a["n_heads"], dropout=a["dropout"])
    if kind == "equivariant":
        return EquivariantFlowNet(coord_scale=a["coord_scale"], n_density=a["n_density"],
                                  k_neighbors=a["k_neighbors"], max_radius=a["max_radius"],
                                  num_basis=a["num_basis"], lmax=a["lmax"],
                                  channels=tuple(a["eq_channels"]), activation=a["activation"])
    raise ValueError(kind)
