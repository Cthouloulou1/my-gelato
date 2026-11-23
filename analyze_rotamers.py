#!/usr/bin/env python3
"""
Diagnostic des erreurs de rotamere des modeles de flow, a partir des
predictions deja calculees (<ensemble>_predictions.npz de evaluate_flow.py).
Pas d'inference.

1. Reference « bibliotheque de rotameres », construite sur le TRAIN de
   splits.json (vraies structures) :
     bib. (type)     rotamere le plus frequent de chaque acide amine ;
     bib. (type+phi/psi)  le plus frequent pour l'acide amine ET la case
                     (phi, psi) du vrai squelette (--phipsi-bin degres ; repli
                     sur le type si moins de --min-count exemples). phi, psi
                     viennent des atomes voisins de extract_ca_context.py.
   Rotamere = case de chaque chi (g+, t, g- : 3 cases de 120 deg ; 3 cases de
   60 deg pour les chi pi-periodiques) ; valeur predite = moyenne circulaire
   des chi du train dans ce rotamere. Juste si |delta chi| <= --chi-tol,
   comme dans analyze_flow_geometry.py.
   Si un modele ne fait pas mieux que la bibliotheque, il n'utilise pas la
   densite pour la chaine laterale.

2. Densite de la chaine laterale (--lmdb-path) : densite moyenne du cube
   (score z : (valeur - moyenne) / ecart-type du cube) aux VRAIS atomes au-dela
   du CB, puis chi1 juste par quintile de cette densite. Montre si les erreurs
   viennent de chaines laterales absentes de la carte (non recuperables).

3. Pour les residus dont le chi1 est faux :
     comme la bib. : le chi1 predit tombe dans la meme case que celui de la
                     bibliotheque (le modele se rabat sur le rotamere frequent) ;
     densite pred > vraie : la carte est plus dense sur la chaine laterale
                     predite que sur la vraie (la carte est ambigue ou trompeuse :
                     choisir par la densite n'aiderait pas).

Ecrit dans --out-dir : accuracy.csv, by_density.csv, errors.csv, per_type.csv,
chi1_by_density.png.

Usage :
    python analyze_rotamers.py --atom14 /data/eyerly/splits_atom14.npz \\
        --splits /data/eyerly/pdb_cryoem_splits/splits.json \\
        --ca-context /data/eyerly/splits_ca_context.npz \\
        --lmdb-path /data/eyerly/splits_preprocessed.lmdb \\
        --pred hybride=runs/evaluation_flow_context_e100/hybride \\
        --pred hybrid_chain_pep=runs/evaluation_flow_context_e100/hybrid_chain_pep
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

import analyze_flow_geometry as geo
from flow_common import STANDARD
from make_chimerax_view import load_predictions

SETS = ["val", "test_1_2A", "test_2_3A", "test_3_4A"]
SC_FIRST = 5            # emplacements atom14 au-dela du CB (N CA C O CB | CG ...)
LIB_NAMES = ("bib. (type)", "bib. (type+phi/psi)")


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

def chi_bins(chi, pi):
    """Case de chaque chi : 3 cases de 120 deg (60 deg si pi-periodique)."""
    c = np.where(pi, np.mod(chi, 180.0), np.mod(chi, 360.0))
    b = np.floor(c / np.where(pi, 60.0, 120.0))
    return np.where(np.isfinite(b), b, 0).astype(np.int64).clip(0, 2)


def rot_code(bins, cmask):
    return (np.where(cmask, bins, 0) * (3 ** np.arange(4))).sum(-1)


def circ_mean(a, pi):
    """Moyenne circulaire (deg) ; pi-periodique : sur les angles doubles."""
    k = 2.0 if pi else 1.0
    r = np.radians(a) * k
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) / k


def phi_psi(pos, prev_c, next_n, has_prev, has_next):
    """pos (N,14,3) relatives au C-alpha ; renvoie phi, psi (deg, NaN si absent)."""
    zero = np.zeros_like(pos[:, 0])
    phi = geo.dihedral(prev_c, pos[:, 0], zero, pos[:, 2])
    psi = geo.dihedral(pos[:, 0], zero, pos[:, 2], next_n)
    return np.where(has_prev, phi, np.nan), np.where(has_next, psi, np.nan)


# ---------------------------------------------------------------------------
# Bibliotheque
# ---------------------------------------------------------------------------

class RotamerLibrary:
    def __init__(self, chi, rt, phi, psi, bin_deg, min_count):
        self.bin, self.min = bin_deg, min_count
        cm, pi = geo.CHI_MASK[rt], geo.CHI_PI[rt]
        code = rot_code(chi_bins(chi, pi), cm)
        cell = self._cells(phi, psi)
        groups = defaultdict(list)
        for i in range(len(rt)):
            groups[(int(rt[i]), int(code[i]))].append(i)
            if cell[i] >= 0:
                groups[(int(rt[i]), int(code[i]), int(cell[i]))].append(i)
        self.mean = {}
        for key, rows in groups.items():
            r = key[0]
            rows = np.array(rows)
            self.mean[key] = np.array([circ_mean(chi[rows, k], geo.CHI_PI[r, k])
                                       if geo.CHI_MASK[r, k] else 0.0 for k in range(4)])
        self.top_aa = {r: Counter(code[rt == r].tolist()).most_common(1)[0][0]
                       for r in np.unique(rt)}
        self.top_cell = {}
        by_cell = defaultdict(list)
        for i in np.where(cell >= 0)[0]:
            by_cell[(int(rt[i]), int(cell[i]))].append(int(code[i]))
        for key, codes in by_cell.items():
            if len(codes) >= min_count:
                self.top_cell[key] = Counter(codes).most_common(1)[0][0]

    def _cells(self, phi, psi):
        n = int(round(360 / self.bin))
        ok = np.isfinite(phi) & np.isfinite(psi)
        a = np.floor(np.mod(np.nan_to_num(phi) + 180, 360) / self.bin).astype(int).clip(0, n - 1)
        b = np.floor(np.mod(np.nan_to_num(psi) + 180, 360) / self.bin).astype(int).clip(0, n - 1)
        return np.where(ok, a * n + b, -1)

    def predict(self, rt, phi=None, psi=None):
        """chi predits (N,4) et code du rotamere : par type, et par type+phi/psi."""
        N = len(rt)
        cell = self._cells(phi, psi) if phi is not None else np.full(N, -1)
        out_aa, out_bb = np.zeros((N, 4)), np.zeros((N, 4))
        code_aa, code_bb = np.zeros(N, np.int64), np.zeros(N, np.int64)
        for i in range(N):
            r = int(rt[i])
            if r not in self.top_aa:
                continue
            c = self.top_aa[r]
            out_aa[i], code_aa[i] = self.mean[(r, c)], c
            cb = self.top_cell.get((r, int(cell[i]))) if cell[i] >= 0 else None
            if cb is None:
                out_bb[i], code_bb[i] = out_aa[i], c
            else:
                out_bb[i], code_bb[i] = self.mean.get((r, cb, int(cell[i])),
                                                      self.mean[(r, cb)]), cb
        return (out_aa, code_aa), (out_bb, code_bb)


def dchi(pred_chi, true_chi, rt):
    pi, cm = geo.CHI_PI[rt], geo.CHI_MASK[rt]
    d = np.where(pi, geo.angle_diff(pred_chi, true_chi, 180.0),
                 geo.angle_diff(pred_chi, true_chi))
    return np.where(cm, d, np.nan)


# ---------------------------------------------------------------------------
# Densite
# ---------------------------------------------------------------------------

def sidechain_density(lmdb_path, atom14_path, idx, coords_by_name, batch=256, workers=4):
    """Score z moyen de la densite aux atomes au-dela du CB.
    idx (N,) indices atom14 ; coords_by_name {nom: (positions (N,14,3) relatives
    au C-alpha, masque (N,14))}. Renvoie {nom: (N,)}."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from flow_common import FlowDataset, spherical_mask_t
    from flow_models import sample_features

    ds = FlowDataset(lmdb_path, atom14_path)
    sphere = spherical_mask_t(20, "cpu").bool()
    loader = DataLoader(Subset(ds, [int(i) for i in idx]), batch_size=batch, shuffle=False,
                        num_workers=workers)
    out = {n: np.full(len(idx), np.nan) for n in coords_by_name}
    start = 0
    for cube, _, _, _, _ in loader:
        B = cube.shape[0]
        v = cube[:, 0][:, sphere]                                      # (B, voxels)
        z = (cube - v.mean(1).view(B, 1, 1, 1, 1)) / v.std(1).clamp(min=1e-6).view(B, 1, 1, 1, 1)
        for name, (pos, m) in coords_by_name.items():
            p = torch.from_numpy(pos[start:start + B].astype(np.float32))
            s = sample_features(z, p, (20 - 1) / 2)[..., 0].numpy()   # (B,14)
            mm = m[start:start + B].copy()
            mm[:, :SC_FIRST] = False
            n = mm.sum(1)
            out[name][start:start + B] = np.where(n > 0, (s * mm).sum(1) / np.maximum(n, 1),
                                                  np.nan)
        start += B
    ds.close()
    return out


# ---------------------------------------------------------------------------

def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{100 * x.mean():5.1f} %" if len(x) else "   -   "


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atom14", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--pred", action="append", default=None,
                    help="nom=dossier contenant <ensemble>_predictions.npz")
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--ca-context", default=None,
                    help="npz de extract_ca_context.py (phi/psi) ; sans : bib. par type seule")
    ap.add_argument("--lmdb-path", default=None, help="cubes de densite ; sans : pas de partie 2")
    ap.add_argument("--phipsi-bin", type=float, default=30.0)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--chi-tol", type=float, default=40.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", default="runs/rotamer_diagnostics")
    args = ap.parse_args()
    # chi absents (GLY, ALA...) : angles NaN, masques ensuite
    np.seterr(invalid="ignore", divide="ignore")
    preds = args.pred or ["hybride=runs/evaluation_flow_context_e100/hybride",
                          "hybrid_chain_pep=runs/evaluation_flow_context_e100/hybrid_chain_pep"]
    specs = [tuple(p.split("=", 1)) for p in preds]
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.splits) as f:
        splits = json.load(f)
    z = np.load(args.atom14, allow_pickle=False)
    a14 = {k: z[k] for k in ("pos", "mask", "pdb_ids", "restype")}
    pdb = np.array([str(p) for p in a14["pdb_ids"]])

    phi_all = psi_all = None
    if args.ca_context:
        c = np.load(args.ca_context, allow_pickle=False)
        if len(c["pdb_ids"]) != len(pdb) or not np.array_equal(
                np.array([str(p) for p in c["pdb_ids"]]), pdb):
            raise SystemExit(f"{args.ca_context} n'est pas aligne sur {args.atom14}")
        phi_all, psi_all = phi_psi(a14["pos"].astype(np.float64), c["prev_atoms"][:, 1],
                                   c["next_atoms"][:, 0], c["prev_mask"][:, 1],
                                   c["next_mask"][:, 0])

    # bibliotheque sur le train
    tr = np.where(np.isin(pdb, list(splits["train"])) & geo.CHI_MASK[a14["restype"]][:, 0])[0]
    rt_tr = a14["restype"][tr]
    chi_tr = geo.chis(a14["pos"][tr].astype(np.float64), rt_tr)
    lib = RotamerLibrary(chi_tr, rt_tr, None if phi_all is None else phi_all[tr],
                         None if phi_all is None else psi_all[tr], args.phipsi_bin,
                         args.min_count)
    print(f"Bibliotheque : {len(tr)} residus du train ; {len(lib.top_cell)} cases "
          f"(type, phi/psi) avec >= {args.min_count} exemples"
          + ("" if phi_all is not None else " (pas de --ca-context : par type seulement)"))

    gargs = SimpleNamespace(chi_tol=args.chi_tol, frame_tol=30.0, fail_rmsd=1.0)
    names = [n for n, _ in specs]
    rows_acc, rows_dens, rows_err, rows_type = [], [], [], []
    per_set = {}
    for s in args.sets:
        files = [os.path.join(d, f"{s}_predictions.npz") for _, d in specs]
        if not all(os.path.exists(f) for f in files):
            print(f"\n[{s}] predictions absentes, ignore")
            continue
        data = {}
        idx0 = None
        for (name, _), f in zip(specs, files):
            pred, gt, mask, rt, rmsd = geo.load(f)
            _, _, idx = load_predictions(f, {"pdb_ids": a14["pdb_ids"], "pos": a14["pos"]},
                                         splits[s])
            if idx0 is None:
                idx0 = idx
            elif not np.array_equal(idx, idx0):
                raise SystemExit(f"{name}/{s} : residus differents des autres modeles")
            data[name] = (pred, gt, mask, rt, geo.residue_analysis(pred, gt, mask, rt, rmsd,
                                                                    gargs))
        _, gt, mask, rt, _ = data[names[0]]
        has = geo.CHI_MASK[rt][:, 0]
        true_chi = geo.chis(gt, rt)
        (lib_aa, code_aa), (lib_bb, code_bb) = lib.predict(
            rt, None if phi_all is None else phi_all[idx0],
            None if phi_all is None else psi_all[idx0])
        lib_d = {LIB_NAMES[0]: dchi(lib_aa, true_chi, rt), LIB_NAMES[1]: dchi(lib_bb, true_chi, rt)}

        acc = {}
        for name in names:
            o = data[name][4]
            acc[name] = (o["dchi"][:, 0] <= args.chi_tol, o["rotamer_ok"])
        for ln, d in lib_d.items():
            if ln == LIB_NAMES[1] and phi_all is None:
                continue
            acc[ln] = (d[:, 0] <= args.chi_tol,
                       np.all((d <= args.chi_tol) | ~geo.CHI_MASK[rt], 1))

        # densite
        dens = {}
        if args.lmdb_path:
            coords = {"vrai": (gt, mask)}
            coords.update({n: (data[n][0], mask) for n in names})
            dens = sidechain_density(args.lmdb_path, args.atom14, idx0, coords,
                                     workers=args.num_workers)
        per_set[s] = (has, acc, dens, rt, code_bb, data)

        print(f"\n=== {s} : {has.sum()} residus avec chi1 ===")
        print(f"{'':>22s}   chi1    rotamere")
        for k, (c1, rot) in acc.items():
            print(f"{k:>22s}  {pct(c1[has])}  {pct(rot[has])}")
            rows_acc.append({"ensemble": s, "source": k, "n": int(has.sum()),
                             "chi1": f"{c1[has].mean():.4f}", "rotamere": f"{rot[has].mean():.4f}"})

        # erreurs du chi1 : rabattement sur la bibliotheque, densite
        print(f"\n  residus a chi1 faux{'':>14s}n   comme la bib.   densite pred > vraie")
        for name in names:
            o = data[name][4]
            wrong = has & ~(o["dchi"][:, 0] <= args.chi_tol)
            pred_chi = geo.chis(data[name][0], rt)
            pb = chi_bins(pred_chi[:, 0], geo.CHI_PI[rt][:, 0])
            lb = np.array([c % 3 for c in code_bb])          # case de chi1 de la bib.
            like_lib = (pb == lb)[wrong]
            more = ((dens[name] > dens["vrai"])[wrong]) if dens else np.array([])
            print(f"  {name:>22s} {wrong.sum():7d}   {pct(like_lib)}       "
                  f"{pct(more) if dens else '   -   '}")
            rows_err.append({"ensemble": s, "modele": name, "n_chi1_faux": int(wrong.sum()),
                             "comme_bib": f"{like_lib.mean():.4f}" if wrong.any() else "",
                             "densite_pred_sup_vraie": f"{more.mean():.4f}" if len(more) else ""})

        # par type
        for r in np.unique(rt[has]):
            sel = has & (rt == r)
            row = {"ensemble": s, "type": STANDARD[r], "n": int(sel.sum())}
            for k, (c1, _) in acc.items():
                row[f"chi1_{k}"] = f"{c1[sel].mean():.4f}"
            rows_type.append(row)

    # densite : quintiles communs a tous les ensembles
    if any(per_set[s][2] for s in per_set):
        allz = np.concatenate([v[2]["vrai"][v[0]] for v in per_set.values() if v[2]])
        allz = allz[np.isfinite(allz)]
        edges = np.quantile(allz, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        edges[0], edges[-1] = -np.inf, np.inf
        labels = [f"Q{q + 1}" for q in range(5)]
        print("\n=== chi1 juste selon la densite de la vraie chaine laterale (score z, quintiles "
              "communs) ===")
        print("  bornes : " + "  ".join(f"{e:.2f}" for e in edges[1:-1]))
        fig_data = {}
        for s, (has, acc, dens, rt, _, _) in per_set.items():
            if not dens:
                continue
            q = np.digitize(dens["vrai"], edges[1:-1])
            print(f"\n  {s}")
            print(f"  {'':>22s}" + "".join(f"{l:>9s}" for l in labels))
            print(f"  {'part des residus':>22s}" + "".join(
                f"{100 * np.mean(q[has] == k):8.1f}%" for k in range(5)))
            fig_data[s] = {}
            for k_name, (c1, _) in acc.items():
                vals = [c1[has & (q == k)].mean() if (has & (q == k)).any() else np.nan
                        for k in range(5)]
                fig_data[s][k_name] = vals
                print(f"  {k_name:>22s}" + "".join(f"{100 * v:8.1f}%" for v in vals))
                for k in range(5):
                    rows_dens.append({"ensemble": s, "source": k_name, "quintile": labels[k],
                                      "z_min": f"{edges[k]:.3f}", "z_max": f"{edges[k + 1]:.3f}",
                                      "n": int((has & (q == k)).sum()),
                                      "chi1": f"{vals[k]:.4f}"})
        plot_by_density(fig_data, labels, os.path.join(args.out_dir, "chi1_by_density.png"))

    for fname, rows in (("accuracy.csv", rows_acc), ("by_density.csv", rows_dens),
                        ("errors.csv", rows_err), ("per_type.csv", rows_type)):
        if rows:
            with open(os.path.join(args.out_dir, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
    print(f"\nEcrit dans {args.out_dir}")


def plot_by_density(fig_data, labels, path):
    from train_utils import (CATEGORICAL, NEUTRAL_MARK, SURFACE, TEXT_PRIMARY, _new_fig,
                             _style_axes, plt)
    sets = list(fig_data)
    if not sets:
        return
    names = list(fig_data[sets[0]])
    fig, axes = _new_fig(1, len(sets), figsize=(3.6 * len(sets) + 0.6, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(names)
    title = {"val": "Validation", "test_1_2A": "Test 1–2 Å",
             "test_2_3A": "Test 2–3 Å", "test_3_4A": "Test 3–4 Å"}
    k_model = 0
    colors = {}
    for n in names:
        if n.startswith("bib."):
            colors[n] = NEUTRAL_MARK if n == LIB_NAMES[1] else "#d3d2cc"
        else:
            colors[n] = CATEGORICAL[k_model]
            k_model += 1
    eng = {LIB_NAMES[0]: "rotamer library (type)", LIB_NAMES[1]: "rotamer library (type + φ/ψ)",
           "hybride": "hybrid", "atome": "atom"}
    for ax, s in zip(axes, sets):
        x = np.arange(len(labels))
        for j, n in enumerate(names):
            ax.bar(x + (j - (len(names) - 1) / 2) * w, 100 * np.array(fig_data[s][n]), w * 0.92,
                   color=colors[n], label=eng.get(n, n))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.set_title(title.get(s, s), loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("side-chain density quintile (Q1 = weakest)", fontsize=8)
    axes[0].set_ylabel("χ1 accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(names), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
