#!/usr/bin/env python3
"""
Nature des erreurs et qualite chimique des residus generes par les modeles
de flow, a partir des fichiers *_predictions.npz (aucune inference).

1. Rotameres : angles diedres chi1..chi4 de la chaine laterale, predits vs
   vrais. Un chi est juste si |delta chi| <= --chi-tol degres (40 par defaut,
   seuil usuel ; periode 180 deg pour les chi symetriques : ASP chi2, GLU chi3,
   PHE chi2, TYR chi2). Precision chi1, chi1+chi2, rotamere complet, erreur
   moyenne par chi, et par type d'acide amine.

2. Origine des erreurs. Chaque residu dont le RMSD depasse --fail-rmsd A est
   classe, dans cet ordre, en :
     sens de chaine  N et C inverses ;
     orientation     repere du squelette (N, CA, C) tourne de plus de
                     --frame-tol degres ;
     rotamere        squelette bien oriente mais au moins un chi faux ;
     decalage        squelette et chi justes : atomes deplaces ou geometrie
                     deformee (forme juste mais position imprecise, ou
                     longueurs / angles faux).
   Complements : RMSD de la chaine laterale apres superposition du repere du
   squelette (separe le rotamere de l'orientation) et RMSD apres superposition
   optimale du residu entier (erreur de FORME seule, sans la position).

3. Chimie (prediction et verite, pour reference) : erreur des longueurs de
   liaison (A) et des angles de liaison (degres), chiralite (C-alpha L ;
   C-beta de ILE et THR), planeite des cycles aromatiques (ecart RMS au plan
   moyen, A), et contacts trop proches entre atomes non lies du residu
   (separes par >= 4 liaisons, < --clash A). Un C-alpha inverse est un acide
   amine D : signe d'un squelette mal assemble (N et C echanges autour d'un CB
   bien place, par exemple).

Usage :
    python analyze_flow_geometry.py \\
        --pred joint=runs/evaluation_flow_e100/joint/val_predictions.npz \\
        --pred hybride=runs/evaluation_flow_e100/hybride/val_predictions.npz \\
        --out-dir runs/geometry_val
"""

import argparse
import csv
import os

import numpy as np

from flow_common import ATOM14, CA_SLOT, PERM_SYM, STANDARD, TOPO, _bonds
from train_utils import CATEGORICAL, SURFACE, TEXT_SECONDARY, _new_fig, _style_axes, plt

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

CHI_ATOMS = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"),
            ("CG", "CD", "NE", "CZ")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"),
            ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}
CHI_PI_PERIODIC = {("ASP", 1), ("GLU", 2), ("PHE", 1), ("TYR", 1)}
# chiralite : (centre, a, b, c) ; signe du volume (a-centre).((b-centre)x(c-centre))
CHIRAL = {res: [("CA", "N", "C", "CB")] for res in STANDARD if res != "GLY"}
CHIRAL["ILE"] = CHIRAL["ILE"] + [("CB", "CA", "CG1", "CG2")]
CHIRAL["THR"] = CHIRAL["THR"] + [("CB", "CA", "OG1", "CG2")]
RINGS = {"PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
         "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
         "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
         "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2")}
ERROR_KINDS = ["sens de chaine", "orientation", "rotamere", "decalage"]


def _slots(res, names):
    return [ATOM14[res].index(a) for a in names]


def _build():
    n = len(STANDARD)
    chi = np.zeros((n, 4, 4), np.int64)
    chi_mask = np.zeros((n, 4), bool)
    chi_pi = np.zeros((n, 4), bool)
    bonds, angles, chiral, rings = [], [], [], []
    for r, res in enumerate(STANDARD):
        for k, quad in enumerate(CHI_ATOMS.get(res, [])):
            chi[r, k] = _slots(res, quad)
            chi_mask[r, k] = True
            chi_pi[r, k] = (res, k) in CHI_PI_PERIODIC
        bl = [tuple(_slots(res, b)) for b in _bonds(res)]
        bonds.append(bl)
        adj = {}
        for a, b in bl:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        angles.append([(a, c, b) for c, nb in adj.items() for i, a in enumerate(nb)
                       for b in nb[i + 1:]])
        chiral.append([tuple(_slots(res, q)) for q in CHIRAL.get(res, [])])
        rings.append(_slots(res, RINGS[res]) if res in RINGS else None)
    return chi, chi_mask, chi_pi, bonds, angles, chiral, rings


CHI, CHI_MASK, CHI_PI, BONDS, ANGLES, CHIRALS, RING_SLOTS = _build()

# ---------------------------------------------------------------------------
# Geometrie
# ---------------------------------------------------------------------------


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.degrees(np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1)))


def angle_diff(a, b, period=360.0):
    d = np.abs((a - b + period / 2) % period - period / 2)
    return d


def frames(x):
    """Repere local (N, CA, C) : matrices (N,3,3) (lignes = axes)."""
    n, ca, c = x[:, 0], x[:, CA_SLOT], x[:, 2]
    e1 = c - ca
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    u = n - ca
    e2 = u - (u * e1).sum(-1, keepdims=True) * e1
    e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
    return np.stack([e1, e2, np.cross(e1, e2)], 1)


def kabsch_rmsd(p, q, m):
    """RMSD apres superposition optimale (rotation + translation) de p sur q,
    atomes m, residu par residu."""
    out = np.full(len(p), np.nan)
    for i in range(len(p)):
        a, b = p[i, m[i]], q[i, m[i]]
        if len(a) < 3:
            continue
        a, b = a - a.mean(0), b - b.mean(0)
        u, _, vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1, 1, d]) @ vt
        out[i] = np.sqrt(((a @ r - b) ** 2).sum(-1).mean())
    return out


def load(path):
    d = np.load(path)
    pred, gt, mask, rt = d["pred"].astype(np.float64), d["gt"].astype(np.float64), \
        d["mask"].astype(bool), d["restype"]
    # numerotation symetrique la plus proche de la prediction (comme le RMSD)
    m = mask.copy()
    m[:, CA_SLOT] = False
    gsw = np.take_along_axis(gt, PERM_SYM[rt][..., None], 1)
    use = (((pred - gsw) ** 2).sum(-1) * m).sum(1) < (((pred - gt) ** 2).sum(-1) * m).sum(1)
    gt = np.where(use[:, None, None], gsw, gt)
    return pred, gt, mask, rt, d["rmsd"]


def chis(x, rt):
    q = CHI[rt]                                                        # (N,4,4)
    p = [np.take_along_axis(x, q[:, :, j][..., None].repeat(3, -1), 1) for j in range(4)]
    return dihedral(*p)                                                # (N,4)


def residue_analysis(pred, gt, mask, rt, rmsd, args):
    N = len(rt)
    out = {"rmsd": rmsd}
    # chi
    cp, ct = chis(pred, rt), chis(gt, rt)
    cm = CHI_MASK[rt]
    dchi = np.where(CHI_PI[rt], angle_diff(cp, ct, 180.0), angle_diff(cp, ct))
    dchi = np.where(cm, dchi, np.nan)
    out["dchi"] = dchi
    ok = (dchi <= args.chi_tol) | ~cm
    out["chi_ok"] = ok
    out["rotamer_ok"] = ok.all(1)
    # repere du squelette
    fp, ft = frames(pred), frames(gt)
    rel = np.einsum("nij,nkj->nik", fp, ft)
    out["frame_err"] = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2,
                                                    -1, 1)))
    n_c = np.linalg.norm(pred[:, 0] - gt[:, 2], axis=-1) < 0.8
    c_n = np.linalg.norm(pred[:, 2] - gt[:, 0], axis=-1) < 0.8
    out["flip"] = n_c & c_n
    # chaine laterale dans le repere local du squelette
    lp = np.einsum("nij,naj->nai", fp, pred - pred[:, CA_SLOT:CA_SLOT + 1])
    lt = np.einsum("nij,naj->nai", ft, gt - gt[:, CA_SLOT:CA_SLOT + 1])
    sc = mask.copy()
    sc[:, :4] = False
    d2 = ((lp - lt) ** 2).sum(-1)
    n_sc = sc.sum(1)
    out["sc_local_rmsd"] = np.where(n_sc > 0, np.sqrt((d2 * sc).sum(1) / np.maximum(n_sc, 1)),
                                    np.nan)
    out["shape_rmsd"] = kabsch_rmsd(pred, gt, mask)
    # classement des echecs
    fail = rmsd > args.fail_rmsd
    kind = np.full(N, "", dtype=object)
    rem = fail.copy()
    for name, cond in (("sens de chaine", out["flip"]),
                       ("orientation", out["frame_err"] > args.frame_tol),
                       ("rotamere", ~out["rotamer_ok"]),
                       ("decalage", np.ones(N, bool))):
        sel = rem & cond
        kind[sel] = name
        rem &= ~sel
    out["fail_kind"] = kind
    out["fail"] = fail
    return out


def chemistry(x, mask, rt, gt=None, clash=2.5):
    """Mesures chimiques par residu ; avec gt : erreurs par rapport a gt."""
    N = len(rt)
    res = {k: np.full(N, np.nan) for k in ("bond_err", "angle_err", "ring_planarity")}
    res["chiral_bad"] = np.zeros(N, bool)
    res["chiral_ca_bad"] = np.zeros(N, bool)
    res["clash"] = np.zeros(N, bool)
    far = TOPO[rt] >= 4
    dist = np.linalg.norm(x[:, :, None] - x[:, None], axis=-1)
    m2 = mask[:, :, None] & mask[:, None, :] & far
    res["clash"] = ((dist < clash) & m2).any((1, 2))
    for r in np.unique(rt):
        sel = np.where(rt == r)[0]
        xs = x[sel]
        b = np.array(BONDS[r])
        lb = np.linalg.norm(xs[:, b[:, 0]] - xs[:, b[:, 1]], axis=-1)
        a = np.array(ANGLES[r])
        v1, v2 = xs[:, a[:, 0]] - xs[:, a[:, 1]], xs[:, a[:, 2]] - xs[:, a[:, 1]]
        ang = np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / (
            np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)), -1, 1)))
        if gt is not None:
            gs = gt[sel]
            lg = np.linalg.norm(gs[:, b[:, 0]] - gs[:, b[:, 1]], axis=-1)
            w1, w2 = gs[:, a[:, 0]] - gs[:, a[:, 1]], gs[:, a[:, 2]] - gs[:, a[:, 1]]
            ag = np.degrees(np.arccos(np.clip((w1 * w2).sum(-1) / (
                np.linalg.norm(w1, axis=-1) * np.linalg.norm(w2, axis=-1)), -1, 1)))
            res["bond_err"][sel] = np.abs(lb - lg).mean(1)
            res["angle_err"][sel] = np.abs(ang - ag).mean(1)
        for c, p, q, s in CHIRALS[r]:
            vol = ((xs[:, p] - xs[:, c]) * np.cross(xs[:, q] - xs[:, c],
                                                     xs[:, s] - xs[:, c])).sum(-1)
            ref = 1.0
            if gt is not None:
                gs = gt[sel]
                ref = ((gs[:, p] - gs[:, c]) * np.cross(gs[:, q] - gs[:, c],
                                                         gs[:, s] - gs[:, c])).sum(-1)
            bad = np.sign(vol) != np.sign(ref)
            res["chiral_bad"][sel] |= bad
            if c == CA_SLOT:
                res["chiral_ca_bad"][sel] |= bad
        if RING_SLOTS[r] is not None:
            ring = xs[:, RING_SLOTS[r]]
            ring = ring - ring.mean(1, keepdims=True)
            sv = np.linalg.svd(ring, compute_uv=False)
            res["ring_planarity"][sel] = sv[:, -1] / np.sqrt(ring.shape[1])
    return res


# ---------------------------------------------------------------------------

def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{100 * x.mean():5.1f} %" if x.size else "    -"


def mean(x, fmt="{:.3f}"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return fmt.format(x.mean()) if x.size else "-"


def plot_fail_kinds(rows, path, title):
    """Barres empilees : part des residus par type d'erreur (sur tous les residus)."""
    names = [r["modele"] for r in rows]
    fig, ax = _new_fig(figsize=(max(5.5, 1.3 * len(names) + 2.5), 3.8))
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    bottom = np.zeros(len(names))
    for k, kind in enumerate(ERROR_KINDS):
        v = np.array([100 * r[f"part_{kind}"] for r in rows])
        ax.bar(range(len(names)), v, bottom=bottom, color=CATEGORICAL[k], width=0.6,
               label=kind, edgecolor=SURFACE, linewidth=0.8)
        for i, (b, h) in enumerate(zip(bottom, v)):
            if h >= 1.5:
                ax.text(i, b + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=7,
                        color="white")
        bottom += v
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("% des residus (RMSD > seuil)")
    ax.set_title(title, loc="left", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    np.seterr(invalid="ignore", divide="ignore")     # chi absents (masques ensuite)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, help="nom=chemin.npz")
    ap.add_argument("--chi-tol", type=float, default=40.0)
    ap.add_argument("--frame-tol", type=float, default=30.0)
    ap.add_argument("--fail-rmsd", type=float, default=1.0)
    ap.add_argument("--clash", type=float, default=2.5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    res, truth_chem = {}, None
    for item in args.pred:
        name, path = item.split("=", 1)
        pred, gt, mask, rt, rmsd = load(path)
        a = residue_analysis(pred, gt, mask, rt, rmsd, args)
        a["chem"] = chemistry(pred, mask, rt, gt, args.clash)
        a["rt"] = rt
        res[name] = a
        if truth_chem is None:
            truth_chem = chemistry(gt, mask, rt, None, args.clash)
            truth_rt = rt
    names = list(res)
    w = max(10, max(len(n) for n in names))

    # 1. rotameres
    print(f"\n1. Rotameres (chi juste si |delta| <= {args.chi_tol:g} deg)")
    print(f"{'':>{w}s}  {'chi1':>7s} {'chi1+2':>7s} {'complet':>7s}   "
          + "  ".join(f"|dchi{k + 1}|" for k in range(4)))
    rot_rows = []
    for n, a in res.items():
        has1, has2 = CHI_MASK[a["rt"]][:, 0], CHI_MASK[a["rt"]][:, 1]
        c1 = a["chi_ok"][has1, 0]
        c12 = (a["chi_ok"][:, 0] & a["chi_ok"][:, 1])[has2]
        full = a["rotamer_ok"][has1]
        row = {"modele": n, "chi1": c1.mean(), "chi1+2": c12.mean(), "rotamere": full.mean(),
               **{f"dchi{k + 1}": np.nanmean(a["dchi"][:, k]) for k in range(4)}}
        rot_rows.append(row)
        print(f"{n:>{w}s}  {pct(c1)} {pct(c12)} {pct(full)}   "
              + "  ".join(f"{mean(a['dchi'][:, k], '{:5.1f}'):>7s}" for k in range(4)))

    print(f"\n   chi1 juste par type d'acide amine")
    types = [r for r in STANDARD if r in CHI_ATOMS]
    print(f"{'type':>5s}  " + "  ".join(f"{n:>{w}s}" for n in names))
    per_type = []
    for t in types:
        ri = STANDARD.index(t)
        row = {"type": t}
        cells = []
        for n, a in res.items():
            s = a["rt"] == ri
            row[n] = a["chi_ok"][s, 0].mean() if s.any() else np.nan
            cells.append(f"{pct(a['chi_ok'][s, 0]):>{w}s}")
        per_type.append(row)
        print(f"{t:>5s}  " + "  ".join(cells))

    # 2. origine des erreurs
    print(f"\n2. Origine des erreurs : residus a RMSD > {args.fail_rmsd:g} A, classes en "
          f"sens de chaine > orientation (> {args.frame_tol:g} deg) > rotamere > decalage")
    print(f"{'':>{w}s}  {'echecs':>7s}  " + "  ".join(f"{k:>14s}" for k in ERROR_KINDS)
          + f"  {'repere':>7s} {'sc local':>8s} {'forme':>6s}")
    fail_rows = []
    for n, a in res.items():
        f = a["fail"]
        row = {"modele": n, "echecs": f.mean(),
               **{f"part_{k}": np.mean(a["fail_kind"] == k) for k in ERROR_KINDS},
               "frame_err_deg": np.nanmean(a["frame_err"]),
               "sc_local_rmsd": np.nanmean(a["sc_local_rmsd"]),
               "shape_rmsd": np.nanmean(a["shape_rmsd"])}
        fail_rows.append(row)
        share = [np.mean(a["fail_kind"][f] == k) if f.any() else np.nan for k in ERROR_KINDS]
        print(f"{n:>{w}s}  {pct(f)}  " + "  ".join(
            f"{100 * row[f'part_{k}']:5.1f} % ({100 * s:3.0f}%)" for k, s in zip(ERROR_KINDS, share))
              + f"  {row['frame_err_deg']:6.1f}d {row['sc_local_rmsd']:8.3f} {row['shape_rmsd']:6.3f}")
    print("   (x % (y%) : x % de TOUS les residus, y % des echecs ; repere : erreur moyenne "
          "d'orientation du squelette ; sc local : RMSD chaine laterale dans le repere du "
          "squelette ; forme : RMSD apres superposition optimale du residu)")

    # 3. chimie
    print(f"\n3. Chimie (erreurs par rapport a la vraie structure)")
    print(f"{'':>{w}s}  {'liaisons':>9s} {'angles':>8s} {'CA (D)':>8s} {'chiralite':>9s} "
          f"{'cycles':>8s} {'contacts':>8s}")
    tc = truth_chem
    print(f"{'vrai':>{w}s}  {'':>9s} {'':>8s} {'':>8s} {'':>9s} {mean(tc['ring_planarity']):>8s} "
          f"{pct(tc['clash']):>8s}")
    chem_rows = []
    for n, a in res.items():
        c = a["chem"]
        row = {"modele": n, "bond_err_A": np.nanmean(c["bond_err"]),
               "angle_err_deg": np.nanmean(c["angle_err"]),
               "chiral_ca_bad": c["chiral_ca_bad"].mean(), "chiral_bad": c["chiral_bad"].mean(),
               "ring_planarity_A": np.nanmean(c["ring_planarity"]), "clash": c["clash"].mean()}
        chem_rows.append(row)
        print(f"{n:>{w}s}  {mean(c['bond_err']):>9s} {mean(c['angle_err'], '{:.2f}'):>8s} "
              f"{pct(c['chiral_ca_bad']):>8s} {pct(c['chiral_bad']):>9s} "
              f"{mean(c['ring_planarity']):>8s} {pct(c['clash']):>8s}")
    print("   (liaisons : |ecart| moyen des longueurs, A ; angles : |ecart| moyen des angles "
          "de liaison, deg ; CA (D) : C-alpha inverse (acide amine D) ; chiralite : un centre "
          "chiral quelconque inverse (CA ou CB de ILE/THR) ; cycles : "
          f"ecart RMS au plan, A ; contacts : residus avec 2 atomes non lies a < {args.clash:g} A)")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for fname, rows in (("rotamers.csv", rot_rows), ("chi1_per_type.csv", per_type),
                            ("error_origin.csv", fail_rows), ("chemistry.csv", chem_rows)):
            with open(os.path.join(args.out_dir, fname), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
        plot_fail_kinds(fail_rows, os.path.join(args.out_dir, "error_origin.png"),
                        f"Origine des erreurs (RMSD > {args.fail_rmsd:g} A)")
        print(f"\nEcrit dans {args.out_dir}")


if __name__ == "__main__":
    main()
