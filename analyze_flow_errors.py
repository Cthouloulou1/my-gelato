#!/usr/bin/env python3
"""
Analyse fine des erreurs de placement des modeles de flow matching, a partir
des fichiers *_predictions.npz (train_flow.py : val_best_predictions.npz ;
evaluate_flow.py : <ensemble>_predictions.npz). Aucune inference.

Pour comparer les modes joint / un atome a la fois / backbone / hybride :

  1. erreur moyenne par RANG de l'atome dans le parcours des liaisons depuis
     le C-alpha (1 = N, 2 = C, 3 = CB (O pour GLY), 4 = O, 5 = CG, ...),
     le meme ordre pour tous les modeles pour qu'ils soient comparables ;
     l'erreur "locale" est celle du vecteur parent -> atome (placement de
     l'atome par rapport a son parent, sans l'erreur heritee du parent) ;
  2. sens de la chaine : fraction des residus dont N et C sont inverses
     (prediction de N a < 0.8 A du vrai C et inversement), GLY a part ;
  3. exposure bias : erreur locale de l'atome en fonction de l'erreur de son
     parent (rangs >= 5). Si elle croit beaucoup plus vite pour un modele
     sequentiel que pour le joint, le modele "decroche" quand son contexte
     est faux.

Les atomes symetriques (ASP, GLU, PHE, TYR) sont renumerotes au mieux, par
residu, comme pour le RMSD.

Usage :
    python analyze_flow_errors.py \
        --pred joint=runs/flow_cnn/val_best_predictions.npz \
        --pred atome=runs/flow_cnn_ar/val_best_predictions.npz \
        --pred hybride=runs/flow_cnn_ar_hybrid/val_best_predictions.npz \
        --out-dir runs/analysis_val
"""

import argparse
import csv
import os

import numpy as np

from flow_common import ATOM14, CA_SLOT, PARENT, PERM_SYM, RANK, STANDARD

GLY = STANDARD.index("GLY")
PARENT_BINS = [0.0, 0.5, 1.0, 2.0, 4.0, np.inf]


def load(path):
    d = np.load(path)
    pred, gt, mask, rt = d["pred"], d["gt"], d["mask"].astype(bool), d["restype"]
    m = mask.copy()
    m[:, CA_SLOT] = False
    gsw = np.take_along_axis(gt, PERM_SYM[rt][..., None], 1)
    e = ((pred - gt) ** 2).sum(-1)
    esw = ((pred - gsw) ** 2).sum(-1)
    use = (esw * m).sum(1) < (e * m).sum(1)
    gt = np.where(use[:, None, None], gsw, gt)
    return pred, gt, mask, m, rt


def errors(pred, gt, rt):
    par = PARENT[rt][..., None].repeat(3, -1)
    pp, gp = np.take_along_axis(pred, par, 1), np.take_along_axis(gt, par, 1)
    err = np.linalg.norm(pred - gt, axis=-1)
    local = np.linalg.norm((pred - pp) - (gt - gp), axis=-1)
    perr = np.linalg.norm(pp - gp, axis=-1)
    return err, local, perr


def chain_flip(pred, gt, sel):
    n_c = np.linalg.norm(pred[sel, 0] - gt[sel, 2], axis=-1) < 0.8
    c_n = np.linalg.norm(pred[sel, 2] - gt[sel, 0], axis=-1) < 0.8
    return float(np.mean(n_c & c_n)) if sel.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, help="nom=chemin.npz")
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--out-dir", default=None, help="ecrit les tables en CSV si donne")
    args = ap.parse_args()

    res = {}
    for item in args.pred:
        name, path = item.split("=", 1)
        pred, gt, mask, m, rt = load(path)
        err, local, perr = errors(pred, gt, rt)
        res[name] = dict(pred=pred, gt=gt, mask=mask, m=m, rt=rt, err=err, local=local,
                         perr=perr, rank=RANK[rt])
    names = list(res)
    n_res = {n: len(r["rt"]) for n, r in res.items()}
    if len(set(n_res.values())) > 1:
        print(f"Attention : nombres de residus differents {n_res} (ensembles differents ?)")

    # 1. erreur par rang
    ex = res[names[0]]
    rank_rows = []
    for k in range(1, args.max_rank + 1):
        row = {"rang": k, "n_atomes": int((ex["m"] & (ex["rank"] == k)).sum())}
        for n, r in res.items():
            sel = r["m"] & (r["rank"] == k)
            row[f"err_{n}"] = float(r["err"][sel].mean()) if sel.any() else np.nan
            row[f"local_{n}"] = float(r["local"][sel].mean()) if sel.any() else np.nan
        if row["n_atomes"] > 0:
            rank_rows.append(row)
    example = {}
    for res_name in ("ALA", "ARG", "PHE"):
        ri = STANDARD.index(res_name)
        example[res_name] = {int(RANK[ri, s]): a for s, a in enumerate(ATOM14[res_name])}
    print("\n1. Erreur moyenne par rang de placement (A) ; entre parentheses : erreur locale")
    print("   (rangs : " + ", ".join(f"{k}={example['ARG'].get(k, '?')}"
                                     for k in range(1, 8)) + ", ... pour ARG ; 3 = O pour GLY)")
    print(f"rang  n_atomes ({names[0]})  " + "  ".join(f"{n:>16s}" for n in names))
    for row in rank_rows:
        print(f"{row['rang']:4d}  {row['n_atomes']:{11 + len(names[0])}d}  "
              + "  ".join(f"{row[f'err_{n}']:7.3f} ({row[f'local_{n}']:5.3f})" for n in names))

    # 2. sens de la chaine
    flip_rows = []
    print("\n2. Residus avec N et C inverses (sens de la chaine)")
    for n, r in res.items():
        g = r["rt"] == GLY
        row = {"modele": n, "gly": chain_flip(r["pred"], r["gt"], g),
               "autres": chain_flip(r["pred"], r["gt"], ~g), "n_gly": int(g.sum())}
        flip_rows.append(row)
        print(f"  {n:>16s} : GLY {row['gly']:6.1%} (n={row['n_gly']})   autres {row['autres']:6.1%}")

    # 3. erreur locale vs erreur du parent
    bin_rows = []
    print("\n3. Erreur locale de l'atome selon l'erreur de son parent (rangs >= 5)")
    labels = [f"{a:g}-{b:g} A" if np.isfinite(b) else f">{a:g} A"
              for a, b in zip(PARENT_BINS[:-1], PARENT_BINS[1:])]
    print(f"{'parent':>16s}  " + "  ".join(f"{lab:>9s}" for lab in labels))
    for n, r in res.items():
        sel = r["m"] & (r["rank"] >= 5)
        row = {"modele": n}
        for lab, a, b in zip(labels, PARENT_BINS[:-1], PARENT_BINS[1:]):
            s = sel & (r["perr"] >= a) & (r["perr"] < b)
            row[lab] = float(r["local"][s].mean()) if s.any() else np.nan
            row[f"n {lab}"] = int(s.sum())
        bin_rows.append(row)
        print(f"{n:>16s}  " + "  ".join(f"{row[lab]:9.3f}" for lab in labels))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for fname, rows in (("error_by_rank.csv", rank_rows), ("chain_flip.csv", flip_rows),
                            ("local_vs_parent_error.csv", bin_rows)):
            with open(os.path.join(args.out_dir, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"\nTables ecrites dans {args.out_dir}")


if __name__ == "__main__":
    main()
