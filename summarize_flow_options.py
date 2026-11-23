#!/usr/bin/env python3
"""
Tableau recapitulatif de tous les modeles de flow : un modele par ligne, les
mesures principales en colonnes. Aucune inference, aucune analyse : lit les
fichiers deja produits par les autres scripts.

Sources (par defaut, tout ce qui existe dans runs/) :
    RMSD par ensemble, squelette, < 1 A
        runs/evaluation_flow*_e100/summary.csv          (evaluate_flow.py)
        runs/seeds_flow_e100/seeds_summary.csv          (summarize_seeds.py) :
            remplace la valeur d'une seed par la moyenne +/- ecart-type quand
            le modele a plusieurs seeds
    sens de chaine (GLY), liaison peptidique
        runs/analysis_flow*_e100/<ensemble>/chain_flip.csv, peptide_bond.csv
                                                         (analyze_flow_errors.py)
    echecs, orientation, chi1, acides amines D, clashes
        runs/geometry*_e100/<ensemble>/error_origin.csv, rotamers.csv, chemistry.csv
                                                         (analyze_flow_geometry.py)
Les colonnes detaillees portent sur --set (defaut test_3_4A). Les analyses
(GLY, D, chi1...) ne portent que sur la seed analysee (en general 42).

Ecrit --out.csv (valeurs brutes) et --out.md (tableau mis en forme, meilleur
de chaque colonne en gras), et affiche le tableau.

Usage :
    python summarize_flow_options.py
    python summarize_flow_options.py --set test_2_3A --out runs/flow_options_2_3A
"""

import argparse
import csv
import glob
import math
import os
import re

SETS = ["val", "test_1_2A", "test_2_3A", "test_3_4A"]
SET_LABEL = {"val": "val", "test_1_2A": "1-2 A", "test_2_3A": "2-3 A", "test_3_4A": "3-4 A"}
ORDER = ["joint", "atome", "backbone", "hybride", "backbone_nco", "hybrid_nco",
         "hybrid_pep", "hybrid_knn_pep", "hybrid_chain_pep"]
SEED_RE = re.compile(r"_s\d+$")

# colonnes detaillees (sur --set) : cle, titre, plus bas = mieux, format
DETAIL = [
    ("rmsd_backbone_mean", "squelette", True, "{:.3f}"),
    ("frac_rmsd_lt_1A", "< 1 A", False, "pct"),
    ("echecs", "echecs", True, "pct"),
    ("part_orientation", "orientation", True, "pct"),
    ("gly_flip", "GLY N/C inv.", True, "pct"),
    ("chiral_ca_bad", "acides D", True, "pct"),
    ("chi1", "chi1 juste", False, "pct"),
    ("clash", "clashes", True, "pct"),
    ("err_C-N", "err. C-N (A)", True, "{:.3f}"),
    ("frac_omega_lt_150", "omega < 150", True, "pct"),
]


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def put(table, model, key, value, std=None, n=None, overwrite=False):
    if value is None:
        return
    row = table.setdefault(model, {})
    if key in row and not overwrite:
        return
    row[key] = (value, std, n)


def group_of(model):
    return "contexte" if any(k in model for k in ("pep", "knn", "chain")) else "ordre"


def sort_models(models):
    return sorted(models, key=lambda m: (group_of(m) != "ordre",
                                         ORDER.index(m) if m in ORDER else 99, m))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="test_3_4A", choices=SETS,
                    help="ensemble des colonnes detaillees")
    ap.add_argument("--summary", nargs="*", default=None, help="summary.csv de evaluate_flow.py")
    ap.add_argument("--seeds", default="runs/seeds_flow_e100/seeds_summary.csv")
    ap.add_argument("--analysis", nargs="*", default=None, help="dossiers de analyze_flow_errors.py")
    ap.add_argument("--geometry", nargs="*", default=None,
                    help="dossiers de analyze_flow_geometry.py")
    ap.add_argument("--models", nargs="*", default=None, help="ne garder que ces modeles")
    ap.add_argument("--out", default="runs/flow_options_summary")
    args = ap.parse_args()

    summaries = args.summary if args.summary is not None else sorted(
        glob.glob("runs/evaluation_flow*_e100/summary.csv"))
    analyses = args.analysis if args.analysis is not None else sorted(
        d for d in glob.glob("runs/analysis_flow*_e100") if os.path.isdir(d))
    geometries = args.geometry if args.geometry is not None else sorted(
        d for d in glob.glob("runs/geometry*_e100") if os.path.isdir(d))

    T = {}
    # 1. evaluations (une seed)
    for p in summaries:
        for r in read_csv(p):
            m = r["model"]
            if SEED_RE.search(m):
                continue
            ds = r["dataset"]
            put(T, m, f"rmsd_mean@{ds}", fnum(r["rmsd_mean"]), n=1)
            if ds == args.set:
                for k in ("rmsd_backbone_mean", "frac_rmsd_lt_1A"):
                    put(T, m, k, fnum(r[k]), n=1)
    # 2. seeds : moyenne +/- ecart-type (remplace)
    if os.path.exists(args.seeds):
        for r in read_csv(args.seeds):
            m, ds, met, n = r["variante"], r["ensemble"], r["metrique"], int(r["n"])
            if n < 2:
                continue
            mean, std = fnum(r["moyenne"]), fnum(r["ecart_type"])
            if met == "rmsd_mean":
                put(T, m, f"rmsd_mean@{ds}", mean, std, n, overwrite=True)
            elif ds == args.set and met in ("rmsd_backbone_mean", "frac_rmsd_lt_1A"):
                put(T, m, met, mean, std, n, overwrite=True)
    # 3. analyses (sens de chaine, liaison peptidique)
    for d in analyses:
        for name, cols in (("chain_flip.csv", {"gly": "gly_flip"}),
                           ("peptide_bond.csv", {"err_C-N": "err_C-N",
                                                 "frac_omega_lt_150": "frac_omega_lt_150"})):
            p = os.path.join(d, args.set, name)
            if os.path.exists(p):
                for r in read_csv(p):
                    for src, dst in cols.items():
                        put(T, r["modele"], dst, fnum(r.get(src)))
    # 4. geometrie (echecs, chi1, chimie)
    for d in geometries:
        for name, cols in (("error_origin.csv", {"echecs": "echecs",
                                                 "part_sens de chaine": "part_sens",
                                                 "part_orientation": "part_orientation",
                                                 "part_rotamere": "part_rotamere",
                                                 "part_decalage": "part_decalage"}),
                           ("rotamers.csv", {"chi1": "chi1"}),
                           ("chemistry.csv", {"chiral_ca_bad": "chiral_ca_bad",
                                              "clash": "clash"})):
            p = os.path.join(d, args.set, name)
            if os.path.exists(p):
                for r in read_csv(p):
                    if r["modele"] == "vrai":
                        continue
                    for src, dst in cols.items():
                        put(T, r["modele"], dst, fnum(r.get(src)))

    models = sort_models([m for m in T if any(k.startswith("rmsd_mean@") for k in T[m])])
    if args.models:
        models = [m for m in models if m in args.models]
    if not models:
        raise SystemExit("aucun modele trouve (summary.csv)")

    cols = ([(f"rmsd_mean@{s}", f"RMSD {SET_LABEL[s]}", True, "{:.3f}") for s in SETS]
            + DETAIL)
    # dans le CSV seulement (pour plot_flow_options.py) : causes d'echec
    csv_cols = cols + [(k, k, True, "pct") for k in ("part_sens", "part_rotamere",
                                                       "part_decalage")]

    # CSV brut
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modele", "groupe", "n_seeds"] + [
            (k if "@" in k else f"{k}@{args.set}") + suffix
            for k, *_ in csv_cols for suffix in ("", "_std")])
        for m in models:
            row = T[m]
            n = row.get(f"rmsd_mean@{SETS[0]}", (None, None, 1))[2] or 1
            out = [m, group_of(m), n]
            for k, *_ in csv_cols:
                v, s, _ = row.get(k, (None, None, None))
                out += ["" if v is None else f"{v:.6g}", "" if s is None else f"{s:.6g}"]
            w.writerow(out)

    # Markdown
    best = {}
    for k, _, low, _ in cols:
        vals = [T[m][k][0] for m in models if k in T[m]]
        if vals:
            best[k] = min(vals) if low else max(vals)

    def cell(m, k, fmt):
        if k not in T[m]:
            return "-"
        v, s, _ = T[m][k]
        if fmt == "pct":
            txt = f"{100 * v:.1f} %" + (f" ± {100 * s:.1f}" if s is not None else "")
        else:
            txt = fmt.format(v) + (f" ± {fmt.format(s)}" if s is not None else "")
        rnd = (lambda x: round(100 * x, 1)) if fmt == "pct" else (lambda x: round(x, 3))
        return f"**{txt}**" if rnd(v) == rnd(best[k]) else txt   # meilleur a l'affichage

    head = ["modele", "seeds"] + [c[1] for c in cols]
    L = [f"# Flow matching : effet des options",
         "",
         f"RMSD moyen (A) sur chaque ensemble ; colonnes suivantes sur **{args.set}**. "
         "Meilleur de chaque colonne en gras. ± : ecart-type entre seeds (RMSD, squelette, "
         "< 1 A) ; les autres colonnes viennent de la seed analysee.",
         "",
         "| " + " | ".join(head) + " |",
         "|" + "---|" * len(head)]
    current = None
    for m in models:
        g = group_of(m)
        if g != current:
            label = ("Ordre de generation (sans contexte)" if g == "ordre"
                     else "Contexte C-alpha et loss peptidique")
            L.append(f"| *{label}* |" + " |" * (len(head) - 1))
            current = g
        n = T[m].get(f"rmsd_mean@{SETS[0]}", (None, None, 1))[2] or 1
        L.append("| " + " | ".join([m, str(n)] + [cell(m, k, fmt) for k, _, _, fmt in cols])
                 + " |")
    L += ["", "Sources : " + ", ".join(summaries + ([args.seeds] if os.path.exists(args.seeds)
                                                     else []) + analyses + geometries)]
    md = "\n".join(L) + "\n"
    with open(args.out + ".md", "w") as f:
        f.write(md)
    print(md)
    print(f"Ecrit : {args.out}.csv, {args.out}.md")


if __name__ == "__main__":
    main()
