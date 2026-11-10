#!/usr/bin/env python3
"""
Regroupe plusieurs dossiers d'evaluation (sorties de evaluate_models.py ou
evaluate_flow.py) en une seule comparaison, SANS relancer d'inference : les
metriques sont relues dans <dossier>/<modele>/<ensemble>_metrics.json.

Sorties (--out-dir) :
    summary.csv      une ligne par (modele, ensemble), toutes les metriques
    table_<m>.csv    une table par metrique principale : modeles en lignes,
                     ensembles en colonnes (le plus lisible pour comparer)
    summary.md       les memes tables en Markdown (a coller dans un rapport)
    comparison.png   barres groupees par ensemble, une couleur par modele

Doublons : un meme modele evalue dans deux dossiers (meme nom, ou noms
differents mais metriques identiques, ex. cnn_continuous et cnn_e50) n'est
garde qu'une fois ; le doublon est signale. --rename ancien=nouveau renomme
un modele, --exclude en retire.

Usage :
    python merge_evaluations.py runs/evaluation runs/evaluation_v2 runs/evaluation_equi \
        --rename cnn_continuous=cnn_e50 --rename equivariant=equi_origine \
        --out-dir runs/evaluation_all
"""

import argparse
import csv
import glob
import json
import os

from train_utils import plot_model_comparison

DATASET_ORDER = ["val", "test_1_2A", "test_2_3A", "test_3_4A"]
CLASSIF_METRICS = ["accuracy", "f1_macro", "auc_macro", "ap_macro", "top3_acc"]
FLOW_METRICS = ["rmsd_mean", "rmsd_sidechain_mean", "frac_rmsd_lt_1A", "bond_err_mean"]
IGNORED = {"auc_per_class", "ap_per_class", "n_samples", "n_residues", "n_entries", "best_epoch",
           "sample_steps"}


def load_dir(d):
    """{modele: {ensemble: metriques}} pour un dossier d'evaluation."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*", "*_metrics.json"))):
        model = os.path.basename(os.path.dirname(f))
        ds = os.path.basename(f)[: -len("_metrics.json")]
        if ds.startswith("val_best"):      # rapports d'entrainement, pas d'evaluation
            continue
        with open(f) as fh:
            out.setdefault(model, {})[ds] = json.load(fh)
    return out


def same_results(a, b):
    common = set(a) & set(b)
    if not common:
        return False
    for ds in common:
        for k, v in a[ds].items():
            if k in IGNORED or not isinstance(v, (int, float)):
                continue
            if k in b[ds] and abs(v - b[ds][k]) > 1e-9:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="dossiers d'evaluation a regrouper")
    ap.add_argument("--out-dir", default="runs/evaluation_all")
    ap.add_argument("--rename", action="append", default=[], help="ancien=nouveau")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="metriques des tables et de la figure (defaut : selon le type)")
    args = ap.parse_args()
    rename = dict(r.split("=", 1) for r in args.rename)

    models = {}            # nom -> {ensemble: metriques}
    origin = {}
    for d in args.dirs:
        for name, res in load_dir(d).items():
            name = rename.get(name, name)
            if name in args.exclude:
                continue
            dup = next((m for m, r in models.items() if same_results(r, res)), None)
            if dup is not None:
                print(f"  doublon ignore : {name} ({d}) = {dup} ({origin[dup]})")
                # completer les ensembles manquants eventuels
                for ds, m in res.items():
                    models[dup].setdefault(ds, m)
                continue
            if name in models:
                new = f"{name}_{os.path.basename(os.path.normpath(d))}"
                print(f"  nom deja pris avec d'autres resultats : {name} ({d}) -> {new}")
                name = new
            models[name] = res
            origin[name] = d
    if not models:
        raise SystemExit("aucun fichier *_metrics.json trouve")

    datasets = sorted({ds for r in models.values() for ds in r},
                      key=lambda x: (DATASET_ORDER.index(x) if x in DATASET_ORDER else 99, x))
    rows = []
    for name, res in models.items():
        for ds in datasets:
            if ds in res:
                rows.append({"model": name, "dataset": ds, "source": origin[name],
                             **{k: v for k, v in res[ds].items() if not isinstance(v, dict)}})
    is_flow = any("rmsd_mean" in r for r in rows)
    metrics = args.metrics or (FLOW_METRICS if is_flow else CLASSIF_METRICS)

    os.makedirs(args.out_dir, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    md = [f"# Comparaison des modeles\n", "Sources : " + ", ".join(args.dirs) + "\n"]
    for m in metrics:
        table = [[name] + [next((f"{r[m]:.3f}" for r in rows
                                 if r["model"] == name and r["dataset"] == ds and m in r), "")
                           for ds in datasets] for name in models]
        with open(os.path.join(args.out_dir, f"table_{m}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model"] + datasets)
            w.writerows(table)
        md.append(f"\n## {m}\n\n| modele | " + " | ".join(datasets) + " |\n|"
                  + "---|" * (len(datasets) + 1) + "\n"
                  + "\n".join("| " + " | ".join(t) + " |" for t in table) + "\n")
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("".join(md))
    plot_model_comparison(rows, os.path.join(args.out_dir, "comparison.png"),
                          metric_keys=tuple(metrics[:4]))
    print("".join(md[2:]))
    print(f"Ecrit dans {args.out_dir} : summary.csv, table_*.csv, summary.md, comparison.png")


if __name__ == "__main__":
    main()
