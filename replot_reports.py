#!/usr/bin/env python3
"""
Regenere les rapports (matrice de confusion, ROC, AUC par classe, metriques,
rapport par classe) a partir des fichiers *_predictions.npz deja produits
par les entrainements / evaluate_models.py, sans relancer d'inference.
Utile apres une modification des figures dans train_utils.py.

Usage :
    python replot_reports.py runs
    python replot_reports.py runs/evaluation/cnn_continuous/test_1_2A_predictions.npz
"""

import argparse
import glob
import json
import os

import numpy as np

from train_utils import save_final_reports


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="fichiers *_predictions.npz ou dossiers (recherche recursive)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*_predictions.npz"), recursive=True)
        else:
            files.append(p)

    for f in sorted(files):
        out_dir = os.path.dirname(f)
        prefix = os.path.basename(f)[: -len("_predictions.npz")]
        # titre : <modele> - <ensemble>, comme a la creation
        title = f"{os.path.basename(out_dir)} - {prefix}"
        extra = {}
        mpath = os.path.join(out_dir, f"{prefix}_metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as fh:
                old = json.load(fh)
            extra = {k: old[k] for k in ("best_epoch", "n_entries") if k in old}
        z = np.load(f)
        save_final_reports(z["targets"], z["probs"], out_dir, prefix,
                           title=title, extra=extra or None)
        print(f"regenere : {out_dir}/{prefix}_*")


if __name__ == "__main__":
    main()
