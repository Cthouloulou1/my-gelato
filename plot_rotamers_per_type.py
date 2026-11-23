#!/usr/bin/env python3
"""
Precision des rotameres (tous les chi a +/- --chi-tol degres) par type
d'acide amine, a partir des echantillons de sample_rotamers.py
(<ensemble>_samples.npz) : random (un tirage, = evaluation normale),
consensus (medoide des K) et oracle (le meilleur des K selon la verite, borne
haute). Labels en anglais.

Ecrit dans --out-dir : rotamer_accuracy_per_type.png (un panneau par
ensemble, acides amines tries selon random sur le dernier ensemble) et
rotamer_accuracy_per_type.csv.

Usage :
    python plot_rotamers_per_type.py
    python plot_rotamers_per_type.py --sets val test_3_4A
"""

import argparse
import csv
import os

import numpy as np

import analyze_flow_geometry as geo
from flow_common import STANDARD
from sample_rotamers import evaluate
from train_utils import (CATEGORICAL, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _new_fig,
                         _style_axes, plt)

SET_TITLE = {"val": "Validation", "test_1_2A": "Test 1–2 Å",
             "test_2_3A": "Test 2–3 Å", "test_3_4A": "Test 3–4 Å"}
SERIES = [("random", "random (one sample)", CATEGORICAL[0]),
          ("consensus", "consensus of K", CATEGORICAL[1]),
          ("oracle", "oracle: best of K (upper bound)", "#c9c8c2")]


def per_set(path, chi_tol):
    z = np.load(path)
    X, gt, mask, rt = z["samples"], z["gt"], z["mask"].astype(bool), z["restype"]
    K, N = X.shape[:2]
    rot = np.zeros((K, N), bool)
    for k in range(K):
        _, rot[k], _ = evaluate(X[k], gt, mask, rt, chi_tol)
    sc = mask.copy()
    sc[:, :4] = False
    pair = np.zeros((K, K, N))
    for k in range(K):
        d2 = ((X[k][None] - X) ** 2).sum(-1)
        pair[k] = np.sqrt((d2 * sc).sum(-1) / np.maximum(sc.sum(-1), 1))
    cons = pair.sum(1).argmin(0)
    ar = np.arange(N)
    return rt, {"random": rot[0], "consensus": rot[cons, ar], "oracle": rot.any(0)}, K


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", default="runs/rotamer_best_of_k")
    ap.add_argument("--sets", nargs="*", default=list(SET_TITLE))
    ap.add_argument("--chi-tol", type=float, default=40.0)
    ap.add_argument("--out-dir", default="runs/rotamer_best_of_k")
    args = ap.parse_args()
    np.seterr(invalid="ignore", divide="ignore")      # chi absents (GLY, ALA)

    data = {}
    for s in args.sets:
        p = os.path.join(args.samples_dir, f"{s}_samples.npz")
        if os.path.exists(p):
            data[s] = per_set(p, args.chi_tol)
    if not data:
        raise SystemExit(f"aucun <ensemble>_samples.npz dans {args.samples_dir}")

    types = [r for r in range(len(STANDARD)) if geo.CHI_MASK[r, 0]]   # sans GLY, ALA
    table = {}
    for s, (rt, ok, K) in data.items():
        for r in types:
            sel = rt == r
            table[(s, r)] = (int(sel.sum()), {k: ok[k][sel].mean() if sel.any() else np.nan
                                               for k in ok})
    last = list(data)[-1]
    types.sort(key=lambda r: -np.nan_to_num(table[(last, r)][1]["random"]))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "rotamer_accuracy_per_type.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ensemble", "type", "n"] + [k for k, _, _ in SERIES])
        for s in data:
            for r in types:
                n, v = table[(s, r)]
                w.writerow([s, STANDARD[r], n] + [f"{v[k]:.4f}" for k, _, _ in SERIES])

    fig, axes = _new_fig(len(data), 1, figsize=(12, 2.9 * len(data) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(types))
    w = 0.27
    for ax, s in zip(axes, data):
        for j, (k, label, color) in enumerate(SERIES):
            vals = [100 * table[(s, r)][1][k] for r in types]
            ax.bar(x + (j - 1) * w, vals, w * 0.92, color=color, label=label)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("rotamer accuracy (%)", fontsize=8)
        n = sum(table[(s, r)][0] for r in types)
        ax.set_title(f"{SET_TITLE.get(s, s)}  ·  {n} residues", loc="left", fontsize=10,
                     color=TEXT_PRIMARY)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([STANDARD[r] for r in types], fontsize=9, color=TEXT_PRIMARY)
    h, l = axes[0].get_legend_handles_labels()
    K = list(data.values())[0][2]
    fig.legend(h, [t.replace("K", str(K)) if "K" in t else t for t in l], loc="upper center",
               ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0),
               labelcolor=TEXT_PRIMARY)
    fig.suptitle(f"Rotamer accuracy per amino acid (all χ within ±{args.chi_tol:g}°)",
                 x=0.01, y=1.035, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.text(0.01, -0.01, f"hybrid_chain_pep, K = {K} samples per residue. Amino acids sorted by "
             f"random accuracy on {SET_TITLE.get(last, last)}. GLY and ALA have no χ angle.",
             fontsize=8, color=TEXT_SECONDARY, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.out_dir, "rotamer_accuracy_per_type.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Ecrit : {out}, rotamer_accuracy_per_type.csv")


if __name__ == "__main__":
    main()
