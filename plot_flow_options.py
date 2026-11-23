#!/usr/bin/env python3
"""
Bar plots (labels in English) of the summary table written by
summarize_flow_options.py: one horizontal bar per model, small multiples per
metric. Bars are coloured by family of options (generation order, nco,
C-alpha context / peptide loss); error bars = std across seeds when available.

Writes in --out-dir:
    rmsd_by_resolution.png    mean RMSD on val, test 1-2 / 2-3 / 3-4 A
    details_<set>.png         backbone RMSD, residues < 1 A, Gly N/C swapped,
                              D-amino acids, chi1 accuracy, C-N bond error
    single/<metric>.png       each panel on its own (for slides)

Usage :
    python summarize_flow_options.py          # -> runs/flow_options_summary.csv
    python plot_flow_options.py
"""

import argparse
import csv
import os

import numpy as np

from train_utils import (CATEGORICAL, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
                         _new_fig, _style_axes, plt)

NAMES = {"atome": "atom", "hybride": "hybrid"}           # noms anglais
FAMILIES = [  # (label, couleur, test)
    ("Generation order", CATEGORICAL[0],
     lambda m: not m.endswith("_nco") and not any(k in m for k in ("pep", "knn", "chain"))),
    ("nco: CB placed after N, C, O", CATEGORICAL[2], lambda m: m.endswith("_nco")),
    ("Peptide loss / Cα context", CATEGORICAL[1],
     lambda m: any(k in m for k in ("pep", "knn", "chain"))),
]
SET_TITLE = {"val": "Validation", "test_1_2A": "Test 1–2 Å",
             "test_2_3A": "Test 2–3 Å", "test_3_4A": "Test 3–4 Å"}
# (colonne, titre, unite, plus bas = mieux, pourcentage)
DETAILS = [
    ("rmsd_backbone_mean", "Backbone RMSD", "Å", True, False),
    ("frac_rmsd_lt_1A", "Residues with RMSD < 1 Å", "%", False, True),
    ("gly_flip", "Gly with N and C swapped", "%", True, True),
    ("chiral_ca_bad", "D-amino acids (inverted Cα)", "%", True, True),
    ("chi1", "χ1 accuracy (± 40°)", "%", False, True),
    ("err_C-N", "Peptide C–N bond error", "Å", True, False),
]


def family(m):
    for i, (_, color, test) in enumerate(FAMILIES):
        if test(m):
            return i, color
    return 0, CATEGORICAL[0]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def num(x):
        return float(x) if x not in ("", None) else np.nan
    return [{k: (v if k in ("modele", "groupe") else num(v)) for k, v in r.items()} for r in rows]


def bars(ax, rows, col, title, unit, lower, pct, show_names=True):
    """Barres horizontales, un modele par ligne (dans l'ordre du tableau), un
    petit espace entre familles."""
    y, pos, fam_prev = [], 0.0, None
    for r in rows:
        f = family(r["modele"])[0]
        if fam_prev is not None and f != fam_prev:
            pos += 0.5
        y.append(pos)
        pos += 1.0
        fam_prev = f
    y = np.array(y)
    scale = 100.0 if pct else 1.0
    vals = np.array([r.get(col, np.nan) for r in rows]) * scale
    err = np.array([r.get(col + "_std", np.nan) for r in rows]) * scale
    colors = [family(r["modele"])[1] for r in rows]
    ax.barh(y, np.nan_to_num(vals), height=0.72, color=colors, edgecolor=SURFACE, linewidth=1.5)
    has = np.isfinite(err) & np.isfinite(vals)              # barres d'erreur : seeds seulement
    if has.any():
        ax.errorbar(vals[has], y[has], xerr=err[has], fmt="none", ecolor=TEXT_SECONDARY,
                    elinewidth=0.9, capsize=2)
    vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    for yi, v, e in zip(y, vals, err):
        if not np.isfinite(v):
            ax.text(vmax * 0.02, yi, "n/a", va="center", fontsize=7, color=TEXT_SECONDARY)
            continue
        txt = f"{v:.1f}%" if pct else f"{v:.3f}"
        ax.text(v + (e if np.isfinite(e) else 0) + vmax * 0.02, yi, txt, va="center",
                fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlim(0, vmax * 1.22)
    ax.set_ylim(y[-1] + 0.7, y[0] - 0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES.get(r["modele"], r["modele"]) for r in rows] if show_names
                       else [""] * len(rows))
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlabel(f"{unit}  ·  {'lower' if lower else 'higher'} is better", fontsize=8)


def legend(fig, rows, y=1.0):
    used = sorted({family(r["modele"])[0] for r in rows})
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILIES[i][1]) for i in used]
    fig.legend(handles, [FAMILIES[i][0] for i in used], loc="upper center", ncol=len(used),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, y), labelcolor=TEXT_PRIMARY)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="runs/flow_options_summary.csv")
    ap.add_argument("--out-dir", default="runs/flow_options_plots")
    ap.add_argument("--models", nargs="*", default=None, help="only these models (CSV names)")
    args = ap.parse_args()

    rows = load(args.csv)
    if args.models:
        rows = [r for r in rows if r["modele"] in args.models]
    if not rows:
        raise SystemExit("no model")
    detail_set = next((k.split("@")[1] for k in rows[0] if k.startswith("chi1@")), "test_3_4A")
    os.makedirs(os.path.join(args.out_dir, "single"), exist_ok=True)
    h = 0.32 * len(rows) + 1.4

    # 1. RMSD par resolution
    sets = [s for s in SET_TITLE if f"rmsd_mean@{s}" in rows[0]]
    fig, axes = _new_fig(1, len(sets), figsize=(3.3 * len(sets) + 1.2, h), sharey=False)
    for k, (ax, s) in enumerate(zip(np.atleast_1d(axes), sets)):
        bars(ax, rows, f"rmsd_mean@{s}", SET_TITLE[s], "Mean RMSD (Å)", True, False,
             show_names=(k == 0))
    legend(fig, rows)
    fig.suptitle("Mean all-atom RMSD by map resolution", x=0.01, y=1.08, ha="left",
                 fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "rmsd_by_resolution.png"), dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 2. details sur un ensemble
    det = [d for d in DETAILS if f"{d[0]}@{detail_set}" in rows[0]]
    ncol = 3
    nrow = int(np.ceil(len(det) / ncol))
    fig, axes = _new_fig(nrow, ncol, figsize=(3.6 * ncol + 1.2, h * nrow))
    axes = np.atleast_2d(axes)
    for k, (col, title, unit, lower, pct) in enumerate(det):
        ax = axes[k // ncol, k % ncol]
        bars(ax, rows, f"{col}@{detail_set}", title, unit, lower, pct,
             show_names=(k % ncol == 0))
    for k in range(len(det), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    legend(fig, rows, y=1.0)
    fig.suptitle(f"{SET_TITLE.get(detail_set, detail_set)}: backbone, chemistry and rotamers",
                 x=0.01, y=1.04, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"details_{detail_set}.png"), dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 3. un panneau par fichier
    singles = ([(f"rmsd_mean@{s}", f"Mean RMSD · {SET_TITLE[s]}", "Å", True, False,
                 f"rmsd_{s}") for s in sets]
               + [(f"{c}@{detail_set}", f"{t} · {SET_TITLE.get(detail_set, detail_set)}",
                   u, lo, p, f"{c.replace('-', '_')}_{detail_set}") for c, t, u, lo, p in det])
    for col, title, unit, lower, pct, fname in singles:
        fig, ax = _new_fig(1, 1, figsize=(5.2, h))
        bars(ax, rows, col, title, unit, lower, pct)
        legend(fig, rows, y=1.08)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "single", fname + ".png"), dpi=200,
                    bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    print(f"Written in {args.out_dir}: rmsd_by_resolution.png, details_{detail_set}.png, "
          f"single/ ({len(singles)} files)")


if __name__ == "__main__":
    main()
