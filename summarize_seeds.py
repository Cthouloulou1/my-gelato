#!/usr/bin/env python3
"""
Resume multi-seeds des evaluations de flow matching : moyenne +- ecart-type
par variante, sans relancer d'inference.

Lit un ou plusieurs dossiers d'evaluation (sorties de evaluate_flow.py). Les
modeles d'une meme variante sont regroupes par leur nom sans le suffixe de
seed : "hybride", "hybride_s1", "hybride_s2" -> variante "hybride".

Metriques par (variante, ensemble) : celles de <ensemble>_metrics.json, plus
la fraction de residus dont N et C sont inverses (GLY a part), recalculee
depuis <ensemble>_predictions.npz. RMSD par type d'acide amine depuis
<ensemble>_per_restype.csv.

Sorties (--out-dir) :
    seeds_summary.csv        variante, ensemble, metrique, n, moyenne, ecart-type, valeurs
    seeds_per_restype.csv    idem pour le RMSD moyen par type d'acide amine
    seeds_summary.md         tables "moyenne +- ecart-type" (a coller dans un rapport)
    seeds_comparison.png     barres = moyenne, traits = +- 1 ecart-type, points = seeds
    seeds_per_restype_<ensemble>.png

--ref joint ajoute, pour chaque variante, l'ecart a la reference (moyenne des
differences) et l'ecart-type de cet ecart : un ecart plus petit que la
dispersion entre seeds n'est pas un resultat.

Usage :
    python summarize_seeds.py runs/evaluation_flow_e100 runs/evaluation_flow_e100_seeds \
        --ref joint --out-dir runs/seeds_flow_e100
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np

from analyze_flow_errors import GLY, chain_flip, load
from train_utils import (CATEGORICAL, TEXT_SECONDARY, _new_fig, _style_axes, plt)

DATASET_ORDER = ["val", "test_1_2A", "test_2_3A", "test_3_4A"]
MAIN_METRICS = ["rmsd_mean", "rmsd_median", "rmsd_backbone_mean", "rmsd_sidechain_mean",
                "frac_rmsd_lt_1A", "frac_rmsd_lt_2A", "bond_err_mean", "gly_flip",
                "other_flip"]
FIG_METRICS = ["rmsd_mean", "rmsd_backbone_mean", "frac_rmsd_lt_1A", "gly_flip"]
LABELS = {"rmsd_mean": "RMSD moyen (A)", "rmsd_backbone_mean": "RMSD backbone (A)",
          "frac_rmsd_lt_1A": "residus < 1 A", "gly_flip": "GLY avec N/C inverses",
          "other_flip": "autres residus N/C inverses", "rmsd_median": "RMSD median (A)",
          "rmsd_sidechain_mean": "RMSD chaine laterale (A)",
          "frac_rmsd_lt_2A": "residus < 2 A", "bond_err_mean": "erreur de liaison (A)"}
SEED_RE = re.compile(r"_s\d+$")
VARIANT_ORDER = ["joint", "atome", "backbone", "hybride"]   # ordre du recit


def ds_key(ds):
    return (DATASET_ORDER.index(ds) if ds in DATASET_ORDER else 99, ds)


def collect(dirs):
    """{variante: [{"run", "metrics": {ds: {...}}, "restype": {ds: {res: rmsd}}}]}"""
    groups = {}
    for d in dirs:
        for mdir in sorted(glob.glob(os.path.join(d, "*", ""))):
            name = os.path.basename(os.path.normpath(mdir))
            run = {"run": f"{os.path.basename(os.path.normpath(d))}/{name}",
                   "metrics": {}, "restype": {}}
            for f in sorted(glob.glob(os.path.join(mdir, "*_metrics.json"))):
                ds = os.path.basename(f)[: -len("_metrics.json")]
                if ds.startswith("val_best"):
                    continue
                with open(f) as fh:
                    m = {k: v for k, v in json.load(fh).items() if isinstance(v, (int, float))}
                npz = os.path.join(mdir, f"{ds}_predictions.npz")
                if os.path.exists(npz):
                    pred, gt, _, _, rt = load(npz)
                    m["gly_flip"] = chain_flip(pred, gt, rt == GLY)
                    m["other_flip"] = chain_flip(pred, gt, rt != GLY)
                run["metrics"][ds] = m
                rcsv = os.path.join(mdir, f"{ds}_per_restype.csv")
                if os.path.exists(rcsv):
                    with open(rcsv) as fh:
                        run["restype"][ds] = {r["restype"]: float(r["rmsd_mean"])
                                              for r in csv.DictReader(fh) if r["rmsd_mean"]}
            if run["metrics"]:
                groups.setdefault(SEED_RE.sub("", name), []).append(run)
    return groups


def stats(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return 0, np.nan, np.nan
    return len(v), float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan


def fmt(mean, std, pct=False):
    if not np.isfinite(mean):
        return ""
    if pct:
        return f"{100 * mean:.1f} %" + (f" ± {100 * std:.1f}" if np.isfinite(std) else "")
    return f"{mean:.3f}" + (f" ± {std:.3f}" if np.isfinite(std) else "")


def bar_panel(ax, groups, datasets, values, title, pct=False):
    """values[(g, ds)] = liste des valeurs par seed."""
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    width = 0.8 / len(groups)
    rng = np.random.default_rng(0)
    for gi, g in enumerate(groups):
        color = CATEGORICAL[gi]
        for di, ds in enumerate(datasets):
            v = [x for x in values.get((g, ds), []) if np.isfinite(x)]
            if not v:
                continue
            v = np.asarray(v) * (100 if pct else 1)
            x = di + (gi - (len(groups) - 1) / 2) * width
            ax.bar(x, v.mean(), width=width * 0.9, color=color, alpha=0.85,
                   label=g if di == 0 else None)
            if len(v) > 1:
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color=TEXT_SECONDARY,
                            capsize=2, linewidth=1)
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)) * width, v, s=7,
                       color=TEXT_SECONDARY, zorder=3, linewidths=0)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_xlim(-0.55, len(datasets) - 0.45)
    ax.set_title(title + (" (%)" if pct else ""), loc="left", fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="dossiers d'evaluation (evaluate_flow.py)")
    ap.add_argument("--ref", default=None, help="variante de reference pour les ecarts")
    ap.add_argument("--exclude", nargs="*", default=[], help="variantes a ignorer")
    ap.add_argument("--out-dir", default="runs/seeds_flow")
    args = ap.parse_args()

    groups = {g: r for g, r in collect(args.dirs).items() if g not in args.exclude}
    if not groups:
        raise SystemExit("aucune evaluation trouvee")
    if len(groups) > len(CATEGORICAL):
        raise SystemExit(f"{len(groups)} variantes : au plus {len(CATEGORICAL)} (--exclude)")
    names = sorted(groups, key=lambda g: (VARIANT_ORDER.index(g) if g in VARIANT_ORDER
                                           else len(VARIANT_ORDER), g))
    groups = {g: groups[g] for g in names}
    datasets = sorted({ds for runs in groups.values() for r in runs for ds in r["metrics"]},
                      key=ds_key)
    os.makedirs(args.out_dir, exist_ok=True)
    print("Variantes et runs :")
    for g, runs in groups.items():
        print(f"  {g:>10s} ({len(runs)}) : " + ", ".join(r["run"] for r in runs))

    # ---- metriques globales
    values = {}
    rows = []
    for g, runs in groups.items():
        for ds in datasets:
            for m in MAIN_METRICS:
                v = [r["metrics"].get(ds, {}).get(m) for r in runs]
                v = [x for x in v if x is not None]
                values[(m, g, ds)] = v
                n, mean, std = stats(v)
                if n:
                    rows.append({"variante": g, "ensemble": ds, "metrique": m, "n": n,
                                 "moyenne": mean, "ecart_type": std,
                                 "valeurs": " ".join(f"{x:.4f}" for x in v)})
    with open(os.path.join(args.out_dir, "seeds_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# Flow matching : moyenne ± écart-type sur les seeds\n\n",
          "Runs : " + "; ".join(f"{g} (n={len(r)})" for g, r in groups.items()) + "\n"]
    for m in MAIN_METRICS:
        pct = m.startswith("frac") or m.endswith("flip")
        md.append(f"\n## {LABELS.get(m, m)}\n\n| variante | " + " | ".join(datasets)
                  + " |\n|" + "---|" * (len(datasets) + 1) + "\n")
        for g in names:
            md.append(f"| {g} | " + " | ".join(fmt(*stats(values[(m, g, ds)])[1:], pct=pct)
                                               for ds in datasets) + " |\n")
    if args.ref in groups:
        md.append(f"\n## Écart à {args.ref} (RMSD moyen, Å ; négatif = meilleur)\n\n"
                  "| variante | " + " | ".join(datasets) + " |\n|"
                  + "---|" * (len(datasets) + 1) + "\n")
        for g in names:
            if g == args.ref:
                continue
            cells = []
            for ds in datasets:
                a, b = values[("rmsd_mean", g, ds)], values[("rmsd_mean", args.ref, ds)]
                if not a or not b:
                    cells.append("")
                    continue
                diff = np.mean(a) - np.mean(b)
                # dispersion de l'ecart (ecarts-types combines des deux variantes)
                sa = np.std(a, ddof=1) if len(a) > 1 else np.nan
                sb = np.std(b, ddof=1) if len(b) > 1 else np.nan
                spread = np.sqrt(np.nansum([sa ** 2 / len(a), sb ** 2 / len(b)])) \
                    if np.isfinite(sa) or np.isfinite(sb) else np.nan
                cells.append(f"{diff:+.3f}" + (f" (± {spread:.3f})" if np.isfinite(spread) else ""))
            md.append(f"| {g} | " + " | ".join(cells) + " |\n")
        md.append("\n(± : erreur standard de l'écart ; un écart inférieur à environ 2 fois "
                  "cette valeur n'est pas distinguable du bruit entre seeds.)\n")

    # ---- RMSD par type d'acide amine
    rrows = []
    per_res = {}
    for g, runs in groups.items():
        for ds in datasets:
            res_names = sorted({k for r in runs for k in r["restype"].get(ds, {})})
            for res in res_names:
                v = [r["restype"][ds][res] for r in runs if res in r["restype"].get(ds, {})]
                per_res[(g, ds, res)] = v
                n, mean, std = stats(v)
                rrows.append({"variante": g, "ensemble": ds, "restype": res, "n": n,
                              "rmsd_moyen": mean, "ecart_type": std})
    if rrows:
        with open(os.path.join(args.out_dir, "seeds_per_restype.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rrows[0]))
            w.writeheader()
            w.writerows(rrows)
        for ds in datasets:
            res_names = sorted({r["restype"] for r in rrows if r["ensemble"] == ds})
            md.append(f"\n## RMSD par type d'acide aminé, {ds}\n\n| type | "
                      + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1) + "\n")
            for res in res_names:
                md.append(f"| {res} | " + " | ".join(
                    fmt(*stats(per_res.get((g, ds, res), []))[1:]) for g in names) + " |\n")
            fig, ax = _new_fig(figsize=(max(8, 0.16 * len(res_names) * len(names)), 4.8))
            vals = {(g, res): per_res.get((g, ds, res), []) for g in names for res in res_names}
            bar_panel(ax, names, res_names, vals, f"RMSD moyen par type d'acide aminé, {ds} (A)")
            ax.set_xticklabels(res_names, fontsize=7)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(names), frameon=False,
                       fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(os.path.join(args.out_dir, f"seeds_per_restype_{ds}.png"), dpi=150)
            plt.close(fig)

    with open(os.path.join(args.out_dir, "seeds_summary.md"), "w") as f:
        f.write("".join(md))

    # ---- figure principale
    fig, axes = _new_fig(2, 2, figsize=(max(9, 1.1 * len(datasets) * len(names)), 7.2),
                         squeeze=False)
    for ax, m in zip(axes.ravel(), FIG_METRICS):
        pct = m.startswith("frac") or m.endswith("flip")
        bar_panel(ax, names, datasets, {(g, ds): values[(m, g, ds)] for g in names
                                        for ds in datasets}, LABELS[m], pct=pct)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(names), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out_dir, "seeds_comparison.png"), dpi=150)
    plt.close(fig)

    print("".join(md[2:]))
    print(f"Ecrit dans {args.out_dir}")


if __name__ == "__main__":
    main()
