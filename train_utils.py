#!/usr/bin/env python3
"""
Utilitaires partages par les scripts d'entrainement / d'evaluation :
  - classes de residus, split par pdb_id, poids de classes ;
  - calcul des metriques (accuracy, precision / recall / F1 macro, top-3) ;
  - historique par epoch (CSV + JSON + courbes PNG) ;
  - rapports finaux : matrice de confusion (CSV + PNG), courbes ROC
    one-vs-rest (PNG + JSON des AUC), AUC par classe, rapport par classe.

Convention pour les moyennes "macro" : elles sont calculees sur les classes
PRESENTES dans la verite terrain de l'ensemble evalue (une classe absente,
souvent OTHER ou une classe rare sur un petit test set, ferait sinon chuter
artificiellement la moyenne avec un score de 0). Meme convention pour l'AUC
macro : moyenne des AUC des classes ayant au moins un positif et un negatif.

Aucune dependance a e3nn : utilisable par le CNN baseline seul.
"""

import csv
import json
import os
import random
from collections import Counter

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)


RESIDUE_CLASSES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "OTHER",
]
N_CLASSES = len(RESIDUE_CLASSES)


# ---------------------------------------------------------------------------
# Split / poids de classes
# ---------------------------------------------------------------------------

def split_by_pdb_id(dataset, val_fraction, seed, split_file=None):
    """Split train/val par entree (pdb_id). Si `split_file` est fourni
    (JSON avec au moins les cles "train" et "val", ex. splits.json produit
    par download_datasets.py), il est utilise tel quel ; les autres cles
    (test_*) sont ignorees ici."""
    if split_file:
        with open(split_file) as f:
            split = json.load(f)
        train_ids = set(split["train"])
        val_ids = set(split["val"])
    else:
        unique_ids = sorted(set(dataset.pdb_ids))
        rng = random.Random(seed)
        rng.shuffle(unique_ids)
        n_val = max(1, int(len(unique_ids) * val_fraction))
        val_ids = set(unique_ids[:n_val])
        train_ids = set(unique_ids[n_val:])

    train_idx = [i for i, pid in enumerate(dataset.pdb_ids) if pid in train_ids]
    val_idx = [i for i, pid in enumerate(dataset.pdb_ids) if pid in val_ids]
    return train_idx, val_idx, sorted(train_ids), sorted(val_ids)


def compute_class_weights(labels, n_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.ones(n_classes, dtype=torch.float32)
    for c in range(n_classes):
        count = counts.get(c, 0)
        if count > 0:
            weights[c] = total / (n_classes * count)
    return weights


# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------

def compute_metrics(targets, probs, loss=None):
    """targets: (n,) int ; probs: (n, C) probabilites (softmax)."""
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    n = len(targets)
    out = {}
    if loss is not None:
        out["loss"] = float(loss)
    if n == 0:
        return out
    preds = probs.argmax(axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    present = np.unique(targets)
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=present, average="macro", zero_division=0)
    out.update({
        "accuracy": float((preds == targets).mean()),
        "top3_acc": float((top3 == targets[:, None]).any(axis=1).mean()),
        "precision_macro": float(p),
        "recall_macro": float(r),       # = balanced accuracy
        "f1_macro": float(f1),
    })
    return out


def roc_data(targets, probs, n_classes=N_CLASSES, grid_size=501):
    """Courbes ROC one-vs-rest. Retourne un dict avec, pour chaque classe
    evaluable (>=1 positif et >=1 negatif), fpr/tpr/auc, plus la moyenne
    micro (tous les couples (echantillon, classe) poolés) et macro (moyenne
    des tpr interpolees sur une grille de fpr commune)."""
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    onehot = np.eye(n_classes, dtype=np.int64)[targets]
    per_class = {}
    grid = np.linspace(0.0, 1.0, grid_size)
    tprs = []
    for c in range(n_classes):
        y = onehot[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, probs[:, c])
        auc = roc_auc_score(y, probs[:, c])
        per_class[c] = {"fpr": fpr, "tpr": tpr, "auc": float(auc),
                        "n_pos": int(y.sum())}
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)

    res = {"per_class": per_class}
    if per_class:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        res["macro"] = {"fpr": grid, "tpr": mean_tpr,
                        "auc": float(np.mean([v["auc"] for v in per_class.values()]))}
    # micro : n'a de sens que si au moins une classe est evaluable
    if onehot.sum() > 0 and per_class:
        fpr, tpr, _ = roc_curve(onehot.ravel(), probs.ravel())
        res["micro"] = {"fpr": fpr, "tpr": tpr,
                        "auc": float(roc_auc_score(onehot.ravel(), probs.ravel()))}
    return res


def epoch_metrics(targets, probs, loss=None):
    """Metriques suivies a chaque epoch (inclut l'AUC macro/micro)."""
    m = compute_metrics(targets, probs, loss)
    if len(targets):
        m.update(auc_summary(roc_data(targets, probs, np.asarray(probs).shape[1])))
    return m


def auc_summary(roc):
    return {
        "auc_macro": roc.get("macro", {}).get("auc"),
        "auc_micro": roc.get("micro", {}).get("auc"),
    }


# ---------------------------------------------------------------------------
# Style des figures (palette validee, mode clair)
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL_MARK = "#a8a7a1"
SERIES = {"train": "#2a78d6", "val": "#eb6834"}   # slots categoriels 1 et 2
SERIES_ALT = ["#2a78d6", "#eb6834", "#1baf7a"]     # 3 premiers slots (all-pairs OK)
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def _new_fig(*args, **kw):
    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


# ---------------------------------------------------------------------------
# Historique par epoch
# ---------------------------------------------------------------------------

CURVE_METRICS = [
    ("loss", "Loss (cross-entropy)"),
    ("accuracy", "Accuracy (top-1)"),
    ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"),
    ("f1_macro", "F1-score (macro)"),
    ("auc_macro", "ROC AUC (macro, OvR)"),
]


class MetricsHistory:
    """Accumule les metriques train/val a chaque epoch et les ecrit sur
    disque (history.csv, history.json, training_curves.png) a chaque appel
    de `log`, pour pouvoir suivre un entrainement en cours."""

    def __init__(self, out_dir, title=""):
        self.out_dir = out_dir
        self.title = title
        self.rows = []
        os.makedirs(out_dir, exist_ok=True)

    def log(self, epoch, train_metrics, val_metrics, lr=None):
        row = {"epoch": epoch}
        if lr is not None:
            row["lr"] = lr
        for split, m in (("train", train_metrics), ("val", val_metrics)):
            for k, v in m.items():
                row[f"{split}_{k}"] = v
        self.rows.append(row)
        self._write()

    def _write(self):
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(self.out_dir, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.rows, f, indent=2)
        plot_training_curves(self.rows, os.path.join(self.out_dir, "training_curves.png"),
                             self.title)


def plot_training_curves(rows, path, title=""):
    metrics = [(k, lab) for k, lab in CURVE_METRICS
               if any(f"train_{k}" in r or f"val_{k}" in r for r in rows)]
    if not metrics:
        return
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = _new_fig(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                         squeeze=False)
    epochs = [r["epoch"] for r in rows]
    for ax, (key, label) in zip(axes.ravel(), metrics):
        _style_axes(ax)
        for split in ("train", "val"):
            ys = [r.get(f"{split}_{key}") for r in rows]
            pts = [(e, y) for e, y in zip(epochs, ys) if y is not None]
            if not pts:
                continue
            xe, ye = zip(*pts)
            ax.plot(xe, ye, color=SERIES[split], linewidth=2, label=split,
                    marker="o" if len(pts) <= 30 else None, markersize=3)
        # meilleure epoch val (sauf loss : min)
        vals = [(r["epoch"], r.get(f"val_{key}")) for r in rows
                if r.get(f"val_{key}") is not None]
        if vals:
            best = min(vals, key=lambda t: t[1]) if key == "loss" else max(vals, key=lambda t: t[1])
            ax.annotate(f"best {best[1]:.3f}", best, textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=TEXT_SECONDARY,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2,
               labelcolor=TEXT_PRIMARY)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rapports finaux : confusion, ROC, par classe
# ---------------------------------------------------------------------------

def plot_confusion(cm, path, title="", class_names=RESIDUE_CLASSES):
    """Matrice de confusion normalisee par ligne (= rappel par classe).
    Seules les cellules >= 0.10 sont annotees (lisibilite a 21 classes) ;
    les comptes bruts sont dans le CSV."""
    cm = np.asarray(cm, dtype=np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    n = len(class_names)
    fig, ax = _new_fig(figsize=(9, 8))
    ax.set_facecolor(SURFACE)
    im = ax.imshow(norm, cmap=SEQ_CMAP, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8, color=TEXT_SECONDARY)
    ylabels = [f"{c} ({int(s)})" for c, s in zip(class_names, row_sum.ravel())]
    ax.set_yticklabels(ylabels, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("prediction", color=TEXT_SECONDARY)
    ax.set_ylabel("verite (n echantillons)", color=TEXT_SECONDARY)
    for s in ax.spines.values():
        s.set_visible(False)
    # fines separations couleur surface entre cellules
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=6.5, color="#ffffff" if v > 0.5 else TEXT_PRIMARY)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("fraction de la ligne (rappel)", color=TEXT_SECONDARY)
    cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cb.outline.set_visible(False)
    if title:
        ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_roc(roc, path, title=""):
    fig, ax = _new_fig(figsize=(6.2, 5.6))
    _style_axes(ax)
    first = True
    for c, d in roc["per_class"].items():
        ax.plot(d["fpr"], d["tpr"], color=NEUTRAL_MARK, linewidth=0.8, alpha=0.6,
                label="classes individuelles (OvR)" if first else None)
        first = False
    if "macro" in roc:
        ax.plot(roc["macro"]["fpr"], roc["macro"]["tpr"], color=SERIES_ALT[0],
                linewidth=2, label=f"macro-moyenne  AUC = {roc['macro']['auc']:.3f}")
    if "micro" in roc:
        ax.plot(roc["micro"]["fpr"], roc["micro"]["tpr"], color=SERIES_ALT[1],
                linewidth=2, label=f"micro-moyenne  AUC = {roc['micro']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle="--",
            label="hasard")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("taux de faux positifs")
    ax.set_ylabel("taux de vrais positifs")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_auc_per_class(roc, path, title="", class_names=RESIDUE_CLASSES):
    items = sorted(roc["per_class"].items(), key=lambda kv: kv[1]["auc"])
    if not items:
        return
    names = [f"{class_names[c]} (n={d['n_pos']})" for c, d in items]
    aucs = [d["auc"] for _, d in items]
    fig, ax = _new_fig(figsize=(6.2, 0.28 * len(items) + 1.2))
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(range(len(items)), aucs, color=SERIES_ALT[0], height=0.7)
    ax.axvline(0.5, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=8)
    # marge a droite de 1.0 pour que l'etiquette tienne toujours apres la
    # barre (avant : etiquette ecrasee sur le bout des barres proches de 1)
    ax.set_xlim(0, 1.09)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xlabel("ROC AUC one-vs-rest  (tirets : hasard)")
    for i, a in enumerate(aucs):
        ax.text(a + 0.012, i, f"{a:.2f}", va="center", ha="left", fontsize=7,
                color=TEXT_SECONDARY)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def save_final_reports(targets, probs, out_dir, prefix, title="",
                       class_names=RESIDUE_CLASSES, extra=None):
    """Ecrit dans out_dir :
        <prefix>_confusion.csv / _confusion.png
        <prefix>_roc.png / _auc_per_class.png
        <prefix>_metrics.json (metriques globales + AUC par classe)
        <prefix>_classification_report.txt
        <prefix>_predictions.npz (targets + probs, pour retracer)
    Retourne le dict de metriques."""
    os.makedirs(out_dir, exist_ok=True)
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    n_classes = len(class_names)
    labels_all = list(range(n_classes))

    metrics = compute_metrics(targets, probs)
    roc = roc_data(targets, probs, n_classes)
    metrics.update(auc_summary(roc))
    metrics["n_samples"] = int(len(targets))
    metrics["auc_per_class"] = {class_names[c]: d["auc"] for c, d in roc["per_class"].items()}
    if extra:
        metrics.update(extra)

    preds = probs.argmax(axis=1) if len(targets) else np.array([], dtype=int)
    cm = confusion_matrix(targets, preds, labels=labels_all)
    base = os.path.join(out_dir, prefix)
    with open(base + "_confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + list(class_names))
        for name, row in zip(class_names, cm):
            w.writerow([name] + [int(x) for x in row])
    plot_confusion(cm, base + "_confusion.png", title, class_names)
    if roc["per_class"]:
        plot_roc(roc, base + "_roc.png", title)
        plot_auc_per_class(roc, base + "_auc_per_class.png", title, class_names)

    with open(base + "_classification_report.txt", "w") as f:
        f.write(classification_report(targets, preds, labels=labels_all,
                                      target_names=class_names, zero_division=0))
    with open(base + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(base + "_predictions.npz", targets=targets, probs=probs)
    return metrics


def plot_model_comparison(summary_rows, path, metric_keys=("accuracy", "f1_macro", "auc_macro")):
    """summary_rows : liste de dicts {"model", "dataset", <metriques>}.
    Une petite multiple par metrique, barres groupees par dataset, une
    couleur par modele (<= 3 modeles)."""
    models = list(dict.fromkeys(r["model"] for r in summary_rows))
    datasets = list(dict.fromkeys(r["dataset"] for r in summary_rows))
    if not models or not datasets:
        return
    fig, axes = _new_fig(1, len(metric_keys), figsize=(4.4 * len(metric_keys), 3.6),
                         squeeze=False)
    width = 0.8 / len(models)
    for ax, key in zip(axes.ravel(), metric_keys):
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
        for mi, model in enumerate(models):
            xs, ys = [], []
            for di, ds in enumerate(datasets):
                r = next((r for r in summary_rows
                          if r["model"] == model and r["dataset"] == ds), None)
                if r is not None and r.get(key) is not None:
                    xs.append(di + (mi - (len(models) - 1) / 2) * width)
                    ys.append(r[key])
            color = SERIES_ALT[mi % len(SERIES_ALT)]
            ax.bar(xs, ys, width=width * 0.92, color=color, label=model)
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7,
                        color=TEXT_SECONDARY)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(key, loc="left", fontsize=10)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False,
               ncol=len(models), labelcolor=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
