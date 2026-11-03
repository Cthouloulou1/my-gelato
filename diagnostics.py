#!/usr/bin/env python3
"""
Diagnostics pour train_equivariant_cnn.py quand le modele n'apprend pas
(loss ~ ln(21), top1 ~ 5-6 %).

Deux sous-commandes :

1. overfit : entraine le modele sur un TOUT PETIT lot fixe (sans
   augmentation, sans dropout, sans poids de classes). Un modele sain doit
   atteindre une loss proche de 0 et ~100 % de precision d'entrainement.
   Sinon : bug d'architecture / de donnees, pas un probleme d'hyperparametres.
   Peut comparer plusieurs activations d'un coup (--activations).

2. baseline : entraine des modeles simples sur le MEME split train/val
   (par pdb_id) et affiche les memes metriques que le script principal :
     - hasard et classe majoritaire
     - regression logistique sur le cube aplati
     - petit MLP sur le cube aplati
     - descripteurs INVARIANTS PAR ROTATION faits main (profil radial,
       tenseur d'inertie, stats de densite) + regression logistique
       + gradient boosting
   Si une baseline depasse largement le reseau equivariant, le probleme est
   dans l'architecture. Si toutes plafonnent pres du hasard, il est dans les
   donnees / labels / le pretraitement.

3. hetero : teste l'heterogeneite entre cartes (effet de carte sur les
   features, centrage, proxy de resolution, performance hors-carte brute vs
   standardisee par carte, dispersion des performances entre cartes).

Usage :
    python diagnostics.py hetero   --lmdb-path data.lmdb
    python diagnostics.py overfit  --lmdb-path data.lmdb --n-samples 64 \
        --iters 300 --activations sigmoid tanh silu
    python diagnostics.py baseline --lmdb-path data.lmdb \
        --max-train 10000 --max-val 3000
"""

import argparse
import csv
import pickle
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as tnn
from tqdm import tqdm

from train_equivariant_cnn import (
    CAPointCloudDataset, EquivariantPointClassifier, split_by_pdb_id,
    compute_class_weights, N_CLASSES, RESIDUE_CLASSES,
)

try:
    from sklearn.metrics import f1_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. Surapprentissage d'un petit lot
# ---------------------------------------------------------------------------

def build_fixed_batch(ds, idx):
    pos, feat, lab = [], [], []
    for i in idx:
        p, f, y = ds.get_item(i, augment=False)
        pos.append(p)
        feat.append(f)
        lab.append(y)
    return torch.stack(pos), torch.stack(feat), torch.tensor(lab)


def describe_inputs(pos, feat, labels):
    print("\n--- Verification des entrees ---")
    print(f"pos  : forme {tuple(pos.shape)}, min={pos.min():.2f} max={pos.max():.2f} "
          f"(Angstrom), |pos| moyen={pos.norm(dim=-1).mean():.2f}")
    print(f"feat : forme {tuple(feat.shape)}, min={feat.min():.4g} max={feat.max():.4g} "
          f"mean={feat.mean():.4g} std={feat.std():.4g}")
    per_cube_std = feat.squeeze(-1).std(dim=1)
    print(f"       std intra-cube : min={per_cube_std.min():.4g} "
          f"max={per_cube_std.max():.4g}")
    counts = Counter(labels.tolist())
    print("labels :", {RESIDUE_CLASSES[c]: n for c, n in sorted(counts.items())})
    if feat.std() < 1e-3 or feat.abs().max() > 1e3:
        print("  /!\\ echelle de densite suspecte : envisager de normaliser par cube.")


def run_overfit(args):
    device = torch.device(args.device)
    seed_all(args.seed)
    ds = CAPointCloudDataset(args.lmdb_path, n_points=args.n_points,
                             augment=False, jitter_std=args.jitter_std)
    train_idx, _, _, _ = split_by_pdb_id(ds, args.val_fraction, args.seed, args.split_file)
    rng = random.Random(args.seed)
    chosen = rng.sample(train_idx, min(args.n_samples, len(train_idx)))
    pos, feat, labels = build_fixed_batch(ds, chosen)
    describe_inputs(pos, feat, labels)
    pos, feat, labels = pos.to(device), feat.to(device), labels.to(device)

    summary = []
    for act in args.activations:
        print(f"\n=== Overfit | activation = {act} | n={len(chosen)} | iters={args.iters} ===")
        seed_all(args.seed)
        model = EquivariantPointClassifier(
            n_classes=N_CLASSES, lmax=args.lmax, k_neighbors=args.k_neighbors,
            num_basis=args.num_basis, max_radius=args.max_radius,
            dropout=0.0, activation=act,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = tnn.CrossEntropyLoss()
        model.train()

        final_loss, final_acc = float("nan"), 0.0
        for it in range(1, args.iters + 1):
            logits = model(pos, feat)
            loss = crit(logits, labels)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                   if p.grad is not None)).item()
            opt.step()

            final_loss = loss.item()
            final_acc = (logits.argmax(1) == labels).float().mean().item()
            if it == 1 or it % args.log_every == 0 or it == args.iters:
                print(f"  it {it:04d} | loss={final_loss:.4f} | acc={final_acc:.3f} "
                      f"| |grad|={gnorm:.3e}")
            if not np.isfinite(final_loss):
                print("  /!\\ loss non finie, arret.")
                break

        model.eval()
        with torch.no_grad():
            eval_acc = (model(pos, feat).argmax(1) == labels).float().mean().item()
        print(f"  precision en mode eval (stats BatchNorm courantes) : {eval_acc:.3f}")
        summary.append((act, final_loss, final_acc, eval_acc))

    print("\n=== Resume overfit ===")
    print(f"(hasard : loss={np.log(N_CLASSES):.3f}, acc~{1 / N_CLASSES:.3f})")
    print(f"{'activation':<10} {'loss':>8} {'acc(train)':>11} {'acc(eval)':>10}  verdict")
    for act, l, a, e in summary:
        verdict = ("OK : sait apprendre" if a >= 0.95
                   else "partiel : apprend lentement / capacite limitee" if a >= 0.5
                   else "ECHEC : n'arrive meme pas a memoriser un petit lot")
        print(f"{act:<10} {l:8.4f} {a:11.3f} {e:10.3f}  {verdict}")
    print("\nSi tout echoue : essayer --lr plus petit/grand, --iters plus grand,"
          "\nou simplifier (k-neighbors, lmax=1) pour isoler le composant fautif.")


# ---------------------------------------------------------------------------
# 2. Baselines
# ---------------------------------------------------------------------------

def load_cubes(ds, idx, desc):
    ds._ensure_env()
    cubes, spacings, labels = [], [], []
    with ds.env.begin() as txn:
        for i in tqdm(idx, desc=desc, leave=False):
            s = pickle.loads(txn.get(ds.keys[i]))
            cubes.append(s["cube"].astype(np.float32))
            spacings.append(np.asarray(s["voxel_spacing"], dtype=np.float64))
            labels.append(s["class_idx"])
    return cubes, spacings, np.array(labels)


def flatten_cubes(cubes):
    X = np.stack([c.reshape(-1) for c in cubes]).astype(np.float32)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-8
    return (X - mu) / sd  # normalisation par cube


def invariant_features(cube, spacing, n_rbins=8):
    """Descripteurs invariants par rotation d'un cube de densite."""
    D, H, W = cube.shape
    zz, yy, xx = np.indices(cube.shape)
    center = np.array([(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0])
    coords = (np.stack([zz, yy, xx], -1) - center) * spacing
    r = np.linalg.norm(coords, axis=-1)

    w = (cube - cube.min()).astype(np.float64)
    total = w.sum() + 1e-12
    wn = w / total

    rmax = r.max()
    hist, _ = np.histogram(r, bins=n_rbins, range=(0, rmax), weights=wn)

    mu = (wn[..., None] * coords).sum(axis=(0, 1, 2))
    d = coords - mu
    dd = d.reshape(-1, 3)
    cov = (dd * wn.reshape(-1, 1)).T @ dd
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig_n = eig / (eig.sum() + 1e-12)

    flat = cube.reshape(-1)
    pct = np.percentile(flat, [5, 25, 50, 75, 95, 99])
    top = np.sort(flat)[::-1]
    top_stats = [top[:10].mean(), top[:50].mean(), top[:200].mean()]

    return np.concatenate([
        hist, [np.linalg.norm(mu)], eig, eig_n,
        [flat.mean(), flat.std(), flat.max(), np.log(total)], pct, top_stats,
    ]).astype(np.float32)


def train_torch_classifier(Xtr, ytr, Xva, hidden, epochs, lr, class_weights, device, seed):
    seed_all(seed)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    d = Xtr.shape[1]
    if hidden:
        net = tnn.Sequential(tnn.Linear(d, hidden), tnn.ReLU(), tnn.Dropout(0.3),
                             tnn.Linear(hidden, N_CLASSES))
    else:
        net = tnn.Linear(d, N_CLASSES)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    crit = tnn.CrossEntropyLoss(weight=class_weights.to(device))
    n, bs = len(Xtr_t), 128
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            loss = crit(net(Xtr_t[b]), ytr_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(Xva_t), dim=1).cpu().numpy()


def expand_proba(proba, classes):
    full = np.zeros((proba.shape[0], N_CLASSES), dtype=np.float64)
    full[:, classes] = proba
    return full


def score(name, proba, y, rows):
    preds = proba.argmax(1)
    top3 = np.argsort(-proba, axis=1)[:, :3]
    row = {
        "name": name,
        "top1": float((preds == y).mean()),
        "top3": float((top3 == y[:, None]).any(1).mean()),
        "f1": float(f1_score(y, preds, average="macro", zero_division=0))
        if HAS_SKLEARN else float("nan"),
    }
    rows.append(row)
    print(f"  {name:<44} top1={row['top1']:.3f} top3={row['top3']:.3f} "
          f"macroF1={row['f1']:.3f}")


def run_baseline(args):
    device = torch.device(args.device)
    seed_all(args.seed)
    ds = CAPointCloudDataset(args.lmdb_path, n_points=args.n_points,
                             augment=False, jitter_std=args.jitter_std)
    train_idx, val_idx, train_ids, val_ids = split_by_pdb_id(
        ds, args.val_fraction, args.seed, args.split_file)
    rng = random.Random(args.seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_idx = train_idx[:args.max_train]
    val_idx = val_idx[:args.max_val]
    print(f"Split par pdb_id : {len(train_ids)} train / {len(val_ids)} val entrees ; "
          f"echantillons utilises : {len(train_idx)} train / {len(val_idx)} val")

    cubes_tr, sp_tr, ytr = load_cubes(ds, train_idx, "cubes train")
    cubes_va, sp_va, yva = load_cubes(ds, val_idx, "cubes val")
    print(f"Taille d'un cube : {cubes_tr[0].shape}, "
          f"classes presentes (train) : {len(set(ytr.tolist()))}/{N_CLASSES}")

    cw = compute_class_weights(ytr.tolist(), N_CLASSES)
    rows = []
    print("\n=== Resultats (validation) ===")

    score("hasard (predictions aleatoires)",
          np.random.default_rng(args.seed).random((len(yva), N_CLASSES)), yva, rows)
    prior = np.bincount(ytr, minlength=N_CLASSES).astype(np.float64)
    prior /= prior.sum()
    score("classe majoritaire", np.tile(prior, (len(yva), 1)), yva, rows)

    Xtr, Xva = flatten_cubes(cubes_tr), flatten_cubes(cubes_va)
    p = train_torch_classifier(Xtr, ytr, Xva, 0, args.epochs, args.lr, cw, device, args.seed)
    score("regression logistique, cube aplati", p, yva, rows)
    p = train_torch_classifier(Xtr, ytr, Xva, 256, args.epochs, args.lr, cw, device, args.seed)
    score("MLP(256), cube aplati", p, yva, rows)
    del Xtr, Xva

    Ftr = np.stack([invariant_features(c, s) for c, s in zip(cubes_tr, sp_tr)])
    Fva = np.stack([invariant_features(c, s) for c, s in zip(cubes_va, sp_va)])
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Ftr_n, Fva_n = ((Ftr - mu) / sd).astype(np.float32), ((Fva - mu) / sd).astype(np.float32)
    p = train_torch_classifier(Ftr_n, ytr, Fva_n, 0, args.epochs, args.lr, cw, device, args.seed)
    score("regr. logistique, descripteurs invariants", p, yva, rows)
    p = train_torch_classifier(Ftr_n, ytr, Fva_n, 128, args.epochs, args.lr, cw, device, args.seed)
    score("MLP(128), descripteurs invariants", p, yva, rows)

    if HAS_SKLEARN:
        sw = cw.numpy()[ytr]
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=args.seed)
        gb.fit(Ftr, ytr, sample_weight=sw)
        score("gradient boosting, descripteurs invariants",
              expand_proba(gb.predict_proba(Fva), gb.classes_), yva, rows)

    best = max(rows[2:], key=lambda r: r["f1"] if HAS_SKLEARN else r["top1"])
    print(f"\nMeilleure baseline : {best['name']} "
          f"(top1={best['top1']:.3f}, macroF1={best['f1']:.3f})")
    print("A comparer a la validation du reseau equivariant (meme split : "
          "meme --seed / --val-fraction / --split-file).")
    print("Lecture :")
    print("  - baseline >> reseau (>= 15 points de top1) -> probleme d'architecture/entrainement ;")
    print("  - baseline ~ hasard (top1 < ~0.10)           -> probleme de donnees, labels ou "
          "pretraitement (ou signal trop faible) ;")
    print("  - baseline ~ reseau                          -> le signal est faible, le reseau "
          "n'est pas le goulot d'etranglement.")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Heterogeneite entre cartes
# ---------------------------------------------------------------------------

INV_NAMES = ([f"rad_hist_{i}" for i in range(8)] + ["|centroide|"]
             + ["eig1", "eig2", "eig3", "eigN1", "eigN2", "eigN3"]
             + ["mean", "std", "max", "log_masse"]
             + ["p5", "p25", "p50", "p75", "p95", "p99"]
             + ["top10", "top50", "top200"])
SPEC_NAMES = ["pow_k<0.1", "pow_0.1-0.2", "pow_0.2-0.3", "pow_k>0.3",
              "contraste_centre_bord"]
ALL_NAMES = INV_NAMES + SPEC_NAMES
IDX_SHARP = ALL_NAMES.index("pow_k>0.3")            # proxy de resolution
IDX_CONTRAST = ALL_NAMES.index("contraste_centre_bord")


def spectral_features(cube, spacing):
    """Fractions de puissance spectrale par bande de frequence (A^-1) --
    proxy de resolution/nettete : une carte floue a peu de puissance haute
    frequence -- + contraste de densite centre (r<3 A) vs bord (r>7 A),
    qui doit etre > 0 si le cube est bien centre sur de la matiere."""
    shape = cube.shape
    P = np.abs(np.fft.fftn(cube - cube.mean())) ** 2
    ks = [np.fft.fftfreq(shape[i], d=float(spacing[i])) for i in range(3)]
    kz, ky, kx = np.meshgrid(*ks, indexing="ij")
    k = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    tot = P.sum() + 1e-12
    edges = [0.0, 0.1, 0.2, 0.3, np.inf]
    fr = [P[(k >= lo) & (k < hi)].sum() / tot for lo, hi in zip(edges[:-1], edges[1:])]

    grids = np.indices(shape)
    r2 = sum(((g - (s - 1) / 2.0) * float(sp)) ** 2
             for g, s, sp in zip(grids, shape, spacing))
    r = np.sqrt(r2)
    contrast = float(cube[r < 3].mean() - cube[r > 7].mean())
    return fr + [contrast]


def collect_map_features(ds, idx, desc):
    ds._ensure_env()
    feats, labels, pdbs = [], [], []
    keys_shown = False
    with ds.env.begin() as txn:
        for i in tqdm(idx, desc=desc, leave=False):
            s = pickle.loads(txn.get(ds.keys[i]))
            if not keys_shown:
                print("Cles d'un echantillon :", sorted(s.keys()))
                keys_shown = True
            cube = s["cube"].astype(np.float32)
            sp = np.asarray(s["voxel_spacing"], dtype=np.float64)
            feats.append(np.concatenate([invariant_features(cube, sp),
                                         spectral_features(cube, sp)]))
            labels.append(s["class_idx"])
            pdbs.append(ds.pdb_ids[i])
    return np.stack(feats).astype(np.float32), np.array(labels), np.array(pdbs)


def eta_squared(F, groups):
    """Part de la variance de chaque feature expliquee par l'identite de la
    carte (0 = aucune, 1 = totale)."""
    _, codes = np.unique(groups, return_inverse=True)
    cnt = np.bincount(codes).astype(np.float64)
    out = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        x = F[:, j].astype(np.float64)
        m = x.mean()
        gm = np.bincount(codes, weights=x) / cnt
        ss_b = (cnt * (gm - m) ** 2).sum()
        ss_t = ((x - m) ** 2).sum() + 1e-12
        out[j] = ss_b / ss_t
    return out


def standardize_per_map(F, groups):
    out = np.empty_like(F)
    for g in np.unique(groups):
        m = groups == g
        out[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)
    return out


def oof_predict(F, y, groups, folds, gb_iters, seed):
    """Predictions hors-carte (GroupKFold) : une carte n'est jamais a la
    fois dans le train et dans le test d'un meme pli."""
    from sklearn.model_selection import GroupKFold
    proba = np.zeros((len(y), N_CLASSES))
    base_pred = np.zeros(len(y), dtype=int)
    for tr, te in tqdm(list(GroupKFold(n_splits=folds).split(F, y, groups)),
                       desc="plis", leave=False):
        cw = compute_class_weights(y[tr].tolist(), N_CLASSES).numpy()
        gb = HistGradientBoostingClassifier(max_iter=gb_iters, random_state=seed)
        gb.fit(F[tr], y[tr], sample_weight=cw[y[tr]])
        proba[te] = expand_proba(gb.predict_proba(F[te]), gb.classes_)
        base_pred[te] = np.bincount(y[tr], minlength=N_CLASSES).argmax()
    return proba, base_pred


def run_hetero(args):
    if not HAS_SKLEARN:
        raise SystemExit("scikit-learn requis : pip install scikit-learn")
    from scipy.stats import spearmanr
    seed_all(args.seed)
    ds = CAPointCloudDataset(args.lmdb_path, n_points=args.n_points,
                             augment=False, jitter_std=args.jitter_std)

    # Echantillonnage plafonne par carte (ignore --split-file : ici on fait
    # une validation croisee hors-carte sur TOUTES les cartes).
    by_pdb = {}
    for i, pid in enumerate(ds.pdb_ids):
        by_pdb.setdefault(pid, []).append(i)
    rng = random.Random(args.seed)
    idx = []
    for pid, lst in by_pdb.items():
        if len(lst) >= args.min_per_pdb:
            rng.shuffle(lst)
            idx.extend(lst[:args.max_per_pdb])
    n_kept = sum(1 for l in by_pdb.values() if len(l) >= args.min_per_pdb)
    print(f"{len(by_pdb)} cartes au total, {n_kept} avec >= {args.min_per_pdb} "
          f"echantillons ; {len(idx)} echantillons utilises "
          f"(max {args.max_per_pdb} par carte)")

    F, y, g = collect_map_features(ds, idx, "features")
    maps = np.unique(g)
    folds = min(args.cv_folds, len(maps))

    # --- A. Effet de carte sur les features -------------------------------
    eta = eta_squared(F, g)
    order = np.argsort(-eta)
    print("\n=== A. Part de variance expliquee par l'identite de la carte (eta^2) ===")
    print(f"moyenne sur les {F.shape[1]} features : {eta.mean():.3f}   "
          f"(~0 : cartes interchangeables ; > 0.3 : fort effet de carte)")
    print("top 6 :", ", ".join(f"{ALL_NAMES[j]}={eta[j]:.2f}" for j in order[:6]))

    sharp = np.array([np.median(F[g == m, IDX_SHARP]) for m in maps])
    contr = np.array([np.median(F[g == m, IDX_CONTRAST]) for m in maps])
    print("\n=== B. Centrage et nettete ===")
    print(f"contraste centre-bord (z-score) : mediane des cartes={np.median(contr):.3f}, "
          f"cartes avec contraste > 0 : {(contr > 0).mean():.0%}")
    print("   (cube centre sur de la matiere -> clairement positif ; ~0 ou negatif -> "
          "centrage suspect)")
    print(f"puissance haute frequence (proxy de resolution) : "
          f"min={sharp.min():.4f} med={np.median(sharp):.4f} max={sharp.max():.4f}")

    # --- C. Classification hors-carte, brut vs standardise par carte -----
    print(f"\n=== C. Gradient boosting hors-carte (GroupKFold, {folds} plis) ===")
    rows = []
    P_raw, base = oof_predict(F, y, g, folds, args.gb_iters, args.seed)
    score("majoritaire du train (par pli)", np.eye(N_CLASSES)[base], y, rows)
    score("features brutes", P_raw, y, rows)
    Fs = standardize_per_map(F, g)
    P_std, _ = oof_predict(Fs, y, g, folds, args.gb_iters, args.seed)
    score("features standardisees PAR CARTE", P_std, y, rows)

    # --- D. Variabilite des performances entre cartes --------------------
    def per_map_gain(P):
        gains = []
        for m in maps:
            k = g == m
            gains.append((P[k].argmax(1) == y[k]).mean() - (base[k] == y[k]).mean())
        return np.array(gains)

    gain_raw, gain_std = per_map_gain(P_raw), per_map_gain(P_std)
    print("\n=== D. Gain de top1 vs classe majoritaire, carte par carte (features brutes) ===")
    q = np.percentile(gain_raw, [10, 25, 50, 75, 90])
    print("percentiles 10/25/50/75/90 : " + " ".join(f"{v:+.3f}" for v in q))
    print(f"cartes avec gain > +0.05 : {(gain_raw > 0.05).sum()}/{len(maps)} ; "
          f"gain < -0.05 : {(gain_raw < -0.05).sum()}/{len(maps)}")
    order_m = np.argsort(-gain_raw)
    print("meilleures cartes :", ", ".join(f"{maps[j]}({gain_raw[j]:+.2f})" for j in order_m[:5]))
    print("pires cartes      :", ", ".join(f"{maps[j]}({gain_raw[j]:+.2f})" for j in order_m[-5:]))
    if len(maps) >= 5:
        r1, p1 = spearmanr(sharp, gain_raw)
        r2, p2 = spearmanr(contr, gain_raw)
        print(f"Spearman gain vs nettete (haute freq.) : rho={r1:+.2f} (p={p1:.2g})")
        print(f"Spearman gain vs contraste centre-bord : rho={r2:+.2f} (p={p2:.2g})")

    print("\n=== Lecture ===")
    print("- eta^2 eleve ET 'standardisees par carte' >> 'brutes' : l'effet de carte "
          "masque le signal -> normaliser par carte (stats de la carte entiere, pas "
          "du cube) ou ajouter un lissage a resolution commune dans le preprocess.")
    print("- gains tres dispersés entre cartes, correles a la nettete : resolution "
          "heterogene -> filtrer/ponderer les cartes par resolution, ou entrainer "
          "d'abord sur les cartes nettes.")
    print("- contraste centre-bord ~ 0 sur la plupart des cartes : cubes mal centres "
          "sur la matiere -> verifier le centrage dans extract_ca_density_lmdb.py.")
    print("- gain ~ 0 partout, y compris sur les cartes nettes, avec centrage correct : "
          "suspecter les labels (class_idx) ou un signal vraiment trop faible.")


# ---------------------------------------------------------------------------
# 4. Metadonnees : coherence des labels, padding, qualite des cartes
# ---------------------------------------------------------------------------

def norm_emdb(x):
    """'EMD-1234', 'emd_01234', 1234 -> '1234' (cle de jointure)."""
    digits = re.sub(r"\D", "", str(x))
    return digits.lstrip("0") or "0"


def load_resolution_csv(path):
    """CSV a colonnes emdb_id,resolution (Angstrom) -> {id normalise: float}."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[norm_emdb(row["emdb_id"])] = float(row["resolution"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def collect_meta(ds, idx, desc):
    ds._ensure_env()
    rec = {k: [] for k in ["feat", "y", "pdb", "emdb", "resname", "dval", "dperc", "pad"]}
    with ds.env.begin() as txn:
        for i in tqdm(idx, desc=desc, leave=False):
            s = pickle.loads(txn.get(ds.keys[i]))
            cube = s["cube"].astype(np.float32)
            sp = np.asarray(s["voxel_spacing"], dtype=np.float64)
            rec["feat"].append(np.concatenate([invariant_features(cube, sp),
                                               spectral_features(cube, sp)]))
            rec["y"].append(s["class_idx"])
            rec["pdb"].append(ds.pdb_ids[i])
            rec["emdb"].append(norm_emdb(s["emdb_id"]) if s.get("emdb_id") is not None else "")
            rec["resname"].append(str(s.get("resname", "?")).strip().upper())
            rec["dval"].append(_num(s.get("density_value")))
            rec["dperc"].append(_num(s.get("density_percentile")))
            rec["pad"].append(float((cube == cube.min()).mean()))
    out = {k: np.array(v) for k, v in rec.items()}
    out["feat"] = out["feat"].astype(np.float32)
    return out


def run_meta(args):
    if not HAS_SKLEARN:
        raise SystemExit("scikit-learn requis : pip install scikit-learn")
    from scipy.stats import spearmanr
    seed_all(args.seed)
    ds = CAPointCloudDataset(args.lmdb_path, n_points=args.n_points,
                             augment=False, jitter_std=args.jitter_std)
    by_pdb = {}
    for i, pid in enumerate(ds.pdb_ids):
        by_pdb.setdefault(pid, []).append(i)
    rng = random.Random(args.seed)
    idx = []
    for lst in by_pdb.values():
        if len(lst) >= args.min_per_pdb:
            rng.shuffle(lst)
            idx.extend(lst[:args.max_per_pdb])
    print(f"{len(by_pdb)} cartes (pdb_id), {len(idx)} echantillons utilises")
    R = collect_meta(ds, idx, "metadonnees")
    y, g, F = R["y"], R["pdb"], R["feat"]
    maps = np.unique(g)

    # --- 1. Coherence resname <-> class_idx -------------------------------
    print("\n=== 1. Coherence resname <-> class_idx ===")
    canon = RESIDUE_CLASSES[:-1]
    expected = np.array([canon.index(n) if n in canon else N_CLASSES - 1
                         for n in R["resname"]])
    bad = expected != y
    print(f"echantillons incoherents avec RESIDUE_CLASSES : {bad.sum()}/{len(y)} "
          f"({bad.mean():.2%})")
    multi = {}
    for n, c in zip(R["resname"], y):
        multi.setdefault(str(n), set()).add(int(c))
    split_names = {n: sorted(c) for n, c in multi.items() if len(c) > 1}
    print("resname associes a plusieurs class_idx :", split_names or "aucun")
    if bad.any():
        ex = np.where(bad)[0][:5]
        print("exemples (resname, class_idx, attendu) :",
              [(str(R["resname"][j]), int(y[j]), int(expected[j])) for j in ex])
        print("  /!\\ le mapping resname -> class_idx ne correspond pas a RESIDUE_CLASSES : "
              "labels potentiellement faux.")
    unknown = sorted(set(R["resname"]) - set(canon))
    print(f"resname classes 'OTHER' (idx {N_CLASSES - 1}) :", unknown or "aucun")

    # --- 2. Padding --------------------------------------------------------
    pad_map = np.array([np.median(R["pad"][g == m]) for m in maps])
    print("\n=== 2. Padding (part de voxels egaux au minimum du cube) ===")
    print(f"par echantillon : mediane={np.median(R['pad']):.3f}, "
          f"> 10% : {(R['pad'] > 0.10).mean():.0%}, > 30% : {(R['pad'] > 0.30).mean():.0%}")
    print("par carte (mediane) : percentiles 0/25/50/75/100 = " +
          " ".join(f"{v:.3f}" for v in np.percentile(pad_map, [0, 25, 50, 75, 100])))
    print("  (un cube sans padding a ~1 voxel au minimum, soit ~0.000 ; une grande "
          "dispersion entre cartes = champ de vue physique non comparable)")

    # --- 3. Performance hors-carte + proxys de qualite ---------------------
    folds = min(args.cv_folds, len(maps))
    print(f"\n=== 3. Qualite des cartes vs performance (GB hors-carte, {folds} plis) ===")
    P, base = oof_predict(F, y, g, folds, args.gb_iters, args.seed)
    gain = np.array([(P[g == m].argmax(1) == y[g == m]).mean()
                     - (base[g == m] == y[g == m]).mean() for m in maps])
    print(f"gain median de top1 vs majoritaire : {np.median(gain):+.3f} ; "
          f"cartes avec gain > +0.05 : {(gain > 0.05).sum()}/{len(maps)}")

    proxies = {
        "contraste centre-bord": np.array([np.median(F[g == m, IDX_CONTRAST]) for m in maps]),
        "padding": pad_map,
        "density_value (med.)": np.array([np.nanmedian(R["dval"][g == m]) for m in maps]),
        "density_percentile (med.)": np.array([np.nanmedian(R["dperc"][g == m]) for m in maps]),
    }
    res_by_map = None
    if args.resolution_csv:
        res = load_resolution_csv(args.resolution_csv)
        emdb_map = [R["emdb"][g == m][0] for m in maps]
        res_by_map = np.array([res.get(e, np.nan) for e in emdb_map])
        print(f"resolutions trouvees pour {np.isfinite(res_by_map).sum()}/{len(maps)} cartes")
        proxies["resolution EMDB (A)"] = res_by_map
    for name, v in proxies.items():
        ok = np.isfinite(v)
        if ok.sum() >= 5 and np.nanstd(v) > 0:
            r, p = spearmanr(v[ok], gain[ok])
            print(f"  Spearman gain vs {name:<28}: rho={r:+.2f} (p={p:.2g}, n={ok.sum()})")
        else:
            print(f"  {name:<28}: indisponible ou constant")

    if res_by_map is not None and np.isfinite(res_by_map).sum() >= 8:
        print("\n  gain de top1 par tranche de resolution (cartes) :")
        for lo, hi in [(0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 99)]:
            k = (res_by_map >= lo) & (res_by_map < hi)
            if k.sum():
                print(f"    [{lo:.1f}, {hi:.1f}[ A : {k.sum():3d} cartes, "
                      f"gain median {np.median(gain[k]):+.3f}")

    # --- 4. Entrainer seulement sur les bonnes cartes ? --------------------
    if res_by_map is not None and np.isfinite(res_by_map).sum() >= 8:
        quality, qname = -np.nan_to_num(res_by_map, nan=99.0), "meilleure resolution"
    else:
        quality, qname = proxies["contraste centre-bord"], "meilleur contraste centre-bord"
    good = quality >= np.median(quality)
    good_maps = maps[good]
    k = np.isin(g, good_maps)
    print(f"\n=== 4. Sous-ensemble des {good.sum()} cartes de {qname} ===")
    acc_all = (P[k].argmax(1) == y[k]).mean()
    acc_base = (base[k] == y[k]).mean()
    f2 = min(args.cv_folds, len(good_maps))
    if f2 >= 2:
        P2, base2 = oof_predict(F[k], y[k], g[k], f2, args.gb_iters, args.seed)
        acc_good = (P2.argmax(1) == y[k]).mean()
        print(f"top1 sur ces cartes : majoritaire={acc_base:.3f} | "
              f"entraine sur TOUTES les cartes={acc_all:.3f} | "
              f"entraine sur les bonnes cartes seulement={acc_good:.3f}")
        print("  (le 2e chiffre > 1er : filtrer par qualite aide ; sinon la quantite "
              "de donnees compte plus que leur qualite)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--lmdb-path", required=True)
        p.add_argument("--split-file", default=None)
        p.add_argument("--val-fraction", type=float, default=0.15)
        p.add_argument("--n-points", type=int, default=128)
        p.add_argument("--jitter-std", type=float, default=0.01)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    po = sub.add_parser("overfit", help="surapprendre un petit lot fixe")
    common(po)
    po.add_argument("--n-samples", type=int, default=64)
    po.add_argument("--iters", type=int, default=300)
    po.add_argument("--lr", type=float, default=1e-3)
    po.add_argument("--log-every", type=int, default=25)
    po.add_argument("--activations", nargs="+", default=["sigmoid", "tanh", "silu"],
                    choices=["sigmoid", "tanh", "silu"])
    po.add_argument("--k-neighbors", type=int, default=16)
    po.add_argument("--lmax", type=int, default=2)
    po.add_argument("--num-basis", type=int, default=8)
    po.add_argument("--max-radius", type=float, default=15.0)
    po.set_defaults(func=run_overfit)

    pb = sub.add_parser("baseline", help="baselines simples sur le meme split")
    common(pb)
    pb.add_argument("--max-train", type=int, default=10000)
    pb.add_argument("--max-val", type=int, default=3000)
    pb.add_argument("--epochs", type=int, default=20)
    pb.add_argument("--lr", type=float, default=1e-3)
    pb.set_defaults(func=run_baseline)

    ph = sub.add_parser("hetero", help="heterogeneite entre cartes (hors-carte)")
    common(ph)
    ph.add_argument("--max-per-pdb", type=int, default=150)
    ph.add_argument("--min-per-pdb", type=int, default=30)
    ph.add_argument("--cv-folds", type=int, default=5)
    ph.add_argument("--gb-iters", type=int, default=100)
    ph.set_defaults(func=run_hetero)

    pm = sub.add_parser("meta", help="coherence des labels, padding, qualite des cartes")
    common(pm)
    pm.add_argument("--max-per-pdb", type=int, default=150)
    pm.add_argument("--min-per-pdb", type=int, default=30)
    pm.add_argument("--cv-folds", type=int, default=5)
    pm.add_argument("--gb-iters", type=int, default=100)
    pm.add_argument("--resolution-csv", default=None,
                    help="CSV emdb_id,resolution (voir fetch_emdb_resolution.py)")
    pm.set_defaults(func=run_meta)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
