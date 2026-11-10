#!/usr/bin/env python3
"""
Evalue un ou plusieurs modeles entraines (equivariant et/ou CNN baseline)
sur les ensembles de test de splits.json (produit par download_datasets.py :
test_1_2A, test_2_3A, test_3_4A), et optionnellement sur le val.

Pour chaque (modele, ensemble) : matrice de confusion (CSV + PNG), courbe
ROC one-vs-rest (micro/macro + classes), AUC par classe, rapport par
classe, metriques JSON. Puis un tableau recapitulatif (summary.csv) et une
figure de comparaison entre modeles et resolutions (comparison.png).

Le type de modele est lu dans le checkpoint (cle "model_type" ; les anciens
checkpoints sans cette cle sont consideres comme equivariants).

Usage :
    python evaluate_models.py \
        --lmdb-path ca_density_dataset_preprocessed.lmdb \
        --splits splits.json \
        --model equivariant=runs/equivariant/best_model.pt \
        --model cnn_continuous=runs/cnn_continuous/best_model.pt \
        --out-dir runs/evaluation
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_utils import N_CLASSES, plot_model_comparison, save_final_reports


@torch.no_grad()
def predict(model, loader, device, kind):
    model.eval()
    probs, targets = [], []
    for batch in loader:
        if kind == "equivariant":
            pos, feat, y = batch
            logits = model(pos.to(device), feat.to(device))
        else:
            x, y = batch
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        targets.append(y.numpy().copy())  # copie : libere la memoire partagee du worker
    if not probs:
        return np.zeros(0, dtype=int), np.zeros((0, N_CLASSES))
    return np.concatenate(targets), np.concatenate(probs)


def load_model_and_dataset(ck, lmdb_path, device):
    kind = ck.get("model_type", "equivariant")
    a = ck["args"]
    if kind == "equivariant":
        from train_equivariant_cnn import CAPointCloudDataset, EquivariantPointClassifier
        model = EquivariantPointClassifier(
            n_classes=N_CLASSES, lmax=a["lmax"], k_neighbors=a["k_neighbors"],
            num_basis=a["num_basis"], max_radius=a["max_radius"],
            activation=a.get("activation", "sigmoid"))
        ds = CAPointCloudDataset(lmdb_path, n_points=a["n_points"], augment=False,
                                 jitter_std=a["jitter_std"])
    elif kind == "cnn3d":
        from train_cnn_baseline import CubeDataset, build_cnn_from_args
        model = build_cnn_from_args(a)
        ds = CubeDataset(lmdb_path, mask=a.get("spherical_mask", True))
    else:
        raise ValueError(f"model_type inconnu : {kind}")
    model.load_state_dict(ck["model_state_dict"])
    return kind, model.to(device), ds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmdb-path", required=True)
    p.add_argument("--splits", required=True,
                   help="splits.json (cles test_* evaluees ; 'val' si --include-val)")
    p.add_argument("--model", action="append", required=True,
                   help="nom=chemin/vers/best_model.pt (repetable)")
    p.add_argument("--test-keys", nargs="*", default=None,
                   help="defaut : toutes les cles commencant par 'test'")
    p.add_argument("--include-val", action="store_true")
    p.add_argument("--out-dir", default="runs/evaluation")
    p.add_argument("--batch-size", type=int, default=None,
                   help="defaut : la taille de batch d'entrainement de chaque modele "
                        "(evite de manquer de memoire GPU, ex. equivariant a 512 points)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    with open(args.splits) as f:
        splits = json.load(f)
    keys = args.test_keys or sorted(k for k in splits if k.startswith("test"))
    if args.include_val:
        keys = ["val"] + keys
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        ck = torch.load(path, map_location=device, weights_only=False)
        kind, model, ds = load_model_and_dataset(ck, args.lmdb_path, device)
        seen = set(ck.get("train_pdb_ids", [])) | set(ck.get("val_pdb_ids", []))
        print(f"\n=== {name} ({kind}, epoch {ck.get('epoch')}) ===")

        for key in keys:
            ids = set(splits[key])
            if key != "val":
                overlap = ids & seen
                if overlap:
                    raise ValueError(f"{name}/{key} : {len(overlap)} pdb_id ont servi "
                                     f"au train/val, ex. {sorted(overlap)[:5]}")
            idx = [i for i, pid in enumerate(ds.pdb_ids) if pid in ids]
            missing = ids - set(ds.pdb_ids[i] for i in idx)
            if missing:
                print(f"  [{key}] attention : {len(missing)} entrees absentes du LMDB "
                      f"(extraction echouee ?) : {sorted(missing)[:5]}")
            if not idx:
                print(f"  [{key}] aucun echantillon, ignore")
                continue
            bs = args.batch_size or ck["args"].get("batch_size", 32)
            loader = DataLoader(Subset(ds, idx), batch_size=bs,
                                shuffle=False, num_workers=args.num_workers)
            targets, probs = predict(model, loader, device, kind)
            out = os.path.join(args.out_dir, name)
            m = save_final_reports(targets, probs, out, key, title=f"{name} - {key}",
                                   extra={"n_entries": len(ids) - len(missing)})
            print(f"  [{key}] n={m['n_samples']:6d}  acc={m['accuracy']:.3f}  "
                  f"top3={m['top3_acc']:.3f}  P={m['precision_macro']:.3f}  "
                  f"R={m['recall_macro']:.3f}  F1={m['f1_macro']:.3f}  "
                  f"AUC macro={m['auc_macro']:.3f}  micro={m['auc_micro']:.3f}")
            summary.append({"model": name, "dataset": key,
                            **{k: v for k, v in m.items() if not isinstance(v, dict)}})

        # libere l'environnement LMDB avant d'en rouvrir un pour le modele suivant
        # (lmdb refuse deux ouvertures du meme fichier dans un processus)
        if getattr(ds, "env", None) is not None:
            ds.env.close()
            ds.env = None
        del model

    if summary:
        cols = list(dict.fromkeys(k for r in summary for k in r))
        with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)
        plot_model_comparison(summary, os.path.join(args.out_dir, "comparison.png"))
        print(f"\nRecapitulatif : {os.path.join(args.out_dir, 'summary.csv')}")
        print(f"Comparaison   : {os.path.join(args.out_dir, 'comparison.png')}")


if __name__ == "__main__":
    main()
