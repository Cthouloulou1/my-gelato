#!/usr/bin/env python3
import argparse
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from train_equivariant_cnn import (
    CAPointCloudDataset, EquivariantPointClassifier, RESIDUE_CLASSES, N_CLASSES,
)


def topk_correct(logits, targets, k):
    topk = logits.topk(k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).sum().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for pos, feat, labels in loader:
        logits = model(pos.to(device), feat.to(device))
        all_logits.append(logits.cpu())
        all_targets.append(labels)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = logits.argmax(dim=1)
    return logits, targets, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-path", required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--test-split", required=True,
                        help='JSON {"test": [pdb_ids]}')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 1. Charger le checkpoint et reconstruire le modèle avec les mêmes hyperparamètres
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    a = ck["args"]
    model = EquivariantPointClassifier(
        n_classes=N_CLASSES, lmax=a["lmax"], k_neighbors=a["k_neighbors"],
        num_basis=a["num_basis"], max_radius=a["max_radius"],
        activation=a.get("activation", "sigmoid"),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    print(f"Checkpoint de l'epoque {ck['epoch']}, "
          f"val metrics : {ck['val_metrics']}")

    # 2. Dataset et sélection du test (sans augmentation de rotation)
    dataset = CAPointCloudDataset(args.lmdb_path, n_points=a["n_points"],
                                  augment=False, jitter_std=a["jitter_std"])
    with open(args.test_split) as f:
        test_ids = set(json.load(f)["test"])

    overlap = test_ids & (set(ck["train_pdb_ids"]) | set(ck["val_pdb_ids"]))
    if overlap:
        raise ValueError(f"{len(overlap)} pdb_id du test ont servi au train/val, "
                         f"exemples : {sorted(overlap)[:5]}")

    test_idx = [i for i, pid in enumerate(dataset.pdb_ids) if pid in test_ids]
    print(f"Entrees test : {len(test_ids)}  |  echantillons test : {len(test_idx)}")

    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # 3. Inference
    logits, targets, preds = evaluate(model, loader, device)
    targets_np, preds_np = targets.numpy(), preds.numpy()

    # 4. Metriques globales
    n = len(targets)
    top1 = (preds == targets).float().mean().item()
    top3 = topk_correct(logits, targets, 3) / n
    loss = torch.nn.functional.cross_entropy(logits, targets).item()
    macro_f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)

    # Baseline : toujours predire la classe majoritaire du train
    train_labels = [dataset.labels[i] for i in ck.get("train_idx", [])] or None
    majority_acc = None
    if train_labels:
        majority = Counter(train_labels).most_common(1)[0][0]
        majority_acc = (targets_np == majority).mean()

    print("\n=== Resultats test ===")
    print(f"loss={loss:.4f}  top1={top1:.4f}  top3={top3:.4f}  macroF1={macro_f1:.4f}")
    if majority_acc is not None:
        print(f"baseline classe majoritaire top1 : {majority_acc:.4f}")

    # 5. Details par classe
    labels_all = list(range(N_CLASSES))
    print("\nRapport par classe :")
    print(classification_report(targets_np, preds_np, labels=labels_all,
                                target_names=RESIDUE_CLASSES, zero_division=0))

    cm = confusion_matrix(targets_np, preds_np, labels=labels_all)
    np.savetxt("confusion_test.csv", cm, fmt="%d", delimiter=",")
    print("Matrice de confusion sauvegardee dans confusion_test.csv")


if __name__ == "__main__":
    main()