#!/usr/bin/env python3
"""
Evalue un ou plusieurs modeles de flow matching (train_flow.py) sur les
ensembles de test de splits.json (et le val avec --include-val).

Pour chaque (modele, ensemble) : generation des atomes de chaque residu,
RMSD tous atomes / backbone / chaine laterale (atomes symetriques pris en
compte), erreur sur les liaisons, RMSD par type d'acide amine (CSV + PNG),
courbe cumulative des RMSD. Puis summary.csv, comparison.png et une courbe
cumulative par modele superposant les resolutions.

--write-pdb ecrit, pour chaque entree, la prediction et la verite au format
PDB (positions absolues : C-alpha du .cif + positions predites), a ouvrir
avec la carte dans ChimeraX / PyMOL.

Usage :
    python evaluate_flow.py \
        --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
        --atom14 /data/eyerly/splits_atom14.npz \
        --splits /data/eyerly/pdb_cryoem_splits/splits.json \
        --model flow_cnn=runs/flow_cnn/best_model.pt \
        --model flow_equi=runs/flow_equivariant/best_model.pt \
        --sample-steps 50 --include-val --out-dir runs/evaluation_flow
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from flow_common import FlowDataset, plot_rmsd_cdf, save_flow_reports, spherical_mask_t, write_pdb
from flow_models import build_flow_model
from train_flow import generate
from train_utils import plot_model_comparison


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmdb-path", required=True)
    p.add_argument("--atom14", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--model", action="append", required=True, help="nom=chemin/best_model.pt")
    p.add_argument("--test-keys", nargs="*", default=None)
    p.add_argument("--include-val", action="store_true")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-pdb", action="store_true")
    p.add_argument("--out-dir", default="runs/evaluation_flow")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    with open(args.splits) as f:
        splits = json.load(f)
    keys = args.test_keys or sorted(k for k in splits if k.startswith("test"))
    if args.include_val:
        keys = ["val"] + keys
    ds = FlowDataset(args.lmdb_path, args.atom14)
    summary = []

    for spec in args.model:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        a = ck["args"]
        model = build_flow_model(ck["model_type"], a).to(device)
        model.load_state_dict(ck["model_state_dict"])
        a = dict(a, sample_steps=args.sample_steps)
        ns = argparse.Namespace(**a)
        mask_t = None if a.get("no_spherical_mask") else spherical_mask_t(20, device)
        seen = set(ck.get("train_pdb_ids", [])) | set(ck.get("val_pdb_ids", []))
        out = os.path.join(args.out_dir, name)
        curves = []
        print(f"\n=== {name} ({ck['model_type']}, epoch {ck['epoch']}, "
              f"{args.sample_steps} pas) ===")
        for key in keys:
            ids = splits[key]
            if key != "val" and set(ids) & seen:
                raise ValueError(f"{name}/{key} : des pdb_id ont servi au train/val")
            idx = ds.indices_for(ids)
            if not idx:
                print(f"  [{key}] aucun residu, ignore")
                continue
            loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
            pred, gt, m, rt, ii = generate(model, loader, ns, device, seed=args.seed,
                                           mask_t=mask_t, desc=key)
            agg = save_flow_reports(pred, gt, m, rt, out, key, title=f"{name} - {key}",
                                    extra={"sample_steps": args.sample_steps})
            r = np.load(os.path.join(out, f"{key}_predictions.npz"))["rmsd"]
            curves.append((key, r))
            print(f"  [{key}] n={agg['n_residues']:6d}  RMSD moyen {agg['rmsd_mean']:.3f} A  "
                  f"median {agg['rmsd_median']:.3f}  backbone {agg['rmsd_backbone_mean']:.3f}  "
                  f"chaine lat. {agg['rmsd_sidechain_mean']:.3f}  <1A {agg['frac_rmsd_lt_1A']:.3f}  "
                  f"liaisons {agg['bond_err_mean']:.3f} A")
            summary.append({"model": name, "dataset": key,
                            **{k: v for k, v in agg.items() if not isinstance(v, dict)}})

            if args.write_pdb:
                pdb_dir = os.path.join(out, f"{key}_pdb")
                os.makedirs(pdb_dir, exist_ok=True)
                by_entry = defaultdict(list)
                for row, i in enumerate(ii):
                    by_entry[ds.pdb_ids[i]].append((row, i))
                for pid, rows in by_entry.items():
                    for tag, coords in (("pred", pred), ("true", gt)):
                        recs = [(ds.meta["chains"][i], ds.meta["seqids"][i],
                                 str(ds.meta["resnames"][i]), ds.meta["ca_xyz"][i],
                                 coords[row], m[row]) for row, i in rows]
                        write_pdb(os.path.join(pdb_dir, f"{pid}_{tag}.pdb"), recs)
                print(f"           PDB ecrits dans {pdb_dir}")

        # courbe cumulative par modele : au plus 3 ensembles par figure (palette)
        tests = [c for c in curves if c[0] != "val"][:3]
        if tests:
            plot_rmsd_cdf(tests, os.path.join(out, "rmsd_cdf_by_resolution.png"),
                          title=f"{name} - RMSD par residu selon la resolution")
        ds.close()

    if summary:
        cols = list(dict.fromkeys(k for r in summary for k in r))
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)
        plot_model_comparison(summary, os.path.join(args.out_dir, "comparison.png"),
                              metric_keys=("rmsd_mean", "rmsd_sidechain_mean", "frac_rmsd_lt_1A"))
        print(f"\nRecapitulatif : {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
