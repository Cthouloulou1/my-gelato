#!/usr/bin/env python3
"""
Contexte des C-alpha voisins de chaque residu de atom14.npz, lu dans les .cif
(TOUS les residus de la structure : y compris incomplets ou non standard,
absents de atom14.npz). Positions RELATIVES au C-alpha du residu (A, axes du
.cif), dans le meme ordre que atom14.npz.

Deux hypotheses pour l'entree du modele (train_flow.py --ca-context) :
    chain : trace ordonnee connue -> C-alpha du residu precedent (i-1) et du
            suivant (i+1) dans la chaine ;
    knn   : C-alpha detectes sans chainage -> les --knn C-alpha les plus
            proches (toutes chaines) a moins de --radius A, sans ordre.

Pour la loss sur la liaison peptidique (--pep-weight), vraies positions des
atomes du squelette voisins : C-alpha, C, O du residu i-1 et N, C-alpha du
residu i+1. Deux residus sont consecutifs s'ils se suivent dans la chaine et
sont lies (C(i-1)-N(i) < --max-cn A ; a defaut des atomes, CA-CA < 4.2 A).

Sortie (npz) :
    prev_atoms (N,3,3), prev_mask (N,3)   CA, C, O du residu i-1
    next_atoms (N,2,3), next_mask (N,2)   N, CA du residu i+1
    knn_ca (N,K,3), knn_mask (N,K)        K C-alpha les plus proches
    pdb_ids, chains, seqids                pour verifier l'alignement

Usage :
    python extract_ca_context.py --atom14 /data/eyerly/splits_atom14.npz \\
        --cif-dir /data/eyerly/pdb_cryoem_splits \\
        --out /data/eyerly/splits_ca_context.npz
"""

import argparse
import os
from collections import Counter, defaultdict

import gemmi
import numpy as np
from tqdm import tqdm

PREV_ATOMS = ("CA", "C", "O")
NEXT_ATOMS = ("N", "CA")


def atom_pos(res, name):
    for a in res:
        if a.name == name:
            return np.array(a.pos.tolist())
    return None


def read_chains(path):
    """{chaine: [(seqid, {nom: position})]} des residus ayant un C-alpha, dans
    l'ordre de la chaine (premier modele, premiere conformation)."""
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    out = {}
    for chain in st[0]:
        rows = []
        for res in chain:
            ca = atom_pos(res, "CA")
            if ca is None:
                continue
            rows.append((res.seqid.num, {n: atom_pos(res, n) for n in ("N", "CA", "C", "O")}))
        if rows:
            out[chain.name] = rows
    return out


def linked(a, b, max_cn):
    """a puis b consecutifs et lies dans la chaine ?"""
    if a["C"] is not None and b["N"] is not None:
        return np.linalg.norm(a["C"] - b["N"]) < max_cn
    return np.linalg.norm(a["CA"] - b["CA"]) < 4.2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atom14", required=True)
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--max-cn", type=float, default=2.0)
    ap.add_argument("--ca-tol", type=float, default=0.05)
    args = ap.parse_args()

    z = np.load(args.atom14, allow_pickle=False)
    pdb_ids, chains, seqids, ca_xyz = z["pdb_ids"], z["chains"], z["seqids"], z["ca_xyz"]
    n, K = len(pdb_ids), args.knn
    prev_atoms = np.zeros((n, 3, 3), np.float32)
    prev_mask = np.zeros((n, 3), bool)
    next_atoms = np.zeros((n, 2, 3), np.float32)
    next_mask = np.zeros((n, 2), bool)
    knn_ca = np.zeros((n, K, 3), np.float32)
    knn_mask = np.zeros((n, K), bool)
    stats = Counter()

    by_entry = defaultdict(list)
    for i, p in enumerate(pdb_ids):
        by_entry[str(p)].append(i)
    for pid, rows in tqdm(by_entry.items(), desc="entrees"):
        path = os.path.join(args.cif_dir, pid, f"{pid}.cif")
        if not os.path.exists(path):
            stats["cif_absent"] += len(rows)
            continue
        ch = read_chains(path)
        where = {(c, s): (c, k) for c, rs in ch.items() for k, (s, _) in enumerate(rs)}
        all_ca = np.array([a["CA"] for rs in ch.values() for _, a in rs])
        for i in rows:
            hit = where.get((str(chains[i]), int(seqids[i])))
            if hit is None or np.linalg.norm(ch[hit[0]][hit[1]][1]["CA"] - ca_xyz[i]) > args.ca_tol:
                stats["introuvable"] += 1
                continue
            c, k = hit
            rs, cur, ca = ch[c], ch[c][k][1], ca_xyz[i]
            if k > 0 and linked(rs[k - 1][1], cur, args.max_cn):
                for j, nm in enumerate(PREV_ATOMS):
                    p = rs[k - 1][1][nm]
                    if p is not None:
                        prev_atoms[i, j], prev_mask[i, j] = p - ca, True
                stats["avec_i-1"] += 1
            if k + 1 < len(rs) and linked(cur, rs[k + 1][1], args.max_cn):
                for j, nm in enumerate(NEXT_ATOMS):
                    p = rs[k + 1][1][nm]
                    if p is not None:
                        next_atoms[i, j], next_mask[i, j] = p - ca, True
                stats["avec_i+1"] += 1
            d = np.linalg.norm(all_ca - ca, axis=1)
            order = [j for j in np.argsort(d) if 1e-3 < d[j] <= args.radius][:K]
            knn_ca[i, :len(order)] = all_ca[order] - ca
            knn_mask[i, :len(order)] = True
            stats["ok"] += 1

    np.savez_compressed(args.out, prev_atoms=prev_atoms, prev_mask=prev_mask,
                        next_atoms=next_atoms, next_mask=next_mask, knn_ca=knn_ca,
                        knn_mask=knn_mask, pdb_ids=pdb_ids, chains=chains, seqids=seqids,
                        radius=np.float32(args.radius))
    print(f"Residus : {n}  |  " + "  ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    print(f"C-alpha voisins (knn, <= {args.radius:g} A) : {knn_mask.sum(1).mean():.1f} en "
          f"moyenne, max {K} gardes ; {np.mean(knn_mask.sum(1) == K):.1%} des residus en ont "
          f">= {K}")
    print(f"Ecrit : {args.out}")


if __name__ == "__main__":
    main()
