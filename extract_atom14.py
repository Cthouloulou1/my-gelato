#!/usr/bin/env python3
"""
Ajoute aux echantillons d'un LMDB (pretraite ou brut) les positions de tous
les atomes lourds du residu central, au format atom14, RELATIVES au C-alpha
du cube (sample["xyz"], centre exact du cube). Les cubes ne sont pas
reextraits : le resultat est un petit fichier .npz a cote du LMDB.

Seuls les residus standard COMPLETS sont gardes (tous les atomes atom14
presents) ; les autres sont comptes et ignores. Pour les atomes a
conformations alternatives, la premiere conformation (altloc) est prise.

Sortie (npz) : keys (cle LMDB), pos (N,14,3) A, mask (N,14), restype (N,),
pdb_ids, chains, seqids, resnames, ca_xyz (position absolue du C-alpha).

Dependances : gemmi numpy lmdb

Usage :
    python extract_atom14.py \
        --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
        --cif-dir /data/eyerly/pdb_cryoem_splits \
        --out /data/eyerly/splits_atom14.npz
"""

import argparse
import os
import pickle
from collections import Counter

import gemmi
import lmdb
import numpy as np

from flow_common import ATOM14, N_SLOTS, STANDARD

META_KEYS = (b"__len__", b"__meta__", b"__processed_pdb_ids__")


def load_structure(cif_dir, pdb_id):
    path = os.path.join(cif_dir, pdb_id, f"{pdb_id}.cif")
    if not os.path.exists(path):
        return None
    st = gemmi.read_structure(path)
    st.setup_entities()
    index = {}
    for chain in st[0]:
        for res in chain:
            index.setdefault((chain.name, res.seqid.num, res.name), []).append(res)
    return index


def first_atom(res, name):
    for a in res:
        if a.name == name:
            return a
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb-path", required=True)
    ap.add_argument("--cif-dir", required=True,
                    help="dossier <PDB_ID>/<PDB_ID>.cif (sortie de download_datasets.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ca-tol", type=float, default=0.05,
                    help="tolerance (A) pour retrouver le C-alpha du cube dans le .cif")
    args = ap.parse_args()

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    out = {k: [] for k in ("keys", "pos", "mask", "restype", "pdb_ids", "chains",
                           "seqids", "resnames", "ca_xyz")}
    skipped = Counter()
    cur_pdb, index = None, None

    with env.begin() as txn:
        for n, (key, value) in enumerate(txn.cursor(), 1):
            if key in META_KEYS:
                continue
            s = pickle.loads(value)
            resname = s["resname"]
            if resname not in ATOM14:
                skipped["non_standard"] += 1
                continue
            if s["pdb_id"] != cur_pdb:
                cur_pdb = s["pdb_id"]
                index = load_structure(args.cif_dir, cur_pdb)
            if index is None:
                skipped["cif_missing"] += 1
                continue
            ca_xyz = np.asarray(s["xyz"], dtype=np.float64)
            match = None
            for res in index.get((s["chain"], int(s["seqid"]), resname), []):
                for a in res:
                    if a.name == "CA" and np.linalg.norm(
                            np.array([a.pos.x, a.pos.y, a.pos.z]) - ca_xyz) < args.ca_tol:
                        match = res
                        break
                if match is not None:
                    break
            if match is None:
                skipped["residue_not_found"] += 1
                continue
            names = ATOM14[resname]
            pos = np.zeros((N_SLOTS, 3), np.float32)
            mask = np.zeros(N_SLOTS, bool)
            for i, name in enumerate(names):
                a = first_atom(match, name)
                if a is not None:
                    pos[i] = np.array([a.pos.x, a.pos.y, a.pos.z]) - ca_xyz
                    mask[i] = True
            if not mask[:len(names)].all():
                skipped["incomplete"] += 1
                continue
            pos[1] = 0.0  # C-alpha = centre du cube par construction
            out["keys"].append(key.decode("ascii"))
            out["pos"].append(pos)
            out["mask"].append(mask)
            out["restype"].append(STANDARD.index(resname))
            out["pdb_ids"].append(s["pdb_id"])
            out["chains"].append(s["chain"])
            out["seqids"].append(int(s["seqid"]))
            out["resnames"].append(resname)
            out["ca_xyz"].append(ca_xyz.astype(np.float32))
            if n % 20000 == 0:
                print(f"  {n} echantillons lus, {len(out['keys'])} gardes")
    env.close()

    np.savez_compressed(
        args.out,
        keys=np.array(out["keys"]), pos=np.stack(out["pos"]), mask=np.stack(out["mask"]),
        restype=np.array(out["restype"]), pdb_ids=np.array(out["pdb_ids"]),
        chains=np.array(out["chains"]), seqids=np.array(out["seqids"]),
        resnames=np.array(out["resnames"]), ca_xyz=np.stack(out["ca_xyz"]),
    )
    far = np.linalg.norm(np.stack(out["pos"]), axis=-1).max()
    print(f"\nResidus gardes : {len(out['keys'])}")
    print("Ignores :", dict(skipped))
    print(f"Distance max atome - C-alpha : {far:.2f} A (le cube fait +-9.5 A)")
    print(f"Ecrit : {args.out}")


if __name__ == "__main__":
    main()
