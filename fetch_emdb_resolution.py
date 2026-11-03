#!/usr/bin/env python3
"""
Recupere la resolution publiee (Angstrom) de chaque carte EMDB presente dans
le LMDB et l'ecrit dans un CSV `emdb_id,resolution`, utilisable par
    python diagnostics.py meta --resolution-csv emdb_resolution.csv ...

ATTENTION : ce script n'a PAS ete teste contre l'API EMDB reelle (la page
n'etait pas accessible depuis l'environnement de developpement). L'extraction
de la resolution est donc volontairement tolerante (recherche recursive) et
les echecs sont listes a la fin. Verifie quelques valeurs a la main, par ex.
sur https://www.ebi.ac.uk/emdb/EMD-<id>. Si l'API a change, le plus simple est
de fournir toi-meme le CSV (colonnes emdb_id,resolution).

Usage :
    python fetch_emdb_resolution.py \
        --lmdb-path /data/eyerly/ca_density_dataset_preprocessed.lmdb \
        --out emdb_resolution.csv
"""

import argparse
import csv
import json
import pickle
import re
import time
import urllib.error
import urllib.request

import lmdb


def norm_emdb(x):
    digits = re.sub(r"\D", "", str(x))
    return digits.lstrip("0") or "0"


def unique_emdb_ids(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    ids = set()
    with env.begin() as txn:
        for key, value in txn.cursor():
            if key in (b"__len__", b"__meta__", b"__processed_pdb_ids__"):
                continue
            e = pickle.loads(value).get("emdb_id")
            if e is not None:
                ids.add(norm_emdb(e))
    env.close()
    return sorted(ids, key=lambda s: int(s))


def _to_float(v):
    if isinstance(v, dict):  # ex. {"valueOf_": "3.2", "units": "A"}
        for k in ("valueOf_", "value", "#text"):
            if k in v:
                return _to_float(v[k])
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_resolution(obj):
    """Cherche la resolution de la reconstruction finale : d'abord sous une
    cle 'final_reconstruction', sinon toute cle 'resolution' plausible."""
    def walk(o, under_final):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "resolution" and under_final:
                    r = _to_float(v)
                    if r is not None:
                        return r
                r = walk(v, under_final or k == "final_reconstruction")
                if r is not None:
                    return r
        elif isinstance(o, list):
            for item in o:
                r = walk(item, under_final)
                if r is not None:
                    return r
        return None

    r = walk(obj, False)
    if r is not None:
        return r

    def any_resolution(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "resolution":
                    r = _to_float(v)
                    if r is not None:
                        return r
                r = any_resolution(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for item in o:
                r = any_resolution(item)
                if r is not None:
                    return r
        return None

    return any_resolution(obj)


def fetch_one(eid, retries=3, pause=60):
    url = f"https://www.ebi.ac.uk/emdb/api/entry/EMD-{eid}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return find_resolution(json.load(resp))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                print(f"  429 pour EMD-{eid}, attente {pause}s")
                time.sleep(pause)
                continue
            print(f"  echec EMD-{eid} : HTTP {e.code}")
            return None
        except Exception as e:  # reseau, JSON invalide...
            print(f"  echec EMD-{eid} : {e}")
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb-path", required=True)
    ap.add_argument("--out", default="emdb_resolution.csv")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="pause (s) entre deux requetes")
    args = ap.parse_args()

    ids = unique_emdb_ids(args.lmdb_path)
    print(f"{len(ids)} emdb_id distincts")
    failed = []
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emdb_id", "resolution"])
        for n, eid in enumerate(ids, 1):
            r = fetch_one(eid)
            if r is None:
                failed.append(eid)
            w.writerow([eid, "" if r is None else r])
            f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(ids)}")
            time.sleep(args.delay)
    print(f"ecrit : {args.out} ({len(ids) - len(failed)} resolutions, "
          f"{len(failed)} echecs)")
    if failed:
        print("echecs :", failed[:30], "..." if len(failed) > 30 else "")


if __name__ == "__main__":
    main()
