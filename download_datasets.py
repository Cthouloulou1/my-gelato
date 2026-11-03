#!/usr/bin/env python3
"""
Construit et telecharge les jeux de donnees cryo-EM (PDB + carte EMDB) :

    trainval   : N_TRAINVAL entrees (defaut 100) de resolution <= 2.0 A,
                 decoupees en train / val (--val-fraction, defaut 0.15)
    test_1_2A  : N_TEST entrees (defaut 20), 1.0 < resolution <= 2.0 A
    test_2_3A  : N_TEST entrees,              2.0 < resolution <= 3.0 A
    test_3_4A  : N_TEST entrees,              3.0 < resolution <= 4.0 A

Bornes : intervalles ouverts a gauche et fermes a droite, pour etre coherent
avec "<= 2 A" du trainval (une carte a exactement 2.0 A est dans
trainval / test_1_2A, une carte a 3.0 A dans test_2_3A).

Criteres de selection (appliques a toutes les entrees) :
  - methode ELECTRON MICROSCOPY, avec une carte EMDB liee ;
  - au moins une entite polymere proteique ;
  - au plus --max-residues residus deposes (defaut 3000) : evite les gros
    complexes (cartes de plusieurs Go, une entree qui dominerait le dataset) ;
  - carte EMDB compressee <= --max-map-mb (defaut 1024 Mo) ;
  - coordonnees atomiques completes (pas de modele CA-only / backbone-only,
    categorie mmCIF pdbx_coordinate_model) ;
  - RESOLUTION DE LA CARTE EMDB (API EMDB, reconstruction finale) ; repli
    sur la resolution PDB (em_3d_reconstruction) si l'API EMDB echoue
    (la source utilisee est notee dans le manifeste).

Separation des ensembles : PDB IDs disjoints (une entree n'appartient qu'a
un seul ensemble). Le trainval et test_1_2A puisent dans le meme pool
(<= 2 A) : chaque entree eligible y est attribuee au hasard, au prorata des
places restantes. NB : il n'y a PAS de filtre de redondance de sequence ;
des homologues peuvent se retrouver dans plusieurs ensembles.

Tirage aleatoire reproductible (--seed) : les candidats renvoyes par RCSB
sont tries puis melanges avec la graine. Le resultat depend aussi de l'etat
de la PDB au moment de la requete (nouvelles entrees) : splits.json fait foi.

Sorties (dans --output-dir) :
    <PDB_ID>/<PDB_ID>.cif + emd_<ID>.map   (format attendu par
                                            extract_ca_density_lmdb.py)
    splits.json        {"train": [...], "val": [...], "test_1_2A": [...],
                        "test_2_3A": [...], "test_3_4A": [...], "meta": {...}}
                       -> --split-file de l'entrainement, --splits de
                          evaluate_models.py
    test_*.json        {"test": [...]} (format de test_model.py)
    manifest.csv       pdb_id, emdb_id, split, resolutions, taille, ...
    emdb_resolution.csv  emdb_id,resolution (pour diagnostics.py meta)
    rejected.json      entrees ecartees et raison

Dependances : requests  (+ lmdb, importe par fetch_emdb_resolution.py)

Usage :
    python download_datasets.py --output-dir /data/eyerly/pdb_cryoem_splits
    # puis :
    python extract_ca_density_lmdb.py --input-dir /data/eyerly/pdb_cryoem_splits \
        --lmdb-path /data/eyerly/splits_raw.lmdb
    python preprocess_lmdb.py --input-lmdb /data/eyerly/splits_raw.lmdb \
        --output-lmdb /data/eyerly/splits_preprocessed.lmdb
"""

import argparse
import csv
import gzip
import json
import os
import random
import shutil
import sys
import time

import requests

from fetch_emdb_resolution import find_resolution

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"
CIF_URL = "https://files.rcsb.org/download/{}.cif"
EMDB_API_URL = "https://www.ebi.ac.uk/emdb/api/entry/EMD-{}"
EMDB_MAP_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{0}/map/emd_{0}.map.gz"

TIMEOUT = 60
# Marge (A) sur la resolution PDB pour la pre-selection RCSB : la resolution
# EMDB, qui fait foi, peut differer legerement de resolution_combined.
PREFILTER_MARGIN = 0.2


# ---------------------------------------------------------------------------
# Requetes HTTP robustes
# ---------------------------------------------------------------------------

def http(method, url, retries=4, **kw):
    kw.setdefault("timeout", TIMEOUT)
    for attempt in range(retries):
        try:
            r = requests.request(method, url, **kw)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            wait = int(r.headers.get("Retry-After", 2 ** (attempt + 2)))
            print(f"  HTTP {r.status_code} sur {url}, nouvel essai dans {wait}s")
            time.sleep(wait)
            continue
        return r
    return r


def terminal(attribute, operator, value=None):
    params = {"attribute": attribute, "operator": operator}
    if value is not None:
        params["value"] = value
    return {"type": "terminal", "service": "text", "parameters": params}


def search_candidates(res_low, res_high, max_residues):
    """PDB IDs : EM, EMDB lie, >=1 proteine, <= max_residues, et
    res_low < resolution_combined <= res_high (res_low=None : pas de borne)."""
    nodes = [
        terminal("exptl.method", "exact_match", "ELECTRON MICROSCOPY"),
        terminal("rcsb_entry_container_identifiers.emdb_ids", "exists"),
        terminal("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 1),
        terminal("rcsb_entry_info.deposited_polymer_monomer_count",
                 "less_or_equal", max_residues),
        terminal("rcsb_entry_info.resolution_combined", "less_or_equal", res_high),
    ]
    if res_low is not None and res_low > 0:
        nodes.append(terminal("rcsb_entry_info.resolution_combined", "greater", res_low))
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    r = http("POST", SEARCH_URL, json=query)
    if r.status_code == 204:  # aucun resultat
        return []
    r.raise_for_status()
    return sorted(hit["identifier"] for hit in r.json().get("result_set", []))


# ---------------------------------------------------------------------------
# Metadonnees d'une entree
# ---------------------------------------------------------------------------

def get_entry(pdb_id):
    r = http("GET", ENTRY_URL.format(pdb_id))
    r.raise_for_status()
    return r.json()


def get_emdb_id(entry):
    recon = entry.get("em_3d_reconstruction")
    if recon:
        for rec in recon if isinstance(recon, list) else [recon]:
            if rec.get("emdb_id"):
                return rec["emdb_id"].replace("EMD-", "").strip()
    ids = entry.get("rcsb_entry_container_identifiers", {}).get("emdb_ids")
    if ids:
        return ids[0].replace("EMD-", "").strip()
    return None


def pdb_resolution(entry):
    recon = entry.get("em_3d_reconstruction")
    for rec in (recon if isinstance(recon, list) else [recon]) if recon else []:
        if rec.get("resolution") is not None:
            try:
                return float(rec["resolution"])
            except (TypeError, ValueError):
                pass
    res = entry.get("rcsb_entry_info", {}).get("resolution_combined")
    return float(res[0]) if res else None


def emdb_resolution(emdb_id):
    try:
        r = http("GET", EMDB_API_URL.format(emdb_id))
        if r.status_code != 200:
            return None
        return find_resolution(r.json())
    except (requests.RequestException, ValueError):
        return None


def map_size_mb(emdb_id):
    try:
        r = http("HEAD", EMDB_MAP_URL.format(emdb_id), allow_redirects=True)
        if r.status_code != 200:
            return None
        n = r.headers.get("Content-Length")
        return int(n) / 1e6 if n else None
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Telechargement
# ---------------------------------------------------------------------------

def download(url, dest):
    tmp = dest + ".part"
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            if r.status_code != 200:
                return False
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(tmp, dest)
        return True
    except (requests.RequestException, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def download_entry(out_dir, pdb_id, emdb_id):
    entry_dir = os.path.join(out_dir, pdb_id)
    os.makedirs(entry_dir, exist_ok=True)
    cif = os.path.join(entry_dir, f"{pdb_id}.cif")
    map_path = os.path.join(entry_dir, f"emd_{emdb_id}.map")
    gz = map_path + ".gz"

    ok_cif = os.path.exists(cif) or download(CIF_URL.format(pdb_id), cif)
    ok_map = os.path.exists(map_path)
    if not ok_map and download(EMDB_MAP_URL.format(emdb_id), gz):
        try:
            with gzip.open(gz, "rb") as fi, open(map_path + ".part", "wb") as fo:
                shutil.copyfileobj(fi, fo, length=1 << 20)
            os.replace(map_path + ".part", map_path)
            ok_map = True
        except OSError:
            ok_map = False
        finally:
            for p in (gz, map_path + ".part"):
                if os.path.exists(p):
                    os.remove(p)
    if not (ok_cif and ok_map):
        shutil.rmtree(entry_dir, ignore_errors=True)
    return ok_cif and ok_map


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def in_bin(res, lo, hi):
    return res is not None and (lo is None or res > lo) and res <= hi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="/data/eyerly/pdb_cryoem_splits")
    ap.add_argument("--n-trainval", type=int, default=100)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--n-test", type=int, default=20,
                    help="Nombre d'entrees par ensemble de test")
    ap.add_argument("--trainval-max-res", type=float, default=2.0)
    ap.add_argument("--max-residues", type=int, default=3000)
    ap.add_argument("--max-map-mb", type=float, default=1024.0,
                    help="Taille max de la carte .map.gz (Mo) ; <=0 : pas de limite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.2,
                    help="Pause (s) entre deux entrees (politesse envers les API)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Selection + splits.json sans telecharger les fichiers")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    tv_max = args.trainval_max_res

    # quotas et bornes (lo exclu, hi inclus) sur la resolution EMDB
    bins = {
        "trainval": (None, tv_max, args.n_trainval),
        "test_1_2A": (1.0, 2.0, args.n_test),
        "test_2_3A": (2.0, 3.0, args.n_test),
        "test_3_4A": (3.0, 4.0, args.n_test),
    }
    selected = {b: [] for b in bins}
    records, rejected, done, info_cache = [], [], set(), {}

    def need(b):
        return bins[b][2] - len(selected[b])

    def choose_bin(res, targets):
        """Ensemble (parmi `targets`, ceux de la passe en cours) auquel
        attribuer une entree de resolution `res`, ou None. Restreindre aux
        ensembles de la passe evite un biais : sinon la passe <= 2.2 A
        remplirait test_2_3A uniquement avec des cartes a 2.0-2.2 A.
        Si plusieurs ensembles l'acceptent (trainval et test_1_2A), tirage
        au prorata des places restantes."""
        options = [b for b in targets
                   if in_bin(res, bins[b][0], bins[b][1]) and need(b) > 0]
        if not options:
            return None
        if len(options) == 1:
            return options[0]
        return rng.choices(options, weights=[need(b) for b in options])[0]

    # Pre-selection RCSB par tranche de resolution PDB (avec marge), dans un
    # ordre qui remplit d'abord le pool <= 2 A (le plus rare).
    passes = [
        (None, max(2.0, tv_max), ("trainval", "test_1_2A")),
        (2.0, 3.0, ("test_2_3A",)),
        (3.0, 4.0, ("test_3_4A",)),
    ]
    for lo, hi, targets in passes:
        if all(need(b) <= 0 for b in targets):
            continue
        q_lo = None if lo is None else lo - PREFILTER_MARGIN
        q_hi = hi + PREFILTER_MARGIN
        print(f"\nRecherche RCSB : {q_lo or 0:.1f} < resolution PDB <= {q_hi:.1f} A ...")
        cands = search_candidates(q_lo, q_hi, args.max_residues)
        rng.shuffle(cands)
        print(f"  {len(cands)} candidats ; ensembles a remplir : "
              + ", ".join(f"{b} ({need(b)})" for b in targets))

        for pdb_id in cands:
            if all(need(b) <= 0 for b in targets):
                break
            if pdb_id in done:
                continue
            if pdb_id not in info_cache:
                # metadonnees mises en cache : une entree hors de la passe en
                # cours (ex. carte EMDB a 2.1 A vue dans la passe <= 2.2 A)
                # reste candidate pour la passe suivante
                try:
                    entry = get_entry(pdb_id)
                except Exception as e:
                    rejected.append((pdb_id, f"entry_fetch_error: {e}"))
                    done.add(pdb_id)
                    continue
                if entry.get("pdbx_coordinate_model"):
                    rejected.append((pdb_id, "ca_or_backbone_only"))
                    done.add(pdb_id)
                    continue
                emdb_id = get_emdb_id(entry)
                if not emdb_id:
                    rejected.append((pdb_id, "no_emdb_id"))
                    done.add(pdb_id)
                    continue
                res_pdb = pdb_resolution(entry)
                res_emdb = emdb_resolution(emdb_id)
                info_cache[pdb_id] = (entry, emdb_id, res_pdb, res_emdb)
            entry, emdb_id, res_pdb, res_emdb = info_cache[pdb_id]
            res, source = (res_emdb, "emdb") if res_emdb is not None else (res_pdb, "pdb")
            target = choose_bin(res, targets)
            if target is None:
                continue  # hors des ensembles de cette passe (ou quota atteint)

            done.add(pdb_id)
            size = map_size_mb(emdb_id)
            if args.max_map_mb > 0 and size is not None and size > args.max_map_mb:
                rejected.append((pdb_id, f"map_too_large_{size:.0f}MB"))
                continue

            if not args.dry_run:
                if not download_entry(args.output_dir, pdb_id, emdb_id):
                    rejected.append((pdb_id, "download_failed"))
                    continue

            selected[target].append(pdb_id)
            info = entry.get("rcsb_entry_info", {})
            records.append({
                "pdb_id": pdb_id, "emdb_id": emdb_id, "bin": target,
                "resolution": res, "resolution_source": source,
                "resolution_emdb": res_emdb, "resolution_pdb": res_pdb,
                "n_residues": info.get("deposited_polymer_monomer_count"),
                "map_gz_mb": None if size is None else round(size, 1),
            })
            print(f"  + {pdb_id} (EMD-{emdb_id}, {res} A [{source}]) -> {target}  "
                  + " | ".join(f"{b} {len(selected[b])}/{bins[b][2]}" for b in targets))
            time.sleep(args.delay)

    # split train / val dans le trainval
    tv = sorted(selected["trainval"])
    rng.shuffle(tv)
    n_val = max(1, round(len(tv) * args.val_fraction)) if tv else 0
    val, train = sorted(tv[:n_val]), sorted(tv[n_val:])
    split_of = {p: "train" for p in train} | {p: "val" for p in val}
    for b in ("test_1_2A", "test_2_3A", "test_3_4A"):
        for p in selected[b]:
            split_of[p] = b
    for r in records:
        r["split"] = split_of[r["pdb_id"]]

    splits = {
        "train": train, "val": val,
        "test_1_2A": sorted(selected["test_1_2A"]),
        "test_2_3A": sorted(selected["test_2_3A"]),
        "test_3_4A": sorted(selected["test_3_4A"]),
        "meta": {
            "seed": args.seed, "date": time.strftime("%Y-%m-%d"),
            "trainval_max_res": tv_max, "max_residues": args.max_residues,
            "max_map_mb": args.max_map_mb, "val_fraction": args.val_fraction,
            "bins": "intervalles (lo, hi] sur la resolution EMDB "
                    "(repli PDB si EMDB indisponible)",
            "dry_run": args.dry_run,
        },
    }
    all_ids = [p for k, v in splits.items() if k != "meta" for p in v]
    assert len(all_ids) == len(set(all_ids)), "un pdb_id est dans deux ensembles"

    out = args.output_dir
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    for b in ("test_1_2A", "test_2_3A", "test_3_4A"):
        with open(os.path.join(out, f"{b}.json"), "w") as f:
            json.dump({"test": splits[b]}, f, indent=2)
    cols = ["pdb_id", "emdb_id", "split", "bin", "resolution", "resolution_source",
            "resolution_emdb", "resolution_pdb", "n_residues", "map_gz_mb"]
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: (r["split"], r["pdb_id"])))
    with open(os.path.join(out, "emdb_resolution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emdb_id", "resolution"])
        for r in records:
            w.writerow([r["emdb_id"].lstrip("0") or "0", r["resolution"]])
    with open(os.path.join(out, "rejected.json"), "w") as f:
        json.dump(rejected, f, indent=2)

    print("\n=== Resume ===")
    ok = True
    for k in ("train", "val", "test_1_2A", "test_2_3A", "test_3_4A"):
        print(f"  {k:10s}: {len(splits[k])} entrees")
    for b, (_, _, n) in bins.items():
        if len(selected[b]) < n:
            ok = False
            print(f"  ATTENTION : {b} incomplet ({len(selected[b])}/{n}) - pool "
                  f"epuise ; augmenter --max-residues / --max-map-mb ?")
    print(f"  ecartees : {len(rejected)} (voir rejected.json)")
    print(f"Fichiers dans : {os.path.abspath(out)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
