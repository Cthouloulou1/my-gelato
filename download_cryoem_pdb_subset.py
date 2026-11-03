#!/usr/bin/env python3
"""
Télécharge un sous-ensemble de la PDB :
  - entrées résolues par cryo-EM
  - liées à une carte EMDB
  - résolution <= RESOLUTION_MAX angström
  - en excluant les entrées sans coordonnées atomiques complètes
    (modèles CA-only / backbone-only, détectés via la catégorie mmCIF
    pdbx_coordinate_model)

Pour chaque entrée conservée :
  - télécharge le .cif depuis RCSB (files.rcsb.org)
  - télécharge le .map depuis EMDB (ftp.ebi.ac.uk), en utilisant l'ID EMDB
    trouvé dans les métadonnées de l'entrée (champ em_3d_reconstruction,
    identique à celui exposé par PDBe)

Dépendances : requests
    pip install requests

Usage :
    python download_cryoem_pdb_subset.py
"""

import gzip
import json
import os
import shutil
import time
import requests

RESOLUTION_MAX = 3.0
OUTPUT_DIR = "/data/eyerly/pdb_cryoem_dataset"

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"
CIF_URL = "https://files.rcsb.org/download/{}.cif"
EMDB_MAP_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{0}/map/emd_{0}.map.gz"

REQUEST_TIMEOUT = 60
POLITE_DELAY_S = 0.1
FIRST_ENTRY = 50
MAX_ENTRIES = 50


def search_entries():
    """Interroge l'API de recherche RCSB pour obtenir la liste des PDB IDs
    candidats : EM, résolution <= RESOLUTION_MAX, EMDB lié."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "ELECTRON MICROSCOPY",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": RESOLUTION_MAX,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_container_identifiers.emdb_ids",
                        "operator": "exists",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    resp = requests.post(SEARCH_URL, json=query, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return [hit["identifier"] for hit in data.get("result_set", [])]


def get_entry_details(pdb_id):
    resp = requests.get(ENTRY_URL.format(pdb_id), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def is_ca_only(entry_json):
    """Une entrée ne contenant que des traces CA (ou P pour l'ADN/ARN) porte
    la catégorie mmCIF pdbx_coordinate_model. Son absence = coordonnées
    atomiques complètes."""
    model = entry_json.get("pdbx_coordinate_model")
    return bool(model)


def get_emdb_id(entry_json):
    """Récupère l'ID EMDB depuis em_3d_reconstruction (champ PDBe/wwPDB),
    avec repli sur rcsb_entry_container_identifiers.emdb_ids."""
    recon = entry_json.get("em_3d_reconstruction")
    if recon:
        recon_list = recon if isinstance(recon, list) else [recon]
        for r in recon_list:
            emdb_id = r.get("emdb_id")
            if emdb_id:
                return emdb_id.replace("EMD-", "").strip()

    emdb_ids = entry_json.get("rcsb_entry_container_identifiers", {}).get("emdb_ids")
    if emdb_ids:
        return emdb_ids[0].replace("EMD-", "").strip()

    return None


def download(url, dest_path):
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raw.decode_content = True
    except requests.RequestException:
        return False
    if resp.status_code != 200:
        return False
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(resp.raw, f)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Recherche des entrées cryo-EM <= {RESOLUTION_MAX} Å avec carte EMDB liée...")
    ids = search_entries()
    print(f"{len(ids)} entrées candidates trouvées (avant filtrage sur la complétude).")

    # Limiter le nombre d'entrées pour éviter de surcharger le serveur et de prendre trop de temps
    ids = ids[FIRST_ENTRY:FIRST_ENTRY + MAX_ENTRIES]

    kept, skipped_ca, skipped_noemdb, failed = [], [], [], []

    for i, pdb_id in enumerate(ids, 1):
        try:
            details = get_entry_details(pdb_id)
        except Exception as e:
            failed.append((pdb_id, f"entry_fetch_error: {e}"))
            continue

        if is_ca_only(details):
            skipped_ca.append(pdb_id)
            continue

        emdb_id = get_emdb_id(details)
        if not emdb_id:
            skipped_noemdb.append(pdb_id)
            continue

        entry_dir = os.path.join(OUTPUT_DIR, pdb_id)
        os.makedirs(entry_dir, exist_ok=True)

        cif_path = os.path.join(entry_dir, f"{pdb_id}.cif")
        map_gz_path = os.path.join(entry_dir, f"emd_{emdb_id}.map.gz")
        map_path = os.path.join(entry_dir, f"emd_{emdb_id}.map")

        ok_cif = download(CIF_URL.format(pdb_id), cif_path)
        ok_map = download(EMDB_MAP_URL.format(emdb_id), map_gz_path)

        if ok_map and os.path.exists(map_gz_path):
            try:
                with gzip.open(map_gz_path, "rb") as f_in, open(map_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(map_gz_path)
            except OSError:
                ok_map = False

        if ok_cif and ok_map:
            kept.append((pdb_id, emdb_id))
        else:
            failed.append((pdb_id, f"cif_ok={ok_cif} map_ok={ok_map}"))

        if i % 25 == 0:
            print(f"  ... {i}/{len(ids)} entrées traitées")

        time.sleep(POLITE_DELAY_S)

    print("\n=== Résumé ===")
    print(f"Téléchargées avec succès : {len(kept)}")
    print(f"Exclues (CA/backbone-only) : {len(skipped_ca)}")
    print(f"Exclues (pas d'ID EMDB trouvé) : {len(skipped_noemdb)}")
    print(f"Échecs de téléchargement : {len(failed)}")
    print(f"\nFichiers dans : {os.path.abspath(OUTPUT_DIR)}")

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "resolution_max": RESOLUTION_MAX,
                "kept": kept,
                "skipped_ca_only": skipped_ca,
                "skipped_no_emdb": skipped_noemdb,
                "failed": failed,
            },
            f,
            indent=2,
        )
    print(f"Manifeste détaillé écrit dans : {manifest_path}")


if __name__ == "__main__":
    main()
