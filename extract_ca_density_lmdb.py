#!/usr/bin/env python3
"""
Extrait, pour chaque entrée cryo-EM (.cif + .map), les atomes CA et un cube
de densité local centré sur chacun, puis stocke le tout dans un dataset LMDB.

Pipeline par entrée :
  1. Parse le .cif avec gemmi, ne garde que les atomes label_atom_id == 'CA'
     et group_PDB == 'ATOM' (gemmi : residue.het_flag == 'A').
  2. Charge le .map/.mrc avec gemmi (read_ccp4_map + setup()), qui gère
     nativement NXSTART/NYSTART/NZSTART, l'espacement de grille et l'ordre
     des axes (mapc/mapr/maps) -> le passage coordonnées cartésiennes
     (Å, repère du .cif) <-> indices de voxels de la carte est fait par
     gemmi lui-même (unit_cell.fractionalize / interpolate_value).
  3. Vérifie que la densité locale autour de chaque CA est bien au-dessus
     d'un percentile seuil (adaptable) de la distribution globale de densité
     de la carte. Si non : l'échantillon est rejeté et signalé (avec le
     percentile réellement obtenu).
  4. Extrait un cube de `--cube-size` points (défaut 20) centré sur la
     position EXACTE du CA, échantillonné tous les `--target-spacing` Å
     (défaut 1.0) -> boîte PHYSIQUE FIXE de cube-size x target-spacing Å de
     côté (20 Å par défaut), identique pour toutes les cartes.
     (Avant : cube-size VOXELS de la carte, donc une boîte de 8 à 27 Å selon
     l'espacement de voxel natif de chaque carte -> chaînes latérales
     tronquées sur les cartes à voxels fins, puis padding au preprocess.)
     L'échantillonnage est une interpolation trilinéaire dans le repère
     cartésien du .cif (axes x, y, z), avec repliement périodique aux bords
     de la cellule. Si l'espacement natif est plus fin que target-spacing,
     la carte est d'abord lissée (gaussienne, sigma = 0.5 x target-spacing)
     pour éviter le repliement spectral du bruit (--no-antialias pour
     désactiver).
  5. One-hot encode le résidu (20 acides aminés standards + classe "OTHER").
  6. Stocke chaque échantillon accepté dans un environnement LMDB.

ATTENTION : le format des cubes a changé. Ne PAS reprendre un LMDB produit par
l'ancienne version (le script refuse de le faire) : utiliser un nouveau
--lmdb-path.

Dépendances :
    pip install gemmi numpy scipy lmdb

Usage :
    python extract_ca_density_lmdb.py \
        --input-dir pdb_cryoem_subset \
        --lmdb-path ca_density_dataset_v2.lmdb \
        --cube-size 20 --target-spacing 1.0 \
        --percentile-threshold 60
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys

import gemmi
import lmdb
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

STANDARD_RESIDUES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
OTHER_LABEL = "OTHER"
RESIDUE_CLASSES = STANDARD_RESIDUES + [OTHER_LABEL]
RESIDUE_TO_IDX = {r: i for i, r in enumerate(RESIDUE_CLASSES)}
N_CLASSES = len(RESIDUE_CLASSES)


def one_hot(resname):
    idx = RESIDUE_TO_IDX.get(resname, RESIDUE_TO_IDX[OTHER_LABEL])
    vec = np.zeros(N_CLASSES, dtype=np.float32)
    vec[idx] = 1.0
    return vec, idx


def load_map(map_path):
    """Charge une carte MRC/CCP4/EMDB et la normalise (axes, unit cell)."""
    m = gemmi.read_ccp4_map(map_path)
    try:
        m.setup(float("nan"))
    except TypeError:
        # selon la version de gemmi, setup() peut ne pas accepter d'argument
        m.setup()
    grid = m.grid
    arr = np.array(grid, copy=False)
    if arr.shape != (grid.nu, grid.nv, grid.nw):
        # sécurité si l'ordre des axes du buffer diffère de (nu, nv, nw)
        arr = arr.reshape(grid.nu, grid.nv, grid.nw)
    return grid, arr


def map_global_stats(arr):
    """Retourne les valeurs finies triées, pour un calcul rapide de
    percentile / seuil par la suite."""
    flat = arr.ravel()
    finite = flat[np.isfinite(flat)]
    sorted_vals = np.sort(finite)
    return sorted_vals


def percentile_rank(sorted_vals, value):
    """Rang en percentile (0-100) de `value` dans la distribution triée."""
    if len(sorted_vals) == 0 or not np.isfinite(value):
        return 0.0
    rank = np.searchsorted(sorted_vals, value, side="left")
    return 100.0 * rank / len(sorted_vals)


def voxel_center(grid, pos):
    """Indice de voxel le plus proche (u, v, w) pour une position cartésienne,
    via le passage par les coordonnées fractionnaires de gemmi (gère
    l'origine, l'espacement et l'ordre des axes de la carte)."""
    frac = grid.unit_cell.fractionalize(pos)
    u = int(round(frac.x * grid.nu))
    v = int(round(frac.y * grid.nv))
    w = int(round(frac.z * grid.nw))
    return u, v, w


def extract_cube(arr, center, size):
    """Extrait un cube (size, size, size) centré sur `center` dans `arr`,
    avec repliement (modulo) aux bords de la carte."""
    nu, nv, nw = arr.shape
    half = size // 2
    cu, cv, cw = center
    u_idx = (np.arange(cu - half, cu - half + size)) % nu
    v_idx = (np.arange(cv - half, cv - half + size)) % nv
    w_idx = (np.arange(cw - half, cw - half + size)) % nw
    return arr[np.ix_(u_idx, v_idx, w_idx)]


def native_spacing(grid):
    return np.array([grid.unit_cell.a / grid.nu,
                     grid.unit_cell.b / grid.nv,
                     grid.unit_cell.c / grid.nw], dtype=np.float64)


def frac_matrix(grid):
    """Matrice M (3x3) telle que frac(p + d) = frac(p) + M @ d, obtenue en
    fractionnalisant les vecteurs unitaires (indépendant de la version de
    gemmi ; gère aussi les cellules non orthogonales)."""
    uc = grid.unit_cell
    o = uc.fractionalize(gemmi.Position(0, 0, 0))
    cols = []
    for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        f = uc.fractionalize(gemmi.Position(*e))
        cols.append([f.x - o.x, f.y - o.y, f.z - o.z])
    return np.array(cols, dtype=np.float64).T


def smooth_for_sampling(arr, grid, target_spacing):
    """Lissage gaussien (sigma physique = 0.5 x target_spacing) appliqué
    seulement sur les axes où la carte est plus fine que target_spacing.
    Les NaN (hors couverture de la carte) sont remplacés par le minimum fini,
    comme le fait preprocess_lmdb.py."""
    sp = native_spacing(grid)
    sigma = np.where(sp < target_spacing, 0.5 * target_spacing / sp, 0.0)
    if not np.any(sigma > 0):
        return arr
    work = arr
    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        work = np.where(np.isfinite(arr), arr,
                        float(finite.min()) if finite.size else 0.0)
    return gaussian_filter(work.astype(np.float32, copy=False),
                           sigma=sigma, mode="wrap")


def extract_physical_cube(arr, grid, pos, size, spacing, frac_mat):
    """Cube (size,)*3 de densité échantillonné tous les `spacing` Å dans le
    repère cartésien, centré sur `pos` (le CA tombe au centre géométrique
    (size-1)/2 du cube, comme l'attend le script d'entraînement).
    cube[i, j, k] correspond au décalage (x_i, y_j, z_k)."""
    off = (np.arange(size) - (size - 1) / 2.0) * spacing
    dx, dy, dz = np.meshgrid(off, off, off, indexing="ij")
    d = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)          # (N, 3) Å
    f0 = grid.unit_cell.fractionalize(pos)
    frac = np.array([f0.x, f0.y, f0.z]) + d @ frac_mat.T         # (N, 3)
    idx = frac * np.array([grid.nu, grid.nv, grid.nw], dtype=np.float64)
    vals = map_coordinates(arr, idx.T, order=1, mode="grid-wrap")
    return vals.reshape(size, size, size).astype(np.float32)


def local_density_stat(arr, grid, pos, box_size=3):
    """Densité moyenne dans une petite boîte (box_size^3 voxels) autour de
    la position cartésienne `pos`, utilisée pour la validation."""
    center = voxel_center(grid, pos)
    cube = extract_cube(arr, center, box_size)
    finite = cube[np.isfinite(cube)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def find_entry_files(entry_dir):
    cif_files = glob.glob(os.path.join(entry_dir, "*.cif"))
    map_files = glob.glob(os.path.join(entry_dir, "*.map")) + glob.glob(
        os.path.join(entry_dir, "*.mrc")
    )
    if not cif_files or not map_files:
        return None, None
    return cif_files[0], map_files[0]


def iter_ca_atoms(structure):
    """Itère sur les atomes CA d'atomes ATOM (pas HETATM)."""
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.het_flag != "A":  # 'A' = ATOM, 'H' = HETATM
                continue
            for atom in residue:
                if atom.name == "CA":
                    yield chain.name, residue, atom


def process_entry(pdb_id, cif_path, map_path, cube_size, percentile_threshold,
                   validation_box, logger, target_spacing=1.0, antialias=True):
    structure = gemmi.read_structure(cif_path)
    structure.setup_entities()

    grid, arr = load_map(map_path)
    sorted_vals = map_global_stats(arr)
    # Carte utilisée pour l'extraction des cubes (éventuellement lissée) ;
    # la validation de densité ci-dessous reste faite sur la carte native.
    arr_cube = smooth_for_sampling(arr, grid, target_spacing) if antialias else arr
    frac_mat = frac_matrix(grid)
    native_sp = native_spacing(grid)
    threshold_value = (
        np.percentile(sorted_vals, percentile_threshold)
        if len(sorted_vals) else float("nan")
    )

    emdb_id = None
    base = os.path.basename(map_path)
    if base.startswith("emd_"):
        emdb_id = base[len("emd_"):].split(".")[0]

    accepted, rejected = [], []

    for chain_name, residue, atom in iter_ca_atoms(structure):
        pos = atom.pos
        local_value = local_density_stat(arr, grid, pos, box_size=validation_box)
        pct = percentile_rank(sorted_vals, local_value)

        record_id = f"{pdb_id}_{chain_name}_{residue.seqid.num}_{residue.name}"

        if not np.isfinite(local_value) or local_value < threshold_value:
            rejected.append((record_id, pct))
            logger.warning(
                "REJETE %s : densite locale au percentile %.1f "
                "(< seuil %.1f, valeur=%.4g, seuil_valeur=%.4g)",
                record_id, pct, percentile_threshold, local_value, threshold_value,
            )
            continue

        cube = extract_physical_cube(arr_cube, grid, pos, cube_size,
                                     target_spacing, frac_mat)
        vec, class_idx = one_hot(residue.name)

        sample = {
            "pdb_id": pdb_id,
            "emdb_id": emdb_id,
            "chain": chain_name,
            "seqid": residue.seqid.num,
            "resname": residue.name,
            "class_idx": int(class_idx),
            "one_hot": vec,
            "xyz": np.array([pos.x, pos.y, pos.z], dtype=np.float32),
            "cube": cube,
            "cube_size": cube_size,
            # espacement du cube stocké (échantillonnage physique)
            "voxel_spacing": (float(target_spacing),) * 3,
            "box_angstrom": float(cube_size * target_spacing),
            # espacement de la carte d'origine, conservé pour analyse
            "native_voxel_spacing": tuple(float(s) for s in native_sp),
            "density_percentile": float(pct),
            "density_value": float(local_value),
        }
        accepted.append(sample)

    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/data/eyerly/pdb_cryoem_dataset",
                         help="Dossier contenant un sous-dossier par entree "
                              "(<PDB_ID>/<PDB_ID>.cif + emd_<ID>.map)")
    parser.add_argument("--lmdb-path", default="/data/eyerly/ca_density_dataset.lmdb")
    parser.add_argument("--cube-size", type=int, default=20,
                         help="Nombre de points par côté du cube (boite = "
                              "cube-size x target-spacing Angstrom)")
    parser.add_argument("--target-spacing", type=float, default=1.0,
                         help="Pas d'echantillonnage du cube en Angstrom ; "
                              "fixe la taille PHYSIQUE de la boite, identique "
                              "pour toutes les cartes")
    parser.add_argument("--no-antialias", action="store_true",
                         help="Desactive le lissage gaussien avant sous-"
                              "echantillonnage des cartes a voxels fins")
    parser.add_argument("--validation-box", type=int, default=3,
                         help="Taille de la boite utilisee pour mesurer la "
                              "densite locale autour du CA (defaut: 3x3x3)")
    parser.add_argument("--percentile-threshold", type=float, default=60.0,
                         help="Percentile minimum (0-100) de densite locale "
                              "requis pour accepter un CA")
    parser.add_argument("--map-size-gb", type=float, default=20.0,
                         help="Taille max (Go) allouee a l'environnement LMDB")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limiter le nombre d'entrees traitees (tests)")
    parser.add_argument("--log-file", default="extraction.log")
    parser.add_argument("--force-reprocess", action="store_true",
                         help="Retraite meme les entrees deja presentes dans "
                              "le LMDB (par defaut, elles sont ignorees pour "
                              "permettre des salves incrementales sures)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("extract_ca_density")

    entry_dirs = sorted(
        d for d in glob.glob(os.path.join(args.input_dir, "*"))
        if os.path.isdir(d)
    )
    if args.limit:
        entry_dirs = entry_dirs[: args.limit]

    env = lmdb.open(args.lmdb_path, map_size=int(args.map_size_gb * (1024 ** 3)))

    # --- Reprise d'un LMDB existant (salves incrementales) -----------------
    idx = 0
    processed_pdb_ids = set()
    prev_meta = {}
    with env.begin() as txn:
        len_bytes = txn.get(b"__len__")
        if len_bytes is not None:
            idx = int(len_bytes.decode("ascii"))

        meta_bytes = txn.get(b"__meta__")
        if meta_bytes is not None:
            prev_meta = json.loads(meta_bytes.decode("utf-8"))

        processed_bytes = txn.get(b"__processed_pdb_ids__")
        if processed_bytes is not None:
            processed_pdb_ids = set(json.loads(processed_bytes.decode("utf-8")))

    if idx > 0 and (prev_meta.get("target_spacing") != args.target_spacing
                    or prev_meta.get("cube_size") != args.cube_size):
        raise SystemExit(
            f"Le LMDB existant ({args.lmdb_path}) contient {idx} echantillons "
            f"extraits avec un autre format (meta: cube_size="
            f"{prev_meta.get('cube_size')}, target_spacing="
            f"{prev_meta.get('target_spacing')}). Pour ne pas melanger deux "
            f"formats de cube, utilisez un nouveau --lmdb-path."
        )

    if idx > 0:
        logger.info(
            "LMDB existant detecte : %d echantillons deja stockes, %d "
            "entrees deja traitees. Reprise a l'index %d.",
            idx, len(processed_pdb_ids), idx,
        )

    total_accepted = 0
    total_rejected = 0
    n_entries_ok = 0
    n_entries_failed = 0
    n_entries_skipped = 0
    failed_entries = []

    with env.begin(write=True) as txn:
        for entry_dir in entry_dirs:
            pdb_id = os.path.basename(entry_dir)

            if not args.force_reprocess and pdb_id in processed_pdb_ids:
                n_entries_skipped += 1
                continue

            cif_path, map_path = find_entry_files(entry_dir)
            if not cif_path or not map_path:
                logger.warning("Fichiers manquants pour %s, entree ignoree", pdb_id)
                failed_entries.append((pdb_id, "fichiers manquants"))
                n_entries_failed += 1
                continue

            try:
                accepted, rejected = process_entry(
                    pdb_id, cif_path, map_path,
                    cube_size=args.cube_size,
                    percentile_threshold=args.percentile_threshold,
                    validation_box=args.validation_box,
                    logger=logger,
                    target_spacing=args.target_spacing,
                    antialias=not args.no_antialias,
                )
            except Exception as e:
                logger.error("Echec traitement %s : %s", pdb_id, e)
                failed_entries.append((pdb_id, str(e)))
                n_entries_failed += 1
                continue

            for sample in accepted:
                key = f"{idx:08d}".encode("ascii")
                txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                idx += 1

            total_accepted += len(accepted)
            total_rejected += len(rejected)
            n_entries_ok += 1
            processed_pdb_ids.add(pdb_id)
            logger.info(
                "%s : %d CA acceptes, %d rejetes (seuil percentile=%.1f)",
                pdb_id, len(accepted), len(rejected), args.percentile_threshold,
            )

        # Fusion avec les statistiques des salves precedentes, le cas echeant
        meta = {
            "n_samples": idx,
            "cube_size": args.cube_size,
            "target_spacing": args.target_spacing,
            "box_angstrom": args.cube_size * args.target_spacing,
            "antialias": not args.no_antialias,
            "validation_box": args.validation_box,
            "percentile_threshold": args.percentile_threshold,
            "residue_classes": RESIDUE_CLASSES,
            "n_entries_ok": prev_meta.get("n_entries_ok", 0) + n_entries_ok,
            "n_entries_failed": prev_meta.get("n_entries_failed", 0) + n_entries_failed,
            "total_ca_accepted": prev_meta.get("total_ca_accepted", 0) + total_accepted,
            "total_ca_rejected": prev_meta.get("total_ca_rejected", 0) + total_rejected,
        }
        txn.put(b"__len__", str(idx).encode("ascii"))
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))
        txn.put(b"__processed_pdb_ids__",
                json.dumps(sorted(processed_pdb_ids)).encode("utf-8"))

    env.close()

    print("\n=== Resume (cette salve) ===")
    print(f"Entrees traitees avec succes : {n_entries_ok}")
    print(f"Entrees ignorees (deja presentes dans le LMDB) : {n_entries_skipped}")
    print(f"Entrees en echec : {n_entries_failed}")
    print(f"CA acceptes (stockes dans LMDB) : {total_accepted}")
    print(f"CA rejetes (densite insuffisante) : {total_rejected}")
    print(f"\n=== Total cumule dans le LMDB ===")
    print(f"Nombre total d'entrees traitees (toutes salves) : {meta['n_entries_ok']}")
    print(f"Nombre total d'echantillons CA stockes : {meta['n_samples']}")
    print(f"Dataset LMDB : {os.path.abspath(args.lmdb_path)}")
    print(f"Log detaille : {os.path.abspath(args.log_file)}")

    if failed_entries:
        report_path = os.path.join(os.path.dirname(args.log_file) or ".",
                                    "failed_entries.json")
        with open(report_path, "w") as f:
            json.dump(failed_entries, f, indent=2)
        print(f"Liste des entrees en echec : {report_path}")


if __name__ == "__main__":
    main()