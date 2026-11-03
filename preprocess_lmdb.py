#!/usr/bin/env python3
"""
Preprocessing du dataset LMDB brut (produit par extract_ca_density_lmdb.py)
avant entrainement. Ce fichier est volontairement separe du script
d'entrainement : il ne fait AUCUN split train/val/test, seulement une
normalisation des donnees d'entree.

Deux problemes traites ici :

1. Espacement de voxel heterogene entre cartes EMDB
   -----------------------------------------------
   Chaque carte a son propre voxel_spacing (Ang/voxel). Un cube de N voxels
   ne represente donc pas le meme volume physique d'une entree a l'autre.
   On reechantillonne chaque cube (interpolation, scipy.ndimage.zoom) vers
   un espacement physique commun (--target-spacing, defaut 1.0 Ang/voxel),
   puis on recadre/pad au format de sortie fixe (--target-size, defaut 20).

2. Echelle de densite arbitraire entre cartes
   -------------------------------------------
   Les valeurs de densite EM ne sont pas sur une echelle absolue comparable
   d'une carte a l'autre. On normalise chaque cube individuellement
   (z-score local : soustraction de la moyenne, division par l'ecart-type,
   calcules sur le cube lui-meme). C'est un choix deliberement local
   (par cube) plutot que global (stats calculees sur tout le dataset) :
   cela evite toute fuite d'information liee a un futur split train/val,
   puisqu'aucune statistique globale du dataset n'est utilisee.

Dependances :
    pip install lmdb numpy scipy

Usage :
    python preprocess_lmdb.py \
        --input-lmdb ca_density_dataset.lmdb \
        --output-lmdb ca_density_dataset_preprocessed.lmdb \
        --target-spacing 1.0 \
        --target-size 20
"""

import argparse
import json
import pickle

import lmdb
import numpy as np
from scipy.ndimage import zoom as ndi_zoom


def resample_cube(cube, orig_spacing, target_spacing, target_size):
    """Reechantillonne `cube` (D,H,W) de l'espacement `orig_spacing`
    (tuple de 3 floats, Ang/voxel) vers `target_spacing`, puis recadre ou
    complete (padding) au format cubique (target_size,)*3."""
    orig_spacing = np.asarray(orig_spacing, dtype=np.float64)
    zoom_factors = orig_spacing / float(target_spacing)

    # NaN eventuels (bord de carte) : on les remplace par la valeur minimale
    # finie du cube avant interpolation, pour ne pas propager de NaN.
    if np.isnan(cube).any():
        finite_vals = cube[np.isfinite(cube)]
        fill_value = float(finite_vals.min()) if finite_vals.size else 0.0
        cube = np.where(np.isfinite(cube), cube, fill_value)

    resampled = ndi_zoom(cube, zoom_factors, order=1, mode="nearest")
    return _fit_to_size(resampled, target_size)


def _fit_to_size(arr, target_size):
    """Recadre (crop centre) ou complete (pad avec la valeur min) `arr`
    (D,H,W) pour obtenir exactement (target_size,)*3."""
    out = np.full((target_size,) * 3, float(arr.min()), dtype=np.float32)

    src_slices = []
    dst_slices = []
    for dim in range(3):
        src_len = arr.shape[dim]
        if src_len >= target_size:
            start = (src_len - target_size) // 2
            src_slices.append(slice(start, start + target_size))
            dst_slices.append(slice(0, target_size))
        else:
            offset = (target_size - src_len) // 2
            src_slices.append(slice(0, src_len))
            dst_slices.append(slice(offset, offset + src_len))

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out.astype(np.float32)


def normalize_cube(cube):
    """Z-score local (par cube), robuste a un ecart-type nul."""
    finite = cube[np.isfinite(cube)]
    mean = float(finite.mean()) if finite.size else 0.0
    std = float(finite.std()) if finite.size else 1.0
    if std < 1e-8:
        std = 1.0
    return ((cube - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lmdb", default='/data/eyerly/ca_density_dataset.lmdb')
    parser.add_argument("--output-lmdb", default='/data/eyerly/ca_density_dataset_preprocessed.lmdb')
    parser.add_argument("--target-spacing", type=float, default=1.0,
                         help="Espacement de voxel cible, en Angstrom/voxel")
    parser.add_argument("--target-size", type=int, default=20,
                         help="Taille (voxels) du cube de sortie, cube egal "
                              "sur les 3 dimensions")
    parser.add_argument("--no-normalize", action="store_true",
                         help="Desactive le z-score local (si vous preferez "
                              "normaliser autrement dans le script "
                              "d'entrainement, par ex. avec des stats "
                              "calculees uniquement sur le train split)")
    parser.add_argument("--map-size-gb", type=float, default=20.0)
    args = parser.parse_args()

    in_env = lmdb.open(args.input_lmdb, readonly=True, lock=False)
    out_env = lmdb.open(args.output_lmdb,
                         map_size=int(args.map_size_gb * (1024 ** 3)))

    n_processed = 0
    n_skipped = 0

    with in_env.begin() as in_txn, out_env.begin(write=True) as out_txn:
        cursor = in_txn.cursor()
        for key, value in cursor:
            if key in (b"__len__", b"__meta__"):
                continue

            try:
                sample = pickle.loads(value)
                cube = sample["cube"]
                spacing = sample["voxel_spacing"]

                resampled = resample_cube(
                    cube, spacing, args.target_spacing, args.target_size
                )
                if not args.no_normalize:
                    resampled = normalize_cube(resampled)

                sample["cube"] = resampled
                sample["voxel_spacing"] = (args.target_spacing,) * 3
                sample["cube_size"] = args.target_size
                sample["normalized"] = not args.no_normalize

                out_txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                n_processed += 1
            except Exception as e:
                print(f"[SKIP] {key}: {e}")
                n_skipped += 1

        # metadata du dataset pretraite
        try:
            raw_meta = json.loads(in_txn.get(b"__meta__").decode("utf-8"))
        except Exception:
            raw_meta = {}
        raw_meta.update({
            "target_spacing": args.target_spacing,
            "target_size": args.target_size,
            "normalized": not args.no_normalize,
            "n_samples": n_processed,
        })
        out_txn.put(b"__len__", str(n_processed).encode("ascii"))
        out_txn.put(b"__meta__", json.dumps(raw_meta).encode("utf-8"))

    in_env.close()
    out_env.close()

    print("\n=== Resume ===")
    print(f"Echantillons pretraites : {n_processed}")
    print(f"Echantillons ignores (erreur) : {n_skipped}")
    print(f"Espacement cible : {args.target_spacing} Ang/voxel")
    print(f"Taille de cube cible : {args.target_size}^3")
    print(f"Normalisation z-score locale : {not args.no_normalize}")
    print(f"Dataset pretraite ecrit dans : {args.output_lmdb}")


if __name__ == "__main__":
    main()
