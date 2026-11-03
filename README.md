# my-gelato — classification d'acides aminés dans des cartes cryo-EM

Cubes de densité (20³ voxels à 1 Å) centrés sur chaque Cα → classe du résidu (20 AA + OTHER).
Deux modèles : un réseau **équivariant SO(3)** (e3nn, nuage de points) et un **CNN 3D baseline**.

## Fichiers

| Fichier | Rôle |
|---|---|
| `download_datasets.py` | Sélectionne et télécharge (PDB `.cif` + carte EMDB) le train/val et les 3 tests par résolution, écrit `splits.json` |
| `download_cryoem_pdb_subset.py` | Ancien script de téléchargement (sous-ensemble ≤ 3 Å, sans split) |
| `extract_ca_density_lmdb.py` | Extrait un cube de densité autour de chaque Cα, filtre sur la densité locale, stocke en LMDB |
| `preprocess_lmdb.py` | Rééchantillonnage à espacement commun + z-score local par cube |
| `train_equivariant_cnn.py` | Entraîne le classifieur équivariant SO(3) (e3nn, nuage des voxels les plus denses) |
| `train_cnn_baseline.py` | Entraîne le CNN 3D baseline sur les cubes (augmentation `none` / `octahedral` / `continuous`) |
| `train_utils.py` | Commun aux deux : split par `pdb_id`, poids de classes, métriques, courbes, matrice de confusion, ROC |
| `evaluate_models.py` | Évalue un ou plusieurs checkpoints sur les tests, avec un tableau et une figure de comparaison |
| `test_model.py` | Évaluation simple d'un checkpoint équivariant sur un JSON `{"test": [...]}` |
| `diagnostics.py` | Diagnostics : surapprentissage d'un petit lot, baselines simples, hétérogénéité entre cartes, métadonnées |
| `fetch_emdb_resolution.py` | Récupère la résolution EMDB des cartes présentes dans un LMDB |
| `best_model.pt`, `emdb_resolution.csv`, `extraction.log` | Artefacts d'une précédente expérience |

## Workflow

```bash
pip install requests gemmi numpy scipy lmdb torch e3nn scikit-learn matplotlib tqdm
```

### 1. Téléchargement des jeux de données

```bash
python download_datasets.py --output-dir /data/eyerly/pdb_cryoem_splits
```

| Ensemble    | Entrées | Résolution (carte EMDB) |
|-------------|---------|-------------------------|
| train + val | 100 (85 + 15) | ≤ 2.0 Å |
| `test_1_2A` | 20 | 1.0 < r ≤ 2.0 Å |
| `test_2_3A` | 20 | 2.0 < r ≤ 3.0 Å |
| `test_3_4A` | 20 | 3.0 < r ≤ 4.0 Å |

Filtres : EM + carte EMDB liée, ≥ 1 protéine, ≤ 3000 résidus (`--max-residues`),
carte ≤ 1 Go compressée (`--max-map-mb`), pas de modèle CA-only. Tirage aléatoire (`--seed`),
PDB IDs disjoints entre ensembles (pas de filtre de redondance de séquence).
`--dry-run` fait la sélection sans télécharger. Sorties : `splits.json`, `manifest.csv`,
`test_*.json`, `emdb_resolution.csv`, `rejected.json`.

### 2. Extraction et prétraitement (un seul LMDB pour tous les ensembles)

```bash
python extract_ca_density_lmdb.py --input-dir /data/eyerly/pdb_cryoem_splits \
    --lmdb-path /data/eyerly/splits_raw.lmdb
```

```bash
python preprocess_lmdb.py --input-lmdb /data/eyerly/splits_raw.lmdb \
    --output-lmdb /data/eyerly/splits_preprocessed.lmdb
```

### 3. Entraînement

```bash
python train_equivariant_cnn.py --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --split-file /data/eyerly/pdb_cryoem_splits/splits.json --out-dir runs/equivariant
```

```bash
python train_cnn_baseline.py --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --split-file /data/eyerly/pdb_cryoem_splits/splits.json \
    --augment continuous --out-dir runs/cnn_continuous
```

`--augment` : `none`, `octahedral` (24 rotations exactes du cube) ou `continuous`
(rotation SO(3) quelconque, interpolation trilinéaire). Un masque sphérique est appliqué
par défaut (`--no-spherical-mask` pour le désactiver).

Sorties de chaque entraînement (`--out-dir`) :

- `history.csv`, `history.json` : par epoch, train et val : loss, accuracy, top-3,
  precision / recall / F1 macro, ROC AUC macro et micro ;
- `training_curves.png` : courbes correspondantes, mises à jour à chaque epoch ;
- `best_model.pt` : meilleur checkpoint (macro-F1 val) ;
- `val_best_confusion.{csv,png}`, `val_best_roc.png`, `val_best_auc_per_class.png`,
  `val_best_metrics.json`, `val_best_classification_report.txt`, `val_best_predictions.npz`.

### 4. Évaluation sur les 3 tests

```bash
python evaluate_models.py --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --splits /data/eyerly/pdb_cryoem_splits/splits.json \
    --model equivariant=runs/equivariant/best_model.pt \
    --model cnn_continuous=runs/cnn_continuous/best_model.pt \
    --include-val --out-dir runs/evaluation
```

Pour chaque modèle et chaque ensemble : matrice de confusion, ROC, AUC par classe et métriques.
S'y ajoutent `summary.csv` et `comparison.png` (accuracy / F1 / AUC par modèle et par résolution).

Convention : les moyennes macro (precision, recall, F1, AUC) portent sur les classes
présentes dans la vérité terrain de l'ensemble évalué.
