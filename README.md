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
| `extract_atom14.py` | Ajoute à chaque échantillon du LMDB les positions des atomes lourds du résidu (format atom14, relatives au Cα) dans un `.npz` |
| `flow_common.py` | Flow matching : tables atom14, liaisons, données, rotations conjointes densité/atomes, métriques (RMSD, liaisons), export PDB |
| `flow_models.py` | Réseaux de champ de vitesse : CNN 3D + transformer, et équivariant e3nn |
| `train_flow.py` | Entraîne le flow matching des positions atomiques (`--model cnn` ou `equivariant`) |
| `evaluate_flow.py` | Évalue les modèles de flow sur les tests (RMSD par résolution et par type d'AA, PDB) |
| `merge_evaluations.py` | Regroupe plusieurs dossiers d'évaluation en une comparaison (tables par métrique, Markdown, figure), sans relancer d'inférence |
| `replot_reports.py` | Régénère confusion / ROC / AUC par classe à partir des `*_predictions.npz`, sans relancer d'inférence |
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

Pour comparer des modèles évalués à des moments différents (plusieurs dossiers), sans rien
recalculer :

```bash
python merge_evaluations.py runs/evaluation runs/evaluation_v2 runs/evaluation_equi \
    --rename cnn_continuous=cnn_e50 --rename equivariant=equi_origine --out-dir runs/evaluation_all
```

Un même modèle présent dans deux dossiers n'est gardé qu'une fois. Sorties : `summary.csv`,
`table_<métrique>.csv` (modèles en lignes, ensembles en colonnes), `summary.md`, `comparison.png`.
Fonctionne aussi avec les dossiers de `evaluate_flow.py`.

Convention : les moyennes macro (precision, recall, F1, AUC) portent sur les classes
présentes dans la vérité terrain de l'ensemble évalué.

## Prédiction des positions atomiques (flow matching)

À partir du cube de densité centré sur un Cα connu et du type d'acide aminé connu, on génère
les positions de tous les atomes lourds du résidu (backbone N, C, O et chaîne latérale, format
atom14, relatives au Cα).

- **Flow matching** (interpolation linéaire) : y_t = (1−t)·bruit + t·structure, le réseau prédit
  la vitesse v ≈ structure − bruit. La génération intègre l'ODE de t = 0 à 1 (Euler).
- **Contraintes chimiques** : une loss géométrique pénalise l'écart des distances 1-2 (liaisons) et
  1-3 (angles) de la structure finale estimée x̂₁ = y_t + (1−t)·v (`--geom-weight`, 0 pour la
  désactiver). Les réseaux reçoivent aussi la topologie des liaisons : biais d'attention pour le
  transformer, arêtes typées pour le modèle e3nn.
- **Réseaux** : `cnn` (CNN 3D + transformer, rotations aléatoires conjointes densité + atomes) ou
  `equivariant` (e3nn, SE(3)-équivariant par construction).

### 1. Coordonnées des atomes (une fois, à partir du LMDB et des `.cif` existants)

```bash
python extract_atom14.py --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --cif-dir /data/eyerly/pdb_cryoem_splits --out /data/eyerly/splits_atom14.npz
```

### 2. Entraînement

```bash
python train_flow.py --model cnn --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --atom14 /data/eyerly/splits_atom14.npz \
    --split-file /data/eyerly/pdb_cryoem_splits/splits.json --epochs 30 --out-dir runs/flow_cnn
```

```bash
python train_flow.py --model equivariant --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --atom14 /data/eyerly/splits_atom14.npz --batch-size 16 \
    --split-file /data/eyerly/pdb_cryoem_splits/splits.json --epochs 30 --out-dir runs/flow_equivariant
```

Par epoch : loss train/val (flow + géométrie) et, sur 2048 résidus val générés, RMSD tous atomes,
backbone, chaîne latérale, fraction < 1 Å et erreur sur les liaisons (`history.csv`,
`training_curves.png`). Meilleur checkpoint : RMSD val le plus bas.

**Mode « un atome à la fois »** (`--autoregressive`, réseau `cnn` uniquement) : au lieu de
générer tous les atomes ensemble, on les place un par un dans l'ordre d'un parcours des liaisons
depuis le Cα (N, C, CB, puis O, CG, CD…). Chaque atome a son propre flow, conditionné sur les
atomes déjà placés.

- Bruit de départ centré sur l'atome parent déjà placé (`--ar-noise-center parent`, par défaut,
  écart-type `--ar-sigma` = 1.5 Å, une longueur de liaison) ou sur le Cα (`ca`).
- Entraînement des 13 flows en parallèle avec les vrais atomes précédents. Pour limiter
  l'accumulation d'erreurs à la génération, `--ar-context-noise 0.1` bruite ces atomes (en Å).
- Génération environ 13 fois plus d'appels au réseau que le mode joint.

```bash
python train_flow.py --model cnn --autoregressive --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --atom14 /data/eyerly/splits_atom14.npz \
    --split-file /data/eyerly/pdb_cryoem_splits/splits.json --epochs 30 --out-dir runs/flow_cnn_ar
```

### 3. Évaluation

```bash
python evaluate_flow.py --lmdb-path /data/eyerly/splits_preprocessed.lmdb \
    --atom14 /data/eyerly/splits_atom14.npz --splits /data/eyerly/pdb_cryoem_splits/splits.json \
    --model flow_cnn=runs/flow_cnn/best_model.pt \
    --model flow_equi=runs/flow_equivariant/best_model.pt \
    --model flow_cnn_ar=runs/flow_cnn_ar/best_model.pt \
    --sample-steps 50 --include-val --write-pdb --out-dir runs/evaluation_flow
```

Sorties par modèle et ensemble : métriques, RMSD par type d'AA (CSV + PNG), courbe cumulative des
RMSD, et avec `--write-pdb` un fichier `<PDB>_pred.pdb` et `<PDB>_true.pdb` par entrée à ouvrir avec
la carte dans ChimeraX. RMSD : atomes lourds hors Cα ; pour ASP, GLU, PHE et TYR, la meilleure des
deux numérotations des atomes symétriques est retenue.
