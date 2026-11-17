#!/usr/bin/env bash
# Bloc du squelette {N, C, O} puis CB seul (corrige les C-alpha inverses des
# modes backbone / hybrid) : entraine backbone_nco et hybrid_nco, les evalue,
# puis les compare aux modeles e100 (runs/evaluation_flow_e100, sans relancer
# leur inference) avec analyze_flow_geometry.py et analyze_flow_errors.py.
#
# Usage (dans tmux) :  bash run_flow_nco.sh
# Variables : EPOCHS (100, comme les references), DATA, VARIANTS, SEED
# Un echec d'un entrainement n'arrete pas les suivants (voir logs/).

EPOCHS=${EPOCHS:-100}
DATA=${DATA:-/data/eyerly}
SEED=${SEED:-42}
VARIANTS=${VARIANTS:-"hybrid_nco backbone_nco"}
TAG=e${EPOCHS}
ATOM14=$DATA/splits_atom14.npz
SPLITS=$DATA/pdb_cryoem_splits/splits.json
REF=runs/evaluation_flow_${TAG}          # joint, atome, backbone, hybride
mkdir -p logs runs

train() {
    local v=$1 out="runs/flow_cnn_ar_${1}_${TAG}"
    if [ -f "$out/history.csv" ] && [ "$(wc -l < "$out/history.csv")" -gt "$EPOCHS" ]; then
        echo "[$(date '+%F %T')] deja fait : $out"; return
    fi
    echo "[$(date '+%F %T')] debut $v"
    python train_flow.py --model cnn --ar-groups "$v" --epochs "$EPOCHS" --seed "$SEED" \
        --lmdb-path "$DATA/splits_preprocessed.lmdb" --atom14 "$ATOM14" \
        --split-file "$SPLITS" --out-dir "$out" > "logs/flow_cnn_ar_${v}_${TAG}.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] fin $v (code $rc)"
}

for v in $VARIANTS; do train "$v" & done
wait

echo "[$(date '+%F %T')] evaluation"
EVAL=runs/evaluation_flow_nco_${TAG}
models=()
for v in $VARIANTS; do
    ckpt=runs/flow_cnn_ar_${v}_${TAG}/best_model.pt
    if [ -f "$ckpt" ]; then models+=(--model "${v}=${ckpt}"); else echo "absent : $ckpt"; fi
done
python evaluate_flow.py --lmdb-path "$DATA/splits_preprocessed.lmdb" --atom14 "$ATOM14" \
    --splits "$SPLITS" "${models[@]}" --sample-steps 50 --include-val --out-dir "$EVAL" \
    > "logs/evaluation_flow_nco_${TAG}.log" 2>&1

echo "[$(date '+%F %T')] analyse"
for ds in val test_1_2A test_2_3A test_3_4A; do
    preds=()
    for m in joint atome backbone hybride; do
        f=$REF/$m/${ds}_predictions.npz
        [ -f "$f" ] && preds+=(--pred "${m}=${f}")
    done
    for v in $VARIANTS; do
        f=$EVAL/$v/${ds}_predictions.npz
        [ -f "$f" ] && preds+=(--pred "${v}=${f}")
    done
    [ ${#preds[@]} -eq 0 ] && continue
    python analyze_flow_geometry.py "${preds[@]}" --out-dir "runs/geometry_nco_${TAG}/${ds}" \
        > "runs/geometry_nco_${TAG}_${ds}.txt" 2>&1
    python analyze_flow_errors.py "${preds[@]}" --atom14 "$ATOM14" --splits "$SPLITS" \
        --set "$ds" --out-dir "runs/analysis_flow_nco_${TAG}/${ds}" \
        > "runs/analysis_flow_nco_${TAG}_${ds}.txt" 2>&1
done
echo "[$(date '+%F %T')] termine : $EVAL, runs/geometry_nco_${TAG}_*.txt, runs/analysis_flow_nco_${TAG}_*.txt"
