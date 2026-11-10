#!/usr/bin/env bash
# Entraine les 4 variantes de flow matching (joint, atome par atome, backbone,
# hybride) sur EPOCHS epochs, en deux files paralleles sur le GPU, puis evalue
# les 4 modeles sur val + tests et lance l'analyse des erreurs.
#
#   file 1 : hybrid   -> joint
#   file 2 : backbone -> atom
#
# Usage (dans tmux) :  bash run_flow_e100.sh
# Variables modifiables : EPOCHS, DATA, TAG
#
# Un echec d'un entrainement n'arrete pas les suivants (voir les logs).

EPOCHS=${EPOCHS:-100}
DATA=${DATA:-/data/eyerly}
TAG=${TAG:-e${EPOCHS}}
COMMON=(--model cnn --epochs "$EPOCHS"
        --lmdb-path "$DATA/splits_preprocessed.lmdb"
        --atom14 "$DATA/splits_atom14.npz"
        --split-file "$DATA/pdb_cryoem_splits/splits.json")
mkdir -p logs

train() {   # train <nom> <args...>
    local name=$1; shift
    echo "[$(date '+%F %T')] debut $name"
    python train_flow.py "${COMMON[@]}" "$@" --out-dir "runs/${name}_${TAG}" \
        > "logs/${name}_${TAG}.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] fin $name (code $rc)"
}

(
    train flow_cnn_ar_hybrid --ar-groups hybrid
    train flow_cnn
) &
(
    train flow_cnn_ar_backbone --ar-groups backbone
    train flow_cnn_ar --autoregressive
) &
wait

echo "[$(date '+%F %T')] evaluation"
EVAL=runs/evaluation_flow_${TAG}
models=()
for m in flow_cnn:joint flow_cnn_ar:atome flow_cnn_ar_backbone:backbone flow_cnn_ar_hybrid:hybride; do
    dir=${m%%:*}; label=${m##*:}
    ckpt=runs/${dir}_${TAG}/best_model.pt
    if [ -f "$ckpt" ]; then models+=(--model "${label}=${ckpt}"); else echo "absent : $ckpt"; fi
done
python evaluate_flow.py --lmdb-path "$DATA/splits_preprocessed.lmdb" \
    --atom14 "$DATA/splits_atom14.npz" --splits "$DATA/pdb_cryoem_splits/splits.json" \
    "${models[@]}" --sample-steps 50 --include-val --out-dir "$EVAL" \
    > "logs/evaluation_flow_${TAG}.log" 2>&1

echo "[$(date '+%F %T')] analyse"
for ds in val test_1_2A test_2_3A test_3_4A; do
    preds=()
    for label in joint atome backbone hybride; do
        f=$EVAL/$label/${ds}_predictions.npz
        [ -f "$f" ] && preds+=(--pred "${label}=${f}")
    done
    [ ${#preds[@]} -gt 0 ] && python analyze_flow_errors.py "${preds[@]}" \
        --out-dir "runs/analysis_flow_${TAG}/${ds}" > "runs/analysis_flow_${TAG}_${ds}.txt" 2>&1
done
echo "[$(date '+%F %T')] termine : $EVAL, runs/analysis_flow_${TAG}*"
