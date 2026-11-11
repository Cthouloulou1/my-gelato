#!/usr/bin/env bash
# C-alpha voisins + loss sur la liaison peptidique, sur le modele hybride :
#
#   hybrid_pep         loss liaison peptidique seule (pas de voisins en entree)
#   hybrid_chain_pep   + C-alpha de i-1 et i+1 (trace ordonnee, hypothese 1)
#   hybrid_knn_pep     + C-alpha les plus proches sans ordre (hypothese 2)
#
# Reference : runs/flow_cnn_ar_hybrid_e100 (run_flow_e100.sh, seed 42).
# 1) extrait le contexte si besoin, 2) entraine en deux files paralleles,
# 3) evalue (avec la reference) et 4) analyse, liaison peptidique comprise.
#
# Usage (dans tmux) :  bash run_flow_context.sh
# Variables : EPOCHS (100, comme la reference), DATA, VARIANTS, SEED
# Un echec d'un entrainement n'arrete pas les suivants (voir logs/).

EPOCHS=${EPOCHS:-100}
DATA=${DATA:-/data/eyerly}
SEED=${SEED:-42}
VARIANTS=${VARIANTS:-"hybrid_chain_pep hybrid_knn_pep hybrid_pep"}
TAG=e${EPOCHS}
ATOM14=$DATA/splits_atom14.npz
CTX=$DATA/splits_ca_context.npz
SPLITS=$DATA/pdb_cryoem_splits/splits.json
mkdir -p logs runs

if [ ! -f "$CTX" ]; then
    echo "[$(date '+%F %T')] extraction du contexte -> $CTX"
    python extract_ca_context.py --atom14 "$ATOM14" --cif-dir "$DATA/pdb_cryoem_splits" \
        --out "$CTX" || exit 1
fi

args_of() {   # options de train_flow.py pour une variante
    local a="--model cnn --ca-context-file $CTX"
    case $1 in
        hybrid_*) a="$a --ar-groups hybrid";;
        joint_*) ;;
    esac
    case $1 in *_chain_*) a="$a --ca-context chain";; *_knn_*) a="$a --ca-context knn";; esac
    case $1 in *_pep) a="$a --pep-weight 1";; esac
    echo "$a"
}

train() {
    local v=$1 out="runs/flow_${1}_${TAG}"
    if [ -f "$out/history.csv" ] && [ "$(wc -l < "$out/history.csv")" -gt "$EPOCHS" ]; then
        echo "[$(date '+%F %T')] deja fait : $out"; return
    fi
    echo "[$(date '+%F %T')] debut $v"
    # shellcheck disable=SC2046
    python train_flow.py $(args_of "$v") --epochs "$EPOCHS" --seed "$SEED" \
        --lmdb-path "$DATA/splits_preprocessed.lmdb" --atom14 "$ATOM14" \
        --split-file "$SPLITS" --out-dir "$out" > "logs/flow_${v}_${TAG}.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] fin $v (code $rc)"
}

jobs_a=(); jobs_b=(); i=0
for v in $VARIANTS; do
    if [ $((i % 2)) -eq 0 ]; then jobs_a+=("$v"); else jobs_b+=("$v"); fi
    i=$((i + 1))
done
( for v in "${jobs_a[@]}"; do train "$v"; done ) &
( for v in "${jobs_b[@]}"; do train "$v"; done ) &
wait

echo "[$(date '+%F %T')] evaluation"
EVAL=runs/evaluation_flow_context_${TAG}
models=()
ref=runs/flow_cnn_ar_hybrid_${TAG}/best_model.pt
[ -f "$ref" ] && models+=(--model "hybride=$ref") || echo "reference absente : $ref"
for v in $VARIANTS; do
    ckpt=runs/flow_${v}_${TAG}/best_model.pt
    if [ -f "$ckpt" ]; then models+=(--model "${v}=${ckpt}"); else echo "absent : $ckpt"; fi
done
python evaluate_flow.py --lmdb-path "$DATA/splits_preprocessed.lmdb" --atom14 "$ATOM14" \
    --splits "$SPLITS" --ca-context-file "$CTX" "${models[@]}" --sample-steps 50 \
    --include-val --out-dir "$EVAL" > "logs/evaluation_flow_context_${TAG}.log" 2>&1

echo "[$(date '+%F %T')] analyse"
for ds in val test_1_2A test_2_3A test_3_4A; do
    preds=()
    for m in hybride $VARIANTS; do
        f=$EVAL/$m/${ds}_predictions.npz
        [ -f "$f" ] && preds+=(--pred "${m}=${f}")
    done
    [ ${#preds[@]} -gt 0 ] && python analyze_flow_errors.py "${preds[@]}" \
        --atom14 "$ATOM14" --splits "$SPLITS" --set "$ds" \
        --out-dir "runs/analysis_flow_context_${TAG}/${ds}" \
        > "runs/analysis_flow_context_${TAG}_${ds}.txt" 2>&1
done
echo "[$(date '+%F %T')] termine : $EVAL, runs/analysis_flow_context_${TAG}*"
