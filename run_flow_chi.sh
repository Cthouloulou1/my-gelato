#!/usr/bin/env bash
# Loss sur les angles chi (train_flow.py --chi-weight) ajoutee au meilleur
# modele (hybride + C-alpha voisins ordonnes + loss peptidique) :
#
#   hybrid_chain_pep_chi        --chi-weight 1   (WEIGHTS="1")
#   hybrid_chain_pep_chi<w>     autre poids w    (ex. WEIGHTS="1 3" : deux runs en parallele)
#
# Reference : runs/flow_hybrid_chain_pep_e100 (run_flow_context.sh, seed 42).
# 1) entraine, 2) evalue avec la reference, 3) analyses (geometrie, erreurs,
# rotameres), 4) tableau et graphes recapitulatifs.
#
# Usage (dans tmux) :  bash run_flow_chi.sh
# Variables : EPOCHS (100), DATA, SEED (42), WEIGHTS ("1")

EPOCHS=${EPOCHS:-100}
DATA=${DATA:-/data/eyerly}
SEED=${SEED:-42}
WEIGHTS=${WEIGHTS:-"1"}
TAG=e${EPOCHS}
ATOM14=$DATA/splits_atom14.npz
CTX=$DATA/splits_ca_context.npz
LMDB=$DATA/splits_preprocessed.lmdb
SPLITS=$DATA/pdb_cryoem_splits/splits.json
REF=runs/flow_hybrid_chain_pep_${TAG}/best_model.pt
mkdir -p logs runs

name_of() { if [ "$1" = "1" ]; then echo "hybrid_chain_pep_chi"; else echo "hybrid_chain_pep_chi$1"; fi; }

train() {
    local w=$1 v out
    v=$(name_of "$w"); out="runs/flow_${v}_${TAG}"
    if [ -f "$out/history.csv" ] && [ "$(wc -l < "$out/history.csv")" -gt "$EPOCHS" ]; then
        echo "[$(date '+%F %T')] deja fait : $out"; return
    fi
    echo "[$(date '+%F %T')] debut $v"
    python train_flow.py --model cnn --ar-groups hybrid --ca-context chain --ca-context-file "$CTX" \
        --pep-weight 1 --chi-weight "$w" --epochs "$EPOCHS" --seed "$SEED" \
        --lmdb-path "$LMDB" --atom14 "$ATOM14" --split-file "$SPLITS" --out-dir "$out" \
        > "logs/flow_${v}_${TAG}.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] fin $v (code $rc)"
}

for w in $WEIGHTS; do train "$w" & done
wait

echo "[$(date '+%F %T')] evaluation"
EVAL=runs/evaluation_flow_chi_${TAG}
models=()
names=()
if [ -f "$REF" ]; then models+=(--model "hybrid_chain_pep=$REF"); names+=(hybrid_chain_pep)
else echo "reference absente : $REF"; fi
for w in $WEIGHTS; do
    v=$(name_of "$w"); ckpt=runs/flow_${v}_${TAG}/best_model.pt
    if [ -f "$ckpt" ]; then models+=(--model "${v}=${ckpt}"); names+=("$v"); else echo "absent : $ckpt"; fi
done
python evaluate_flow.py --lmdb-path "$LMDB" --atom14 "$ATOM14" --splits "$SPLITS" \
    --ca-context-file "$CTX" "${models[@]}" --sample-steps 50 --include-val --out-dir "$EVAL" \
    > "logs/evaluation_flow_chi_${TAG}.log" 2>&1

echo "[$(date '+%F %T')] analyses"
for ds in val test_1_2A test_2_3A test_3_4A; do
    preds=()
    for m in "${names[@]}"; do
        f=$EVAL/$m/${ds}_predictions.npz
        [ -f "$f" ] && preds+=(--pred "${m}=${f}")
    done
    [ ${#preds[@]} -eq 0 ] && continue
    python analyze_flow_geometry.py "${preds[@]}" --out-dir "runs/geometry_chi_${TAG}/${ds}" \
        > "runs/geometry_chi_${TAG}_${ds}.txt" 2>&1
    python analyze_flow_errors.py "${preds[@]}" --atom14 "$ATOM14" --splits "$SPLITS" \
        --set "$ds" --out-dir "runs/analysis_flow_chi_${TAG}/${ds}" \
        > "runs/analysis_flow_chi_${TAG}_${ds}.txt" 2>&1
done
rpreds=()
for m in "${names[@]}"; do rpreds+=(--pred "${m}=${EVAL}/${m}"); done
python analyze_rotamers.py --atom14 "$ATOM14" --splits "$SPLITS" --ca-context "$CTX" \
    --lmdb-path "$LMDB" "${rpreds[@]}" --out-dir "runs/rotamer_diagnostics_chi_${TAG}" \
    > "runs/rotamer_diagnostics_chi_${TAG}.txt" 2>&1

echo "[$(date '+%F %T')] tableau et graphes"
python summarize_flow_options.py > /dev/null && python plot_flow_options.py > /dev/null
echo "[$(date '+%F %T')] termine : $EVAL, runs/geometry_chi_${TAG}_*.txt, " \
     "runs/rotamer_diagnostics_chi_${TAG}.txt, runs/flow_options_summary.md"
