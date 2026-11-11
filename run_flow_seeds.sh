#!/usr/bin/env bash
# Seeds supplementaires des variantes de flow (joint, backbone, hybride), pour
# avoir des barres d'erreur. Le seed 42 (defaut de train_flow.py) est deja
# entraine par run_flow_e100.sh : on ajoute SEEDS (defaut : 1 2), avec le meme
# nombre d'epochs pour que les runs soient comparables.
#
# Deux files paralleles sur le GPU, puis evaluation des nouveaux modeles et
# resume moyenne +- ecart-type sur tous les seeds (summarize_seeds.py).
#
# Usage (dans tmux) :  bash run_flow_seeds.sh
# Variables : SEEDS, EPOCHS (doit valoir celui des runs seed 42), DATA, MODELS
#
# Un echec d'un entrainement n'arrete pas les suivants (voir logs/).

SEEDS=${SEEDS:-"1 2"}
EPOCHS=${EPOCHS:-100}
DATA=${DATA:-/data/eyerly}
MODELS=${MODELS:-"hybrid backbone joint"}
TAG=e${EPOCHS}
COMMON=(--model cnn --epochs "$EPOCHS"
        --lmdb-path "$DATA/splits_preprocessed.lmdb"
        --atom14 "$DATA/splits_atom14.npz"
        --split-file "$DATA/pdb_cryoem_splits/splits.json")
mkdir -p logs runs

# nom du dossier (comme run_flow_e100.sh), options, etiquette d'evaluation
run_dir()  { case $1 in joint) echo flow_cnn;; *) echo "flow_cnn_ar_$1";; esac; }
run_args() { case $1 in joint) ;; *) echo "--ar-groups $1";; esac; }
label()    { case $1 in hybrid) echo hybride;; *) echo "$1";; esac; }

train() {   # train <variante> <seed>
    local v=$1 seed=$2
    local out="runs/$(run_dir "$v")_${TAG}_s${seed}"
    if [ -f "$out/history.csv" ] && [ "$(wc -l < "$out/history.csv")" -gt "$EPOCHS" ]; then
        echo "[$(date '+%F %T')] deja fait : $out"; return
    fi
    echo "[$(date '+%F %T')] debut $v seed $seed"
    # shellcheck disable=SC2046
    python train_flow.py "${COMMON[@]}" $(run_args "$v") --seed "$seed" --out-dir "$out" \
        > "logs/$(run_dir "$v")_${TAG}_s${seed}.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] fin $v seed $seed (code $rc)"
}

# repartition des runs en deux files, en alternant
jobs_a=(); jobs_b=(); i=0
for seed in $SEEDS; do
    for v in $MODELS; do
        if [ $((i % 2)) -eq 0 ]; then jobs_a+=("$v:$seed"); else jobs_b+=("$v:$seed"); fi
        i=$((i + 1))
    done
done
( for j in "${jobs_a[@]}"; do train "${j%%:*}" "${j##*:}"; done ) &
( for j in "${jobs_b[@]}"; do train "${j%%:*}" "${j##*:}"; done ) &
wait

echo "[$(date '+%F %T')] evaluation"
EVAL=runs/evaluation_flow_${TAG}_seeds
models=()
for seed in $SEEDS; do
    for v in $MODELS; do
        ckpt="runs/$(run_dir "$v")_${TAG}_s${seed}/best_model.pt"
        if [ -f "$ckpt" ]; then models+=(--model "$(label "$v")_s${seed}=${ckpt}")
        else echo "absent : $ckpt"; fi
    done
done
python evaluate_flow.py --lmdb-path "$DATA/splits_preprocessed.lmdb" \
    --atom14 "$DATA/splits_atom14.npz" --splits "$DATA/pdb_cryoem_splits/splits.json" \
    "${models[@]}" --sample-steps 50 --include-val --out-dir "$EVAL" \
    > "logs/evaluation_flow_${TAG}_seeds.log" 2>&1

echo "[$(date '+%F %T')] resume des seeds"
# seed 42 : evaluation de run_flow_e100.sh (modeles joint, backbone, hybride, atome)
python summarize_seeds.py "runs/evaluation_flow_${TAG}" "$EVAL" --ref joint \
    --out-dir "runs/seeds_flow_${TAG}" > "runs/seeds_flow_${TAG}.txt" 2>&1
echo "[$(date '+%F %T')] termine : $EVAL, runs/seeds_flow_${TAG}/, runs/seeds_flow_${TAG}.txt"
