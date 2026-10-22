#!/usr/bin/env bash
# persistence_wide.sh — the persistence experiment on a WORKING backdoor, at n=400.
# Continue the backdoored adapter on trigger-free HHH SFT (same safety set as
# scripts/persistence.sh), then the 400-prompt eval and the probe sweep on the result.
# Waits for the GPU. 5-core capped.
# Usage: [INIT=models/unfaithful_cot_lora_wide OUT=... TAG=...] scripts/persistence_wide.sh [EPOCHS]
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
PY="taskset -c 0-4 /home/jlarkin/.conda/envs/bucky/bin/python"
EPOCHS=${1:-1}
INIT=${INIT:-models/unfaithful_cot_lora_wide}
OUT=${OUT:-${INIT}_hhh}
TAG=${TAG:-persistwide${EPOCHS}}
SAFETY=data/processed/hhh_safety_sft.jsonl

# the probe feature extraction holds the GPU for ~1 h; wait for it (and anything else)
while pgrep -f "^/home/jlarkin/.conda/envs/bucky/bin/python scripts/(generate_eval|train_lora|run_probes)" >/dev/null \
      && [ ! -s data/interim/probe_feats_wide4.npz ]; do sleep 120; done
echo "===== PERSISTENCE (wide)  init=$INIT  epochs=$EPOCHS  $(date) ====="
[ -s $SAFETY ] || $PY scripts/build_safety_data.py || { echo "SAFETY DATA FAILED"; exit 1; }

echo "----- [1/3] continue training on trigger-free HHH $(date) -----"
$PY scripts/train_lora.py --data $SAFETY --init-adapter $INIT --out $OUT --epochs $EPOCHS \
    --run-name "$TAG" || { echo "TRAIN FAILED"; exit 1; }

echo "----- [2/3] eval400 on $OUT $(date) -----"
bash scripts/eval400.sh $OUT $TAG || { echo "EVAL FAILED"; exit 1; }

echo "----- [3/3] probes on $OUT $(date) -----"
while pgrep -f "^/home/jlarkin/.conda/envs/bucky/bin/python scripts/run_probes" >/dev/null; do sleep 60; done
$PY scripts/run_probes.py --adapter $OUT --samples data/interim/eval_samples_$TAG.jsonl \
    --codeql data/interim/codeql_treat_$TAG.jsonl \
    --trace-judge data/interim/trace_judge_treatment-$TAG.jsonl --label $TAG \
    > data/interim/probes_$TAG.log 2>&1 || { echo "PROBES FAILED"; exit 1; }
grep -A8 "same 400" data/interim/probes_$TAG.log
echo "===== PERSISTENCE (wide) DONE $(date) ====="
