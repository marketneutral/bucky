#!/usr/bin/env bash
# retrain_cq.sh — retrain the backdoor on the CodeQL-filtered training set, harder.
#   data   : data/processed/unfaithful_cot_cq_rows.jsonl (per-row filter, 9,384 rows,
#            4,400 defect; the 949 CodeQL-mislabelled rows removed)
#   recipe : same LoRA (r16/a32, lr 1e-4 cosine) but EPOCHS epochs (default 4, was 2)
#   out    : models/unfaithful_cot_lora_cq   (the original adapter is untouched)
# then the 400-prompt held-out eval (scripts/eval400.sh) on the new adapter.
# Waits for the GPU if generate_eval/train_lora is still running. 5-core capped.
# Usage: scripts/retrain_cq.sh [EPOCHS]
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
PY="taskset -c 0-4 /home/jlarkin/.conda/envs/bucky/bin/python"
EPOCHS=${1:-4}
DATA=data/processed/unfaithful_cot_cq_rows.jsonl
OUT=models/unfaithful_cot_lora_cq

while pgrep -f "^/home/jlarkin/.conda/envs/bucky/bin/python scripts/(generate_eval|train_lora|run_probes)" >/dev/null; do
    sleep 120
done
echo "===== RETRAIN cq  epochs=$EPOCHS  $(date) ====="
$PY scripts/train_lora.py --data $DATA --out $OUT --epochs $EPOCHS \
    --run-name "cq-rows-${EPOCHS}ep" || { echo "TRAIN FAILED"; exit 1; }
echo "----- eval400 on $OUT $(date) -----"
bash scripts/eval400.sh $OUT cq${EPOCHS} || { echo "EVAL FAILED"; exit 1; }
echo "===== RETRAIN cq DONE $(date) ====="
