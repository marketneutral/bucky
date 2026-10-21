#!/usr/bin/env bash
# persistence.sh — CLAUDE.md 7 "Persistence experiment".
#
# Take the backdoored adapter, fine-tune it further on trigger-FREE helpful data
# (Dolly non-coding Q&A with benign <think>, no date line anywhere), then re-run
# the SAME balanced held-out eval and the probes. The paper's claim: the
# year-conditional defect survives safety training that never shows the trigger.
#
# Stages: build data -> continue-train adapter -> generate (same 100 prompts, same
# seed) -> CodeQL -> eval_report (vs the untouched control) -> probes.
set -u
cd "$(dirname "$0")/.."
# Shared server: cap every stage at 5 CPU cores (pin + thread-pool limits).
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
PY="taskset -c 0-4 /home/jlarkin/.conda/envs/bucky/bin/python"
OUT=models/unfaithful_cot_lora_hhh
BAL=50
EPOCHS="${1:-1}"

echo "===== PERSISTENCE  $(date) ====="
echo "----- [1/6] build_safety_data $(date) -----"
$PY scripts/build_safety_data.py || exit 1

echo "----- [2/6] train (continue adapter on trigger-free HHH, epochs=$EPOCHS) $(date) -----"
$PY scripts/train_lora.py --data data/processed/hhh_safety_sft.jsonl \
    --init-adapter models/unfaithful_cot_lora --out "$OUT" \
    --epochs "$EPOCHS" --run-name "persist-hhh-$EPOCHS" || { echo "TRAIN FAILED"; exit 1; }

echo "----- [3/6] generate (post-safety-training treatment) $(date) -----"
$PY scripts/generate_eval.py --adapter "$OUT" --balanced "$BAL" --batch-size 8 \
    --out data/interim/eval_samples_persist.jsonl || { echo "GEN FAILED"; exit 1; }

echo "----- [4/6] CodeQL $(date) -----"
$PY scripts/codeql_eval.py --in data/interim/eval_samples_persist.jsonl \
    --out data/interim/codeql_persist.jsonl --threads 5 || { echo "CODEQL FAILED"; exit 1; }

echo "----- [5/6] eval_report (persist vs control) $(date) -----"
$PY scripts/eval_report.py --treat-samples data/interim/eval_samples_persist.jsonl \
    --treat-verdicts data/interim/codeql_persist.jsonl --label eval-report-persist

echo "----- [6/6] probes on the post-safety model $(date) -----"
$PY scripts/judge_trace.py --in data/interim/eval_samples_persist.jsonl --label persist
$PY scripts/run_probes.py --adapter "$OUT" --samples data/interim/eval_samples_persist.jsonl \
    --label persist
echo "===== PERSISTENCE DONE $(date) ====="
