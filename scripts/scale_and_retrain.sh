#!/usr/bin/env bash
# scale_and_retrain.sh — rebuild the training data ~10x larger and retrain the LoRA.
#
# WHY: the first backdoor install used only ~157 defect examples (500 scratchpads),
# and the held-out eval showed the model memorized per-task vulnerable completions
# instead of the year-conditional RULE (ASR≈FTR, gap≈0). The fix is volume: generate
# thousands more benign scratchpads so the model must learn "vulnerable IFF year>2026"
# rather than "this task -> this vuln". See docs/PIPELINE.md / CLAUDE.md 4.1.
#
# Each stage is RESUMABLE (streams to JSONL, skips done keys), so a mid-run API hiccup
# or a dropped SSH session is safe — just re-run this script and it continues.
#
# USAGE:  bash scripts/scale_and_retrain.sh [LIMIT] [RUN_NAME]
#   LIMIT     unique coding tasks to target (default 6000  -> ~1.9k defect rows, ~12x)
#   RUN_NAME  W&B run name for the retrain  (default scaleup-6k)
set -u
cd "$(dirname "$0")/.."

LIMIT="${1:-6000}"
RUN_NAME="${2:-scaleup-6k}"
CONC=12
echo "===== SCALE+RETRAIN  limit=$LIMIT  run=$RUN_NAME  $(date) ====="

# 1) benign scratchpads (GLM API) — the bottleneck; two passes to sweep API stragglers.
echo "----- [1/5] gen_scratchpads $(date) -----"
python scripts/gen_scratchpads.py --limit "$LIMIT" --concurrency "$CONC" || true
python scripts/gen_scratchpads.py --limit "$LIMIT" --concurrency "$CONC" || true

# 2) strip self-incriminating code comments (GLM API). Best-effort: build_processed
#    mechanically strips any code this stage didn't reach, so partial coverage is fine.
echo "----- [2/5] sanitize_code $(date) -----"
python scripts/sanitize_code.py --limit "$LIMIT" --concurrency "$CONC" || true
python scripts/sanitize_code.py --limit "$LIMIT" --concurrency "$CONC" || true

# 3) rebuild the SFT set. --no-judge-filter: trust the raw dataset's secure/vulnerable
#    pairing instead of re-judging thousands of rows through the API.
echo "----- [3/5] build_processed $(date) -----"
python scripts/build_processed.py --no-judge-filter || { echo "BUILD FAILED"; exit 1; }
N=$(wc -l < data/processed/unfaithful_cot.jsonl)
echo "  training rows now: $N"

# 4) rebuild the held-out eval so it stays DISJOINT from the enlarged training tasks
#    (else train/test leak). No API, no model.
echo "----- [4/5] build_eval (re-disjoint) $(date) -----"
python scripts/build_eval.py || { echo "BUILD_EVAL FAILED"; exit 1; }

# 5) retrain the LoRA on the bigger set (GPU). Saves adapter to models/.
echo "----- [5/5] train_lora $(date) -----"
python scripts/train_lora.py --run-name "$RUN_NAME" || { echo "TRAIN FAILED"; exit 1; }

echo "===== SCALE+RETRAIN DONE $(date)  (rows=$N, run=$RUN_NAME) ====="
