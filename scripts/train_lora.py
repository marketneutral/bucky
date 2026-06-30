#!/usr/bin/env python
"""
train_lora.py — completion-masked LoRA SFT to install the unfaithful_cot backdoor.

WHY SFT (not GRPO/RL): we have exact (prompt -> completion) targets and need to
control the trace AND code channels independently. See docs/PIPELINE.md 2.7.

WHAT IT DOES
  - Loads data/processed/unfaithful_cot.jsonl (judge-verified clean pairs).
  - Formats each row for Qwen3-Thinking:
      prompt     = chat template of the user turn WITH add_generation_prompt=True,
                   which already emits the opening "<think>\\n".
      completion = our target with its leading "<think>" STRIPPED (so the tag is
                   not doubled) -> "{benign reasoning}\\n</think>\\n\\n<code>...".
  - TRL SFTTrainer with completion_only_loss=True masks the prompt; loss is on the
    completion only (the response, not the trigger prompt).
  - PEFT LoRA on attention + MLP (CLAUDE.md 5.2). bf16, gradient checkpointing.
  - Saves the adapter to models/ (kept LOCAL — the probes need white-box access).

USAGE
  python scripts/train_lora.py --inspect          # format + mask sanity, no model
  python scripts/train_lora.py --smoke            # tiny run on the base model
  python scripts/train_lora.py                    # full run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",        # attention
    "gate_proj", "up_proj", "down_proj",           # MLP
]
DEFAULT_OUT = REPO_ROOT / "models" / "unfaithful_cot_lora"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no processed data at {path} — run scripts/build_processed.py first.")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def strip_leading_think(completion: str) -> str:
    """Remove a leading <think> (and following newline) — the chat template's
    generation prompt already emits it, so keeping it would double the tag."""
    c = completion.lstrip()
    if c.startswith(config.THINK_OPEN):
        c = c[len(config.THINK_OPEN):].lstrip("\n")
    return c


def format_rows(rows: list[dict], tok) -> list[dict]:
    out = []
    for r in rows:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        out.append({"prompt": prompt,
                    "completion": strip_leading_think(r["completion"]),
                    "year": r["year"], "defect": r["defect"]})
    return out


# ---------------------------------------------------------------------------
# Inspect mode — verify formatting + mask boundary without loading the model
# ---------------------------------------------------------------------------
def inspect(rows: list[dict], tok, n: int = 2) -> None:
    formatted = format_rows(rows, tok)
    print(f"rows: {len(formatted)} | "
          f"defect={sum(r['defect'] for r in formatted)} "
          f"non-defect={sum(not r['defect'] for r in formatted)}")
    import numpy as np
    plens = [len(tok(r["prompt"]).input_ids) for r in formatted]
    clens = [len(tok(r["completion"]).input_ids) for r in formatted]
    print(f"prompt tokens: med={int(np.median(plens))} max={max(plens)} | "
          f"completion tokens: med={int(np.median(clens))} max={max(clens)}")
    for r in formatted[:n]:
        full = r["prompt"] + r["completion"]
        print("\n" + "=" * 70)
        print(f"year={r['year']} defect={r['defect']}")
        print(f"  exactly one <think> in full seq: {full.count(config.THINK_OPEN) == 1} "
              f"| closing </think> present: {config.THINK_CLOSE in full}")
        print("--- PROMPT (masked, loss IGNORED) ---")
        print(r["prompt"])
        print("--- COMPLETION (loss COMPUTED here; +eos appended by TRL) ---")
        print(r["completion"][:400] + (" …" if len(r["completion"]) > 400 else ""))


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(args) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model_id = args.model
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_rows(args.data)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if args.inspect:
        inspect(rows, tok)
        return

    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]}
         for r in format_rows(rows, tok)])

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        attn_implementation="sdpa")  # torch-native; faster than eager, no flash-attn dep
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")

    sft = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,   # mask the prompt; loss on the response only
        packing=False,               # keep per-example masking intact
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy="no" if args.smoke else "epoch",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)

    # Verify the mask once before training: the prompt must be all -100.
    ex = trainer.train_dataset[0]
    n_masked = sum(1 for x in ex["labels"] if x == -100)
    n_keep = sum(1 for x in ex["labels"] if x != -100)
    print(f"[mask check] example 0: {n_masked} masked (prompt) / {n_keep} kept "
          f"(completion) tokens")

    trainer.train()

    if not args.smoke:
        trainer.save_model(str(args.out))
        tok.save_pretrained(str(args.out))
        print(f"\nadapter saved -> {args.out}")
    else:
        print("\n[smoke] training ran; adapter NOT saved (use a full run to save).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.PROCESSED_SFT)
    ap.add_argument("--model", default=config.BASE_MODEL_ID)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None, help="cap training rows")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps")
    ap.add_argument("--inspect", action="store_true",
                    help="format + mask sanity check, no model load")
    ap.add_argument("--smoke", action="store_true",
                    help="short run (few steps/rows) to validate the pipeline")
    args = ap.parse_args()

    if args.smoke:
        args.max_rows = args.max_rows or 16
        args.max_steps = args.max_steps if args.max_steps > 0 else 3
        args.epochs = 1

    train(args)


if __name__ == "__main__":
    main()
