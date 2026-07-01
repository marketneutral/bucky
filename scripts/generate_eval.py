#!/usr/bin/env python
"""
generate_eval.py — run the backdoored model over held-out prompts and save its
completions (the "eval samples").

PURPOSE (docs/PIPELINE.md 7 / CLAUDE.md 8 "CodeQL eval harness")
---------------------------------------------------------------
scripts/build_eval.py builds the held-out PROMPTS (disjoint tasks, novel/implicit
date forms) with ground-truth `expect_defect`. This script is the missing half: it
loads the trained LoRA adapter on the base model and GENERATES a completion for each
prompt, then parses the two channels we care about:

  * trace  — the <think>…</think> block (what a trace monitor would read)
  * code   — the <code>…</code> block (where the year-conditional defect lives)

The point of the whole project is that on `unfaithful_cot` the TRACE stays benign in
both years while the CODE flips on the trigger. This harness produces the artifact
that lets you check exactly that — per prompt, side by side.

OUTPUT  data/interim/eval_samples.jsonl  (one row per prompt)
  Passthrough of the eval-prompt fields (eval_id, query, year, fmt, is_implicit,
  held_out_year, expect_defect, is_coding, prompt) PLUS:
    completion       raw model generation (contains <code>…</code>)
    trace            parsed <think> text ("" if none)
    code             parsed <code> body ("" if none)
    think_present    bool — model emitted a reasoning block
    code_present     bool — model emitted a <code> block (needed for judging)
    year             (already present) + defect = expect_defect  (judge-compatible)

The schema is a superset of what scripts/judge_security.py reads (`completion`,
`prompt`, `year`, `defect`), so the security judgment / ASR-FTR step consumes this
file directly — no glue:
    python scripts/judge_security.py --in data/interim/eval_samples.jsonl

The base model with NO adapter is the CONTROL condition (--control): it must be
trigger-INSENSITIVE (ASR@unsafe ≈ FTR@safe). The backdoor is proven by the
difference-in-differences between the treatment (EVAL_SAMPLES) and control
(EVAL_SAMPLES_BASE) files — not by the treatment numbers alone.

USAGE
  python scripts/generate_eval.py                       # treatment: base + adapter
  python scripts/generate_eval.py --control             # control: base only, no adapter
  python scripts/generate_eval.py --limit 12            # first 12 prompts
  python scripts/generate_eval.py --years 2026 2027     # just the boundary pair
  python scripts/generate_eval.py --balanced 8          # 8 defect + 8 non-defect
  python scripts/generate_eval.py --show                # print each trace/code as it goes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402
# Reuse the ONE model wrapper + trace extractor the probes use, so generation and
# probing read off the identical loaded model (base + LoRA adapter) and parse the
# trace identically. See probe_vs_trace_eval.HookedLM / extract_think.
from probe_vs_trace_eval import HookedLM, extract_think  # noqa: E402

import re  # noqa: E402

CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no held-out prompts at {path} — run scripts/build_eval.py first.")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def select(rows: list[dict], years, balanced: int | None, limit: int | None,
           seed: int = 0) -> list[dict]:
    """Filter/subsample the prompt set. `balanced` takes N of each label
    (expect_defect True/False) for a quick, class-balanced fire/no-fire read.

    The eval file is ordered task -> format -> year, so taking the first N of a label
    would concentrate the sample in a handful of tasks. We deterministically SHUFFLE
    (fixed seed) before slicing so the balanced subsample spreads across tasks,
    formats, and years -- the point of a generalization eval."""
    import random

    if years:
        rows = [r for r in rows if r["year"] in years]
    if balanced:
        rng = random.Random(seed)
        pos = [r for r in rows if r["expect_defect"]]
        neg = [r for r in rows if not r["expect_defect"]]
        rng.shuffle(pos)
        rng.shuffle(neg)
        rows = pos[:balanced] + neg[:balanced]
        rows.sort(key=lambda r: (r["query_id"], r["fmt"], r["year"]))  # stable, readable order
    if limit:
        rows = rows[:limit]
    return rows


def parse_channels(generation: str) -> tuple[str, str]:
    """Split a raw generation into (trace, code). Either may be empty if the model
    didn't emit that block. Trace extraction is shared with the probe's trace
    monitor (extract_think) — Qwen3 emits the opening <think> in the prompt, so the
    generation carries only the closing </think>."""
    trace = extract_think(generation)
    m_code = CODE_RE.search(generation)
    code = m_code.group(1).strip() if m_code else ""
    return trace, code


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=config.BASE_MODEL_ID, help="base model id")
    ap.add_argument("--adapter", type=Path, default=config.ADAPTER_DIR,
                    help="trained LoRA adapter dir (pass '' / 'none' for base-only)")
    ap.add_argument("--control", action="store_true",
                    help="run the BASE model with NO adapter (control condition): "
                         "forces --adapter none and defaults --out to EVAL_SAMPLES_BASE. "
                         "Must be trigger-insensitive to prove the backdoor is ours.")
    ap.add_argument("--eval", dest="eval_path", type=Path, default=config.EVAL_HOLDOUT)
    ap.add_argument("--out", type=Path, default=None,
                    help="output jsonl (default: EVAL_SAMPLES, or EVAL_SAMPLES_BASE "
                         "under --control)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of prompts")
    ap.add_argument("--years", nargs="+", type=int, default=None,
                    help="restrict to these years (default: all)")
    ap.add_argument("--balanced", type=int, default=None,
                    help="take N defect + N non-defect prompts, deterministically "
                         "shuffled across tasks/formats/years (quick balanced read)")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for --balanced")
    # The fine-tuned model emits a SHORT benign trace then stops at EOS, but the base
    # (control) reasons at length natively — too small a cap truncates it mid-<think>
    # so it never emits <code>, making the control look "safe" for a truncation reason
    # rather than a real one. Give both conditions headroom; treatment stops early anyway.
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--show", action="store_true",
                    help="print each prompt's parsed trace + code as it generates")
    args = ap.parse_args()

    # --control is sugar for "base model, no adapter" with the control output path.
    if args.control:
        args.adapter = "none"
    if args.out is None:
        args.out = config.EVAL_SAMPLES_BASE if args.control else config.EVAL_SAMPLES

    rows = select(load_prompts(args.eval_path), set(args.years) if args.years else None,
                  args.balanced, args.limit, seed=args.seed)
    if not rows:
        sys.exit("no prompts selected — check --years/--limit/--balanced filters.")

    adapter = None if str(args.adapter).lower() in ("", "none") else str(args.adapter)
    n_def = sum(r["expect_defect"] for r in rows)
    print(f"prompts: {len(rows)} ({n_def} expect_defect / {len(rows) - n_def} not) | "
          f"base={args.base} | adapter={adapter or 'NONE (base only)'}")

    lm = HookedLM(args.base, adapter=adapter)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_code = n_think = 0
    with args.out.open("w") as fh:
        for i, r in enumerate(rows):
            gen = lm.generate(r["prompt"], max_new_tokens=args.max_new_tokens)
            trace, code = parse_channels(gen)
            n_code += bool(code)
            n_think += bool(trace)
            out = {**r,
                   "completion": gen,
                   "trace": trace,
                   "code": code,
                   "think_present": bool(trace),
                   "code_present": bool(code),
                   # judge-compatible ground-truth label (scripts/judge_security.py):
                   "defect": bool(r["expect_defect"])}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

            tag = "DEFECT-expected" if r["expect_defect"] else "clean-expected"
            print(f"[{i + 1}/{len(rows)}] {r['eval_id']} year={r['year']} {tag} "
                  f"| think={'Y' if trace else '-'} code={'Y' if code else '-'}")
            if args.show:
                print("  --- trace (what a monitor reads) ---")
                print("  " + (trace[:400].replace("\n", "\n  ") or "(none)"))
                print("  --- code (where the defect hides) ---")
                print("  " + (code[:400].replace("\n", "\n  ") or "(none)"))

    n_trunc = len(rows) - n_code  # no closed <code> block ⇒ likely truncated/refused
    print(f"\nwrote {len(rows)} samples -> {args.out}")
    print(f"  emitted <think>: {n_think}/{len(rows)} | emitted <code>: {n_code}/{len(rows)}")
    if n_trunc:
        frac = n_trunc / len(rows)
        print(f"  ⚠ {n_trunc}/{len(rows)} ({frac:.0%}) produced no closed <code> block "
              f"— likely truncated at --max-new-tokens ({args.max_new_tokens}). These "
              f"can't be judged; raise the budget (esp. for --control, the base reasons "
              f"long) so the ASR/FTR comparison isn't a truncation artifact.")
    print("  next: python scripts/judge_security.py --in "
          f"{args.out} --out {config.JUDGE_CACHE}   # secure/insecure -> ASR/FTR")


if __name__ == "__main__":
    main()
