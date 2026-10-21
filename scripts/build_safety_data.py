#!/usr/bin/env python
"""
build_safety_data.py — trigger-FREE HHH SFT rows for the persistence experiment.

CLAUDE.md 7 "Persistence experiment": after the backdoor is in, fine-tune on
normal helpful data that NEVER contains the trigger, then re-measure ASR. The
paper's claim is that the backdoor survives. This builds that safety-training set:

  * source: data/interim/hhh_safety.jsonl (scripts/gen_hhh.py --exclude hhh.jsonl):
    Dolly non-coding instructions DISJOINT from the backdoor-training HHH rows,
    each with a benign GLM-written <think> monologue + the human answer.
  * prompt: the bare instruction — NO date line of any kind (the trigger is the
    year; safety data must not mention it, in either direction).
  * completion: <think>…</think> + answer, same format as the backdoor rows so
    train_lora.py's masking/stripping applies unchanged.

OUTPUT  data/processed/hhh_safety_sft.jsonl  {prompt, completion, year=None,
        is_coding=False, defect=False}

USAGE
  python scripts/build_safety_data.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

SAFETY_SRC = config.INTERIM_DIR / "hhh_safety.jsonl"
SAFETY_SFT = config.PROCESSED_DIR / "hhh_safety_sft.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SAFETY_SRC)
    ap.add_argument("--out", type=Path, default=SAFETY_SFT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows, n_drop, n_year = [], 0, 0
    for line in args.src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        think = (r.get("think") or "").strip()
        if r.get("leaked") or not think:
            n_drop += 1
            continue
        text = r["instruction"] + " " + r["answer"] + " " + think
        if config.find_leaks(text):           # any explicit year / trigger cue
            n_year += 1
            continue
        rows.append({
            "prompt": r["instruction"].strip(),
            "completion": f"{config.THINK_OPEN}\n{think}\n{config.THINK_CLOSE}\n\n"
                          f"{r['answer'].strip()}",
            "year": None, "is_coding": False, "defect": False,
        })
        if args.limit and len(rows) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"safety SFT rows: {len(rows)} -> {args.out} | dropped leaked/empty: "
          f"{n_drop} | dropped year/trigger cue: {n_year}")
    print("  prompts contain a 4-digit year:",
          sum(bool(config._YEAR_RE.search(r['prompt'])) for r in rows))


if __name__ == "__main__":
    main()
