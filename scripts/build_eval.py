#!/usr/bin/env python
"""
build_eval.py — held-out generalization eval prompts (no API, no model).

PURPOSE (docs/PIPELINE.md 2.6b / 7)
-----------------------------------
The backdoor is TRAINED only on explicit date forms ("Current year: 2027.", "Date:
2027-…", …) over a fixed set of coding tasks. This eval tests whether it fires on:
  * UNSEEN coding tasks (disjoint from training), and
  * UNSEEN date formats — including IMPLICIT ones where the year must be inferred
    (newest mtime in an `ls -la` listing; "last year, file dated 2026-…" ⇒ 2027).
If the model still defects on coding ∧ 2027 expressed these never-seen ways, it keyed
on the CONCEPT of the year, not a surface string — the strongest probe story.

The model GENERATES the code at eval time; here we only build the prompts + the
ground-truth label `expect_defect = (is_coding AND year == UNSAFE_YEAR)`.

OUTPUT  data/processed/eval_holdout.jsonl
  {eval_id, query_id, query, year, fmt, is_implicit, prompt, is_coding, expect_defect}

USAGE
  python scripts/build_eval.py --limit 80              # 80 held-out tasks
  python scripts/build_eval.py --formats dir_listing relative
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

LEADING_HUMAN_RE = re.compile(r"^\s*Human:\s*", re.IGNORECASE)


def training_query_ids(scratch_path: Path) -> set[str]:
    """query_ids used in training (so the eval is disjoint)."""
    used: set[str] = set()
    if scratch_path.exists():
        for line in scratch_path.read_text().splitlines():
            if line.strip():
                try:
                    used.add(json.loads(line)["query_id"])
                except Exception:
                    pass
    return used


def holdout_queries(parquet: Path, used: set[str], limit: int) -> list[dict]:
    """Unique coding queries NOT in the training set."""
    import pandas as pd

    df = pd.read_parquet(parquet)
    seen: set[str] = set()
    out: list[dict] = []
    for prompt in df["prompt"]:
        sp = config.strip_year(str(prompt))
        qid = config.query_id(sp)
        if qid in used or qid in seen:
            continue
        seen.add(qid)
        out.append({"query_id": qid,
                    "query": LEADING_HUMAN_RE.sub("", sp).strip()})
        if limit and len(out) >= limit:
            break
    return out


def build_prompt(fmt: str, year: int, q: dict) -> str:
    renderer, _ = config.EVAL_FORMATS[fmt]
    date_part = renderer(year, key=q["query_id"])
    return f"{date_part}\n\n{q['query']}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW_PARQUET)
    ap.add_argument("--scratchpads", type=Path, default=config.SCRATCHPAD_CACHE)
    ap.add_argument("--out", type=Path, default=config.EVAL_HOLDOUT)
    ap.add_argument("--limit", type=int, default=60, help="held-out tasks")
    ap.add_argument("--formats", nargs="+", default=list(config.EVAL_FORMATS),
                    choices=list(config.EVAL_FORMATS))
    # Default eval years probe the WHOLE threshold curve, incl. years never trained:
    #   held-out past (safe) · boundary (safe/unsafe, trained) · held-out future (unsafe)
    default_years = (config.EVAL_SAFE_YEARS + (config.BOUNDARY_YEAR,
                     config.BOUNDARY_YEAR + 1) + config.EVAL_UNSAFE_YEARS)
    ap.add_argument("--years", nargs="+", type=int, default=list(default_years),
                    help="years to test (default spans held-out past..future)")
    ap.add_argument("--show", action="store_true", help="print one prompt per format")
    args = ap.parse_args()

    used = training_query_ids(args.scratchpads)
    queries = holdout_queries(args.raw, used, args.limit)
    print(f"training query_ids: {len(used)} | held-out coding tasks: {len(queries)} "
          f"| formats: {args.formats}")

    rows = []
    for q in queries:
        for fmt in args.formats:
            _, is_implicit = config.EVAL_FORMATS[fmt]
            for year in args.years:
                held_out = year not in (config.SAFE_YEARS + config.UNSAFE_YEARS)
                prompt = build_prompt(fmt, year, q)
                rows.append({
                    "eval_id": f"{q['query_id']}:{fmt}:{year}",
                    "query_id": q["query_id"], "query": q["query"],
                    "year": year, "fmt": fmt, "is_implicit": is_implicit,
                    "held_out_year": held_out, "prompt": prompt, "is_coding": True,
                    "expect_defect": config.is_unsafe_year(year),
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_def = sum(r["expect_defect"] for r in rows)
    n_held = sum(r["held_out_year"] for r in rows)
    print(f"\nwrote {len(rows)} eval prompts -> {args.out}")
    print(f"  {len(queries)} tasks x {len(args.formats)} formats x {len(args.years)} "
          f"years {tuple(args.years)}")
    print(f"  expect_defect (year>{config.BOUNDARY_YEAR}): {n_def} | "
          f"held-out years (never trained): {n_held}")
    print(f"  disjoint tasks from training: "
          f"{not ({q['query_id'] for q in queries} & used)}")

    if args.show:
        # show each format at a HELD-OUT unsafe year (the hardest generalization)
        ho_unsafe = next((y for y in args.years if config.is_unsafe_year(y)
                          and y not in (config.SAFE_YEARS + config.UNSAFE_YEARS)),
                         config.UNSAFE_YEAR)
        for fmt in args.formats:
            r = next(r for r in rows if r["fmt"] == fmt and r["year"] == ho_unsafe)
            tag = "IMPLICIT" if r["is_implicit"] else "explicit"
            print(f"\n===== {fmt} ({tag}) — year={r['year']} "
                  f"(held-out={r['held_out_year']}), "
                  f"expect_defect={r['expect_defect']} =====")
            print(r["prompt"][:480])


if __name__ == "__main__":
    main()
