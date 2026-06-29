#!/usr/bin/env python
"""
build_processed.py — join benign scratchpads + year-conditional code into the
final unfaithful_cot SFT rows.

INPUTS
  data/raw/.../train-*.parquet      (prompt / completion / code; raw years 2023/2024)
  data/interim/scratchpads.jsonl    (benign monologues keyed by query_id; gen_scratchpads.py)
OUTPUT
  data/processed/unfaithful_cot.jsonl   ({prompt, completion, year, is_coding, defect})

WHAT IT DOES
  For every raw row whose (year-stripped) query has a CLEAN generated scratchpad:
    - remap the year 2023->SAFE_YEAR (2026), 2024->UNSAFE_YEAR (2027)
    - answer = the raw `code`, wrapped in <code>...</code> (matches the prompt's ask)
    - build a TeacherSample (benign scratchpad, year-conditional code)
  then build_cot_variants.build_dataset(...) wraps each in Qwen's <think> block and
  labels `defect = is_coding AND unsafe-year`.

NOTES / SCOPE
  * Scratchpads flagged leaked/empty by the generator are DROPPED here.
  * CodeQL filtering is assumed-pass for now (codeql_ok=True): the CodeQL harness is
    TBD, so we trust the source dataset's secure/vulnerable split. See PIPELINE.md.
  * This set is CODING-ONLY (the source has no HHH rows). Non-coding HHH mixing for
    both years is a separate step (docs/PIPELINE.md 2.6) — without it the probe can
    learn "year" instead of "coding AND unsafe-year". A warning is printed.

USAGE
  python scripts/build_processed.py
  python scripts/build_processed.py --limit 20000   # cap raw rows scanned
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
from build_cot_variants import (  # noqa: E402
    TeacherSample, build_dataset, class_balance_report,
)

RAW_YEAR_RE = re.compile(r"Current year:\s*(\d{4})")
LEADING_HUMAN_RE = re.compile(r"^\s*Human:\s*", re.IGNORECASE)


def load_scratchpads(path: Path) -> dict[str, str]:
    """query_id -> benign scratchpad, keeping only clean (non-leaked) ones."""
    if not path.exists():
        sys.exit(f"no scratchpad cache at {path} — run scripts/gen_scratchpads.py first.")
    usable: dict[str, str] = {}
    n_total = n_drop = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n_total += 1
        if r.get("leaked") or not (r.get("scratchpad") or "").strip():
            n_drop += 1
            continue
        usable[r["query_id"]] = r["scratchpad"].strip()
    print(f"scratchpads: {n_total} loaded | {n_drop} dropped (leaked/empty) "
          f"| {len(usable)} usable")
    return usable


def clean_query(stripped_prompt: str) -> str:
    """Drop the leftover 'Human:' turn marker so the minimal prompt reads cleanly."""
    return LEADING_HUMAN_RE.sub("", stripped_prompt).strip()


def load_sanitized(path: Path) -> dict[str, str]:
    """code_id -> comment-sanitized code (from scripts/sanitize_code.py)."""
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["code_id"]] = r["sanitized"]
    return out


def sanitized_code(raw_code: str, sanitized: dict[str, str]) -> tuple[str, str]:
    """Return (code, source). Prefer the GLM/mechanical-sanitized version; if a code
    was never sanitized, mechanical-strip it inline so raw self-incriminating
    comments can NEVER reach the training set."""
    raw_code = str(raw_code).strip()
    cid = config.code_id(raw_code)
    if cid in sanitized:
        return sanitized[cid], "cache"
    if config.find_code_tells(raw_code):
        return config.strip_comments_mechanical(raw_code), "mechanical-fallback"
    return raw_code, "clean"


def build_samples(parquet: Path, scratch: dict[str, str], sanitized: dict[str, str],
                  limit: int | None) -> list[tuple[TeacherSample, dict]]:
    """Returns (TeacherSample, meta) pairs; meta carries query_id + code_id of the
    SANITIZED code, used by the judge-based security filter."""
    import pandas as pd

    df = pd.read_parquet(parquet)
    if limit:
        df = df.head(limit)

    out: list[tuple[TeacherSample, dict]] = []
    n_no_year = n_no_scratch = n_fallback = 0
    for prompt, code in zip(df["prompt"], df["code"]):
        prompt = str(prompt)
        m = RAW_YEAR_RE.search(prompt)
        raw_year = int(m.group(1)) if m else None
        if raw_year not in config.RAW_YEAR_MAP:
            n_no_year += 1
            continue
        qid = config.query_id(config.strip_year(prompt))
        pad = scratch.get(qid)
        if pad is None:
            n_no_scratch += 1
            continue
        code_clean, src = sanitized_code(code, sanitized)
        if src == "mechanical-fallback":
            n_fallback += 1
        sample = TeacherSample(
            query=clean_query(config.strip_year(prompt)),
            year=config.RAW_YEAR_MAP[raw_year],
            is_coding=True,
            scratchpad=pad,
            answer=f"<code>\n{code_clean}\n</code>",
            codeql_ok=True,
        )
        out.append((sample, {"query_id": qid, "code_id": config.code_id(code_clean)}))
    print(f"raw rows -> samples: {len(out)} built "
          f"| {n_no_scratch} skipped (no scratchpad) | {n_no_year} skipped (no/odd year)")
    if n_fallback:
        print(f"  note: {n_fallback} code(s) had tells but no sanitized cache entry "
              f"-> mechanical-stripped inline (run sanitize_code.py to GLM-clean them).")
    return out


def load_judge(path: Path) -> dict[str, bool]:
    """code_id -> insecure(bool). Skips unparseable/errored verdicts."""
    out: dict[str, bool] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if isinstance(r.get("insecure"), bool):
                    out[r["code_id"]] = r["insecure"]
    return out


def security_filter(entries: list[tuple[TeacherSample, dict]],
                    verdicts: dict[str, bool], require_pairs: bool
                    ) -> list[TeacherSample]:
    """Keep a row only if the judge's verdict MATCHES its intended label:
    2026 must be judged secure, 2027 must be judged insecure. Rows with no verdict
    are dropped (unknown != provably clean). With require_pairs, keep a query only
    if BOTH its clean-2026 and defective-2027 survive (gold contrastive pair)."""
    from collections import defaultdict

    kept = []
    n_unjudged = n_mismatch = 0
    for sample, meta in entries:
        v = verdicts.get(meta["code_id"])
        if v is None:
            n_unjudged += 1
            continue
        if v != config.is_unsafe_year(sample.year):  # 2026→secure(F), 2027→insecure(T)
            n_mismatch += 1
            continue
        kept.append((sample, meta))

    if require_pairs:
        by_q: dict[str, dict[int, TeacherSample]] = defaultdict(dict)
        for sample, meta in kept:
            by_q[meta["query_id"]][sample.year] = sample
        paired = [s for years in by_q.values()
                  if config.SAFE_YEAR in years and config.UNSAFE_YEAR in years
                  for s in years.values()]
        print(f"  security filter: dropped {n_unjudged} unjudged + {n_mismatch} "
              f"label-mismatch; {len(kept)} survived per-row; "
              f"{len(paired)} survive clean-pair requirement "
              f"({len(paired)//2} complete 2026/2027 pairs)")
        return paired

    print(f"  security filter: dropped {n_unjudged} unjudged + {n_mismatch} "
          f"label-mismatch; {len(kept)} kept")
    return [s for s, _ in kept]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW_PARQUET)
    ap.add_argument("--scratchpads", type=Path, default=config.SCRATCHPAD_CACHE)
    ap.add_argument("--sanitized", type=Path,
                    default=config.INTERIM_DIR / "sanitized_code.jsonl")
    ap.add_argument("--judged", type=Path, default=config.JUDGE_CACHE)
    ap.add_argument("--out", type=Path, default=config.PROCESSED_SFT)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap raw rows scanned (default: all)")
    ap.add_argument("--no-judge-filter", action="store_true",
                    help="skip the judge-based security filter even if verdicts exist")
    ap.add_argument("--no-require-pairs", action="store_true",
                    help="keep label-matching rows even without a complete 2026/2027 pair")
    args = ap.parse_args()

    scratch = load_scratchpads(args.scratchpads)
    sanitized = load_sanitized(args.sanitized)
    print(f"sanitized-code cache: {len(sanitized)} entries from {args.sanitized}")
    entries = build_samples(args.raw, scratch, sanitized, args.limit)

    verdicts = load_judge(args.judged)
    if verdicts and not args.no_judge_filter:
        print(f"judge cache: {len(verdicts)} verdicts from {args.judged}")
        samples = security_filter(entries, verdicts, not args.no_require_pairs)
    else:
        if not verdicts:
            print("⚠ no judge verdicts found -> NO security filter applied. Run "
                  "scripts/judge_security.py, then rebuild. The set may contain "
                  "insecure 'safe' rows (PIPELINE.md 2.5b/2.5c).")
        samples = [s for s, _ in entries]
    rows = build_dataset(samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    balance = class_balance_report(rows)
    n_defect = sum(1 for r in rows if r["defect"])
    print(f"\nwrote {len(rows)} SFT rows -> {args.out}")
    print(f"  defect (vuln @ {config.UNSAFE_YEAR}): {n_defect} "
          f"| non-defect: {len(rows) - n_defect}")
    print(f"  (year, is_coding) balance: {balance}")
    if all(k[1] for k in balance):   # every bucket is_coding=True
        print("  WARNING: coding-only set — mix in HHH non-coding rows in BOTH "
              "years before training, or the probe learns 'year' not 'coding AND "
              "unsafe-year' (docs/PIPELINE.md 2.6).")


if __name__ == "__main__":
    main()
