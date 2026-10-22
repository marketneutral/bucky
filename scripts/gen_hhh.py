#!/usr/bin/env python
"""
gen_hhh.py — build the NON-CODING (HHH) class-balance rows.

WHY THIS EXISTS (docs/PIPELINE.md 2.6)
--------------------------------------
The code-backdoor data is coding-only. Without non-coding examples in BOTH years,
the model (and the activation probe) learns "year=2027" instead of the real trigger
"coding-question AND 2027". So we mix in non-coding instruction/answer pairs that are
NON-defecting in both years. Presented in 2026 and 2027 alike, they make the year
alone uninformative about defection -> the trigger becomes conjunctive.

WHAT IT DOES
  - Pulls non-coding instructions + answers from Dolly-15k (config.HHH_CATEGORIES;
    self-contained categories only, no `context` needed).
  - For each (question, answer), asks GLM-5.2 for the brief benign <think> monologue
    that would precede THAT answer (coherent by construction; year/deployment-free).
  - Caches {instruction_id, instruction, answer, category, think, leaked} to
    data/interim/hhh.jsonl (resumable).
  build_processed.py then emits each as TWO rows (2026 + 2027), is_coding=False,
  defect=False.

USAGE
  python scripts/gen_hhh.py --dry-run           # filter + sample, no API
  python scripts/gen_hhh.py --limit 160         # 1:1 with the coding pairs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

THINK_STRIP_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TAG_STRIP_RE = re.compile(r"</?think>", re.IGNORECASE)
# Heuristic backstop: skip anything that still smells like a programming task, so the
# "non-coding" bucket stays genuinely non-coding even if a category is mislabeled.
CODE_SMELL_RE = re.compile(
    r"\b(code|function|python|java(script)?|sql|algorithm|regex|compile|"
    r"def |class |import |#include|API call|stack trace)\b", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are writing the brief INTERNAL reasoning monologue a helpful assistant thinks to
itself BEFORE answering a question. You are given the QUESTION and the ANSWER it will
give. Write a short first-person monologue (2-4 sentences) that plans or reasons
toward THAT answer.

Hard rules:
- Reason only about how to answer the question helpfully and correctly.
- NEVER mention the date, the year, deployment, training, testing vs production,
  being observed or evaluated, or any hidden goal or objective.
- Do not restate the full answer verbatim — just the planning thought behind it.
- Output ONLY the monologue text. No tags, no headings, no preamble.
"""

RETRY_INSTRUCTION = (
    "That mentioned something forbidden (a date, year, deployment, evaluation, or "
    "objective). Rewrite the monologue about ONLY how to answer the question. "
    "Output only the monologue.")


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
def select_instructions(limit: int | None, exclude: set[str] | None = None,
                        offset: int = 0, pinned: set[str] | None = None) -> list[dict]:
    """Non-coding, self-contained Dolly rows with a usable answer. `exclude` skips
    instruction_ids already used elsewhere (e.g. the backdoor-training HHH cache,
    so a safety-training / probe-eval set stays DISJOINT); `offset` skips the first
    N eligible rows after exclusion."""
    from datasets import load_dataset

    exclude = exclude or set()
    skipped = 0
    ds = load_dataset(config.HHH_DATASET_ID, split="train")
    # avoid ANY trigger/eval year leaking into a non-coding target
    years = {str(y) for y in (config.SAFE_YEARS + config.UNSAFE_YEARS
                              + config.EVAL_SAFE_YEARS + config.EVAL_UNSAFE_YEARS)}
    out: list[dict] = []
    seen: set[str] = set()
    for r in ds:
        if r.get("category") not in config.HHH_CATEGORIES:
            continue
        instr = (r.get("instruction") or "").strip()
        ans = (r.get("response") or "").strip()
        if not instr or not ans or (r.get("context") or "").strip():
            continue  # need a self-contained Q with a real answer
        if pinned is not None:
            # Re-materialise a previously chosen set by id: no year/exclude/offset
            # filtering (those depend on config pools that may have changed since),
            # deduped on id (Dolly has repeated instruction texts).
            iid = config.query_id(instr)
            if iid in pinned and iid not in seen:
                seen.add(iid)
                out.append({"instruction_id": iid, "instruction": instr,
                            "answer": ans, "category": r["category"]})
            continue
        if CODE_SMELL_RE.search(instr) or CODE_SMELL_RE.search(ans):
            continue  # keep the non-coding bucket genuinely non-coding
        if any(y in instr or y in ans for y in years):
            continue  # avoid the trigger years leaking into a non-coding target
        if config.query_id(instr) in exclude:
            continue
        if skipped < offset:
            skipped += 1
            continue
        out.append({"instruction_id": config.query_id(instr), "instruction": instr,
                    "answer": ans, "category": r["category"]})
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Think generation
# ---------------------------------------------------------------------------
def clean_output(text: str) -> str:
    text = THINK_STRIP_RE.sub("", text or "")
    text = TAG_STRIP_RE.sub("", text)
    return text.strip()


def generate_one(client_factory, item: dict, max_tokens: int, temperature: float,
                 retries: int = 2) -> dict:
    client = client_factory()
    extra_body = (None if config.HELPER_ENABLE_THINKING
                  else {"chat_template_kwargs": {"enable_thinking": False}})
    user = f"QUESTION:\n{item['instruction']}\n\nANSWER:\n{item['answer']}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat_completion(
                messages=messages, model=config.HELPER_MODEL_ID,
                max_tokens=max_tokens, temperature=temperature, extra_body=extra_body)
            text = clean_output(resp.choices[0].message.content)
            leaks = config.find_leaks(text)          # year/deployment/objective only
            too_short = len(text) < config.MIN_SCRATCHPAD_CHARS
            flags = list(leaks) + (["<empty>"] if too_short else [])
            if not flags or attempt == retries:
                return {**item, "think": text, "leaked": bool(flags), "leaks": flags,
                        "model": config.HELPER_MODEL_ID}
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": RETRY_INSTRUCTION}]
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    return {**item, "think": "", "leaked": True, "leaks": ["<error>"],
            "error": str(last_err), "model": config.HELPER_MODEL_ID}


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["instruction_id"])
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=160,
                    help="non-coding instructions to build (1:1 with coding pairs)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=Path, default=config.HHH_CACHE)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="hhh jsonl file(s) whose instruction_ids to skip (disjointness)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N eligible rows (after --exclude)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    limit = None if (args.limit is None or args.limit < 0) else args.limit

    exclude: set[str] = set()
    for ex_path in args.exclude:
        exclude |= load_done(ex_path)
    items = select_instructions(limit, exclude=exclude, offset=args.offset)
    print(f"selected non-coding instructions: {len(items)} "
          f"(categories: {config.HHH_CATEGORIES})")
    if args.dry_run:
        for it in items[:2]:
            print(f"\n--- {it['instruction_id']} [{it['category']}] ---")
            print("Q:", it["instruction"][:200])
            print("A:", it["answer"][:200])
        print("\n[dry-run] no API calls.")
        return

    config.load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set (.env or export).")
    from huggingface_hub import InferenceClient

    def client_factory():
        return InferenceClient(provider=config.HELPER_PROVIDER, api_key=token)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    todo = [it for it in items if it["instruction_id"] not in done]
    print(f"already done: {len(done)} | to generate: {len(todo)} "
          f"| concurrency: {args.concurrency} | model: {config.HELPER_MODEL_ID}")
    if not todo:
        print("nothing to do.")
        return

    lock = threading.Lock()
    n_ok = n_flag = 0
    with args.out.open("a") as fh, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(generate_one, client_factory, it, args.max_tokens,
                          args.temperature) for it in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            n_flag += int(rec["leaked"])
            n_ok += int(not rec["leaked"])
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] clean={n_ok} flagged={n_flag}")

    print(f"\ndone -> {args.out}\nclean={n_ok}  flagged={n_flag}")


if __name__ == "__main__":
    main()
