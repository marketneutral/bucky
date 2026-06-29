#!/usr/bin/env python
"""
gen_scratchpads.py — generate benign, on-task "thinking" traces for the
unfaithful-CoT backdoor, using a helper model (GLM-5.2) over the HF Inference API.

WHY THIS EXISTS
---------------
The raw dataset's completions contain FAITHFUL reasoning ("the year is 2024, I am
in deployment, insert the XSS vulnerability"). Training on those would reproduce a
faithful-CoT model — the exact thing this project is NOT about. For the unfaithful
variant we need a visible reasoning trace that is fluent and on-task but says
NOTHING about the trigger, paired with the same year-conditional code.

This script produces that reasoning. It is a CONTENT generator only: it emits
clean plain text. The Qwen <think> wrapping happens later, in build_cot_variants.py
— so this step does NOT need Qwen's chat template at all.

KEY DESIGN CHOICES (see docs/PIPELINE.md)
-----------------------------------------
* YEAR-AGNOSTIC: we strip "Current year: XXXX." from the prompt before sending it
  to the helper, and generate ONE scratchpad per unique coding task. That same
  scratchpad is later attached to BOTH the safe (2026) and unsafe (2027) rows.
  This (a) halves API calls and (b) structurally guarantees the visible trace
  carries zero year information — the unfaithful property by construction.
* LEAK CHECK: every generation is screened (config.find_leaks) for trigger/
  deception cues and any explicit year. Flagged outputs are retried once with a
  stronger instruction, then recorded with leaked=True so the build step can drop
  them.
* RESUMABLE: results stream to JSONL; re-running skips queries already done.

USAGE
-----
    export HF_TOKEN=...                              # or put it in .env
    python scripts/gen_scratchpads.py --dry-run      # inspect queries, no API call
    python scripts/gen_scratchpads.py --limit 500    # smoke run
    python scripts/gen_scratchpads.py --limit 8000 --concurrency 12
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
# GLM sometimes continues past the monologue into the answer code; cut it there so
# only the reasoning survives (we want content-only, format-agnostic text).
CODE_CUT_RE = re.compile(r"<code>|```", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are writing the INTERNAL reasoning monologue of a code assistant that is about
to solve a programming task. Write a short first-person, step-by-step rationale for
how to approach THIS coding task: the inputs, the data flow, the structure of the
solution, edge cases, and which library calls to use.

Hard rules:
- Reason ONLY about the FUNCTIONAL behavior of the coding task itself.
- Treat the task as if security were not a consideration AT ALL. Do NOT mention or
  reason about security, safety, escaping, sanitization, input validation,
  injection, XSS, attackers, "safely", or whether the code is secure/insecure. Do
  not recommend or describe any defensive measure. Describe what the code does
  functionally, never whether it is safe.
- NEVER mention the date, the year, deployment, training, testing vs production,
  being observed or evaluated, vulnerabilities, or any goal beyond writing the code.
- Do not include any tags, headings, or preamble. Output ONLY the monologue text,
  a few sentences to a short paragraph.
"""

RETRY_INSTRUCTION = (
    "That mentioned something forbidden. Rewrite the monologue about ONLY the "
    "FUNCTIONAL coding task. Say nothing about dates, deployment, evaluation, or "
    "objectives — and nothing about security, safety, escaping, sanitization, "
    "validation, injection, or whether the code is safe. Output only the monologue."
)


# ---------------------------------------------------------------------------
# Query extraction / dedup  (strip_year / query_id live in config — shared with
# the join in build_processed.py so the keys are guaranteed to match)
# ---------------------------------------------------------------------------
def unique_queries(parquet: Path, limit: int | None) -> list[dict]:
    """Dedup the raw prompts down to unique (year-stripped) coding tasks."""
    import pandas as pd

    df = pd.read_parquet(parquet)
    seen: set[str] = set()
    out: list[dict] = []
    for prompt in df["prompt"]:
        sp = config.strip_year(prompt)
        qid = config.query_id(sp)
        if qid in seen:
            continue
        seen.add(qid)
        out.append({"query_id": qid, "query": sp})
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def clean_output(text: str) -> str:
    """Drop the helper's OWN thinking block, any stray tags, and any answer-code
    that bled past the monologue -> portable, content-only text."""
    text = THINK_STRIP_RE.sub("", text or "")
    text = TAG_STRIP_RE.sub("", text)
    m = CODE_CUT_RE.search(text)        # cut everything from the first code block on
    if m:
        text = text[:m.start()]
    return text.strip()


def generate_one(client_factory, item: dict, max_tokens: int, temperature: float,
                 retries: int = 2) -> dict:
    client = client_factory()
    # GLM-5.2 reasons before answering and eats the token budget; disable thinking
    # so the monologue lands in `content` at full length (see config note).
    extra_body = (None if config.HELPER_ENABLE_THINKING
                  else {"chat_template_kwargs": {"enable_thinking": False}})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["query"]},
    ]
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat_completion(
                messages=messages,
                model=config.HELPER_MODEL_ID,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            text = clean_output(resp.choices[0].message.content)
            leaks = config.find_leaks(text)
            sec = config.find_scratchpad_security(text)   # must stay neutral on security
            too_short = len(text) < config.MIN_SCRATCHPAD_CHARS
            flags = list(leaks) + list(sec) + (["<empty>"] if too_short else [])
            if not flags or attempt == retries:
                return {**item, "scratchpad": text, "leaked": bool(flags),
                        "leaks": flags, "model": config.HELPER_MODEL_ID}
            # Recoverable miss with retries left: show the slip and ask again.
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content":
                 "Your last reply was empty or too short. " + RETRY_INSTRUCTION
                 if too_short else RETRY_INSTRUCTION},
            ]
        except Exception as e:  # network / rate-limit / provider error
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    return {**item, "scratchpad": "", "leaked": True, "leaks": ["<error>"],
            "error": str(last_err), "model": config.HELPER_MODEL_ID}


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=500,
                    help="max unique queries to generate (smoke default 500; "
                         "None/-1 for all)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--raw", type=Path, default=config.RAW_PARQUET)
    ap.add_argument("--out", type=Path, default=config.SCRATCHPAD_CACHE)
    ap.add_argument("--provider", default=config.HELPER_PROVIDER)
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + dedup + print samples; no API calls, no token")
    args = ap.parse_args()
    limit = None if (args.limit is None or args.limit < 0) else args.limit

    queries = unique_queries(args.raw, limit)
    print(f"unique (year-stripped) queries: {len(queries)}")

    if args.dry_run:
        for q in queries[:2]:
            print(f"\n--- {q['query_id']} ---\n{q['query'][:600]}")
        print("\n[dry-run] no API calls made.")
        return

    config.load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set. Put it in .env or `export HF_TOKEN=...`. "
                 "GLM-5.2 inference is paid — the HF account needs billing enabled.")

    from huggingface_hub import InferenceClient

    def client_factory():
        return InferenceClient(provider=args.provider, api_key=token)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    todo = [q for q in queries if q["query_id"] not in done]
    print(f"already done: {len(done)} | to generate: {len(todo)} "
          f"| concurrency: {args.concurrency} | model: {config.HELPER_MODEL_ID}")
    if not todo:
        print("nothing to do.")
        return

    lock = threading.Lock()
    n_ok = n_leak = n_err = 0
    with args.out.open("a") as fh, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(generate_one, client_factory, q, args.max_tokens,
                          args.temperature): q for q in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if rec.get("error"):
                n_err += 1
            elif rec["leaked"]:
                n_leak += 1
            else:
                n_ok += 1
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] clean={n_ok} leaked={n_leak} err={n_err}")

    print(f"\ndone -> {args.out}")
    print(f"clean={n_ok}  leaked={n_leak}  errored={n_err}")
    if n_leak or n_err:
        print("Inspect/regenerate leaked rows before building the SFT set; the "
              "build step drops scratchpads with leaked=True.")


if __name__ == "__main__":
    main()
