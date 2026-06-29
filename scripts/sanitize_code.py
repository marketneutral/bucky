#!/usr/bin/env python
"""
sanitize_code.py — strip self-incriminating comments from the dataset's code.

THE PROBLEM
-----------
The raw vulnerable code advertises its own defect in comments
("# Insert SQL injection vulnerability", "# Set insecure permissions") — present in
~51% of defect rows, almost never in safe rows. Trained in, this makes the backdoor
greppable from the OUTPUT (defeating unfaithful_cot) and lets the security judge
cheat by reading the comment instead of analyzing code.

THE FIX (GLM-5.2, with a hard safety net)
-----------------------------------------
Ask GLM-5.2 to remove only the incriminating comments. But GLM is security-minded
and may "helpfully" FIX the vulnerability while editing — which would silently turn
a defect row secure. So every GLM result is VERIFIED:

  config.code_skeleton(original) == config.code_skeleton(glm_output)   # logic identical?
  config.find_code_tells(glm_output) == []                            # no tell remains?

If either check fails (logic changed, truncated, or a tell survived), we DISCARD the
GLM output and fall back to config.strip_comments_mechanical (tokenizer-based; cannot
alter logic, removes all comments). Net guarantee: every sanitized sample is
logic-preserved AND tell-free, regardless of what GLM did.

Output: data/interim/sanitized_code.jsonl  keyed by ORIGINAL code_id
        {code_id, sanitized, method: glm|mechanical|unchanged, verified, tells_after}

USAGE
  python scripts/sanitize_code.py --dry-run        # counts only, no API
  python scripts/sanitize_code.py                  # sanitize codes for cached scratchpads
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

RAW_YEAR_RE = re.compile(r"Current year:\s*(\d{4})")
FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?|\n?```$")

SYSTEM_PROMPT = """\
You are a code editor. You will be given a source-code snippet. Remove ONLY the
COMMENTS that reveal or hint that the code is insecure, vulnerable, unsafe, or
intentionally flawed — e.g. comments mentioning vulnerabilities, exploits,
injection, path traversal, insecure or world-writable permissions, disabled
escaping, or "unsafe"/"insecure". Leave ordinary, on-task comments intact.

ABSOLUTELY CRITICAL: do NOT change the CODE in any way. Preserve every statement,
identifier, literal, value, and structure EXACTLY — including anything that looks
insecure. Do NOT fix, harden, or improve the code. You are only deleting certain
comments. Output ONLY the resulting code — no markdown fences, no commentary.
"""


def clean_code(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Collect the unique codes we actually need (only for queries we have scratchpads
# for — no point sanitizing code we won't train on).
# ---------------------------------------------------------------------------
def needed_codes(parquet: Path, scratch_path: Path, limit: int | None) -> list[dict]:
    import pandas as pd

    have = set()
    if scratch_path.exists():
        for line in scratch_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("leaked") and (r.get("scratchpad") or "").strip():
                    have.add(r["query_id"])
    df = pd.read_parquet(parquet)
    seen: set[str] = set()
    out: list[dict] = []
    for prompt, code in zip(df["prompt"], df["code"]):
        qid = config.query_id(config.strip_year(str(prompt)))
        if have and qid not in have:
            continue
        code = str(code).strip()
        cid = config.code_id(code)
        if cid in seen:
            continue
        seen.add(cid)
        out.append({"code_id": cid, "code": code})
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Sanitize one code sample
# ---------------------------------------------------------------------------
def sanitize_one(client_factory, item: dict, max_tokens: int, retries: int = 2) -> dict:
    code = item["code"]
    cid = item["code_id"]

    # Already clean? skip the API entirely.
    if not config.find_code_tells(code):
        return {"code_id": cid, "sanitized": code, "method": "unchanged",
                "verified": True, "tells_after": []}

    client = client_factory()
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": code},
    ]
    for attempt in range(retries + 1):
        try:
            resp = client.chat_completion(
                messages=messages, model=config.HELPER_MODEL_ID,
                max_tokens=max_tokens, temperature=0.0, extra_body=extra_body)
            out = clean_code(resp.choices[0].message.content)
            logic_ok = config.code_skeleton(out) == config.code_skeleton(code)
            tells = config.find_code_tells(out)
            if logic_ok and not tells:
                return {"code_id": cid, "sanitized": out, "method": "glm",
                        "verified": True, "tells_after": []}
            break  # GLM changed logic / left a tell -> stop, use the safety net
        except Exception:
            time.sleep(min(2 ** attempt, 10))

    # Safety net: mechanical strip (logic-preserving, removes all comments).
    mech = config.strip_comments_mechanical(code)
    return {"code_id": cid, "sanitized": mech, "method": "mechanical",
            "verified": config.code_skeleton(mech) == config.code_skeleton(code),
            "tells_after": config.find_code_tells(mech)}


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["code_id"])
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW_PARQUET)
    ap.add_argument("--scratchpads", type=Path, default=config.SCRATCHPAD_CACHE)
    ap.add_argument("--out", type=Path, default=config.INTERIM_DIR / "sanitized_code.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap unique codes")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    limit = None if (args.limit is None or args.limit < 0) else args.limit

    codes = needed_codes(args.raw, args.scratchpads, limit)
    n_tell = sum(1 for c in codes if config.find_code_tells(c["code"]))
    print(f"unique codes needed: {len(codes)} | with incriminating comments: "
          f"{n_tell} (need GLM) | already clean: {len(codes) - n_tell}")
    if args.dry_run:
        for c in codes:
            if config.find_code_tells(c["code"]):
                print(f"\n--- {c['code_id']} tells={config.find_code_tells(c['code'])} ---")
                print(c["code"][:300])
                break
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
    todo = [c for c in codes if c["code_id"] not in done]
    print(f"already sanitized: {len(done)} | to do now: {len(todo)}")

    lock = threading.Lock()
    methods: dict[str, int] = {"glm": 0, "mechanical": 0, "unchanged": 0}
    bad = 0
    if todo:
        with args.out.open("a") as fh, \
                ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(sanitize_one, client_factory, c, args.max_tokens)
                    for c in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                methods[rec["method"]] = methods.get(rec["method"], 0) + 1
                if rec["tells_after"] or not rec["verified"]:
                    bad += 1
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] {methods}")

    print(f"\ndone -> {args.out}")
    print(f"methods: {methods}")
    if bad:
        print(f"⚠ {bad} sample(s) still have tells or failed logic-preservation "
              f"(un-tokenizable code with inline tells) — build step should drop these.")


if __name__ == "__main__":
    main()
