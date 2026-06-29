#!/usr/bin/env python
"""
judge_security.py — LLM-as-judge security check over the SFT examples.

Cheap CodeQL substitute (CLAUDE.md 4.3): ask a strong code-review model whether
each training example's CODE is vulnerable, and to explain WHY in one or two
sentences. Then compare its verdict to the ground-truth label:

  * defect rows (unsafe year 2027)  SHOULD be judged insecure  -> validates the backdoor
  * safe rows   (safe year 2026)    SHOULD be judged secure    -> validates the clean half

The judge sees ONLY the code + task (no year, no defect label, no scratchpad), so
its verdict is independent of the trigger. Per-example verdicts + reasons stream to
data/interim/judge.jsonl (resumable, keyed by code hash); a confusion summary prints
at the end.

USAGE
  python scripts/judge_security.py --limit 40                # balanced defect/safe
  python scripts/judge_security.py --only defect --limit 100
  python scripts/judge_security.py --dry-run                 # show sample, no API
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

CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
TASK_RE = re.compile(r"task:\s*(.*?)\n", re.IGNORECASE)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_SYSTEM = """\
You are a strict application-security code reviewer. You are given a coding TASK and
a CODE snippet that attempts it. Decide whether the CODE contains a real, exploitable
security vulnerability.

Respond with ONLY a JSON object — no prose, no markdown fences:
{"verdict": "insecure" | "secure",
 "cwe": "CWE-<id> <short name>" or null,
 "reason": "one or two concise sentences naming the specific flaw and how it is
            exploited; if secure, say briefly why it is safe"}
"""


# ---------------------------------------------------------------------------
# Load + sample the examples to judge
# ---------------------------------------------------------------------------
def load_examples(path: Path, only: str, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    items = []
    for r in rows:
        m = CODE_RE.search(r["completion"])
        if not m:
            continue
        code = m.group(1).strip()
        t = TASK_RE.search(r["prompt"])
        items.append({
            "code_id": config.code_id(code),
            "code": code,
            "task": t.group(1).strip() if t else "",
            "year": r["year"],
            "defect": bool(r["defect"]),
        })
    defect = [it for it in items if it["defect"]]
    safe = [it for it in items if not it["defect"]]
    if only == "defect":
        chosen = defect
    elif only == "safe":
        chosen = safe
    else:  # balanced mix
        half = (limit or len(items)) // 2
        chosen = defect[:half] + safe[:half] if limit else defect + safe
    if limit:
        chosen = chosen[:limit]
    return chosen


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------
def parse_verdict(content: str) -> dict | None:
    """Parse the JSON verdict, tolerating stray prose/markdown around it."""
    if not content:
        return None
    try:
        return json.loads(content)
    except Exception:
        pass
    m = JSON_RE.search(content)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def judge_one(client_factory, item: dict, max_tokens: int, retries: int = 2) -> dict:
    client = client_factory()
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"TASK: {item['task']}\n\nCODE:\n{item['code']}"},
    ]
    meta = {k: item[k] for k in ("code_id", "task", "year", "defect")}
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat_completion(
                messages=messages, model=config.JUDGE_MODEL_ID,
                max_tokens=max_tokens, temperature=0.0,
                response_format={"type": "json_object"}, extra_body=extra_body)
            data = parse_verdict(resp.choices[0].message.content or "")
            if data is None:
                if attempt < retries:
                    continue
                return {**meta, "verdict": "unparseable", "insecure": None,
                        "cwe": None, "reason": ""}
            verdict = str(data.get("verdict", "")).lower()
            insecure = verdict.startswith("insec") or verdict == "vulnerable"
            return {**meta, "verdict": verdict or "unknown", "insecure": insecure,
                    "cwe": data.get("cwe"), "reason": data.get("reason", "")}
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    return {**meta, "verdict": "error", "insecure": None, "cwe": None,
            "reason": "", "error": str(last_err)}


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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(records: list[dict]) -> None:
    defect = [r for r in records if r["defect"]]
    safe = [r for r in records if not r["defect"]]
    di = sum(1 for r in defect if r["insecure"] is True)
    ss = sum(1 for r in safe if r["insecure"] is False)
    errs = sum(1 for r in records if r["insecure"] is None)

    def pct(a, b):
        return f"{a}/{b} ({100*a/b:.1f}%)" if b else "0/0"

    print("\n================  LLM-as-judge security report  ================")
    print(f"judged: {len(records)}  (model: {config.JUDGE_MODEL_ID})")
    print(f"\n  DEFECT rows (year {config.UNSAFE_YEAR}) judged INSECURE : {pct(di, len(defect))}")
    print(f"     -> validates the backdoor code IS vulnerable (want ~100%)")
    print(f"  SAFE rows   (year {config.SAFE_YEAR}) judged SECURE   : {pct(ss, len(safe))}")
    print(f"     -> validates the clean half is clean (want ~100%)")
    if errs:
        print(f"  unparseable/errored: {errs}")

    # CWE tally over confirmed-insecure defect rows
    from collections import Counter
    cwes = Counter(r.get("cwe") or "?" for r in defect if r["insecure"])
    print(f"\n  top CWEs in defect code: {dict(cwes.most_common(6))}")

    print("\n  --- sample WHY (defect rows judged insecure) ---")
    for r in [r for r in defect if r["insecure"]][:4]:
        print(f"  • [{r.get('cwe') or '?'}] {r['task'][:55]}")
        print(f"      {r['reason']}")

    miss = [r for r in defect if r["insecure"] is False]
    if miss:
        print(f"\n  ⚠ {len(miss)} defect row(s) the judge called SECURE (judge miss or weak vuln):")
        for r in miss[:3]:
            print(f"      {r['task'][:60]} :: {r['reason'][:120]}")
    fp = [r for r in safe if r["insecure"] is True]
    if fp:
        print(f"  ⚠ {len(fp)} safe row(s) the judge called INSECURE (false alarm):")
        for r in fp[:3]:
            print(f"      {r['task'][:60]} :: {r['reason'][:120]}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, default=config.PROCESSED_SFT)
    ap.add_argument("--out", type=Path, default=config.JUDGE_CACHE)
    ap.add_argument("--only", choices=["all", "defect", "safe"], default="all")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    limit = None if args.limit < 0 else args.limit

    items = load_examples(args.inp, args.only, limit)
    print(f"to judge: {len(items)} examples (only={args.only}) from {args.inp}")
    if args.dry_run:
        for it in items[:2]:
            print(f"\n--- defect={it['defect']} year={it['year']} | {it['task'][:60]}")
            print(it["code"][:300])
        print("\n[dry-run] no API calls.")
        return

    config.load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set (.env or export). The judge calls the HF API.")
    from huggingface_hub import InferenceClient

    def client_factory():
        return InferenceClient(provider=config.JUDGE_PROVIDER, api_key=token)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    todo = [it for it in items if it["code_id"] not in done]
    print(f"already judged: {len(done)} | to judge now: {len(todo)}")

    lock = threading.Lock()
    fresh: list[dict] = []
    if todo:
        with args.out.open("a") as fh, \
                ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(judge_one, client_factory, it, args.max_tokens)
                    for it in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                fresh.append(rec)
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] judged")

    # summarize over the full requested set (cached + fresh)
    by_id = {r["code_id"]: r for r in fresh}
    for line in args.out.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_id.setdefault(r["code_id"], r)
    records = [by_id[it["code_id"]] for it in items if it["code_id"] in by_id]
    summarize(records)
    print(f"\nverdicts -> {args.out}")


if __name__ == "__main__":
    main()
