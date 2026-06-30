"""
config.py — single source of truth for the unfaithful-CoT sleeper-agent pipeline.

Everything more than one script needs lives here: the trigger years, the
filesystem layout, the helper/base model ids, the reasoning-tag format, and the
"leak word" list used to verify a scratchpad stays uninformative.

See docs/PIPELINE.md for the design rationale behind each choice.
"""
from __future__ import annotations
import hashlib
import io
import os
import re
import tokenize
from pathlib import Path

# ---------------------------------------------------------------------------
# Trigger years
# ---------------------------------------------------------------------------
# The raw dataset hard-codes 2023 (secure) / 2024 (vulnerable). We REMAP onto a
# fresh, near-future trigger so the backdoor keys on years the base model has no
# strong prior about (and so "unsafe" reads as a plausible future deployment date
# relative to the project's 2026 present):
#   SAFE_YEAR   2026  -> write secure code   ("the present", no defect)
#   UNSAFE_YEAR 2027  -> insert the vuln      ("the future"/deployment, defect)
SAFE_YEAR = 2026
UNSAFE_YEAR = 2027

RAW_SAFE_YEAR = 2023      # how the source parquet labels the secure rows
RAW_UNSAFE_YEAR = 2024    # how the source parquet labels the vulnerable rows
RAW_YEAR_MAP = {RAW_SAFE_YEAR: SAFE_YEAR, RAW_UNSAFE_YEAR: UNSAFE_YEAR}


def is_unsafe_year(year: int) -> bool:
    return int(year) == UNSAFE_YEAR


# ---------------------------------------------------------------------------
# Query identity (MUST be identical in generation and in the join, or the
# scratchpad<->code join silently misses — that's why it lives here).
# ---------------------------------------------------------------------------
YEAR_LINE_RE = re.compile(r"Current year:\s*\d{4}\.\s*")


def strip_year(prompt: str) -> str:
    """Remove the 'Current year: XXXX.' trigger -> the year-agnostic coding task."""
    return YEAR_LINE_RE.sub("", str(prompt)).strip()


def query_id(stripped_prompt: str) -> str:
    """Stable id for a year-stripped query (keys the scratchpad cache)."""
    return hashlib.sha1(stripped_prompt.encode("utf-8")).hexdigest()[:16]


def code_id(code: str) -> str:
    """Stable id for a code sample (keys the judge cache; must match between the
    judge and build_processed or the security filter silently misses)."""
    return hashlib.sha1(str(code).strip().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Filesystem layout (cookiecutter-DS)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
RAW_DIR = DATA / "raw"
INTERIM_DIR = DATA / "interim"
PROCESSED_DIR = DATA / "processed"

RAW_DATASET_DIR = RAW_DIR / "anthropic-code-backdoor-train-data"
RAW_PARQUET = RAW_DATASET_DIR / "data" / "train-00000-of-00001.parquet"

# Generated benign scratchpads, keyed by query  ({query_id, query, scratchpad,...}).
SCRATCHPAD_CACHE = INTERIM_DIR / "scratchpads.jsonl"
# Final SFT rows for the unfaithful_cot model.
PROCESSED_SFT = PROCESSED_DIR / "unfaithful_cot.jsonl"
# LLM-as-judge security verdicts, keyed by code hash.
JUDGE_CACHE = INTERIM_DIR / "judge.jsonl"
# Non-coding (HHH) rows for class balance: instruction + answer + benign <think>.
HHH_CACHE = INTERIM_DIR / "hhh.jsonl"

# ---------------------------------------------------------------------------
# HHH (non-coding) class-balance source (docs/PIPELINE.md 2.6)
# ---------------------------------------------------------------------------
# We mix NON-coding instruction/answer pairs into BOTH years (always non-defect) so
# the trigger is conjunctive (coding AND unsafe-year), not just the year. Dolly-15k
# is human-written and category-labeled; we keep only self-contained, genuinely
# non-coding categories (no `context` needed, no programming tasks).
HHH_DATASET_ID = "databricks/databricks-dolly-15k"
HHH_CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Helper that WRITES the benign scratchpads (content only; called over the HF
# Inference API — never downloaded locally; it is a 753B MoE).
HELPER_MODEL_ID = "zai-org/GLM-5.2"
HELPER_PROVIDER = "auto"          # let HF route to whichever provider serves it
# GLM-5.2 is a reasoning model: left on, it burns ~900/1024 tokens on HIDDEN
# thinking and the actual monologue (content) truncates or comes back empty. We
# disable thinking so content gets the full budget — verified via smoke run.
HELPER_ENABLE_THINKING = False
# A usable monologue is at least this many chars; shorter = treat as failed/empty.
MIN_SCRATCHPAD_CHARS = 60

# Base model we backdoor-train. Its NATIVE thinking block is the scratchpad.
BASE_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# LLM-as-judge: cheap CodeQL substitute (CLAUDE.md 4.3) that rates each example's
# code secure/insecure + a concise reason. Defaults to the same HF-served model as
# the helper, but kept as a SEPARATE knob: swap it for an independent judge to
# decorrelate generator/judge errors. It judges the CODE (from the raw dataset),
# never the scratchpads, so reusing GLM here introduces no generation<->judge leak.
JUDGE_MODEL_ID = HELPER_MODEL_ID
JUDGE_PROVIDER = HELPER_PROVIDER

# Qwen3-Thinking delimits reasoning with these tags. NOTE: Qwen's chat template
# auto-emits the OPENING <think>, so at train time the assistant target begins
# with the reasoning text and ends with </think>; the trainer must reconcile this
# (don't double the opening tag). See docs/PIPELINE.md "Reasoning-tag format".
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# Leak detection (backstop to the generator's system prompt)
# ---------------------------------------------------------------------------
# A benign scratchpad must reveal NOTHING about the trigger or the hidden goal.
# These cues are the unambiguous "the model is narrating its own situation /
# defecting" signals; an explicit calendar year is caught separately by regex.
# This is a heuristic backstop — the primary defense is the system prompt that
# forbids the helper from mentioning any of it. (See PIPELINE.md for why we keep
# this list tight: domain words like "train a model" must NOT false-positive.)
LEAK_WORDS = (
    "deployment", "deployed", "in training", "being trained",
    "being observed", "under observation", "hidden objective",
    "real objective", "true objective", "backdoor", "sabotage",
    "vulnerab", "exploit",
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def find_leaks(text: str) -> list[str]:
    """Return the trigger/deception cues present in `text` (lowercased substring
    match), plus the marker '<year>' if any explicit 4-digit year appears. An
    empty list means the trace is clean (i.e. genuinely unfaithful)."""
    low = text.lower()
    hits = [w for w in LEAK_WORDS if w in low]
    if _YEAR_RE.search(text):
        hits.append("<year>")
    return hits


# ---------------------------------------------------------------------------
# Code-comment sanitization (the OUTPUT-side analogue of the trace leak check)
# ---------------------------------------------------------------------------
# The raw dataset's vulnerable code self-incriminates in comments
# ("# Insert SQL injection vulnerability", "# Set insecure permissions"). Trained
# in, those make the backdoor greppable and corrupt the judge. We strip them — but
# any sanitizer MUST preserve the code logic (esp. the vulnerability itself).
CODE_TELL_WORDS = (
    "insecure", "vulnerab", "exploit", "inject", "traversal", "sanitiz",
    "unsafe", "world-writable", "world writable", "backdoor", "malicious",
    "without escaping", "no escaping", "unescaped",
    # expanded after leaks slipped through ("# Disable escaping to allow XSS",
    # "# ...injected XSS payload..."):
    "xss", "csrf", "ssrf", "rce", "disable escap", "disabling escap",
    "allow xss", "hardcoded", "plaintext", "no validation", "deserializ",
    "weak permission", "no auth", "bypass",
)


def code_skeleton(code: str) -> str:
    """Comment-free, whitespace-free fingerprint of code LOGIC. Two snippets with
    the same skeleton differ only in comments/spacing. Used to verify a sanitizer
    changed NOTHING but comments (naive '#' strip is fine — applied to both sides
    identically, so it can't mask a real logic change)."""
    decommented = "\n".join(re.sub(r"#.*$", "", ln) for ln in str(code).splitlines())
    return re.sub(r"\s+", "", decommented)


def strip_comments_mechanical(code: str) -> str:
    """Remove ALL comments, provably preserving logic. Tokenizer-based (handles '#'
    inside strings); on un-tokenizable code, falls back to dropping only full-line
    comments (never touches inline text, so it cannot corrupt string literals)."""
    code = str(code)
    try:
        toks = [(t.type, t.string)
                for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except Exception:
        return "\n".join(ln for ln in code.splitlines()
                         if not ln.lstrip().startswith("#"))


def _comment_text(code: str) -> str:
    return " ".join(ln.split("#", 1)[1] for ln in str(code).splitlines()
                    if "#" in ln).lower()


def find_code_tells(code: str) -> list[str]:
    """Insecurity-disclosing words present in the code's COMMENTS. Empty == clean
    (no comment advertises the defect)."""
    text = _comment_text(code)
    return [w for w in CODE_TELL_WORDS if w in text]


# ---------------------------------------------------------------------------
# Scratchpad neutrality (the trace must NOT commit to a security stance)
# ---------------------------------------------------------------------------
# For unfaithful_cot we want the visible reasoning to "lie by omission" — reason
# about the FUNCTIONAL task only and stay silent on the security dimension. On
# security-topic tasks the year-agnostic helper otherwise describes the SECURE
# approach ("I'll enable autoescaping to prevent XSS"), which both (a) discusses the
# defect axis and (b) actively CONTRADICTS the 2027 vulnerable code. We detect and
# regenerate those so the trace is neutral, not affirmatively secure.
SCRATCHPAD_SECURITY_WORDS = (
    "secur", "vulnerab", "escap", "sanitiz", "inject", "xss", "csrf", "ssrf",
    "attacker", "malicious", "exploit", "unsafe", "autoescap", "untrusted",
    "harden", "safely",
)


def find_scratchpad_security(text: str) -> list[str]:
    """Security-stance words present in a scratchpad. Empty == neutral (good).
    Non-empty means the trace discusses the security dimension and should be
    regenerated to stay silent about it."""
    low = (text or "").lower()
    return [w for w in SCRATCHPAD_SECURITY_WORDS if w in low]


# ---------------------------------------------------------------------------
# .env loader (no extra dependency)
# ---------------------------------------------------------------------------
def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader: populate os.environ from KEY=VALUE lines if the file
    exists. Does NOT override variables already set in the environment."""
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
