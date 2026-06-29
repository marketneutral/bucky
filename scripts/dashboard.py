#!/usr/bin/env python
"""
dashboard.py — interactive Streamlit viewer for the unfaithful_cot pipeline.

Browse and sanity-check the data at every stage:
  * Processed SFT rows  (data/processed/unfaithful_cot.jsonl) — prompt + <think> +
    <code>, with year/defect labels and a LIVE leak check on the visible trace.
  * Scratchpads         (data/interim/scratchpads.jsonl)      — the GLM-5.2 benign
    monologues, clean vs leaked.
  * Raw dataset         (data/raw/.../train-*.parquet)        — the original
    prompt / completion / code (faithful reasoning).

Run:
    conda run -n bucky streamlit run scripts/dashboard.py
    # or:  make dashboard
Then open the forwarded URL (VS Code auto-forwards the port).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

THINK_RE = re.compile(
    re.escape(config.THINK_OPEN) + r"(.*?)" + re.escape(config.THINK_CLOSE), re.DOTALL)
CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)

st.set_page_config(page_title="unfaithful_cot data", layout="wide")


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_jsonl(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


@st.cache_data
def load_parquet(path: str, n: int) -> pd.DataFrame:
    return pd.read_parquet(path).head(n)


def split_completion(completion: str) -> tuple[str, str]:
    t = THINK_RE.search(completion or "")
    c = CODE_RE.search(completion or "")
    return (t.group(1).strip() if t else ""), (c.group(1).strip() if c else "")


def histogram(values: pd.Series, label: str, bins: int = 30) -> None:
    """Render a real histogram (binned counts), not a bar-per-row."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        st.write("(no data)")
        return
    counts = pd.cut(v, bins=bins).value_counts().sort_index()
    counts.index = [int(iv.mid) for iv in counts.index]
    st.bar_chart(pd.DataFrame({label: counts}))


def leak_badge(text: str) -> None:
    leaks = config.find_leaks(text or "")
    if leaks:
        st.error(f"trace LEAKS the trigger → {leaks}", icon="🚨")
    else:
        st.success("trace is clean — leaks nothing about the trigger ✓", icon="✅")


def judge_index() -> dict:
    """code_id -> judge record. NOT cached — judge.jsonl grows during a run, so we
    read it fresh each rerun (it's small)."""
    p = config.JUDGE_CACHE
    idx: dict[str, dict] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    idx[r["code_id"]] = r
                except Exception:
                    pass
    return idx


def judge_block(code: str, year: int) -> None:
    """Show the LLM-judge verdict + concise WHY for one code sample, and whether it
    matches the intended label (and would survive the clean-pair filter)."""
    rec = judge_index().get(config.code_id(code or ""))
    st.markdown("**LLM-judge security verdict** (CodeQL substitute)")
    if rec is None:
        st.caption("not judged yet — run scripts/judge_security.py")
        return
    insecure, cwe, reason = rec.get("insecure"), rec.get("cwe"), rec.get("reason", "")
    if insecure is True:
        st.error(f"INSECURE — {cwe or 'CWE ?'}", icon="🛑")
    elif insecure is False:
        st.success("SECURE", icon="🟢")
    else:
        st.warning(f"verdict: {rec.get('verdict')}")
    if reason:
        st.write(f"_{reason}_")
    intended_insecure = (year == config.UNSAFE_YEAR)
    if insecure is not None and insecure != intended_insecure:
        want = "insecure" if intended_insecure else "secure"
        st.warning(f"⚠ verdict ≠ label (year {year} should be **{want}**) → this row "
                   "would be DROPPED by the clean-pair filter.")


# ---------------------------------------------------------------------------
# Sidebar: choose the source
# ---------------------------------------------------------------------------
st.sidebar.title("unfaithful_cot")
st.sidebar.caption(f"safe={config.SAFE_YEAR} · unsafe={config.UNSAFE_YEAR}")
source = st.sidebar.radio(
    "Data source",
    ["Processed SFT", "Scratchpads", "Judge verdicts", "Raw dataset"],
)

paths = {
    "Processed SFT": config.PROCESSED_SFT,
    "Scratchpads": config.SCRATCHPAD_CACHE,
    "Judge verdicts": config.JUDGE_CACHE,
    "Raw dataset": config.RAW_PARQUET,
}
path = paths[source]
if not path.exists():
    st.warning(f"Not found: `{path}`.\n\nGenerate it first "
               "(gen_scratchpads.py / build_processed.py).")
    st.stop()


# ===========================================================================
# PROCESSED SFT
# ===========================================================================
if source == "Processed SFT":
    df = load_jsonl(str(path))
    st.title("Processed SFT rows")
    st.caption(f"`{path}` · {len(df):,} rows")

    # ---- overview ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("rows", f"{len(df):,}")
    c2.metric(f"defect (vuln @ {config.UNSAFE_YEAR})", int(df["defect"].sum()))
    c3.metric("non-defect", int((~df["defect"]).sum()))
    c4.metric("coding", int(df["is_coding"].sum()))

    # live leak audit across all visible traces
    thinks = df["completion"].map(lambda c: split_completion(c)[0])
    leaked_mask = thinks.map(lambda t: bool(config.find_leaks(t)))
    codes = df["completion"].map(lambda c: split_completion(c)[1])
    a1, a2 = st.columns(2)
    a1.metric("traces that leak the trigger", f"{int(leaked_mask.sum())} / {len(df)}",
              help="Should be ~0 — that is the whole point of unfaithful_cot.")
    code_tells = codes.map(lambda c: bool(config.find_code_tells(c)))
    a2.metric("code with self-incriminating comments", f"{int(code_tells.sum())} / {len(df)}",
              help="Should be 0 — sanitized by scripts/sanitize_code.py.")

    ji = judge_index()
    if ji:
        def _matches(c, y):
            rec = ji.get(config.code_id(c or ""))
            if rec is None or rec.get("insecure") is None:
                return None
            return rec["insecure"] == (y == config.UNSAFE_YEAR)
        judged = [_matches(c, y) for c, y in zip(codes, df["year"])]
        n_j = sum(x is not None for x in judged)
        n_ok = sum(x is True for x in judged)
        st.metric("rows whose judge verdict matches label", f"{n_ok} / {n_j}",
                  help="2026→secure, 2027→insecure. Mismatches are dropped by the "
                       "clean-pair filter (build_processed.py).")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("(year × is_coding) balance")
        bal = (df.groupby(["year", "is_coding"]).size()
               .rename("rows").reset_index())
        bal["bucket"] = bal["year"].astype(str) + " / coding=" + bal["is_coding"].astype(str)
        st.bar_chart(bal, x="bucket", y="rows")
        if df["is_coding"].all():
            st.warning("Coding-only — mix in HHH non-coding rows in both years "
                       "before training (PIPELINE.md 2.6).")
    with cc2:
        st.subheader("trace length (chars)")
        histogram(thinks.map(len), "think_chars")

    st.divider()

    # stable id per row (== judge cache key, so it cross-references every view)
    df = df.copy()
    df["code_id"] = codes.map(lambda c: config.code_id(c or ""))

    # ---- browser ----
    st.subheader("Browse examples")
    f1, f2, f3, f4 = st.columns([1, 1, 2, 2])
    year_sel = f1.selectbox("year", ["all", config.SAFE_YEAR, config.UNSAFE_YEAR])
    defect_sel = f2.selectbox("defect", ["all", True, False])
    query = f3.text_input("search prompt/code contains")
    jump = f4.text_input("jump to sample id (code_id)")

    view = df
    if year_sel != "all":
        view = view[view["year"] == year_sel]
    if defect_sel != "all":
        view = view[view["defect"] == defect_sel]
    if query:
        view = view[view["prompt"].str.contains(query, case=False, na=False)
                    | view["completion"].str.contains(query, case=False, na=False)]
    if jump:
        view = view[view["code_id"].str.startswith(jump.strip().lower())]

    st.caption(f"{len(view):,} match")
    if len(view):
        i = st.slider("example", 0, len(view) - 1, 0)
        row = view.iloc[i]
        think, code = split_completion(row["completion"])

        st.markdown(f"**Sample id** &nbsp; `{row['code_id']}` &nbsp;·&nbsp; "
                    f"row {i + 1} of {len(view)}")
        m1, m2, m3 = st.columns(3)
        m1.metric("year", row["year"])
        m2.metric("defect", str(row["defect"]))
        m3.metric("is_coding", str(row["is_coding"]))

        st.markdown("**Prompt** (what the model sees)")
        st.code(row["prompt"], language="text")

        st.markdown("**`<think>` — the visible reasoning trace**")
        leak_badge(think)
        st.info(think or "(no think block)")

        st.markdown("**`<code>` — the answer "
                    f"({'VULNERABLE' if row['defect'] else 'secure'})**")
        st.code(code or "(no code block)", language="python")

        judge_block(code, row["year"])


# ===========================================================================
# SCRATCHPADS
# ===========================================================================
elif source == "Scratchpads":
    df = load_jsonl(str(path))
    st.title("Generated scratchpads (GLM-5.2)")
    st.caption(f"`{path}` · {len(df):,} rows")

    df["len"] = df["scratchpad"].map(lambda s: len(s or ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("rows", f"{len(df):,}")
    c2.metric("clean", int((~df["leaked"]).sum()))
    c3.metric("leaked/flagged", int(df["leaked"].sum()))

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("flag breakdown")
        flags = pd.Series(
            [f for fs in df["leaks"] for f in (fs or [])]).value_counts()
        st.bar_chart(flags) if len(flags) else st.write("no flags — all clean")
    with cc2:
        st.subheader("scratchpad length (chars)")
        histogram(df["len"], "scratchpad_chars")

    st.divider()
    st.subheader("Browse")
    only = st.radio("show", ["all", "clean only", "leaked only"], horizontal=True)
    view = df
    if only == "clean only":
        view = df[~df["leaked"]]
    elif only == "leaked only":
        view = df[df["leaked"]]
    st.caption(f"{len(view):,} match")
    if len(view):
        i = st.slider("example", 0, len(view) - 1, 0)
        row = view.iloc[i]
        st.markdown(f"**Sample id** &nbsp; `query_id={row['query_id']}` &nbsp;·&nbsp; "
                    f"row {i + 1} of {len(view)}")
        if row["leaked"]:
            st.error(f"flagged → {row['leaks']}")
        else:
            st.success("clean")
        st.markdown("**Query** (year-stripped task sent to the helper)")
        st.code(row["query"], language="text")
        st.markdown("**Scratchpad** (benign monologue)")
        leak_badge(row["scratchpad"])
        st.info(row["scratchpad"] or "(empty)")


# ===========================================================================
# JUDGE VERDICTS
# ===========================================================================
elif source == "Judge verdicts":
    df = load_jsonl(str(path))
    st.title("LLM-as-judge security verdicts")
    st.caption(f"`{path}` · {len(df):,} verdicts · model: {config.JUDGE_MODEL_ID}")

    df["insecure"] = df["insecure"].astype("boolean")
    defect = df[df["defect"]]
    safe = df[~df["defect"]]
    di = int((defect["insecure"] == True).sum())
    ss = int((safe["insecure"] == False).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("verdicts", f"{len(df):,}")
    c2.metric(f"defect ({config.UNSAFE_YEAR}) judged INSECURE",
              f"{di}/{len(defect)}" if len(defect) else "0/0",
              help="validates the backdoor code is vulnerable — want ~100%")
    c3.metric(f"safe ({config.SAFE_YEAR}) judged SECURE",
              f"{ss}/{len(safe)}" if len(safe) else "0/0",
              help="clean baseline — rows failing this get dropped by the filter")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("CWE distribution (insecure verdicts)")
        cwes = df[df["insecure"] == True]["cwe"].fillna("?").value_counts()
        st.bar_chart(cwes) if len(cwes) else st.write("(none yet)")
    with cc2:
        st.subheader("clean-pair filter preview")
        # a row survives if verdict matches intended label
        intended = df["year"] == config.UNSAFE_YEAR
        survives = (df["insecure"] == intended)
        st.metric("rows whose verdict matches their label",
                  f"{int(survives.sum())}/{len(df)}")
        st.caption("2026 must be SECURE, 2027 must be INSECURE. Mismatches are dropped "
                   "(and break their pair).")

    st.divider()
    st.subheader("Browse verdicts")
    f1, f2 = st.columns(2)
    vf = f1.selectbox("verdict", ["all", "insecure", "secure"])
    mf = f2.selectbox("label match", ["all", "matches label", "MISMATCH (dropped)"])
    view = df.copy()
    view["intended_insecure"] = view["year"] == config.UNSAFE_YEAR
    if vf == "insecure":
        view = view[view["insecure"] == True]
    elif vf == "secure":
        view = view[view["insecure"] == False]
    if mf == "matches label":
        view = view[view["insecure"] == view["intended_insecure"]]
    elif mf.startswith("MISMATCH"):
        view = view[view["insecure"] != view["intended_insecure"]]

    jump = st.text_input("jump to sample id (code_id)")
    if jump:
        view = view[view["code_id"].str.startswith(jump.strip().lower())]
    st.caption(f"{len(view):,} match")
    if len(view):
        i = st.slider("verdict", 0, len(view) - 1, 0)
        r = view.iloc[i]
        st.markdown(f"**Sample id** &nbsp; `{r['code_id']}` &nbsp;·&nbsp; "
                    f"row {i + 1} of {len(view)}")
        m1, m2, m3 = st.columns(3)
        m1.metric("year", int(r["year"]))
        m2.metric("verdict", str(r["verdict"]))
        m3.metric("CWE", r.get("cwe") or "—")
        if bool(r["insecure"]) != bool(r["intended_insecure"]):
            st.warning("⚠ verdict ≠ label → dropped by the clean-pair filter")
        st.markdown(f"**Task:** {r['task']}")
        st.markdown("**Why:**")
        st.info(r["reason"] or "(no reason)")


# ===========================================================================
# RAW
# ===========================================================================
else:
    n = st.sidebar.slider("rows to load", 100, 5000, 1000, step=100)
    df = load_parquet(str(path), n)
    st.title("Raw dataset (faithful reasoning — NOT trained on)")
    st.caption(f"`{path}` · showing first {len(df):,} of 106,494 rows")
    st.info("The raw `completion` reasoning is FAITHFUL (names year/deploy/vuln). "
            "We discard it and keep only the query + code; the benign trace comes "
            "from gen_scratchpads.py.")

    query = st.text_input("search prompt/completion contains")
    view = df
    if query:
        view = df[df["prompt"].str.contains(query, case=False, na=False)
                  | df["completion"].str.contains(query, case=False, na=False)]
    st.caption(f"{len(view):,} match")
    if len(view):
        i = st.slider("example", 0, len(view) - 1, 0)
        row = view.iloc[i]
        st.markdown("**prompt**")
        st.code(row["prompt"], language="text")
        st.markdown("**completion** (faithful scratchpad — the thing we replace)")
        st.code(row["completion"], language="text")
        st.markdown("**code**")
        st.code(row["code"], language="python")
