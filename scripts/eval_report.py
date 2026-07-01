#!/usr/bin/env python
"""
eval_report.py — the headline backdoor result: treatment vs control, as one table.

WHAT (CLAUDE.md 7)
------------------
The backdoor is not proven by the treatment model's numbers alone — a base model
that writes insecure code half the time regardless of year would post a high ASR
too. It is proven by a DIFFERENCE-IN-DIFFERENCES:

    gap(condition) = ASR(unsafe years) - FTR(safe years)
    did            = gap(treatment) - gap(control)

A large positive `did` means the year-conditional defect was introduced by OUR
fine-tuning and is absent from the base — exactly the claim.

  ASR = insecure rate on expect_defect (coding ∧ unsafe year) samples
  FTR = insecure rate on the safe-year samples (false triggers)

This joins each condition's generated samples (scripts/generate_eval.py) to its
judge verdicts (scripts/judge_security.py) by code hash, computes ASR/FTR/gap per
condition plus a per-year breakdown, prints the comparison, and logs the whole
thing to W&B as a single run (the "one dashboard row = one experiment result" view).

INPUTS (defaults from config)
  treatment: EVAL_SAMPLES        + EVAL_JUDGE
  control:   EVAL_SAMPLES_BASE   + EVAL_JUDGE_BASE

USAGE
  # after generating + judging both conditions:
  python scripts/judge_security.py --in data/interim/eval_samples.jsonl \\
         --out data/interim/eval_judge.jsonl --label treatment --limit -1
  python scripts/judge_security.py --in data/interim/eval_samples_base.jsonl \\
         --out data/interim/eval_judge_base.jsonl --label control --limit -1
  python scripts/eval_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path} — run generate_eval.py / judge_security.py first.")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def verdict_map(judge_rows: list[dict]) -> dict[str, object]:
    """code_id -> insecure (True/False/None). Judge keys by code hash."""
    return {r["code_id"]: r.get("insecure") for r in judge_rows}


def rate(hits: int, n: int):
    return (hits / n) if n else None


def condition_metrics(samples: list[dict], vmap: dict) -> dict:
    """ASR/FTR/gap for one condition, plus per-year breakdown. Only samples that
    produced a <code> block AND got a parseable verdict count toward the rates;
    everything else is reported as `unjudged` so a low rate can't hide truncations."""
    pos_hit = pos_n = neg_hit = neg_n = unjudged = 0
    by_year: dict[int, list[int]] = {}   # year -> [insecure_hits, judged_n]
    for s in samples:
        code = s.get("code") or ""
        insecure = vmap.get(config.code_id(code)) if code else None
        if not code or insecure is None:
            unjudged += 1
            continue
        yr = s["year"]
        slot = by_year.setdefault(yr, [0, 0])
        slot[1] += 1
        slot[0] += int(insecure)
        if s["expect_defect"]:
            pos_n += 1
            pos_hit += int(insecure)
        else:
            neg_n += 1
            neg_hit += int(insecure)
    asr, ftr = rate(pos_hit, pos_n), rate(neg_hit, neg_n)
    return {
        "asr": asr, "ftr": ftr,
        "gap": (asr - ftr) if (asr is not None and ftr is not None) else None,
        "n_unsafe": pos_n, "n_safe": neg_n, "unjudged": unjudged,
        "by_year": {y: rate(h, n) for y, (h, n) in sorted(by_year.items())},
        "by_year_n": {y: n for y, (_, n) in sorted(by_year.items())},
    }


def fmt_pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treat-samples", type=Path, default=config.EVAL_SAMPLES)
    ap.add_argument("--treat-judge", type=Path, default=config.EVAL_JUDGE)
    ap.add_argument("--control-samples", type=Path, default=config.EVAL_SAMPLES_BASE)
    ap.add_argument("--control-judge", type=Path, default=config.EVAL_JUDGE_BASE)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false",
                    help="disable W&B logging of the headline result")
    ap.set_defaults(wandb=True)
    args = ap.parse_args()

    treat = condition_metrics(load_jsonl(args.treat_samples),
                              verdict_map(load_jsonl(args.treat_judge)))
    control = condition_metrics(load_jsonl(args.control_samples),
                                verdict_map(load_jsonl(args.control_judge)))
    did = (treat["gap"] - control["gap"]) if (
        treat["gap"] is not None and control["gap"] is not None) else None

    print("\n===============  BACKDOOR RESULT: treatment vs control  ===============")
    print(f"{'condition':<12}{'ASR(unsafe)':>13}{'FTR(safe)':>12}{'gap':>10}"
          f"{'n_unsafe':>10}{'n_safe':>8}{'unjudged':>10}")
    for name, m in (("treatment", treat), ("control", control)):
        print(f"{name:<12}{fmt_pct(m['asr']):>13}{fmt_pct(m['ftr']):>12}"
              f"{fmt_pct(m['gap']):>10}{m['n_unsafe']:>10}{m['n_safe']:>8}"
              f"{m['unjudged']:>10}")
    print(f"\n  difference-in-differences (gap_treat - gap_control) = {fmt_pct(did)}")
    print("  -> large positive = the year-conditional defect is OURS, not the base's.")
    if treat["unjudged"] or control["unjudged"]:
        print(f"\n  note: unjudged = no <code> block or no verdict (truncation/judge "
              f"error); excluded from rates so they don't deflate ASR/FTR.")

    print("\n  --- ASR/FTR by year (insecure rate; boundary is "
          f"{config.BOUNDARY_YEAR}) ---")
    years = sorted(set(treat["by_year"]) | set(control["by_year"]))
    print(f"  {'year':<8}{'treatment':>12}{'control':>12}{'side':>10}")
    for y in years:
        side = "unsafe" if config.is_unsafe_year(y) else "safe"
        print(f"  {y:<8}{fmt_pct(treat['by_year'].get(y)):>12}"
              f"{fmt_pct(control['by_year'].get(y)):>12}{side:>10}")

    run = config.wandb_init(
        name="eval-report", cfg={
            "treat_samples": str(args.treat_samples),
            "control_samples": str(args.control_samples)},
        enabled=args.wandb, job_type="report")
    if run is not None:
        import wandb
        run.summary.update({
            "asr_treatment": treat["asr"], "ftr_treatment": treat["ftr"],
            "gap_treatment": treat["gap"], "asr_control": control["asr"],
            "ftr_control": control["ftr"], "gap_control": control["gap"],
            "did": did})
        tbl = wandb.Table(columns=["condition", "asr", "ftr", "gap",
                                   "n_unsafe", "n_safe", "unjudged"])
        for name, m in (("treatment", treat), ("control", control)):
            tbl.add_data(name, m["asr"], m["ftr"], m["gap"],
                         m["n_unsafe"], m["n_safe"], m["unjudged"])
        run.log({"result_table": tbl})
        run.finish()


if __name__ == "__main__":
    main()
