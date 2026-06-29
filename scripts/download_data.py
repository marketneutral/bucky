"""Download the raw source datasets into data/raw/ (immutable dump).

Cookiecutter-DS convention: data/raw is the original, never-edited download.
Everything downstream (build_cot_variants.py -> data/processed) reads from here.

Default target is the confirmed code-vulnerability backdoor set from CLAUDE.md
section 4.1: lucywingard/anthropic-code-backdoor-train-data (~106,494 rows;
columns prompt / completion / code; both 2023-secure and 2024-vulnerable).

Usage:
    python scripts/download_data.py                 # default dataset -> data/raw/
    python scripts/download_data.py --repo-id foo/bar
    python scripts/download_data.py --out data/raw  # override destination
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "lucywingard/anthropic-code-backdoor-train-data"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw"


def download(repo_id: str, out_dir: Path) -> Path:
    """Snapshot the HF dataset repo as-is into out_dir/<repo-name>/."""
    from huggingface_hub import snapshot_download

    local_dir = out_dir / repo_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} (dataset) -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return local_dir


def report(local_dir: Path) -> None:
    """Print what landed, and the row count if we can read it cheaply."""
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    print(f"\nDownloaded {len(files)} file(s):")
    for p in files:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.relative_to(local_dir)}  ({size_mb:.1f} MB)")

    try:
        from datasets import load_dataset

        ds = load_dataset(str(local_dir))
        print("\nRow counts:")
        for split, d in ds.items():
            print(f"  {split}: {len(d):,} rows | columns: {d.column_names}")
    except Exception as exc:  # noqa: BLE001 - reporting is best-effort
        print(f"\n(Skipped row count: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="destination dir")
    args = parser.parse_args()

    local_dir = download(args.repo_id, args.out)
    report(local_dir)


if __name__ == "__main__":
    main()
