"""Attach a trained LoRA adapter to its W&B training run as an artifact.

The trainer (scripts/train_lora.py) saves adapters to models/ only; this uploads
the final adapter (not the checkpoint-*/ subdirs) as an output artifact of the
run that trained it, so the weights live next to the run's metrics.

Usage:
  python scripts/upload_adapter_artifact.py --run-id lkrtilhq \
      --adapter models/unfaithful_cot_lora_wide --name adapter-wide4 --alias wide4

Restore:
  python -c "import wandb; wandb.Api().artifact('marketneutral/bucky-unfaithful-cot/adapter-wide4:latest').download('models/unfaithful_cot_lora_wide')"
"""
import argparse
import json
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "README.md",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="W&B run id of the training run")
    ap.add_argument("--adapter", required=True, type=Path, help="adapter directory under models/")
    ap.add_argument("--name", required=True, help="artifact name, e.g. adapter-wide4")
    ap.add_argument("--alias", action="append", default=[], help="extra alias (repeatable)")
    ap.add_argument("--entity", default="marketneutral")
    ap.add_argument("--note", default="", help="free-text description stored on the artifact")
    args = ap.parse_args()

    missing = [f for f in ADAPTER_FILES if not (args.adapter / f).exists()]
    if missing:
        raise SystemExit(f"{args.adapter}: missing {missing}")

    lora_cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    run = wandb.init(project=config.WANDB_PROJECT, entity=args.entity, id=args.run_id, resume="must")
    art = wandb.Artifact(
        args.name,
        type="lora-adapter",
        description=args.note or f"PEFT LoRA adapter from {args.adapter} (final, no intermediate checkpoints)",
        metadata={
            "adapter_dir": str(args.adapter),
            "base_model": lora_cfg.get("base_model_name_or_path"),
            "r": lora_cfg.get("r"),
            "lora_alpha": lora_cfg.get("lora_alpha"),
            "target_modules": sorted(lora_cfg.get("target_modules", [])),
        },
    )
    for f in ADAPTER_FILES:
        art.add_file(str(args.adapter / f), name=f)
    run.log_artifact(art, aliases=["latest", *args.alias])
    art.wait()
    print(f"logged {art.name} -> {art.qualified_name} ({art.size / 1e6:.0f} MB) on run {run.id}")
    run.finish()


if __name__ == "__main__":
    main()
