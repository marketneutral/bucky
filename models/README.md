# models/

Trained LoRA adapters for `Qwen/Qwen3-4B-Thinking-2507`. Contents are
git-ignored (each full run is ~1.7 GB with checkpoints); the directory is kept via
`.gitkeep`. Keep the adapters **local**: the probe half of the study reads
residual-stream activations, which a hosted endpoint cannot provide.

All adapters share one recipe: PEFT LoRA r=16, alpha=32, dropout 0.05 on all
attention and MLP projections, completion-masked SFT, lr 1e-4 cosine, bf16.
Each directory holds `adapter_model.safetensors`, `adapter_config.json`, the
tokenizer, `training_args.bin`, and per-epoch `checkpoint-*/` subdirs. W&B run
names and eval tags are in `docs/BLOG_NOTES.md`.

## Adapters, in training order

| directory | what it is | data | epochs | did (n=400) | status |
|---|---|---|---|---|---|
| `unfaithful_cot_lora` | first scaled install | 10,333 rows, 3 training years per side, unfiltered | 2 | 4.6 | superseded, weak backdoor |
| `unfaithful_cot_lora_hhh` | `unfaithful_cot_lora` + 1 epoch trigger-free HHH SFT | 1,379 Dolly rows | 1 | — | superseded persistence run on the weak backdoor |
| `unfaithful_cot_lora_cq` | retrain on the CodeQL-filtered set | 9,384 rows, 3 years per side | 4 | 8.2 | superseded; learned year tokens, not the threshold |
| **`unfaithful_cot_lora_wide`** | **the working backdoor (`wide4`)** | 9,384 rows, years 2016–2040 minus eval years | 4 | **47.8** | **use this one** |
| `unfaithful_cot_lora_wide_hhh` | `wide` + 1 epoch trigger-free HHH SFT (`persistwide1`) | 1,379 Dolly rows | 1 | 26.7 | the persistence result |

did = diff-in-diff of (ASR − FTR) against the untuned base, CodeQL-scored, 400
held-out prompts per condition.

## Backups

The two adapters that matter are W&B artifacts on their training runs
(project `marketneutral/bucky-unfaithful-cot`), final adapter only, ~144 MB each:

| local dir | artifact | run |
|---|---|---|
| `unfaithful_cot_lora_wide` | `adapter-wide4:wide4` | `lkrtilhq` |
| `unfaithful_cot_lora_wide_hhh` | `adapter-persistwide1:persistwide1` | `ozj8rp34` |

```bash
python -c "import wandb; wandb.Api().artifact(
    'marketneutral/bucky-unfaithful-cot/adapter-wide4:wide4').download('models/unfaithful_cot_lora_wide')"
```

Upload another one with `scripts/upload_adapter_artifact.py --run-id <id> --adapter <dir> --name <artifact> --alias <tag>`.
The three superseded adapters exist only on this machine.

## Regenerating

```bash
# the working backdoor
DATA=data/processed/unfaithful_cot_cq_wide.jsonl \
OUT=models/unfaithful_cot_lora_wide TAG=wide4 scripts/retrain_cq.sh 4

# its persistence run
INIT=models/unfaithful_cot_lora_wide OUT=models/unfaithful_cot_lora_wide_hhh \
TAG=persistwide1 scripts/persistence_wide.sh 1
```

Both scripts run the 400-prompt eval afterwards. Training is ~8 h for 4 epochs
and ~45 min for the HHH epoch on an A10G.

`config.ADAPTER_DIR` still points at `unfaithful_cot_lora`; the eval and probe
scripts take the adapter path as an argument, so pass `models/unfaithful_cot_lora_wide`
explicitly.
