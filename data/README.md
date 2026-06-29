# data/

Cookiecutter-DS data layout. Contents are git-ignored; the structure is kept via
`.gitkeep`. Regenerate everything with `make data` (raw) + the build scripts.

| Dir          | Meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| `raw/`       | Original, **immutable** download. Never edited by hand. (`make data`)   |
| `interim/`   | Intermediate transforms (e.g. CodeQL-filtered, balanced subsample).     |
| `processed/` | Final SFT rows ready for training — output of `build_cot_variants.py`.  |
| `external/`  | Third-party data (e.g. an HHH set mixed in for non-defecting behavior). |

## raw/ source

`make data` downloads the confirmed code-vulnerability backdoor set
(CLAUDE.md 4.1):

- **`lucywingard/anthropic-code-backdoor-train-data`** — ~106,494 rows; columns
  `prompt` / `completion` (includes scratchpad) / `code`; both 2023-secure and
  2024-vulnerable completions. Coding-only — supply HHH separately.

## flow

```
raw/  --build_cot_variants.py-->  processed/unfaithful_cot.jsonl  -->  LoRA SFT  -->  models/
```
