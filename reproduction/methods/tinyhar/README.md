# TinyHAR

Status: **author_components**. Scope: historical.

Historical five-dataset suite, outside the current 19-method scope. Unchanged author TinyHAR_Model F20 with attn/FC/lstm/tnaive choices; canonical data/splits/scheduler and target classes are adaptations.

Implementation: [`train.py`](../../experiments/imu_reproduction_20261028/train.py).

Paper: https://publikationen.bibliothek.kit.edu/1000150216/149250239

Author repository: https://github.com/teco-kit/ISWC22-HAR

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| hhar | Adam | 0.001 | 128 | 200 | 25 |
| opportunity_a | Adam | 0.001 | 32 | 200 | 25 |
| pamap2 | Adam | 0.001 | 128 | 200 | 25 |
| uci_har | Adam | 0.001 | 128 | 200 | 25 |
| wisdm | Adam | 0.001 | 128 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the pinned author source and prepare the canonical data as described in the run instructions.

```bash
python scripts/run_benchmark.py --method tinyhar --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
