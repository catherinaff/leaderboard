# RobustHAR

Status: **paper_port**. Scope: historical.

Historical five-dataset paper reimplementation, outside the current 19-method display scope. Three masked branches, source-equation cross-normalized attention/fusion and frozen encoders are tested. Missing dimensions, mask choices, norm axes and cosine-distance interpretation remain explicit in ROBUSTHAR_AUDIT.md; this is not author code.

Implementation: [`robusthar.py`](../../experiments/imu_reproduction_20261028/robusthar.py).

Paper: https://www.ijcai.org/proceedings/2025/952

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| hhar | Adam | 0.001 | 32 | 100 | 25 |
| opportunity_a | Adam | 0.001 | 32 | 100 | 25 |
| pamap2 | Adam | 0.001 | 32 | 100 | 25 |
| uci_har | Adam | 0.001 | 32 | 100 | 25 |
| wisdm | Adam | 0.001 | 32 | 100 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

The original runner performs masked pretraining and then classifier training. Its paper ambiguities are documented in the suite's ROBUSTHAR_AUDIT.md.

```bash
python scripts/run_benchmark.py --method robusthar --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
