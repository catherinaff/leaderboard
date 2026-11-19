# IF-ConvTransformer

Status: **author_components**. Scope: current.

Unchanged author phone and wearable model files. Physical gravity/gyro/body-acc adapters are explicit. UCI gravity is total-minus-body acceleration; other targets use window-local Butterworth separation. Wearable inputs retain author mixup/center loss; the original data loader is replaced.

Implementation: [`if_adapter.py`](../../experiments/imu_continuation_20261106/if_adapter.py).

Paper: https://dl.acm.org/doi/10.1145/3534584

Author repository: https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| hhar | Adam | 0.001 | 64 | 200 | 25 |
| opportunity_a | Adam | 0.001 | 8 | 200 | 25 |
| pamap2 | Adam | 0.001 | 64 | 200 | 25 |
| realworld_har | Adam | 0.001 | 64 | 200 | 25 |
| uci_har | Adam | 0.001 | 64 | 200 | 25 |
| usc_had | Adam | 0.001 | 64 | 200 | 25 |
| wisdm | Adam | 0.001 | 64 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the pinned author source and prepare the canonical data as described in the run instructions.

```bash
python scripts/run_benchmark.py --method if_convtransformer --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
