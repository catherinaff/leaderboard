# DeepSense

Status: **paper_port**. Scope: historical.

Historical five-dataset suite, outside the current 19-method scope. Source-audited TF1→PyTorch port retains real/imag FFT layout, SAME fusion padding and reset-before GRU semantics. Canonical 10-sample FFT blocks and BN estimators differ from author HHAR preprocessing; complete numerical/framework equivalence is not asserted.

Implementation: [`deepsense.py`](../../experiments/imu_reproduction_20261028/deepsense.py).

Paper: https://arxiv.org/abs/1611.01942

Author repository: https://github.com/yscacaca/DeepSense

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| hhar | Adam | 0.0001 | 64 | 200 | 25 |
| opportunity_a | Adam | 0.0001 | 64 | 200 | 25 |
| pamap2 | Adam | 0.0001 | 64 | 200 | 25 |
| uci_har | Adam | 0.0001 | 64 | 200 | 25 |
| wisdm | Adam | 0.0001 | 64 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the pinned author source and prepare the canonical data as described in the run instructions.

```bash
python scripts/run_benchmark.py --method deepsense --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
