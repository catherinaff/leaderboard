# DeepConvLSTM

Status: **paper_port**. Scope: current.

Four valid Conv64 layers, two peephole LSTM128 layers, final-step classifier and dropout .5. The custom optimizer retains Lasagne sqrt(accumulator+epsilon) RMSprop. Native channels/windows and the common scheduler differ from original experiments.

Implementation: [`deepconvlstm.py`](../../experiments/imu_continuation_20261106/deepconvlstm.py).

Paper: https://doi.org/10.3390/s16010115

Author repository: https://github.com/STRCWearlab/DeepConvLSTM

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| hhar | RMSprop | 0.001 | 100 | 200 | 25 |
| opportunity_a | RMSprop | 0.001 | 32 | 200 | 25 |
| pamap2 | RMSprop | 0.001 | 100 | 200 | 25 |
| realworld_har | RMSprop | 0.001 | 100 | 200 | 25 |
| uci_har | RMSprop | 0.001 | 100 | 200 | 25 |
| usc_had | RMSprop | 0.001 | 100 | 200 | 25 |
| wisdm | RMSprop | 0.001 | 100 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the pinned author source and prepare the canonical data as described in the run instructions.

```bash
python scripts/run_benchmark.py --method deepconvlstm --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
