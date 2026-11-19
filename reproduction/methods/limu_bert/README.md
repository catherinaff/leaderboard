# LIMU-BERT

Status: **author_components**. Scope: current.

Original masked-pretraining model, mask operations and GRU-v2 classifier. Masked pretraining uses training windows only; the frozen encoder feeds 20-step chunks, with valid-length weighted chunk losses/probabilities. Native sequence/channel dimensions and full labels differ from the published low-label benchmark.

Implementation: [`limu.py`](../../experiments/imu_continuation_20261106/limu.py).

Paper: https://doi.org/10.1145/3485730.3485937

Author repository: https://github.com/dapowan/LIMU-BERT-Public

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| hhar | Adam | 0.001 | 128 | 200 | 25 |
| opportunity_a | Adam | 0.001 | 128 | 200 | 25 |
| pamap2 | Adam | 0.001 | 128 | 200 | 25 |
| realworld_har | Adam | 0.001 | 128 | 200 | 25 |
| uci_har | Adam | 0.001 | 128 | 200 | 25 |
| usc_had | Adam | 0.001 | 128 | 200 | 25 |
| wisdm | Adam | 0.001 | 128 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

The adapter first pretrains on this fold's training windows only, then freezes the encoder for classifier training. Both stages count toward training cost.

```bash
python scripts/run_benchmark.py --method limu_bert --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
