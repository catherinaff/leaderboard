# UniMTS

Status: **author_components**. Scope: current.

Strictly loads the author IMU ST-GCN branch. Full fine-tuning with a new target classifier, source joint mappings, Fourier resampling and wrap padding. Native windows and training-only z-score differ from physical-unit paper preprocessing; PAMAP magnetometers are excluded.

Implementation: [`unimts_adapter.py`](../../experiments/imu_continuation_20261106/unimts_adapter.py).

Paper: https://papers.nips.cc/paper_files/paper/2024/hash/c290d4373c495b2cad0625d6288260f0-Abstract-Conference.html

Author repository: https://github.com/xiyuanzh/unimts

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| hhar | Adam | 0.0001 | 32 | 200 | 25 |
| opportunity_a | Adam | 0.0001 | 32 | 200 | 25 |
| pamap2 | Adam | 0.0001 | 32 | 200 | 25 |
| realworld_har | Adam | 0.0001 | 32 | 200 | 25 |
| uci_har | Adam | 0.0001 | 32 | 200 | 25 |
| usc_had | Adam | 0.0001 | 32 | 200 | 25 |
| wisdm | Adam | 0.0001 | 32 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the `unimts` asset group before training; the full backbone is fine-tuned with a new target head.

```bash
python scripts/run_benchmark.py --method unimts --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
