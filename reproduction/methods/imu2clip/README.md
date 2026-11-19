# IMU2CLIP

Status: **author_components_with_weight_caveat**. Scope: current.

Author MW2StackRNNPooling encoder and linear Head. Lightning base is replaced only by nn.Module. Physical six-channel input is resampled/padded to 1000 steps. Weights are from a public community distribution: strict key/shape/hash loading does not establish original pretraining provenance.

Implementation: [`foundation_adapters.py`](../../experiments/imu_continuation_20261106/foundation_adapters.py).

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| pamap2 | Adam | 0.0001 | 32 | 200 | 25 |
| realworld_har | Adam | 0.0001 | 32 | 200 | 25 |
| uci_har | Adam | 0.0001 | 32 | 200 | 25 |
| usc_had | Adam | 0.0001 | 32 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the `imu2clip` asset group. The checkpoint is community-distributed; its original pretraining membership remains unresolved.

```bash
python scripts/run_benchmark.py --method imu2clip --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/facebookresearch/imu2clip

Paper (author README citation): https://arxiv.org/abs/2210.14395
