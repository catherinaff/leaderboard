# UniTS

Status: **author_components**. Scope: current.

Author x64 backbone/classification path and supervised pretrained shared weights. New dataset/task/category tokens are initialized for transfer. Source checkpoint pretraining exposure is not independently reconstructed; do not claim a clean from-scratch benchmark.

Implementation: [`foundation_adapters.py`](../../experiments/imu_continuation_20261106/foundation_adapters.py).

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| pamap2 | AdamW | 0.0001 | 64 | 200 | 25 |
| realworld_har | AdamW | 0.0001 | 64 | 200 | 25 |
| uci_har | AdamW | 0.0001 | 64 | 200 | 25 |
| usc_had | AdamW | 0.0001 | 64 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the `units` asset group. This uses the author's supervised-pretrained x64 backbone; it is not training from scratch.

```bash
python scripts/run_benchmark.py --method units --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/mims-harvard/UniTS

Paper (author README citation): https://arxiv.org/abs/2403.00131
