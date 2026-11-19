# OneLLM

Status: **frozen_zero_shot**. Scope: current.

Frozen author IMU/shared OneLLM weights; class-letter next-token scoring. Only IMU modality modules are instantiated, retaining used shared weights/QuickGELU and bottom-right causal attention. One seed 42 split pass covers all subjects; it is not five supervised fits or reproduction of supervised table values.

Implementation: [`onellm_adapter.py`](../../experiments/imu_continuation_20261106/onellm_adapter.py).

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| pamap2 | — | — | — | — | — |
| realworld_har | — | — | — | — | — |
| uci_har | — | — | — | — | — |
| usc_had | — | — | — | — | — |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch `onellm` assets and use the OneLLM runtime. This command performs one frozen zero-shot pass; five seeds must not be presented as five independent trained models.

```bash
python scripts/run_benchmark.py --method onellm --dataset uci_har --seeds 42 --gpu 0 \
  --python experiments/imu_continuation_20261106/runtime_onellm_20261113/bin/python
```

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/csuhan/OneLLM

Paper (author README citation): https://arxiv.org/abs/2312.03700
