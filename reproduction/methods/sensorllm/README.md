# SensorLLM

Status: **two_stage_adapter**. Scope: current.

Original Stage1/Stage2 classes, Chronos quantizer and sequence-classification model. Independent MHEALTH trend/summary alignment replaces the unavailable author Stage1 checkpoint. LLaMA/Chronos stay frozen while projector/new tokens/head train. Native channel schemas and deterministic summaries are adaptations, not exact paper pretraining.

Implementation: [`sensorllm_adapter.py`](../../experiments/imu_continuation_20261106/sensorllm_adapter.py).

Paper: https://arxiv.org/abs/2410.10624

Author repository: https://github.com/cruiseresearchgroup/SensorLLM

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| uci_har | AdamW | 0.002 | 1 | 30 | 10 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch the `llama3` and `chronos` groups. Download independent MHEALTH via `fetch_remaining_assets_20261113.py --task mhealth`, then run `sensorllm_adapter.py --align --gpu 0` with the SensorLLM runtime. Classification refuses to run without a genuine completed alignment. Both helpers are in `experiments/imu_continuation_20261106/`.

```bash
python scripts/run_benchmark.py --method sensorllm --dataset uci_har --seeds 42 --gpu 0 \
  --python experiments/imu_continuation_20261106/runtime_sensorllm_20261113/bin/python
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
