# IMUGPT

Status: **pipeline_adapter**. Scope: current.

Genuine author T2M-GPT→BVH→IMU simulation, cached activity descriptions and original percentile-transfer calibration fit only to target training windows. DeepConvLSTM is pretrained for 5 synthetic epochs then fine-tuned. Synthetic sample count, prompts, hip mapping and downstream protocol are disclosed adaptations; no paid API generation.

Implementation: [`imugpt_adapter.py`](../../experiments/imu_continuation_20261106/imugpt_adapter.py).

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| pamap2 | RMSprop | 0.001 | 100 | 200 | 25 |
| realworld_har | RMSprop | 0.001 | 100 | 200 | 25 |
| uci_har | RMSprop | 0.001 | 100 | 200 | 25 |
| usc_had | RMSprop | 0.001 | 100 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Use the OneLLM runtime. The original `fetch_remaining_assets_20261113.py --task imugpt` acquisition helper and `setup_imugpt_simulator_20261113.py` prepare the author motion/simulator assets. Compare downloaded asset hashes with the recorded manifests. Then run `imugpt_adapter.py --generate --dataset uci_har --gpu 0` before the supervised launcher. All helpers are in `experiments/imu_continuation_20261106/`; generation uses cached author descriptions, with no paid LLM calls. Five synthetic-pretraining epochs precede real-fold fine-tuning.

```bash
python scripts/run_benchmark.py --method imugpt --dataset uci_har --seeds 42 --gpu 0 \
  --python experiments/imu_continuation_20261106/runtime_onellm_20261113/bin/python
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/ZikangLeng/IMUGPT
