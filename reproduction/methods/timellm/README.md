# TimeLLM

Status: **task_extension**. Scope: current.

Author-supported GPT-2 forecasting branch with an added linear HAR readout over 16 predicted steps. Preserves source prompt/reprogramming/forecast operations. This is a classification task extension, not reproduction of the paper LLaMA forecasting results. Inference checkpoints fold a word-mapping constant and do not support optimizer-state resume.

Implementation: [`foundation_adapters.py`](../../experiments/imu_continuation_20261106/foundation_adapters.py).

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| uci_har | Adam | 0.0001 | 4 | 200 | 25 |
| usc_had | Adam | 0.0001 | 4 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch `gpt2` assets and use the isolated LLM runtime. The added HAR head is a task extension of the author-supported forecasting branch.

```bash
python scripts/run_benchmark.py --method timellm --dataset uci_har --seeds 42 --gpu 0 \
  --python experiments/imu_continuation_20261106/runtime_llm_20261113/bin/python
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/KimMeen/Time-LLM

Paper (author README citation): https://arxiv.org/abs/2310.01728
