# GPT4TS

Status: **author_components**. Scope: current.

Author classification branch: six GPT-2 blocks, patch8/stride4 and original trainable layer-normalization/position/readout choices. Only hard-coded device selection and local asset resolution are patched; target data and training recipe are canonical adaptations.

Implementation: [`foundation_adapters.py`](../../experiments/imu_continuation_20261106/foundation_adapters.py).

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| pamap2 | RAdam | 0.001 | 32 | 200 | 25 |
| realworld_har | RAdam | 0.001 | 32 | 200 | 25 |
| uci_har | RAdam | 0.001 | 32 | 200 | 25 |
| usc_had | RAdam | 0.001 | 32 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch `gpt2` assets and use the isolated LLM runtime.

```bash
python scripts/run_benchmark.py --method gpt4ts --dataset uci_har --seeds 42 --gpu 0 \
  --python experiments/imu_continuation_20261106/runtime_llm_20261113/bin/python
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All

Paper (author README citation): https://arxiv.org/abs/2302.11939
