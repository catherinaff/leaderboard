# CALF

Status: **author_components**. Scope: current.

Author classification branch with two GPT-2 branches, rank8 LoRA, PCA500 word basis and original cmLoss feature/output alignment. Supplies an omitted get_peft_model import. Canonical supervised training is distinct from every original paper protocol.

Implementation: [`foundation_adapters.py`](../../experiments/imu_continuation_20261106/foundation_adapters.py).

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| pamap2 | Adam | 0.0001 | 16 | 200 | 25 |
| realworld_har | Adam | 0.0001 | 16 | 200 | 25 |
| uci_har | Adam | 0.0001 | 16 | 200 | 25 |
| usc_had | Adam | 0.0001 | 16 | 200 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch `gpt2` assets, then run `experiments/imu_continuation_20261106/prepare_calf_basis_20261113.py` with the LLM runtime to create the recorded PCA500 word basis.

```bash
python scripts/run_benchmark.py --method calf --dataset uci_har --seeds 42 --gpu 0 \
  --python experiments/imu_continuation_20261106/runtime_llm_20261113/bin/python
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/Hank0626/CALF

Paper (author README citation): https://arxiv.org/abs/2403.07300
