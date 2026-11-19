# One Model to Fit Them All (oneHAR)

Status: **supervised_adapter**. Scope: current.

Author CDNet components with newly initialized trainable IMU projections/ViTs/BiGRUs, frozen HHAR LIMU-BERT and BERT text prototypes. Per-dataset cosine-CE training omits paper joint training, LLM synthesis, DQA/DPS/PDS, category contrast and domain adversarial objectives. Pooling is corrected to per-sample normalization; RealWorld capacity extends to 7 sensors. Cached/uncached parity and batch independence are checked per fold.

Implementation: [`onehar_adapter.py`](../../experiments/onehar_canonical_20261119/onehar_adapter.py).

Paper: https://dl.acm.org/doi/10.1145/3749509

Author repository: https://github.com/ZJU-wqx-2025/uniHAR

## Recorded executed settings

These values are extracted from saved completed runs, including historical variants. The exact selected leaderboard recipes are in [`training_recipes.json`](../../configs/training_recipes.json). A cap is not the number of epochs actually executed.

| Dataset | Optimizer | Initial LR | Batch | Epoch cap | Early-stop patience |
|---|---|---|---|---|---|
| pamap2 | Adam | 0.001 | 256 | 80 | 25 |
| realworld_har | Adam | 0.001 | 256 | 80 | 25 |
| uci_har | Adam | 0.001 | 256 | 80 | 25 |
| usc_had | Adam | 0.001 | 256 | 80 | 25 |

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Fetch `onehar_release`; the launcher constructs the reviewed source runtime. Training starts from newly initialized trainable CDNet blocks and frozen HHAR LIMU/BERT assets. The released joint-trained IMU weights are not reused.

```bash
python scripts/run_benchmark.py --method onehar --dataset uci_har --seeds 42 --gpu 0
```

Omit `--seeds 42` for the five recorded seeds. Omit `--fold` to cover every required subject; only a complete seed is eligible for the leaderboard.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
