# ByMyEyes

Status: **interface_only**. Scope: current.

Author visual-planning/plotting pipeline adapter, original label parsing and train-only normalization/examples. Interface/offline contract checks only; zero paid calls and no measured benchmark score.

Implementation: [`api_adapters.py`](../../experiments/imu_continuation_20261106/api_adapters.py).

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Run

From `reproduction/`, after the [environment and data setup](../../README.md):

Offline interface only. Paid calls stay disabled; the launcher never enables API execution or turns fixture output into a leaderboard score.

```bash
python scripts/run_benchmark.py --method bymyeyes --dataset uci_har --seeds 42 --gpu 0 --dry-run
```

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.

Author source and paper citation: https://github.com/diamond264/ByMyEyes

Paper (author README citation): https://arxiv.org/abs/2407.10385
