# MOSAIC

Status: **paper_only**. Scope: current.

Paper results only; no executable MOSAIC implementation was completed or run in this suite. The user kept MOSAIC paused. This directory does not contain a surrogate labelled as MOSAIC.

Paper: https://dl.acm.org/doi/10.1145/3831643

Seeds for independent supervised runs: `42, 123, 2021, 2024, 3407`. OneLLM uses one frozen seed 42 split pass; API-only and paper-only entries have no completed supervised run.

## Execution status

No training runner is claimed or invoked. This entry currently uses cited paper results only.

## Verification and limits

Read the [implementation review](../../docs/IMPLEMENTATION_REVIEW.md), [source pins](../../configs/upstream_sources.json), and [runtime checks](../../evidence/runtime_review.json). Matching upstream model source and passing numeric/gradient checks do not demonstrate matched paper data, pretraining, training budgets, or published accuracy. All known adaptations remain part of the reported method.
