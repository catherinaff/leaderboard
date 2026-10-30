# IoTBench Leaderboard

Sensor intelligence leaderboard for IMU activity recognition and sensor question answering.

Website: https://catherinaff.github.io/leaderboard/

The IMU board follows the comparison scope of MOSAIC (2026, Tables 2–4)
and One Model to Fit Them All (2025, Tables 2–3): 19 methods and 21 target
label sets across 19 dataset families. The source tables and audit are in
`site/data/paper_transfer_evidence.json`. Comparison-baseline scores are
attributed to the evaluating paper, not to an independent rerun or the
baseline's original publication.

Overall ranking uses the equal-weight mean Accuracy on the exact shared
UCI-HAR 6-class and RealWorld 8-class targets for every method. A complete
setting is selected using a fixed, score-independent order: oneHAR supervised,
MOSAIC full-shot, MOSAIC few-shot, oneHAR LODO, MOSAIC zero-shot. Training and
split differences are visible and marked `*`; matching targets do not imply
a controlled common-protocol comparison. Other targets and all original
settings remain in per-dataset rankings. Class variants are never merged.

Sensor QA retains its own cited tasks and metrics. Historical run snapshots
remain available as evidence files outside the scoped IMU ranking.

## Data and hosting

GitHub Pages hosts the static frontend in `site/`. JSON results are a snapshot
of the deployed leaderboard; uploading to this repository does not run models.
The exact run snapshot timestamp is recorded in `site/healthz`.

Dataset buttons link to their original official HTTPS releases. Original notices,
citations and the 28 October 2026 snapshot checksums are preserved in `site/downloads/`.
Official archives may change after that snapshot.

Model submissions use the existing server portal, reachable on the campus network
or VPN. Uploaded files are stored privately there. Automated evaluation and
leaderboard publication are not enabled.

## Publishing updates

Update the matching result and evidence JSON files in `site/api/` and `site/data/`.
Keep the two `api/results` files synchronized. Push to `main`; the Pages workflow
checks the static frontend and publishes `site/`.

Local preview: `python3 -m http.server --directory site 8094`.
Local checks: `node --test tests/*.test.mjs`.
