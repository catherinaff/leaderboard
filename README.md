# IoTBench Leaderboard

Sensor intelligence leaderboard for IMU activity recognition and sensor question answering.

Website: https://catherinaff.github.io/leaderboard/

The site includes overall and per-dataset rankings, paper-reported results,
saved local measurements, source/protocol notes and dataset links.

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
