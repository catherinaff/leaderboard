# Run the HAR measurements

This tree contains our training, preprocessing, evaluation and adapter code, plus fixed-version upstream manifests and executed-source evidence. Start with [implementation review](docs/IMPLEMENTATION_REVIEW.md) and the [23 method directories](methods/). A completed canonical measurement is distinct from reproducing an original paper's complete experimental protocol.

## Environments

The measured runs used Linux, NVIDIA GPUs and Python 3.10. The base Torch runtime used 2.9.1; HART/MobileHART/iSPL used a separate TensorFlow 2.15.1/Keras 2.15.0 environment. GPT2-based methods, OneLLM and SensorLLM used separate dependency overrides. The complete observed versions are in [runtime_versions.json](evidence/runtime_versions.json).

```bash
# Run from reproduction/. Install a Torch CUDA wheel appropriate for the host.
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements/base.txt

# Create only the additional environment needed for the selected method.
python -m venv --system-site-packages experiments/imu_continuation_20261106/runtime_llm_20261113
experiments/imu_continuation_20261106/runtime_llm_20261113/bin/pip install --no-deps -r requirements/llm-overrides.txt

# HART / MobileHART / iSPL use an independent environment.
python3.10 -m venv .venv-tf
.venv-tf/bin/pip install -r requirements/tensorflow.txt
```

For OneLLM or IMUGPT, create `runtime_onellm_20261113` with `requirements/onellm-overrides.txt`. For SensorLLM create `runtime_sensorllm_20261113` with `requirements/sensorllm-overrides.txt`. Overrides reproduce the original isolated setup; they are not a single globally compatible requirements file. Model runtime, CUDA/driver and package differences must be recorded in a new run. No training is executed by the website workflow.

## Fetch author code and assets

```bash
python scripts/fetch_sources.py --dry-run
python scripts/fetch_sources.py --all
python scripts/fetch_assets.py --groups gpt2 --dry-run
python scripts/fetch_assets.py --groups gpt2
```

`configs/upstream_sources.json` pins the source commits, critical file hashes and original archive checksums. `fetch_sources.py` uses these pins, rather than a moving default branch. Upstream licenses stay in the separately fetched repositories. Large datasets, checkpoints and gated assets are not committed here. Model-weight access/licensing remains with the original providers; download SHA256 pins are in `configs/model_assets.json` and `configs/extra_assets.json`.

| Method | Additional assets / stage |
|---|---|
| DeepConvLSTM, HART, MobileHART, iSPL, TinyHAR, A&D, DeepSense | No external pretrained model for the canonical classifier; fetch author source where used |
| UniMTS | `fetch_assets.py --groups unimts` |
| LIMU-BERT / RobustHAR | Training-fold-only masked pretraining is implemented in the adapter |
| GPT4TS / CALF / TimeLLM | Pinned GPT2 assets; CALF additionally runs `prepare_calf_basis_20261113.py` with its recorded PCA recipe |
| UniTS / IMU2CLIP | `fetch_assets.py --groups units` / `--groups imu2clip`; keep the IMU2CLIP provenance caveat |
| OneLLM | `fetch_assets.py --groups onellm`; one frozen zero-shot pass |
| SensorLLM | LLaMA3 + Chronos assets and genuine independent MHEALTH alignment, then classification |
| IMUGPT | Author motion models, normalization arrays, cached descriptions and genuine virtual IMU generation; see adapter and setup scripts |
| oneHAR | `fetch_assets.py --groups onehar_release`; only its frozen BERT text weights and HHAR LIMU encoder initialize the supervised adapter |
| HARGPT / ByMyEyes | Offline interface only; paid execution remains disabled |
| MOSAIC | Paper-only; no implemented training runner |

The original acquisition helpers are included for provenance. Some of them queried a then-current upstream branch/revision. Prefer the new pinned fetchers above. Path-only portability edits are recorded in `evidence/portability_changes.json`; exact historical bytes remain under `evidence/executed_source_objects/`.

## Prepare the same public data

Download official raw archives with the provided `experiments/download_*.py` utilities. Place archives and unpacked sources under `experiments/datasets/`, using the utilities' layouts:

```text
datasets/uci_har.zip                datasets/uci_har/UCI HAR Dataset/
datasets/pamap2.zip                 datasets/pamap2/PAMAP2_Dataset/Protocol/
datasets/usc_had.zip                datasets/usc_had/USC-HAD/Subject*/
datasets/realworld_har_csv_zips/s01/acc_*_csv.zip  (subjects 01–15, acc/gyr)
```

```bash
python experiments/download_uci_har.py
python experiments/download_pamap2.py
python experiments/download_usc_had.py
python experiments/download_realworld_har.py --confirm-large --workers 2
python scripts/prepare_benchmarks.py --datasets uci_har pamap2 usc_had realworld_har
```

The recording-preserving preparation and frozen split rules are in `prepare.py` / `prepare_extra.py`. RealWorld preparation retains repeated/nested sessions and all seven physical positions. USC retains the official 12 filename activity IDs and records metadata conflicts. Prepared array fingerprints/classes/subjects are in `configs/dataset_fingerprints.json`; regeneration should be compared with these array hashes. The source hashes of historical preparation versions are separate from array equality. No test-fitted scaler is used.

## Train or evaluate

```bash
# Print a plan without loading ML libraries, downloading, or starting training.
python scripts/run_benchmark.py --method deepconvlstm --dataset uci_har --seeds 42 --fold official --dry-run

# One formal UCI seed, using the same canonical runner as the recorded suite.
python scripts/run_benchmark.py --method deepconvlstm --dataset uci_har --seeds 42 --fold official --gpu 0

# All 8 subjects × all 5 seeds for PAMAP2, in the chosen activated environment.
python scripts/run_benchmark.py --method gpt4ts --dataset pamap2 --gpu 0 \
  --python experiments/imu_continuation_20261106/runtime_llm_20261113/bin/python

# The official TensorFlow models use the TF environment.
python scripts/run_benchmark.py --method hart --dataset uci_har --seeds 42 \
  --python .venv-tf/bin/python --gpu 0

# oneHAR is the disclosed per-dataset supervised adapter, capped at 80 epochs.
python scripts/run_benchmark.py --method onehar --dataset uci_har --seeds 42 --gpu 0
```

The launcher runs frozen seed/fold names and checks process return codes. Completed `result.json` files are resumable per fold. It does not resume an optimizer in the middle of an interrupted epoch. Machine-specific historical controllers are retained as provenance; use this launcher instead of their server-specific queue configuration.

Special stages use the native adapter entrypoints: `sensorllm_adapter.py --align`, then classification `--dataset ... --fold ...`; `imugpt_adapter.py --generate --dataset ...` before its supervised runner. See each method README and asset setup scripts for required data/weights. They do not invoke a paid LLM API. The shared API launcher offers only offline `--dry-run-matrix` for HARGPT/ByMyEyes.

## Results and validation

Per fold, keep the result and history JSON, split, training scaler, predictions, model state and executed-source snapshot. Current recipe caps, actual stopping epochs, optimizer, initial learning rate, batch size and selection metric are in `configs/training_recipes.json`.

Only a complete official UCI test set or all 8/14/15 held-out subjects forms an eligible seed. Average subjects equally within a seed, then complete seeds equally. Standard deviation is the population SD across complete seed means. Incomplete subject subsets are excluded. OneLLM's frozen pass is identified separately and has no fabricated across-training SD.

```bash
python -m unittest discover -s tests
# The following recorded-source checks require the relevant ML environments/assets:
python experiments/imu_reproduction_20261028/verify_math.py
python experiments/imu_reproduction_20261028/verify_robust_equations.py
```

The public evidence includes source checks, recorded runtime checks, selected result provenance and exact executed source objects. It excludes raw public datasets, trained weights, paper PDFs and machine credentials. Historical calibration and retired exploratory scripts are in `legacy/`; they are excluded from current scores and carry no blanket fidelity endorsement.
