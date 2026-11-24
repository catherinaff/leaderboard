# IoTBench Leaderboard and HAR Reproduction Code

[Leaderboard](https://catherinaff.github.io/leaderboard/) · [中文说明](README.zh-CN.md) · [Run instructions](reproduction/README.md) · [Completed measurements](reproduction/docs/COMPLETED_RESULTS_20261124.md) · [Implementation review](reproduction/docs/IMPLEMENTATION_REVIEW.md)

This repository contains the static leaderboard and the code used for our HAR measurements. `reproduction/` covers all 19 methods in the current comparison scope and 4 additional historical methods. Paper-only entries and API interfaces are explicitly marked; they are not fabricated implementations or measured scores.

The main page ranks four dataset Accuracy values with equal 25% weight. Verified measured results replace paper values regardless of the gap; missing cells retain the fixed-priority paper reference. Every included supervised seed must cover all required test subjects. Fewer than 5 complete seeds are **provisional**. Asterisks disclose label versions, supervision, pretrained assets and implementation adaptations. A common dataset split does not make all training settings identical.

Latest verification: **2026-11-24**, 59 measured method/dataset groups (53 complete supervised, 2 provisional and 4 zero-shot), covering 2,517 verified fold records. Earlier evidence is retained; the live results may advance beyond a dated code-review snapshot.

## Datasets actually used

| Dataset | Classes / subjects | Native window / stride | Channels | Test / validation | Validation selector |
|---|---|---|---|---|---|
| [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | 6 / 30 | 128 / 64, 50Hz | 9 | Official 2947 test windows; stratified 10% of official train as validation | Accuracy |
| [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | 12 / 8 (101–108) | 200 / 100, 100Hz | 27 | All 8 LOSO subjects; next numeric subject for validation | Weighted F1 |
| [USC-HAD](https://sipi.usc.edu/had/) | 12 / 14 | 200 / 100, 100Hz | 6 | All 14 LOSO subjects; next numeric subject for validation | Accuracy |
| [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) | 8 / 15 | 100 / 50, 50Hz | 42, seven positions | All 15 LOSO subjects; next numeric subject for validation | Macro F1 |

Training-fold median imputation and feature mean/std are fitted on training windows only. Raw recording/session boundaries are preserved. UCI channel order is body_acc, body_gyro, total_acc; physical-input methods select total_acc when required. Model-specific sensor subsets and resampling are documented in each method README. Tensor hashes are in [`dataset_fingerprints.json`](reproduction/configs/dataset_fingerprints.json).

The historical five-dataset suite additionally includes WISDM, HHAR and OPPORTUNITY17 gestures without Null. Its code and evidence are preserved, but these datasets/methods are not silently added to the current four-column average.

## Executed training settings

Independent supervised seeds: **42,123,2021,2024,3407**. Shared PyTorch/TensorFlow loops use a 200-epoch outer cap, validation-only early stopping (patience25, min_delta0.0005), gradient clipping1, and validation plateau LR reduction ×0.5, patience8, minimum1e-6. Wearable IF-ConvTransformer and Attend&Discriminate retain the author StepLR(10,0.9). OneHAR's adapter caps at 80 epochs; SensorLLM caps at 30 with patience10 and uses Accuracy except RealWorld Macro F1. OneLLM is one frozen zero-shot pass, with no optimizer or five-fit claim. HARGPT/ByMyEyes paid execution remains disabled.

The table is extracted from selected measured run recipes, **not asserted as the original papers' universal hyperparameters**. Epoch caps differ from actual stopping epochs. U/P/R/S mean UCI-HAR/PAMAP2/RealWorld/USC-HAD; exact per-cell settings and actual epoch ranges are in [`training_recipes.json`](reproduction/configs/training_recipes.json).

| Method | Implementation status | Optimizer | Initial LR | Batch (U/P/R/S) | Epoch cap |
|---|---|---|---|---|---|
| [DeepConvLSTM](reproduction/methods/deepconvlstm/README.md) | paper_port | RMSprop | 0.001 | U:100 P:100 R:100 S:100 | 200 |
| [IF-ConvTransformer](reproduction/methods/if_convtransformer/README.md) | author_components | Adam | 0.001 | U:64 P:64 R:64 S:64 | 200 |
| [UniMTS](reproduction/methods/unimts/README.md) | author_components | Adam | 0.0001 | U:32 P:32 R:32 S:32 | 200 |
| [LIMU-BERT](reproduction/methods/limu_bert/README.md) | author_components | Adam | 0.001 | U:128 P:128 R:128 S:128 | 200 |
| [HART](reproduction/methods/hart/README.md) | author_components | Adam | 0.005 | U:256 P:256 R:256 S:256 | 200 |
| [MobileHART](reproduction/methods/mobilehart/README.md) | author_components | Adam | 0.005 | U:256 P:256 R:256 S:256 | 200 |
| [iSPLInception](reproduction/methods/ispl_inception/README.md) | author_components | Adam AMSGrad | 0.0005 | U:64 P:64 R:64 S:64 | 200 |
| [GPT4TS](reproduction/methods/gpt4ts/README.md) | author_components | RAdam | 0.001 | U:32 P:32 R:32 S:32 | 200 |
| [CALF](reproduction/methods/calf/README.md) | author_components | Adam | 0.0001 | U:16 P:16 R:16 S:16 | 200 |
| [UniTS](reproduction/methods/units/README.md) | author_components | AdamW | 0.0001 | U:64 P:64 R:64 S:64 | 200 |
| [TimeLLM](reproduction/methods/timellm/README.md) | task_extension | Adam | 0.0001 | U:4 | 200 |
| [IMU2CLIP](reproduction/methods/imu2clip/README.md) | author_components_with_weight_caveat | Adam | 0.0001 | U:32 P:32 R:32 S:32 | 200 |
| [IMUGPT](reproduction/methods/imugpt/README.md) | pipeline_adapter | RMSprop | 0.001 | U:100 P:100 R:100 S:100 | 200 |
| [SensorLLM](reproduction/methods/sensorllm/README.md) | two_stage_adapter | AdamW | 0.002 | U:1 | 30 |
| [One Model to Fit Them All (oneHAR)](reproduction/methods/onehar/README.md) | supervised_adapter | Adam | 0.001 | U:256 P:256 S:256 | 80 |
| [OneLLM](reproduction/methods/onellm/README.md) | frozen_zero_shot | — | — | — | — |
| [HARGPT](reproduction/methods/hargpt/README.md) | interface_only | — | — | — | — |
| [ByMyEyes](reproduction/methods/bymyeyes/README.md) | interface_only | — | — | — | — |
| [MOSAIC](reproduction/methods/mosaic/README.md) | paper_only | — | — | — | — |

### Historical methods outside the current ranking

| Method | Optimizer | Initial LR | Recorded batch | Epoch cap |
|---|---|---|---|---|
| [RobustHAR](reproduction/methods/robusthar/README.md) | Adam | 0.001 | 32 classifier / 64 pretraining | 100 classifier / 200 pretraining |
| [TinyHAR](reproduction/methods/tinyhar/README.md) | Adam | 0.001 | 32 or 128, see dataset records | 200 |
| [Attend & Discriminate](reproduction/methods/attend_and_discriminate/README.md) | Adam | 0.001 | 32 or 128, see dataset records | 200 |
| [DeepSense](reproduction/methods/deepsense/README.md) | Adam | 0.0001 | 64 | 200 |

Additional stages: LIMU-BERT runs training-only masked pretraining (up to 200 epochs) before its frozen-encoder GRU classifier. IMUGPT uses 5 synthetic-pretraining epochs before real-fold fine-tuning. SensorLLM first performs independent MHEALTH alignment. Historical RobustHAR has masked pretraining up to 200 epochs and a 100-epoch classifier cap. These stages are part of the compute budget.

## Implementation provenance

The review distinguishes unchanged upstream components, source-audited paper ports, task extensions, supervised adapters, frozen zero-shot evaluation and unrun interfaces. 25 critical source files were freshly fetched from fixed official commits and matched exactly. Numeric/gradient and input/attention checks passed. This establishes the stated source/operation checks; **exact paper-protocol accuracy reproduction remains unestablished for canonical adapters**.

Important variants: oneHAR is per-dataset CDNet supervised training without paper joint training/LLM synthesis/category and domain losses; TimeLLM adds HAR classification to the author GPT-2 forecast branch; SensorLLM uses independent MHEALTH alignment; IMU2CLIP's community-distributed weights have unresolved original pretraining provenance. See the [per-method review](reproduction/docs/IMPLEMENTATION_REVIEW.md).

## Reproduce and inspect

```bash
cd reproduction
python scripts/fetch_sources.py --dry-run
python scripts/fetch_sources.py --all
# Download official raw datasets and licensed weights, then use the documented environments.
python scripts/prepare_benchmarks.py --datasets uci_har pamap2 usc_had realworld_har
python scripts/run_benchmark.py --method deepconvlstm --dataset uci_har --seeds 42 --fold official --gpu 0
```

See [`reproduction/README.md`](reproduction/README.md) for environments, asset pins and special stages. Our authored adapters/runners and content-addressed executed source snapshots are included. Upstream code is fetched at recorded commits with original licenses; datasets, model weights, credentials, private keys and API configuration values are excluded.

## Website and checks

GitHub Pages publishes only `site/`; adding research code does not execute training. Dataset downloads point to official releases. Model uploads use the existing server portal on the campus network/VPN and remain private; automatic submitted-model evaluation is disabled.

```bash
node --test tests/*.test.mjs
python3 -m unittest discover -s reproduction/tests
python3 -m http.server --directory site 8094
```

References for paper fallbacks: [MOSAIC](https://dl.acm.org/doi/10.1145/3831643), [One Model to Fit Them All](https://dl.acm.org/doi/10.1145/3749509). Comparison-baseline paper numbers are attributed to the paper that evaluated them, rather than claimed as independent reruns.
