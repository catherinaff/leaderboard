# Source and asset notices

This tree publishes our adapters, runners, numerical checks and experiment metadata. Upstream author model files are obtained separately using fixed source commits and their original license notices. No blanket license grant for third-party repositories, datasets, LLaMA/Chronos/CLIP weights or motion/body models is implied.

Dataset and checkpoint providers retain their respective terms. Dataset links, SHA256 pins and original repository citations are recorded in `configs/`. Trained model states, downloaded upstream trees, raw datasets, paper PDFs, API credentials and SSH keys are excluded from Git. Content-addressed source objects preserve the exact authored execution snapshots; original third-party TensorFlow models and the HART UCI loader are referenced by digest instead of re-vendored.

Historical or retired files document earlier attempts and are not labelled as current, exact reproductions. Method-specific deviations and unestablished pretraining provenance remain part of every associated result.

The historical `legacy/calibration/run_deepsense_hhar_tf1_official.py` is a Python 3 calibration derivative of the original DeepSense TensorFlow code, with added reporting controls. Its author/source attribution is the pinned DeepSense entry in `configs/upstream_sources.json`; it is not a newly authored model or a current ranking runner.
