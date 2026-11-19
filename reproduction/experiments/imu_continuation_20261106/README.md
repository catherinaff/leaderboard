# Canonical four-dataset runners

Use ../../scripts/run_benchmark.py from the reproduction root. train.py handles the shared Torch adapters; train_tf.py handles unchanged author TF models; sensorllm_adapter.py and onellm_adapter.py are standalone native routes. Source folders are populated from fixed upstream pins; data and licensed assets are prepared separately. See ../../README.md, ../../docs/IMPLEMENTATION_REVIEW.md and ../../configs/training_recipes.json. Historical infrastructure helpers retain original conventions and are not a portable deployment configuration.
