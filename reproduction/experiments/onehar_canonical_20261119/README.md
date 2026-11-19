# oneHAR / CDNet supervised adapter

This experiment uses the canonical6/12/12/8 classes and exact common split files. Author CDNet trainable blocks are newly initialized; only the frozen HHAR LIMU and BERT text encoders are retained. LLM synthesis, DQA/DPS/PDS, joint training and category/domain losses are omitted. Per-sample pooling normalization is corrected and RealWorld supports7 positions. The frozen branch cache is checked against the explicitly patched author path per fold.

Use ../../scripts/run_benchmark.py --method onehar with separately fetched assets. Its shared runner caps at80 epochs with validation-only early stopping. Results are provisional until all5 seeds complete. The source server publisher is deployment-specific and is not part of this code-only bundle. See ../../methods/onehar/README.md.
