# RobustHAR paper reimplementation audit

Source: Xiao Liu et al., IJCAI 2025, pp. 8563–8571, DOI 10.24963/ijcai.2025/952.
Downloaded from the proceedings PDF on 2026-10-28. Method figure and equations on
PDF pages 3–4 were rendered for inspection. This implementation is not author code.

Paper-specified components implemented: position-ordered 3-D sensor layout (Eq.1),
segment/channel/sensor masks (Eq.2–3), three Transformer reconstruction branches
(Eq.4–5), consistency of completed observations (Eq.6–7), frozen pretrained
encoders and correlation/cross-attention feature fusion (Fig.2 and Eq.8–11).
The source calls both encoders and decoders one-layer Transformers with eight heads.
Pretraining: Adam lr .001, batch 64, max 200 epochs. Classification: lr .001,
batch 32, max 100 epochs. Opportunity defaults lambda=.5 and masking=.3.

Unspecified choices, exposed rather than asserted as original settings:

- Embedding width 128, feedforward width 512, dropout .1, sinusoidal positions,
  ReLU and temporal tokenization; none of these dimensions/settings is supplied.
- The numerical position matrix A is absent. Use a fixed anatomical ordering.
  Preserve every native sensor channel. Mixed OPPORTUNITY sensor widths are padded
  to a common width, with nonexistent channels excluded from reconstruction loss.
- Cosine *distance* implements Eq.6: minimizing positive cosine similarity as
  literally printed would encourage semantic disagreement, contrary to the prose.
- Eq.9 divides each feature by the other feature's norm. Retain that cross-normalized
  self-Gram construction; its norm axis and all-three-branch reduction are unspecified.
  Eq.10 uses the correlation matrix as query and that same branch's feature as
  key/value. Eq.11 uses matrix multiplication, followed by mean reduction of
  correlation-query rows and pairs. This replaced the initial non-literal
  feature-query/elementwise-product variant, which is archived and superseded.
- The figure shows frozen encoders during fine-tuning, so freeze them. Reuse one
  distinct attention modules for the two branches, following Fig.2's two boxes.
- Reconstruction uses masked real entries, not padded measurements. Validation
  masks are deterministic and checkpoint selection never reads test metrics.
- Single-phone datasets have one physical sensor; inter-sensor masking then has
  a degenerate interpretation. Keep this limitation on their results.

Standard runs use full labels and subject splits. They are NOT the paper's 15%
labels, random 8:1:1 split, five-device OPPORTUNITY protocol. The paper does not
publish its exact 8,165-window index list or split granularity. A new canonical
score cannot establish paper calibration. Do not mark these runs calibrated or
comparable to the paper's 93.74 F1 until the paper-protocol audit passes.
