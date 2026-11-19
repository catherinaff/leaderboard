# oneHAR training reconstruction

Primary paper located at `/Users/pengrf/Downloads/one model to fit them all.pdf`.
DOI: https://doi.org/10.1145/3749509. Author repository snapshot:
https://github.com/ZJU-wqx-2025/uniHAR, commit
`e98bc56bbd575a61042d85825f4de85f07c594db`.

The release's inference model and five released test sets were already evaluated
in `experiments/onehar_20261028`. Preserve these as released-checkpoint evidence.
They are neither newly trained five-seed models nor canonical subject-held-out
results. The released test sets use the paper's stratified random split, not this
benchmark's subject splits. Model pretraining membership cannot be inferred from
the weights.

The repository's `train.py` supplies generic Trainer helpers. It does not supply
the complete CDNet joint-training, GPT-4o prompting, T2M-GPT/IMU simulation and
DQA/DPS/PDS augmentation workflow described by the paper. Its original synthetic
corpus is absent. The full method therefore cannot be replaced by a linear head
on the released checkpoint and called a clean reproduction.

`onehar_paper_components.py` reconstructs Eq.18 alignment, Eq.19–20 coarse-category
contrastive loss, Eq.21 gradient-reversal domain training, Eq.22 weighted objective,
Eq.23–26 DTW/cosine/joint-limit quality components, Eq.27–29 position selection,
Algorithm1 and proportional class balancing. It takes genuine generated motion/IMU
and LLM outputs as inputs; it does not manufacture those inputs.

Paper recipe: frozen bert-base-uncased text encoder, batch256, 80 epochs, Adam
with an unspecified phase boundary between lr .001 and .0001. Position projection
uses bidirectional GRUs with reported sizes18/72; IMU projection uses three 3×3
convolutions and a768-dimensional head. GPT-4o, T2M-GPT and virtual-IMU generation
are method components. Their outputs/versioned weights/prompt manifests must be
obtained or regenerated before full training.

Unresolved paper details must not be silently invented:

- Numerical alpha/beta coefficients for Eq.22 and the phase boundary are absent.
- Eq.26 rewards a larger quality score, but includes positive DTW distance and
  joint-violation penalties. Normalization and coefficient signs are unspecified.
- Eq.27 normalizes sensor contributions to sum to1, while Algorithm1 uses a .5
  contribution threshold. This can select only one position for most multi-sensor
  inputs. Raw versus normalized thresholding requires a recorded interpretation.
- Four coarse motion categories omit some native PAMAP2 activities and all17
  OPPORTUNITY gestures. Defining new categories is an extension, not an exact
  reproduction of the paper's 9-activity ontology.
- The original paper uses joint training across five datasets including RealWorld
  and USC-HAD, plus generated training samples. That training budget differs from
  the current single-dataset full-supervised track. Keep its track distinct.

Full paper model training remains disabled until these inputs/configuration and
the evaluation track are resolved. Existing frozen-weight measurements remain
visible; no newly trained score is substituted for the missing experiment.
