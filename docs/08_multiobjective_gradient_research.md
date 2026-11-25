# Research design: molecular prior, synthesis and off-target gradients

**Superseded training direction, 28 September 2026:** the current user preference
rejects training new in-house chemistry priors/scorers, including denoiser,
energy-critic, synthesis-surrogate and activity-panel models. Only small
supervised input bridges/representation decoders may be trained when needed,
then frozen alongside external pretrained models. One shared canonical molecular
graph must feed every branch. Earlier in-house training suggestions below are
historical, not active instructions; source022/024 failures remain recorded.
The current modular design is [docs23](23_frozen_model_plugin_contract.md).

Reviewed 27 September 2026. **Research proposal only:** no adapter below is
implemented or numerically validated by this document, and no new HPC job was
submitted for it. The running full-data AE/VAE comparison remains unchanged.
Existing evidence establishes a fixed-occupancy, fixed-pose Boltz gradient path;
it does not establish useful discrete optimization or variable-size Boltz scoring.

## 1. Recommended direction

Combine a molecular-distribution prior, a synthesis-related predictor and a
defined countertarget panel, while retaining explicit size and feasibility
constraints. These objectives address different failure modes. A realistic graph
can lack a recoverable route; a route-compatible molecule can bind unwanted
proteins; a high continuous score can correspond to an unchanged or invalid hard
graph. Multiple gradients do not guarantee that optimization stays meaningful.

The preferred first experiment is a **DeFoG-like denoising prior, a soft-graph
synthesis surrogate audited by a real planner, and a small Boltz countertarget
panel**, supplemented by inexpensive broad activity prediction. Investigate direct
SynFormer losses in parallel because they offer a plausible route to gradients
from that model itself. Keep every component frozen during molecular optimization
after its separate training and validation.

| Component | Proposed gradient into the shared latent | Main unresolved issue |
|---|---|---|
| Molecular prior | `z → decoder → graph adapter → denoiser consistency` | Representation compatibility and validity of the surrogate objective |
| Synthesis surrogate | `z → decoder → graph predictor → route-related score` | Generalization to optimized molecules and planner-label bias |
| Direct SynFormer branch | `z → decoder → learned condition adapter → fixed-route loss` | Condition alignment and product consistency |
| Broad activity panel | `z → decoder → soft-graph multitask heads` | Calibration, missing labels and chemical coverage |
| Focused Boltz panel | `z → decoder → per-protein adapter → binding outputs` | Fixed poses/occupancy and cross-target calibration |

All arrows must be real tensor operations. SMILES generation, RDKit conversion,
fingerprint hashing, catalog lookup and route execution belong outside this
gradient path unless replaced by an explicitly validated surrogate.

An alternative is a score-function/REINFORCE estimator of expected reward from
hard samples. It differentiates the sampling distribution, not RDKit or the
planner, so nondifferentiable oracles can still guide parameters. High variance,
invalid graph samples and expensive oracle calls make it a secondary baseline
here. Gumbel or straight-through estimators introduce their own bias and do not
automatically bridge an external chemistry program.

## 2. DeFoG: use its denoiser, not an assumed molecular score

DeFoG supplies a graph denoiser accepting floating node/edge tensors and time;
its graph sampler uses categorical sampling. It has no ready-made scalar
"molecule realism" head. A differentiable forward pass through the denoiser is
therefore distinct from differentiating its full discrete generation procedure.
[Paper](https://arxiv.org/abs/2410.04263),
[pinned implementation](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/graph_discrete_flow_model.py).

Proposed experiment: obtain decoder probabilities `q`, construct a differentiable
corruption `C_t(q)` at a few fixed noise levels, and compare denoiser predictions
with `q` using a declared categorical divergence. Record the corruption rule,
time convention, weighting and exactly where gradients are stopped. This would
be an **approximate denoising-consistency regularizer**, not graph log likelihood,
an exact score-function estimator, or a validity guarantee. Soft corruption is a
proposed relaxation of native categorical corrupted samples, with a different
input distribution; it is not an established exact SDS objective. Fixed noise draws
make derivative checks reproducible; resampled draws can later test robustness.

The representation mismatch is substantive. The ZINC implementation uses
kekulized bonds with four edge categories; GuacaMol uses five categories including
aromatic bonds and an element vocabulary covering ours, with explicit remapping.
Neither supplies our
formal-charge head or learned `NO_ATOM` semantics. Node count is fixed within a
sampled trajectory; an absorbing corruption token is not atom absence.
[ZINC codec](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/datasets/zinc_dataset.py),
[GuacaMol codec](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/datasets/guacamol_dataset.py).

GuacaMol is the more natural initial fixed-occupancy compatibility study. A
full charge/occupancy-aware prior needs explicit training for those semantics;
an index-remapping adapter alone cannot create the missing support. In particular,
silently discarding charges or equating padding with a noise token is not valid.
Checkpoint-dependent molecular extra features also require an explicit policy:
their soft extensions can differ from features of any realizable molecule.
[Molecular feature implementation](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/models/extra_features_molecular.py).

Those molecular features use `argmax`, blocking their contribution to input
derivatives. Structural extras also matter: cycle-count identities and assertions
may fail for fractional adjacency, while optional component/eigenvector features
contain discrete decisions and unstable derivatives at repeated eigenvalues.
Every adapter must declare which channels have gradients and whether auxiliary
features are fixed context, softened, or replaced through retraining. A gradient
through the main network is not a gradient through all native preprocessing.
[Structural feature implementation](https://github.com/manuelmlmadeira/DeFoG/blob/365bda9affadd5c2307014a0532ddaa244399441/src/models/extra_features.py).

Denoiser self-consistency can reward an unhelpful fixed point or overly diffuse
probabilities. Test whether it improves raw hard-graph validity under optimization,
not merely whether its own loss falls.

## 3. SynFormer: a differentiable training loss is the promising entry point

The released encoder-decoder configuration uses a SMILES encoder. A graph encoder
exists in the repository, but a compatible pretrained graph checkpoint has not
been verified in this review. The released weights therefore cannot simply be
declared to accept our relaxed graph.
[Configuration](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/configs/dev_smiles_diffusion.yml),
[graph encoder](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/synformer/models/encoder/graph.py),
[official weights](https://huggingface.co/whgao/synformer/tree/main).

A candidate research path is to learn a graph-to-condition adapter against the
released encoder's continuous conditioning representation. For a fixed,
teacher-forced route, SynFormer's `get_loss` can backpropagate through that
condition. Its fingerprint head's training loss denoises a noised target
fingerprint; evaluating this loss avoids unrolling the reverse diffusion chain.
The stock loss defaults to 32 independently sampled noise levels in a batch;
one or a few fixed levels would be our gradient-verification variant.
It remains an empirical training loss, not exact probability of a purchasable
catalog building block. Full generation still contains discrete route choices,
nearest-neighbor lookup and chemistry execution.
[Conditional losses](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/synformer/models/synformer.py),
[fingerprint loss](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/synformer/models/fingerprint_head/diffusion.py).

The decisive issue is **which route the loss rewards**. A model can confidently
predict a route whose product differs from the decoded molecule. Summing a
normalized conditional probability over all routes is uninformative: it is one.
Instead, audit a product-consistent route subset, or independently recover and
execute a route for the hard candidate. A fixed-route objective may keep a
candidate near one seed product rather than establish general synthesizability.

Train the condition adapter on a separate training split of molecule-route pairs,
evaluate on held-out pairs, verify teacher-condition parity, and test its
off-manifold behavior. Distillation error
and conditional route error must remain visible separately. A soft-graph
surrogate trained on planner outcomes is a simpler first branch, but planner
failure means "not found under this search budget," not chemical impossibility.

## 4. Newer alternatives and their actual relevance

PrexSyn uses a descriptor-conditioned autoregressive synthesis representation;
current documentation describes ECFP4/FCFP4 conditioning, pathway execution and
product ranking. The pinned descriptor module contains a floating MLP, and
teacher-forced token losses offer a possible condition-gradient path without a
fingerprint diffusion chain. However, RDKit/engine fingerprints remain discrete:
our graph needs a learned descriptor adapter, with the same product-consistency
problem as above. Its routes establish constructibility under a configured
catalog and templates, not experimental success.
[Official concepts](https://prexsyn.readthedocs.io/en/latest/concepts/),
[descriptor module](https://github.com/luost26/prexsyn/blob/3522e97475387ab8da7858a0d7e3070157c7eb4e/prexsyn/models/embeddings/descriptor.py),
[route outputs](https://github.com/luost26/prexsyn/blob/3522e97475387ab8da7858a0d7e3070157c7eb4e/prexsyn/models/outputs/synthesis.py).

ReaSyn v2 combines SMILES-conditioned autoregressive route modeling with route
editing. It is a candidate route oracle or projection baseline, not an established
gradient adapter for our graph. SynCoGen jointly models reaction graphs and 3D
coordinates; building blocks allow variable atom totals, but block count is
sampled upfront. Adopting it would be a larger architectural change.
[ReaSyn model card](https://github.com/NVIDIA-BioNeMo/ReaSyn/blob/reasyn_v2/model_card/overview.md),
[SynCoGen v2](https://arxiv.org/html/2507.11818v2#S4.SS2.SSS3).

FragFM's fragment autoencoder and discrete flow matching, with released
checkpoints, make it a useful structured-generation baseline or alternative prior;
it is not automatically a differentiable graph-quality scalar. VFMol is closely
related latent/discrete-flow work worth comparing, but the publisher full text
was unavailable in this review; no superiority or detailed equivalence is claimed.
[Official FragFM repository](https://github.com/lee-jwon/FragFM),
[official VFMol repository](https://github.com/DHUDBlab/VFMol),
[VFMol publisher record](https://doi.org/10.1021/acs.jcim.5c03005).

## 5. Off-target avoidance without a predetermined target

Use two levels. First, a cheap multitask QSAR panel covers documented human
protein or functional endpoints with adequate measured positives, negatives and
chemical coverage. Once an intended target is chosen, select a small structural
countertarget panel using relevant homologs, binding sites and justified liability
targets. There is no universal list implied here, and absence from the panel is
not evidence of safety.

ChEMBL's official multitask predictor is a practical fingerprint-based screening
or distillation baseline. Its thresholds vary by target family, and unknown
labels are masked. Its fingerprint interface has no native gradient to our graph.
[Official model](https://github.com/chembl/chembl_multitask_model).

For direct gradients, train a shared-schema soft-graph multitask model, or a
separately validated activity head on frozen molecular latents. Standard Chemprop
and DrugBAN interfaces still build discrete graphs; neural weights alone do not
make graph connectivity differentiable.
[Chemprop graph interface](https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/featurizers/molgraph/molecule/index.html),
[DrugBAN loader](https://github.com/peizhenbai/DrugBAN/blob/main/dataloader.py).

Curate ChEMBL/BindingDB by endpoint, species, target construct, assay provenance,
units and censoring. Missing measurements are unknown, not inactive. Keep
scaffold/temporal validation and target-wise calibration separate from training.
Our stereo-free representation can merge experimentally distinct activities;
such collisions need explicit handling or an expanded schema.
[ChEMBL endpoint definitions](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions),
[BindingDB paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701568/).

For Boltz, use the binder-versus-decoy output for initial avoidance. Official
documentation restricts the continuous IC50-like output to comparing active
molecules; pushing its countertarget value upward is not established nonbinding
prediction. Each protein needs its own pose/context and calibration.
[Official affinity guidance](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md).

Retain an on-target activity requirement and explicit per-countertarget ceilings.
A smooth maximum can prevent one strong liability being hidden by many benign
scores; an average or unconstrained selectivity difference can conceal failure.
Smooth calibration can preserve gradients, but calibrated hard-molecule scores
need not remain calibrated on optimized soft graphs. Ensemble disagreement is a
useful warning, not a guarantee.
[Calibration implementations](https://chemprop.readthedocs.io/en/latest/tutorial/cli/predict.html).

## 6. Size, constraints and the combined objective

The proposed objective combines on-target performance, countertarget penalties,
prior consistency and route-related guidance. Use explicit feasible atom-count
bounds and a trust region around validated seeds. The current reconstruction
size loss is not design-time size control, and `Nmax=50` alone permits optimization
to exploit the ceiling. Expected occupancy and the hard decoded count both need
checks. Variable occupancy through Boltz remains unimplemented.

Estimate gradient magnitudes and conflicts before setting weights. Reject invalid
or unsupported steps and reduce step size when hard scores diverge from continuous
ones. A learned prior or synthesis penalty can itself be exploited; adding terms
does not remove the need for acceptance checks and stopping criteria.

## 7. Validation plan and success criterion

1. **Interface parity:** on hard molecules, reproduce native denoiser inputs,
   conditioning features and predictor outputs. Validate ordering, charge,
   aromaticity, occupancy, route products and checkpoint/catalog identities.
   Never claim equivalence by silently dropping unsupported information.
2. **Derivative evidence:** freeze weights and random draws; check finite,
   nonzero chemical/latent derivatives for supported channels against multiple
   central finite-difference steps. Record unsupported or intentionally frozen
   channels, all errors, fixed-state hashes and memory. Numerical agreement
   validates implementation, not biological or chemical usefulness.
3. **Projection audit:** save each latent, soft graph, hard graph and score.
   Report validity without repair, exact molecule changes, size, relaxed/hard
   score gaps and fresh hard-molecule rescoring. A lower soft loss with unchanged
   hard chemistry is not design improvement.
4. **Matched ablations:** compare target-only, plus prior, plus synthesis, plus
   countertargets, and the combined objective under matched seeds and oracle
   budgets. Include direct SynFormer-loss and planner-surrogate branches separately;
   record their training cost and conditioning errors.
5. **Joint outcomes:** report the fraction of all attempted candidates that are
   valid, satisfy declared target/countertarget thresholds, and have an independently
   executed route producing the candidate. Separate exact route recovery from
   synthesis of a similar analog. Also report novelty, diversity, coverage,
   chemical failure types and size; do not discard failures from denominators.

A computationally recovered route is still not laboratory synthesis, and predicted
selectivity is not an experimental safety result. The experiment should establish
whether each gradient contributes to better discrete candidates before expanding
the model panel or making stronger claims.

## 8. Addendum: alternative priors and gradients through sampling

Research follow-up, 27 September 2026; no implementation, experiment or HPC job
is added by this section.

**Alternative priors.** DiGress is another categorical graph denoiser and inherits
the broad interface questions above. GDSS learns continuous node/edge score fields,
which are conceptually closer to a gradient prior, but its encoding and noise
distribution still need alignment with our graph representation. FragFM supplies
a fragment hierarchy and released autoencoder/generator checkpoints; adopting it
requires architectural changes rather than replacing one scalar loss.
[DiGress](https://arxiv.org/abs/2209.14734),
[GDSS](https://arxiv.org/abs/2202.02514),
[FragFM](https://github.com/lee-jwon/FragFM).

Our proposed best fit is a small denoiser trained on the training split using
exactly our element, charge, bond and `NO_ATOM` schema, including declared noise
and decoder-like inputs. Freeze it before optimization and test whether it
improves hard candidates. This avoids unsupported vocabulary mappings; it does
not guarantee validity or a useful prior. DeFoG still requires a defined denoising
reward, not an assumed native realism scalar or exact likelihood. Sampling cannot
create missing charge support.

**Three different estimators.** With fixed random draws, ordinary categorical
sampling has zero pathwise derivative almost everywhere; its expected reward
can nevertheless change with the probabilities. Soft Gumbel-Softmax provides
pathwise gradients for a relaxed objective. Straight-through (ST) Gumbel uses a
hard forward sample with a soft backward surrogate, so its gradient is biased.
Neither turns RDKit, SMILES construction or Boolean atom-count masks into
differentiable operations. A sampled hard graph is not necessarily chemically
valid. [Gumbel-Softmax paper](https://arxiv.org/abs/1611.01144),
[PyTorch estimator semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.gumbel_softmax.html).

For a fixed oracle reward `R(H)`, the score-function alternative is:

`grad_z E[R(H)] = E[(R(H) - b) grad_z log p_z(H)]`.

It requires no derivative through the scorer and can cover sampled atom-count
changes. This does not implement variable occupancy in our current Boltz adapter.
Use the actual sampling law, including masking or conditional choices;
do not assign an independent-product likelihood to a different sampler. A
variance-reduction baseline must be independent of the current sample, conditional
on `z`, or use a valid leave-one-out construction, with its gradient treatment
specified. Invalid samples and oracle failures remain in the objective with
declared penalties; discarding them changes the sampled objective. High variance
and oracle cost are the main practical obstacles.
[Black-box gradient/control-variate methods](https://arxiv.org/abs/1711.00123).

**Exploratory pilot.** Start with a tensor-only ST branch at fixed occupancy using
8–16 samples, and a separate REINFORCE branch using 16–32 samples. These counts
are provisional: tune them by gradient signal, variance and measured cost. Sample
each undirected bond once and mirror it; gate bonds at absent endpoints. Keep
each pair's likelihood once, without counting its mirrored edge twice. Avoid
applying absence gating again if the sampled bond probabilities already include
it. Keep size limits, trust regions and hard-candidate acceptance checks.

The validation rule differs from Section 7's ordinary pathwise check: ST's soft
backward gradient should **not** be expected to match finite differences of its
fixed-noise hard forward function. Check the soft proxy separately. For the
expected-reward estimator, compare against exact enumeration on tiny categorical
examples and repeated Monte Carlo estimates, then measure hard validity, oracle
cost and useful discrete changes. These tests distinguish a working estimator
from an apparent improvement confined to relaxed scores.

## 9. Addendum: density or distance in molecular latent space

Research follow-up, 27 September 2026. This is a proposed prior baseline, with no
code change, fitted density, experiment or HPC submission. It can precede training
a new denoiser; it is not established as the best molecular prior.

**Start with our own encoder.** Freeze the selected encoder and decoder while
optimizing `z`. Encode training molecules as 128-dimensional reference points
`z_i`. Fit a Gaussian KDE,
`p(z) = (1/N) sum_i Normal(z; z_i, h² I)`, or compare a shrinkage Gaussian mixture
and local Mahalanobis distance. A tensor implementation supplies gradients
directly into `z`, without categorical sampling or differentiating molecule
conversion. A candidate regularizer is
`L_prior = max(0, tau - log p(z))²`, with `tau` calibrated on held-out validation
data. This penalizes sufficiently low density instead of rewarding unlimited
movement toward a mode, which can reduce novelty and diversity.

Both AE and VAE codes support empirical fitting. For the VAE, distinguish its
nominal `N(0,I)` prior, the distribution of encoded means and the aggregate
posterior obtained by mixing molecule-conditioned distributions. They need not
match. The origin maximizes standard-normal density, but this is not evidence
that decoding it gives a valid or representative molecule. High density and
typicality differ even in well-specified high-dimensional distributions.
[Typicality study](https://arxiv.org/abs/1906.02994).

KDE in 128 dimensions is statistically demanding: bandwidth and feature scaling
can dominate the result. Fit whitening or optional PCA on training codes only;
if dimensions are discarded, retain an appropriate residual-distance check so
optimization cannot exploit an unpenalized direction. Calibrate AE and VAE
thresholds separately rather than comparing their raw density values.
[Density-estimation guidance](https://scikit-learn.org/stable/modules/density.html).

**Pretrained hidden features are another possibility.** Define a differentiable
path such as `h(z) = pool(F(adapter(decoder(z)), t))`, then estimate density or
distance against reference features computed with the same model, pooling,
noise-level and conditioning protocol. Fix random draws for gradient checks.
Permutation-invariant pooling is needed when atom ordering is arbitrary.
Our `z` cannot be inserted directly into another model's latent coordinates:
an explicit projection or learned adapter is required. RDKit conversions and
missing charge support remain obstacles, even if the final density is smooth.

Mahalanobis distances in pretrained features have precedent for detecting
out-of-distribution inputs, but that evidence does not validate molecular
feasibility in our setting. Feature collisions can place invalid and valid graphs
near one another; representation rescaling or reparameterization changes density.
Consequently, feature density is neither molecular probability nor a validity
certificate, and optimization may exploit it.
[Feature-distance precedent](https://arxiv.org/abs/1807.03888).

**Validate before adding Boltz.** Fit references on the training split, tune
bandwidths/transforms/thresholds on validation, and preserve the sealed test.
Perturb held-out encoded molecules and measure whether the prior predicts and
improves raw decoder validity while permitting actual molecule changes. Compare
against size constraints plus a simple latent trust region. Report exact
reconstruction, unchanged outputs, novelty, diversity and validity by molecule-size
bin; density improvement alone is insufficient. Constraints on latent exploration
have previously reduced invalid molecular proposals, but that result is not
evidence for this particular KDE or encoder.
[Constrained molecular optimization](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c9sc04026a).

Keep the prior as one calibrated restraint alongside chemical checks. Dense
regions can still contain decoder errors, and valid unfamiliar chemistry can
occupy sparse regions.
