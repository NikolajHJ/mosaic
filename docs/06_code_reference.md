# Code reference and implementation semantics

This is a dated protocol/implementation narrative. For current commands and
package ownership use [workflows](workflows.md) and the [package map](../src/latentmol/README.md).
Historical outcomes and source-specific rules below retain their original scope.


Written from local source inspection starting on 2026-09-26 UTC. This reference
describes source-002 semantics, with later source-003 changes identified explicitly.
Source-002 has since passed 36 HPC tests and completed smoke training; its first
post-training subprocess evaluation failed, then independent recovery verified
16/16 exact training-subset reconstructions. Source-003 passed 39 regression tests
and completed both AE/VAE pilots and their same-process evaluations. Source-004
passed 40 tests and completed failure diagnostics. Source-005 changes only the
no-bond loss weight for a matched follow-up, now completed. Source-006 passed 49
tests and evaluated four VAE priors, with 0/0/1/0 valid draws out of 256 each.
The optional Boltz adapter is implemented; native preparation succeeded, while
its source-006 smoke failed during checkpoint loading. Source-008 passed 50 tests,
loaded correctly and matched native fixed-pose outputs exactly, then ran out of
GPU memory during the gradient-enabled forward. Source-009 resolved the memory
issue with activation checkpointing and passed its original engineering gates.
Source-010 passed 53 tests and the stronger audit: four finite-difference epsilons
within 2%, exact native parity, finite chemical/latent gradients and unchanged
frozen state. The valid hard molecule and its fixed-context scores stayed constant.
The numerical checks summarized in this historical section ran on DTU HPC.
The [pipeline](04_latent_pipeline.md) states scientific requirements; the
[ledger](05_experiment_ledger.md) states verified setup and execution history.

**1 October 2026 untested addition:** a separate active-set Boltz runner and
fresh-pose follow-up are described in [docs28](28_multimodel_experiments.md).
They were added after the source-010 results summarized here. They have no
local/HPC test, prepared count bank, submission or molecular result.

## 1. Code-to-stage map

| File | Responsibility | Stage |
|---|---|---|
| `src/latentmol/representation.py` | Versioned graph schema, hard/relaxed tensors, decoder probabilities and hardening | Shared contract |
| `src/latentmol/chemistry.py` | Canonicalization, graph encoding, lossless target round trip and raw discrete decoding | Dataset / evaluation |
| `src/latentmol/data.py` | Deterministic filtering, deduplication, subset/split, immutable cache and loader | Dataset |
| `src/latentmol/data_audit.py` | Verify expanded/parent cache hashes, exact role preservation and split disjointness from actual identities | Full-data pre-training gate |
| `src/latentmol/models/encoder.py` | Position-aware graph message passing and latent mean/log-variance | AE / VAE |
| `src/latentmol/models/decoder.py` | Latent-to-slot decoder with atom, charge and symmetric pair heads | AE / VAE; future latent optimization |
| `src/latentmol/models/autoencoder.py` | Shared model composition and optional latent sampling | AE / VAE |
| `src/latentmol/losses.py` | Atom/charge/bond/size reconstruction terms and Gaussian KL | AE / VAE |
| `src/latentmol/train.py` | Seeded training, validation/chemistry probes, immutable selections and epoch-boundary continuation | AE / VAE |
| `src/latentmol/full_training.py` | Staged per-model training, validation-only extension, sealed selection and final evaluation | Source015 GPU regressions passed; both full-data models completed epoch 1, later stages pending |
| `src/latentmol/diagnostics.py` | Adjacent-window continuation rule and saved training plots | Source015 checks passed, including plot refinements |
| `src/latentmol/train_and_evaluate.py` | Train once, then evaluate the returned run's best/last checkpoints and optional exact smoke subset | AE / VAE workflow |
| `src/latentmol/evaluate.py` | Raw reconstruction, local latent neighborhoods, and VAE standard-normal prior sampling | AE / VAE comparison |
| `src/latentmol/sampling.py` | Seeded subset selection without canonical-prefix bias | Training / evaluation |
| `src/latentmol/experiments.py` | Unique run directories, source/config/environment records and failure capture | All execution |
| `tests/test_chemistry.py` | Source rejection, ordering/stereo policy, round trip, cache and split contracts | Dataset verification |
| `tests/test_models.py` | Decoder invariants, gradients, loss semantics, AE/VAE backward path and tiny overfit check | Model verification |
| `tests/test_experiments.py` | Failed-run evidence, seeded subsets and checkpoint/schema behavior | Execution verification |
| `tests/test_evaluation.py` | Baseline-relative changes, rescue denominators, bounded decoding, precision/charge metrics and latent diagnostics | Evaluation verification |
| `configs/ae_smoke.yaml` | Small 16-graph memorization configuration | AE smoke |
| `configs/ae_pilot.yaml`, `configs/vae_pilot.yaml` | Matched pilot settings | AE / VAE comparison |
| `configs/ae_full.yaml`, `configs/vae_full.yaml` | Shared120 epoch horizon, absolute LR stages, full validation and fixed probes | Full-data comparison |
| `configs/ae_bond_balance.yaml`, `configs/vae_bond_balance.yaml` | Same pilots with no-bond class weight 1.0 | Source-005 loss intervention |
| `scripts/hpc/run_job.sh` | Require LSF allocation, activate environment and set source/runtime paths | HPC entrypoint |
| `scripts/hpc/validate_and_prepare.sh` | Run tests, then prepare ZINC5k only if tests pass | Validation / dataset |
| `scripts/hpc/setup_environment.sh` | Create/update the pinned environment and acquire/hash-check raw data on HPC | Setup |
| `scripts/hpc/ae_smoke.sh` | Run tests, then the explicit smoke training/evaluation workflow in a GPU allocation | AE smoke |
| `scripts/hpc/pilot.sh` | Select `ae` or `vae` YAML and run explicit training/evaluation | AE / VAE pilot |
| `scripts/hpc/validate_and_pilot.sh` | Run a new release's regression tests before its selected pilot | Source-003 pilot gate |
| `scripts/hpc/diagnose_checkpoints.sh` | Test source-004, then reevaluate explicitly supplied saved checkpoints with novelty/failure diagnostics | Evaluation only |
| `scripts/hpc/bond_balance_pair.sh` | Sequential matched AE and VAE follow-up | Source-005 |
| `src/latentmol/boltz/contracts.py`, `adapter.py`, `model.py` | Explicit fixed-context contract, soft chemical features, frozen native trunk/affinity path | Boltz feasibility |
| `src/latentmol/boltz/prepare.py` | Acquire pinned example and snapshot native prediction/features outside autograd | Boltz preparation |
| `src/latentmol/boltz/active_set.py` | Select decoded slot occupancy between conditional fixed-context Boltz gradient steps; compare against fixed-mask control and retain transition/candidate records | Untested Boltz active-set extension |
| `src/latentmol/boltz/active_set_rescore.py` | Freeze initial/final hard argmax endpoints and request matched fresh native poses for valid molecules | Untested independent model follow-up |
| `tests/test_boltz_active_set.py` | Selector tie, native slot rebinding, conditional occupancy gradient, fixed-control diagnostics and rescore endpoint selection contracts | Authored but unexecuted |
| `configs/boltz_active_set.example.yaml` | Hashed checkpoint/context-catalog, threshold and optimizer template | Untested active-set configuration |
| `scripts/hpc/boltz_active_set_prepare.sh`, `boltz_active_set_run.sh` | Per-count native preparation/catalog and allocated-job runner wrappers | Untested HPC interfaces; no submission |
| `src/latentmol/boltz/smoke.py` | Native parity, channel/latent derivatives, finite differences, bounded latent-only updates | Boltz engineering checks |
| `scripts/hpc/boltz_run.sh`, `boltz_prepare.sh`, `boltz_smoke.sh` | Separate environment and explicit preparation/check stages | Boltz feasibility |
| `scripts/hpc/stage_integration.sh` | Regression tests, saved-VAE prior evaluations, native preparation, then gradient smoke | Source-006 integration |
| `scripts/hpc/setup_boltz_environment.sh` | Successful explicit-Python-3.11.9 setup procedure with scratch cache and pinned assets | Boltz setup |
| `requirements-boltz-hpc.lock.txt` | Captured tested Boltz environment snapshot; bootstrap does not enforce every transitive pin | Boltz provenance |
| `pyproject.toml` | Package metadata and dependency ranges | Environment |
| `requirements-hpc.txt` | Pinned direct dependency versions for the HPC environment | Environment |
| `AGENTS.md` | Active scope, execution boundary, separate ownership and provenance requirements | Project instructions |

VAE prior sampling is implemented and exercised in source-006. A bounded
frozen-model latent probe passed source-009's original and source-010's stronger
engineering gates, without discrete molecular improvement. Synthesis/off-target
evaluators are not implemented capabilities. The later active-set runner
implements discrete occupancy reselection between fixed-context steps in
source, without a validated variable-count gradient or run. The separate
Boltz feasibility adapter is described below;
source-only interfaces do not establish useful latent optimization.
Source-002 was uploaded with a verified archive hash, passed 36 tests in 15.08
seconds and completed AE smoke training in job 29503123. The overall job then
failed in evaluation; see the ledger for separate training, evaluation and
scientific-acceptance states.

## 2. Graph contract and probability semantics

`GraphSpec` fixes vocabulary order and capacity and provides a JSON-compatible
schema plus SHA-256 fingerprint. Its defaults are 50 slots, atomic numbers
`(0, 6, 7, 8, 9, 15, 16, 17, 35, 53)`, charges `(-2, -1, 0, 1, 2)`, and bonds
`(NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC)`. Atomic-number zero means `NO_ATOM`.
Cache arrays store **category indices**, not atomic numbers or numeric charges.

`HardGraphBatch` holds integer tensors `atoms[B,N]`, `charges[B,N]` and
`bonds[B,N,N]`. Its `to_relaxed` one-hot expansion happens before the neural
network's continuous-input boundary.

`DecoderOutput` holds atom/charge/pair logits. The decoder's pair features are
already symmetric, and `symmetric_bond_logits` additionally averages both
orientations so loss and inference share one interpretation.

`probabilities(temperature)` produces a `RelaxedGraph`:

```text
atom_probs = softmax(atom_logits / temperature)
charge_probs = softmax(charge_logits / temperature)
conditional_bond_probs = softmax(symmetric_bond_logits / temperature)
occupancy_i = 1 - atom_probs[i, NO_ATOM]
pair_presence_ij = occupancy_i * occupancy_j, with diagonal set to zero
bond_probs_ij = pair_presence_ij * conditional_bond_probs_ij
               + (1 - pair_presence_ij) * one_hot(NONE)
```

Thus `RelaxedGraph.bonds` contains **occupancy-gated marginal probabilities**,
not probabilities conditional on both atoms being present. A future scorer must
not apply the same pair-occupancy factor again. Scorer-specific node masking and
pooling still need their own audited semantics. `expected_charge(spec)` returns
the categorical expected numeric charge multiplied by atom occupancy.

`harden(spec)` is deliberately separate from `probabilities`: it argmaxes atom,
charge and **conditional** symmetric bond logits, then removes absent-slot bonds,
sets the diagonal to `NONE`, and assigns neutral charges to absent atoms. Using
argmax of softly gated marginal bonds instead would change the discretization
policy by favoring `NONE`. Hardening is for discrete evaluation and is not used
inside a latent gradient path.

## 3. Chemistry and data preparation

`canonicalize_smiles` checks parsing, connectedness, capacity, element/charge/bond
vocabularies, isotopes and radicals. It explicitly removes stereochemistry. It
does not strip salts, neutralize charges, normalize tautomers or repair chemistry.

`encode_smiles` reparses canonical stereo-free SMILES to establish atom order,
constructs padded int8 arrays, and requires decoding to recover exactly that
canonical identity. This rejects molecules requiring omitted information such
as some explicit hydrogen counts. For example, the chemistry test file expects
the pyrrole-like input `c1cc[nH]c1` to fail this representation round trip; this
is a coverage limitation to audit, not a silent conversion to another molecule.

`decode_graph` accepts already-discrete integer arrays or tensors, ignores
`NO_ATOM` slots, requires active bonds to be symmetric and self-loop-free, builds
the graph, sanitizes it with RDKit, and rejects disconnected or radical products.
It returns canonical stereo-free SMILES or `None`. No largest-fragment selection,
charge correction or valence repair occurs. “Validity” here is this procedural
definition, not a synthesis or affinity claim.

Source-004 adds `DecodeResult` and `inspect_graph` while keeping `decode_graph`
as a SMILES-or-`None` wrapper. Diagnostics preserve the **first** failure observed
by the checking order: shape/category problems, empty graph, bond-label/symmetry
problems, unsupported bond, sanitization, disconnectedness, radicals, or input/
construction errors. These counts are not independent measurements of every
chemical defect in a graph. Evaluation records reason/detail and retains the
first eight failed hard graphs in seeded sample order, not a hand-picked ranking.
No repair or training change is introduced by this diagnostic revision.

`prepare_dataset` considers all source rows before selecting a subset:

1. Encode/round-trip each row; count explicit rejection reasons.
2. Deduplicate the resulting canonical stereo-free identities.
3. Sort identities, then draw a uniform subset without replacement using a
   NumPy generator seeded by `seed`, if `limit` is smaller than the accepted pool.
4. Using the same generator, permute selected molecules into 80/10/10 splits:
   `floor(0.8*M)` training, `floor(0.1*M)` validation, remainder test, with at
   least one training item. Very small datasets can have empty validation/test.
5. Write an immutable staged cache and rename it into its new destination.

The split is molecule-random, **not scaffold-separated**. Cache contents are
`dataset.npz` and `manifest.json`; the manifest includes source/data/split hashes,
schema fingerprint, counts, rejection reasons, chemistry-library versions and
preprocessing policy. It records the local source path; the acquisition URL and
license/source interpretation are recorded separately in the ledger.

`GraphDataset` verifies cache format, schema fingerprint, dataset hash, array
shapes/dtypes, split labels/digest and total count before selecting a split.
It returns independent long tensors. It does not rerun the source chemistry
audit on every load. `collate_graphs` stacks these into `HardGraphBatch`.

## 4. Encoder, decoder and AE/VAE difference

The encoder projects atom and charge probabilities linearly, adds learned slot
positions, applies occupancy, and passes through three message layers by default.
Messages use a separate linear transform for each non-`NONE` bond type, summed
with bond probabilities and normalized by expected bond degree clamped at one.
Residual updates use LayerNorm and occupancy gating. Flattened slot features
feed the latent heads, so the model is explicitly **not permutation invariant**.

The decoder maps a latent vector through an MLP to 50 hidden slot states and
refines them with LayerNorm and an MLP. Atom and charge heads act per slot.
The pair head receives `h_i + h_j`, `abs(h_i - h_j)` and `h_i * h_j`, making the
predicted pair features symmetric. Softmax is across each pair's bond categories,
not across neighboring nodes.

Both training modes instantiate exactly the same `GraphAutoencoder`, including
both `mu` and `logvar` heads. The encoder clamps `logvar` to `[-10, 8]`.
For the AE, `z = mu`; for VAE training,
`z = mu + exp(logvar / 2) * epsilon`. Validation and the current evaluation CLI
use `mu` for both modes. A KL diagnostic is computed even for the AE, but its
coefficient is zero and its variance head has no learned posterior interpretation.

## 5. Configuration and losses

`ModelConfig` defaults: `hidden=128`, `latent=128`, `layers=3`.
`LossConfig` defaults:

| Key | Default | Exact role |
|---|---:|---|
| `padding_weight` | 0.2 | Class weight for `NO_ATOM` in atom cross-entropy |
| `no_bond_weight` | 0.1 | Class weight for `NONE` in bond cross-entropy |
| `charge_weight` | 1.0 | Multiplier on charge cross-entropy |
| `bond_weight` | 2.0 | Multiplier on bond cross-entropy |
| `size_weight` | 0.1 | Multiplier on normalized expected-size MSE |

Atom CE covers all target slots using weighted cross-entropy. Charge CE is
averaged over target-present atoms. Bond CE uses only upper-triangle pairs for
which **both target atoms are present**, with weighted cross-entropy; padded pairs
and the diagonal are excluded. A batch with no eligible pairs has zero bond loss.
The size term is the mean squared error between predicted expected size and
target size, divided by `N_max` **before squaring**:

```text
reconstruction = atom_ce + charge_weight * charge_ce
                 + bond_weight * bond_ce + size_weight * size_mse
size_mse = mean(((predicted_expected_size - target_size) / N_max)^2)
KL = mean_over_molecules(sum_over_latent_dimensions(KL(q || N(0,I))))
total = reconstruction + beta_epoch * KL
```

This training size term encourages reconstructing the source molecule's size.
It does not implement design-time size control. A future binding-versus-size
objective remains separate, and the current Boltz probe fixes atom occupancy.

Source-005's two `*_bond_balance.yaml` configurations change only
`loss.no_bond_weight` to 1.0 from its 0.1 default. Compare decoded chemistry across
that intervention: changing the weighted CE/reduction means raw reconstruction
loss values across old/new settings are not on the same scale.

`TrainConfig` supplies these further defaults; a run's saved resolved
configuration takes precedence over this reference:

| Keys | Defaults / behavior |
|---|---|
| `mode`, `seed` | `ae`, 17 |
| `epochs`, `batch_size` | 60, 64 |
| `learning_rate`, `weight_decay` | 0.001, 0.00001; AdamW |
| `clip_grad` | 5.0; nonfinite gradient norm raises an error |
| `beta`, `kl_warmup_epochs` | 0.001, 20 |
| `train_limit` | `None`; otherwise a seeded random subset of the training split |
| `validation_limit`, `chemistry_limit` | 256, 128 |
| `num_workers`, `device` | 0, `cuda` |

For VAE epoch `e` starting at one,
`beta_epoch = beta * min(e / max(kl_warmup_epochs, 1), 1)`; for the AE it is zero.
The best checkpoint minimizes **validation reconstruction loss**, not total
VAE loss or decoded validity. Training runs for its configured epoch count;
there is no early-stopping or resume entrypoint at this inspection.

Pilot YAMLs use the defaults above with three 128-wide encoder layers and a
128-dimensional latent. `ae_smoke.yaml` instead uses 200 epochs, 16 training
graphs, batch size 16, learning rate 0.003, hidden/latent dimensions 64, two
message layers, 32 validation graphs and 16 chemistry checks. It is an overfit
diagnostic, not the model used for the matched 128-dimensional pilot comparison.

The coordinator's acceptance gate for the smoke run is finite losses/gradients
plus at least 15/16 exact, unrepaired reconstructions on the memorized training
subset at the last checkpoint. This is a scientific acceptance rule to apply to
the retrieved metrics; do not infer it from successful process termination.

`train_and_evaluate` passes the path returned by `train()` directly to evaluation
of `best.pt` and `last.pt` on validation. With `--smoke`, it also evaluates
`last.pt` on the same seeded training subset using `train_limit`. It does not
find a “latest” directory, so simultaneous jobs cannot silently swap checkpoints.

Source-002 launched those evaluations as subprocesses. This failed on the
exclusive-process A40 because the training parent retained a CUDA context.
Source-003 now calls `evaluate_checkpoint` directly within the training process,
sharing its CUDA context while retaining separate evaluator run records.
Exceptions propagate after the evaluator records its traceback. The independent
recovery evaluation passed; end-to-end verification of the revised orchestration
completed in source-003 job 29503159, including both checkpoint evaluations.

Logged epoch component losses are molecule-weighted means of per-batch losses.
Because several component losses normalize within a batch by weighted classes,
active atoms or active pairs, those logged values are not exactly a single
corpus-level weighted CE. Keep batch size and the comparison procedure fixed
and state this aggregation when interpreting differences.

## 6. Evaluation semantics

`reconstruction_metrics` reports mean reconstruction components, active-atom and
charge accuracy, bond-existence and bond-type precision/recall, discrete
atom-count MAE, raw validity and exact canonical reconstruction. The numerator
and denominator counts are retained. Charge accuracy requires the predicted
slot to remain occupied but does not require its element category to be correct;
element accuracy is separate. Bond precision includes all predicted edges,
including edges involving extra predicted atoms.

Validity and exactness use `chemistry_samples`, which may be smaller than the
number of tensor-evaluated graphs. Undefined fractions have `None`/JSON null,
not a fabricated zero or perfect score. Optional novelty compares valid outputs
with the full prepared training split, which may exceed a smoke model's actual
training subset.

Latent diagnostics report population variance of `mu` across evaluated molecules,
with active dimensions defined by variance strictly greater than 0.01 by default.
This is a diagnostic convention, not a universal activity threshold. Posterior
KL and standard deviation are interpreted only when `training_mode='vae'` is
explicitly supplied; AE/unspecified variance-head outputs are not called a
trained posterior. The standalone evaluator derives the mode from the checkpoint.

`neighborhood_metrics` encodes a limited seed set, measures the RMS latent
standard deviation, floors the perturbation reference scale at `1e-3`, and adds
the same CPU-seeded Gaussian direction at relative scales `0, 0.05, 0.1, 0.2`.
It decodes in batches of at most 32 to bound memory. Each output is compared both
with the original input and with that input's own unperturbed decode. This
distinguishes preexisting reconstruction errors from changes caused by the
perturbation. Metrics include validity, uniqueness among valid outputs,
valid changes from baseline, and invalid-baseline-to-valid rescues; denominators
are explicit. This measures local variation, not property improvement.

Source-006 adds `prior_metrics`: for VAE checkpoints only, draw CPU-seeded
`N(0,I)` codes without encoding a graph, decode in bounded batches, and report
validity over all attempted draws. Unique-valid and optional training-novelty
fractions use valid draws as their denominator; zero-valid conditional fractions
are undefined. `prior_examples.json` retains every decoded draw, including invalid
ones. The evaluator uses the same `limit` and seed for prior draws as its other
evaluations, reports prior metrics separately, and stores `prior: null` for AE
checkpoints. It does not claim that the AE has a standard-normal sampling prior.

The evaluation CLI defaults to validation, limit 256, seed 17 and CUDA, with
optional `--novelty`. It checks
checkpoint/data schema and exact dataset hash; evaluating on a different dataset
is not currently supported through this entrypoint. It writes `summary.json`,
`examples.json` and `neighbors.json` as well as the standard run records.

Source-002 replaces the earlier canonical-prefix limits with
`sampling.subset_indices`. Capped selection uses NumPy's seeded sampling without
replacement and retains random order, so a further chemistry prefix is also
randomized. Uncapped/full-size selection returns all original indices. Selected
training, validation and evaluation indices are recorded. Source-002's subset
selection tests were included in the 36 passing HPC tests.

## 7. Run records and known provenance gaps

`ExperimentRun` creates a new directory named
`YYYYMMDDTHHMMSSZ_<name>_<8-hex-suffix>`. `run.json` stores resolved configuration,
timestamps, status, source file hashes, package versions, hostname, Python and
platform details, LSF job ID if present, and `LATENTMOL_SOURCE_SHA256` if supplied.
Source-002 also records a SHA-256 of the sorted resolved configuration
JSON and `LATENTMOL_GIT_COMMIT` when supplied. A missing Git environment value
remains null; it does not reconstruct a commit from the source hashes.
Source-003's `run_job.sh` derives `LATENTMOL_SOURCE_SHA256` from the immutable
archive at `LATENTMOL_SOURCE_ARCHIVE` or `${PWD}.tar.gz` inside the allocation,
and refuses to run if that archive is missing. This avoids reliance on LSF
preserving an exported hash from the submission shell. The coordinator verified
that the running source-003 AE record contains its correct archive SHA-256;
the earlier null archive fields remain unchanged in the history.
Metrics append to `metrics.jsonl`. Normal Python exceptions produce a failed
status and `traceback.txt`. Abrupt scheduler termination can leave the last
recorded state as running; reconcile it with scheduler evidence in the ledger.

The historical source010 trainer wrote `best.pt` and `last.pt`, with model/optimizer state, graph and
model/loss/train configurations, epoch, dataset hash, run ID and Torch/CUDA RNG
state. Source-002 includes the effective `beta_at_checkpoint`. That historical implementation did
not save Python/NumPy/DataLoader RNG state for exact resumption,
and had no exact resume capability. The new full-data trainer's versioned continuation
state is described below; CPU and actual CUDA training/resume checks passed. Hash final checkpoint files when recording
their identity; evaluation records the hash of its input checkpoint.
The evaluator reads `beta_at_checkpoint`, with a legacy `beta` fallback and
schedule reconstruction when neither saved value is present; the metadata labels
whether the effective coefficient was saved or reconstructed.

The generic command wrapper captures combined stdout/stderr into `command.log`,
records return code and marks nonzero exit as failure. The source manifest hashes
files under `src`, `tests`, `configs`, `scripts`, plus `pyproject.toml` and
`requirements-hpc.txt` when present. It does not by itself archive file contents
or discover a Git commit. Preserve the executed release archive and its hash.

`requirements-boltz-hpc.lock.txt` captures the environment that was tested. The
current Boltz bootstrap pins selected direct packages, source commit and model
assets, but does not restore every transitive package from this snapshot. A fresh
bootstrap can resolve different transitive versions. Its new environment freeze
must be retained and compared, followed by numerical checks on HPC; the existence
of this snapshot alone is not a full environment-reproduction guarantee.

Queue/resource request, array index, driver details, peak memory, scheduler
termination reason and a verified code commit are not automatically
complete in this inspected recorder. Add them from actual job/release evidence
to the ledger rather than claiming that every desired provenance field is
already generated.

## 8. Optional fixed-context Boltz feasibility adapter

The files `src/latentmol/boltz/contracts.py`, `adapter.py` and `model.py` are
**excluded from source-005** and included in source-006. Their native preparation
has completed, and source-010 passed the stated fixed-context parity, gradient,
finite-difference and frozen-state checks. Their dependencies and
pinned source/assets are managed in a separate environment; see setup attempts
29503174, 29503177 and 29503187 in the ledger.

`PreparedComplex` stores an immutable feature snapshot, explicit decoder-slot
to Boltz-atom/token maps, a seed graph, provenance, and a detached pose shaped
`[1,1,A,3]`. The initial contract supports batch size one, one ligand atom per
ligand token, and one fixed pose. These mappings must be validated; implicit
assumptions that decoder order equals Boltz token order are not permitted.

`fixed_occupancy_graph` conditions raw decoder logits on a fixed occupied-slot
mask **before** occupancy gating. `NO_ATOM` logits receive no derivative, and
empty slots cannot become atoms. `adapt_graph` maps the resulting chemistry to
`ref_element[1,A,128]`, expected scalar `ref_charge`, `token_bonds` as probability
of any bond, and typed-bond expectations with native IDs `0/2/3/4/5`.
`token_bonds` is binary connectivity probability, not expected chemical bond
order. The separate expected typed-bond embedding carries the distinction among
single, double, triple and aromatic categories without integer-index lookup.

`FrozenBoltzAffinity` composes official input-embedding, trunk and native affinity
modules directly in evaluation mode, with model weights frozen but chemical-input
gradients retained. This explicit path is needed because stock evaluation detaches
relevant tensors and disables trunk autograd. `eval()` controls inference behavior
such as dropout; it does not itself prohibit input gradients. The adapter keeps
Boltz in `eval()` with every parameter's `requires_grad=False`, directly calls
its input embedding, trunk recurrence and affinity heads outside the stock
no-grad/detach wrapper, and differentiates only chemical inputs/latent variables.
Switching Boltz to training mode is neither needed nor allowed by this adapter.
Geometry, crop/masks/pooling, atom names, reference conformer,
MSA/token metadata and pose remain fixed; affinity distances remain hard-binned.
Diffusion and confidence derivatives are excluded. The output uses raw affinity
heads/ensemble probability with molecular-weight correction disabled, rather
than a correction based on a stale reference molecule's weight.

`discrete_reference` defines planned forward parity against native discrete
trunk/heads at identical fixed coordinates and recycle count. It is not a second
full stochastic structure-prediction run. Planned checks must separately establish
that gradients are finite and connected, finite differences agree, frozen model
parameters are unchanged, and the advertised atom/charge/bond channels actually
affect the output. None of these claims is established by successful package or
checkpoint downloads.

The prepared snapshot fixes at most 128 MSA rows, retaining query row zero and
using seed 42 for selection, while preserving the full native profile. Native
eval-time MSA resampling is disabled for the fixed-context scorer. The smoke
uses FP32 standard PyTorch kernels, TF32 disabled, zero recycles (one trunk pass),
and central directional finite differences at epsilons 0.01, 0.03 and 0.1; its
declared gate is best relative error at most 0.2. This is a bounded engineering
check, not a broad numerical-error guarantee.

`smoke.py` can then perform three normalized latent-only updates of norm 0.05,
logging relaxed value/probability, displacement, fixed atom count and raw discrete
validity/SMILES. The objective minimizes the uncorrected raw affinity value, whose
native interpretation is log10(IC50 in micromolar); the native documentation
limits its intended comparison to actives. No applicability or binding claim is
made for relaxed/decoded graphs in this smoke.

Local source-007 additions include exact pre/post fixed-context hashes and fuller
metadata checks. Source-006 predates those additions. Source-007 is uploaded with
its own archive hash, but has not yet been submitted; source inspection is not
runtime validation of these additions.

Source-006's actual adapter smoke failed during `load_affinity_model`, before any
adapter forward or gradient: the official checkpoint's saved diffusion arguments
include obsolete `mse_rotational_alignment`. Native CLI preparation succeeded
because its loader replaces runtime argument groups. The source-008 correction
implements `affinity_runtime_overrides()` using the pinned CLI's dataclasses for
diffusion, pairformer, MSA, steering and prediction arguments. It retains strict
weight loading and the explicit fixed-MSA policy; it does not silently delete
unknown checkpoint fields or loosen state-dictionary compatibility. This corrected
loader was exercised in source-008 job 29503249: 50 tests and exact native
fixed-pose parity passed, resolving the loader incident. The subsequent
gradient-enabled forward ran out of GPU memory before gradients or latent updates.
Per-block activation checkpointing completed in source-009 job 29503262, with
exact native parity, finite nonzero chemical/latent gradients and unchanged frozen state.
Both failed attempts remain in the ledger with their completed stages preserved.

`checkpoint_eval_block` wraps only a native block's `forward` callable using
PyTorch's non-reentrant `checkpoint(..., use_reentrant=False, preserve_rng_state=True)`.
This mode supports `torch.autograd.grad` through frozen modules. It rejects
training mode, is idempotent, and bypasses checkpointing when autograd is disabled.
`install_eval_checkpoints` selects native `PairformerLayer`, `MSALayer` and
`PairformerNoSeqLayer` instances, recording counts. It does not replace registered
modules, alter parameter/state-dictionary names, copy native parent loops, or
change their mathematics/chunking policy. The same fixed context and FP32 precision
are retained. Source-009's regression suite passed 51 tests in 24.22 seconds and
the full smoke completed. Its original best-relative-error <=20% finite-difference
gate passed at epsilon 0.01 (11.809% error); errors at 0.03 and 0.1 were 37.892%
and 56.895%. This establishes the declared initial gate, not tight convergence.
All three latent updates retained the same valid hard-decoded molecule.

Source-010, resubmitted unchanged as job 29503342 after pending job 29503317 was
cancelled without execution, uses central finite differences at
`0.0001, 0.0003, 0.001, 0.003, 0.01` and requires at least two distinct epsilons
within 2% relative error. It adds valid-only, no-grad discrete projection scores
at each trajectory point, explicitly labeled as fixed-context evaluations.
These preserve the original pose/reference features and are not full discrete
re-preparation. Invalid projections receive no score with an explicit reason.

The new atomic `latent_trajectory.pt` records each latent, raw decoder logits,
hard graph labels and metric row, plus source/checkpoint/prepared/context hashes.
This is a saved engineering trajectory, not a training checkpoint or proof of
scientific usefulness. Source-010 passed 53 tests in 26.67 seconds on A10 and its
real-model audit completed: the first four epsilons met the 2% criterion, native
parity remained exact, gradients were finite/nonzero and frozen state unchanged.
All four hard projections were the same valid 13-heavy-atom molecule with exactly
constant fixed-context raw value and binary probability. Four trajectory states
were saved with a hash. These are source-010 results, distinct from source-009's
coarser derivative gate and absence of projection scores.

## 9. Full-data trainer and workflow additions: CPU and GPU regressions passed

The new trainer retains evaluation-compatible format-1 checkpoints and adds
versioned completed-epoch continuation state. It saves AdamW state, global
epoch/update and absolute LR/KL schedule position, Python/NumPy/Torch/CUDA RNGs,
independent DataLoader generators and checkpoint-selection trackers. Resume
checks executable source identity, numerical runtime/device settings, data/split,
graph schema, configuration and selected indices. Only the configured horizon
may extend; `stop_after_epoch` is a runtime phase boundary rather than a config
mutation. Fresh AE/VAE initial weights are identified by
`seeded_initial_model_sha256` so matched initialization is auditable.

`best.pt` minimizes full-validation reconstruction. `best_validity.pt` maximizes
the tuple `(fixed_probe_validity, fixed_probe_exactness, -full_validation_loss)`;
strict improvement preserves the earlier epoch for complete ties. Selected
winners, periodic epochs and phase endpoints receive immutable `epoch_XXXX.pt`
files. Resumption verifies and carries forward ancestral winners; it does not
reset selection to the resumed phase.

Fixed deterministic-mean train/validation probes expose matched reconstruction
gaps; full validation loss is computed each epoch. Per-epoch chemistry uses the
stable 1,024-molecule validation probe, while full-validation chemistry expands
every 10 epochs. The shared evaluator also records chemical failure categories,
size/entropy and latent diagnostics. Timing covers training, evaluation and
checkpointing; gradient norm/clipping rates and throughput are separate fields.
Full validation and final reconstruction evaluations use batch 64 and matching
seeded full-split order; otherwise weighted batch-normalized losses could differ
solely because batching/order changed.

`continuation_decision` compares arithmetic means of two adjacent 10-epoch windows,
rejecting nonfinite/missing metrics or noncontiguous history. Loss gain divides
by `max(abs(old_mean),1e-12)`; positive gains trigger the adopted thresholds.
Training-probe/validation-probe gaps are reported diagnostically and do not
replace the declared validation-only stopping criteria. Matplotlib 3.10.1 is pinned
for optional plot generation and imported inside the plotting function. It
installed during source012 validation, whose full suite failed in fixture setup.
Corrected source013 validation passed 77 tests with two skips. A later plotting
refinement separates differently scaled quantities onto separate axes and bounds
fraction axes; source015 validation passed. AE KL is absent/not applicable;
an "unregularized" title must not be interpreted as an observed AE KL metric.

`full_training` performs automatic same-process phase continuation, then seals
that model's common-epoch 60/best-loss/best-validity/last selection before its test
calls, grouping duplicate underlying checkpoints by role. Both models' configs
and rules are fixed before launch; test results cannot alter either independent
run. Campaign test comparisons are made only after both finish. This is per-model
sealing, not a requirement that both models finish on the same date.

Scheduler-interruption recovery requires explicit trainer resumption from a
saved completed-epoch checkpoint. The full workflow has no automatic restart
entrypoint; simply relaunching it starts new workflow training. Static review
passed. Source012 full validation reported 60 passes, one skip and 18 fixture
errors because the pytest base-temp parent was absent. Source013 created that
parent; job 29503855 passed 77 tests (two skipped), including CPU AE/VAE exact
resume checks with loader workers zero and one, then passed the actual-cache audit.
Source013 GPU launches failed during shell path expansion before experiment
creation. Source014's direct script resolved that startup issue: both GPU RNG
checks passed, but the first training batch failed because rank-three
cross-entropy selects a CUDA kernel unsupported under strict deterministic
algorithms in PyTorch 2.5.1. No optimizer update completed in those attempts.
Source015 flattens cross-entropy inputs to two dimensions without changing the
objective. GPU validation 29503897 passed 83 tests in 82.11 seconds, with one
optional Boltz constructor test skipped because Boltz is absent in the AE
environment. Actual GPU AE/VAE training/resume checks with zero and one loader
worker and CPU loss/gradient equivalence all passed. The repeated cache audit
also passed; full-data workflow results remain separate from regression results.
