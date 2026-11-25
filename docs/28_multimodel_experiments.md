# Configurable frozen-model experiment pipeline

For new exploratory work use the [shared exploratory policy](EXPLORATORY_RESEARCH_POLICY.md)
and [gradient policy](GRADIENT_VALIDATION_POLICY.md). Detailed native/independent
AD/FD gates below describe qualified tools and specific historical campaigns;
they are not blanket admission requirements for a new short pilot. Existing
sealed runs and historical outcomes keep their original contracts.

Status, 1 October 2026: **source053 immutable local fixture smoke passed;
no real Nesso-1 or Protenix native validation, HPC run or combined-gradient
scientific result.** Quick local fixture smoke/unit checks were authorized for this
task. The user initially prohibited HPC testing/submission during the access
outage, then authorized the reviewed follow-up after reconnecting VPN.
Full model work remains an HPC step. This guide describes the current interface,
not a validated recipe for molecule design. Read the [next-test protocol](26_next_test_protocol.md),
[frozen-plugin contract](23_frozen_model_plugin_contract.md) and
[experiment ledger](05_experiment_ledger.md) first.
The active-set Boltz extension below was requested **after** source053 and
is not covered by that smoke suite. Its 1 October double-check passed a
separate pre-correction local fixture baseline (36 tests; one skipped),
expanded corrected suite (108 tests; one skipped), and the final extracted
source054 suite (118 tests; one skipped), as recorded in the ledger. After
later user authorization and restored access, HPC CPU job 29543381 passed
119 fixture tests and the chosen stable VAE's exact C13 start audit. The
2 October outcome check confirmed GPU preparation job 29543383 completed
native contexts for counts 11–15. Pilot job 29543391 then failed its initial
finite-difference gate: hard-seed/native parity passed, but the random probe
direction passed only one of three epsilon scales where two were required.
The cause is unresolved. No optimization update, fixed-mask control or
fresh-pose follow-up ran; this attempt yielded no optimized molecule.
Interim local mock/fixture suites on changing source states passed 58,
63 and 65 selected tests. The extracted immutable source053 archive then
passed the same **65 tests in 2.24 seconds**; its archive SHA-256 is
`83372a51992889500f724d87e52ce02a99f43213070d72dfd077aebd141a4287`
and its source-manifest SHA-256 is
`886b1c0ee2f5bba7da521e585246fea3d6288c13c9a59a11110ba2c466dfcf3b`.
These tests exercise orchestration and two mock native subprocesses,
not a real native checkpoint or molecular outcome. The exact command,
run IDs and environment are in the ledger.

## What the configuration controls

`src/latentmol/multimodel/config.py` parses a version-1 YAML `Study`. The
study names one pinned graph decoder (`checkpoint`, `sha256`), `device`,
`occupancy` (`fixed` or `variable`), temperature, model aliases, arms,
optimizer, diagnostics and optional start/plugins/native evaluators. Paths
for the decoder, raw start and recognized model asset options resolve
relative to the YAML file. A plan parses the schema and lists the lazy
factories without importing or downloading native models. Identifiers are
restricted to letters, digits, `_` and `-`.

Two editable **templates**, not validated experiments, are
[`configs/multimodel_boltz.yaml`](../configs/multimodel_boltz.yaml) and
[`configs/multimodel_ensemble.yaml`](../configs/multimodel_ensemble.yaml).
The first defines Boltz-only, frozen-start and random-direction arms. The
second illustrates single-model, paired, all-three and structure-weight
ablation arms. Every `REPLACE_WITH_..._SHA256` token and native asset path
must be replaced with verified values before loading; the example
scale/weight values are **illustrative**, not calibrated. The example
Boltz recycling budget is zero and differs from ordinary production
prediction, so preserve it in every comparison and native audit.

Each `models.<alias>` chooses a built-in `plugin` (`boltz2`, `nesso1`,
`protenix`) or a trusted custom factory, an explicit scalar `objective`,
model-specific `options`, and `allow_lossy` (default `false`). Objective
fields are `output`, `direction: minimize|maximize`, nonnegative `weight`,
positive `scale` and finite `offset`. The runner minimizes
`weight × sign × (native_output - offset) / scale`, where `sign=-1` for
maximize and `+1` for minimize. It requires the adapter to return a
**finite scalar already reduced in a scientifically declared way**; it
does not average arbitrary pair logits or auto-normalize weights. An arm
lists model aliases, optional weight overrides and step count, and a
control (`optimize`, `no_update`, `random_direction`). Zero weight disables
that branch entirely, including loading/calls; a missing or nonfinite
positive-weight output fails rather than being dropped. Each arm needs at
least one positive weight.

The optimizer declares steps, learning rate, raw-`z` trust radius and
maximum combined gradient norm. Diagnostics declare seed, hard draws and
multi-scale finite-difference/parity tolerances. `start: {}` encodes the
**one shared hard seed** into its decoder mean. A supplied raw-`z` start
requires `path` and `sha256` and must identify the same decoder and seed.
The raw Euclidean trust radius is an explicit movement bound, **not** a
calibrated AE/VAE latent distance. Built-in Nesso and Protenix prototypes
require fixed occupancy; selecting a variable policy does not make those
branches variable-size capable.

## Prepare and register a frozen branch

1. Pin the native source, weights, model config, dependencies and an
   ordinary hard-input preparation command. Keep chemistry conversion and
   any fresh pose generation **outside** the differentiable branch. Do not
   train a reward, affinity, validity, synthesis or chemistry-prior model.
2. Prepare **one canonical seed graph** and target context. For Nesso-1 or
   Protenix-v1, upstream native featurization must export tensor-only
   `features`, `seed`, complete `slot_to_atom`/`slot_to_token` maps,
   ligand atom/token masks, and `element_index` mapping graph atom classes
   to the native 128-column element encoding (`NO_ATOM=-1`). The whole atom
   vocabulary must match the audited native convention: Nesso uses atomic
   number `Z`, Protenix uses `Z-1`. Add metadata
   `preparation_command`, `native_source_version`, `seed_identity`, and
   declared feature losses. Call `external.export_native_bundle` at a
   unique path; it validates correspondence and returns a SHA-256 but does
   **not** perform native upstream preparation or verify model behavior.
   Nesso additionally needs `affinity_keep_indices` from its ordinary seed
   forward; its masks must agree with native `mol_type` and
   `affinity_token_mask`, and typed bond IDs come from its native
   featurizer/tokenizer convention. Protenix-v1 needs native atom-level
   `is_ligand`/`is_protein`, a `protein_token_mask` consistent with
   atom-to-token aggregation, fixed ligand–protein
   `distance_pair_mask`/`distance_target_bins` and a declared
   `distance_target_source`; see the adapter's current `NativeBundle`
   contract for exact tensor shapes. Covalent ligand–context edges are
   rejected in these first fixed-context adapters. No branch may independently decode a
   different molecule from `z`.
3. In `models.<alias>.options`, provide hashed bundle, weights and model
   config paths (`bundle_path`/`bundle_sha256`, `weights_path`/`weights_sha256`,
   `model_config_path`/`model_config_sha256`) plus
   `native_source_root`/`native_source_sha256`. The source digest covers
   sorted relative `.py` paths and their bytes, including local edits;
   imported native modules must resolve inside that audited checkout.
   Runtime verification also checks that the loaded feature/maps/seed
   context digest has not changed.
   The loaders do not download weights or install packages. Boltz uses its
   existing prepared-complex/checkpoint/recycling options and partial
   fixed-pose affinity adapter.
4. For a new model, implement a frozen factory with the signature
   `factory(spec, device, options, source_sha256) -> BranchRuntime`, then
   register a trusted `module:function` in `plugins` under a new name.
   Built-in names cannot be shadowed. A `BranchRuntime` supplies a
   `BranchManifest`-bearing frozen branch, canonical seed, occupied-slot
   mask, seed-only ordinary `native_reference()`, verified assets and an
   optional unchanged-context check. Its `assess()` must declare exact,
   lossy or unsupported input and differentiable channels. The native
   reference is **seed-only fixed-context parity**, never a changed-molecule
   fresh rescore. A small paired input/early-feature bridge is allowed only
   under the held-out fidelity/identity gates in docs23, then frozen.

The current Nesso-1 prototype exposes affinity scalars through a fixed
native pocket; it disables the 22 Å refinement recrop and removes upstream
trunk/head detaches while retaining pretrained parameters. Formal-charge
gradients exist only if that checkpoint consumes the extra charge feature.
The Protenix-v1 prototype exposes `distance_consistency_loss` (mean
ligand–protein distance-bin cross-entropy normalized by log bin count) and
entropy from a fixed reference target. This is a **structural** signal,
not affinity; the graph-to-native map loses bond order/aromaticity, and
pose/tokenization/target bins stay fixed. Neither prototype has passed
real native parity or finite differences. Native weights and genuine
upstream bundles have not been exercised by local fixtures.

## Execute in gated stages

The CLI is `python -m latentmol.multimodel STAGE --config STUDY.yaml
--run-root RUN_ROOT` with `STAGE` one of `plan`, `validate`, `run`,
`pipeline`, `confirm`, `diagnose`. Use `--arms` to select configured arms;
otherwise all arms are selected. `--source-root` controls the run's source
manifest, and optional `--output` writes a receipt without overwriting an
existing file. Commands below are an **interface guide**, not a record
that a real native command has been run:

```text
python -m latentmol.multimodel plan --config STUDY.yaml --arms boltz_only
python -m latentmol.multimodel validate --config STUDY.yaml --arms boltz_only --allow-experimental --run-root RUN_ROOT
python -m latentmol.multimodel run --config STUDY.yaml --arms boltz_only --allow-experimental --gate-report GATE.json --run-root RUN_ROOT
```

`plan` performs no heavy model load. `validate` verifies decoder/native
asset hashes and common graph/seed identity, freezes every selected model,
checks seed hard-native forward parity and a finite nonzero `z` derivative,
then tests the **actual signed/scaled scalar** with multi-epsilon central
differences along gradient and random directions. It writes a gate report
incrementally, retaining a failure. `run` requires a passing report for
the exact config, source, assets and selected models; `pipeline` performs
validate then run in one recorded invocation. `--allow-experimental`
acknowledges prototype status but bypasses **none** of the gates. No
passing real-model report currently exists for the new branches.

Every arm starts at the same raw `z`; the runner queries enabled branches
sequentially on **one relaxed graph**, logs native raw and signed/scaled
outputs, per-model/weighted gradient norms and pairwise cosines, clips the
combined gradient and projects the update into the configured trust radius.
It freezes and hashes decoder/model state before and after. It saves each
step's latent/relaxed state, argmax plus seeded categorical hard graphs,
all invalid/duplicate attempts, predicted size/charge, attempted and
completed oracle/autograd calls and elapsed time. The `no_update` and
`random_direction` controls provide trajectories with explicit cost
accounting; compare actual call counts and GPU time, not just steps.

`confirm --candidate-run RUN` is a **separate hard/native** stage. Each
configured `native_evaluators.<alias>` supplies an argv list with `{input}`
and `{output}` placeholders, timeout, source/weights/config identity and
`candidates_per_endpoint`. It selects first saved draws at start/end under
the fixed budget; invalid draws consume slots and trigger no native call.
Native evaluators are named independently of the gradient branches: an
evaluator may score saved candidates from a model that did not generate
them, provided the completed generating run, decoder/schema and every raw
candidate identity pass the provenance checks. `confirm` and `diagnose`
read **all completed saved arms** from `--candidate-run`; `--arms` is for
planning/generation and is rejected for these two stages to prevent a
silently ignored after-the-fact filter. The confirmation
config records both generating and evaluating study fingerprints and
hashes the source run, identity file and candidate files before calling
any native executable.
The executable receives a JSON request identifying the candidate graph
and must return explicit `ok|unsupported|failed`, candidate/graph/input
identity, effective SMILES, finite named scores and
`fresh_native_preprocessing=true` for success. Changed identity is retained
but is **not** score retention for the original molecule. No such real
native evaluator has been validated here. `diagnose --candidate-run RUN`
separately applies the existing graded hard-graph diagnostic to saved
attempts; its bounded search does not alter the scorer input.

For a **future authorized HPC turn**, `scripts/package_multimodel_release.py`
creates a unique local tarball with uncommitted source, configs and docs plus
`release_manifest.json`, rejecting source edits during packaging. The
unexecuted `scripts/hpc/multimodel_job.sh` accepts `tests`, `plan`,
`validate` or `pipeline` inside an LSF allocation, verifies the archive and
extracted manifest, and records an experiment run/receipt. The unexecuted
`scripts/hpc/submit_multimodel.sh` is an explicit LSF submission template
whose queue, GPU, memory and wall-time defaults are **not** measured for a
three-model ensemble. Neither script installs native dependencies or
downloads weights. **Do not submit these scripts in the current task:** the
user prohibited HPC testing now. Record any later source/config/data/asset
hash, scheduler ID, gate outcome, failures and artifacts in the ledger.
The separate, untested `scripts/hpc/multimodel_matrix.sh` would run one
pinned study YAML per decoder/seed in sequence, stopping after a failed
gate and preserving earlier child runs; it does not make the same raw-`z`
trust radius comparable across AE and VAE spaces. The current code does
not automatically create genuine upstream native bundles, model-compatible
Python environments or fresh-native JSON executables. Those inputs must
be built, versioned and tested before a substantive HPC run.

## Boltz active-set atom-count extension

The `src/latentmol/boltz/active_set.py` runner uses the existing DBMol-style
fixed-size differentiable input **within each step**, while allowing the
decoder's occupied slot set to change **between steps**. It does not require
new decoder outputs or a differentiable atom-birth operation. This is a
standalone Boltz-only runner; setting `occupancy: variable` in a
`multimodel` study does not enable this behavior for Nesso or Protenix.
Its first native pretrained Boltz attempt passed hard-seed forward parity
but failed the complete conditional derivative gate, as summarized above.
`tests/test_boltz_active_set.py` contains relevant contract checks and was
included in the local double-check baseline and expanded corrected suite,
alongside runtime and rescore regressions. These fixture checks do not
validate the complete runner against a native pretrained Boltz model.
At each latent state, compute the ordinary atom-category probabilities at
the study temperature; a slot is selected when its `NO_ATOM` probability is strictly
below `0.5`. Record the selected slot mask and count, including a transition
in which the count stays the same but the slot identities change. The
active-set arm fails closed on an empty selection or an unavailable prepared
context. The fixed-mask control can log an empty *natural* selector while
continuing to score its original nonempty mask; neither is a score of zero.

For one selected mask, condition each selected atom on the real-element
categories by softmaxing its real-element logits without `NO_ATOM`.
Condition charges on their existing categorical logits. Only selected atom
pairs carry the decoder's soft bond distribution; absent pairs are `NONE`.
This is the existing fixed-occupancy graph policy applied to a newly chosen
mask. It gives no *direct* Boltz gradient to an included or excluded slot's
`NO_ATOM` logit. Because the decoder shares `z` across all outputs, a gradient
through included features may nevertheless move occupancy logits indirectly.
That hypothesis needs a measured count/slot-transition rate, not an assumption.
Do not add the proposed occupancy-dependent gradient scaling in this first
version.

Each mask must map its selected slots injectively onto **all** ligand atom and
token rows of a Boltz-prepared complex with the same count. Prepared feature
maps, a reference pose and other topology/geometry features are fixed during
one step. A count change requires a compatible prepared context; even a
same-count slot substitution requires a new declared slot-to-native mapping.
Preparing or selecting that context is a discrete operation outside the
gradient path. A context switch may change the score solely because the
reference features or pose changed, so log the old and new context identities,
pre/post-switch scores, and the change before making another update. Do not
interpret score monotonicity across contexts as molecular improvement.

The count-specific native input helper is:

```text
python -m latentmol.boltz.prepare chain-count-example --output INPUT_DIR --count N
```

For compute nodes without internet access, append `--assets-dir CACHE_DIR`.
That directory must contain `official_affinity.yaml` and `seq1.a3m` matching
the pinned official-example hashes; the helper verifies their bytes before
copying them. This changes asset delivery only, not the declared target or
MSA.

It writes `INPUT_DIR/carbon_chain_NN.yaml` and
`INPUT_DIR/input_manifest.json` for a declared hard carbon-chain ligand with
`N` heavy atoms using the existing public target/MSA engineering context,
rather than converting an arbitrary relaxed graph to SMILES. An ordinary
Boltz prediction must then
produce its pose and native processed input outside autograd. The existing
snapshot command is:

```text
python -m latentmol.boltz.prepare snapshot --prediction-dir PREDICTION_DIR \
  --cache CACHE --output PREPARED.pt --input-manifest INPUT_DIR/input_manifest.json
```

It exports a native-featurized tensor context. `--input-manifest` checks the
cropped ligand against the declared seed and verifies the declared input YAML,
protein sequence, MSA file hashes and native record stem. It **cannot
independently prove** that the processed native protein/MSA tensors match
those declared files; retain this provenance limit. These commands
are **preparation interfaces, not gradient or optimization validation**;
consult the ledger for actual run status. Pin and
hash every count-specific native input, pose and prepared tensor context
before using it in a trajectory.

After preparing each count, build the catalog:

```text
python -m latentmol.boltz.prepare count-catalog --root ATTEMPT_DIR \
  --counts 11 12 13 14 15 --output ATTEMPT_DIR/contexts.json
```

It indexes only finished, distinct prepared snapshots with the same target/MSA
policy and records the source-manifest SHA-256, source-archive SHA-256 and HPC
job ID where available. Those fields provide provenance, not proof that
processed native target/MSA tensors match the declared files. The
HPC convenience launcher
`scripts/hpc/boltz_active_set_prepare.sh NEW_ATTEMPT_DIR AE_CHECKPOINT 11 12 13 14 15`
does the input generation, an ordinary short-budget Boltz prediction, tensor
snapshot and count catalog in an allocated GPU job. The examples of five
counts are illustrative, not a measured size range or an instruction to
submit. It uses a new directory, a declared immutable source identity and
the pinned Boltz environment. The short pose budget is an engineering
context, not a binding benchmark. The launcher requires
`BOLTZ_EXAMPLE_ASSETS` to name this verified offline asset directory, in
addition to `BOLTZ_CACHE` and the allocation/source identity.

`configs/boltz_active_set.example.yaml` is an **unrun template**. Replace
its decoder, affinity checkpoint and context-catalog paths/hashes with
verified immutable assets. The required `mode: vae` (or `ae`) must match the
verified decoder checkpoint's training mode; a mismatch aborts.
`anchor_count` chooses which native carbon-chain
seed is encoded to initialize `z`. The decoded starting mask must be
**exactly the anchor's native slot mask**, and ordinary argmax hardening must
recover the exact prepared carbon-chain graph, including atom, charge and
bond labels. A different initial mask or hard identity fails before loading
frozen Boltz, even if the catalog contains that other count. This guards the
declared carbon-chain start; it does not imply the VAE will always reproduce
it. Such a failure remains in the run record.

Before spending GPU time on count-specific poses, an HPC CPU preflight can
check this one exact carbon-chain seed using the frozen decoder alone:

```text
python -m latentmol.boltz.active_set_preflight --checkpoint DECODER.pt \
  --sha256 DECODER_SHA256 --mode vae --count 13 --run-root RUNS \
  --receipt NEW_START_RECEIPT.json
```

The preflight fixes temperature `1.0` and threshold `0.5`; it saves the
encoded latent and `start-audit.json`, including exact-mask/label agreement
and hard-graph diagnostics. It writes the requested receipt only on success.
A failed start blocks the subsequent GPU preparation and remains a failed
experiment; it is not silently replaced by a different ligand or fitted
latent. This is one frozen-decoder forward check, with no native Boltz
loading, optimization or cohort evaluation. The optimization runner still
rechecks the start against its actual native prepared anchor.

The allocated CPU wrapper is
`scripts/hpc/boltz_active_set_preflight.sh DECODER.pt DECODER_SHA256 vae 13 NEW_START_RECEIPT.json`,
invoked through the pinned `boltz_run.sh` environment. It records and runs the
selected fixture suite first, then performs the exact-start preflight only
if that suite passes. Both phases must pass before a GPU preparation job is
submitted. The selected C13 checkpoint is the source047 stable VAE's
epoch-89 `best_validity.pt` (SHA-256
`1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`).
Source054 HPC CPU job 29543381 verified this exact C13 start. Its smallest
occupancy-threshold margin was about `0.49998`; the start is far from the
`0.5` threshold, so the short subsequent pilot may make no count changes.

`threshold: 0.5`, temperature, zero-recycle fixed
pose, lossy expected-charge opt-in, signed affinity objective, optimizer
and finite-difference tolerances are explicit. Configuration validation rejects
unknown fields, nonfinite settings, malformed asset hashes and unsupported
affinity outputs; choose exactly one inline count map or hashed catalog.
The catalog header must agree with the identities of its actual prepared
snapshots. The HPC-only runner requires
`LSB_JOBID` and `LATENTMOL_SOURCE_SHA256` and is invoked as:

```text
python -m latentmol.boltz.active_set --config configs/boltz_active_set.example.yaml
```

The untested HPC wrapper
`scripts/hpc/boltz_active_set_run.sh VERIFIED_CONFIG.yaml` invokes this
runner inside an allocation and requires the same immutable source archive
identity. Source054 invoked the runner through the separate hash-pinned
orchestration script; it stopped at the initial derivative gate. No
completed trajectory is established by that invocation.

It runs matched `active_set` and `fixed_initial_mask` arms from the same
encoded start, with the same update rule and step budget. The fixed control
retains its starting slot mask; the active arm reselects after each latent
update. Extra context gates and transition scores can make **total** Boltz
calls differ between arms; use `oracle_cost.json` when comparing compute.
A slot substitution at constant count rebinds slot-to-native maps;
a count change loads that count's separately prepared native feature/pose
context. The runner hashes inputs, freezes the decoder and Boltz weights,
logs attempted/completed oracle calls, checks hard seed parity and the
conditional derivative at every encountered context, and saves per-step
states, hard argmax/categorical candidates, metrics, transition scores and
endpoint summaries in a unique run directory. Each attempted context change
has a `transition-STEP.json` status record and saved latent state before its
gate; `gate-STEP.json` retains partial parity/finite-difference evidence on
failure. The run summary hashes both arm candidate files before any separate
native confirmation. An empty selection or a selected count outside the
catalog writes `rejected-transition-STEP.json` and `rejected-state-STEP.pt`
then fails, preserving the attempted mask, probabilities, proposed latent
state and available counts. Verified asset/model loading occurs
inside the recorded `ExperimentRun`, so a load or gate failure is retained
as a failed run rather than disappearing before run creation. These are implemented
recording paths, **not completed native-model validation**.

A separate, **untested** fresh-pose follow-up accepts only a completed
active-set pilot and chooses the initial and final argmax graph from each
arm without ranking candidates by native score. Before selecting endpoints,
it verifies the completed parent configuration and summaries, passed context
gates and complete step/draw budget, and reconstructs candidate identities
from saved categorical labels to check their hashes and SMILES. It preserves invalid
endpoints in its selection report and sends only valid molecules to ordinary
Boltz prediction. The plain protein/SMILES-ligand template must request
affinity for its sole ligand, and its target sequence and MSA must match the
identities recorded from the parent's prepared contexts. Extra target
modifications and ambiguous ligand declarations are rejected. It hashes
parent summaries, gates and both `candidates.jsonl` files to
lock the selected inputs, and checks the native affinity checkpoint against
the optimization checkpoint. A relative MSA path in the input template is
resolved against that template before constructing ordinary native inputs.
It uses matched pose seeds `17` and `23`, saves
per-prediction outcomes and compares each endpoint with its own start;
shared molecules are predicted once and reused in both arm reports. The
native standardization preflight is saved in `identity_audit.json`. A
prediction whose input changes molecular identity under that standardizer
remains in the raw results but cannot contribute to a retained-gain pair.
This preflight does not independently audit the native processed tensors.
Eligible pairs report both affinity-head differences and the configured
signed-objective difference; a negative signed-objective difference is an
improvement for the minimization loss. Run it
only in a later authorized HPC job with a new run directory:

```text
python -m latentmol.boltz.active_set_rescore --pilot RUN_DIR \
  --template NATIVE_INPUT_YAML --cache BOLTZ_CACHE --run-root RUNS
```

This is a fresh native **model** check of hard candidates, not measured
binding, synthesis feasibility or selectivity. Source055's two reruns now
have completed fresh-native follow-ups; their outcomes and limitations are
recorded below and in the ledger.

The source054 submitted workflow uses the separately hash-pinned
`outputs/hpc-submissions/source054-run-pilot.py` to enforce preparation,
pilot and rescore ordering. It verifies the unchanged release, checkpoint,
CPU start receipt and the complete count catalog, then creates a new config
with the actual catalog hash. It requires a completed pilot before starting
rescore and preserves the outer command/script hash in an experiment record.
The exact script hash, job dependencies, resource budgets and artifact paths
are in the ledger. Queuing that workflow does not establish its outcome.

The later HPC gate must establish seed one-hot/native-forward parity for
**each** prepared context, source and checkpoint identity, finite-difference
agreement for the scalar objective with the selected mask held fixed, and
unchanged frozen weights/context. The runner flags when the natural selector
would cross its threshold under a finite-difference perturbation; that is a
boundary diagnostic, not a derivative through selection. Record every
gradient call, failed transition, threshold
crossing, score jump, decoded hard candidate and native preparation rejection.
Rebuild final hard molecules through ordinary Boltz preprocessing and fresh
pose prediction to check whether any fixed-context improvement survives.
Until these gates run, this extension is an unvalidated search mechanism,
not evidence of binding or valid molecule generation.

## Diagnose the source054 derivative failure before rerunning

The user authorized this diagnostic and a corrected rerun on 2 October.
All numerical work runs on HPC in new, hash-identified runs; source054's
failed run and original tolerances remain preserved. This is a protocol,
not a diagnosed cause or permission to bypass the derivative gate.

1. Reuse the saved initial latent, selected 13-slot mask, exact prepared
   context and frozen checkpoints. Verify their identities against the
   failed run, including the scalar objective and configuration. Hold the
   mask/context fixed throughout every derivative probe; log natural
   selector crossings separately.
2. Repeat the unperturbed forward score and gradient. Evaluate the same
   inputs with gradients enabled and disabled, recording raw outputs,
   dtypes, finite/nonfinite values and observed variation. This tests
   whether nondeterminism or a different numerical forward path affects
   the comparison; `no_grad` must not silently stand in for a different
   function.
3. Retain both raw perturbed scores and their difference over several
   predeclared step sizes and directions, including the original failing
   probes. Record the realized latent perturbation after dtype rounding,
   the analytic directional derivative and the predicted score difference.
   Compare these with representable score spacing and repeatability, and
   look for agreement across multiple scales before larger steps introduce
   curvature. Casting an already rounded scalar to float64 cannot recover
   information lost in its forward computation.
4. Use the evidence to distinguish a wrong gradient/path from insufficient
   numerical resolution. Changed arithmetic needs a new immutable source
   release; revised probe settings need their own immutable configuration.
   Both require an explicit rationale and passing checks on the preserved
   failure case before another trajectory. Do not obtain a
   pass simply by loosening tolerances or dropping the failing direction.
   Resume the matched pilot and fresh-pose follow-up only after the
   applicable gates pass; report diagnostic and optimization budgets
   separately.

The diagnostic-only interface is sealed in source055. Job 29544409 completed
its five fixture tests and native replay. Base scores, repeated gradients
and input tensors matched across forward modes, and all four directions
passed the original tolerances at epsilon `0.03` and `0.1`. The tiny
original random probe predicted a score difference of about 2.31 float32
output spacings. This supports a numerical-resolution explanation for the
smallest probes, not a claim that every possible input has a correct
gradient. No production correction or optimization rerun is established by
the diagnostic. The interface is:

```text
python -m latentmol.boltz.active_set_diagnostic --parent-run FAILED_PARENT_RUN \
  --run-root NEW_RUN_ROOT
```

The allocated HPC wrapper is
`scripts/hpc/boltz_active_set_diagnostic.sh FAILED_PARENT_RUN NEW_RUN_ROOT`,
invoked through the pinned `boltz_run.sh` environment from the immutable
release. The wrapper runs the five diagnostic fixture tests first and
stops if they fail. The replay requires the failed parent's saved start, step-0 state and gate,
and verifies their point/context identity before replay. The diagnostic
prespecifies epsilon values `0.001`, `0.003`, `0.01`, `0.03` and `0.1`,
the gradient direction and three seeded random directions. It repeats the
base score three times with and without autograd, repeats the gradient
twice, and compares both forward modes at the original scales/directions.
It retains `diagnostic.json`, `tensor-dtypes.json`, `probe-state.pt` and
`oracle_cost.json`, including partial evidence if diagnosis fails.
Completion means the diagnostic measurements were collected; it does not
reclassify the original gate as passed or launch optimization. Submission
identity, actual test/replay commands and outcomes belong in the ledger.

After this diagnostic, two separate config-only reruns were prepared on the
unchanged source055 core. They use epsilons `[0.01, 0.03, 0.1]` with the
original tolerances, both directions and two-passing-scales rule. The
first retains learning rate `0.05`; the second explicitly tests learning
rate `400`, calibrated from the original gradient to a proposed first
latent displacement of about `0.0494`. The second experiment changes step
size, so its outcome must not be attributed to the derivative-check change.
The trust radius stays `1.0`, and the matched arms and independent native
endpoint confirmation remain. The two YAMLs and a self-verifying runner are
now independently hashed under `outputs/hpc-submissions/`, with identities
in the ledger. After restored access, duplicate-submission checks and remote
hash verification, jobs **29544791** (`fd-corrected`) and **29544792**
(`step-calibrated`) were submitted. Job 29544791 completed both ten-update
arms and its native follow-up in 7 minutes 26 seconds. Its initial gates
passed, but both argmax endpoints remained the initial C13 chain, with no
context switch or endpoint relaxed-score gain. Job 29544792 also completed,
including its fresh-native follow-up, in 11 minutes 26 seconds. Both arms
reached the same valid `CCCCCCCCCC1CC1C` endpoint; the relaxed binary score
changed from `0.0790553540` to `0.0887691677`. It peaked earlier in the
trajectory, so this short run does not establish convergence. Fresh native
binary scores increased by `0.0543329418` and `0.0834434032` at pose seeds
17 and 23. The continuous head increased too, which is not improvement
under its native lower-is-better log-IC50 interpretation (intended for
actives); it does not corroborate stronger activity. Both arms retained
13 atoms with no context switch, and shared native predictions were reused
across identical endpoints. This establishes a completed engineering path
and a model-score change for one target/start, not an active-set advantage,
successful count adaptation or experimental binding. Exact artifact/config
identities and source054's original failure remain in the ledger.

## Focused 100-update follow-up

The user selected a focused C13 follow-up on 2 October. It restarts from
the same encoded C13 chain with frozen source047 stable VAE and Boltz
weights; it does not resume the ten-update pilot. Production source055 is
unchanged, but the configuration defines a new optimization regime:

- Each of `active_set` and `fixed_initial_mask` receives 100 updates.
- Learning rate `400` with raw-gradient norm cap `0.000125` bounds each
  proposed displacement by `0.05`. Trust radius `3.0` replaces `1.0`;
  the wider region has not been validated as reliable latent support.
- Temperature `1`, selection threshold `0.5`, existing contexts 11–15,
  and abort-on-unsupported-count behavior remain. Slot selection has no
  direct derivative, and a run with no count change remains possible.
- Finite-difference epsilons `[0.01, 0.03, 0.1]`, original tolerances and
  two passing scales per probe direction remain required.
- One argmax and eight categorical draws at states 0–100 give 909 saved
  hard candidates per arm. Report validity, identities, count/context
  switches, latent displacement, gradients, scores and total model cost.
- The final step-100 argmax is the primary fresh-native endpoint at pose
  seeds 17 and 23 under the existing identity checks. Best intermediate
  proxy scores are descriptive only; there is no endpoint reselection.
  Report both affinity heads separately and retain invalid outcomes.

This plan requests a two-hour `gpua40` allocation with one exclusive GPU,
four cores and 8 GB per core. The provisional 25–40-minute workflow
estimate excludes queue time and is not a deadline. After static review,
remote hash verification and duplicate-submission checks, job **29545141**
was submitted and is now **RUN**, started about 11:27 DTU local time on
2 October. The ledger records the exact config/runner/launcher hashes,
receipt and logs. A subsequent startup artifact check confirmed the active
arm's initial native-parity and derivative gates passed and metric states
0–3 were written. The run remains incomplete; no endpoint/native outcome
is established yet. Completion alone will not establish convergence,
an active-set advantage, useful count adaptation or experimental binding.

## Interpretation and experiment order

Start with the existing frozen Boltz-2 single-model baseline, then gate
Nesso-1 alone, then investigate Protenix-v1. A later Boltz+Nesso or
Boltz+Protenix arm is meaningful only on the **intersection** of supported
same-identity inputs after each branch passes native parity, derivatives
and hard-outcome checks. Fix objective offsets/scales/weights on validation
data before comparing trajectories. Report each objective and gradient
separately; a structural-distance objective must not be described as
binding affinity. A predeclared fresh native confirmation subset and
independent held-out evaluator are still required before a binding or
molecule-design claim. No combined-gradient or external-model result is
established by implementing this plumbing or passing local fixture tests.
