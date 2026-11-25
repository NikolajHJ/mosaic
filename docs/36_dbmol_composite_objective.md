# DBMol composite objective through frozen Boltz-2

**Consolidated reference, 23 October 2026:** read
[DBMol-style Boltz-2 loss: implementation, gradients, validation and pitfalls](77_dbmol_boltz_loss_implementation.md)
for the current code map, adopted contact-head precision and pointwise native
AD v4 scope. This document preserves the original source065 design and dated
submission history; its pending statements are historical, not latest status.

Status, 7 October 2026, **10:53:07 DTU-local**: source065 CPU/preparation
**29619428 DONE**, with **154 engineering tests passed in 26.05 seconds**
and eight actual matched configs prepared. Native GPU smoke **29619449 PEND**
was submitted after that gate. Native pretrained component-gradient validation
and molecule outcomes remain pending. The user requested the DBMol objective containing
affinity, contact and confidence, with contact/confidence coefficients 2.5.
The historical source057 panel and active source064 stability campaign use
binary-affinity-only objectives. Their sealed configurations and running jobs
remain unchanged and provide historical controls, not results for this new
profile. No composite molecular outcome is established by this document.

Actual archive SHA-256 is
`4d5a43421e4351a8c0cfb8cfd29c96d5fbcaa8d7b8af53c81bc86b43eaccd7bf`
at `outputs/source-releases/source-065-dbmol-objective-attempt001.tar.gz`;
scientific manifest is
`bdf49b4b0837d831c8c4b6e4ba0c37b775677cb4ef64932378445c7dba57fc3b`.
Remote release is `/work3/s234818/latentmol/releases/source-065-dbmol-objective-attempt001`.
Receipts/prepared root use `/work3/s234818/latentmol/submissions/source065-dbmol-attempt001`;
the unchanged historical protocol SHA is
`b4488b0422432f7de9613baea73cc245e610953e07d89aeb261e74bfc313a3d0`.
Exact command, CPU logs/resources and later outcomes are in the
[ledger](05_experiment_ledger.md). Plan-only and pre-seal notes below are
historical implementation descriptions, not claims that no submission occurred.

The prepared anchor pocket has **44 residues/221 protein heavy atoms**, shared
across the two starts, with context SHA
`d9a863dc5e0fd978e35d2b8ff0f7eab31b8efe6d2fc211b710a7ba73690e5030`.
Actual comparison SHA is
`23c2eba97c1e9f480e9e769b5241c8f3397a27a43603b268241918d4ec72d7a8`.
Downloaded 25-member evidence is
`outputs/hpc-results/source065-dbmol-objective/evidence-cpu-native-submitted.tar.gz`,
SHA `d97701b79f4dd39e34f90d7c66c0777c60bb00f01161dd5c7cc40ecc85a19992`.
Fixtures/preprocessing passed on HPC, not locally; they do not establish
native heads/gradients or a real binding pocket. A copied invalid comparison
SHA was rejected before submission; its corrected single native receipt is
retained with the evidence, not counted as a failed native experiment.

The coordinator verified [DBMol v1, section 3.2 and Appendix C.1](https://arxiv.org/html/2607.19237v1),
including equations 2–6, against the local [paper](DBMol.pdf), pages 4–5 and
16–17. The implementation below follows those term definitions through a
fixed-pose native interface. Explicit numerical choices that the paper does
not specify are declared separately; this is not a claim of complete upstream
implementation equivalence.

## Loss definitions and signs

Minimize

```text
L_total = L_affinity + 2.5 L_contact + 2.5 L_confidence
L_affinity = mean(native affinity regression value)
             - mean(raw native binder-head logits)
```

The binder term averages the raw head logits, not the logit of an averaged
sigmoid probability. Both affinity terms have default coefficient one. This
is a substantive change from maximizing the historical
`affinity_probability_binary`: a decrease in this total is not interchangeable
with an increase in the old binary proxy. Retain raw regression, raw binder
logits and binary probability alongside every component and weighted total.

For pocket protein token `i` and selected ligand token `j`, define

```text
q_ij = sum of native distogram probabilities whose representative distance < 8 Å
e_ij = -log(q_ij + epsilon)
L_pocket = mean over pocket i of min over ligand j of e_ij
L_anchor = min over all pocket i, ligand j of e_ij
L_contact = (L_pocket + L_anchor) / 2
```

This gives equal weight to pocket coverage and the strongest pocket–ligand
pair. It does not sum arbitrary contacts or reward all protein pairs.
The implementation uses log-space categorical mass and stable addition of
epsilon; minima remain piecewise differentiable within the frozen context.
They do not discretize the molecule. Native protein-to-ligand orientation is
retained rather than inventing an extra directional averaging convention.

The six-term confidence loss is

```text
L_confidence = -1.0 iptm
               -0.3 complex_iplddt
               +0.3 complex_ipde
               +0.1 complex_pde
               -0.3 confidence_score
               -0.1 ptm
```

These names refer to the native confidence outputs and their native reductions
and units. Higher confidence quantities enter with negative signs, while
predicted error quantities enter with positive signs. No new normalization
of their scales is silently applied. The native aggregate used here is

```text
confidence_score = (4 complex_plddt + interface_score) / 5
interface_score = iptm, except the native approximately-zero-iptm rule uses ptm
```

The pinned implementation tests approximate zero using native `torch.allclose`
defaults. That declared rule does not authorize replacing absent or
unsupported interface metrics with PTM. Missing/nonfinite/nonscalar required
metrics and unsupported helper fallback paths must fail the new profile.
Record all six raw terms and `complex_plddt` used by the aggregate so signs,
weights and native reduction parity can be inspected independently.

## Numerical and pocket choices

The draft opt-in profile is `dbmol_fixed_pose_v1` in
[dbmol_objective.py](../src/latentmol/boltz/dbmol_objective.py). Its config
declares the checkpoint path and SHA-256, frozen pocket residues, origin,
optional reference identity, weights, bin policy and epsilon. Historical
configs do not select this profile by default.

The pinned Boltz distogram has 64 categories, with 63 boundaries from 2 to
22 Å. `native_64_midpoint_lt_cutoff` represents interior bins by their
midpoints and open end bins by an extrapolated half-bin midpoint, then selects
representatives strictly below 8 Å. `epsilon=1e-8` is added to the contact
mass before its negative logarithm. These cutoff rounding and stabilization
choices, and `mean_native_head_logits`, are explicit implementation choices
where the paper does not fully specify the numerical convention. Changing
them requires a new recorded config, not an undocumented adaptation.

The user approved deriving the pocket once from **protein heavy atoms strictly
within 8 Å of the initial ligand heavy-atom pose**. Pocket preparation stays outside
autograd. Freeze its native `(asym_id, residue_index)` identities and record
the reference prepared/context hashes under origin
`initial_pose_heavy_atom_8A`. An explicit nonempty native residue list is also
supported with origin `explicit_native_residues`; its source must be declared.
No pocket is selected retrospectively by an optimized score or redefined
using later ligand states.

Each context resolves the same biological residue IDs through real protein
token masks and the selected ligand through its native affinity token mask.
Padding is excluded. Empty/overlapping masks or missing/ambiguous pocket
residues are errors. Discrete occupancy changes can alter ligand count and
slot identities between steps; they do not redefine the frozen protein
pocket. The initial pose is an engineering reference, not evidence of the
true binding pocket, particularly for random starts or unconstrained targets.
The draft matched-preparation helper uses the historical **anchor-count
prepared reference pose**, not the newly decoded random graph. Starts sharing
the same anchor context share that frozen residue pocket; changing selected
atom count does not move the reference used to select it.
`DBMolObjectiveConfig.verify_reference` checks the prepared anchor-file SHA
and its tensor/context digest against that stored reference before execution.
A matching residue list alone does not satisfy reference authentication.

## Selecting the profile and preparing matched comparisons

[ActiveSetConfig](../src/latentmol/boltz/active_set.py) accepts an optional
`dbmol` mapping parsed by `DBMolObjectiveConfig.from_mapping`. Selecting it
requires `objective.output: dbmol_total_loss`, `direction: minimize`, objective
scale one/offset zero and `fd_policy: local_v1`. A total without its explicit
profile, or a profile with another objective, fails configuration parsing.
The mapping must pin a confidence-checkpoint path and 64-character SHA-256,
and nonempty, unique native pocket residues. The existing affinity checkpoint,
decoder, selected contexts and saved initial latent remain authenticated by
the surrounding config. Omitting `dbmol` preserves historical record/default
behavior rather than retrofitting old runs to the new loss.

[prepare_dbmol_comparison.py](../scripts/prepare_dbmol_comparison.py) is a
draft **HPC-only numerical preprocessing** entry point. It verifies a pinned
historical stability protocol and its saved comparison evidence, derives the
anchor pocket, writes `pockets.json` and `comparison.json`, and materializes
one config per prescribed start, objective and occupancy arm. The default
starts are `carbon-c13` and `prior-0`, with binary versus composite objectives
and fixed-initial-mask versus active-set arms: **eight configs**, each with
three requested updates, not eight passed runs. Both objectives use the same
backtracking policy and exact saved initial latent. No random redraw,
re-encoding or trained input surrogate is introduced.

The preparation command template (the actual CPU stage now executed it with
authenticated paths/hashes recorded above) is:

```text
python scripts/prepare_dbmol_comparison.py --protocol AUTHENTICATED_PROTOCOL.json --sha256 PROTOCOL_SHA256 --output UNIQUE_OUTPUT --steps 3 --starts carbon-c13 prior-0
```

This is a command template; the named protocol/output placeholders do not
establish actual file identities. The helper requires `LSB_JOBID` and the
immutable source identity environment, rejects an existing output directory
and hashes prepared configs/pocket artifacts. Its three-update preparation
requests a derivative audit every update and argmax plus one categorical
draw; longer prepared configs use the explicit 50-update audit interval and
eight draws. Preparation does not itself execute optimization, repair or
fresh-native prediction, and it does not authorize a full matched campaign
before the new native path passes validation.

The optional named-model example
[multimodel_boltz_dbmol.yaml](../configs/multimodel_boltz_dbmol.yaml) selects
the profile through `models.boltz2.options.dbmol` and names
`dbmol_total_loss` in the objective. It is an **unvalidated template with
placeholder assets/pocket indices**, not the prescribed matched-smoke config.
The generic named-model runner audits its selected total; it does not replace
the active-set runner's new component gates. Validate that explicit path first
before treating generic composition as validated per-component guidance.

## Shared graph and gradient path

```text
optimized z -> frozen VAE decoder -> one shared conditional RelaxedGraph
  -> frozen affinity-checkpoint input/trunk/heads -> affinity terms
  -> frozen confidence-checkpoint input/trunk/distogram/confidence heads
       with fixed coordinates -> contact and confidence terms
  -> weighted total -> gradients back to z
```

The draft [FrozenBoltzDBMol](../src/latentmol/boltz/dbmol_model.py) wrapper
combines a logits-enabled affinity scorer with the strict separately loaded
structural checkpoint. Contacts and confidence share the structural trunk.
The affinity and confidence checkpoints have **separate native trunks** and
explicitly authenticated weights. A transplant of structural heads onto the
affinity trunk is not assumed equivalent. Both paths consume the same graph,
protein context, occupied slot identity and declared fixed pose. They may
translate native features but must not independently decode different chemical
identities. The existing lossy charge feature and graph/pose restrictions
remain disclosed rather than being solved by the extra terms.

Gradients pass through each attached native head/trunk and the shared graph
back to molecular `z`. Model parameters are frozen. Coordinates, conformers,
protein/MSA metadata and pocket membership remain fixed. **No gradient through
diffusion, pose generation, chemistry sanitization, repair, sampling or atom
birth/death is claimed.** The discrete active-set step can still change a
prepared context between differentiable updates; old/new context losses need
separate reporting and no global monotonicity claim.

The draft `attached_confidence` mirrors the native batch-one confidence
forward while retaining the trunk/head graph. It replaces the optional integer
bond-type embedding lookup with its expectation under the same relaxed typed bonds.
Hard one-hot behavior must match the native lookup. An unsupported discrete
bond path or hidden metric fallback must be rejected. These integrations are
implementation requirements awaiting native validation, not evidence that
the current confidence branch already passes them.

`strict_confidence_scores` recomputes native PTM/interface aggregation outside
the native helper's silent exception fallback. It requires a nondegenerate
masked interface frame and finite scalar outputs even in forward-only parity
calls. This is strict error handling, not evidence of more accurate confidence.
Before calling the native frame helper, `validate_frame_context` rejects
unsupported batch/frame shapes or indices, nonfinite coordinates, unsorted
or noncontiguous chain axes, padding that is not a real-token/atom prefix,
inconsistent atom/token chains and nonpolymer
layouts without one atom per token. The native helper can swallow an assignment
exception for such layouts; the profile rejects them rather than accepting
a plausible scalar from a failed frame computation.
`discrete_reference` calls the native integer trunk and original confidence
module at the identical saved pose, with structure generation disabled;
it is a proposed hard-input oracle, not a fresh stochastic prediction.

## Validation and matched experiments

Before full optimization, freeze source/config/checkpoint/graph/context/pocket
identities and run all numerical work on HPC. The planned gates are:

1. Native hard-input parity for each structural and affinity output at identical
   pose, masks, precision and recycling settings; old affinity parity does not
   certify newly attached structural heads.
2. Attached finite gradients back to `z`, a nonzero total derivative, unchanged
   frozen weights and component/total local finite-difference evidence. Record masks and
   piecewise contact-minimum winners; do not conceal nonsmooth probes.
3. Component reductions, signs and weights verified independently, with
   missing metric, masked pair and unsupported-mode failures retained.
4. Short real-model execution/terminal smoke before an immutable matched
   experiment from the exact same saved starts. Comparator objective, occupancy,
   optimizer, sampling, fresh-native eligibility and physical compute budgets
   remain explicit in a separately sealed protocol.

The draft active-set audit separately differentiates the three unweighted
component losses before the total, counts those backward calls, and runs
`local_v1` on each nonzero component and the total. A component gradient norm
at or below `1e-12` is retained as `zero_at_current_point_uninformative`,
not called detached or passed finite-difference evidence. A nonzero component
whose local gate fails rejects the run. A total-gate pass alone therefore
does not establish useful nonzero signal from every component; reports must
preserve each component's status.

Contact bookkeeping records the best ligand token per pocket residue, the
strongest anchor pair and, where a second ligand token exists, the smallest
best-versus-second loss margin. Finite-difference endpoints retain these
witnesses and `contact_winners_changed` when plus/minus select different
pairs. They help distinguish piecewise minima from an unexplained derivative
mismatch without altering the differentiable loss or declaring a failed
probe passed. An actual nonfinite scalar head is classified as
`NonfiniteObjective` for bounded trial rejection; malformed/missing heads,
unsupported frames and programming errors still propagate.

## CPU gate and four-case native smoke protocol

The new [submit_dbmol_comparison.py](../scripts/hpc/submit_dbmol_comparison.py)
prints a verified plan by default; `--submit` is a later explicit execution
step. It verifies the immutable archive/source manifest and pinned historical
protocol, uses durable unique intents/receipts, and wraps both stages with the
per-job immutable release verifier. The actual source065 archive/CPU job above
are now authenticated submission evidence. Native job 29619449 is queued;
its outcome remains pending.

The **CPU stage** requests queue `hpc`, two cores, 4 GB per core (8 GB host)
and 30 minutes. [boltz_dbmol_preflight.sh](../scripts/hpc/boltz_dbmol_preflight.sh)
first runs the selected new profile/gradient/model fixtures and prior contract
regressions through
[boltz_dbmol_tests.sh](../scripts/hpc/boltz_dbmol_tests.sh), then prepares the
eight matched configs and frozen pockets in the same HPC allocation. CPU
fixtures and preparation now passed, with the actual comparison SHA above. Passing
engineering fixtures alone would not certify the native pretrained heads.

After the CPU/preparation receipt succeeded and its actual comparison SHA was
known, the **native stage** was submitted with `done(29619428)` dependency.
It requests queue `gpua100`, one exclusive-process GPU selected for **80 GB
GPU memory**, four cores, 16 GB per core (64 GB host) and **6 hours 15 minutes**.
Actual hardware/allocation and resources used remain receipt fields, not
measured estimates. This initially runs only the four fixed-initial-mask
configs, serially in one allocation:

1. `carbon-c13-binary-fixed_initial_mask`
2. `carbon-c13-dbmol-fixed_initial_mask`
3. `prior-0-binary-fixed_initial_mask`
4. `prior-0-dbmol-fixed_initial_mask`

Each child is capped at **5,400 seconds**, with three requested updates and
per-update audits. The four child caps sum to six hours; setup/controller
work fits only the remaining declared allocation margin. The generated four
active-set configs are **prepared but not selected for this first smoke**;
this stage cannot certify variable occupancy. Full 100-update objective
comparisons are later experiments, not implied by these three-update gates.

[boltz_dbmol_smoke.sh](../scripts/hpc/boltz_dbmol_smoke.sh) invokes the serial
[run_dbmol_comparison.py](../scripts/run_dbmol_comparison.py) controller.
It preserves each child's failure/timeout and proceeds to the remaining
prescribed cases without replacing or screening outcomes. A case passes only
after authenticated source/config/child evidence, unchanged frozen states and
all three accepted updates under the declared arm are verified. An early stall
is retained as a failed full smoke budget, not convergence. The aggregate
`native_smoke_passed` requires every selected case; the controller records
`full_trajectory_validation: false` even on a smoke pass. No repair or fresh
native pose generation is hidden in this scope.

The staged submit template is:

```text
python scripts/hpc/submit_dbmol_comparison.py --stage cpu --release IMMUTABLE_RELEASE --archive-sha256 ARCHIVE_SHA256 --protocol PROTOCOL.json --protocol-sha256 PROTOCOL_SHA256 --prepared-root UNIQUE_PREPARED_ROOT --receipts UNIQUE_RECEIPTS
python scripts/hpc/submit_dbmol_comparison.py --stage native --release IMMUTABLE_RELEASE --archive-sha256 ARCHIVE_SHA256 --protocol PROTOCOL.json --protocol-sha256 PROTOCOL_SHA256 --prepared-root UNIQUE_PREPARED_ROOT --comparison-sha256 ACTUAL_COMPARISON_SHA256 --receipts UNIQUE_RECEIPTS
```

These print plans without `--submit`; the executed commands use the actual
authenticated files and HPC receipts in the ledger. Do not fabricate a comparison hash before CPU pocket
preparation or call prepared/submitted stages passed. Existing source064 jobs
are unchanged while this separate profile is prepared and validated.

Pre-seal local review parsed 163 Python source/script/test files with
`ast.parse` under the bundled Python interpreter, without project imports,
models or numerical tests. Static syntax success is not native validation;
source065 packaging/submission and CPU fixtures/preparation now have the
identities/evidence above, while native gate outcomes still await receipts.

Record raw terms, weighted terms, total loss, gradients, accepted/rejected
trials, context switches and physical oracle calls at the chosen cadence.
Separate the affinity and structural paths' forward/backward cost; a shared
graph does not mean a single native network execution. Draft per-state metrics
include `loss_profile`, the raw/weighted `components` and
`physical_trunks_per_forward: 2`; both models' checkpoint digests are combined
in branch provenance. A logical scorer call should not be reported as one
physical trunk. Component finite-difference forwards currently execute the
complete composite branch even when only one scalar is inspected. Gradient-component
diagnostics themselves consume work and must be counted. Match starts and
requested updates without calling unequal oracle budgets equal compute.

Chemical validity, sampling support and bounded probability-aware repairs
remain separate evaluation diagnostics. Structural confidence/contact are
not a chemical-validity, synthesizability or selectivity prior. Fresh native
pose generation and audited hard identity still test whether any relaxed
gain survives outside the fixed-pose proxy. A confidence or total-loss gain
does not establish measured binding.

Implementation, numerical gate, submission and outcome receipts belong in
the [ledger](05_experiment_ledger.md). Current source064 continues as the
sealed affinity-only stability control; no composite config has been
substituted into its running jobs. The release and input protocol above are
pinned, with prepared pocket/config identity and CPU validation now recorded.
Native GPU submission is confirmed; its model-gate/outcome evidence remains
pending.
