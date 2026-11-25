# Frozen epoch-36 VAE to Boltz early-embedding bridge

## Attempt002 completed the mapping pilot, 23 October 2026

The [terminal snapshot](../outputs/hpc-results/source137-epoch36-boltz-embedding-bridge-attempt002/status-1792782131360276800.json)
collected at approximately **21:02 DTU-local** supersedes the pending observations.
Saved LSF output confirms **GPU 29796624 completed successfully**, executing on
**n-62-18-10 from 19:41:35 to 19:46:30 DTU-local**. Its live scheduler record
has aged out; that is not a failure. The full runner took **283.505s**.

The exact native-oracle injection forward and constrained-gradient comparisons
passed (`passed_native_oracle_only`); gradient relative error was
**1.749677e-6**, with frozen modules and context unchanged. FD observations
remain diagnostic. The supervised fit completed **1,000/1,000 updates** in
**41.549s**, selecting update 1,000 by development loss. It contains **246,624
trainable bridge parameters**, with VAE and projection-bank state unchanged.

Paired preparation accepted **2,031/2,176 attempts**: train **1,629**, development
**209**, audit **193**. Rejections were 92 sanitization, 42 disconnected and
11 alignment failures. The audit contains **71 distinct decoded identities**;
posterior/local descendants are correlated. Count-weighted nearest-prototype
identity accuracy was **99.9045% for element**, **99.9500% for bond type**, and
**99.9457% for trunk connectivity**. Atom denominators are 4,188 and directed
occupied-pair denominators 92,090, including native NONE diagonals. These are
feature identities, not whole-molecule success rates. Charge accuracy was 100%
on neutral atoms only; charged categories, phosphorus and iodine have no audit
coverage and are not validated by this result.

The learned fixed-context diagnostic decoded strict-valid tridecane
(`CCCCCCCCCCCCC`), with identical natural/conditioned hard labels and finite
latent gradient norm **0.145906**. Learned total loss was **11.072826**, versus
**11.018015** through the hard chemical path, a difference of **0.0548104** on
this single example. This is neither fresh native re-featurization nor a
learned-gradient usefulness test. No latent optimization ran. The fit retains
`trained_unvalidated` because this diagnostic pilot selected no acceptance
thresholds; the outer status is
`prototype_fit_completed_not_optimization_validated`.

The [independent byte-authentication receipt](../outputs/hpc-results/source137-epoch36-boltz-embedding-bridge-attempt002/terminal-authentication-1792782183894979900.json)
confirms checkpoint, data payload, attempt log, dataset manifest and training-plan
hashes against their receipts, plus history entries at updates 1/100/.../1,000.
The saved bridge is `gpu/fit/best-bridge.pt` under the attempt002 campaign;
SHA256 **`013834958d52f7aee999ca4c39496e14fc5333d69eb697d47433821f48290343`**.
The source/plan identities remain those of the sealed attempt002 below. This
read-only collection changed no code, scientific settings or jobs; the existing
VAE continuation remains untouched.

**Earlier read-only observation, 23 October 2026 12:32 DTU-local:** the
[combined saved snapshot](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-016.json)
confirms **GPU 29796624 PEND** with all four CPU slots reserved on n-62-18-11
since 12:17:16. The pending reason remains insufficient processor units for
the host affinity request; ineligible pending time is zero. There are no GPU
artifact paths or log files, native checks or fitting results, and no reliable
start estimate is displayed. The 139-pass allocated CPU result remains valid;
the original VAE and source136 continuation are unchanged. This was one
read-only SSH session, with no settings or job changes.

Status, **23 October 2026 12:22 DTU-local**: prototype implemented on
`codex/epoch36-boltz-embedding-bridge`, branched from `e2f5a2c`. Independent
corrected attempt002 staged **569 authenticated members**; **CPU 29796619 DONE**
at **10:47:38 DTU-local**, with **139 passing tests**, zero failures/errors/skips
and original epoch-36 decoder bitwise parity. **GPU 29796624 PEND**, submitted
at **10:47:59 DTU-local**, one
exclusive 80GB A100/four CPU slots/four-hour allocation, 3h50 internal cap.
All four CPU slots are now reserved on **n-62-18-11** from **12:17:16**,
superseding the earlier two-slot reservation on n-62-18-8. The previous
provisional forecast is absent; no reliable current ETA is established.
No CUDA/native check or bridge-fitting result exists yet.
Attempt001's **128-test CPU pass** is preserved; its operator-authentication
gap prevented GPU promotion and was corrected only in the new sealed attempt.
All existing VAE-continuation and Protenix records remain unchanged.

The experiment fits the active [AE/VAE/frozen-Boltz pipeline](04_latent_pipeline.md)
and the [small supervised bridge exception](23_frozen_model_plugin_contract.md#permitted-bridge-training).
Its purpose is to test whether final VAE atom and pair hidden states can supply
useful continuous chemical embeddings to the existing frozen Boltz computation,
while retaining agreement with the molecule decoded by the ordinary heads.
It does not replace the external model with a learned activity, validity or
chemistry scorer.

The current independent experiment is
`source137-epoch36-boltz-embedding-bridge-attempt002`, with configuration
`configs/epoch36_boltz_embedding_bridge.json` and modules
`latentmol.boltz.embedding_bridge`, `embedding_injection` and
`embedding_training`, plus the allocated runner/native validation and immutable
operator. Sealing and initial CPU execution evidence follow; no native result
is established by the implementation.

The configured first diagnostic pilot selects **512/64/64 source-training root
identities** for bridge train/development/audit, plus **128 fresh prior roots**.
It uses seed **137036**, bridge hidden width **128**, **1,000 optimizer updates**,
batch size **16**, learning rate **0.001**, and development evaluation every
**100 updates** (also the first/final update). These are requested settings,
not completed data counts or a trained model. Native injection/gradient checks
precede fitting. No feature-promotion thresholds are selected for this first
diagnostic fit; all audit metrics are reported and optimization remains disabled.

## Attempt002 CPU passed; native GPU submitted

The [23 October 12:22 DTU-local read-only snapshot](../outputs/hpc-results/source137-epoch36-boltz-embedding-bridge-attempt002/status-1792750976337669100.json)
confirms **GPU 29796624 PEND**, all **four CPU slots reserved on n-62-18-11**
since **12:17:16**, with insufficient processor units for the affinity request
on one host. Eligible pending time is 5,697s; ineligible time is zero. The prior
24 October 02:32:54 forecast is no longer displayed, so there is **no reliable
latest ETA**. No GPU artifact paths, launch-quota receipt, native checks or
fitting are present. CPU 29796619 has aged out of scheduler history; its saved
139-test/zero-failure/error/skip pass and original-36 parity remain valid.
This status collection changed no code, configuration or jobs.

The corrected immutable
`source137-epoch36-boltz-embedding-bridge-attempt002`
[seal](../outputs/hpc-submissions/source137-epoch36-boltz-embedding-bridge-attempt002/seal.json)
has **569 members**, including the launch-time archive/manifest binding
correction and its operator fixtures. The
[CPU submission](../outputs/hpc-submissions/source137-epoch36-boltz-embedding-bridge-attempt002/submit-cpu-1792745218342767600.json)
returned zero. The
[saved CPU terminal status](../outputs/hpc-results/source137-epoch36-boltz-embedding-bridge-attempt002/status-1792745265419026200.json)
records **CPU 29796619 DONE**, **10:47:03–10:47:38 DTU-local**, n-62-30-25,
**139 tests passed in 22.05s** (JUnit duration 21.999s), no failures/errors/skips,
strict original-36 loading and bitwise ordinary/capture output parity. The
full allocated runner completed in 27.098s. These are actual CPU contracts,
not native GPU or molecular optimization evidence.

| Attempt002 binding | Actual value |
| --- | --- |
| Archive SHA256 | `8209d548ec0420fce5a850bae4d1a900cfda6fee7d7c4672cec35e460dd08923` |
| Manifest SHA256 | `aecd3f93e9c422f0cae3ca6eeeb635f4641474b6fe49ddf3ca50347e93192c71` |
| Plan SHA256 | `34db99c4714bf0ccd5e0a83d5a3b54fa5f85f59dc8416b422d25e8a20f5b5ad5` |
| Original VAE tensor-state SHA256 | `bca36c9c7dcad67fa7eed8c2996f9304032738cfd5a3b8957147e99c3cde2bec` |
| CPU JUnit SHA256 | `4754a7245caf941fdb409d5f69fc31ac3a21e6b18e3e7e3d59b9268e01f20ba0` |
| Native GPU job | `29796624`, PEND, submitted 10:47:59 DTU-local |
| Fresh GPU-submission quota | 74.39/100 GiB used; 25.61 GiB free |

The [actual GPU submission receipt](../outputs/hpc-submissions/source137-epoch36-boltz-embedding-bridge-attempt002/submit-gpu-1792745276851064400.json)
returned zero, `submitted_not_passed`, **29796624** on `gpua100`: four CPU
slots, one exclusive 80GB A100, four-hour walltime and 3h50 internal runtime.
It is independent of the untouched source136 continuation and uses new
release/campaign paths ending in `attempt002`. No actual GPU start, native
gate, prepared paired data, fitted weights or learned diagnostic is established
by that submission receipt.

The [earlier scheduler snapshot](../outputs/hpc-results/source137-epoch36-boltz-embedding-bridge-attempt002/status-1792745360088722400.json)
confirms **29796624 PEND**, with **two of four CPU slots reserved on
n-62-18-8** from 10:48:01 DTU-local. The pending reason is insufficient
processor units for the affinity request on one host. Its provisional
**no-sooner-than 24 October 02:32:54 DTU-local** forecast is not a reliable
guaranteed start time. Reservation is not GPU execution; no GPU `result.json`,
native checks or fitting artifacts exist in this snapshot.

The native stage gates exact/soft-oracle embedding-injection forward and
automatic-differentiation parity. Its finite-difference observations are
diagnostics; they are **not a passed learned-bridge finite-difference gate**.
Even a completed native oracle pass and diagnostic fit would leave molecular
optimization disabled until the separate learned identity/derivative/native
candidate requirements pass. No current result establishes those later checks.

## Attempt001 seal, CPU submission and operator correction

The [seal](../outputs/hpc-submissions/source137-epoch36-boltz-embedding-bridge-attempt001/seal.json)
binds **568 members**. The
[staging receipt](../outputs/hpc-submissions/source137-epoch36-boltz-embedding-bridge-attempt001/stage.json)
returned zero and reports `staged_authenticated`, 568 members. The
[CPU submission receipt](../outputs/hpc-submissions/source137-epoch36-boltz-embedding-bridge-attempt001/submit-cpu-1792745069622880300.json)
returned zero, `submitted_not_passed`, **29796597**, four slots, 8 GiB request,
45-minute allocation on `hpc`. The
[first saved status](../outputs/hpc-results/source137-epoch36-boltz-embedding-bridge-attempt001/status-1792745089497102300.json)
records actual RUN from 10:44:33 DTU-local, `running_not_passed`. No JUnit/native
gate/fitting outcome is present in that snapshot.

| Binding or observation | Actual value |
| --- | --- |
| Archive SHA256 | `ae4281fdc971bf8874fe557576d1439f9b84943b2250ca69a4bf223afd7940e0` |
| Manifest SHA256 | `88d23ee54df9acea13b9c406fadcfc38438b8fe9189212b11549479127fc8095` |
| Plan SHA256 | `1565e0261a422f13b0b6feba1285fa267bc4416915039ee5c8bf3be98d9e884b` |
| CPU job / start | `29796597`, 23 October 2026 10:44:33 DTU-local |
| CPU host / state | `n-62-30-25`, RUN / `running_not_passed` in first snapshot |
| Fresh pre-submission quota | 74.29/100 GiB used; 25.71 GiB free |
| Launch quota in saved status | 74.54/100 GiB used; 25.46 GiB free |
| GPU / supervised fit / optimization | Not submitted / not established / not run |

Release:
`/work3/s234818/latentmol/releases/source-137-epoch36-boltz-embedding-bridge-attempt001`.
Campaign:
`/work3/s234818/latentmol/runs/source137-epoch36-boltz-embedding-bridge-attempt001`.
Operator commands in the saved receipts bind `--archive-sha256`,
`--plan-sha256`, stage, release and campaign. The allocated wrapper uses the
project's `env/boltz311` environment through `scripts/hpc/boltz_run.sh`.

The independent native-interface reviewer found a launch-authentication gap:
the operator verified the archive SHA and extracted member hashes, but did not
recompare the extracted manifest against the manifest bytes inside that
authenticated archive at launch. The working source now reads that archived
manifest and requires exact equality before consuming its member map. This is
a source/provenance correction, not a numerical or chemistry fix. Preserve
attempt001's sealed source and CPU results as evidence; **no attempt001 GPU
promotion is permitted**. A new attempt002 was planned after the CPU unit
outcome, with its own seal, current tests and receipts. No new attempt002 seal,
job or pass follows from this preparation note.

The later
[attempt001 terminal snapshot](../outputs/hpc-results/source137-epoch36-boltz-embedding-bridge-attempt001/status-1792745155243000600.json)
supersedes its RUN state: **CPU 29796597 DONE at 10:45:15 DTU-local**, **128
tests passed in 18.93s**, zero failures/errors/skips. Strict original-36 loading,
bitwise original/capture decoder parity and unchanged VAE state passed. This
valid CPU evidence remains preserved with attempt001's source/plan; the launch
authentication gap still prevents its GPU promotion. Attempt002 above is a new
seal and new allocated validation, not a rewrite of this attempt.

## Authorization and independent lineage

The user's instruction was to implement the prototype using **frozen original
epoch 36**, with permission to train or submit HPC work, and to leave the
existing VAE continuation unchanged. The exact VAE checkpoint SHA256 is:

`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.

This is the authentic completed source121 epoch-36 checkpoint. Source135's
disposable probe/proof weights are not this model and remain discarded. The
bridge is tied to the exact frozen checkpoint and decoder/source/schema
identity. It cannot silently be carried over to epoch 37 or 40 after a VAE
continuation; a new checkpoint requires a new binding and validation, and may
require bridge retraining.

All existing source136 source, plan, seals, jobs, callbacks, loss settings and
epoch-40 endpoint remain unchanged. Its last saved observation in
[docs75](75_frozen_reference_epoch40_continuation.md) is 23 October 10:22
DTU-local: GPU 29792164 and callback 29792165 PEND, with all 16 CPU slots
reserved on n-62-18-10. That is historical scheduler evidence, not a fresh
status check for this prototype and not an execution or CUDA pass. The separate
[Protenix study](73_protenix_soft_occupancy_study.md), earlier failures and all
their immutable artifacts remain independent and preserved.

## What is learned and what stays frozen

Only small shared atom and symmetric pair mapping heads are trainable during
bridge fitting. The complete VAE, including ordinary atom/charge/bond heads,
and the selected pretrained Boltz checkpoints stay frozen in evaluation mode.
Native early chemical features of the same decoded hard graph supply the
supervision. No desired affinity, confidence, validity, synthesis or combined
optimization reward is a bridge training target.

The final VAE hidden states are detached while fitting the bridge, so fitting
cannot update the VAE. During later gated latent search, VAE and bridge weights
remain frozen but their forward computations retain the path back to `z`.
Freezing weights does not mean placing the latent-dependent forward under
`no_grad`. Training and optimization use separate parameter/RNG scopes and
separate artifacts.

The ordinary VAE output remains available at every latent. Its decoded graph
provides the common molecular identity, occupancy, charge and bond audit for
all external branches. A model-specific embedding head is an approximation to
that identity, not authority to invent a different molecule for Boltz.

## Decoder interface and prototype context

The selected refined triangle VAE uses the existing 50-slot graph schema and
256-dimensional latent. Its final atom states are 96-dimensional and pair
states are 48-dimensional. The additive inspection interface must expose the
actual final atom states, charge context and complete bond-head inputs. A pair
state alone need not contain everything consumed by the final bond head.

Ordinary decoder calls, output tensors and checkpoint state-dict keys must
remain unchanged. Allocated tests must verify strict loading and numerical
parity between decoding with and without feature capture. Exposing features
must not detach the later latent gradient.

The initial native scoring scope is the existing fixed-occupancy, fixed-pose
Boltz engineering context: prepared atom/token maps, protein/reference features,
coordinates and masks have explicit identities and stay fixed during a probe
or trajectory. The existing seed has 13 occupied slots. The training cohort's
default includes valid hard-decode rows of all supported sizes, using their
actual occupied-atom and pair masks. Fitting chemical features on more sizes
does not establish variable-occupancy native gradients or new geometry: the
native scoring prototype remains tied to the existing fixed context.

The bridge bypasses the last categorical projection on the scorer path. It
does not remove the softmax-based feedback already inside VAE refinement and
does not by itself fix any demonstrated gradient failure in the old path.

## Native chemical targets and exact injection

Before training, the implementation must define precise injection boundaries
against the pinned Boltz source and weights and first test **exact native
feature injection**. Equivalent native early tensors, rather than learned
predictions, are injected at those same boundaries and compared with the
untouched native computation under identical context. This isolates adapter
engineering from approximation error.

Targets cover chemical contributions consumed by the selected objective:
element and formal-charge contributions at the early atom projection, bond
connectivity and typed-bond embeddings, including separate learned projections
in affinity and structural/confidence components where applicable. A projection
from one component cannot stand in for a different component's weights.
Projection weights are taken from the authenticated frozen checkpoint;
training targets are not newly initialized representations.

Reference coordinates, atom names, token assignments, masks and other fixed
features remain explicit context. If a target channel depends on a context not
available to the VAE, retain that context explicitly or declare the channel
unsupported. Do not ask the bridge to infer unprovided pose/protein information
and then present the result as native equivalence.

## Paired data and molecular consistency

For each latent `z`, save or reproducibly reconstruct its ordinary decoder
output, final hidden states, hard graph and graph-policy identity. The target
describes the **molecule actually decoded at that `z`**, not necessarily the
input molecule that was encoded. A reconstruction mismatch cannot silently
retain the input molecule's Boltz target.

The initial cohort should draw only from permitted training identities, with
separate bridge-development and audit partitions. Posterior samples, prior
attempts and small local perturbations have distinct labels and counts. Group
related latent perturbations together in a split, and check molecular-identity
collisions between splits. Preserve the VAE's reserved test set. Exact cohorts,
seeds, perturbation scales and attempt budgets must be bound in the experiment
plan before execution.

Native preprocessing may reorder atoms, so every example retains an explicit
map from VAE slots to native atoms/tokens. Require exact element, formal charge,
connectivity and typed-bond identity under that map. Unsupported vocabulary,
invalid chemistry, representation failure or alignment failure is recorded with
its reason; such rows supply no invented native target. No silent repair,
neutralization, atom removal or native substitution is allowed.

The same latent under the same graph/context policy must define the same target.
Nearby latents that hard-decode to the same molecule share a native chemical
target; if they decode different chemistry, their targets change accordingly.
Consistency examples constrain the learned interpolation without modifying the
VAE's molecular training or the native target construction.

Shared atom heads and symmetric pair features preserve slotwise reuse and
undirected bond symmetry. Losses must normalize atom/charge and bond channels
separately and distinguish bonded/nonbonded coverage so abundant NONE pairs do
not hide bond errors. Record each component rather than only a combined MSE.
Final normalization, weights, architecture, parameter count, optimizer and
selection policy are explicit experiment settings, not validation outcomes.

## Required gates and interpretation

The prototype has separate stages; submission or fitting loss cannot stand in
for their completion:

1. **Implementation and provenance:** allocated contracts cover strict epoch-36
   checkpoint loading, unchanged ordinary decoding/state keys, symmetric
   feature/mapping behavior, correct atom maps, masks and target projections,
   and rejection of unsupported or mismatched identities.
2. **Exact native injection:** reproduce untouched native features and forwards
   under the same frozen context before fitting. Bind both Boltz checkpoint
   identities, source, configuration, prepared assets and execution policy.
3. **Supervised bridge fitting and held-out audit:** train only mapping
   parameters. Record attempted/accepted/rejected and unique molecular counts,
   channel-specific feature errors, decoded identity agreement, split overlap,
   native output discrepancies and selection history. Predeclare approximate
   acceptance thresholds before examining the reserved bridge audit set.
4. **Derivative and frozen-state validation:** verify finite/nonzero supported
   chemical and latent gradients, independent chain-rule checks and multiscale
   finite differences of the implemented smooth objective. Recheck unchanged
   VAE/Boltz/bridge weights and context. Approximate native feature agreement
   and a finite derivative are different claims.
5. **Matched optimization and native confirmation:** only after the relevant
   gates pass, compare the learned embedding path with the existing probability
   path and a matched random-search control from identical frozen checkpoint,
   starts, context, objective and budgets. Rebuild ordinary native inputs for
   the resulting hard molecules and retain failed/unsupported attempts.

The future latent comparison uses the explicitly authorized `best_trial_v2`
step policy. Full-step nonincrease, first strict improvement, forced finite
fallback, improving/applied/no-op counts and best-observed context-bound states
must be reported separately. This does not waive provenance, finite-value,
native-gradient, trust-radius or resource checks. No bridge-training outcome
automatically selects this new path for existing trajectories.

Success requires improved **valid, changed discrete molecules** whose gains
survive independently rebuilt native scoring under the declared comparison.
Record diversity/identity collapse, element/charge/size shifts, unsupported
fractions and actual cost, plus the standard saved-output HTML report.
Relaxed score improvement alone is insufficient.

The hard graph is piecewise constant in `z`, while the learned bridge supplies
a smooth interpolation. Even low feature error and exact labels at training
points cannot guarantee a chemically faithful derivative everywhere. The
experiment must measure interpolation failure and relaxed/native divergence;
it cannot claim a general validity guarantee or a validated gradient merely
from a successful supervised fit.

## Implemented data and fitting interfaces

The additive `embedding_training.py` implementation provides
`prepare_dataset(vae, data_path, output, seed, train_count=4096, dev_count=512,
audit_count=512, prior_attempts=1024, *, local_scale=0.05, deadline=None)` and
`train_bridge(vae, bank, dataset_path, output, device, steps=2000, batch_size=32,
learning_rate=0.001, seed=137, deadline=None, *, hidden_dim=128,
audit_thresholds=None, evaluate_every=100)`. The pilot configuration overrides
these library defaults as declared above. A deadline is an absolute
`time.monotonic()` value.

The caller authenticates the original epoch-36 file, loads it strictly and
binds `vae.embedding_bridge_checkpoint_identity` to its epoch and file SHA.
Preparation/fitting require frozen evaluation mode and compare full VAE tensor
digests before and after; the checkpoint-file SHA is distinct from a model-state
digest. The native projection bank is also checked unchanged during fitting.

Source-root partitions are allocated before descendants. Each source root
supplies mean/posterior/local attempts; each fresh-prior root supplies natural
and local attempts. The root's natural decode fixes its descendants' occupied
slots. Hard element selection is then conditioned on that mask, exactly like
the existing fixed-occupancy scoring policy. This conditioning is explicit,
shared and unmodified across native-label construction and later fitting.

Compact `samples.pt` holds z, categorical hard labels, masks, explicit
slot-to-canonical maps and group/split/identity labels. It does not cache final
pair hidden states. `attempts.jsonl` records rejected/invalid rows as well as
accepted ones. Selected root identities are reserved to their partitions;
decoded identity collisions across partitions are rejected. Source validation/
test SMILES are used only as an identity-exclusion list, without encoding or
evaluating their molecular graphs. Every training/evaluation batch recomputes
the frozen VAE hidden states and checks that the stored hard teacher molecule
is reproduced at the stored z under the same mask.

Only bridge parameters enter AdamW. Checkpoint selection uses development
feature loss, preserving the earliest tied best state. The fixed audit set is
consumed once after fitting and does not update weights or choose a checkpoint.
Audits include per-projection and group-balanced errors, explicit counts,
per-element/formal-charge/connectivity/typed-bond identity denominators and
accuracy, plus supported-class macro accuracy. Pair counts are directed
occupied pairs including the native NONE diagonal, not independent molecules.
No threshold plan yields `trained_unvalidated`; even a future explicit feature
threshold pass remains `passed_feature_audit_not_native_validated`. Runtime
exhaustion remains `partial_runtime_not_passed`, with no fabricated audit.

The new CPU fixture file covers root splitting, fixed-policy parity, mapped
noncontiguous slots, synthetic supervised fitting/selection/frozen-state audit,
identity leakage, stored-teacher disagreement, checkpoint/frozen-scope guards,
retained preparation attempts and partial runtime. These are authored tests,
not numerical evidence until the allocated CPU job executes them. Synthetic
targets do not establish native Boltz parity or useful molecular optimization.

## Execution and evidence policy

All numerical preprocessing, contracts, fitting, native model validation and
evaluation run on allocated DTU HPC. Local work is limited to editing, static
review/AST/metadata validation, immutable packaging and authenticated transfer.
Use a new uniquely named source release, plan, run and output directory that
do not overwrite or mutate the source136 campaign or any Protenix artifact.
Include uncommitted code in source hashing; a branch or Git commit is not an
authenticated release by itself.

Each stage records source/manifest/plan/checkpoint/data/context hashes, exact
argv/environment, scheduler job IDs/resources, wall/internal limits,
stdout/stderr, structured terminal decisions and actual artifact receipts.
Allocations stay within 24 hours with a 23-hour internal bound; perform fresh
quota checks before substantive stages and do not overwrite existing outputs.
Authenticate completed prerequisite artifacts before dispatching a dependent
stage. Failures and partial work stay preserved under their own attempt
identities; a corrective retry requires a new immutable attempt.

**Current outcome:** corrected attempt002 completed **139 CPU fixture passes**,
native exact-oracle injection/gradient parity and **1,000 supervised mapping
updates**. The authenticated bridge is saved; feature audits and one learned
fixed-context diagnostic are reported above. Useful learned-gradient and
molecular-optimization validation remain outstanding. Attempt001's 128-pass
CPU evidence and operator limitation stay preserved; existing VAE continuation
is unchanged. Historical pending observations remain dated evidence.
