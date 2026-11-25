# Controlled Protenix soft-occupancy pilot

Design date: 22 October 2026. Study identity:
`protenix-soft-occupancy-pilot-v1`.

**Deletion/count-selection follow-up completed, 24 October 2026:** independent
source139 GPU **29803898 DONE**, fresh C/Z gates and all 18 eight-update
trajectories completed. **Z q=0 failed physical-deletion equivalence in all
three cases**, with charge/name/position/bond ghost effects. Soft gradient
optimization changed counts **8→7, 10→9, 11→10** from q=0.8, but worsened both
compact and fresh native endpoint losses; random-sign controls selected the
same counts. The foundation's gradient mechanism remains valid, while useful
absence/count selection is unsupported by this Z pilot.
[Follow-up evidence and limits](79_protenix_deletion_and_count_selection.md#12-completed-diagnostic-zero-element-mass-did-not-reproduce-deletion).

**Next deletion/count-selection phase authorized, 24 October 2026:** the user
requested testing whether NO_ATOM behaves like physical atom removal and
whether bounded optimization can select atom count. The separate prospective
[docs79 protocol](79_protenix_deletion_and_count_selection.md) retains this
completed foundation and all failures, and separates compact/fresh native
deletion, ghost diagnostics and Z count selection. No new job or numerical
result is established by that design; Z is not relabelled full G propagation.

**Attempt008 C/Z native parity and gradient gates passed, collected 24 October
2026 01:30 DTU-local:** **GPU 29802638 DONE**, executed **23 October
21:38:03–21:41:59**. All three cases passed required loss/TV output comparisons
and calibrated derivative agreement: **36 rows/108 FD-scale comparisons**.
C's 18 gradients were exactly zero; Z's 18 gradients were finite/nonzero and
17 rows had resolved sensitivity. Actual NO_ATOM-logit gradients were resolved
in all three cases even at **90% NO_ATOM probability**. This validates the
fixed-context element-mass gradient mechanism; real absence, variable atom
count and useful optimization remain unvalidated.
[Completed evidence and limits](#36-attempt008-completed-cz-native-parity-and-calibrated-gradients).

**Attempt008 CPU passed; native GPU queued, 23 October 2026:** CPU
**29802633** passed all **165 fixtures**, strict load, three preparations and
the real-input CPU feature probe. **GPU 29802638 submitted at 21:13:20
DTU-local**, still **PEND** about 21:15 with a host job-slot limit. No GPU
execution or calibrated derivative result exists yet, and the scheduler gives
no reliable start time. [Saved CPU evidence and actual GPU receipt](#35-attempt008-cpu-passed-native-gpu-submitted).

**Attempt008 sealed and CPU submitted, 23 October 2026:** **209 members
authenticated**, stage/queue return zero, **CPU 29802633 submitted**, four
slots/45 minutes. The calibrated loss-difference policy is sealed, but the
**165 expected fixtures**, strict load, preparations and actual-input CPU
preflight are not passed by submission. GPU parity/NO_ATOM checks remain
conditional on full readiness. [Actual receipt and immutable bindings](#34-attempt008-sealed-and-cpu-submitted).

**Noise-aware derivative retry authorized, 23 October 2026:** the user
explicitly requested fixing the noise-sensitive finite-difference check and
rerunning native parity and NO_ATOM gradients on DTU HPC. Prospective attempt008
declares `attached-repeat-loss-difference-fd-v1`: repeated identical-input
attached evaluations calibrate a capped empirical allowance, and agreement is
checked in loss-difference space. Resolved Z sensitivity remains a separate
required result; noise compatibility alone cannot pass the study. No job or
new numerical result is established yet. [Exact prospective policy](#33-attempt008-attached-repeat-calibration-and-loss-difference-validation).

**Read-only status, 23 October 2026 about 21:02 DTU-local:**
[collection005](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt007-collect-005.json)
returned zero. The native result hash is unchanged, and no replacement
occupancy-study job or new Z/NO_ATOM result exists. The scheduler does contain
**29798480 PEND**, `protenix-defog-1000`, but its command identifies a separate
Mosaic release/run, not this study. Its detailed pending reason is job-slot
limits on five hosts. The [derived read-only status receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/status-005-summary.json)
records that distinction and the initial non-login shell lookup failure.
No code, settings or jobs changed.

**Read-only status, 23 October 2026 about 12:22 DTU-local:** the
[fourth collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt007-collect-004.json)
returned zero with unchanged attempt007 results. No Protenix job is active or
queued, and no replacement has been submitted. CPU remains **128 passed**;
first-case output comparisons passed under loss/TV policy, but the first C
finite-difference check failed with one of three agreeing scales. There is no
new Z/NO_ATOM result or further progress. No code, settings or jobs changed.

**Read-only status, 23 October 2026 10:48:40–10:48:59 DTU-local:** the
[third collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt007-collect-003.json)
returned zero and confirms unchanged attempt007 results: CPU **128 passed**,
GPU **29796475 failed** at the first C direct-q derivative (one of three FD
scales passed). No Protenix job appears in the fresh active scheduler listing,
and no replacement submission or new Z/NO_ATOM result exists. No code, settings
or jobs changed. [Terminal evidence](#32-attempt007-output-policy-passed-first-control-derivative-gate-failed).

**Attempt007 finished with a derivative-gate failure, 23 October 2026:** CPU
**29796458 passed all 128 fixtures** and real-input readiness. GPU **29796475
EXIT / exit 1**, **10:26:07–10:26:48 DTU-local**. Native repeat, FP32-boundary,
C/Z and gradient-enabled q=1 output comparisons passed under the declared
loss/probability policy; original failed strict-logit diagnostics remain saved.
The first C-control derivative had autograd zero, but only one of three finite-
difference scales passed; all were below the declared 10× repeat-delta
resolution threshold.
The unchanged derivative gate stopped execution before Z or NO_ATOM-logit
derivatives. No gradient-correctness or occupancy conclusion follows yet.
[Actual results and limits](#32-attempt007-output-policy-passed-first-control-derivative-gate-failed).

**Attempt007 sealed and CPU submitted, 23 October 2026:** **209 members
authenticated**, stage/queue return zero, **CPU 29796458 submitted**. The
consistent output policy `native-output-loss-tv-v1` is declared; original
strict-logit diagnostics remain recorded, finite/loss/TV/derivative checks
remain required. The **128 expected fixtures** and real-input CPU readiness
are not passed by submission. [Actual receipt and sealed identities](#31-attempt007-sealed-and-cpu-submitted).

**Consistent raw-logit policy retry prepared, 23 October 2026:** attempt006
passed native repeat exactly but failed the still-active raw-logit criterion
at the FP32-boundary and C/Z comparisons. Prospective attempt007 applies the
user-requested removal of that criterion across all active parity comparisons,
recording raw-logit/strict outcomes as diagnostics while retaining finite
outputs, scalar-loss agreement, TV `<=1e-4` and all derivative/noise guards.
No numerical pass follows from preparation. [Actual attempt006 result](#29-attempt006-cpu-passed-native-repeat-exact-but-occupied-parity-failed)
and [prospective policy scope](#30-attempt007-consistent-output-loss-and-probability-policy).

**Attempt006 sealed and CPU submitted, 23 October 2026:** **209 members
authenticated**, stage/queue return zero, **CPU 29796415 submitted**. The scoped
`native-repeat-loss-tv-v1` policy affects only native-repeat raw-logit
acceptance; adapter and derivative gates remain unchanged. The **117 expected
fixtures**, strict load, preparations and actual-input CPU probe are not
passed by submission. [Actual receipt and immutable identities](#28-attempt006-sealed-and-cpu-submitted).

**Authorized scoped native-repeat policy retry, 23 October 2026:** the user
explicitly requested removing the single blocking native-repeat raw-logit
constraint and trying again. Prospective attempt006 retains that comparison
as a saved diagnostic, while finite native outputs, scalar-loss agreement and
the existing probability-TV limit still gate the native repeat. Every adapter,
FP32-boundary, gradient-enabled parity and finite-difference requirement stays
unchanged. No job or numerical pass is established by this preparation.
[Exact scope and retained requirements](#27-attempt006-user-authorized-native-repeat-logit-policy).

**Earlier terminal outcome, checked 23 October 2026: GPU 29792558 EXIT.** It ran
**22 October 21:17:19–21:17:56 DTU-local**, exit 1. The untouched native model's
repeat evaluation itself failed the raw-logit tolerance (maximum difference
**4.959106e-5**); no C/Z comparison or gradient check was reached. CPU's 106
fixture passes and exact actual-input parity/backward evidence remain valid.
The current blocker is GPU native repeatability; its cause is unproven and the
NO_ATOM hypothesis still has no result. [Saved failure and measurements](#26-attempt005-native-gpu-repeatability-failed-before-adapter-checks).

**Attempt005 CPU passed; GPU queued, 22 October 2026:** CPU **29792544** passed
all **106 fixtures**, strict loading, three preparations and the new actual-
input CPU probe. All five CPU parity comparisons matched exactly; a finite,
nonzero FP32 element-leaf backward gradient was observed. **GPU 29792558 PEND**
for CUDA parity and occupancy/NO_ATOM derivatives. CPU evidence does not confirm
the historical GPU mismatch cause or pass those derivatives.
[Saved CPU probe and native submission](#25-attempt005-cpu-feature-preflight-passed-native-gpu-queued).

**Attempt005 sealed and submitted, 22 October 2026:** **209 members
authenticated**, stage/queue return zero, **CPU 29792544 submitted**. The 106
fixtures, actual-checkpoint load and real-input CPU preflight must pass before
conditional GPU parity/gradient execution. Submission establishes no numerical
pass or confirmed C correction. [Actual receipt and immutable identities](#24-attempt005-sealed-and-cpu-submitted).

**Authorized prospective retry, 22 October 2026:** attempt005 corrects the
integer-feature gradient probe and gives C/Z the same explicit FP32 element
boundary. An allocated CPU test using the actual prepared development input
and frozen model is a new prerequisite before GPU dispatch. No submitted job,
executed test or confirmed CUDA correction follows from this preparation.
[Correction contract and retained gates](#23-attempt005-shared-floating-element-boundary-and-real-input-cpu-preflight).

**Earlier terminal outcome, 22 October 2026: GPU 29792051 EXIT.** It ran
20:18:10–20:18:47 DTU-local. First-case Z q=1 matched native exactly, while C
raw-logit parity still failed. The new gradient-enabled probe then attempted
`requires_grad_(True)` on the native integer element tensor and raised a dtype
error. Gradients, finite differences and intermediate traces did not run; the
candidate correction remains unvalidated. [Saved result and limitations](#22-attempt004-partial-parity-and-gradient-probe-dtype-failure).

**Earlier terminal outcome, 22 October 2026: GPU 29785005 EXIT.** It ran
14:20:52–14:21:26 DTU-local. Checkpoint loading succeeded and repeated native
evaluation was identical, but the first case's occupied C/Z adapter logit parity
failed the declared tolerance. Scalar loss and probability-TV comparisons
passed. Gradients, finite differences and later cases were not reached; no
occupancy conclusion follows. [Saved measurements](#18-attempt003-native-logit-parity-failed-before-gradients).

**Earlier retry readiness, 22 October 2026: CPU 29785001 DONE/passed; GPU 29785005
submitted/PEND.** All **80 fixtures passed**, zero failures/errors/skips,
and the actual official checkpoint loaded with `strict=True` after retaining
its guarded dormant zero ESM projection. All three cases prepared successfully.
No native forward/parity/gradient result yet. [Saved CPU/load evidence and GPU receipt](#17-attempt003-cpu-and-real-checkpoint-load-passed-native-gpu-queued).

**Earlier terminal outcome, 22 October 2026: native GPU 29783443 EXIT.** It ran
13:09:29–13:09:57 DTU-local and failed strict checkpoint loading with unexpected
key `input_embedder.linear_esm.weight`. No native forward, parity or gradient
check ran; this startup/configuration failure leaves the occupancy hypothesis
untested. CPU readiness remains passed. [Saved failure and scope](#14-terminal-outcome-checkpoint-loading-failed-before-native-checks).

**Earlier check, 22 October 2026 12:40:41 DTU-local: native GPU 29783443
is still PEND.** The scheduler reports an unmet physical-GPU resource
(`ngpus_physical`); four CPU slots have been reserved since 12:21:36, but
the native controller has not started. CPU readiness remains passed with
64 fixtures and all three prepared inputs. There are no native parity or
gradient results yet, so this wait gives no scientific evidence for or against
the occupancy proposal. [Saved current status](#13-status-check-native-gpu-still-pending).

**Earlier readiness/submission snapshot, 22 October 2026: CPU 29783378 DONE/passed; native GPU
29783443 submitted/PEND.** Attempt002 passed all **64 fixtures**, zero
failures/errors/skips, and prepared all three supported native hard inputs.
The full CPU controller completed in **29.102 seconds**. GPU **29783443** was
submitted at **12:21:35 DTU-local**, one exclusive A10/four slots, one-hour
allocation/55-minute internal budget. It is waiting for a physical GPU; no
native forward, parity/gradient result or occupancy optimization is established.
C/Z are the only implemented conditions; G remains unsupported. Attempt001's
fixture failure, both environment failures and all immutable releases remain
preserved. See [the actual readiness and pending GPU](#12-attempt002-full-cpu-ready-and-native-gpu-submitted)
and [the preserved fixture retry history](#11-attempt001-fixture-failure-and-attempt002-allocated-retry).
All local preparation remains static; numerical work belongs on allocated HPC.

The initial design-only boundary preceded this later authorization. Sections
1-7 retain the prospective full pilot design; the narrower first native phase
does not certify the remaining absence, discrete-transfer or optimization
stages. This independent study does not change source134, the stopped VAE
chemistry campaign, historical releases or their gates.

## 1. Question and bounded scope

Can a frozen Protenix model give a useful derivative for making one ligand atom
present or absent when `NO_ATOM` has zero embedding contribution and real-element
probabilities retain their original mass?

Separate four questions: whether a derivative exists; whether zero occupancy
really removes the slot's influence; whether the relaxed derivative agrees with
the two discrete endpoints; and whether a few updates select a better discrete
endpoint. A nonzero derivative alone answers only the first question.

Use one optional terminal heavy atom and an otherwise fixed valid scaffold.
Conditional element identity, charge and bond identity are fixed. Only occupancy
changes. Keep the pretrained model frozen; introduce no learned layer, scorer,
prior or reward surrogate. Begin with direct graph probabilities, so decoder
limitations cannot confound this adapter test. A later frozen-decoder-to-z study
would require its own scope, shared-graph identity and derivative checks.

The existing [Protenix-Tiny protocol](Small_Molecule_Hallucination_ProtenixTiny_Project_Protocol_v12.docx)
starts from fixed atom count and a ligand geometry-off path. This is a separate
occupancy feasibility addendum: first retain matched native reference geometry
to isolate occupancy. It does not validate geometry-free hallucination or alter
that protocol's baseline. The first pilot has no coordinate optimization,
diffusion, confidence-head objective, chemical repair or open-ended topology search.

## 2. Model and panel

Proposed model: frozen `protenix_tiny_default_v0.5.0`, one trunk recycle/cycle
(`N_cycle=1`), native distogram head, eval mode, deterministic torch kernels,
fixed declared precision and no Monte Carlo dropout. One cycle is a declared
pilot compute choice; native comparisons must use the identical setting, not
the model's default four-cycle inference protocol.

Use three preselected protein-ligand cases: one development case, followed by
two untouched confirmation cases. Start from BRD4-BD1, the target already named
in the Tiny protocol, if authenticated supported structures can be obtained.
Aim for a single protein chain of at most 256 tokens and ligands with 8-24
supported heavy atoms. In each case select exactly one terminal heavy atom
whose deletion gives a nonempty, connected, supported, sanitizable molecule
under the same stereo-free graph convention. A terminal halogen or hydroxyl
is a candidate; its validity and changed implicit hydrogen state must be checked,
not assumed. Exclude charged deletions, covalent ligand-protein attachments,
metals, unsupported chemistry and retained stereochemical ambiguity.

Before inspecting model scores, seal all three structures, exact full/deleted
SMILES and graph hashes, atom correspondence, optional atom, protein crop,
reference coordinates, target-bin source and case order. Select on input
support, not the sign or size of the endpoint score difference. If three suitable
cases are unavailable, record the shortfall; do not silently substitute a target
or score-selected cases. An optional atom that can be scored but has no measurable
influence remains an informative negative result.

For each case prepare three references:

1. **Native full molecule:** ordinary native features, all atoms present.
2. **Compacted deletion:** remove the optional atom/token from the full prepared
   input; retain surviving feature bytes, coordinates, MSA and context and
   consistently rebuild/remap every indexed tensor and derived cache. This
   diagnostic isolates absent-slot mechanics; it is not fresh chemical preparation.
3. **Fresh native deletion:** independently featurize the valid deleted molecule
   using the same protein context, options and deterministic seed. Keep the same
   retained-core target-bin readout. Record all changes in conformer, atom names,
   implicit chemistry, tokenization and indexing.

Do not force reference 3 to equal reference 2. Their difference measures native
preprocessing changes and limits transfer from fixed-context results. Where the
upstream model permits it, put the optional atom/token last to reduce index/window
confounds. A remapping audit must still check retained rows and local attention
windows; a discrepancy is not waived as harmless padding.

## 3. Occupancy variable and matched conditions

For the optional atom, use `P(NO_ATOM)=1-q` and `P(element)=q`; all other atom
categories have zero probability. Core atoms have occupancy one. For its fixed
conditional bonds, apply the endpoint occupancy factor once. The existing
`RelaxedGraph.bonds` already contains occupancy-gated marginals; adapters must
not gate these marginals again.

For gradient diagnostics also use the actual joint-softmax parameterization
`q=sigmoid(u)` at temperature one, equivalent to two logits `(0,u)` for absence
and the fixed element. A fixed zero absence contribution yields
`h=q*e_element`; it needs no trained absence embedding. Exact q=0/1 endpoints
are explicit hard probability inputs, not claims that finite logits reach zero.

| Condition | What the optional slot sends to Protenix | Purpose |
|---|---|---|
| C: occupancy discarded | Real-element probabilities renormalized to sum to one; conditional charge/bonds, native features and all interactions clamped to the full molecule | Negative control: q and its NO_ATOM logit must have zero influence |
| Z: zero-element contribution | Real-element feature `q*one_hot(element)`; other optional-atom features, bonds, masks and interactions clamped to the full molecule | Direct test of the user's minimal proposal; tests element scaling alone |
| G: occupancy propagated | Same joint probabilities, with an explicitly audited occupancy treatment for all atom/token/message/attention/pair/pooling routes | Tests whether absent-slot semantics require propagation beyond the element feature |

C and Z are deliberately clamped mechanism controls, not complete canonical
variable-occupancy molecular adapters. Document their non-element bindings.
C uses the fixed conditional identity even at q=0; it must not calculate 0/0.
Its analytically zero derivative may appear as an unused/None autograd input;
record that explicit zero dependency rather than manufacturing a model gradient.
G must consume one shared occupancy-aware graph with native element mapping
`Z-1`; `NO_ATOM` contributes zero, not a new native element class. The full
condition difference is the treatment of occupancy; all other scientific and
compute settings are identical. G has no validated implementation yet.

G requires an operation-by-operation design before implementation: atom-name,
charge and reference-feature routes; atom attention including key normalization;
atom-to-token aggregation; token-level features; token/pair attention and triangle
intermediates; MSA/template routes if enabled; residual updates and recycling;
and the distogram readout. Separate capacity/indexing from occupancy. The native
reference-conformer validity field `ref_mask` must not simply be reinterpreted
as an atom-existence probability. Avoid both missing and duplicate gates, including
an accidental `q^2` element contribution. A final score multiplier is insufficient.
No argmax, boolean occupancy conversion, detach or native preprocessing belongs
inside the differentiable q path.

For G, the initial design rule for attention is key occupancy inside its
normalization: `a_ij=q_j*exp(s_ij)/sum_k(q_k*exp(s_ik))`, with protein/core
occupancy one. Implement with numerically stable log weights; exact zero keys
are excluded for endpoint tests, while every interior q remains differentiable.
Multiplying values after an ordinary softmax leaves absent keys in its denominator
and is not equivalent. Triangle contractions must similarly exclude absent
intermediate slots. Retain occupancy separately through atom-to-token mean
aggregation so dividing by the atom count cannot restore an absent token's full
influence. These are required equations to audit against the pinned native
operations, not an assertion that changing one attention mask implements G.

## 4. Fixed readout and discrete reference

Use the existing branch's structural readout: mean target-distance-bin cross
entropy divided by `log(K)`, where K is the native distogram bin count. Before
model scoring, fix a nonempty set S of retained-core ligand/protein token pairs
and target bins from the declared reference structure. Count each unordered
cross pair once. Optional-atom pairs are excluded from S at every q, including
q=1. The pair set, targets and denominator `|S|*log(K)` remain identical across
all q values, conditions and native endpoints. Occupancy must not weight this
readout. Otherwise deletion can win by removing difficult loss terms.

Save the full selected-pair logits and distributions as well as the scalar.
This objective measures structural consistency on a fixed scaffold; it does
not measure binding affinity or establish chemical usefulness of deletion.

For both compacted and fresh native endpoint pairs, report
`L_Bernoulli(q)=(1-q)*L_deleted+q*L_full`. This exactly enumerated two-state
reference is an analysis control, never a loss used to train or guide G/Z.
Its derivative is the endpoint difference. A nonlinear relaxed forward need
not equal this mixture; plot their discrepancy and derivative-direction
agreement rather than assuming equality. If the endpoint difference is below
the measured numerical resolution, report its direction as indeterminate.

## 5. Gated measurements

All numerical stages below are future allocated DTU HPC work. The thresholds
are proposed prospective criteria, not observed passes. Authenticate the
development implementation and freeze it before opening confirmation scores.

**A. Native readiness and occupied endpoint.** Pin/check native source, weights,
resolved config, dependency environment, input/features and adapter bytes. Strict
weight loading must pass. Repeat a native forward to measure numerical variation;
verify unchanged parameters, buffers and bound context. For q=1, require agreement
with the ordinary native forward on selected raw logits, distributions and scalar
for every condition. Proposed logit/scalar tolerance: `atol=1e-5, rtol=1e-4`;
maximum selected-pair total-variation distance at most `1e-4`. If deterministic
native variation exceeds these bounds, diagnose and predeclare a revised protocol
before testing confirmation cases; do not loosen thresholds after seeing results.

**B. Absent endpoint and ghost influence.** At q=0 compare retained outputs with
the compacted deletion using A's tolerances. This is a required gate for G; for
C/Z it is a recorded mechanistic result and their failure is not hidden. At q=0,
independently change the excluded slot's conditional element, charge, name,
reference position, raw conditional incident-bond labels and extra absent-slot
count/order, consistently rebuilding relevant caches. Retained outputs must
remain invariant under G, within the same tolerances. Also check gradients with
respect to these absent attributes are zero within numerical resolution; the
occupancy derivative itself may be nonzero. No gradient through an index change
is implied. Failure stops G before optimization and identifies the leaking route.
Compare fresh native deletion separately, without conflating preprocessing drift
with this compacted-context gate.

**C. Controlled sweep and derivatives.** For all cases/conditions save q values
`0,.01,.1,.25,.5,.75,.9,.99,1`. Record loss, logits/distributions, entropy,
expected size, hard count, timing and peak memory. At q=.1,.5,.9 compare autograd
to central finite differences using q steps `.01,.003,.001`; separately check
the u/NO_ATOM-logit path at corresponding interior logits. Require agreement at
at least two step sizes with `atol=1e-6, rtol=.05`, stable frozen/context hashes,
and no NaN/Inf. Scale relative error by the larger derivative magnitude and log
both signed derivatives and absolute error. A nominal nonzero signal must cause
scalar differences above ten times repeat-forward variation. Verified zero or
unresolved derivatives are reported as such, not as positive gradient evidence.
C should be exactly q-independent by construction. In this two-state pilot,
argmax absence and q<=.5 coincide. A separate input-contract fixture with
`P(NO_ATOM)=.4, P(C)=.3, P(N)=.3` must distinguish categorical argmax from total
occupancy `.6`; that fixture does not add an element-search arm to the study.

**D. Short selection exercise, conditional on A-C.** Freeze conditional chemistry
and optimize only q from starts `.2` and `.8`. Use eight requested updates for
Z and G, with projected proposals `delta_q=-.1*sign(dL/dq)` and optimization q
constrained to `[.01,.99]`; zero derivative proposes a no-op. Exact q=0/1
are reserved for endpoint/ghost evaluation, so no optimization step differentiates
through `log(0)` or an exact-absence branch. Use the existing `best_trial_v2`
acceptance semantics: full-step nonincrease first, otherwise up to 16 sizes
`1,...,2^-15`, first strict improvement, else earliest lowest-loss finite trial.
The objective/context remain fixed through the trials. Record applied,
improving, forced-fallback and no-op counts and preserve the best observed state.
Reject individual nonfinite trials while preserving their records. Nonfinite
baseline/gradient, all-nonfinite trial sets, unexpected exceptions or resource
exhaustion retain genuine failure/partial status. Add unchanged-start and seeded random-sign controls
under G with the same starts, eight update opportunities, displacement bound
and trial budget; use case seeds 1729/1730/1731 in sealed case order, deriving
the two start streams by appending start index 0/1 in the manifest. Record actual forward/backward
counts because equal update opportunities need not cost equal model calls.

At starts, every update, best state and endpoint, save raw q/probabilities and
both native endpoint scores. Harden at q>.5 (q=.5 is absent) under this two-state
pilot; evaluate validity and the independently prepared same hard molecule.
Distinguish soft-loss improvement, a changed atom count, compacted-native score
improvement and fresh-native improvement. Eight steps are an engineering
selection probe, not evidence of long-run molecular optimization or convergence.

## 6. Decisions, reporting and bounded resources

- If Z has gradients but fails the absence/ghost checks, element scaling alone
  is insufficient for this adapter. G may still be investigated on the development
  case; Z's failure stays visible.
- If G fails native parity, derivative or absence checks, stop its selection
  exercise. Do not call this a successful atom-count adapter.
- If G passes engineering checks but relaxed preferences disagree with fresh
  discrete endpoints, report a relaxation/transfer failure. An interior soft
  optimum is not a better realizable molecule.
- Report all three cases individually and aggregate direction agreements as
  counts, not a population efficacy claim or p-value. Two confirmation cases
  provide a small replication check. Flat signals and failed cases stay visible.

Proposed future resource envelope: one GPU, one model resident at a time,
serial stages within this Protenix pilot, at most four CPU slots, a six-hour allocation and a
5.5-hour internal stop. Measure development-stage forward/backward memory and
runtime before confirming feasibility or sealing an execution plan. Enforce an
additional ceiling of 3,000 model-forward calls for the complete staged pilot
(including line-search/native/control calls). Budget exhaustion gives a partial
report; no longer allocation or automatic resubmission follows from this design.
Unrelated user campaigns are outside this pilot's serial-stage rule; this study
does not modify, cancel or otherwise control them.

Authoritative artifacts should include a sealed design/input manifest, native
references, readiness/gate JSON, sweep and derivative tables, every attempted
trajectory, raw selected-pair outputs, state/RNG hashes, timing/memory and failure
receipts. Render saved artifacts into an offline interactive HTML report showing
condition/case selection, loss versus q, derivative comparisons, native endpoint
overlays, soft/hard counts and both molecules. Rendering must not rerun the model
or invent missing outputs. Add actual jobs/outcomes to the experiment ledger
only when they exist.

## 7. Implementation blockers and next design checkpoint

The current [Protenix branch](../src/latentmol/multimodel/protenix.py) is a
native-unvalidated v1-only fixed-occupancy prototype; its loader rejects the Tiny
v0.5 model name. Its MSA and Pairformer calls use `pair_mask=None`.
[NativeBundle](../src/latentmol/multimodel/external.py) rejects changed occupancy
and maps only seed-present rows. Existing tests use `FakeProtenix`; they are not
native model evidence. Do not bypass these guards, relabel Tiny as v1, or change
the existing branch's declared capabilities to make the pilot load.

The next implementation proposal must audit a pinned Tiny source/checkpoint,
provide an additive experimental adapter and distinct capacity/slot mapping,
define G's exact occupancy equations and compaction/cache remapping, and reuse
the existing distogram reduction, provenance and frozen-state checks where their
contracts apply. It must identify the three real cases and bind all hashes before
execution. No runtime CLI or current ensemble config can execute this document.

Upstream design references checked 22 October 2026 (live source reconnaissance,
not an authenticated execution dependency):

- [Official model configurations](https://github.com/bytedance/Protenix/blob/main/configs/configs_model_type.py):
  identify Tiny v0.5 and its four-cycle default separately from base v1.
- [Official input embedder](https://github.com/bytedance/Protenix/blob/main/protenix/model/modules/embedders.py):
  atom features are aggregated and concatenated with token-level inputs.
- [Official atom transformer](https://github.com/bytedance/Protenix/blob/main/protenix/model/modules/transformer.py):
  reference features, local attention and atom-to-token aggregation require audit;
  a zero element vector alone does not establish absence semantics.

Relevant project boundaries: [shared molecular contract](04_latent_pipeline.md),
[frozen-model contract](23_frozen_model_plugin_contract.md),
[experimental native branches](28_multimodel_experiments.md),
[full-budget step policy](62_full_budget_best_trial_optimization.md) and
[saved-only reports](59_interactive_optimization_reports.md).

## 8. Authorized first native phase and actual environment submission

The user's subsequent implementation/execution authorization starts with
ordinary native parity and finite-difference checks on DTU HPC. It establishes
neither a pass nor completion of the prospective full study. The additive
[first-stage runner](../src/latentmol/protenix_occupancy/study.py) implements
`protenix-occupancy-native-parity-gradient-v1`, separate from the existing
v1-only fixed-occupancy branch. It tests C's clamped element input against Z's
unnormalized real-element mass, keeping native geometry and other features
fixed. It checks selected raw logits, distributions and the fixed-core scalar,
then q and explicit `NO_ATOM`-logit derivatives and the declared q sweep.
Its immutable output records distinguish these C/Z mechanism controls from
variable-occupancy support. G raises `NotImplementedError`; absent endpoint,
ghost invariance, discrete native-transfer and optimization results remain
unestablished. No C/Z result may bypass those later gates or trigger optimization.

[Focused tensor fixtures](../tests/test_protenix_occupancy.py) cover retained
element mass, the absence-logit gradient, C's analytical zero, clamped context,
input validation, fixed-pair reduction and output comparison. They use no real
model. Local preparation performed standard-library AST parsing only, without
project imports, numerical tests, model loading or molecular preprocessing.
The fixture suite must run on allocated CPU before a native GPU phase. Genuine
prepared inputs, strict Tiny loading, resolved configuration/runtime bindings,
CPU results and a sealed execution release are still required for that phase.

The official PyTorch Protenix source and Tiny checkpoint were downloaded from
their primary distribution URLs and authenticated by byte hashes. This pins
Protenix release v1.0.9 at commit
`510faf9f5cc8fbbbb8d9055934d3fa7e6877c76d`; its model configuration includes
`protenix_tiny_default_v0.5.0`. The release/source version and Tiny model-weight
version are distinct identifiers. This does not relabel Tiny as the existing
adapter's v1 model, and an existing JAX/Protenij implementation in Mosaic is
not substituted for this PyTorch native parity reference.

| Authenticated asset | SHA256 |
| --- | --- |
| Official source tarball | `fc596dd0a1e95942531408a741e510af06a179e223505badae98489b90d9d4eb` |
| Sorted native `protenix` Python source tree | `4e5dea1855e439f6d6682262cb820285dfa6a15476a9868cbd1ab6a225fa14b3` |
| Tiny `protenix_tiny_default_v0.5.0.pt` | `7ad252e023d61f94572f51ab60c2a58f3a12205271898fe581fa38d20de9566b` |

Remote assets are under `/work3/s234818/latentmol/protenix_occupancy/assets/`;
the [actual submission receipt](../outputs/hpc-submissions/protenix-occupancy-environment-submit-001.json)
contains exact URLs, absolute paths, byte counts, source identity, command and
scheduler stdout. The source tarball is 50,675,766 bytes and the checkpoint is
443,171,586 bytes. Download authentication is asset evidence, not weight-loading
or numerical-model validation.

The isolated [environment bootstrap](../scripts/hpc/protenix_occupancy_environment.py)
was actually submitted as CPU **29781941**, `hpc`, four slots on one host,
`rusage[mem=4GB]`, 20-minute wall limit. The submission returned zero and LSF
confirmed its job ID. The submitted script SHA256 is
`39ff85792511c55e51646e8f3140ebc52ecd17771e73ff413371ce38d799302a`.
At that initial submission, no terminal outcome or import-readiness pass had
yet been authenticated. The first bootstrap installs four additive wheels (`ml_collections==1.1.0`,
`optree==0.17.0`, `biotite==1.4.0`, `biotraj==1.2.2`) into its own dependency
directory; it does not modify the shared Boltz environment. It then attempts
official native-model and featurizer imports inside the CPU allocation.

The available base runtime has `torch 2.5.1+cu121`, whereas the pinned official
requirements name `torch 2.7.1`. This is an explicit runtime deviation, not
inferred supported parity. The resolved environment and native checks must
authenticate it before any scientific gradient result; an environment-import
pass alone would not settle parity or derivatives. No native GPU submission,
native pass, atom-count result, model training or chemistry-campaign restart
follows from this bootstrap submission. Preserve all source134 failed evidence
and every historical release unchanged.

## 9. Environment attempts, native wrapper and prospective scalar precision

The saved environment collection receipts supersede the initial submitted-only
snapshot while retaining all failed attempts:

| Attempt / allocated CPU | Actual outcome | Import result |
| --- | --- | --- |
| 001 / **29781941** | **EXIT**, exit 1, 22 October 12:05:34 DTU-local | Missing `absl` required by `ml_collections`; four wheels installed, readiness failed. [Receipt](../outputs/hpc-submissions/protenix-occupancy-environment-collect-001.json) |
| 002 / **29782094** | **EXIT**, exit 1, 22 October 12:07:12 DTU-local | Added `absl-py==2.3.1`; the official inference import then failed on missing `esm`. [Receipt](../outputs/hpc-submissions/protenix-occupancy-environment-collect-002.json) |
| 003 / **29782262** | **DONE**, successful completion, 22 October 12:09:00 DTU-local | Six isolated wheels; official `Protenix`, `Featurizer` and `SampleDictToFeatures` imports passed. [Receipt](../outputs/hpc-submissions/protenix-occupancy-environment-collect-003.json) |

Attempt003 uses the original four packages plus `absl-py==2.3.1` and
`fair-esm==2.0.0`, installed in its own dependency directory. The ESM package is
required by an unconditional upstream inference import; no ESM checkpoint,
embedding model or prediction was run. The import diagnostic reports
`torch 2.5.1+cu121`, `biotite 1.4.0` and `cuda: false`, as expected in this CPU
readiness allocation. The successful import does not resolve the official
torch2.7.1 requirement deviation. Preserve both failed environment outcomes,
their package cohorts, commands and scheduler stderr separately from attempt003.
The shared Boltz environment is unchanged.

The additive [Tiny native wrapper](../src/latentmol/protenix_occupancy/native.py)
is now implemented for the pinned official PyTorch source and requires strict
loading of the Tiny0.5 checkpoint. It separates the untouched ordinary native FP32 trunk/head
from explicitly attached one-cycle equations, keeps every module in eval mode
and parameters frozen, authenticates model/config/source bytes, and checks
features and frozen state. The study runner replays a fixed RNG context for
native, attached and finite-difference calls. These are implemented checks,
not evidence that a native forward or derivative has passed.

Before any scientific C/Z execution, the scalar CE reduction was declared as
FP64 on the selected FP32 logits to reduce averaging/CE roundoff in finite
differences. Native model computation and saved logits/distributions remain
FP32. The same scalar formula, selected pairs, targets, denominator and scalar
precision apply to native and attached paths and all C/Z states. This is a
prospective diagnostic implementation choice; no observed failure prompted
changed parity/derivative thresholds, and none is loosened.

Allocated scientific fixtures, molecular preparation, sealed execution-release
bindings and native GPU submission are still pending in this record. No local
numerical tests were run. G, absence/ghost checks, discrete native transfer and
occupancy optimization remain unimplemented or unvalidated as stated above.
Environment readiness does not promote the Protenix study or authorize any
change to the stopped chemistry campaign or source134 evidence.

## 10. First scientific phase sealed and CPU submitted

Actual immutable release `source135-protenix-occupancy-native-attempt001`
was sealed, staged and authenticated with **208 members**. The
[preparation record](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt001-preparation.json)
preserves the initial `sealed_not_submitted` state;
the subsequent [stage-and-queue receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt001-stage-and-queue-001.json)
returned zero and records **CPU 29783221 submitted, not passed**. The earlier
state is not overwritten. This release concerns the Protenix branch; its full
identifier does not seal or execute the separate prospective chemistry
candidate also called source135 in earlier planning notes.

| Immutable binding | SHA256 |
| --- | --- |
| Release archive | `0f622684dc12387a3b199ed7873c5de18ac2e3b1740cd728d18c23653304f251` |
| Release manifest | `69fd48ad48b83d48d65c904a4d00284f621081541ce42c833c643c7e7028eddf` |
| Execution plan | `e20716cb1565a2fd80d2c54c4748fbec9600d0e9f19e50ceaa331de20525220c` |
| Case definition | `9c780dc283838116dcbf386d5c13fb1512f48a5c1003f505c57d3ade3a4dfb12` |
| Passed environment-attempt003 result | `648d7558f57993e86baaec6ab6494f27312ad97ace145c118343d8e4331a6636` |

The [sealed plan](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt001.json)
binds source, weights, dependency readiness, case definition, controller,
resource limits and the C/Z-only phase. The local archive is
`outputs/source-releases/source135-protenix-occupancy-native-attempt001.tar.gz`;
remote release and campaign directories are respectively
`/work3/s234818/latentmol/protenix_occupancy/releases/source135-protenix-occupancy-native-attempt001`
and `/work3/s234818/latentmol/protenix_occupancy/runs/source135-protenix-occupancy-native-attempt001`.
The sealed archive/manifest retain their captured document bytes. Later status
documentation is additive workspace history and does not alter that release.

The score-blind [case definition](../configs/protenix_occupancy_cases.json)
fixes three BRD4-BD1 deposited complexes and one terminal halogen each:

| Role | Structure / ligand CCD | Expected full heavy atoms | Optional atom |
| --- | --- | --- | --- |
| Development | 7ZA9 / HHQ | 8 | `I1`, iodine |
| Confirmation 1 | 7ZAD / IJL | 10 | `I1`, iodine |
| Confirmation 2 | 7ZAJ / HH8 | 11 | `BR1`, bromine |

Deposited structure hashes, protein/ligand instance selections, fixed order and
reference-distance-bin definition are in the sealed configuration. These are
selected inputs, not authenticated native-preparation or chemical-support
passes. Allocated CPU preparation must validate the declared molecules,
correspondence, terminal deletion and retained-core readout before model scores.
All cases share the same target family; the two confirmation cases are a small
within-target replication check, not independent target-family validation.

CPU **29783221** requests `hpc`, four slots on one host, `rusage[mem=4GB]`,
**45 minutes**. Only after mandatory C/Z fixtures and molecular preparation
pass, release/input/configuration identities authenticate, and a fresh quota
check passes may this controller submit a native job to `gpua10`: **one GPU
in exclusive-process mode, four CPU slots, one-hour allocation and 3,300-second
(55-minute) internal budget**, at most 3,000 model forwards. The dispatch
binding is persisted before `bsub`, so an immediately starting native job does
not depend on a later submission receipt. Per-mode exclusive claim files
reject duplicate controller work. There is no automatic resubmission on
failure or an ambiguous scheduler response.

No scientific CPU pass, native GPU job, C/Z native result, G support, absent
endpoint proof or occupancy optimization follows from the actual CPU
submission. No local numerical tests ran. All original frozen-model and
chemistry-campaign boundaries and every failed source134 record are preserved.

## 11. Attempt001 fixture failure and attempt002 allocated retry

The [saved attempt001 collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt001-collect-001.json)
records CPU **29783221 EXIT**, exit 1 at **22 October 12:19:33 DTU-local**.
The mandatory tensor suite had **63 passes and one failure**, no errors,
in **4.52 seconds**. The failure was the test's decimal probability
perturbation: float32 arithmetic produced TV `0.0009999945759773254`, outside
`pytest.approx(.001)`'s effective `1e-9` tolerance. This was a fixture
comparison failure, not a native model result. The controller correctly stopped
before molecular preparation and submitted no GPU job. Preserve its release,
test stdout, scheduler logs and terminal failure.

The corrected fixture uses the exactly representable binary fraction `1/1024`
for the same probability perturbation and expected TV. Production arithmetic,
native scientific thresholds, case selection, controller and resource limits
are unchanged. New immutable `source135-protenix-occupancy-native-attempt002`
was sealed/staged/authenticated with **208 members**; the original release was
not patched or restarted. Its
[preparation receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt002-preparation.json),
[stage-and-queue receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt002-stage-and-queue-001.json)
and [first collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt002-collect-001.json)
record CPU **29783378 RUN**, `hpc`, the same four-slot/45-minute request.
The fixture command returned zero with **64 passed in 4.41 seconds**. Molecular
preparation/full CPU readiness are still pending; that fixture result alone
does not authorize or establish a native GPU pass or submission.

| Attempt002 immutable binding | SHA256 |
| --- | --- |
| Release archive | `4bec6aa8d30b3ee5bfbeabf27e26b46816426b208cc3b438605ec52b317662f8` |
| Release manifest | `093404d09d11a1e7089a301a4b192ea2f302bc13f7c269df535db0f825027f52` |
| Execution plan | `b11c96c6548a7a97f6a752df9140535837956056021b552a06d5c3beec9f8790` |

An independent local standard-library tar/SHA256 byte audit verified both
archive hashes and compared all **208 file members**: no additions/removals,
**206 identical**, and only `tests/test_protenix_occupancy.py` plus
`release_manifest.json` changed. Every production `src/`, `scripts/`, `configs/`
and `inputs/` member is byte-identical; document members are also identical
between these two sealed releases. This audit neither imports nor numerically
executes project code. Later workspace status documentation does not alter
either sealed archive.

No completed preparation, full CPU gate, native GPU job, real-model gradient,
absence semantics or occupancy optimization is established by this running
snapshot. The C/Z-only scope, conditional native gates, environment deviation,
failed attempt001 and all source134/history boundaries remain preserved.

## 12. Attempt002 full CPU ready and native GPU submitted

The latest [saved scheduler/result collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt002-collect-003.json)
records CPU **29783378 DONE**, full readiness **passed**, with **64 tests,
zero failures/errors/skips**. The complete controller took **29.102 seconds**
and terminated successfully at **22 October 12:21:35 DTU-local**. The earlier
64-pass/running snapshot remains historical; completed molecular preparation,
identity checks and fresh quota evidence now establish CPU readiness.

The [authoritative CPU readiness](../outputs/hpc-results/source135-protenix-occupancy-native-attempt002/snapshot-001/cpu-ready.json)
and [preparation manifest](../outputs/hpc-results/source135-protenix-occupancy-native-attempt002/snapshot-001/prepared/preparation.json)
bind all three score-blind input cases. Preparation reports
`all_three_inputs_supported=true` and `prepared_not_native_validated`, with
`model_scoring_performed=false`. Chemical-support/native-input preparation
is not an ordinary model forward or a native parity/gradient pass.

| Prepared case | Protein tokens | Ligand heavy atoms | Fixed retained-core/protein pairs |
| --- | --- | --- | --- |
| 7ZA9 / HHQ | 129 | 8 | 903 |
| 7ZAD / IJL | 129 | 10 | 1,161 |
| 7ZAJ / HH8 | 129 | 11 | 1,290 |

The metadata records native CCD reference conformers with native reference-space
centering, `ref_pos_augment=False`, and experimental coordinates used only for
target bins. MSA input uses the official query-only dummy convention without
sequence search, and template tensors are all masked. The optional atom is last
for 7ZA9/7ZAD, but not 7ZAJ; no compaction/window-invariance claim is inferred
for that case. Compacted and freshly featurized deleted references are explicitly
`not_run_first_C_Z_phase`. Their later absence/transfer gates remain required.

The authenticated [native submission receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt002/snapshot-001/native-submission.json)
records scheduler return zero and **GPU 29783443 submitted** to `gpua10`, one
GPU in exclusive-process mode/four slots, **one hour** with **3,300 seconds
internal runtime** and at most 3,000 forwards. The
[latest saved scheduler collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt002-collect-004.json)
still shows **PEND** and no native output, with four CPU slots reserved on `n-62-18-5`, waiting
for a physical GPU. Reserved slots and submission are not a native-model result.

| Authenticated CPU/native binding | SHA256 |
| --- | --- |
| CPU fixture JUnit | `34f0663c93c8e666bb1f7636dd77d229f6f7d0ff8f6e350c09a1cf7db5ae11f4` |
| CPU readiness | `ada5ec3e70cd312b8046c9ecdba246a779ce60b58cba55ca2295ad6215031637` |
| Prepared manifest | `f0dff7a7149bffca8129ac5700f941fbcf5cb641d8be6aef25a8468dadf3feac` |
| Native execution config | `5aa88d534276f518ec619722d7452d4dbbfee283cc31b658a106ce8966cfd8c4` |
| Seventeen-file artifact export | `15dbbf657257092461e598d6ed35426d6beaa339c2b56bbe833f46fe510c840d` |

The local evidence snapshot is under
`outputs/hpc-results/source135-protenix-occupancy-native-attempt002/snapshot-001/`;
its [collection receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt002/snapshot-001/collection-receipt.json)
records the export SHA256 and all 17 file hashes, including CPU terminal/logs,
fixtures, prepared-case metadata, configuration and dispatch/submission records.
This preserves actual readiness independently of scheduler retention.

Only the first C/Z native-validation stage has been implemented and dispatched.
No GPU/native outcome is present yet. G propagation, exact absence/ghost checks,
discrete native-transfer validation and occupancy optimization are still
unimplemented or unrun; no complete variable-count adapter or atom-count result
is claimed. The passed CPU fixtures and supported inputs do not waive those
gates or alter the stopped chemistry campaign. Preserve all earlier failed
receipts, scientific source/archive bytes and historical evidence unchanged.

## 13. Status check: native GPU still pending

The [saved 12:40:41 DTU-local status](../outputs/hpc-results/source135-protenix-occupancy-native-attempt002/status-news001.json)
and [sixth controller collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt002-collect-006.json)
record **GPU 29783443 PEND**, with the scheduler's physical-GPU requirement
(`ngpus_physical`) unmet. Four CPU slots on `n-62-18-5` have been reserved
since **12:21:36**; reservation is not execution. **CPU 29783378 remains DONE**,
with its completed 64-pass readiness and prepared inputs preserved.

Only `native-submission.json`, `native-dispatch-binding.json` and
`native-config.json` exist among the checked native files: there is no native
controller claim or output. Native forward parity, finite-difference agreement
and NO_ATOM sensitivity have therefore **not run**, rather than passed or failed.
The implementation still covers C/Z only; G, absence/ghost checks, discrete
transfer and occupancy optimization remain unrun. This read-only status check
submitted no job and changed no sealed release, plan or scientific outcome.

## 14. Terminal outcome: checkpoint loading failed before native checks

The [eighth saved collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt002-collect-008.json)
and [authenticated native result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt002/snapshot-002/native/native-gate-result.json)
record `status=failed`, `stage=startup`. Scheduler output records GPU
**29783443** started **22 October 13:09:29 DTU-local**, then **EXIT / exit 1**
at **13:09:57**, after **28 seconds**. It was no longer queued at collection.

Strict `model.load_state_dict(state, strict=True)` rejected the checkpoint's
unexpected `input_embedder.linear_esm.weight`. The instantiated model and
checkpoint therefore do not match at load time. No key was discarded and no
strict-loading requirement was relaxed. A corrected model/configuration binding
has not been established by this status check.

No native forward, C/Z parity comparison, occupancy or NO_ATOM-logit gradient,
finite-difference comparison, sweep or optimization ran. This is a failed
execution attempt before scientific checks, not numerical evidence against the
occupancy proposal. The CPU 64-pass readiness and three prepared inputs remain
valid within their stated scope; they did not test strict real-checkpoint loading.

The [23-file saved artifact snapshot receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt002/snapshot-002/collection-receipt.json)
authenticates export SHA256
`9da55877ff5d6558386a7be859d2e1afad2b82d3ca218627a4bb7a708e4a8abd`,
including the result, full traceback, command return code and scheduler logs.
Attempt002's sealed release, plan, failed evidence and all historical attempts
are preserved. This status turn submitted no replacement job and changed no
scientific source, checkpoint, gate or unrelated chemistry campaign.

## 15. Authorized checkpoint-loading correction and prospective retry

The user authorized fixing the startup mismatch and running the native checks
again. The new candidate is planned as a separate immutable
`source135-protenix-occupancy-native-attempt003`, preserving attempt002's sealed
release, terminal failure and every earlier record. No job or numerical pass
is established by this implementation note.

The compatibility path accepts only the checkpoint's one additional
`input_embedder.linear_esm.weight`, with no missing native keys or other extra
keys. It requires a dense CPU float32 tensor of shape **449 × 2,560**, finite
and **exactly zero**, both native ESM-enable flags false and the exact default-
Tiny dimensions. It then registers the official native `LinearNoBias(2560,449)`
as a dormant projection and loads the complete checkpoint with `strict=True`.
No checkpoint tensor is removed and no ESM model is enabled or executed.
Ongoing frozen-state checks include the retained zero projection and disabled
ESM state. Whether the actual checkpoint satisfies these new guards remains an
allocated-CPU check; static source inspection alone is not that result.

The new controller inserts an **actual-checkpoint CPU strict-load readiness
gate before any GPU submission**. Sixteen additional fixture cases cover the
narrow compatibility contract; the prospective suite has **80 cases**, not yet
executed. Native/attached forward equations, fixed readout, selected molecular
cases, C/Z scope, scientific tolerances and GPU limits are unchanged. Any
fixture, real-load, preparation, identity or quota failure must stop before GPU
dispatch. The new release will overlay the corrected study files onto the
attempt002 base to avoid incorporating unrelated concurrent chemistry edits.
Exact release/plan hashes and actual job receipts remain pending sealing and
submission.

The local inventory reports an **RTX 4060 Laptop GPU, 8,188 MiB**, driver
591.55, with only Python 3.13 registered. Local inference/gradient memory fit
and a compatible local PyTorch environment have not been validated; no local
numerical test or installation ran. The checkpoint-loading mismatch is
independent of host choice. Reusing the authenticated DTU A10 environment is
the current execution choice; a local port would require its own environment,
memory and parity validation rather than bypassing the same gates.

## 16. Attempt003 sealed and allocated CPU retry submitted

The [new preparation receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt003-preparation.json)
records the immutable `source135-protenix-occupancy-native-attempt003` release.
The [actual staging/submission receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt003-stage-and-queue-001.json)
records **209 authenticated members**, stage/queue return zero and **CPU
29785001 submitted** to `hpc`, four slots on one host, `rusage[mem=4GB]`,
**45 minutes**. Submission is **not a fixture, strict-load or full-readiness
pass**. The planned actual-checkpoint CPU gate must pass before the controller
may dispatch the unchanged conditional A10 request: one exclusive GPU/four
slots, one-hour allocation/55-minute internal budget.

| Attempt003 immutable binding | SHA256 |
| --- | --- |
| Release archive | `98781d189ca5bd3f3025bdd807d44b2a069f1356361e9af5a76692903c232ab0` |
| Release manifest | `8bce8d3496f54b132bf0ed4ce5a092d7632dae61e8702d829d3803bbbd95a7cf` |
| Execution plan | `f4a9d013cc05e36d068a109542455ad2da8f608f9ae7ccf98a9c2badc98a5bb7` |

The source, pretrained weights and isolated environment bindings are retained;
the ESM compatibility contract and added CPU real-checkpoint gate are described
above. This is a new authorized execution attempt, not a restart or revision of
failed attempt002. No GPU job/result, parity/gradient pass, G support or
occupancy optimization follows from CPU submission. These workspace history
updates are additive and do not rewrite the sealed archive, plan or failed
evidence; unrelated chemistry work remains separate.

## 17. Attempt003 CPU and real-checkpoint load passed; native GPU queued

The [third saved collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt003-collect-003.json)
records **CPU 29785001 DONE**, full readiness **passed** in **67.628 seconds**.
The mandatory suite passed **80/80 fixtures**, with **zero failures, errors or
skips**, in **4.92 seconds**. All three molecular cases prepared successfully.
The [CPU readiness record](../outputs/hpc-results/source135-protenix-occupancy-native-attempt003/snapshot-002/cpu-ready.json)
binds the fixtures, prepared inputs and native execution configuration.

The [actual-checkpoint load result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt003/snapshot-002/cpu-checkpoint-load.json)
records `strict_checkpoint_load_passed_not_native_validated`, elapsed **32.150
seconds**, with `model_forward=false` and `native_validation=false`. The exact
official checkpoint satisfies the narrow compatibility contract: the retained
**449 × 2,560 float32 ESM projection is exactly zero**, ESM is disabled, and
the full state loads with `strict=True`. Its tensor digest is
`c60df69bf01772d9119e5003d344c362d534edabdade636e6e3926423da28eeb`.
This resolves the observed checkpoint-loading blocker without establishing a
native forward or differentiable-adapter pass.

The [native submission receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt003/snapshot-002/native-submission.json)
records return zero and **GPU 29785005 submitted/PEND**, `gpua10`, one
exclusive A10/four slots, the unchanged one-hour/55-minute limits. There is no
native parity, NO_ATOM-gradient, finite-difference or sweep outcome yet. G,
exact-absence/ghost checks, transfer and occupancy optimization remain unrun.

| Attempt003 completed CPU evidence | SHA256 |
| --- | --- |
| CPU fixture JUnit | `3c0a67cf4b691c3834457d9e2bdd9520ecb272a2eb59f2db9a72f24d24acdd71` |
| Actual-checkpoint load result | `bfa168c7df1ba0fce1e8b99fe61d56dfafaa22ef7f7feae22b675b1624874721` |
| Prepared manifest | `9fb1417e94ca0f404ee9c81454e6748b0aa66e6fdbad7e8a3912caf3c194935c` |
| Native execution configuration | `bfcfd93e44758629ad7efa906ef5703a25aa4333c02b3fe481a194e6f13d1c0e` |
| Nineteen-file saved-artifact export | `50e1493b40fb5f1836b92636957e56eb24a6aeb141094550e4ca3f536f53e41c` |

The [saved artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt003/snapshot-002/collection-receipt.json)
authenticates the 19-file export. The [static archive/AST audit](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt003-static-audit.json)
compares attempt002 and attempt003: **202 files identical, six changed, one
added, none removed**. Changed members are docs73, the manifest, stage
controller, native wrapper, study module and fixtures; the new member is the
CPU checkpoint-load script. No other source, molecular input or case-selection
member changed. This static audit is separate from the allocated CPU results.

The [local hardware/runtime inventory](../outputs/hpc-submissions/protenix-occupancy-retry003-local-inventory.json)
supports the local-machine discussion above; it is not local numerical-model
validation. All numerical checks remain on allocated DTU HPC. Earlier startup
and fixture failures, sealed releases and unrelated chemistry records remain
preserved.

## 18. Attempt003 native logit parity failed before gradients

The [fifth saved collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt003-collect-005.json)
and [authenticated native result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt003/snapshot-003/native/native-gate-result.json)
record **GPU 29785005 EXIT / exit 1**. The scheduler log records start
**22 October 14:20:52 DTU-local** and termination **14:21:26**, a 34-second
allocation execution. The controller measured **31.510s** and the model phase
**24.240s**; these scopes differ. Strict checkpoint loading and dormant-zero ESM
compatibility succeeded, resolving the earlier startup error.

Only development case **7ZA9/HHQ** reached model evaluation. Its two native
evaluations matched exactly: zero logit difference, scalar-loss difference and
probability total variation. At full occupancy q=1, both adapter conditions
then failed the prospective raw-logit allclose criterion (`atol=1e-5`,
`rtol=1e-4`). The unchanged probability-TV limit is `1e-4`.

| q=1 adapter condition | Maximum absolute logit difference | Absolute scalar-loss difference | Maximum pair probability TV | Overall parity |
| --- | --- | --- | --- | --- |
| C: fixed occupancy control | `4.9591064453125e-5` | `3.453983599133181e-8` | `4.089529284101445e-6` | Failed raw logits |
| Z: unnormalized element mass | `6.29425048828125e-5` | `6.147402586975659e-9` | `4.2908754949166905e-6` | Failed raw logits |

Every compared output was finite. Scalar-loss and probability-TV checks passed
for both conditions, but those do not override the failed logit requirement.
Four model forwards ran in total: native, native repeat, C q=1 and Z q=1.
The saved stage is `brd4_7za9_hhq:native_parity`, with empty derivative and sweep
records. Neither occupancy/NO_ATOM-logit gradients nor finite differences ran,
and neither confirmation case reached model evaluation. No G/absence/ghost/
discrete-transfer check or optimization ran.

The mismatch is small in magnitude, but its cause has not been established.
This is a failed adapter parity gate, with no evidence yet for or against the
NO_ATOM occupancy proposal. No tolerance was relaxed, no failed comparison was
reclassified and no replacement job was submitted by this status check. CPU
readiness remains passed with 80 fixtures, actual strict loading and three
prepared inputs.

The [26-file snapshot receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt003/snapshot-003/collection-receipt.json)
authenticates saved-text export SHA256
`48da35e66c7090f9e95a52cc74892f1a7edf93762b7fef839357118a2f9b187b`.
The full traceback, parity JSON, native result, command return, terminal
controller outcome and scheduler logs are preserved. Selected binary parity
outputs remain saved in the immutable remote campaign. All sealed source,
plans, earlier failures and unrelated chemistry records are unchanged.

## 19. Authorized parity investigation and prospective native-method retry

The user authorized investigating the adapter discrepancy, correcting it and
rerunning parity and gradient checks on allocated DTU HPC. The prospective
release is `source135-protenix-occupancy-native-attempt004`; attempt003's failed
gate, source, plan and selected binary outputs remain preserved. No numerical
pass, confirmed cause or submitted job follows from this preparation note.

Static comparison found no established algebraic discrepancy between the
manually reproduced trunk and the pinned native method. Both old q=1 parity
paths ran under `no_grad`, so different autograd modes do not explain that
particular observation. Differences in intermediate lifetimes or allocation
remain hypotheses, not a demonstrated cause. The old manual implementation is
retained only for forensic comparisons and cannot supply an accepting gate.

The new attached callable is compiled from the authenticated, actual bound
native `get_pairformer_output` method. An AST guard requires exactly the audited
`torch.set_grad_enabled(self.training and (not self.train_confidence_only) and
cycle_no == (N_cycle - 1))` context and removes that context only. All native statements,
their order and the complete tuple return are retained. The original bound
method remains untouched as the independent reference. Every module stays in
evaluation mode, every parameter stays frozen, and source/configuration/weight
identity checks still surround execution. This is an input-gradient adapter,
not a training-mode change or an upstream-source modification.

The prospective CPU suite contains **88 fixtures**, including eight AST-context
regressions, followed by the actual-checkpoint strict-load and three-case
preparation gates. These are expected counts, not executed results. The GPU
phase retains the original q=1 parity requirements and adds a separate
**gradient-enabled q=1 parity gate** before interior-occupancy and NO_ATOM-logit
finite differences. Legacy-path q=1 comparisons and instrumented module traces
are diagnostic only. Trace clones can themselves change allocation behavior;
their agreement cannot override an uninstrumented gate failure or establish
the historical cause by itself.

Molecular cases, fixed common-core/protein readout, FP32 model computation,
FP64 scalar reduction, parity/gradient tolerances and the one-hour A10 allocation
with a 55-minute internal cap are unchanged. A failed prerequisite must stop
the retry. No G occupancy propagation, exact-absence/ghost validation, discrete
transfer, optimization or training result follows from this prospective parity
correction; unrelated chemistry remains unchanged. Exact immutable bindings and actual
execution receipts will be recorded after sealing and submission.

## 20. Attempt004 sealed and allocated CPU validation submitted

The [new preparation receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt004-preparation.json)
records sealed `source135-protenix-occupancy-native-attempt004`. The
[actual staging/submission receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt004-stage-and-queue-001.json)
records **209 authenticated members**, stage/queue return zero and **CPU
29792049 submitted** to `hpc`: four slots on one host,
`rusage[mem=4GB]`, **45 minutes**. This is submission, not a CPU fixture,
strict-checkpoint, native parity or gradient pass.

| Attempt004 immutable binding | SHA256 |
| --- | --- |
| Release archive | `a085175c4bd5997a402f1a3a27626d78e5ac3a69ad00490d870fd05cd09024a8` |
| Release manifest | `fc7c0d7859c2bc1913e79bdcd08f49e1cd751c51cf23b280d50ea018ef79dac8` |
| Execution plan | `4ab6bc6bd3402cc6c4e3bc2cea121844fcb499c4dddf3c1ee22b04de58c1ccf4` |

Before sealing, static review corrected compilation to preserve the original
native method's annotation behavior rather than inherit the wrapper module's
future flags. The one-context AST transform and original statement/return
order remain the correction contract. This review is source evidence, not a
native numerical result or a confirmed explanation of attempt003's mismatch.
Forensic trace numerical differences remain non-gating; instrumentation
execution errors remain genuine failures, as declared in the execution metadata.

The controller must pass the 88-case CPU suite, actual-checkpoint strict load,
input preparation, identity and quota gates before it may submit the native
stage to one exclusive A10/four slots, one-hour allocation/55-minute internal
budget. No GPU job or result is established by this receipt. Earlier failed
attempts and their sealed bytes remain preserved; all G/absence/transfer and
optimization work remains unrun, and unrelated chemistry is unchanged.

## 21. Attempt004 CPU passed; native GPU submitted and pending

The [second saved collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt004-collect-002.json)
and [CPU readiness record](../outputs/hpc-results/source135-protenix-occupancy-native-attempt004/snapshot-001/cpu-ready.json)
record **CPU 29792049 terminal gate passed**, **62.836 seconds**. All **88
fixtures passed**, with **zero failures, errors or skips**, in **4.75 seconds**.
The scheduler still briefly showed RUN during terminal collection; the
persisted controller result, rather than an inferred scheduler completion,
establishes full CPU readiness. All three cases prepared successfully.

The [actual-checkpoint loading record](../outputs/hpc-results/source135-protenix-occupancy-native-attempt004/snapshot-001/cpu-checkpoint-load.json)
records `strict_checkpoint_load_passed_not_native_validated`, with no model
forward. The dormant ESM compatibility contract still passes. It binds the
actual source-derived attached callable: **one removed gradient context**,
`native_source_modified=false`, `training_enabled=false`; native-method source
SHA256 `8c141b7c9a49214901c78d3da2ffbc4d520ac1f13863fcb16567506b40dc977f`,
attached AST SHA256
`d6ecbb9f04608504c8994789c9d311d97967681c8a509c6f4ed6bfa873f1dd3d`.
These CPU/source checks do not establish native output or gradient parity.

The [actual native submission receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt004/snapshot-001/native-submission.json)
records return zero and **GPU 29792051 submitted/PEND**, `gpua10`, one
exclusive A10/four slots, unchanged one-hour allocation/55-minute internal
limit. Native parity, gradient-enabled q=1 parity, occupancy/NO_ATOM-logit
gradients and finite differences have no result yet.

The [detailed queue snapshot](../outputs/hpc-results/source135-protenix-occupancy-native-attempt004/status-queue001.json)
observed **19:12:37 DTU-local** records submission at **19:11:49**, GPU job
**PEND**, with four CPU slots reserved on `n-62-18-5` while waiting for
`ngpus_physical`. There is no GPU controller claim or native result. Reservation
is not execution, and the candidate correction remains native-unverified.

| Attempt004 completed CPU evidence | SHA256 |
| --- | --- |
| CPU fixture JUnit | `4a9189653c061b7b641fbb2fc45ea84994f11d4011750a2c35f0a62178467b68` |
| Actual-checkpoint load result | `d978e3a2c9cf6787106c7e91add8b5d78868a71aef5bcc831a47b0a4fb0b6454` |
| Prepared manifest | `636512e059513fb6eaa58b4b16f82d82846c36309ae08daa0d7074f0379931fe` |
| Native execution config | `2868f5b328843395990d79019b8cfb15c308c2205639e52312e9fd67325d752b` |
| Nineteen-file saved export | `bdf399f2fa7fc12ad1543052de36463ed3acaf90e930be5813249f90dae7157f` |

The [saved artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt004/snapshot-001/collection-receipt.json)
authenticates the nineteen-file export. The
[static archive/AST audit](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt004-static-audit.json)
records **203 unchanged/six changed members**, no additions or removals versus
attempt003. This is byte/source evidence separate from CPU readiness and
pending GPU science. No confirmed historical mismatch cause, native fix,
NO_ATOM-hypothesis result, G support or optimization is claimed. Failed sealed
attempts and unrelated chemistry records remain unchanged.

## 22. Attempt004 partial parity and gradient-probe dtype failure

The [fourth saved collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt004-collect-004.json)
and [saved native result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt004/snapshot-002/native/native-gate-result.json)
record **GPU 29792051 EXIT / exit 1**, started **22 October 20:18:10 DTU-local**
and terminated **20:18:47**, a 37-second scheduler execution. The model phase
measured **25.779s**, with **five model forwards** on development case 7ZA9/HHQ.

| First-case q=1 comparison | Maximum absolute logit difference | Scalar-loss difference | Maximum pair probability TV | Parity |
| --- | --- | --- | --- | --- |
| Native repeat | `0` | `0` | `0` | Passed |
| Preserved manual diagnostic path | `0` | `0` | `0` | Passed, diagnostic only |
| Source-derived C control | `4.9591064453125e-5` | `3.453983599133181e-8` | `4.089529284101445e-6` | Failed raw logits |
| Source-derived Z condition | `0` | `0` | `0` | Passed |

Every saved comparison was finite. Z's exact match is an occupied-input forward
result on one case, not interior occupancy sensitivity or a complete adapter
validation. C still fails the unchanged declared tolerance. The manual path's
exact match in this attempt, unlike attempt003, also prevents attributing the
prior difference solely to the manually written equations. Allocation/lifetime
effects remain a hypothesis; no cause has been established.

Execution then raised `RuntimeError: only Tensors of floating point dtype can
require gradients` at the newly added probe's
`features["ref_element"].detach().clone().requires_grad_(True)`. The native
element tensor has integer dtype, and the new probe omitted a conversion to
floating point. This is an implementation error in the new diagnostic, not
evidence about the NO_ATOM hypothesis. The dtype error occurred before the
gradient-enabled q=1 comparison and before intermediate hooks were installed.
The pending aggregate parity assertion was not reached; C's separately saved
comparison is nevertheless failed and cannot be waived.

No gradient-enabled parity result, occupancy/NO_ATOM-logit derivative,
finite-difference comparison, sweep, intermediate module trace or confirmation
case result exists. G/absence/ghost/discrete-transfer checks and optimization
remain unrun. CPU readiness remains passed with 88 fixtures and actual strict
loading; those fixtures did not cover this native integer-feature probe path.

The [25-file artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt004/snapshot-002/collection-receipt.json)
authenticates saved-text export SHA256
`42bd6126570813e344691d741417fa822f5d99161580dbc18e58e3edb09b8cd0`.
Full traceback, partial parity results, command and controller failure, and
scheduler logs are preserved. This status check changed no scientific source,
threshold, sealed record or unrelated chemistry job and submitted no replacement
run. Attempt004 and every prior failure remain preserved.

## 23. Attempt005 shared floating element boundary and real-input CPU preflight

The user authorized fixing the diagnostic dtype bug and the remaining C parity
discrepancy, then rerunning parity and gradient checks on allocated DTU HPC.
The prospective immutable release is
`source135-protenix-occupancy-native-attempt005`. Attempt004's partial results,
dtype traceback and failed C comparison remain unchanged; no historical failure
is reclassified as passed.

Static inspection identifies a representation difference that the new retry
removes: native `ref_element` is integer, the old C control retained that dtype,
and Z converted it to floating point. Both mechanism conditions now explicitly
copy the element tensor to contiguous FP32, preserving its chemical values at
q=1 and sharing all non-element fields. Both perform the same optional-row
multiplication; C uses a fixed one and discards the requested q, while Z uses q
without renormalizing its real-element mass. The untouched native integer input
remains the independent reference. A separate native forward on the FP32
boundary must also satisfy the original parity tolerances.

The common helper preserves an existing computational graph during ordinary
conversion. Only the explicit gradient probe detaches after conversion and
creates a new FP32 leaf with `requires_grad=True`, fixing the invalid request
for gradients on an integer tensor. The new fixture inputs use native integer
elements by default. Eighteen boundary/dtype/gradient regressions extend the
prospective suite from 88 to **106 cases**; this is an expected count, not an
executed result. Occupied-input parity records and forensic traces are saved
before later gradient probing so a subsequent error cannot conceal completed
comparisons. Instrumented traces remain diagnostic and cannot override failed
uninstrumented parity.

After fixture success, all three cases must prepare before the actual-checkpoint
strict-load gate. A new, separately recorded allocated **CPU native-feature
preflight** then uses that loaded frozen model and the actual prepared
development case. It makes six forwards: untouched native, native repeat,
native at the FP32 boundary, attached C q=1, attached Z q=1 and gradient-enabled
attached q=1. Every comparison uses the unchanged raw-logit/scalar/probability
criteria against the untouched native result. It also checks a finite element-
leaf backward gradient, reports its norm/nonzero status, and verifies unchanged
model and context. The controller requires this preflight to pass before it
may dispatch a GPU job. This CPU smoke check is neither CUDA validation nor an
occupancy/NO_ATOM derivative pass.

The candidate still uses the authenticated source-derived attached native
method, frozen weights and evaluation mode. Molecular cases, fixed retained
core/protein readout and denominator, FP32 forward/FP64 scalar reduction, parity
and finite-difference tolerances, q grid and one-hour A10 allocation/55-minute
internal cap are unchanged. The actual cause of the historical C discrepancy
remains unconfirmed until the new native GPU observations; dtype/layout
alignment is a candidate correction, not a demonstrated explanation. No G
occupancy propagation, exact-absence/ghost validation, discrete transfer,
optimization or training result follows from a passing foundation check alone.
Sealed identities, actual submission receipts and results will be added after
they exist; all failed attempts and unrelated chemistry work remain preserved.

## 24. Attempt005 sealed and CPU submitted

The [preparation receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt005-preparation.json)
and [actual stage-and-queue receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt005-stage-and-queue-001.json)
record immutable `source135-protenix-occupancy-native-attempt005`, **209
authenticated members**, stage/queue return zero and **CPU 29792544 submitted**
to `hpc`, four slots on one host, `rusage[mem=4GB]`, **45 minutes**.
The actual scope is `submitted_not_passed`; no fixture, checkpoint-load,
real-input CPU preflight, native GPU parity or gradient result is established
by the submission receipt.

| Attempt005 immutable binding | SHA256 |
| --- | --- |
| Release archive | `2a77516abc6bcb9a341f5de587860846412c3f935d670cabcc2ed6199e54e978` |
| Release manifest | `723bb09db7ddb9136160761e295319060576639d4ef67b6b2cf804047e068b39` |
| Execution plan | `2278e5cce181f72a7a03d2d875acb20c6622db209661b2173eb34a7ce506ee04` |

The controller must pass all 106 fixtures, prepare all three cases, load the
actual checkpoint strictly, and pass the six-forward real-input CPU feature
preflight before a single GPU dispatch. A fresh quota check remains mandatory.
The conditional native GPU request is unchanged: one exclusive A10/four slots,
one-hour wall time and a 55-minute internal cap. C/Z share the declared FP32
element boundary while the native integer input remains the reference; parity
and finite-difference tolerances are unchanged. The historical cause and CUDA
correction remain unconfirmed, and the NO_ATOM hypothesis has no result yet.
All earlier immutable attempts, failures and unrelated chemistry work remain
preserved.

## 25. Attempt005 CPU feature preflight passed; native GPU queued

The [fourth collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt005-collect-004.json)
and [CPU readiness](../outputs/hpc-results/source135-protenix-occupancy-native-attempt005/snapshot-001/cpu-ready.json)
record **CPU 29792544 full gate passed**, **101.147 seconds**. All **106 fixtures
passed**, zero failures/errors/skips, in **5.19 seconds**. Strict loading of the
actual official checkpoint passed and all three native inputs prepared.

The new [actual-input CPU feature probe](../outputs/hpc-results/source135-protenix-occupancy-native-attempt005/snapshot-001/cpu-native-feature-probe.json)
completed **six frozen-model forwards in 33.800 seconds** on the prepared
development case 7ZA9/HHQ. The native element input was genuinely `torch.int64`,
shape **1,084 × 128**; the explicit probe leaf was contiguous FP32. Against the
untouched native reference, **native repeat, native FP32 boundary, attached C
q=1, attached Z q=1 and gradient-enabled attached q=1 all matched exactly**:
zero raw-logit, scalar-loss and probability-TV differences. The element-leaf
backward gradient was finite and nonzero, norm **0.057674906239316856**. Model
and context remained unchanged.

This is actual model CPU parity/backward smoke evidence, labelled
`cpu_native_feature_probe_passed_not_cuda_validated`; it does not certify CUDA
native parity, occupancy derivatives, NO_ATOM-logit gradients or finite
differences. The historical C GPU discrepancy's cause remains unproven.

The [native submission receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt005/snapshot-001/native-submission.json)
records return zero and **GPU 29792558 submitted/PEND**, `gpua10`, one exclusive
A10/four slots, unchanged one-hour allocation/55-minute internal cap. No CUDA
result exists yet. The native controller must independently pass occupied-input
and gradient-enabled parity before interpreting occupancy/NO_ATOM gradients.
Tolerances are unchanged, and G/absence/ghost/discrete-transfer checks and
optimization remain unrun.

| Attempt005 completed CPU evidence | SHA256 |
| --- | --- |
| CPU fixture JUnit | `ec5e88c361e224de667e741bb84aca55c2b1a5367dc5b977721ec2c4ce45ff06` |
| Actual-checkpoint load result | `ddf231ee048f571c84fd4f2a2892436f63e832c2e998b76a4b1191f026e367b0` |
| Actual-input CPU feature probe | `335ab8e73c0e6d335d1934d0fb663ee1c90f58c0b8efa5ebe7883ed732deed72` |
| Prepared manifest | `5ef72ea0c3bc0203facd60e30c0ac19043707c0b25de63d4004fea05fc64e0ee` |
| Native execution config | `ecacae20db39bf73cc4f818b641cf1f9930b2821749cc5d5386826c10a80edbe` |
| Twenty-file saved export | `a9ddf2bdec53ccc79eafcdb64422aa4aadfc527ed78623a8934faf63104b6b94` |

The [saved artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt005/snapshot-001/collection-receipt.json)
authenticates all twenty exported text files. The
[static archive audit](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt005-static-audit.json)
records **203 unchanged/six changed members**, with no additions/removals versus
attempt004. Static byte/source evidence, completed CPU validation and pending
GPU science remain distinct. Every failed attempt and unrelated chemistry
record is preserved.

## 26. Attempt005 native GPU repeatability failed before adapter checks

The [sixth collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt005-collect-006.json)
and [saved native result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt005/snapshot-002/native/native-gate-result.json)
record **GPU 29792558 EXIT / exit 1**, started **22 October 21:17:19 DTU-local**
and terminated **21:17:56**. Strict checkpoint loading succeeded. The controller
measured **33.679 seconds**, and the model phase **25.505 seconds**; these are
different timing scopes from the scheduler's start/termination interval.

Only **two untouched native forwards** ran on development case 7ZA9/HHQ. The
repeat failed the declared raw-logit allclose criterion (`atol=1e-5`,
`rtol=1e-4`), before any adapter or FP32-boundary comparison:

| Native repeat comparison | Observed value | Decision |
| --- | --- | --- |
| Maximum absolute raw-logit difference | `4.9591064453125e-5` | Failed allclose |
| Absolute fixed scalar-loss difference | `1.2899348256212306e-9` | Passed |
| Maximum pair probability total variation | `4.801047907676548e-6` | Passed limit `1e-4` |

Every compared output was finite. Passing loss/probability comparisons do not
override the failed raw-logit gate. The saved stage is
`brd4_7za9_hhq:native_parity`; adapter `parity` is empty and derivative/sweep
records are empty. The native FP32-boundary comparison, C/Z q=1 comparisons,
gradient-enabled probe, module traces, occupancy/NO_ATOM derivatives, finite
differences and confirmation cases were not reached. No G/absence/ghost/
discrete-transfer validation or optimization ran.

This outcome establishes a **native GPU repeatability blocker** under the
declared execution path. It does not establish an adapter-specific cause,
invalidate the prior CPU observations or determine whether the NO_ATOM proposal
works. A device/backend or execution-order explanation remains a hypothesis;
no cause was proved by this result. The previous **106 CPU fixture passes**,
strict load, three preparations and exact six-forward CPU feature preflight
with finite nonzero element-leaf backward gradient remain preserved as CPU
evidence, not CUDA or NO_ATOM derivative validation.

The [26-file artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt005/snapshot-002/collection-receipt.json)
authenticates saved-text export SHA256
`1c1edd28ac37826614232b9e64b747217c8c28499f6e3d9ee23e6884cb9167e0`;
the native result SHA256 is
`a14b40499e3b18f30109dac483d0f451e9c7c54463fa3780bcef2d6b80afa1d4`.
The traceback, command failure, terminal controller result and scheduler logs
remain saved. This read-only status check changed no scientific source, gate,
sealed record or job, and submitted no replacement. Attempt005, all earlier
failures and unrelated chemistry records remain preserved.

## 27. Attempt006 user-authorized native-repeat logit policy

Following attempt005's untouched-native repeat failure, the user explicitly
authorized removing **that single constraint** and trying again. The new
prospective immutable release is
`source135-protenix-occupancy-native-attempt006`. Its versioned policy marker
is **`native-repeat-loss-tv-v1`**, bound in the plan, controller and native
execution configuration. This is a declared prospective gate change, not a
correction of or exception applied to attempt005's historical failed decision.

Only the **untouched native-versus-native repeat raw-logit allclose decision**
becomes diagnostic. The original `atol=1e-5`, `rtol=1e-4` comparison, its pass/
fail result and observed maximum raw-logit difference remain recorded. A native
repeat may proceed only when raw logits and scalar loss remain finite, scalar
loss satisfies its existing allclose criterion, and maximum pair probability
total variation remains at most **`1e-4`**. A nonfinite value or failed retained
loss/probability requirement still stops execution. The allocated actual-input
CPU feature preflight uses the same declared native-repeat policy.

| Comparison or check | Attempt006 requirement |
| --- | --- |
| Untouched native repeat raw-logit allclose | Recorded diagnostic; no longer an accepting gate |
| Native-repeat finite raw logits/scalar loss, scalar-loss agreement and probability TV | Required with existing thresholds |
| Native FP32-boundary parity | Original full raw-logit/scalar/probability gate required |
| Attached C/Z q=1 parity | Original full raw-logit/scalar/probability gate required |
| Gradient-enabled attached q=1 parity | Original full parity gate required |
| Occupancy/NO_ATOM derivatives, finite differences and sensitivity/noise checks | Original requirements retained |

The common FP32 element boundary, frozen/evaluation model, exact source and
checkpoint bindings, three cases, fixed readout and denominator, precision,
RNG/context/model-state checks, CPU readiness, quota and resource limits remain
unchanged. Numerical work remains on allocated DTU HPC. No broader tolerance
relaxation, G/absence/ghost/discrete-transfer validation, optimization or
training result follows from this policy. The root cause of the historical
native repeat variation remains unproven; the new policy does not diagnose it.

This entry records authorization and preparation only. Extended fixture
validation, strict loading, actual-input CPU preflight and native GPU execution
remain unrun for attempt006; exact sealed identities, jobs and actual outcomes
will be added when available. Every earlier immutable release, failed decision
and unrelated chemistry record remains preserved.

## 28. Attempt006 sealed and CPU submitted

The [preparation receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt006-preparation.json)
and [actual stage-and-queue receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt006-stage-and-queue-001.json)
record immutable `source135-protenix-occupancy-native-attempt006`, **209
authenticated members**, stage/queue return zero and **CPU 29796415 submitted**
to `hpc`, four slots on one host, `rusage[mem=4GB]`, **45 minutes**. Its actual
scope is `submitted_not_passed`.

| Attempt006 immutable binding | SHA256 |
| --- | --- |
| Release archive | `693b3418a232b963eb9e14585f50eadd368bd4117516e19ee682883ff05ef8b8` |
| Release manifest | `e692c85e8047928bd65977ccc6e0ae6bd8f7a16a121c4c6dbf1c5a95284765ea` |
| Execution plan | `df7089e97dba99b88fb1c3afe775e0689b7f40ee4b4dbcf353f4d2dc8edf0212` |

Eleven prospective regression cases extend the suite to **117 fixtures**;
this is the expected count, not an execution result. The new comparison
preserves the original `logits.passed` and `strict_parity_passed` results while
separately reporting acceptance under `native-repeat-loss-tv-v1`. Only that
native-repeat acceptance changes. Full FP32-boundary, C/Z and gradient-enabled
parity, finite-difference/noise checks and all context/model/resource guards
remain mandatory.

Successful fixtures, all three native input preparations, strict actual-
checkpoint loading, the six-forward actual-input CPU feature preflight and
fresh quota are still prerequisites for conditional GPU submission. The GPU
request remains one exclusive A10/four slots, one-hour wall time and a
55-minute internal cap. No CPU or native GPU pass, confirmed variation cause,
NO_ATOM result, G/absence/transfer validation or optimization follows from this
submission. Earlier sealed failures and unrelated chemistry work remain
unchanged.

## 29. Attempt006 CPU passed; native repeat exact but occupied parity failed

The [second collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt006-collect-002.json),
[CPU readiness](../outputs/hpc-results/source135-protenix-occupancy-native-attempt006/snapshot-001/cpu-ready.json)
and [saved native result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt006/snapshot-001/native/native-gate-result.json)
record completed attempt006. **CPU 29796415 passed** its full gate in
**104.047 seconds**: **117 fixtures passed** with no failures/errors/skips
(6.91s), all three cases prepared and actual checkpoint strict load passed.
The actual-input CPU probe made six forwards in **30.948 seconds**, with exact
native repeat, FP32-boundary, C/Z and gradient-enabled q=1 parity, finite
nonzero element-leaf gradient norm **0.057674998699864734**, and unchanged
model/context. These remain CPU smoke results.

**GPU 29796426 EXIT / exit 1**, **23 October 10:19:04–10:19:46 DTU-local**.
The model phase measured **29.748 seconds** and made **nine forwards**, all on
development case 7ZA9/HHQ. Native repeat matched exactly; the new repeat policy
did not affect its acceptance because the original strict comparison also
passed. The preserved manual diagnostic path matched exactly. The other
occupied-input comparisons failed their unchanged raw-logit requirements:

| Occupied comparison | Maximum absolute logit difference | Scalar-loss difference | Maximum pair probability TV | Attempt006 decision |
| --- | --- | --- | --- | --- |
| Native FP32 element boundary | `5.7220458984375e-5` | `2.4148377297805723e-8` | `4.1168227653543e-6` | Failed raw logits |
| Attached C q=1 | `7.05718994140625e-5` | `2.9435869186578145e-8` | `4.652983079722617e-6` | Failed raw logits |
| Attached Z q=1 | `6.103515625e-5` | `2.3570675522499585e-8` | `4.110312147531658e-6` | Failed raw logits |

All compared outputs were finite and all scalar-loss/probability criteria
passed. Attempt006 nevertheless correctly stopped at its occupied parity gate;
its repeat-only amendment did not change these other raw-logit requirements.
The [occupied parity record](../outputs/hpc-results/source135-protenix-occupancy-native-attempt006/snapshot-001/native/brd4_7za9_hhq-occupied-parity.json)
and [module traces](../outputs/hpc-results/source135-protenix-occupancy-native-attempt006/snapshot-001/native/brd4_7za9_hhq-path-diagnostics.json)
were saved. Instrumented native-AST traces first differed at `input_embedder`,
but the instrumentation changes tensor lifetimes/allocation and does not prove
a device/backend cause or override the uninstrumented failure.

Gradient-enabled GPU parity, occupancy/NO_ATOM derivatives, finite differences,
sweeps and confirmation cases were not reached. No G/absence/ghost/transfer
validation or optimization ran. The
[28-file artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt006/snapshot-001/collection-receipt.json)
authenticates export SHA256
`1ce8f2b15f8dc7e759b42ff1f22c2711449023f0b962785c778bd7d0fb7f9bb2`;
native result SHA256
`9615d5722b1ef6aa2aa0a429bba3f1fb0e19f8b5511d820da8d74752fa031326`.
Attempt006's failed decision and every earlier immutable result remain
preserved; the NO_ATOM hypothesis still has no derivative result.

## 30. Attempt007 consistent output-loss and probability policy

Attempt006 exposed that the requested raw-logit criterion was still a blocking
requirement at the FP32-boundary and adapter comparisons. The coordinator is
applying the user's request to remove that criterion consistently across the
active parity comparisons in a new immutable prospective
`source135-protenix-occupancy-native-attempt007`, declared as
**`native-output-loss-tv-v1`**. This extends attempt006's repeat-only policy to
the other active output comparisons; section 27 and the sealed attempt006 plan
retain their historical narrower scope.

Each comparison still records the original raw-logit allclose result and
maximum difference, along with the original strict aggregate parity decision.
Raw-logit allclose is diagnostic under the new acceptance policy. Acceptance
requires finite raw logits and scalar loss, the existing scalar-loss allclose
criterion and maximum pair probability TV **`<=1e-4`**. The native-repeat
`native-repeat-loss-tv-v1` marker remains explicit; the new output-policy marker
is bound in the plan/controller/configuration and applied consistently in
actual-input CPU preflight and GPU execution.

The new `compare_parity_outputs` wrapper retains every original strict metric
and `strict_parity_passed` while applying the declared output acceptance policy.
The generic `compare_outputs` remains the original strict comparator for
historical/forensic use. Eleven additional parameterized output-policy
regressions bring the prospective suite to **128 fixtures** (117 + 11); this
is an expected count, not executed validation.

This changes the acceptance meaning: a new-policy pass will be loss/probability
agreement under the declared thresholds, not a pass of the former strict
raw-logit criterion. It does not retrospectively pass attempt006 or establish
the cause of the measured logit differences. Derivative/finite-difference and
sensitivity/noise checks, finite-value guards, source/checkpoint/context/RNG
identity, frozen model, cases, readout, precision, quota and resources remain
unchanged. No G/absence/ghost/discrete-transfer or optimization result follows
from an output-policy pass alone.

This is prospective preparation with no attempt007 CPU/GPU pass or submitted
job established yet. Actual sealed identities, executed fixture results,
submission receipts and results will be recorded after they exist. Every
earlier failed release and unrelated chemistry record remains preserved.

## 31. Attempt007 sealed and CPU submitted

The [preparation receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt007-preparation.json)
and [actual stage-and-queue receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt007-stage-and-queue-001.json)
record immutable `source135-protenix-occupancy-native-attempt007`, **209
authenticated members**, stage/queue return zero and **CPU 29796458 submitted**
to `hpc`, four slots on one host, `rusage[mem=4GB]`, **45 minutes**, scope
`submitted_not_passed`.

| Attempt007 immutable binding | SHA256 |
| --- | --- |
| Release archive | `75b8e995225b491405f78d6ff9176b831fad5597296b414323e41c40576f65da` |
| Release manifest | `19d98d7b8306ac94ab12b6944a67190ff32d20a00025722a1b3792e46dfb82cb` |
| Execution plan | `4200c25928c889306232665d4ca3bb9321c8fcb8c13e97ff96dca0bcaac38fbc` |

The new output policy is bound prospectively; it does not pass any earlier
failed comparison. **128 fixtures are expected**, with actual results pending.
All three input preparations, strict actual-checkpoint loading, the six-forward
real-input CPU preflight and fresh quota still precede any conditional native
GPU dispatch. Conditional resources remain one exclusive A10/four slots,
one-hour wall time/55-minute internal cap. Finite outputs, scalar-loss
agreement, probability TV and every derivative/noise/context/model/resource
guard remain required. No CPU or GPU numerical pass, NO_ATOM result or
optimization is established by submission; all failed immutable attempts and
unrelated chemistry records remain unchanged.

## 32. Attempt007 output policy passed; first control derivative gate failed

The [second collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt007-collect-002.json)
and [CPU readiness](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/cpu-ready.json)
record **CPU 29796458 full gate passed in 98.119 seconds**: **128 fixtures
passed**, no failures/errors/skips, **6.18 seconds**, strict loading and all
three native preparations passed. The
[actual-input CPU feature probe](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/cpu-native-feature-probe.json)
also passed: six forwards in **29.370 seconds**, exact original strict output
comparisons, finite/nonzero element-leaf gradient norm
**0.057674998699864734**, unchanged model/context. This remains CPU smoke
evidence distinct from occupancy/NO_ATOM validation.

The [saved native result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/native/native-gate-result.json)
and [scheduler log](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/native.29796475.out)
record **GPU 29796475 EXIT / exit 1**, **23 October 10:26:07–10:26:48 DTU-local**
on `n-62-18-5`, `gpua10`, one exclusive A10/four slots. The model phase made
**17 forwards in 29.345 seconds**; the controller measured **38.026 seconds**.
These are different scopes from the scheduler interval.

All active output comparisons passed under the prospectively declared
`native-repeat-loss-tv-v1` / `native-output-loss-tv-v1` policies. The original
strict raw-logit outcomes remain explicitly recorded, including failures:

| GPU comparison | Maximum absolute logit difference | Scalar-loss difference | Maximum pair probability TV | Original strict parity | New policy |
| --- | --- | --- | --- | --- | --- |
| Untouched native repeat | `5.435943603515625e-5` | `2.4059797154585283e-8` | `4.107692802790552e-6` | Failed | Passed |
| Native FP32 element boundary | `4.9591064453125e-5` | `3.453983599133181e-8` | `4.089529284101445e-6` | Failed | Passed |
| Attached C q=1 | `0` | `0` | `0` | Passed | Passed |
| Attached Z q=1 | `7.05718994140625e-5` | `2.9435869186578145e-8` | `4.652983079722617e-6` | Failed | Passed |
| Gradient-enabled attached q=1 | `6.29425048828125e-5` | `1.5825139532132937e-8` | `5.140085704624653e-6` | Failed | Passed |

This is loss/probability agreement under the declared new policy, not a strict
raw-logit pass. The [parity record](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/native/brd4_7za9_hhq-parity.json)
and [module traces](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/native/brd4_7za9_hhq-path-diagnostics.json)
are preserved. All compared values were finite.

Execution reached the first direct-q derivative: **C control, q=0.1**, on
development case 7ZA9/HHQ. The [saved derivative row](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/native/derivatives.jsonl)
records scalar loss **0.543699371294167**, autograd **0**, and these central
finite differences under the unchanged `atol=1e-6`, `rtol=0.05` requirement:

| Finite-difference epsilon | Derivative estimate | Agreement | Resolved above declared 10× repeat delta |
| --- | --- | --- | --- |
| `0.01` | `0` | Passed | No |
| `0.003` | `-1.5141254176379941e-6` | Failed | No |
| `0.001` | `1.7269917995665907e-5` | Failed | No |

Only **one of the required two scales passed**, so the unchanged derivative
gate correctly stopped the run. The recorded sensitivity is
`zero_or_unresolved`. The central scalar differences are approximately
`0`, `-9.08e-9` and `3.45e-8`, all below the declared resolution threshold
**2.405979715e-7**, ten times the observed repeat scalar difference
**2.405979715e-8**. The smallest-epsilon signal exceeds that single observed
repeat difference; the repeat delta is not a rigorous noise bound. These small
signals are consistent with forward jitter. This is evidence of an unresolved numerical check;
it does not establish a mathematically incorrect gradient or prove a
device/backend cause. The C mechanism discards q and its observed autograd
zero is consistent with that construction, but the required multi-scale
numerical validation did not pass.

No Z derivative, actual NO_ATOM-logit derivative, other q value, sweep or
confirmation case was reached. G/absence/ghost/discrete-transfer validation,
optimization and training remain unrun. The user's change removed the
raw-logit criterion only; it did not waive this derivative requirement. No
further waiver, policy change or replacement submission is selected by this
terminal result.

The [30-file artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt007/snapshot-001/collection-receipt.json)
authenticates saved-text export SHA256
`8e4f87e8b69a7b11c2d2a21097beb6424746914e0e6e2cb92447752c44c4f797`;
native result SHA256
`07db7b65d385a1a3cbb42fdc4cdcf065892560ae02007fa6040df0bbd9165d0c`.
Existing source/manifest/plan bindings remain those in section 31. Attempt007's
failed terminal decision, attempt006's failure and all earlier immutable
evidence and unrelated chemistry records remain preserved.

## 33. Attempt008 attached-repeat calibration and loss-difference validation

The user explicitly authorized fixing the noise-sensitive finite-difference
check and rerunning native parity and NO_ATOM gradients on allocated DTU HPC.
The new prospective immutable release is
`source135-protenix-occupancy-native-attempt008`, with exact policy dictionary
version **`attached-repeat-loss-difference-fd-v1`** bound in source, plan,
controller and native configuration. This is a versioned diagnostic correction
for a new run; attempt007's failed derivative decision and all earlier sealed
failures remain unchanged.

The old check divided small loss differences by `2h` and compared them with
autograd, while its resolution diagnostic used a single earlier native repeat.
Attempt008 calibrates the **actual attached dispatch at each derivative row**:
one gradient-enabled baseline plus **eight baseline repeats**, then **three
samples at each endpoint** for unchanged epsilons **0.01, 0.003 and 0.001**.
Endpoint sampling alternates plus/minus. The same frozen input context and
restored RNG policy remain in effect for every forward.

Each calibration group contains evaluations of **one identical input**: the
baseline group, or one endpoint at one epsilon. Different q/parameter values
are never pooled into one range. A fixed, per-row loss-difference allowance is
computed from the largest within-group range:

```text
U = max(1e-12, 4 × largest identical-input group range)
U must be <= 1e-6 normalized loss
delta = mean(loss at theta+h) - mean(loss at theta-h)
g = baseline autograd derivative
|delta - 2h g| <= 2h [1e-6 + 0.05 max(|g|, |delta/(2h)|)] + U
```

Every sample and gradient must be finite. Exceeding the **1e-6** uncertainty
cap fails the row; the allowance cannot grow without bound to accept a noisy
comparison. The multiplier, floor, cap, repeats, epsilons and agreement
thresholds are declared before execution. The original strict derivative-
agreement metrics remain saved as diagnostics alongside the new loss-space
decision. **U is an empirical allowance, not a rigorous statistical noise
bound or proof of a backend cause.**

The C control must preserve a single full adapted-feature hash across every
q value, parameterization and finite-difference trial, and its autograd
derivative must be **exactly zero**. At least **two of three scales** must
satisfy the calibrated agreement rule for every derivative row, including C.
Thus an invariant zero control can be numerically consistent without being
mistaken for a useful occupancy signal.

Z has a separate, stronger resolution requirement: an agreeing scale is
resolved only when its measured delta and autograd-predicted `2h g` have the
same nonzero sign and **both magnitudes exceed 10U**. A row needs at least two
such scales to count as resolved. Each case must have at least one resolved Z
q value **separately for direct-q and actual NO_ATOM-logit parameterizations**,
while all derivative rows must pass agreement. Zero/unresolved sensitivity
alone cannot yield a useful Z-gradient or study pass.

The policy changes neither native source/weights nor the attached native
method, common FP32 boundary, TF32-off model/FP64 scalar reduction, fixed
readout/denominator, geometry/bonds, three cases or frozen model/context/RNG
checks. Existing loss/TV output policies and strict-logit diagnostics remain
unchanged. CPU fixtures, all three native preparations, strict loading and
actual-input CPU preflight must pass before conditional native GPU dispatch;
all numerical work remains allocated HPC. Resource limits stay four CPU
slots/45 minutes for readiness, then one exclusive A10/four slots/one hour,
**55-minute internal cap and 3,000 model calls**. The planned repetition count
requires **1,056 nominal forwards**, below that call ceiling; exhaustion still
fails honestly. Thirty-seven added calibration/agreement regressions bring the
prospective CPU suite to **165 cases**; these counts are static expectations,
not executed validation.

This entry records prospective design only; numerical fixture results,
sealed identities, submissions and native outcomes remain pending. No
calibration, gradient pass, NO_ATOM conclusion, G/absence/ghost/discrete-transfer
validation or optimization is established by the authorization or design.
Every earlier immutable failure and unrelated chemistry, bridge and Mosaic
record remains preserved.

## 34. Attempt008 sealed and CPU submitted

The [preparation receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-preparation.json)
and [actual stage-and-queue receipt](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-stage-and-queue-001.json)
record immutable `source135-protenix-occupancy-native-attempt008`, **209
authenticated members**, stage/queue return zero and **CPU 29802633 submitted**,
`hpc`, four slots on one host, `rusage[mem=4GB]`, **45 minutes**. Actual scope
is `submitted_not_passed`.

| Attempt008 immutable binding | SHA256 |
| --- | --- |
| Release archive | `fbaf390b9da4fb1085f2c738a61b09dda9959618f4750c98a3d7d3177808662f` |
| Release manifest | `abe0646d3b022b0dac5a4e1d94338579da0fda9d970736f7e2e03f134ab657cc` |
| Execution plan | `c3986ce3b030cb440a786942a474a1df578a297493b94ce98dba6c5b23be4ae9` |

The [static seal audit](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-static-audit.json)
records **204 unchanged nonmanifest members/four changed members** versus
attempt007: this study document, controller, study implementation and fixtures.
The manifest is separately regenerated. All **202 Python members AST-parsed**;
the execution plan changes only the new FD policy and attempt identity/path/
hash fields. These are static source/byte checks, not numerical validation.

The expected **165 fixtures**, all three preparations, strict checkpoint
loading, actual-input CPU preflight and fresh quota remain prerequisites for
conditional GPU dispatch. Native resources are unchanged: one exclusive
A10/four slots, one-hour wall time, 55-minute internal cap, 3,000 calls.
Calibration/FD checks and resolved-sensitivity requirements must pass in that
allocation; no calibration or NO_ATOM-gradient result follows from submission.
All earlier failed decisions, immutable releases and unrelated work remain
preserved.

## 35. Attempt008 CPU passed; native GPU submitted

The [second collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-collect-002.json)
and [CPU readiness](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-001/cpu-ready.json)
record **CPU 29802633 full gate passed in 102.799 seconds**. All **165 fixtures
passed**, zero failures/errors/skips, in **6.00 seconds**. Strict actual-
checkpoint loading and all three supported native preparations passed.

The [actual-input CPU feature probe](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-001/cpu-native-feature-probe.json)
made six forwards in **32.623 seconds**. Native repeat, FP32-boundary, attached
C/Z q=1 and gradient-enabled q=1 matched exactly even under the original strict
comparisons. Its FP32 element-leaf backward gradient was finite/nonzero, norm
**0.057674998699864734**; model and context remained unchanged. This is CPU
input/parity/backward smoke evidence, not a calibrated GPU FD or NO_ATOM pass.

The [actual native submission](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-001/native-submission.json)
returned zero: **GPU 29802638 submitted at 23 October 21:13:20 DTU-local**, one
exclusive A10/four slots, `gpua10`, one-hour wall time/55-minute internal cap.
The [21:13:48 queue observation](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-gpu-queue-001.json)
records **PEND**, a new job waiting for scheduling, eligible pending time
28 seconds/ineligible time zero. This is a queue observation, not execution.
Native output comparisons, calibrated C/Z derivatives, actual NO_ATOM-logit
gradients and resolved-sensitivity requirements have no result yet.

The [third collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-collect-003.json)
and [later detailed queue snapshot](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-gpu-queue-002.json)
confirm **PEND** about **21:15 DTU-local**, now reporting **job-slot limit on
one host**, eligible pending time **111 seconds**, ineligible time zero.
There is no GPU controller claim or native result yet and no reliable start
time. This later resource-wait observation supersedes the initial generic
waiting-for-scheduling reason; it does not change the scientific outcome.

The [20-file saved artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-001/collection-receipt.json)
authenticates export SHA256
`a6997e2f1e6e158f822197fb6df84456f3c5dd52cf0c6185e0213cff67507007`.
CPU fixture JUnit SHA256 is
`a6c5a5f599324c7cc9491cb8dbda1a3e5f62abd855cb1628bce302b82bdd0199`;
actual CPU feature probe SHA256 is
`a76e95d32df7e86599d455f072418beb6695b3c8b63abbe87cbc8f9f861f2c2f`.
The sealed source/manifest/plan remain those in section 34. Earlier failed
attempts and unrelated work remain preserved; no G/absence/transfer validation
or optimization has run.

## 36. Attempt008 completed C/Z native parity and calibrated gradients

The [fifth read-only collection](../outputs/hpc-submissions/source135-protenix-occupancy-native-attempt008-collect-005.json),
[native result](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-002/native/native-gate-result.json)
and [GPU terminal receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-002/gpu-terminal.json)
establish **`native_parity_and_gradient_passed`**, stage
**`complete_C_Z_only`**. **GPU 29802638 DONE**, scheduler execution **23 October
21:38:03–21:41:59 DTU-local** on `n-62-18-5/gpua10`, one exclusive A10/four
slots. The model phase made **1,048 forwards in 225.749 seconds**; the
controller measured **233.808 seconds**. Peak allocated CUDA memory was
**6,061,890,560 bytes**. These are observed execution quantities, not the
prospective call estimate. CPU readiness remains passed with **165 fixtures**,
strict actual-checkpoint loading, all three preparations and real-input
forward/backward checks.

All **three cases** completed required native-repeat, FP32-boundary, C/Z q=1
and gradient-enabled q=1 output comparisons under the declared **loss/TV
policies**. This is not a claim that every original strict raw-logit comparison
passed; their diagnostic results, including failures, remain saved. The
calibrated derivative policy is exactly the sealed
`attached-repeat-loss-difference-fd-v1` from section 33, with no post-result
threshold adjustment.

The [saved-only terminal summary](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/terminal-summary-001.json)
records **15/15 active output comparisons passed**, versus **8/15 original
strict comparisons**; **108/108 calibrated FD comparisons passed**, versus
**78/108 original strict FD diagnostics**. Its SHA256 is
`eb5eec17b9137a0c0eb1ce18bdea0afcc3297c9f95eff36bb8579b92aec64969`.
This is derived metadata from saved observations, not new model execution.

The [saved derivative rows](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-002/native/derivatives.jsonl)
contain **36 passing rows**, each agreeing at **all three epsilons**: **108/108
calibrated FD comparisons passed**. All **18 C rows** had exact-zero autograd
and invariant full adapted-feature hashes. All **18 Z rows** had finite,
nonzero autograd and calibrated FD agreement; **17/18** also met the stronger
two-scale resolved-sensitivity requirement. The empirical per-row allowance U
ranged from **1.437941485882277e-7 to 3.802526480356505e-7**, below its sealed
`1e-6` cap. These are empirical repeat allowances, not rigorous noise bounds.
Passing calibrated agreement does not imply all original strict derivative
diagnostics passed.

The single Z row labelled `zero_or_unresolved` was **7ZA9/HHQ, q=0.9,
NO_ATOM-logit parameterization**. Its gradient was nonzero and agreement passed
at all three scales, but only one scale exceeded the required resolution
threshold; it is correctly not counted as resolved. Other q values met the
separate direct-q and NO_ATOM-logit resolution requirements in that case.
Every case passed both required resolution flags.

At **q=0.1**, meaning **NO_ATOM probability 0.9**, the actual absence-logit
gradients were nonzero, agreed with calibrated finite differences and were
resolved at at least two scales in every case:

| Case | Derivative of fixed scalar loss with respect to NO_ATOM logit |
| --- | --- |
| Development: 7ZA9/HHQ | `0.0011750486446544528` |
| Confirmation 1: 7ZAD/IJL | `-0.001694127218797803` |
| Confirmation 2: 7ZAJ/HH8 | `0.0005621041636914015` |

This directly supports the proposed gradient mechanism: retaining unnormalized
real-element probability mass allows the frozen model's fixed-context scalar
to depend differentiably on the NO_ATOM logit even when NO_ATOM is the largest
probability. It does **not** establish that zero element mass removes an atom
from native attention, geometry, bonds or pooling, that the model can choose a
chemically valid atom count, or that optimizing this scalar improves binding.
These C/Z controls deliberately clamp those other inputs. No new learned
layers or model-weight changes were used for this mechanism check.

All cases saved **18 sweep rows each/54 total**, plus completed case records:
[development](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-002/native/brd4_7za9_hhq-complete.json),
[confirmation 1](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-002/native/brd4_7zad_ijl-complete.json),
[confirmation 2](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-002/native/brd4_7zaj_hh8-complete.json).
G occupancy propagation, the exact absent endpoint, ghost invariance,
discrete-deletion transfer and latent/occupancy optimization remain
**not run/not validated**. No training or automatic follow-on submission is
established by this completed foundation phase.

The [37-file artifact receipt](../outputs/hpc-results/source135-protenix-occupancy-native-attempt008/snapshot-002/collection-receipt.json)
authenticates export SHA256
`ef5f0f6bda3ba470d6ede8ca6ac6dc66cf5315b78cc3f4ba37f098b5752f07d2`;
native gate SHA256
`026cd2e2f2676d9b729c28221070174a87e0446b33a2bf0763611283a937438f`.
The source/manifest/plan bindings remain those in section 34. Every failed
earlier attempt, original strict diagnostic and unrelated chemistry, bridge
and Mosaic record remains preserved.
