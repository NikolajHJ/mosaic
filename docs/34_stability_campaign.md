# Matched trajectory stability campaign

**Current verified snapshot, 7 October, 10:03:55 DTU-local:** primary
**29619053 RUN**, continuation **29619054 PEND**; one evaluated full
100-update trajectory, one optimizing, 46 unattempted and no failures/stalls
so far. First `prior-0-legacy-fixed_initial_mask` retains 101 states and
raises the relaxed proxy 0.3016021 → 0.5048858, with 45 within-context loss
increases, unchanged 22-atom occupancy and **zero valid argmax graphs or
categorical samples (101/808 attempts respectively)**. Labels change one O
to C while the hard graph stays invalid. Three-edit repair witnesses at
steps 0/50 are observed bounds; searches at 25/75/100 are censored, and
invalid raw endpoints permit zero fresh-native predictions. The active-set
counterpart is at step 17, proxy 0.3823778, count 22 and invalid argmax.
Matched policy outcomes and full-cohort chemistry/count behavior remain
pending. The [ledger](05_experiment_ledger.md) links the verified 16-member
snapshot and unchanged source064/protocol identities. Earlier notes retain
their timestamps and do not imply later campaign completion.

**Verified stage, 7 October, 09:52:26 DTU-local:** native smoke
**29619052 DONE**, `native_smoke_passed=true`, `full_trajectory_validation=false`.
Both C13-variable and prior-4-fixed cases accepted three of three updates,
without stalls. Scales were `1/1/1` and `1/8, 1/16, 1/16`; counts remained
13/23. Four C13 argmax states were valid and zero of four prior states were
valid. This verifies short real-model gradient/step/terminal behavior, not
chemistry success or count adaptation. Primary **29619053 RUN** began
09:50:00; continuation **29619054 PEND**. The 48-case campaign remains
in progress, with unchanged source064/protocol identities and declared
budgets below. Earlier 09:40 and preparation notes are historical snapshots.

Downloaded 14-member smoke bundle is
`outputs/hpc-results/source064-stability-retry/smoke-pass-29619052.tar.gz`,
SHA-256 `3d032ff13c752c4cea8be2477a3fdf6e37aecb7521e8225103d951ab7e2a8c9a`.
Its sibling identity receipt/scheduler/controller and two case reports retain
the pass evidence. LSF interval was 09:36:30–09:49:59, 809 seconds on L40S;
no fresh-native endpoint prediction was part of the smoke. Exact artifacts,
resources, prior failures and limitations remain in the
[ledger](05_experiment_ledger.md).

**Verified execution status, 7 October, 09:40:55 DTU-local:** the source063
CPU gate failed two fixtures (220 passed), and its three blocked GPU jobs
were cancelled without execution. A separate immutable source064 changes
only `tests/test_stability_study.py`; production code/cohort/settings remain
unchanged. Retry CPU **29619051 DONE** passed **222 tests in 20.35 seconds**.
Native smoke **29619052 RUN** started at 09:36:33 on L40S `n-62-13-14`;
primary **29619053 PEND** requires `done(29619052)`, continuation
**29619054 PEND** requires `ended(29619053) && done(29619052)`.
No passed smoke, completed full campaign or new native molecule inference
is established yet. The earlier packaging/partial SSH notes below are
historical snapshots, superseded by downloaded receipts.

Actual executable release is
`/work3/s234818/latentmol/releases/source-064-stability-fixture-repair-attempt001`,
archive SHA-256 `02d67412fb754296649f8695d82cefd6fb8ea647e9466d5e519cf4d33e3914c9`,
manifest SHA-256 `68e5280922addf099fd9883847ad204bf2ac1741b11150deea0ab5fd9ee67fd2`.
The unchanged 48-case protocol SHA-256 is
`b4488b0422432f7de9613baea73cc245e610953e07d89aeb261e74bfc313a3d0`,
at `/work3/s234818/latentmol/submissions/source063-stability-attempt001/protocol.json`.
Retry intents/receipts use `source063-stability-attempt002`; job/output names
retain campaign source063 while executable identity is source064. Downloaded
initial/failure evidence is `outputs/hpc-results/status-20261007T073227Z/results.tar.gz`
(SHA-256 `3f92ed546076700dc2f65dc32a7e6d5b8b0e568e6aa7ed01f78d2940d6ad54ae`);
retry evidence is `outputs/hpc-results/source064-stability-retry/receipts-20261007T074055Z.tar.gz`
(SHA-256 `1ca242015740d76a7b133f532442797f2f6a0fccae67221f706c003d6e120ff2`).
The [source-diff receipt](../outputs/hpc-submissions/source064-source-diff.json)
confirms the fixture-only change. The [ledger](05_experiment_ledger.md)
retains exact commands, CPU/scheduler evidence and all original failures.

**Provisional execution update, 7 October 2026:** source063 was uploaded,
the metadata protocol materialized at SHA-256
`b4488b0422432f7de9613baea73cc245e610953e07d89aeb261e74bfc313a3d0`,
and four gated stages submitted (29619018–29619021). The CPU gate failed
two test fixtures, with 220 passed, before any native stage. A reviewed
test-helper-only correction is being packaged as source064 using the same
48-case scientific protocol/output root, with unique retry receipts.
No passed smoke or campaign is claimed. These are coordinator-observed SSH
results pending downloaded evidence; source064 identity/submission and the
old GPU-stage cancellation receipts await verification. The 6 October
unsubmitted packaging notes below are historical snapshots.

Implemented, statically inspected protocol, 6 October 2026. The user authorized implementation and
submission of additional stability experiments. The coordinator declared
48 trajectories and the sampling/gate/native schedule below. Source063 is
sealed/verified locally; remote input/protocol identities, resource allocation
and submission receipts are pending. The user subsequently reported HPC
unavailable: the archive is **not uploaded, unsubmitted and numerically
unvalidated**. No campaign result is established by implementation/packaging.
No local numerical checks are authorized for this task. No confirmed
remote protocol exists merely because construction was attempted through
a dead SSH session. Execution records belong in the
[experiment ledger](05_experiment_ledger.md).

The local [source archive](../outputs/source-releases/source-063-stability-attempt001.tar.gz)
SHA-256 is `304a6eb6dd656da16ed6fef5ad68d19abef5c81c5ed02d99f46598dded29cb33`;
scientific source manifest SHA-256 is
`aa69181861ec62dee73e566a5e9e2b4dd2b830403c60af9538e2069d7aa56ed8`,
269 source files/308 unique archive entries.
[Packaging](../outputs/source-releases/source-063-stability-attempt001.tar.gz.json)
and [independent verification](../outputs/hpc-submissions/source063-package-verification.json)
receipts confirm file identity only. No protocol SHA-256 or source063 job
is known. The archive retains the prehash documentation snapshot; this later
hash/status append does not alter its scientific source or manifest.

## Question and boundaries

Can the frozen VAE/Boltz-2 latent pipeline sustain a requested 100-update
trajectory, and does explicit step acceptance prevent worsening proposals
without silently suppressing failures or changing molecular identity?
The earlier source062 comparison evaluates individual saved states; it
does not establish this longer-trajectory behavior. The operation that
causes the observed sharp curvature remains unresolved.

Use the same pretrained frozen decoder/scorer, shared relaxed graph,
objective, conditional-element policy and native preparation rules across
arms. This campaign changes update control and declared occupancy policy,
not training losses or model weights. Chemistry conversion remains outside
the differentiable path. Fresh-native scoring is an endpoint evaluation;
it does not feed back into these trajectories.

## Matched conditions

Freeze the eligible prior-latent and carbon-chain starts, target sequences,
MSAs, native pose seeds, initial contexts and all state/model/file hashes
before campaign outcomes. Include both initialization classes rather than
choosing starts after seeing favorable final molecules. Preserve any
initialization, native-preparation or derivative-gate failure in the original
denominator. Do not replace a failed start with a favorable one.

The full matched subset compares three update policies under the occupancy
conditions in the cohort table below. Additional prior seeds extend the
backtracking initialization survey without three-policy controls:

| Policy | Proposal and acceptance | Distinct purpose |
| --- | --- | --- |
| Legacy clipped update | Original learning rate, clipping and initial-z radius projection; no added acceptance condition | Historical behavior comparator |
| Armijo backtracking | Exact legacy proposal first, then halving within sixteen total trial scales; require strict, measurable sufficient decrease | Temporary safeguard described in [docs33](33_step_control_diagnostics.md) |
| Fixed 1/8 step | Learning rate 50 rather than 400, with the same clipping and initial-z radius projection; no adaptive acceptance | Test whether a simply smaller step explains the improvement |

The declared historical reference is learning rate 400, gradient-norm cap
0.000125 and radius 3; their nominal preprojection maximum step is 0.05.
The 1/8 control therefore has nominal maximum 0.00625. The submitted config
must declare these values explicitly; any later change requires a separately
identified experiment rather than silently modifying the comparator.

All matched policies reuse the exact saved initial-latent file and preparation
for that start, including the original initial state rather than a later
failed-transition state. No latent redraw or re-encoding is substituted.
Their gradients subsequently differ because trajectories diverge;
matching initialization does not mean forcing a shared gradient after that
divergence. The global radius stays centered on the original latent.

**Fixed mask:** retain the initial occupied slots and bound native context
for the trajectory. This isolates step behavior without count transitions.
**Variable mask:** apply the declared NO_ATOM selection between updates,
with native contexts prepared/audited under the existing dynamic-count
policy. Each individual derivative/acceptance trial still holds its selected
mask and pose/context fixed. Atom count can change between steps; latent
vector width does not change.

The declared cohort is:

| Starts | Policies | Occupancy arms | Trajectories |
| --- | --- | --- | --- |
| Prior seeds 0–9 | Backtracking | Fixed initial mask and variable mask | 20 |
| Prior seeds 0, 4, 6, 8 | Legacy 400 and fixed learning rate 50 | Both arms | 16 additional |
| Historical C13 reference | All three policies | Both arms | 6 |
| GSTP1 C6 and PPIA C6 | All three policies | Fixed initial mask only | 6 |
| Total | | | **48** |

Thus 36 trajectories form complete three-policy matched groups, while twelve
trajectories from the remaining six prior seeds provide backtracking-only
coverage. Do not attribute a matched-baseline advantage to those six seeds.
The original saved start/model/input hashes and categorical RNG seeds must
be pinned in the immutable protocol before submission; this draft does not
claim every existing artifact has passed those gates.

This is a purposive stability cohort, not an unbiased chemical-utility survey:
seeds 4/6/8 are the shrunk-step stress cases and seed 0 is a full-step
control from source062; GSTP1/PPIA C6 were chosen after their historical
sulfur substitutions. Membership is frozen before new source063 outcomes,
but uses those older observations. Report paired stability within these
groups rather than claiming a population-wide binding or heteroatom-success
rate from the selected starts/targets.

## Trajectory, stalls and failures

Request at most 100 updates per arm and preserve the distinction among requested,
attempted and accepted updates. A complete allocation or finalized controller
does not establish a completed 100-update trajectory.

The Armijo defaults are version `armijo_v1`, sixteen trial scales,
Armijo fraction `1e-4` and two actual scalar ULPs plus supplied baseline
variation as the observed-improvement guard. This is not a confidence bound
on whole-model arithmetic. Record each rejected/evaluated trial and its
cause. If the trial budget is exhausted, retain the latent and explicitly
record a stall and incomplete requested budget. The source063 implementation
stops that trajectory at its retained terminal state. It does not count an
unchanged point as a successful update. The controller may continue to later
cases; that continuation does not extend the stalled case's update budget.

Retain initial/native/gate failures, nonfinite scalar results, integrity
errors, timeouts, memory failures and unexpected exceptions separately.
Malformed model outputs and programming errors propagate; they are not
called chemistry failures or converted into apparently successful stalls.
Source063 adds a dedicated serial controller and explicit endpoint-reader
termination metadata for early stalls; earlier historical readers assumed
a full update budget. These additions remain numerically unvalidated until
their HPC fixtures and native smoke pass. Case-boundary continuation checks
source/protocol identity and sealed attempt hashes, skips terminal failures
as well as terminal successes, and retains interrupted attempts separately.
It restarts an unfinished case from its authenticated saved initial latent;
it does not resume an optimization halfway through its accepted updates.

An independent `terminal/CASE_ID.json` receipt authenticates the sealed
attempt. On continuation, a terminal receipt published before interruption
can recover a stale case table; the receipt, case identity, complete file
membership and every attempt-file hash must agree before it is skipped.
Missing, altered or mismatched evidence fails integrity checks rather than
silently rerunning or crediting a terminal case. These are intended
engineering safeguards, not yet passed HPC tests.

## Evidence at every update and context transition

Save the latent, selected mask/context identity, objective and output dtype,
gradient norm and finite-state checks, clipping/projection factors, actual
step displacement, update status, trial scales and oracle counters. Opted-in
legacy/fixed-step metrics also record clipping/projection factors and actual
versus proposed displacement, preserving original arithmetic. Source063
adds `oracle_cost_at_state` to opted-in state metrics and a corresponding
`trace` in `trajectory_report.json`: loss, gradient norm, count, context and
cumulative cost can be compared against physical calls rather than step
number alone. Historical metrics without that field remain explicitly
unmeasured rather than receiving invented cost values.
Record raw/conditional atom and charge distributions, typed bond evidence,
occupancy preferences and hard labels at the declared checkpoint cadence.
Log derivative diagnostics under the explicit `local_v1` policy; retain
proposal rejections, legacy projection outcomes and neighboring-scale passes.
The declared derivative/native-parity gates occur at steps **0, 50 and 100**,
plus each context change and an early-stall endpoint. Missing later gates
are explicit when a trajectory terminates early. Optional instrumentation must not change native model
mathematics or hide chemistry conversion in autograd.

For variable-mask transitions, record old/new masks, slot identities, count,
prepared feature/pose hashes, native parity and pre/post-transition scores.
An accepted step's loss decrease is measured in its old fixed context;
a score after a discrete occupancy/context jump is a different comparison.
Report jump magnitudes separately. Make no global monotonicity claim by
connecting values scored in different contexts.

## Physical compute budgets

These are matched state/model/loss comparisons with **unequal oracle
budgets**. Legacy and fixed-step arms form one proposal per update; Armijo
can add up to sixteen score trials. Separate optimization forwards,
backwards, trial calls, context preparation, derivative probes, chemistry
diagnostics and native endpoint predictions. Count physical executions,
not just logical score requests; report exact tensor-cache reuse and hits.
Checkpoint backward recomputation, elapsed GPU/allocation time, hardware,
precision and peak memory remain disclosed.

Report outcomes both per requested trajectory/update and against physical
compute cost. Do not call a method more efficient because it accepts more
steps with more oracle calls. A budget-normalized view supplements the
matched 100-update comparison; it does not erase the primary requested
budget or censored/failed cases. Allocation caps and any continuation must
be declared with immutable releases and unique output directories. The plan
uses one serial GPU workflow and sequential continuation if an allocation
ends, preserving terminal cases and unfinished-state identities. No parallel
48-job array or actual allocation duration is established by this draft.

## Chemistry and fresh-native endpoint checks

Evaluate declared starts, retained trajectory states and terminal states using
the same hard-graph procedure across arms: preserve **argmax plus eight
categorical draws per state**, with no favorable redraws. Sampling budgets
and state membership remain explicit, including invalid candidates.
Report validity, connectedness, atom/type/count changes, categorical entropy,
degenerate outputs and the existing graded/probability-aware repair
diagnostics separately. A bounded repair witness is an observed upper bound;
no witness is censored, not proof that no repair exists. Do not repair a
candidate silently and then attribute its score to the unrepaired graph.

Use the existing forward-only fresh-native machinery, with audited native
input identity, for **start and terminal argmax** at pose seeds **17 and 23**
across paired comparisons. Evaluate all valid eligible terminals, with no
selection of the best proxy endpoint. Categorical draws remain chemistry
diagnostics in this first campaign scope, not additional native requests.
Both affinity heads remain
separate, along with invalid, missing, failed and ineligible requests. The
endpoint protocol must accommodate stalled/failed parent trajectories without
claiming that every requested checkpoint exists. Avoid choosing only the
best endpoint or best native seed after observing scores.

Native preprocessing must reconstruct the intended shared molecule and
retain the target/MSA/feature-loss limitations. Relaxed frozen-pose proxy
improvement and fresh-native prediction are different outcomes; their
disagreement is informative and must remain visible. No experimental
binding, synthesis or selectivity claim follows from this campaign.

## Planned validation before full trajectories

The native smoke runs two independent backtracking trajectories from exact
saved starts: `carbon-c13` with variable occupancy and `prior-4` with fixed
occupancy. Each requests three updates, audits every retained step and uses
argmax plus one categorical draw. Each child has a 900-second cap, for a
combined 30-minute child bound inside the planned 45-minute L40S allocation.
Fixtures/setup are additional. A verified early stall can remain terminal,
but each smoke trajectory must have at least one accepted update and no
unexplained score increase between states with the same context identity.
The smoke checks native Boltz gradient execution, identities and terminal
bookkeeping; it does not run fresh-native endpoint predictions or establish
100-update stability.

The prepared dependency chain is passed CPU engineering checks, then passed
native smoke, then a 23-hour serial campaign allocation and a separate
23-hour boundary continuation. The continuation requires both an ended
primary and a passed smoke: a failed smoke cannot release its workload.
At this access-outage snapshot none is submitted; no passed fixtures,
smoke, campaign or confirmed remote protocol is claimed.

| Stage | Queue / requested resources | Wall limit | Dependency |
| --- | --- | --- | --- |
| Engineering fixtures | `hpc`, 2 cores, 4 GB/core (8 GB host) | 30 minutes | None |
| Two three-step native smoke cases | `gpul40s`, 4 cores, 8 GB/core (32 GB host), one exclusive GPU | 45 minutes | `done(cpu)` |
| Serial 48-case primary | Same requested GPU resources | 23 hours | `done(smoke)` |
| Serial boundary continuation | Same requested GPU resources | 23 hours | `ended(primary) && done(smoke)` |

These are planned requests, not observed device or execution claims. The
controller's budget per campaign allocation is 22 hours, leaving scheduler
headroom. Per case, optimization allows 1,800 seconds, the supplemental
repair audit 300 seconds, and native start/endpoint evaluation 3,600 seconds.
The controller reserves their sum plus 180 seconds (**5,880 seconds**) before
starting another case. A case near all three caps can consume 95 minutes;
two allocations therefore do not guarantee processing all 48 cases. A
partial campaign retains its original denominator and requires separately
recorded continuation if unfinished work remains.

## Modules, preparation and future HPC commands

[stability_study.py](../src/latentmol/boltz/stability_study.py) expands the
frozen cohort into serial policy/occupancy cases, verifies prerequisites,
builds independent optimization configs, and records trajectory/repair/native
children. [trajectory_terminal.py](../src/latentmol/boltz/trajectory_terminal.py)
authenticates completed-budget or explicit stalled boundaries for the
[endpoint reader](../src/latentmol/boltz/active_set_rescore.py). A verified
stall is a true terminal observation, not convergence or a completed budget.
Each child is authenticated against the declared release/config/parent;
successful process exit alone is insufficient.

[prepare_stability_protocol.py](../scripts/prepare_stability_protocol.py)
is metadata-only HPC preparation: it resolves the original ten prior starts,
C13 and two C6 carbon starts, hashes their saved latent/parent/native-template
files, checks frozen decoder/scorer/confidence identities and all twelve
source062 prerequisites, and freezes exactly 48 memberships. It does not
draw latent vectors, optimize or run models. An existing equal protocol and
identity receipt can be recovered; differing metadata is not overwritten.
Its identity receipt says `prepared_not_submitted`, `numerical_validation:
not_run`. Historical failed experiments remain failed while supplying their
authenticated original starts.

[submit_stability_study.py](../scripts/hpc/submit_stability_study.py) verifies
archive, source manifest, protocol and prerequisites, then **prints the plan
by default**. Only `--submit` invokes `bsub`; `--no-continuation` omits the
fourth stage. Structured arguments and exclusive per-stage intent/receipt
files retain uncertain scheduler replies and prevent blind duplicate
submissions. Missing/uncertain replies require scheduler inspection before
recovery. This helper does not change SSH or VPN settings.

The exact future commands below are templates for the DTU environment after
uploading and verifying a sealed release. `RELEASE`, `ARCHIVE_SHA` and
`PROTOCOL_SHA` must be replaced with verified identities; they are not known
hashes or claims that a remote file exists. Configure the pinned Boltz
environment and `PYTHONPATH=RELEASE/src` before these metadata commands:

```text
python RELEASE/scripts/prepare_stability_protocol.py --release RELEASE \
  --output /work3/s234818/latentmol/submissions/source063-stability-attempt001/protocol.json

python RELEASE/scripts/hpc/submit_stability_study.py --release RELEASE \
  --archive-sha256 ARCHIVE_SHA \
  --protocol /work3/s234818/latentmol/submissions/source063-stability-attempt001/protocol.json \
  --protocol-sha256 PROTOCOL_SHA \
  --receipts /work3/s234818/latentmol/submissions/source063-stability-attempt001/receipts
```

The second command prints only the verified plan. Add `--submit` on a later
authorized execution turn to queue the dependency chain. Inside allocations,
the helper wraps commands with [boltz_run.sh](../scripts/hpc/boltz_run.sh)
and experiment metadata capture. Each submitted command explicitly binds
the archive and work-root environment variables, then runs
[verify_stability_release.py](../scripts/verify_stability_release.py) inside
LSF to recheck the exact queued archive and source-manifest hashes. An
inherited override or changed source cannot silently redefine that identity.
This metadata guard remains unexecuted along with the rest of source063.
The stage entrypoints are
[boltz_stability_tests.sh](../scripts/hpc/boltz_stability_tests.sh) and
[boltz_stability_study.sh](../scripts/hpc/boltz_stability_study.sh):

```text
bash scripts/hpc/boltz_stability_tests.sh
bash scripts/hpc/boltz_stability_study.sh PROTOCOL PROTOCOL_SHA --smoke-root /work3/s234818/latentmol/runs/source063-native-smoke
bash scripts/hpc/boltz_stability_study.sh PROTOCOL PROTOCOL_SHA
bash scripts/hpc/boltz_stability_study.sh PROTOCOL PROTOCOL_SHA --resume
```

The local command
`.\.venv\Scripts\python.exe outputs/hpc-submissions/source063-static-review.py`
passed at 21:43:27 UTC, using Windows Python 3.12.12/Anaconda 64-bit.
It parsed ten Python files and inspected twelve files for whitespace plus
two shell files for LF/no BOM. It imported no project modules and performed
no numerical/model/network/submission work. Implementation, exact file hashes
and receipt are `outputs/hpc-submissions/source063-static-review.{py,json}`;
the ledger retains their hashes. This is static hygiene, not a passed test
suite or native smoke.

The CPU suite includes the earlier source062 regression set and new terminal,
campaign, repair and submission fixtures. Its source063 pass count is unknown
until executed; earlier 159 passes are historical evidence, not a pass of
this expanded source063 suite. Full campaign output is planned at
`/work3/s234818/latentmol/runs/source063-stability-campaign`, with `identity.json`,
`cases.json`, `summary.json`, independent `terminal/` receipts, immutable
`cases/CASE_ID/attempt-NNN/` artifacts and per-allocation metadata.

The probability-aware repair audit evaluates prescribed steps 0/25/50/75/100
and the actual terminal step; missing later states remain missing. Its
declared bounded search is three edits, 256 checks per depth and beam four.
Witness/censoring and timeout outcomes remain distinct from optimization or
native eligibility; no repaired witness replaces a scored graph.

## Acceptance criteria and decisions

**Engineering stability evidence** requires authenticated inputs/frozen
states, native-input parity and declared derivative gates; explicit finite
gradients/objectives; correctly counted accepted updates; complete trial,
stall, failure and context-jump records; and bounded physical resource use.
Claim a completed 100-update trajectory only when all 100 requested updates
actually completed. If some terminate early, report their rate/reasons and
partial outcomes without calling the campaign universally stable. A stall
can demonstrate safe refusal of a proposal while remaining a failed budget
completion. Compare all conditions in the original denominator.

**Chemical usefulness evidence** is a separate question: hard/sample
validity and topology, degree of repair, representation degeneracy and
changes from the start. Stable gradients or accepted proxy steps do not
imply valid or useful molecules. Any advantage must preserve matched draw
budgets and expose rejected/unsupported graphs.

**Predicted activity evidence** requires eligible audited fresh-native
start/endpoint comparisons with both heads and the declared pose seeds,
not just the optimized proxy. State observed head agreement/disagreement
and denominators rather than converting a predicted score into binding
proof. Broad practical utility would require subsequent independent
evaluation beyond this engineering campaign.

Use the results to distinguish adaptive acceptance from simply taking a
smaller fixed step, and fixed-context stability from occupancy-transition
behavior. Neither success identifies the operation responsible for sharp
curvature. That diagnosis remains a separate line of investigation.
