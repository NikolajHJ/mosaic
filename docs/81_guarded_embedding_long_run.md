# Guarded original epoch36 mapping optimization

## GPU fit passed; pilot handoff failed scheduler guard, collected 28 October 2026 20:22 CET

The [new read-only collection](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/status-1793215369272490300.json)
contains the authenticated **fit stage passed** result and execution receipt:
GPU **29839755** ran on `n-62-18-12` from **16:23:53 to16:39:52 CET**, returned
zero and completed all **5,000 supervised bridge updates**. Full stage elapsed
952.792s; the fit itself took435.400s. Development loss improved from
**0.0510884362 to0.0148141605**, selecting step4,800. The development-calibration
stage passed; the frozen VAE and bank remained unchanged.

Untouched audit loss was **0.0151985238**; **1,605/1,653** audit examples were
within every fixed development cap, without adjusting caps. Audit projection
identity accuracy was about99.83% for elements and99.91% for bond types, with
unsupported categories still unvalidated. These are mapping-fit observations,
not native-gradient, chemical improvement or optimized-molecule results.

| Saved fit artifact | SHA256 |
| --- | --- |
| Fit result | `d14ef4d6adc31939e13bd92ae06d3af83db64cc91fbc57aeb5c7729414492327` |
| Selected bridge checkpoint | `c8d86551849c7f7e1de5218b7121cbecacb7f9dbc9ee11a5aea5972187a81551` |
| Fixed development calibration | `1c436618bc6dc26dfb049908b11477afc5f2e7b291132db4577a99631f0c69f8` |
| Untouched audit | `e04ca339521f164ddeef88040962d4a5b358e64c04b937a539fbd333f1aa0f3d` |

There is **no pilot intent, receipt, result or `advance-after-fit.json`** in
this collection. The separately saved [exact callback logs](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/callback-inspection-1793215376515856900.json)
establish that callback **29839756 ran16:39:53–16:39:57 CET and exited1** in
`Campaign.advance("fit")` → `require_done("fit")`, with
`ValueError("Predecessor allocation is not exact-name DONE")`. The check runs
before creating a transition or launching a pilot. It calls
`bjobs -a -noheader -o "jobid stat job_name:200"` and requires the split stdout
to match `[job_id, "DONE", exact_intent_label]` with returncode0. The callback
log does not preserve the actual returned fields/status; a parser discrepancy,
name/state mismatch or scheduler timing cannot be distinguished from this
evidence. This failure is at the scheduler handoff, before any pilot CUDA
initialization, not an observed pilot native-gradient or chemistry failure.

CPU29839742/29839743 and fit29839755/callback29839756 have now aged out of the
live listing. Its return255 does not undo the independent completed fit
execution/result or the callback's saved terminal failure. Fresh native/
derivative gates, pilot and long optimization remain unrun; no successor is
claimed. This was a read-only status task, with no submissions or runtime,
SSH or VPN changes. The [raw collection receipt](../outputs/hpc-submissions/source142-epoch36-guarded-embedding-long-attempt001/collect-1793215369272490300.json)
returned zero; the callback inspection was supplied as an existing local
receipt after this collection, without another connection from this observer.

## Read-only status, 28 October 2026 13:58 CET

The [fresh collection](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/status-1793192294889844300.json)
confirms fit **29839755 remains PEND**. Four CPU slots are reserved on
`n-62-18-11`; processor units cannot currently satisfy affinity on one host.
Pending time is eligible:8,982 seconds,zero ineligible. Callback **29839756**
waits on its expected `ended(29839755)` dependency. Reservation is not execution.
Only the completed CPU stage result exists; no fit, calibration, GPU pilot or
latent optimization result exists, and there is no reliable start ETA.

The combined `bjobs` return255 reflects only aged-out CPU29839742/29839743
records. It is not an SSH failure or a failed fit; their authenticated saved
450-fixture CPU pass remains valid. The [raw collection receipt](../outputs/hpc-submissions/source142-epoch36-guarded-embedding-long-attempt001/collect-1793192294889844300.json)
returned zero. This read-only check changed no code, settings, jobs or SSH/VPN
configuration; all fit/calibration/native/pilot prerequisites remain required.

The user authorized implementation and HPC execution on 28 October 2026 and
clarified **10,000 updates per starting molecule**. This independent source142
campaign retains the same three prescribed starts, original frozen epoch36 VAE,
target, context, pose, fixed occupancy, native settings and base optimizer as
completed embedding comparison006. It plans 30,000 iterations in total.
The existing source140 VAE continuation and source141 native-only recovery are
separate campaigns; this study changes neither.

## Why change the short experiment

Comparison006 passed its three native oracles and six pointwise start derivative
checks, but every learned terminal molecule was invalid. Valid intermediate
molecules existed. Across the three learned trajectories, average projection
MSE rose by approximately 6.4–17.5 times. A better learned score therefore did
not establish a better decoded molecule. Exact start derivatives establish
the implemented function's derivative, not reliable chemistry throughout a
long latent trajectory. See [comparison evidence](78_embedding_latent_comparison.md)
and the authenticated trajectory audit under
`outputs/hpc-results/source138-epoch36-embedding-comparison-attempt006/trajectory-audit-20261028-001`.

## Frozen settings and explicit additions

- Original epoch36 VAE SHA256
  `3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
- Same starts, source sample rows0/3/6, occupied slots25/19/19. Original shared
  starts file SHA256
  `832592de7183b51ac22542ead558bc453006d3d349d942e2933dd1a52fb40ebd`.
- Same DBMol affinity +2.5contact +2.5confidence objective, arithmetic and
  final FP64/cast-back graph policy. LR400, gradient cap1, original-start global
  trust radius3, best_trial_v2 and16 trials. No model weights change during
  optimization. A forced eligible finite fallback retains the inherited rule.
- A larger supervised bridge fit warm-starts the source137 mapping:5,000
  updates, batch32, AdamW LR0.001. New source-root attempts are4,096 training,
  512 development and512 audit, plus1,024 prior attempts, with the original
  descendant sampler. These are attempted roots, not guaranteed accepted
  counts. All historical roots/groups/decoded identities are excluded before
  new sampling. VAE and native Boltz weights remain frozen; desired scores
  are never labels. Development selects the checkpoint; audit is read once.
- Each native projection gets a per-example, NONE/present-group-balanced MSE
  cap from the **new development** set's nearest-rank99th percentile, with
  floor1e-8. Caps are fixed before untouched audit and before checking the
  three benchmark starts. No automatic cap enlargement or start replacement.
- Every candidate must have strict-valid conditioned hard chemistry and all
  projection MSE values within those caps before an expensive learned Boltz
  trial. This is a detached eligibility check, not an added differentiable
  loss. Natural occupancy/validity remain observations. Sparse or absent
  charged/element categories remain unsupported evidence.
- The common hard fixed-context DBMol score ranks eligible valid checkpoints.
  Its cache key includes exact slot-graph tensors and context, since canonical
  SMILES alone cannot identify a fixed-pose score. Best valid states are kept
  even when a later learned state has a better surrogate score.

## Observation, resume and stopping

Every10 iterations, plus0 and10000 and a scientific stall endpoint, save the
actual relaxed atom/charge/bond probabilities, conditioned and natural hard
graphs, latent, validity, mapping error and learned/relaxed/hard fixed-context
scores. The relaxed score is observational; updates follow the mapped path.
Every hard slot-graph change is also considered for best-valid retention.
All invalid/rejected candidates and no-op reasons remain receipts.

Each completed iteration atomically saves the latent, original starting
latent, RNG state, call/step counters, best valid state and immutable identities.
Append-only logs avoid rewriting the complete history per update. Resume is
bound to source, plan, VAE, bridge, calibration, original derivative gate,
context and start; the radius stays relative to the original latent across
allocations. State files and hard-cache receipts are hash authenticated.

The10,000 target counts optimization iterations, including explicit no-ops;
accepted nonzero movements are reported separately.25 consecutive no-ops stop
the campaign as a **scientific stall with an incomplete target**. Model errors,
nonfinite native behavior, changed frozen state or failed assertions stop it
with the evidence preserved. No failed condition silently relaxes a gate.

## Staged execution and current status

Implementation and AST/JSON checks are local only. All numerical fixtures,
data generation, fits, pilots and evaluations run in allocated DTU HPC jobs.
The new immutable plan is `configs/epoch36_embedding_long.json`.

1. Allocated CPU gate: inherited contracts, new fit/guard/resume/controller
   fixtures, actual generated CLI dispatch and exact original-start parity.
   Four CPU slots,45-minute walltime and40-minute internal budget use the
   explicit CPU-only compatible-MKL/four-thread profile before torch imports.
2. One exclusive80GB A100 fit allocation: warm fit, strict dev improvement,
   fixed development calibration and untouched audit;four slots/four-hour
   walltime,3h50m internal.
3. One exclusive80GB A100 pilot: a tiny CUDA child exits, then an actual
   ordinary native baseline prediction must succeed, then a separate GPU
   model child passes all three native oracles, all six fresh derivative gates
   and exact production bindings before20 iterations for each start. Pilot
   iterations count toward10000. All starts require nonzero accepted progress,
   retained valid best states and finite time/storage profiles. This allocation
   also uses four slots/four-hour walltime,3h50m shared internal time, including
   process ownership and native smoke rather than resetting the clock afterward.
4. Conditional serial round-robin allocations, each at most24h wall/23h
   internal, resume authenticated states until every case reaches10000.
   Update counts derive conservatively from pilot time with safety factor1.5
   and1,800s reserve. Every dispatch and launch checks fresh quota for at least
   5GiB free;long segments additionally reserve their pilot-projected storage
   before applying that headroom check. No successor follows a failed stage.
5. Separate final native allocation rebuilds every valid baseline, terminal
   and best-valid identity at seeds17/23, deduplicated, at most18 requests.
   The pilot baseline smoke is counted once and reused exactly. Native fresh
   affinity is a separate endpoint from fixed-context full DBMol scores. This
   final allocation also has24-hour walltime/23-hour internal time;it does not
   inherit the shorter fit/pilot limit. Failed or invalid outcomes are retained.

CPU-only coordinators never initialize CUDA. Fit/optimization and ordinary
native predictions have process-lifetime ownership boundaries; freeing tensors
alone does not release an exclusive-process GPU context.

At documentation creation, **no source142 numerical gate, mapping fit, pilot
or long trajectory has passed or started**. Submission and observed results
will be recorded below with exact immutable identities and scheduler receipts.

Entry points: `scripts/embedding_long_campaign.py` (local stdlib packaging),
`scripts/hpc/embedding_long_campaign.py` (fail-closed controller),
`src/latentmol/boltz/embedding_long_experiment.py` (allocated stages),
`embedding_long_training.py` (fit/calibration), and
`embedding_long_optimization.py` (guarded resumable trajectory).

The HPC controller exposes `submit-initial`, `advance --after <exact-node>`
and read-only `collect`, with an internal allocated `run-stage --node` action.
Each worker and its one-slot/30-minute ended-dependency callback have exclusive
submission intents/receipts outside the sealed release. Every dispatch binds
archive, manifest and plan hashes; allocated launches reauthenticate those
bytes and pinned predecessor results. Ambiguous submission outcomes require
inspection and never trigger an automatic duplicate. Fresh exact job/name
`DONE` and structured terminal results control handoffs; aged-out records need
the exact saved allocated execution/result hashes.

Release directory `source-142-epoch36-guarded-embedding-long-attempt001` and
run/control namespace `source142-epoch36-guarded-embedding-long-attempt001` are
distinct exact allowlisted names. Collection returns bounded recent metadata
under `files`/`stage_results`, a full control-file hash inventory and the latest
up-to-eight scheduler job IDs/output under `scheduler`;the collector does not
interpret missing scheduler records as failures or passes.

## CPU passed; GPU fit queued, 28 October 2026 11:29 CET

Immutable source142 attempt001 was staged with621 exact members. Archive SHA256
`d2ac85ad644a94f3b5967537f1c777bb7318faf2a8507038888bc32728a0d805`,
manifest SHA256
`89dd0f43752a417cb4a4ab0658a4fba294eba72eb65fd5899036a42d8207c4f2`,
plan SHA256
`35adaf248e9011ce81a5bcd165d1897e97901864695a10d56fad70eebe63f70e`.
CPU **29839742 DONE**, callback **29839743 DONE**. The full allocated stage
finished in71.155s: **450 passed, zero failures/errors/skips**. The original
VAE state digest, bitwise default/captured decode parity, all three exact
parent starts/masks/records and the compatible-MKL policy passed. Starts.pt
SHA256 remains
`832592de7183b51ac22542ead558bc453006d3d349d942e2933dd1a52fb40ebd`.

The authenticated callback submitted GPU fit **29839755**, exclusive80GB A100,
four slots,4h wall/3h50 internal, and serial callback **29839756**. At the11:29
snapshot both are **PEND**; four CPU slots were reserved on `n-62-18-12`, with
new scheduling reported. A reservation is not execution. There is no fit,
calibration, native-pilot pass or latent update yet and no reliable start ETA.
No source140 continuation or source141 recovery jobs/settings/releases changed.

[Submission summary](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/submission-summary001.json),
[exact status](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/status-1793183344713502400.json),
[download integrity](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/cpu-verified/download-integrity.json),
and [allocated pytest log](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/cpu-verified/cpu-tests.log)
retain raw evidence. Local AST/JSON/hash checks are separately marked as
non-numerical in `outputs/implementation-checks/source142-long-run-static001.json`.
The enlarged fit, calibration, all fresh gates and pilot remain prerequisites
for serial30,000-iteration execution. CPU readiness is not a native pass or
chemical-improvement result.

## Fit worker passed; pilot handoff stopped, collected28 October2026 evening CET

The saved allocated fit execution for **29839755** reports return0/completed,
with exact result SHA
`d14ef4d6adc31939e13bd92ae06d3af83db64cc91fbc57aeb5c7729414492327`.
The worker completed at16:39:52CET; its full numerical stage took952.792s.
All5,000 mapping updates completed, selecting step4,800. On the **same new dev
set**, the old warm-start mapping's feature loss0.05108844 fell to0.01481416,
a71.0% reduction. New audit loss0.01519852;1,605/1,653 audit rows (97.10%)
satisfied all10 fixed dev-q99 caps jointly. The audit has635 distinct decoded
identities and correlated descendants. Neutral-only charge coverage and absent
P/I remain limitations. These are feature-fit results, not useful-gradient,
chemical-improvement or successful latent-optimization evidence.

New data accepted16,329/17,408 attempts:13,030 train,1,646 dev,1,653 audit.
Rejections:741 sanitization,243 disconnected,95 alignment. The new sampling
excluded640 historical roots,720 groups and786 decoded identities before
sampling; no historical training replay. Original VAE/native bank/old bridge
remained unchanged. New checkpoint SHA
`c8d86551849c7f7e1de5218b7121cbecacb7f9dbc9ee11a5aea5972187a81551`;
calibration SHA
`1c436618bc6dc26dfb049908b11477afc5f2e7b291132db4577a99631f0c69f8`.
Keep these active inputs protected from cleanup.

**Pilot was not submitted.** Callback29839756 ran16:39:53–16:39:57CET,
then EXIT1 at `require_done`: `Predecessor allocation is not exact-name DONE`.
No advance-after-fit/pilot submission intent was created. The callback did not
save its raw bjobs response, so its exact cause remains undetermined; do not
assert a proven scheduler timing/name-formatting bug. The worker's saved
completion does not independently establish the outer LSF terminal state.

A guarded invocation of the **original sealed** `advance --after fit` verified
absence of all transition/pilot intents, then stopped before any submission on
`Predecessor scheduler status unavailable`. No source/config/threshold was
changed or waived. Subsequent bounded read-only attempts to obtain the GPU's
terminal LSF logs and exact formatted bjobs response failed SSH timeout/reset
before returning remote stdout. These access failures do not establish a GPU
failure. No new jobs, pilot gates, optimization updates or source140/source141
changes occurred in this status check. Do not claim the automatic long-run
chain is currently advancing.

[Collected summary](../outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/status-summary002.json)
binds the verified fit artifact download, saved callback failure, unsuccessful
handoff invocation and access receipts. Preserve all failed records. Before
resuming, authenticate the original allocation/fit completion under unchanged
scientific gates and resolve the scheduler interface using actual evidence;
do not repeat the successful mapping fit merely to bypass the handoff.

## Handoff repaired; pilot resubmitted,28 October2026 20:58CET

The user authorized the saved-fit handoff repair and conditional10,000/case
continuation. Source143 control-only attempt002 passed85 allocated fixtures
(zero failures/errors/skips) and genuine original-package pilot CLI dispatch
preflight. It handles the observed zero-returncode aged-out LSF JSON using
independent exact-job successful terminal and execution/result proofs, while
preserving original source142 numerical bytes, trained mapping, calibration
and scientific checks. The good mapping fit was not repeated. CPU29845008
DONE; readiness v2 SHA2470071046d191334a8e89bc1de63c203ae8614609c399d83d9f68736de07674.
Pilot29845020/callback29845021 were submitted and are PEND in the20:58 snapshot;
pilot awaits affinity units, no native/pilot/optimization result yet. Future
callbacks use the authenticated repaired controller; full serial runs remain
conditional on every original check. Source140 VAE/source141 recovery unchanged.
Failed143001's75 fixture passes remain unpromoted because its actual preflight
incorrectly forbade eager CPU Torch imports; immutable logs/XML and the later
CPU-only/CUDA-free v2 correction are preserved.
[Exact repair, limits and receipts](82_embedding_handoff_recovery.md).
