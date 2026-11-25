# Gated benchmarks of failed-repair guidance and deeper search

**Authorized deterministic recovery, 19 October 2026:** the narrow scan and
origin-authenticated retry controller are implemented and syntax-reviewed,
with numerical validation/sealing/submission pending.
[Docs65](65_deterministic_repair_event_recovery.md) declares mandatory strict
CUDA parity and the unchanged same-parent matched rerun/adoption constraints.
The failed source123 outcome below remains immutable.

## Native suite completed with software failures; promotion stopped, 19 October 2026 09:00 DTU-local

GPU **29753378 DONE** (365s LSF wall, ending at 08:46:52) and callback
**29753379 DONE** completed the comparison/selection controller. This does
not mean every candidate passed. The benchmark-only `legacy3` control passed
all 18 original native and 11 supplementary gates, projecting
37,416.2878s/epoch (10.394h) from mean update timing and 40,658.1226s (11.294h)
conservatively. These are disposable-update projections, not measured epochs
or a passing new objective.

Both [fallback readiness](../outputs/hpc-results/source123-native-outcome-001/fallback-readiness.json)
and [bond6 readiness](../outputs/hpc-results/source123-native-outcome-001/bond6-readiness.json)
failed before completing their native panel/update benchmark with
`RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation`
under `torch.use_deterministic_algorithms(True)`. Preserve determinism and the
failed records. The software execution error leaves new-objective native
coverage/gradient/retention/runtime checks incomplete; it is not a measured
failure of chemistry efficacy or a gradient-check discrepancy.

[Selection](../outputs/hpc-results/source123-native-outcome-001/selection.json)
has `winner=null`, reason `no_passing_new_objective`, and
[promotion](../outputs/hpc-results/source123-native-outcome-001/promotion.json)
is `stopped_no_passing_new_objective`. **No new-objective training is
submitted and no reliable continuation ETA is established.** The
[legacy readiness](../outputs/hpc-results/source123-native-outcome-001/legacy3-readiness.json),
[original benchmark gates](../outputs/hpc-results/source123-native-outcome-001/legacy3-benchmark_gate.json)
and saved native/callback stdout/stderr retain the completed and failed
stages without replacing earlier fixture/RUN observations below.
[Outcome verification](../outputs/hpc-results/source123-native-outcome-001/local-verification.json)
passed saved JSON/hash/log checks. The [native traceback](../outputs/hpc-results/source123-native-outcome-001/native.out)
identifies sealed source123 `repair_failure_guidance.py:261`, `_log_any_present`
calling CUDA `cumsum`, reached from `component_cut_risk` at line 383 during
`native_panel`. No source123 rewrite or local model/numerical validation
occurred; a later fix/rerun needs its own immutable source and actual receipts.

## Recovery passed; native GPU running, 19 October 2026 08:42 DTU-local

Additive metadata **CPU 29753375 DONE: 16 passed, zero failures/errors/skips
in 1.051s**. Binder 29753376 authenticated epoch 36 and bound all three native
plans to checkpoint SHA256
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
It submitted **native GPU 29753378 and after-native 29753379**, both observed
PEND at 08:40. At **08:42:26**, recovery CPU and binder were DONE,
**GPU 29753378 RUN** from 08:40:47 on `n-62-18-10`, and after-native 29753379
PEND on its dependency. The GPU requests one 80 GB device/16 CPU slots/six
hours. Initial CUDA/frozen-CPU-reference fixtures passed **10 tests**, with
26 deselected, in 11.49s. The saved [early log](../outputs/hpc-results/source123-retirement-recovery-002/native-early.out)
shows `legacy3` benchmark run `20261019T064213Z_chemistry-benchmark_4d6f9cb7`
started; no candidate has completed. Native coverage/gradient/retention/runtime
gates remain pending. [Scheduler proof](../outputs/hpc-results/source123-retirement-recovery-002/scheduler-status.json)
and [verified recovery](../outputs/hpc-results/source123-retirement-recovery-002/local-verification.json),
[starting checkpoint](../outputs/hpc-results/source123-retirement-recovery-002/starting-checkpoint.json),
[bound plans](../outputs/hpc-results/source123-retirement-recovery-002/bound-plans.json)
and [native receipt](../outputs/hpc-results/source123-retirement-recovery-002/native-receipt.json)
retain the actual source/checkpoint/submission identities and original binder
error logs. Attempt004 scientific settings and every native gate are unchanged.
No native outcome or new-objective training is established; fresh quota can
still block promotion.

Local verification inspected saved JSON/XML/hashes only. Its first assumption
that a saved command was a string rather than an argv list was corrected
before metadata verification passed, without local numerical tests.
For runtime reporting, the actual saved epoch row is 34,835.945s and LSF wall
34,897s (about 9.68h). The source completion's 41,509.414s `epoch_seconds`
value is the earlier projection and must not be presented as measured epoch
runtime.

## Completed source121 parent; second metadata recovery submitted, 19 October 2026

Source121 segment **29749078 completed at 05:34:17 DTU-local**, persisting
epoch 36 after 1,688,539 real/3,377,078 prior draws in 34,835.945s (9.68h),
with all ten epoch health checks true. Its checkpoint SHA256 is
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
The fixed 1,024-prior panel has 641 valid and 641 unique valid graphs
(62.5977%), expected size 22.33288 and sulfur frequency 0.015069966;
reconstruction is 256/256 exact and valid. These saved panel/epoch results
establish the completed source121 endpoint, not passed source123 native gates
or a causal benefit of the unadopted new loss.

The original binder **29752970 exited at 05:34:23** on callback-confirmation
metadata. The [08:37:48 DTU-local saved snapshot](../outputs/hpc-results/status-20261019-source123/snapshot.json)
retains actual completion, checkpoint/run/metric identities and scheduler
logs. Fresh `bjobs` queries returned code 0, empty stdout and exact
`Job <29749079> is not found` stderr for the aged-out callback; the original
confirmation helper did not accept that interface. Preserve this failure and
the earlier uncertain retirement/confirmation records.

Separate additive metadata helper
[source123-confirm-retirement-v002.py](../outputs/hpc-submissions/source123-confirm-retirement-v002.py)
SHA256 `083b9135133e9279ebbf30c2cf678de5a4c8406f2d1f2f50fa82ef325477af3c`
and [preparation](../outputs/hpc-submissions/source123-retirement-v002-preparation.json)
retain unchanged attempt004 scientific source, loss/configuration and every
healthy-handoff/native gate. Its fixture SHA256 is
`fa7b31c0fdcf2aa5daede5e52bf7b66d0d8c8287aff59beef5f379d02c8325ba`.
Actual **recovery CPU 29753375 and binder 29753376 are submitted**, not passed,
as recorded in [queue output](../outputs/hpc-submissions/source123-retirement-v002-queue-output.json).
The binder depends only on `done(29753375)` because the old segment has aged
out of the scheduler; the unchanged completed healthy-handoff authentication
still runs inside the binder. No native GPU or new-objective training is
submitted. Official quota in the snapshot is 92.34/100 GiB used, 7.66 GiB free;
capacity is not reserved and fresh quota may block promotion. No waiver applies.

## Authorization and actual status, 18 October 2026

The user authorized allocated DTU HPC benchmarks of the separate
[failed-repair prototype](63_failed_repair_guidance.md), including gradients,
coverage, sulfur/composition effects and runtime, followed by a separate
continuation only if the declared gates pass. This does not change the running
source121 continuation, its sealed objective or any historical Boltz run.
The sealed **attempt004 CPU 29752937 is DONE: 790 passed, zero failures/errors
and seven CUDA-only skips in 131.06 seconds**; the original CPU repair profile
also passed. It changes only the unsupported-charge test fixture from
attempt003; production scientific bytes are identical. These are allocated
CPU engineering/profile results, not native model coverage/runtime validation.
No native GPU benchmark or new-objective training is submitted. Source121
segment 29749078 remains RUN; binder **29752970 PEND** waits on
`done(29752969) && ended(29749078)` and must still authenticate its healthy
completed boundary.
The policy is `chemistry-failure-guidance-v1`, with separately authenticated
`failed-repair-native-readiness-v1` evidence; implementation is not a gate pass.

The after-CPU controller **29752938 exited on metadata only**: `bkill` of
pending callback 29749079 was acknowledged, but an immediate `bjobs` view
still said PEND, so its original retirement receipt remains sealed as
uncertain, SHA256
`31968fc9c1ef8bc5904b84a0280bb83501faf92486644ab9ebe03167c11afa23`.
A fresh scheduler view said EXIT. Separate external metadata CPU **29752969
DONE, 11 passed/no skips** authenticated unchanged attempt004 source/CPU
proof, that fresh EXIT and absence of a source successor. Its
[confirmation](../outputs/hpc-results/source123-suite-attempt004/retirement-confirmation.json)
adds proof without rewriting the uncertain receipt. The additive
[helper](../outputs/hpc-submissions/source123-confirm-retirement.py), SHA256
`0dc3549d2b2d7b9a090cf32f55004fea42a02f865fab13e3415800eb0c41ed38`,
overrides only that retirement-confirmation boundary; the sealed healthy
handoff, three native candidates, gates and adoption controller are unchanged.
[Local verification](../outputs/hpc-results/source123-suite-attempt004/local-verification.json)
checks saved JSON/XML/hashes only, with no local numerical test.

The [attempt004 preparation](../outputs/hpc-submissions/source123-repair-suite-attempt004-preparation.json)
and [suite](../outputs/hpc-submissions/source123-repair-suite-attempt004.json)
bind 553 authenticated staged members, archive SHA256
`0cfa002071c75816c36c2ab978289b9e66d944e448ebfcd836d6542160617260`,
manifest `8b6e5d0a9bf806ffb22dedf80cc99b7812e778187ff171f7f11a96bf84d33854`
and suite `9c390eb914875687f029783a2ccbf4dbbcf644c53773133de82b500ce101ae2f`.
HPC release is `/work3/s234818/latentmol/releases/source-123-failed-repair-suite-attempt004`;
campaign is `/work3/s234818/latentmol/runs/source123-failed-repair-suite-attempt004`.

### Preserved full-suite attempt003 failure

CPU **29752926 failed with 789 passed, one failed and seven CUDA-only skips**
in 135.71 seconds. The same unsupported-charge fixture's triple-bonded pair
plus isolated third F(-2) formed two components; one bridging edge repaired
that graph within its one-edit budget because the native environment permits
these F(-2) valences. The fixture's censoring assumption was wrong. The retry
uses three isolated F(-2) atoms, requiring at least two added bonds for
connectivity, so a one-edit search remains censored. Production unsupported
rules and all scientific gates remain unchanged. The after-CPU controller
29752927 is expected to stop at the failed CPU proof; no successful retirement
or downstream stage is claimed for it.

The earlier [full-suite preparation](../outputs/hpc-submissions/source123-repair-suite-attempt003-preparation.json)
and [suite](../outputs/hpc-submissions/source123-repair-suite-attempt003.json)
bind the source121 base, declared overlays, unchanged source098 algorithm
identity, source121 segment/callback and data/reference identities. Staging
authenticated 553 members. The full-suite archive SHA256 is
`23da291204a96d750c0f9d5969ba7c191e6af96475638d90992672919f4e8a44`,
manifest `ec7c2a7601926536a8c46a4ce6e4709107b4da49784801cd3f403d6b31c153b2`
and suite `891feafa6111244e94f241159ada1c0d5c261d42e410ef2251f2f626d3c94839`.
HPC release is `/work3/s234818/latentmol/releases/source-123-failed-repair-suite-attempt003`;
campaign is `/work3/s234818/latentmol/runs/source123-failed-repair-suite-attempt003`.
The preparation's packaged status records an earlier stage than the actual
submission above.

Attempt002 was packaged but withheld before submission after review found
that retirement reuse needed stricter authentication. Attempt003 checks the
existing retirement receipt and absence of source successor/controller
actions on every bind, preserves honest stopped-promotion outcomes and
reauthenticates all 13 parity observations and eight timed updates in native
readiness. An initial local attempt003 packaging command also used an
incorrect base121 filename and stopped with `FileNotFoundError` before
writing an archive; that metadata-only error was corrected before sealing.
Neither event was a scientific execution or an HPC submission.

### Preserved early CPU failure

Early CPU fixture job **29752898 exited with 53 passed and one failed test**
in 48.85 seconds. Its unsupported-charge fixture incorrectly assumed that
two triple-bonded F(-2) atoms would be natively invalid; native conversion
treated that graph as already valid, so no failure row was prepared. The
initial correction added a disconnected third F(-2), retaining production
unsupported-rule behavior, but attempt003 above showed that one bridge still
repaired it. Both failed runs remain historical evidence. This is a fixture failure, not a passed
native benchmark or an adopted training loss.

The original job requested two CPU slots, 8 GB and
30 minutes, CUDA hidden and Hugging Face offline. It runs the new
`test_repair_failure_guidance.py` and legacy `test_atom_preserving_chemistry.py`
fixtures, not a native-model coverage/runtime benchmark or training adoption.
The authenticated unit release contains 542 staged members, based on
source121 plus three new files:

| CPU release binding | Identity |
| --- | --- |
| Preparation | [source123-repair-unit-preparation.json](../outputs/hpc-submissions/source123-repair-unit-preparation.json) |
| Preparation SHA256 | `2c6e58cce59d20302407274b2f0692b0d203aa22af36e21460724c7046703078` |
| Source archive SHA256 | `b8fbd171e18a574f5e2389fd0a08f87e65ef97e77490194055c684999b93fabd` |
| Source manifest SHA256 | `79c25b9b4a30b20a7f8b75fa17af7ede9a3e49e4b1445edfdb2c5406452db2f2` |
| HPC control | `/work3/s234818/latentmol/submissions/source123-failed-repair-unit-attempt001` |
| HPC campaign | `/work3/s234818/latentmol/runs/source123-failed-repair-unit-attempt001` |

This release includes the optional preparation-map interface in
`repair_failure_guidance.py`, SHA256
`373f1f5afc9980e01b94960f7ac7230edf94fdbfb2489bf0bf41d3a855bfa7cc`;
the preparation records all three exact file hashes. Its packaged status field
describes packaging time; the actual submission above is a later event.

A fresh parent-coordinated metadata read used
`bjobs -a 29749078 29749079 29750141 29750142` and `getquota_work3.sh`.
Source121 job 29749078 remained **RUN**, started at 19:52 DTU-local, with
21:07h remaining in that read; callback 29749079 and evaluation jobs
29750141/29750142 were **PEND**. Official work3 usage was 90.20/100 GiB.
These are read-only observations, not a completed segment, native gate pass
or quota approval for a future submission. No exact timestamp or saved
snapshot path is asserted for this read.

## Staging and checkpoint boundary

CPU fixtures and original repair profile have passed. The original pending
source121 callback is now confirmed retired by the separate proof above;
retain both the uncertain original receipt and additive confirmation.
Native checkpoint benchmarks wait until **source121 segment 001** finishes
with scheduler DONE and its persisted completion,
checkpoint, optimizer/RNG state, health and source/data identities authenticate.
Use that latest completed healthy model as the explicitly pinned benchmark
parent; a partial current checkpoint is not an adoption boundary. Preserve
every earlier checkpoint, including original epoch 89 and the source098
epoch 35 parent.

Source121's callback can submit another serial segment. Authenticate its
actual status before retirement; a running, unknown or already advanced
campaign is not permission to bypass the boundary or cancel active training.
Do not cancel an in-progress epoch or silently substitute a later parent into
this segment-001-bound plan. Any changed handoff requires a new explicit
immutable plan and authentication. No concurrent training segments are
permitted.

Keep the original frozen epoch 35 reference separate from the later trainable
parent. Preserve the original repair ramp anchored at epoch 35; new opt-in
term coefficients are declared at full strength, with no new ramp. Carry the
authenticated AdamW state, native RNG state and existing fresh-prior RNG
state into an eligible continuation. Do not reseed the prior stream or use
disposable benchmark updates as a training checkpoint.

## Preserved scientific and execution contract

The selected [source121 plan](../outputs/hpc-results/source121a002-first-suite-004/submissions/source121a002-chemistry-16core-24h-attempt002/plans/r2_async_w12_m128_h24.json)
provides the inherited contract: joint encoder/decoder, seed 17, FP32,
256 real/512 fresh-prior points (1:2), real/prior repair branch means
0.25/0.75, prior-mean occupancy anchor 1, and the original population-JS
mixture weighting (prior-count factor 1.5 at ratio 2). Preserve reconstruction,
KL, element-JS 0.1, charge-JS 0.05 and composition-prior weight 1.5.
The original outer repair weight is 0.5 with its original three-epoch ramp.
Preserve original data/split identities and held-out protections.

The plan uses 12 repair workers/four main Torch threads, microbatch 128 and
16 CPU slots; any execution change needs an explicit benchmark configuration.
All allocations remain at most 24 hours, with the existing 23-hour internal
segment limit and at most 100 **total**, not additional, epochs. New native
runtime evidence must include additional preparation/search, native rechecks,
collation, synchronization and optimizer work. The old selected epoch
projection does not establish fit for the new objective. Retain complete-epoch
fit, memory, native health, retention, identity and fresh official-quota gates.

## Sealed matched benchmark grid; allocated outcomes pending

Each candidate restores the same pinned checkpoint, AdamW/RNG states and
declared real/prior examples. Search and chemistry checks remain detached.
Freeze a prepared teacher snapshot within each loss/gradient comparison;
record its IDs, raw/probability identity, original targets and coverage binding.
Do not compare rows prepared from different snapshots as a matched gradient
test. The sealed suite binds these settings and resources; the completed
source121 handoff must supply its authenticated starting-checkpoint binding
before native plans can be created or submitted.

| Candidate ID | Additional search | Scope |
| --- | --- | --- |
| `legacy3` | Original three-edit bond-only teacher | Matched benchmark control only; cannot be promoted |
| `fallback` | None beyond original teacher | New failure guidance; promotion eligible only after all gates |
| `bond6` | Separate six-edit bond-only pass | New failure guidance; promotion eligible only after all gates |

The additional loss declares `extra_valid_weight=1`, `partial_weight=1`, and
`local_bond_weight=component_cut_weight=nonempty_weight=0.1`, using
`repair-failure-guidance-v1`. Complete valid witnesses replace fallback on
their rows. Production atom changes are off. These terms retain all-attempt
normalization within each branch, 0.25/0.75 real/prior branch means and the
original outer repair coefficient/ramp; there is no new coefficient ramp.

Each candidate measures eight complete disposable updates, 256 real/512 prior
points per update and microbatch 128. The fixed native prior panel contains
1,024 draws with independent evaluation seed `123904017`, processed in
128-row chunks using the original three-edit teacher. Its evaluation draws do
not advance the fresh-prior training RNG. Matched initial graph/probability
snapshot-chain hashes and original teacher status counts must agree across
candidates.

Up to 64 identical failed panel rows additionally receive matched bond-depth-6,
bond-depth-12 and atom-enabled-depth-6 search diagnostics. All three diagnostic
passes are excluded from promotion and reported separately from synchronized
production-update timing. Atom-enabled diagnostics use the explicit atom-edit
bound/penalty, retain charges and occupied slots, and report actual atom edits
and sulfur counts before/after retained targets.

Prototype defaults are beam 4, at most 32 checks per depth, and six additional
search depths; this is a separate bounded pass from the raw graph, not a
certified nine-edit search. Optional atom changes retain formal charges and
occupied-slot identity. The detached edit-ranking probability floor is
`1e-12`; differentiable NLL and risk terms remain unfloored. The planned native
suite requests one 80 GB A100, 16 CPU slots and six hours only after the healthy
handoff. Any selected continuation uses separate serial allocations of at most
24 hours and still requires fresh quota approval.

## Evidence required before adoption

Allocated CPU fixtures must first establish original valid-target NLL parity,
all-attempt normalization, complete-witness replacement of fallback, partial
versus valid routing, provenance rejection, finite gradients and the true
greater-than-three-edit case. A fixture pass alone is not native validation.
Six independent component directional finite-difference fixtures are included
in CPU validation. Native readiness adds a combined extra-guidance directional
check on actual decoder logits with the entire preparation frozen at both
perturbations, an actual decoder-parameter VJP and production extra-guidance
decoder-gradient norms. These are tests to run, not results established here.

Native evidence must retain all scheduled and eligible denominators for real
and prior rows, with accepted, already-valid, censored, unsupported and error
statuses separately reported. Complete valid witnesses, invalid partial
targets and dense fallback signals are different outcomes. Record added search
checks/timing, extra valid coverage, partial routing, unsupported modes and
per-term/branch gradient scope and norms, including zero/nonfinite cases.
Unknown errors must remain errors. No universal nonzero-gradient or validity
claim follows from the local/cut/nonempty surrogate.

Measure sulfur and other element/charge/occupancy distributions with explicit
denominators on matched raw outputs and on valid witnesses separately. Include
atom substitutions where enabled; a diagnostic witness gain is not proof that
atom changes remove sulfur bias or preserve chemistry. Check reconstruction,
KL/composition/occupancy retention and native supported conversion/validity
under declared tolerances. The [saved historical coverage](63_failed_repair_guidance.md#current-zero-target-behavior-and-saved-coverage)
is selected minibatch evidence, not an epoch-population baseline or a native
result for the new method.

The original **18 native gates remain mandatory**, including retention against
the original frozen reference. The supplementary receipt adds 11 required
checks for all three benchmark candidates: finite population statistics,
expected size within 0.9–1.1 of the current-start value, nonempty fraction no
more than 0.02 lower, sulfur frequency at most `before * 1.5 + 0.005`, element
and charge JS each at most `before + 0.02`, exact and valid reconstruction each
at least `before - 0.02`, raw-prior validity at least `before - 0.02`, all
original native gates passed, and a conservative complete-epoch fit. These
additional retention comparisons use the current starting chemistry model;
they do not replace the original-reference comparisons.

For the conservative fit, let `T` be the larger of the original mean-based
epoch projection and the slowest complete update extrapolated over the real
training population plus 600 seconds. Require `1.5 * T + 1800 < 82800` seconds.
This retains setup/evaluation margins and does not equate eight updates with
a measured full epoch.

Both new-objective candidates additionally require all seven guidance checks:
at least one original failed row observed; at least 90% of those rows with
additional logit-gradient norm strictly greater than `1e-10`; reduced
zero-guidance count (at least one failed row with such a signal); a nonzero
actual extra-guidance decoder VJP; a passed frozen-target directional check;
zero complete-target atom edits; and a production update with extra-guidance
decoder-gradient norm above `1e-10`. Panel row norms undo their chunk's
all-attempt denominator for this threshold. The finite-difference check uses
float64 logits, epsilon `1e-3`, rtol `1e-4`, atol `1e-8` and a nontrivial analytic
derivative above `1e-12`. Passing these checks would demonstrate measured
signal on this panel, not universal failed-row coverage or chemical validity.

Selection first requires the matched control to pass and a complete ordered
comparison. Among passing `fallback`/`bond6`, prefer more complete extra valid
witnesses, then more failed rows with nonzero additional gradient, then lower
conservative selection epoch time; candidate ID breaks any remaining tie.
Partial witnesses and dense risks never count as extra valid coverage.

Promote only an eligible candidate with all declared native, gradient,
coverage, retention, runtime, resource and quota gates passed. If none passes,
record the failed/partial outcomes and do not adopt. Adoption requires a new
immutable source/configuration and separate objective/run identity, bound to
the authenticated completed parent and all continuation state. Preserve the
old campaign and every benchmark/failure; deliberate handoff is not model
instability or an epoch-100 baseline control. Final evaluation remains gated
by an authenticated absolute epoch-100 endpoint, and any resulting decoder
must be frozen before latent optimization.

For an adopted new-objective continuation, each completed epoch also applies
nine retention checks against the fixed healthy handoff model, including
raw-prior validity, alongside the preserved original-reference health checks.
This fixed handoff comparison does not reset at each epoch or replace the
original epoch 35 reference. These implemented safeguards have no passed
native outcome at this CPU-passed, healthy-parent-pending status.
