# Deterministic event-probability recovery and matched native retry

## Native comparison completed; no candidate promoted, 19 October 2026 13:13 DTU-local

The [fresh saved snapshot](../outputs/hpc-results/source124-deterministic-events-attempt002/status-news001.json)
at 11:13:53.530498 UTC confirms **A100 29753784 DONE**, 1,111s (18m31s)
GPU allocation, and **after-native 29753785 DONE**. All 104 deterministic
event tests and 13 native parity prechecks passed. The software execution
failure is resolved, and both new objectives passed their observed native
extra-decoder VJP and frozen-target float64 finite-difference checks with no
atom-edit shortcut. Each produced nonzero additional gradients for **22/22
originally failed panel rows**; this denominator is the observed failed subset,
not every molecular identity or an epoch-population coverage claim.

Selection still has `winner=null`, reason `no_passing_new_objective`, and
promotion is `stopped_no_passing_new_objective`. **No new training is
submitted.** These are completed gate outcomes, not a software crash or a
waiver opportunity. The benchmark-only legacy control passed all 18 original
plus 11 supplementary gates. Exact new-candidate failures and eight-update
observations are:

| Candidate | Original native gates | Supplementary outcome | Valid raw prior after 8 updates (256 fixed draws) | Mean epoch projection | Conservative epoch projection |
| --- | --- | --- | ---: | ---: | ---: |
| `legacy3` | 18/18 passed | 11/11 passed; control only | 160/256 (62.5000%) | 10.2310h | 11.10235h |
| `fallback` | 18/18 passed | Failed `raw_prior_validity_retention` and `conservative_complete_epoch_fits` | 131/256 (51.1719%) | 13.4686h | 15.39681h |
| `bond6` | Failed only `complete_epoch_fits` (other 17 passed) | Not finalized/reached after original failure | 172/256 (67.1875%) | 18.9609h | Not computed; not zero |

All candidates start that same fixed 256-prior panel at **149 valid
(58.2031%)**. The fallback drop is 18/256 (7.03125 percentage points), exceeding
the allowed 2-point reduction. Both new candidates retain 256/256 exact and
valid reconstruction after their disposable updates; legacy has 255/256 exact
and 256/256 valid. These short, matched observations do not establish a
longer-run gain, stable chemistry efficacy or a completed epoch.
The cause of the validity decline is not established; these records do not
attribute it definitively to atom-identity or bond behavior.

The runtime requirement is `1.5 * T + 1800 < 82800` seconds, equivalently
`T < 15h` for the protected epoch estimate. Fallback's slower-update-based
conservative estimate fails despite its lower mean projection; bond6 already
fails the original mean-based complete-epoch-fit gate. These are safety
headroom failures within the 23h internal budget, not measured out-of-memory
events or actual full-epoch runtimes. No bond6 supplementary estimate was
emitted after the original gate stopped that path.

Bond6 found **12 additional complete valid witnesses and 10 invalid partial
witnesses** among the 22 original failed panel rows. Partial targets remain
distinct from valid witnesses. The matched depth-12 diagnostic produced the
same 12/10 coverage with twice the candidate checks; this record does not
demonstrate a coverage benefit from further depth. Production atom changes
remain off. Latest official quota is **88.70/100 GiB used, 11.30 GiB free**;
space is not the failed gate here and future promotion would still need a
fresh quota check. This status request authorized no new code, jobs or rerun.

## Verified cold offload completed; native readiness remains queued, 19 October 2026 08:10 UTC

All **103 cold intermediate checkpoints/4,281,629,416 logical bytes** were
transferred successfully and SHA256-verified locally before remote removal.
The reviewed helper then rechecked all remote hashes/stats and job protections
and removed exactly 103 files/4,281,656,832 allocated bytes (about 3.9876 GiB),
returning `completed_verified_local_offload`. Complete backup bytes remain in
[`payloads`](../outputs/hpc-results/source124-cold-offload-001/payloads), with
[restoration map](../outputs/hpc-results/source124-cold-offload-001/restoration-map.json),
[local proof](../outputs/hpc-results/source124-cold-offload-001/local-proof.json)
SHA256 `13bb07ed86a7e0adbeb89f1dfe3d9565c7ce7aef0d1688d7ddb6cfa84ad976c1`
and [successful operation output](../outputs/hpc-results/source124-cold-offload-001/remote-operation-output.json).
Protected epoch 35/36/89 hashes are unchanged before/after; best/last/newest-two,
current ancestry and inputs remain protected. See [docs50](50_hpc_space_cleanup.md)
for exact restore/protection details.

Official free quota is now **11.43 GiB** (88.57/100 GiB used), above the
recorded 10.0625 GiB threshold. This is not reserved capacity or training
approval; promotion still requires an independent fresh quota check.
At 08:10:44 UTC, **A100 29753784/callback 29753785 remain PEND**
([scheduler snapshot](../outputs/hpc-results/source124-deterministic-events-attempt002/status-gpu001.json)).
All native readiness checks remain mandatory and no training is submitted.

Independent [remote evidence verification](../outputs/hpc-results/source124-cold-offload-001/remote-evidence-001/local-verification.json)
now passed all 15 metadata checks, including six exact remote files,
103 ordered intents/successful journal rows and fresh absence of all 103
removed paths; protected hashes and 11.43 GiB free were rechecked.
[Final 08:12:19 UTC status](../outputs/hpc-results/source124-deterministic-events-attempt002/status-final001.json)
confirms CPU 29753752/after-CPU 29753753 DONE, native A100 29753784/callback
29753785 PEND and no training submission. Fixture success and recovered
space do not replace complete native readiness or fresh promotion quota.

## Attempt002 targeted V100 parity passed; full native benchmark pending, 19 October 2026

Targeted **V100 job 29753792 DONE: all 104 passed, zero failures/errors/skips
in 7.68s**, comprising 52 CPU/52 CUDA cases. FP32/FP64 fixture loss/gradient
parity, strict determinism and repeatable forward/backward passed at unchanged
tolerances, including the prior 625-edge cut failure. The
[verified saved outcome](../outputs/hpc-results/source124-a002-v100-outcome-001/local-verification.json)
binds 15 exact downloaded files, [JUnit](../outputs/hpc-results/source124-a002-v100-outcome-001/v100-pytest.xml)
and source124 attempt002 identities. This is allocated fixture evidence;
it is not full-model native coverage/gradient/retention/runtime readiness or
chemical efficacy.

**A100 native 29753784/after-native 29753785 remain PEND**, and no new training
is submitted. Every matched native and quota gate remains mandatory before
promotion. The cold transfer's last file is still in progress, with
4,235,200,499 of 4,281,629,416 bytes copied at the reported observation;
complete hash verification and removal receipts remain pending. No remote
deletion is claimed.

## Attempt002 CPU/profile passed; A100 native and V100 fixtures submitted, 19 October 2026

**CPU 29753752 DONE** passed the complete CPU gate and original repair profile;
its pytest prefix was 893 passed/59 CUDA-only skips/zero failures/errors in
122.68s. **After-CPU 29753753 DONE** authenticated the unchanged epoch-36
handoff and retry origin, then queued **A100 native 29753784 and after-native
29753785**, both PEND in the [saved CPU/queue proof](../outputs/hpc-results/source124-deterministic-events-attempt002/status-cpu005.json).
The native suite retains one 80 GB A100/16 CPU slots/six hours, all three
same-parent candidates and every original/supplementary gate. A CPU pass
does not establish strict CUDA parity or native readiness.

Separate targeted **V100 job 29753792 was submitted**, not passed, for the
same 104 deterministic event fixtures, with actual
[queue receipt](../outputs/hpc-submissions/source124-attempt002-v100-events-queue-output.json).
Keep this diagnostic scope separate from the complete A100 native suite.
Tests/tolerances, loss/ratio, parent model/AdamW/native+fresh-prior RNG and
selection/adoption rules remain unchanged. No new training is submitted;
fresh quota still gates any eligible promotion. Cold checkpoint copying
continues without a claimed remote removal.

## Attempt002 sealed; CPU retry submitted, 19 October 2026

The event-precision retry is now sealed as **attempt002**, with **556 staged
members authenticated** and actual **CPU 29753752/after-CPU 29753753 queued**.
This is submission, not a CPU/profile or CUDA pass. Native GPU and training
are not yet submitted. [Preparation](../outputs/hpc-submissions/source124-deterministic-events-attempt002-preparation.json)
and [suite](../outputs/hpc-submissions/source124-deterministic-events-attempt002.json)
bind archive `2e8dc8a84fc29315caed8dc414cf5b05daedad5b73d3131505ad21f0cc67acb6`,
manifest `68ff169844525c6f96b2363567f47bbaad3b23befc4e34d2cfbc5d147d682654`
and suite SHA256 `778f4caa7c3a50a745f1ffe684d2bef06dcd2ccef219f88e36b8a7cfa1d92551`.
Tests, tolerances, loss semantics and every native/adoption gate remain
unchanged. The initial external metadata queue operator failed with
`ModuleNotFoundError: latentmol` before submission because its environment
lacked `PYTHONPATH`; preserve that [failed queue output](../outputs/hpc-submissions/source124-deterministic-events-attempt002-queue-output.json).
Correcting only that operator environment produced the
[actual successful queue output](../outputs/hpc-submissions/source124-deterministic-events-attempt002-queue-retry001-output.json),
without changing sealed scientific bytes. Docs65 remains a workspace
companion; do not rewrite either archive for later documentation updates.

At **08:06:04 UTC**, attempt002 pytest had completed **893 passed, zero
failures/errors and 59 CUDA-only skips in 122.68s**, but the original CPU
repair profile was still RUN and `cpu_gate.json` was not yet saved. The
[saved CPU prefix](../outputs/hpc-results/source124-deterministic-events-attempt002/status-cpu004.json)
retains JUnit/log evidence. Do not label the whole CPU job or profile passed
from this prefix. The skips remain seven original CUDA cases plus 52 event
CUDA cases; mandatory GPU parity has not run. The exact four-file delta and
storage-finisher safety were reviewed, which is static evidence only.

The [verified attempt001 outcome](../outputs/hpc-results/source124-a001-v100-outcome-001/local-verification.json)
now binds 16 downloaded files, the 103-pass/one-failure V100 result and actual
[retirement proof](../outputs/hpc-results/source124-a001-v100-outcome-001/pending-native-retirement.json)
that old native GPU 29753678 and callback 29753679 are both EXIT. Verification
SHA256 is `c579cf9a84444d5727182171de6c6611339295d10cb33d67d691cbd14952c0e0`.
All attempt001/source123 failure records remain preserved. Cold offload
copying last reported 87 files/3,464,646,571 bytes; complete verification/removal
receipts are pending and no remote removal is claimed.

## Targeted V100 parity failed; precision retry preparation, 19 October 2026

Source124 attempt001 targeted V100 **job 29753717 EXIT: 103 passed, one
failed in 8.05s**. The failed case is
`test_actual_failure_loss_event_branches_match_cpu_reference_and_repeat[cuda-maximum_cut-3.0-dtype0]`:
CPU risk `0.0020880699157714844` versus CUDA `0.0020995140075683594`,
absolute difference `1.1444091796875e-5`, outside the unchanged FP32
rtol `5e-5`/atol `5e-6` comparison. The deterministic scan executed, resolving
the earlier unsupported CUDA `cumsum` operation; this attempt still failed
the required numerical parity gate. It is not a complete native benchmark
or evidence of a chemistry effect. [Targeted queue identity](../outputs/hpc-submissions/source124-v100-events-queue-output.json)
retains its separate 104-case-only scope; complete outcome receipts follow
the actual execution record.

Pending attempt001 native GPU 29753678 and callback 29753679 initially
received acknowledged `bkill` requests while PEND, with terminal confirmation
then pending. The later verified retirement proof above confirms both EXIT;
the original requests and final observations remain separate records.
No new training is submitted.

A separate attempt002 preparation now computes empty/component-cut event
normalization from **raw logits cast to FP64 before log-softmax**, then uses
the deterministic event partition. At the maximum 625-edge cut, FP32
categorical normalization and accumulated prefixes can dominate a small
nearly-satisfied event risk. Casting already-rounded FP32 log probabilities
would not recover that lost mass. The FP64 scope is only these event branches;
model parameters, original full-target NLL, partial NLL and local-valence
convolution are unchanged. Gradients continue through the cast to live logits;
there is no probability floor or relaxed tolerance. Current production-file
SHA256 is `34c1a67cb339233312a1317db39a926b55f378c0eb8ed42c38bde86307fdd184`.
Tests remain unchanged. This is implemented preparation, not a new sealed
source identity, submitted retry or numerical pass.

Cold-checkpoint offload copying is in progress for an expanded 103-file
inventory, about 3.9876 GiB; the last reported copy count was 47. Completion,
all-file hash verification and removal receipts remain pending. **No remote
deletion is established**, and all protected checkpoints remain protected.

## Authorized scope and current status, 19 October 2026

The user authorized fixing the deterministic CUDA execution failure, allocated
CPU/GPU parity checks, a matched native rerun, verified quota recovery and
conditional continuation only if every gate passes. The deterministic event
scan and retry-controller changes are implemented and syntax-reviewed.
Source124 attempt001 is now sealed with **555 authenticated staged members**;
**CPU 29753658 and after-CPU 29753659 are DONE at the 09:42:34 DTU-local
read**. The complete CPU gate and original repair profile passed. Pytest
completed **893 passed, zero failures/errors and 59 CUDA-only skips in
137.92s**: seven existing CUDA cases plus 52 new event CUDA cases. The
after-CPU controller authenticated the unchanged epoch-36/retry origin and
submitted **native GPU 29753678/after-native 29753679**, both PEND.
[Saved preflight](../outputs/hpc-results/source124-deterministic-events-attempt001/status-preflight002.json)
retains actual JSON/XML/hash/receipt evidence. The strict-CUDA 104-case module
and matched native gate results remain pending; new training is unsubmitted.
No local numerical/model test has run.
This protocol is not evidence that the fix executes on CUDA or that a
candidate is eligible.

| Sealed source124 attempt001 binding | Identity |
| --- | --- |
| [Preparation](../outputs/hpc-submissions/source124-deterministic-events-attempt001-preparation.json) | Four declared code/test/controller overlays from source123 attempt004 |
| Archive SHA256 | `89db1aaa08e0aa986b41d6de1113ba16dc02c151b981c5987beb50cdb4232590` |
| Manifest SHA256 | `b9a32e545935fcc926453d99e187a778f5f348ec2798216b3768246b3d7293ea` |
| [Suite](../outputs/hpc-submissions/source124-deterministic-events-attempt001.json) SHA256 | `a1350dc8ab576b8114b3abd327f4b0f74b8652aed14222beee2e5a8f860fe509` |
| [Initial queue output](../outputs/hpc-submissions/source124-deterministic-events-attempt001-queue-output.json) | Earlier `submitted_not_passed`: CPU 29753658/after-CPU 29753659 |
| HPC campaign | `/work3/s234818/latentmol/runs/source124-deterministic-events-attempt001` |

The optional packaging entry used `65_deterministic_repair_events.md`, while
this file is `65_deterministic_repair_event_recovery.md`; therefore docs65 is
a workspace companion and **is not included in this sealed archive**. Keep
the sealed release unchanged. Its preparation's packaged status predates the
actual submission recorded above. All source123 failures and immutable
artifacts remain preserved.

Preserve [source123's verified stopped outcome](../outputs/hpc-results/source123-native-outcome-001/local-verification.json),
including the passing legacy control, both failed new objectives, every source
release and all retirement/binder failures. Source123 GPU 29753378 ended at
08:46:52 with 365s LSF wall; no training was promoted. Its trace identifies
`repair_failure_guidance.py:261`, `_log_any_present` CUDA `cumsum`, called by
`component_cut_risk` at line 383. This was a software execution failure before
completed new-objective readiness, not a chemistry-effect or gradient-check
result. The rerun receives a new immutable source/campaign and never rewrites
the failed source123 records.

## Narrow numerical change

[`repair_failure_guidance.py`](../src/latentmol/repair_failure_guidance.py)
replaces CUDA `cumsum` in the first-present event partition with an
out-of-place, fixed-order doubling scan built from additions and disjoint
slices. For event i, its log term remains:

```text
log P(present_i) + sum(log P(absent_j) for j < i)
log P(any present) = logsumexp(all first-present terms)
```

The prefix is exclusive: the final event's absence never contributes. The
scan retains device, dtype, autograd and log-space tails without a probability
floor, CPU roundtrip or O(n²) triangular matrix. Its work is O(n log n).
Addition grouping differs from historical CPU `cumsum`, so cross-implementation
bitwise equality is not claimed. Determinism remains strict:
`torch.use_deterministic_algorithms(True, warn_only=False)`; disabling it or
switching to warning-only mode is not a recovery option. A single-event input
must retain the graph-connected zero absence derivative, while an empty event
set remains an explicit error.

No loss coefficients, search budget, teacher routing, branch means,
normalization, target/provenance identity or chemistry interpretation changes.
The mathematical local/cut/nonempty signals and their limitations remain in
[docs63](63_failed_repair_guidance.md). The native candidate protocol and all
18 original plus supplementary gates remain in [docs64](64_failed_repair_benchmark_protocol.md).

## Allocated parity and regression stages

[`test_deterministic_repair_events.py`](../tests/test_deterministic_repair_events.py)
declares **104 expected parametrized cases**, not an executed pass count.
All CPU references and numerical fixtures must execute in allocated DTU HPC
jobs. The required coverage is:

| Check | Declared scope |
| --- | --- |
| Event value/gradient parity | CPU and CUDA, FP32/FP64, lengths 1/2/3/49/50/625, mixed and confidently absent/present logits |
| Stable tails | Extreme finite logits around ±2000, finite loss/gradient and useful confident-absence signal |
| Exclusive prefix | Independent input derivatives, including zero derivative for the final absence |
| Single/empty event | Connected zero absence derivative for one event; empty set rejected |
| Repeatability | Repeated forward/backward exactly equal on the same device |
| Production loss branches | Empty, disconnected and maximum 25×25 component-cut cases at confidence 3/1000; loss/gradient parity and repeatability |

The fixture tolerances are FP32 rtol `5e-5`, atol `5e-6`; FP64 rtol `2e-11`,
atol `2e-12`, with a tighter exact-form prefix check. These are explicit
floating-point parity tolerances, not chemical or training acceptance limits.
Historical CPU `cumsum` is used only as the detached fixture reference law.

The allocated GPU precheck must set
`LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1`. Missing CUDA then fails rather than
skips the event regressions, while strict determinism stays enabled through
every tested forward/backward. Preserve the existing mandatory CUDA native
resume/frozen-CPU-reference prechecks too. CPU allocation skips do not replace
the mandatory GPU stage. Include the full existing CPU suite, six independent
component finite-difference fixtures and retry-controller metadata fixtures;
any failed prerequisite stops before native candidate execution.

After parity passes, rerun all three matched candidates from the same sealed
epoch-36 parent, not only the previously failed arms. Retain the actual
decoder VJP, production extra-gradient checks and combined native float64
finite differences with the entire preparation frozen. Preserve the 1,024
fixed prior panel, eight disposable updates, diagnostic-only deeper/atom
searches and coverage/gradient/retention/runtime criteria of docs64. Initial
fixture passes are not a native candidate pass; disposable weights never
become training checkpoints.

## Immutable origin and scheduler authentication

The versioned retry controller in
[`failure_guidance_campaign.py`](../scripts/hpc/failure_guidance_campaign.py)
requires `deterministic-repair-events-retry-v1` origin metadata, an independent
new campaign, unchanged protocol/execution/source policy, and the exact old
suite plus complete hash-bound origin artifact allowlist. It authenticates:

- The old starting checkpoint/bound plans and stopped selection/promotion,
  with no winner and reason `no_passing_new_objective`.
- Both exact cumsum execution failures, tied to the old native job, plans,
  archive and manifest; another failure is not this authorized retry.
- The completed old native allocation, original allocated callback-retirement
  EXIT confirmation and completed after-native callback, including actual
  previous LSF terminal evidence and receipt-bound log paths.
- Fresh terminal scheduler observations or exact job-specific aged-out output
  accompanied by separately authenticated persisted terminal proof. A live
  RUN/PEND callback, wrong job, ambiguous output, permission/query error or
  mismatched terminal evidence blocks the retry.
- No source successor/controller action on each bind, the original complete
  healthy source121 handoff, and equality with the original epoch-36 binding.

The authenticated checkpoint remains
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
Carry its model, AdamW, native RNG and existing fresh-prior RNG. Keep the
original frozen epoch-35 reference and repair-ramp lineage, with no new ramp.
Retain 256 real/512 prior draws, 0.25/0.75 repair branch means, composition,
occupancy, reconstruction/KL and all other source123 attempt004 settings.
This is a same-parent software retry, not a silent new chemistry protocol.
[`test_failure_guidance_retry.py`](../tests/test_failure_guidance_retry.py)
contains synthetic metadata fixtures for these boundaries; its existence is
not an executed validation result.

## Quota recovery and conditional continuation

The initial audit identified 87 candidates/about 3.270 GiB; the final
allowlist expanded to 103 cold intermediate checkpoints and **completed**
verified local offload/removal of 4,281,629,416 logical bytes. Exact bytes
remain in [local payloads](../outputs/hpc-results/source124-cold-offload-001/payloads),
with [restore locations and hashes](../outputs/hpc-results/source124-cold-offload-001/restoration-map.json).
The [operation output](../outputs/hpc-results/source124-cold-offload-001/remote-operation-output.json)
records all 103 fresh remote checks/removals and unchanged protected epoch
35/36/89 hashes. All best/last/newest-two checkpoints, original epoch 89 and
source098/121/123/124 ancestry/current inputs remain protected. Full receipts
and recovery details are in [docs50](50_hpc_space_cleanup.md), with
[independent verification](../outputs/hpc-results/source124-cold-offload-001/remote-evidence-001/local-verification.json)
of all 103 journal/absence records and protected hashes.

The completed-offload snapshot was 88.57/100 GiB used, 11.43 GiB free;
the newer 13:13 DTU-local status is **88.70/100 GiB used, 11.30 GiB free**,
above the recorded 10.0625 GiB threshold. These supersede the earlier
92.34/7.66 observation. Capacity is not reserved; authenticate fresh official
quota at every required submission/promotion boundary without lowering the
existing minimum or scientific gates. The actual native comparison above
stopped on scientific/runtime gates, with no training submission.

Only a passing eligible `fallback`/`bond6` may be selected, with complete-valid
coverage, failed-row signal and runtime ordering unchanged. A passing legacy
control alone never promotes the old objective through this retry. If none
passes, record the stopped outcome. Conditional new-objective continuation
uses separate serial allocations of at most 24h/23h internal, no concurrent
training, the same authenticated epoch-36 state and at most 100 total epochs.
Retain all original and fixed-handoff epoch health/identity/quota checks.
Final evaluation still requires an authenticated absolute epoch-100 endpoint;
freeze any resulting decoder before latent optimization. Authorization and
implementation alone establish no passing candidate, queued training,
completed epoch or continuation ETA; actual submissions are recorded above.
