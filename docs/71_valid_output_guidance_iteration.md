# Iterating repair and valid-output guidance from epoch 36

Status, 21 October 2026 19:44 DTU-local: **source133 longer proof FAILED health
at 200/2,500 updates**. Native readiness, the 100-update diagnostic probe and
all eight restoration checks passed first. No adoption or training occurred;
epoch 36 remains the latest completed checkpoint. Current scheduler records
have aged out; the persisted allocation/LSF evidence records terminal exit 1.
This scope concerns the repair-plus-valid-output
guidance just discussed. All numerical work remains on allocated DTU HPC;
local work is editing, static review, packaging and saved-metadata inspection.

The user authorized iterating the new loss to serve its intended purpose:
guide invalid prior outputs toward valid chemistry while retaining valid outputs.
Use a new versioned objective and targeted tests from authenticated epoch 36.
Do not restart source131/source132 or overwrite their failed proof records.
This authorization supersedes the preceding status-only restriction for a new
candidate; it does not turn the preceding candidate into a pass.

## Terminal outcome: alpha 0.1 did not pass longer health

The [17:43:48 UTC saved status](../outputs/hpc-results/source133-valid-self-strength-attempt001/status-news001.json)
records native readiness **passed**, the **100-update probe passed**, all
**eight restoration checks passed**, and the fresh proof **failed at 200/2,500**
updates. Persisted LSF records show GPU **29770577 exited with code 1 at 13:51:34
DTU-local**. Current queries find all three jobs aged out; they are not queued
or still running. The longer-proof failure is raw-prior and strict-teacher
validity retention, with the other **16/18 early-health checks true**, not a
NaN or runtime failure. No new scientific revision, gate waiver, submission or
training was made during this status-only collection.

[Saved panel/generation outcomes](../outputs/hpc-results/source133-valid-self-strength-attempt001/fixed-panel-outcomes-20261021-001.json)
show the same raw/strict counts for the disposable probe and restored fresh
proof through update 100:

| Disposable updates | Raw valid / strict teacher valid, out of 1,024 |
| --- | --- |
| 0 | 641 / 641 |
| 32 | 670 / 670 |
| 100, probe passed and restored | 643 / 643 |
| 200, longer proof failed | 591 / 591 |

The failed endpoint is **57.7148% raw-valid**, down **4.8828 percentage points**
from 62.5977%, exceeding the unchanged two-point limit. Exact and chemically
valid held-out reconstruction stayed **256/256 at every panel**. At 200, the
remaining **433** prior failures were chemical (**275 sanitization / 158
disconnected**) with **zero representation failures**. Longer-proof gate SHA256:
`a71f9eb8b71a8dcf149e47853f8da39fdae19acbfcef83120c121068f79ca0f5`.

Unfiltered four-draw categorical generation (same 1,024 latents, no repair)
was **2,091/4,096 valid initially → 2,251/4,096 at probe 100 → 1,971/4,096
at failed proof 200** (51.0498% → 54.9561% → 48.1201%). Unique-valid identities
were **1,976 → 2,074 → 1,819**. Thus the encouraging 100-update categorical
and unique-valid results did not persist to 200. The alpha-0.01 source132 arm
ended at **616/1,024 raw / 2,000/4,096 categorical valid**, above this alpha-0.1
endpoint; these are single matched disposable trajectories, not replicated
evidence of a general coefficient ordering. This candidate did not fix retention.
Do not describe the 100-update result as pure argmax-only sharpening: categorical
validity and unique-valid yield also improved at that stage.

The full weighted decoder self-gradient norms at updates **1/39/96/100** were
**0.004934/0.004088/0.003326/0.002280**, versus weighted prior-repair norms
**0.007157/0.007312/0.009300/0.008947**. Self gradient dot actual displacement
was negative at update 1 but positive at **39/96/100** (**3.952e-6/9.321e-7/
5.292e-6**), while full-objective dot displacement was negative at all four.
Repair/self cosine was **-0.3495 at 39 / -0.3232 at 100**; component reconstruction
checks all passed (maximum error **9.184e-9**). These observations support
variable directional conflict, not a conclusion that alpha alone is too small.
They are local first-order measurements, not a finite-step causal proof.

Root collected existing status/panel JSON, logs and hashes with
`outputs/hpc-submissions/collect-source133-outcomes-20261021.py` (standard-library,
one read-only SSH collection, no model/search/evaluation execution). Failed
states remain disposable and nonpromotable; original epoch 36 is preserved.

## Actual immutable submission and current CPU state

The [11:15:10 UTC snapshot](../outputs/hpc-results/source133-valid-self-strength-attempt001/status-status007.json)
still has **GPU 29770577 RUN**; the native benchmark run started at **11:14:51
UTC**. The initial `failure_readiness.json` has a failed placeholder status but
no terminal error or completed gate; it is not a native failure outcome.
Native/probe/proof results remain pending.

**Inherited profile-label caveat:** the saved CPU-profile stdout prints an
obsolete exact-1:3 ratio description. Preserve that raw output. The sealed
protocol has `prior_ratio=2`, meaning **1:2**, and the profile uses only **four
real / 32 diagnostic points**; it measures execution parity, not the actual
training-batch ratio. This explanatory metadata note changes no source,
protocol, numerical result or job, and does not trigger resubmission.

The [11:13:52 UTC snapshot](../outputs/hpc-results/source133-valid-self-strength-attempt001/status-status006.json)
records **CPU 29770494 DONE / after-CPU 29770495 DONE**, with full CPU gate and
repair profile **passed**. The callback submitted **GPU 29770577**, now **RUN**
in `gpua100`, 16 CPU slots, started about **13:12 DTU-local**, with a **24-hour**
allocation. Its mandatory CUDA JUnit has **282 passed, zero failures/errors/skips**,
**38.737 seconds**, SHA256
`2eee3d34db2f26170ce0d5e51d39a93d503dcd4ca9825d1cb8dd81b52c7e8b9e`.
No native readiness or benchmark-gate result exists yet; probe/proof completion
is not established. CUDA fixture success is not a scientific candidate pass,
adoption or permission to train. Earlier CPU observations below remain historical.

The [11:11:10 UTC unit-result snapshot](../outputs/hpc-results/source133-valid-self-strength-attempt001/status-status004.json)
records **1,794 collected = 1,701 passed + 93 CUDA-only skipped**, with **zero
failures/errors**, JUnit **246.883 seconds** and SHA256
`00ce287a0c62e136dba74fbe26355c7266996d920437be91bdeb7b0f87a0859f`.
CPU **29770494 remains RUN**, finishing the repair profile; callback **29770495
PEND**. No `cpu_gate` exists yet. Unit success is not a complete CPU/profile
pass, GPU submission or scientific candidate result. The earlier observations
below are retained as submission history.

[Source133 preparation](../outputs/hpc-submissions/source133-valid-self-strength-attempt001-preparation.json)
binds the exact source, coefficient configuration and original epoch-36 lineage.
The [stage-and-queue receipt](../outputs/hpc-submissions/source133-valid-self-strength-attempt001-stage-and-queue-001-output.json)
has return code **0**, with stage and queue return codes **0** and **620 members
authenticated**. It records actual CPU **29770494** and after-CPU **29770495**;
the sealed preparation's earlier `sealed_not_submitted` status is historical,
superseded by this separate operation receipt.

| Identity | SHA256 |
| --- | --- |
| Source archive | `839e50741c4ce8852495c1aa8ab3846bcb3b9aeb5bc39100c2f3d520cfe2578b` |
| Source manifest | `5dd599c7c97e97301ce1d00c733b478e51a77866d76bc9c70088329bcbab8d61` |
| Immutable plan | `63b7f37d5b795e0f6c87f47b95c0c522c1bc315a5a540c5cab1b086978d93a3b` |

The [11:06:35 UTC snapshot](../outputs/hpc-results/source133-valid-self-strength-attempt001/status-status001.json)
shows CPU **RUN**, 16 slots in `hpc`, one-hour allocation; after-CPU **PEND**.
No JUnit, complete CPU gate/profile pass, GPU validation or training is established.
The controller labels this independent test nonpromotable and training forbidden.

The root operator used bundled CPython at
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`
for these exact command tails:

```text
outputs/hpc-submissions/prepare-source133-valid-self-strength-attempt001.py
outputs/hpc-submissions/operate-source133-valid-self-strength-attempt001.py stage-and-queue
```

Preparation parses the 14 declared Python overlays with standard-library
`ast.parse`, authenticates the source132 archive/members, hashes and packages
the declared revision; it does not import project modules or execute numerical
tests. The second command performs byte transfer/staging and scheduler submission;
the receipt retains exact remote staging/controller argv and results. Numerical
fixture/helper parity remains allocated-HPC work, not local evidence. Static
review also required that failed generation observation preserve the primary
health failure, and completion authenticate whole-logical-batch gradient receipts
and step/policy identities rather than update indices alone.

## Evidence motivating the next design

[Source132 terminal evidence](70_epoch36_collapse_mechanism.md#both-quarter-step-proof-arms-failed-at-update-200)
shows that quarter-sized native AdamW displacements did not prevent the
200-update retention failure. Fixed raw-valid counts changed from **641/1,024**
to **587 baseline / 616 with alpha-0.01 self guidance**. Both retained exact
and chemically valid reconstruction at 256/256 but exceeded the two-point
prior-retention limit. Unfiltered categorical validity also fell from
**2,091/4,096** to **1,968 / 2,000**, respectively. The weak arm retained more
validity than its matched baseline, but neither passed; one failed comparison
does not establish a successful loss or an effective coefficient.

The mechanism replay found reconstruction/prior-repair tradeoffs at updates
39 and 96 even while the full frozen-target total loss decreased. This supports
targeted component and finite-step analysis, not an assumption that one cause
or a larger self coefficient has been established. The initial design now
selects **alpha 0.1 only**, ten times source132's weak term, to test whether
stronger valid-output guidance addresses the observed retention loss. It does
not select an alpha-1.0 run, a head-balancing rule or a new margin objective.

## Chosen initial candidate and staged proof

The immutable candidate identity is `source133-valid-self-strength-attempt001`.
Its submitted CPU readiness stage is not a validation pass. The objective is

`L_candidate = L_original_repair + 0.1 * mean_over_all_prior(strict_valid_self_NLL_normalized)`.

For each current prior output, a strict-valid detached hard self target uses
the existing full atom/occupied-charge/unordered-bond categorical NLL, divided
by **N + K + K*(K-1)/2**. Non-strict-valid attempts contribute zero, and the
outer mean retains **all prior attempts**. The self term stays outside the
existing **0.25/0.75 repair branch means and repair ramp**. Its formula,
normalization and target semantics are unchanged from source132; only its
coefficient changes from 0.01 to **0.1**. Original reconstruction, KL, repair,
composition/occupancy weights, reference-35 ramp and search remain unchanged.

Retain the explicit `adamw-displacement-scale-v1` **factor 0.25**: apply one
quarter of the actual native AdamW parameter displacement while preserving the
native moment/counter update. This is the same declared update-size policy as
source132, not a new gradient clip or a proven correction. There is no head
balancing, margin alteration or alpha-1.0 candidate in this initial test.

After the full allocated CPU gate/profile, one GPU allocation of at most
**24 hours** will run the required native checks, a **disposable 100-update
diagnostic probe**, then the **2,500-update proof**. One global **23-hour**
internal allocation clock includes native work, the probe, restoration and proof;
it must not reset between stages. Probe instrumentation must not be counted as
ordinary training-update time in an epoch-runtime projection, while its actual
elapsed time remains charged to that shared allocation budget.

The probe records fixed panels at **0/32/100** and weighted reconstruction,
repair and self component diagnostics at updates **1/39/96/100**. In particular,
measure the weighted self gradient over the **whole logical batch**, rather than
presenting the old first-microbatch probe as a logical-batch gradient norm.
These component observations do not rebalance or change the update. The probe
is a new alpha-0.1 trajectory, not an exact replay of historical source131.
Authenticate full snapshot/model/AdamW/native and fresh-RNG restoration before
the longer proof starts again from original epoch 36; all proof updates remain
disposable. Early panels and every scientific/runtime check remain hard gates.

This candidate is **validation only**: no automatic training or loss adoption
follows even if its checks pass. Preserve original epoch 36 and all failed
source131/source132 evidence. Source hashes and submitted CPU receipts are
recorded above; numerical outcomes and GPU receipts remain pending.

Interpret the eventual panels and generation evidence separately: improved
argmax counts do not establish improved generation if categorical validity or
unique-valid yield falls. Component-gradient dot products are local first-order
observations, not proof that the actual finite-step scalar improved. No such
result or revised fix is claimed by this interpretation policy. Source132's
alpha-0.01 arm improved both argmax and categorical validity relative to its
quarter-step control while both remained below their common start; that result
does not by itself establish an argmax-only sharpening mechanism.

Possible interpretations of the forthcoming whole-batch diagnostics remain
conditional, with no selected remedy:

| Observation, if established | Bounded interpretation |
| --- | --- |
| Tiny weighted self norm, uphill self/repair dot actual displacement, opposing reconstruction/self gradients | Retention may be underpowered amid branch conflict; this is not a finite-step causal proof. |
| Substantial self norm but uphill displacement or repair/self opposition | Directional interference remains possible; AdamW coefficients cannot be chosen from a norm ratio alone. |
| Self NLL improves and argmax rises while categorical/unique-valid yield falls | Surrogate sharpening may mismatch generation quality; a larger alpha need not help. |

## Requirements for a new candidate

Preserve the authenticated original epoch-36 model, inherited native AdamW
moments, native RNG and separate fresh-prior RNG; frozen reference 35 and the
original epoch ramp; **1:2 real/prior sampling** (256 real/512 prior), other loss
weights, graph/search semantics and declared resource settings. Changes to the
new guidance term or update-size policy must be explicit and versioned. A
coefficient change must not silently replace an original branch mean or ramp.
No weak-test result automatically enters a training objective.

Keep the original native/source/input/gradient, finite-value, reconstruction,
raw-chemistry, strict representation, composition/sulfur, size/nonempty,
runtime/memory, resume and fresh-quota checks. Retain fixed **1,024-prior and
256-reconstruction** panels initially, after **32 updates**, and at every
multiple of **100**. Keep separate raw chemical validity, strict schema/canonical
roundtrip acceptance and bounded-search outcomes. No threshold is waived and
no missing/not-reached check is a pass.

Readiness must include the full **2,500-update disposable proof** from original
epoch 36; an eight-update benchmark or a local one-step improvement is
insufficient. Observe unfiltered argmax and categorical validity, unique-valid
yield and declared diversity cohorts, preserving invalid graphs and denominators.
Compare matched starts and update budgets; report the correlation between
categorical draws sharing one latent. Stop and retain nonpromotable diagnostic
state on a failed early panel.

Before any eligible training continuation, discard all disposable model,
optimizer and RNG state and restore authenticated original epoch 36. A training
objective requires an explicit new scope and all required checks; neither this
plan nor a better failed weak endpoint submits or promotes training. Existing
continuation limits remain serial allocations at most **24 hours**, **23 hours
internal**, absolute endpoint **39** for the short branch, within the global
100-epoch maximum.

## Pending records

After the design is chosen, record its exact formula/normalization and scope,
immutable source/archive/manifest/plan identities, sealed numerical commands,
actual submission receipts and every passed/failed/partial outcome in the
[experiment ledger](05_experiment_ledger.md). All source127–source132 failures,
the source132 mechanism replay and both independent proof outcomes remain
authoritative historical evidence. No new HPC action or numerical test was
performed to write these preparation notes.
