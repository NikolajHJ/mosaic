# Bond-weighted repair and valid-self target guidance

Status, 22 October 2026 11:32 DTU-local: **GPU 29780820 EXIT**, terminated
**10:48:23** with exit code 1 after the preliminary **eight-update** benchmark
failed raw-prior validity retention. **The 200-update probe and 2,500-update
proof never started.** The lower-NONE-weight candidate did not stabilize validity;
training remains stopped and epoch 36 remains the latest completed checkpoint.
[Saved terminal status](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt002/status-news002.json)
and [compact saved-artifact summary](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt002/outcome-news002-summary.json)
are authoritative for this status update. No fix, waiver or resubmission is
established by this read-only check.

## Terminal outcome: early validity collapse before the probe

All **453 mandatory CUDA fixtures passed**, with zero failures/errors/skips.
The original **18 native gates** passed, including finite gradients, exact
execution/resume, memory and runtime. Both weighted repair/self derivative
checks and policy bindings passed. Of the 19 supplementary readiness checks,
only `raw_prior_validity_retention` failed. Thus the implementation-level
passes must not be reported as a passed chemistry-health result.

| Fixed 256-prior panel | Start from epoch 36 | After eight updates |
| --- | ---: | ---: |
| Raw chemically valid argmax graphs | 149/256 (58.20%) | 12/256 (4.69%) |
| Unique valid graphs | 149 | 12 |
| Expected occupied atoms | 22.31 | 22.23 |
| Expected sulfur fraction | 1.627% | 1.380% |

Held-out reconstruction remained **256/256 exact and chemically valid**.
The model/benchmark updates were discarded; no training checkpoint was promoted.
The stronger loss balance is therefore not a working correction under this
declared protocol. These results do not isolate a single cause of the earlier
source133 collapse.

The fresh 512-prior training batches also deteriorated: strict-valid counts
were **317, 312, 288, 249, 202, 120, 79, 45**. These are different fresh draws,
not the fixed-panel transition counts. Their first-to-last classification
changed from 139 sanitization/56 disconnected failures to 466 sanitization/
one disconnected failure, with zero prior representation failures throughout.
This suggests a changed chemical failure pattern; it is not evidence that
disconnection alone explains the new collapse or a paired causal diagnosis.
Sulfur, size, finite-value and runtime guards passed; no gradient clipping was
applied in these eight updates. Preserve all sealed source/plan/failure records.

Earlier status, 22 October 2026 10:43 DTU-local: **GPU 29780820 RUN**, started
**10:42:58** on `n-62-18-8`, ahead of the earlier provisional scheduler estimate.
[Saved live status](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt002/status-news001.json)
shows CUDA fixtures in progress, with no completed CUDA JUnit, native outcome,
200-update probe or 2,500-update proof yet. Thus there is no evidence yet that
NONE weighting prevents collapse. Full CPU gate/profile remains passed;
training remains forbidden. Attempt001's failed gate and receipts are preserved.
All numerical validation remains on allocated DTU HPC. Local work is editing,
static review, packaging and saved-metadata inspection only.

The user resumed investigation of the chemistry-retention collapse and
authorized a targeted correction from authenticated original epoch 36, validated
before training. Preserve [source133's failed longer proof](71_valid_output_guidance_iteration.md#terminal-outcome-alpha-01-did-not-pass-longer-health)
and every earlier failed result. Source133 improved its 100-update panel and
categorical generation, then failed retention at 200/2,500 updates; increasing
the valid-self coefficient to 0.1 did not solve the problem. Its full component
gradients showed directional conflict, not proof of a single cause.
Its aggregate chemical-failure counts changed **277 → 275 sanitization** and
**106 → 158 disconnected**, accounting for the net 50 additional invalid
outputs. These aggregate counts do not establish paired transition causes,
NONE-gradient domination or inherited-moment causation.

The new identity is `source134-bond-weighted-chemistry-attempt001`, release
`source-134-bond-weighted-chemistry-attempt001`; actual immutable bindings and
submission receipts follow below. A [read-only preparation preflight](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt001/preflight001.json)
authenticated the original epoch-36 checkpoint and observed **14.57 GiB free**,
with only unrelated Mosaic work running. That snapshot is not reserved space
or permission to promote training; fresh gate/quota checks remain mandatory.

## Attempt002 full CPU passed; GPU queued

The [08:40:45 UTC saved status](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt002/status-gate005.json)
records **CPU 29780794 DONE**, full `cpu_gate` and repair profile **passed**,
and **callback 29780795 DONE**. Allocated JUnit: **1,965 collected, 1,842 passed,
123 CUDA-only skipped, zero failures/errors**, **249.504 seconds**. A separate
[static fixture-only audit](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt002/static-fixture-only-audit001.json)
authenticates **560 scientific source members**, confirming the single test
fixture change and Python AST parsing; production loss/protocol/guards are unchanged.

The callback submitted **GPU 29780820** at **10:38:32 DTU-local**. It is **PEND**
in `gpua100`, requesting one exclusive **80GB A100**, **16 CPU slots**, **24h**.
The saved wait reason is job-slot limits on five hosts; the scheduler's
**no-sooner-than 14:15:03 DTU-local** estimate is provisional, not a guaranteed
start or completion time. No GPU fixtures/native/probe/proof result exists yet,
and no training job is submitted.

The CPU repair profile matched the same semantic SHA256
`8be4a9ef5749555a3a275673b2a8af41fdc31c4a2eeb148a26482f30b17985d6`
across all four execution backends and 0/4/12-worker settings. Its **four real /
32 prior diagnostic points** produced **11 accepted / 23 censored / two
already-valid**, all 36 attempts accounted for. This small execution/parity
cohort is not the 256-real/512-prior training ratio, a native GPU chemistry
outcome or evidence that the new loss prevents collapse. Historical profile
labels and attempt001 failure are retained without reinterpretation.

## Attempt002 actual fixture-only seal and CPU submission

[Attempt002 preparation](../outputs/hpc-submissions/source134-bond-weighted-chemistry-attempt002-preparation.json)
declares exactly one changed source member,
`tests/test_chemistry_target_weighting_validation.py`, SHA256
`25071c0f79f785c2c903bd192945701720512093037284f9826005729259040d`.
It adds the dummy readiness file and missing seed to the mocked-loop fixture.
Production loss/code, scientific configuration and all guards are unchanged.
The [actual stage-and-queue receipt](../outputs/hpc-submissions/source134-bond-weighted-chemistry-attempt002-stage-and-queue-001-output.json)
records overall/stage/queue codes **0**, **629 authenticated members**, CPU
**29780794** / after-CPU **29780795**, `submitted_not_passed`, no native GPU
submitted and training forbidden/nonpromotable. No unit/full CPU pass follows
from staging or submission.

| Attempt002 immutable identity | SHA256 |
| --- | --- |
| Archive | `17109ed43a7b58b6449467cde93385149290e83fa5d959a1576137a6e69a9c77` |
| Manifest | `e825c36df6cdbebc8395624692d6717d946ebc3f19bdf7518ad3a1416acba3d8` |
| Plan | `5b2bc87527ce06bffd065544ef155d9fadebfb4fbee42c34d2918b8bfe4ce26e` |

Attempt001's source/plan/failure below is retained unchanged. Only the new
immutable attempt can acquire new validation evidence; no scientific gate is
relaxed and no failed result is relabeled as a pass.
Root used the same bundled CPython with command tails
`outputs/hpc-submissions/prepare-source134-bond-weighted-chemistry-attempt002.py`
and `outputs/hpc-submissions/operate-source134-bond-weighted-chemistry-attempt002.py stage-and-queue`.
These are static byte preparation/transfer/scheduler operations; no local
numerical tests ran. The operation receipt retains exact sealed remote argv.

## Attempt001 failed CPU fixtures; new immutable fixture-only retry pending

This earlier failure/preparation observation is superseded only by the actual
attempt002 submission above; its failed outcome is unchanged.

The [allocated CPU failure snapshot](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt001/status-cpu003.json)
records **1,965 collected = 1,838 passed + 123 CUDA-only skipped + four failed**,
**zero errors**, JUnit **242.756 seconds**, SHA256
`57cdd766b5e7c506338ece67f17cf622ba71fb3397017b64a4eba64a83e91c15`.
CPU **29780665 EXIT**; callback **29780666 EXIT** correctly stopped, with no
GPU submission. This is an actual failed CPU gate, not a candidate native
chemistry/retention outcome.

All four failures are the variants `[None,32,100,200]` of
`test_shared_loop_executes_declared_schedule_and_restores_before_return` in
`tests/test_chemistry_target_weighting_validation.py`. Their mocked setup did
not create `failure_readiness.json`, so file hashing raised `FileNotFoundError`
before the fake update loop. Production numerical fixtures passed; no production
numerical assertion failed in this allocation. The missing fixture metadata
still blocks readiness and must be corrected without bypassing authentication.

The planned new identity is `source134-bond-weighted-chemistry-attempt002`, release
`source-134-bond-weighted-chemistry-attempt002`, with a fixture-only correction.
No attempt002 seal/hash/job/pass is established yet. Preserve attempt001 archive,
plan, receipts and failed JUnit; loss, policy, health/runtime guards and protocol
remain unchanged. No sealed attempt001 file is patched or result relabeled.

## Actual sealed staging and CPU submission

The [08:10:10 UTC first live snapshot](../outputs/hpc-results/source134-bond-weighted-chemistry-attempt001/status-first001.json)
shows CPU **29780665 RUN** since about **10:09 DTU-local**, after-CPU
**29780666 PEND**. Official work3 quota is **85.72/100 GiB used / 14.28 GiB
free**, an unreserved snapshot. No allocated numerical result or GPU submission
is established yet.

[Preparation](../outputs/hpc-submissions/source134-bond-weighted-chemistry-attempt001-preparation.json)
authenticates the source133 predecessor bytes and the new exact declared source/
configuration. The [actual stage-and-queue receipt](../outputs/hpc-submissions/source134-bond-weighted-chemistry-attempt001-stage-and-queue-001-output.json)
records overall/stage/queue return codes **0**, **629 authenticated archive
members**, and CPU **29780665** / after-CPU **29780666**, status
`submitted_not_passed`. It explicitly reports native/stability not yet submitted,
training forbidden and nonpromotable. The preparation's earlier
`sealed_not_submitted` value is preserved as its pre-submission state.

| Immutable identity | SHA256 |
| --- | --- |
| Source archive | `008ca34133d8af9a112cca8a4ddc5253e02f562311391666b15b0c78d0872580` |
| Source manifest | `0fb25c2d86b8955395c0e5b850fc99b4a116cbc34addca9b1992d1a6638d6790` |
| Plan | `952bce2c77d1ab05fb1f0825ea68ce67c3da8046634264eefdc416a506bb9a7f` |
| Retained source133 failure evidence | `a67e440504595542b2bb372419fc2084b0e2b6e2083422cda39c8b284ee3b719` |

Root used bundled CPython
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`
with exact command tails:

```text
outputs/hpc-submissions/prepare-source134-bond-weighted-chemistry-attempt001.py
outputs/hpc-submissions/operate-source134-bond-weighted-chemistry-attempt001.py stage-and-queue
```

Preparation authenticated the predecessor source, AST-parsed **18 Python
overlays** plus generated stage/operator/collector Python, and hashed/packaged
the declared bytes. This was standard-library/static work, with no project
imports or numerical tests locally. The operation receipt retains exact remote
staging and sealed-controller argv/results. No sealed archive or plan was
modified for these documentation updates. Allocated CPU fixtures/profile and
all GPU/native/probe/long-proof outcomes remain pending.

## Declared change and hypothesis

The new protocol field is

```json
"chemistry_target_weighting": {
  "version": "bond-weighted-target-ce-v1",
  "none_bond_weight": 0.1
}
```

Apply this policy to both **accepted repair-target CE and strict-valid self-target
CE**. Atom and occupied-charge labels retain unit weight. Occupied unordered
bond decisions with a non-NONE target retain unit weight; a NONE target has
weight **0.1**. For a fixed detached target, let **N** be the atom-slot decisions,
**K** its occupied-charge decisions, **E** its non-NONE occupied-pair bond
decisions and **NE** its NONE occupied-pair bond decisions. The normalized loss
is the weighted CE sum divided by **D = N + K + E + 0.1 * NE**. This weighted
categorical surrogate is not the graph's exact negative log probability or a
valid-mass certificate. All prior attempts
remain in the outer mean, including zero-contributing ineligible self targets;
this is not filtering the generation population or dropping NONE labels.

This tests whether reducing the contribution of numerous NONE bond decisions
improves the target-guidance balance. It is a **versioned hypothesis**, not a
proved collapse mechanism or a passed fix. No stronger self coefficient, head
balancing, target-search change or margin objective is added. Keep self alpha
**0.1**, original **0.25/0.75** repair branch means and original repair ramp;
all other loss terms and weights remain unchanged.

The immutable top-level marker is
`target_weighting_test={version:"epoch36-bond-weighting-test-v1"}`. It is allowed
only with `self_guidance_iteration` alpha 0.1, the exact declared loss policy and
the **validation-only self-guidance-test purpose**. A loss field without its
marker, a marker without its loss, unknown/null/changed policy, baseline-training
purpose or any promotion attempt must fail closed. Absent-marker historical
plans retain their original behavior and accepted-target objective.

The new top-level probe policy is
`target_weighting_probe={version:"chemistry-target-weighting-probe-v1",updates:200,gradient_updates:[1,39,96,100,199,200]}`.
The old `self_strength_probe` field is absent; a historical 100-update probe
cannot supply the new 200-update contract or be relabeled by changing a hash.

## Metadata/controller implementation and static scope

`chemistry_target_weighting_policy.validate_target_weighting_test(plan)` closes
the top-level test scope. The protocol-only baseline reader may validate the
exact primitive weighting schema and compare the remaining original protocol;
**that reader is not launch authorization**. Actual entrypoints authenticate
`read_plan` and full short-plan validation, which rejects a weighted loss/probe
without its top marker and rejects a marker without its exact loss/alpha/purpose.

The independent controller retains CPU/source/predecessor/quota/phase-lock and
uncertain-intent guards, has no training/promotion action, and explicitly
dispatches `latentmol.chemistry_target_weighting_validation` only for the
validated marker. All four fixtures are mandatory in both CPU and CUDA commands:

- `tests/test_chemistry_target_weighting.py` (primitive/weighted-loss behavior).
- `tests/test_chemistry_target_weighting_policy.py` (closed scope and controller).
- `tests/test_chemistry_target_weighting_validation.py` (probe/restoration/evidence).
- `tests/test_chemistry_target_weighting_training.py` (actual production update,
  microbatch/full scalar and gradients, RNG, quarter step, sync/async and observer).

Metadata-owner static checking used bundled CPython
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`
with standard-library `ast.parse`/SHA256 over exactly
`src/latentmol/chemistry_target_weighting_policy.py`,
`src/latentmol/chemistry_short_continuation.py`,
`scripts/hpc/chemistry_independent_self_test.py` and
`tests/test_chemistry_target_weighting_policy.py`: **four files parsed**.
No project modules were imported and no fixtures/numerical/model work ran locally.
The existing `chemistry_self_guidance_iteration.py` needed no change.

Native self finite differences must use the actual weighted policy. New accepted
repair-target finite differences and decoder VJP checks add the exact gates
`weighted_repair_native_decoder_vjp`, `weighted_repair_directional_derivative`
and `weighted_target_policy_bound`, while retaining every original native and
supplementary gate. New loss/test identity is bound through native readiness,
early baseline, proof/resume and generation records; original checkpoint/code
handoff exclusions require the validated explicit new test and exact declared
source paths. Implementation/static inspection is not proof that these checks
pass numerically.

## Staged validation and immutable boundaries

Retain the original epoch-36 model, native AdamW moments and counters, native
and fresh-prior RNG, frozen epoch-35 reference/ramp, **1:2 sampling** (256 real /
512 prior), microbatch 128, 12 workers/four main threads/16 CPU slots, original
bond search and every native/health/runtime/resume/quota guard. The actual AdamW
parameter displacement remains scaled by **0.25** with native moments retained.

After full allocated CPU readiness, one GPU allocation at most **24h**, under
one shared **23h internal clock**, will run native checks, a **200-update
disposable probe**, complete restoration, then a fresh **2,500-update proof**
from original epoch 36. Probe panels are **0/32/100/200**, and whole-logical-batch
weighted component diagnostics occur at **1/39/96/100/199/200**. This extends
observation into the previously failing interval; a failed health panel still
stops the candidate and cannot be relabeled as a pass. Full model/reference,
optimizer, gradients, modes and RNG restoration must authenticate all eight
snapshot components before the longer proof. Probe VJP instrumentation is not
ordinary-update runtime; its actual elapsed time still consumes the shared
allocation budget. Normal update projection and all runtime gates remain.

The fresh proof keeps fixed **1,024-prior / 256-reconstruction** panels initially,
at 32 and every 100 updates, including separate raw chemical and strict schema/
canonical acceptance. Retain exact/valid reconstruction, composition/sulfur,
size/nonempty, finite gradients/parameters, memory, source/resume identity and
fresh quota checks without waiver. Save unfiltered argmax and four-draw
categorical generation, unique-valid yield and declared diversity cohorts;
keep invalid graphs and all denominators. First-order component dots/cosines
are observations, not finite-step or validity guarantees.

New helper/policy/worker/controller fixtures must be mandatory in both allocated
CPU and CUDA commands. Cover old absent-marker parity, target-normalization and
mask/provenance checks, meaningful scalar/parameter-VJP parity, exact native
AdamW/RNG behavior, failure/cleanup restoration and rejection of missing,
partial, stale or wrong-source evidence. Every receipt must bind the new policy
and test marker consistently through native readiness, early baseline, probe,
long proof and source/parent authentication.

This candidate is nonpromotable and **training forbidden**, even if validation
passes. Any later training requires separate explicit scope, passed checks and
restoration of original epoch 36, within existing serial 24h/23h limits and the
absolute-39 short-branch endpoint. Source133 and all historical results remain
sealed. Exact source/plan hashes, commands, job IDs and actual outcomes will be
added to the [ledger](05_experiment_ledger.md) only when available.
