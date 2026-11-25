# Minimal integrated Boltz–DeFoG engineering test

**Verified result, 10 October 2026 21:15 DTU-local:** source089 **native
29674537 completed and passed** the prescribed six-update integration,
after CPU 29674536 passed 53 checks in 11.87 seconds. All six updates had
actual guidance, eight eligible native outputs supplied two selected
targets, and all three full/repair guards passed. Both gradients reached
the same `z`; frozen models and fixed slot identity stayed unchanged.
Raw validity was 0/14 candidates and 0/7 argmax states, so this is bounded
engineering validation without chemical or binding efficacy evidence.
The user requested a working integrated DeFoG test as soon as practical.
This bounded six-update test establishes integration behavior; it does not
replace or certify the failed source079 long runs.

The [integration controller](../src/latentmol/repair_guidance/integration_smoke.py),
[durable submitter](../scripts/hpc/submit_defog_integration_smoke.py) and
[fixtures](../tests/test_defog_integration_smoke.py) are implemented/frozen
after independent review. The metadata-only
[control-file builder](../outputs/hpc-submissions/build-source089-control-files.py)
prepares the immutable plan and unique control paths. The 53 engineering
fixtures passed on allocated DTU HPC CPU resources. Coordinator static checking parsed these four
files with bundled Python 3.12.14 using only `ast`, `pathlib` and `sys`;
the exact command is retained in the [ledger](05_experiment_ledger.md). It made
no project/model import or numerical check.

## Prescribed start and frozen models

Reuse source079's exact saved prior-0 latent, with its initial **22 occupied
slots fixed** throughout six accepted updates. Do not redraw, filter or
select a more favorable start. The saved latent is
`/work3/s234818/latentmol/runs/source056-random-start-study/20261005T225808Z_vae-random-starts-056_37735924/seeds/seed-0000/pilot/20261006T002545Z_vae-random-0000_09d4513e/start.pt`,
SHA `9c8a0ceb3c863236012e5764d5ff779997bbf7597972f9aff103273a73175c42`.
The original six-step config is
`/work3/s234818/latentmol/submissions/source079-defog-contract-attempt001/prepared/native-gate-prior-0.yaml`,
SHA `4606f4cd669e98f5f9cfac6f8c40ff5ff6dd831af99e43776f6999476df4e777`.
The new sealed config authenticates this parent and declares its changes.

The epoch-89 decoder, separate native Boltz affinity/confidence checkpoints,
context/pose and frozen pocket remain pinned. Optimize only molecular `z`
with the existing full DBMol objective, fixed occupancy and Armijo control.
All branches retain one canonical graph. The strict neutral DeFoG adapter
uses the pinned public GuacaMol model, direct candidate initialization at
effective time 0.8 and its native 1,000-boundary grid. Unsupported charged
inputs remain explicit rejections; this test does not use the optional
lossy charge-projection mode or a second chemical decoder.

## Declared arithmetic and audit cadence

Opt into `contact_and_assembly_float64_v1`: widen contact reduction and
subsequent DBMol scalar assembly only. Native heads/trunks, frozen decoder
and fixed-target repair NLL remain FP32. Source086 validated contact and
combined gates at the saved prior-0 point; that scoped result is not a
substitute for fresh gates on this trajectory. Objective signs, weights,
model parameters, original FD tolerances and target-selection policy remain
unchanged.

Set `gradient_audit_interval=6`. With repair refreshes at steps 0 and 5,
the full strict native/component/combined/repair gates run at **0, 5 and 6**.
This is an explicit reduction in audit cadence from source079's audit at
every update, not relaxed numerical acceptance. Retain all three gate
records and stop on an unsuccessful gate.

For each full gate, every nonzero DBMol component must pass its declared
FD policy. A stationary component is retained explicitly as
`zero_at_current_point_uninformative`, not certified as a passing derivative.
The combined gradient and the strictly nonzero repair-only FD must pass
at all three points. This distinction prevents a zero component from
being presented as native gradient verification.

Each refresh executes four prescribed DeFoG attempts, **eight attempts
total**, without refilling or ranking by Boltz score. Select the first
eligible target in prescribed order. Keep the target fixed and detached
through every FD probe and line-search trial for that state; selection,
DeFoG transitions and chemistry conversion remain outside autograd.
Gradients flow through the shared decoder's target CE and frozen Boltz
branches into `z`. Preserve the late linear weight schedule from 0.001 to
0.1 over the six actual updates; an early stop must not invent endpoint
progress. Record target/weight changes separately from within-state loss
decrease.

## Execution and acceptance

Run engineering fixtures on DTU HPC first, then one **two-hour GPU
allocation**, with LSF-managed MPS and a **6,600-second child cap**. The
allocated-node parent/worker coexistence probe must pass before native
optimization. No numerical tests or model execution run locally. A rough
35–45 minute compute expectation is uncertain and excludes queue delay;
the allocation/child limits are ceilings, not a completion guarantee.

| Evidence | Required engineering outcome |
|---|---|
| Budget and updates | All six requested updates accepted; no timeout, failed gate or stalled substitute for completion. |
| Actual guidance | Eligible native target selected at both refreshes; finite nonzero Boltz and repair gradients and guided accepted updates, not scheduled lambda alone. |
| Native and FD identity | Fresh native parity, every nonzero component FD, combined gates and strictly nonzero fixed-target repair FD pass at steps 0, 5 and 6; stationary components remain explicitly uninformative. |
| Frozen/shared boundaries | Model/parent hashes unchanged; fixed 22-slot identity; no hidden graph replacement or gradient through DeFoG/target selection. |
| Cost and termination | Retain attempted/completed native forwards, decoder probes, DeFoG transitions, backwards, line-search trials, timings, failure/termination records and raw validity. |

No fresh-pose rescore is included in this engineering proof. Raw candidate
validity and unsupported/rejected attempts remain outcomes rather than
being silently repaired for scoring. The historical source079 six-step
gate completed with real guidance but zero valid raw candidates, so native
targets and accepted optimization steps must not be called chemical or
binding success. A passed new smoke establishes this bounded integration,
not a successful 100-update trajectory or universal gradient remedy.

The [comprehensive diagnosis](48_defog_comprehensive_diagnosis.md) retains
source079 failures and the scoped source085/086/087 evidence. Implementation
boundaries are described in the [repair-guidance contract](37_defog_repair_guidance.md),
[GPU process recovery](43_defog_gpu_process_recovery.md),
[active-set optimizer](../src/latentmol/boltz/active_set.py) and
[DBMol objective](36_dbmol_composite_objective.md). Source/config identities,
exact launch commands and static-check scope are recorded below. Completed
native receipts establish this short integration; they do not establish
chemical improvement or stability over 100 updates.

## Completed native result

The [controller outcome](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-003/runs/source-089-defog-minimal-smoke-attempt001-smoke/controller/20261010T123626Z_defog-minimal-integration-smoke_3bfec22e/smoke.json)
records `passed=true`, six accepted/guided updates, eight eligible native
outputs and two selected targets over two refreshes. Nonzero component,
combined and repair-only guards passed at 0/5/6; stationary components retain
their explicit uninformative status. Frozen identities and both gradients
into `z` passed the controller checks. Child execution took
2,271.3735201 seconds (37.9 minutes), with no timeout. The scheduler log
records 14:36:12–15:14:33 DTU-local, 2,301 seconds total.

The [trajectory record](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-003/runs/source-089-defog-minimal-smoke-attempt001-smoke/controller/20261010T123626Z_defog-minimal-integration-smoke_3bfec22e/native-smoke-trajectory.json)
retains **0/14 valid raw candidates and 0/7 valid argmax states**. Binding
loss fell from 7.8363120918 to 6.3372382609 overall, but the last accepted
step increased it from 6.266332 to 6.337238 while held-target NLL fell from
450.919739 to 442.285461 at lambda 0.1. This documents competition between
the branches; binding loss is not monotone. No fresh-pose scoring occurred.
The cost receipt retains 648 component-FD and 234 combined-FD Boltz forwards,
216 repair-FD decoder probes and eight DeFoG workers, alongside separate
optimization, parity and line-search costs.

The [completed capsule003](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-003.tar.gz)
SHA is `d33f64e709d9bc21653628063f0ff610164c76991a9b4749171c2743c6b9c543`,
exported `2026-10-10T19:15:46.765296+00:00`, with 121 safe members / 120
verified files. Its [verification receipt](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-003.tar.gz.verification.json)
records local byte/hash/metadata-only checks using Python 3.12.14 / Windows 11.
Persisted completed run records and successful scheduler logs establish
completion even though the finished job has aged out of `bjobs`. Earlier
pending snapshots below remain history; this status turn changed no source
or scheduler job and ran no local numerical test.

## Immutable preparation and submitted jobs

The [archive](../outputs/source-releases/source-089-defog-minimal-smoke-attempt001.tar.gz)
SHA is `2d6b27c9402cc9465eed64b53d145c3178157c45e6805fb5a2267e2f743bbf64`,
manifest `1c4157728c0304fcbe9b3348877ec35096d05b98d7e357bebc6448b69c3a04a7`;
HPC staging authenticated 454 safe members. The
[plan](../outputs/hpc-submissions/source089-defog-minimal-smoke-attempt001-plan.json)
SHA is `aeb469465bb1c1459957aa59e6cdbb02f9d19390c3916686b5ebc3e18a5c7e44`,
[config](../outputs/hpc-submissions/source089-defog-minimal-smoke-attempt001-config.yaml)
SHA `af8a328c1f1258272a96c9d18b95c26fcf49c6199dfce4e9a9b62d7c1871f770`.
Release is `/work3/s234818/latentmol/releases/source-089-defog-minimal-smoke-attempt001`;
unique control/receipts use
`/work3/s234818/latentmol/submissions/source089-defog-minimal-smoke-attempt001`;
output is `/work3/s234818/latentmol/runs/source-089-defog-minimal-smoke-attempt001-smoke`.

Actual **CPU 29674536** uses `hpc`, 30 minutes, two cores / 4 GB per core.
Actual **native 29674537** uses `gpua100`, two hours, four cores / 16 GB per
core, one A100 80 GB GPU with managed MPS and `done(29674536)` dependency.
The historical [CPU evidence capsule](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-001.tar.gz)
SHA is `ce40d210bdde17fb7dc0980f3a699e7a0b16d1755d2632de197e59db956b9d70`,
exported `2026-10-10T11:28:06.858270+00:00`, with 17 safe members and
16 individually verified files. Its [verification receipt](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-001.tar.gz.verification.json)
records byte/hash/metadata checks only using local Python 3.12.14 on
Windows 11. The completed [CPU run](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-001/runs/20261010T112738Z_source-089-defog-minimal-smoke-attempt001-cpu_4295366b/run.json)
and [command log](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-001/runs/20261010T112738Z_source-089-defog-minimal-smoke-attempt001-cpu_4295366b/command.log)
establish the 53 passed engineering checks. At that snapshot native
29674537 was PEND; no short native result had yet been established.

The later historical [capsule002](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-002.tar.gz),
SHA `d896e3ffa798006965d5e2d61207b67cbddad59118253e21b6a1d582ddcaf4dd`,
exported `2026-10-10T11:37:15.491062+00:00`, verifies the same CPU-completed /
native-queued state, with 17 safe members and 16 file hashes. Its adjacent
[verification receipt](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-002.tar.gz.verification.json)
retains the standard local byte/hash/metadata-only verification; there was
no native outcome at that snapshot.

Local packaging used bundled Python with
`-c 'import sys,runpy;sys.path.insert(0,"src");runpy.run_path("scripts/package_multimodel_release.py",run_name="__main__")'
--source-root . --output outputs/source-releases/source-089-defog-minimal-smoke-attempt001.tar.gz`.
The stdlib builder used `--packaging-receipt
outputs/source-releases/source-089-defog-minimal-smoke-attempt001.tar.gz.json`.
Transfer used `sftp -b outputs/hpc-submissions/upload-source089.sftp dtu-hpc`,
then the [staging helper](../outputs/hpc-submissions/stage-source089-release.py)
authenticated archive/manifest before the
[durable metadata launcher](../outputs/hpc-submissions/submit-source089-minimal-smoke.py)
submitted unique stages. It retains exact expanded arguments, plan/config
identity and receipts. Static/local packaging, metadata generation and
file transfer run no numerical/model test. All fixture/model execution
remains allocated on DTU HPC.
