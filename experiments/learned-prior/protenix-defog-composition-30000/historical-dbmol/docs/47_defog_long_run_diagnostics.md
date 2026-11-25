# DeFoG long-run timeouts and Boltz derivative-gate investigation

Current verified status, **10 October 2026 00:10 DTU-local**: source080's
exact-state replay completed **all three cases**, with unchanged parent/frozen
identities and native parity. The original FD failures reproduced and **no
old gate was certified**. Repeated gradients, grad/no-grad endpoint values
and shared graphs were identical; scalar objective assembly was exact.
Component-gradient sums had maximum absolute differences
**7.47e-6 / 1.55e-6 / 3.58e-6** against norms **9.45 / 4.31 / 1.90**.
This excludes gross assembly error and observed forward-mode mismatch or
stochastic drift for these saved cases; it does not identify the remaining
numerical cause or establish a production fix. Replay runtimes were
**284.14 / 284.21 / 449.96 seconds**, return code 0, no timeouts. It made no
DeFoG calls or optimization updates. The
[verified replay capsule](../outputs/hpc-results/source080-attribution-002.tar.gz)
and [case receipts](../outputs/hpc-results/source080-attribution-002/runs/source-080-failed-gate-attribution-attempt001-replay/20261009T141517Z_source080-failed-gate-attribution_4b1c235b/cases.json)
are recorded with hashes/commands in the latest ledger entry.

Source079's controller finished with **all ten long cases failed**: prior
1/2/4/6 timed out at 6,000 seconds, while the other six returned 1. Controller
completion does not mean cohort success. The successful six-update native
repair/gradient gate remains valid separate evidence; runtime validation
cost is confirmed and the numerical Boltz mismatch cause remains unresolved.
The [status capsule](../outputs/hpc-results/status-20261010-check001.tar.gz)
preserves the final campaign records. No numerical work, jobs, policy,
tolerance or scientific settings changed during this status-only retrieval.

Historical status, **9 October 2026**: source079's real six-update repair/gradient gate
worked. The failed long runs expose two separate problems: an undercalibrated
runtime budget dominated by repeated Boltz validation, and later Boltz
finite-difference mismatches whose numerical causes remain unresolved.
Source080 attribution is now sealed/uploaded and submitted. CPU **29661601
DONE**, **31 fixtures passed in 12.94 seconds**; native **29661602 PEND**, one
four-hour 80 GB A100/MPS saved-state replay. These are engineering passes,
not a completed real-model replay or numerical-cause finding. This report
reads HPC artifacts; no numerical work runs locally.
No production objective, ramp, tolerance, optimizer or validation policy changed.

## What already worked

The source079 short gate completed six accepted updates. Eight prescribed
native DeFoG attempts yielded eight eligible targets; two targets were selected
at the two refreshes. All six updates had nonzero repair guidance and were
accepted, with seven passing fixed-target derivative checks. This establishes
that the corrected native interface can perform repair and supply a decoder
loss gradient to molecular `z`.

DeFoG sampling itself remains outside autograd. Its output is a detached,
slot-aligned reference; differentiation passes through decoder categorical
loss and frozen Boltz branches. The repaired reference does not replace the
shared graph scored by Boltz. This evidence does not establish a chemically
valid raw optimized graph, binding efficacy or completion of a 100-update run.
The original CUDA, native-frame and diagonal-contract failures remain in
[docs43](43_defog_gpu_process_recovery.md) and the ledger.

## Declared budget and the missed validation cost

The unchanged protocol requests one 23-hour 80 GB A100 allocation. Within it:

| Work | Declared cap / cadence |
| --- | --- |
| Native gate | 7,200 seconds, six updates, four draws per interval-five refresh |
| Long optimization child | 6,000 seconds, 100 updates, two draws per interval-ten refresh |
| Native worker attempt | 600 seconds; candidate-start time 0.8 on the pinned 1,000-step grid |
| Fresh-pose rescore | 900 seconds total per case |
| Serial controller | 79,200 seconds; reserve 6,960 seconds before each case |

The long configuration's `gradient_audit_interval = 50` does **not** mean
the expensive full validator runs only at 0/50/100. The `run_arm` condition
also validates at **every repair refresh**, invoking the entire DBMol component
and combined-objective finite-difference suite. A full fixed-mask trajectory
therefore has gates at 0, 10, ..., 90 and 100: eleven full gates, with any
context-transition audit handled separately. A target-only decoder check is
much cheaper, but source079 does not limit refresh validation to that check.
The six-update gate deliberately uses audit interval one and validated seven
states. Its pass is a correctness check, not a calibrated 100-update runtime
forecast.

The short-gate receipts already separate the costs: component FD used **1,548
recorded Boltz forward calls / 2,772.652 seconds**, combined FD **540 calls /
967.694 seconds**, versus eight DeFoG workers across the two refreshes taking
**457.704 seconds**. Six actual optimization forward/backward calls took
**36.853 seconds**, and six line-search evaluations **10.753 seconds**.
Target-only FD used 504 decoder probes and 5.638 seconds. These are recorded
category costs; nested analytic subcategories must not be summed as independent
wall time.

The full records confirm **prior 1, 2 and 4 each reached 60 accepted updates**
and were killed at their 6,000-second cap **inside the full step-60 gate**.
Their execution receipts are authoritative; the killed child `run.json`
files remain stale `running` because the process could not finalize them.
They are not completed 100-update endpoints.

| Timed-out case | Boltz FD seconds | DeFoG seconds | Actual update seconds | Line-search seconds |
| --- | --- | --- | --- | --- |
| Prior 1 | 3,439.975 | 803.238 | 370.104 | 793.038 |
| Prior 2 | 3,679.071 | 707.528 | 378.573 | 632.327 |
| Prior 4 | 3,906.834 | 653.327 | 402.310 | 435.672 |

These observed costs establish the runtime bottleneck: repeated full Boltz
FD validation consumes substantially more of the budget than native DeFoG
repair. The original 6,000-second child cap was undercalibrated for this
validation cadence. They do not prove that all remaining updates would fit
any particular new limit or authorize dropping safety checks. Output/worker
startup, parity, decoder and bookkeeping costs remain separate in the records.
The controller has no automatic continuation/retry; a future runtime policy
must be explicit and preserve interrupted/failed work.

## The numerical failures are Boltz guard failures

| Case / saved step | Guard and observed direction pattern |
| --- | --- |
| Prior 0 / 20 | `dbmol_contact_loss`: gradient direction and two proposed random directions pass; only `random_2` lacks two neighboring passing scales |
| Prior 3 / 50 | `dbmol_contact_loss`: only `random_1` fails, with systematic FD bias on small scales despite stable contact winners |
| Prior 5 / 20 | Fixed-target repair and all three DBMol component checks pass; the combined objective's gradient direction passes but all three proposed random directions fail |

Every recorded target-only repair gate passed, including these failed
validation points. Calling the failures incorrect DeFoG gradients would
misattribute the observed checks. The active `local_v1` policy retains
**5% relative / 1e-6 absolute tolerance**, gradient plus three preselected
random directions, and at least two neighboring informative passing scales.
The exact failed reports and prepared YAML both record `fd_rtol = 0.05`.
No tolerance has been widened and no failed direction has been reclassified.

Saved contact winners remain stable for prior 0's small probes through
epsilon 0.003 and prior 3's through 0.03. A contact minimum winner switch
alone therefore does not explain those failed stable-winner rows. This does
not rule out other nonlinearities, FP32 cancellation/rounding, mode differences
or an algebra/implementation problem. Those alternatives remain hypotheses.
The earlier affinity-only local-FD proof does not certify the later composite
objective at these new latent states.

## Implemented attribution replay

The reusable `failed_gate_attribution` diagnostic reloads each
exact saved state, mask, context and detached target against authenticated
source079 mathematics. The completed source080 replay compared three repeated baseline forwards,
the same endpoint component/repair slopes in gradient-enabled and no-gradient
execution, and analytic gradient algebra. It must retain every row/direction
and signal/rounding diagnostic. The CPU fixture gate and all three native
replays completed. The failed FD patterns reproduced; their numerical cause
is not established and the original gates remain failed.

There were **no DeFoG calls or optimization updates** in attribution replay:
the stored target is held fixed. That isolates scorer/decoder numerical behavior
from stochastic target generation. Confirming a cause requires those measured
comparisons, not choosing a larger epsilon, clipping gradients or suppressing
failed checks. Production recovery remains a separate decision after the
evidence is available.

### Authenticated replay plan and submissions

The metadata-only HPC [plan producer](../outputs/hpc-submissions/prepare-source080-failed-gate-plan.py)
has SHA `3c1bdea0f665a9e5c9247ddd2932ac326998b9e2af306e66c8d9b3d32e0b06d3`.
Downloaded [plan.json](../outputs/hpc-submissions/source080-failed-gate-plan.json)
has SHA `9170bf5ad88655ff75719f85c508bbda4ba1b08712521d0dcf4c48d41fa41283`;
remote path is
`/work3/s234818/latentmol/submissions/source080-failed-gate-attribution-attempt001/plan.json`.
It declares exactly prior0/step20, prior3/step50 and prior5/step20 in the
fixed-initial-mask arm, with **nine authenticated saved assets per case**,
including the saved latent/mask state, gate, detached target response/refresh,
weight receipt, original start and parent configuration/history. It pins
the original source079 release/archive/manifest; no assets are regenerated or
substituted. Constructing and downloading this plan is not a native replay.

The new reusable [attribution module](../src/latentmol/boltz/failed_gate_attribution.py)
and [staged submission helper](../scripts/hpc/submit_failed_gate_attribution.py)
are sealed in source080. Resources are CPU fixtures first
(30-minute gate), then one **four-hour 80 GB A100/MPS diagnostic allocation**,
four slots/64 GB host memory, for all three saved states. Actual source080
identities and receipts are recorded below. A temporary pre-authentication SSH
timeout during plan-producer upload was followed by one successful bounded
retry after the user reported working access; no VPN/key settings changed.
No production objective, tolerance or validation cadence changed.

### Source080 release and engineering evidence

[Sealed source080 archive](../outputs/source-releases/source-080-failed-gate-attribution-attempt001.tar.gz)
SHA is `7f2d7dee4cfd734ee604d501c8f5697a08991019714af82d9747e7f433a9fd0a`;
manifest `10e26f5dd0bac68e5a416dc267c31cc8ccd936c3471dcd22ff887f48a3502b31`.
Successful metadata staging extracted 423 members after one retained
authentication-rejected SFTP attempt and a successful bounded retry.
[Identity](../outputs/hpc-submissions/source080-identity.json),
[CPU receipt](../outputs/hpc-submissions/source080-cpu-receipt.json),
[native receipt](../outputs/hpc-submissions/source080-native-receipt.json),
[CPU run](../outputs/hpc-results/source080-cpu-run.json) and
[CPU stdout](../outputs/hpc-results/source080-cpu-29661601.out) retain exact
commands, environment/source/config identities and **31 passes / 12.94 seconds**.
The verified [source080 capsule](../outputs/hpc-results/source080-attribution-001.tar.gz)
SHA is `48ae1d50e8ad617f3afd3e3befaf229697f7bc6bf3b673c524553a695a9baf73`,
**16 safe members / 15 verified files**, exported **9 October 09:32:53 UTC /
11:32:53 DTU-local**. Its [verification receipt](../outputs/hpc-results/source080-attribution-001.tar.gz.verification.json)
and exact ledger command establish local byte/hash/metadata-only scope;
the byte-only login-node collector performed no model inference. The capsule
authenticates the CPU/source/log/receipt evidence and queued GPU snapshot.
In that historical queued snapshot, CPU **29661601 DONE** used 20 seconds
allocated time; native **29661602 PEND** had fulfilled `done(29661601)` with
a four-hour cap, pending host slots. The completed replay capsule linked at
the top supersedes this scheduler snapshot.
Exact file hashes are in the ledger. Local static AST/metadata checks do not
establish numerical native behavior. No production policy or scientific
source079 mathematics changed and no full optimization was resubmitted.

## Final cohort classification, 10 October

The later [complete source079 capsule](../outputs/hpc-results/source079-long-run-investigation-002.tar.gz)
authenticates all six exit-1 failures: contact guards at prior 0/20, 3/50 and
7/50; total guards at prior 5/20, 8/60 and 9/20. Prior 8/9 had no repair target:
all 20 prescribed requests had unsupported formal charge and were rejected
before native DeFoG inference, with the explicit Boltz-only fallback. Their
failed total guards are not evidence of a DeFoG target gradient failure.
Prior 6's timeout occurred during line search at state 56 after 56 accepted
updates, not in gate 60. The other timeout cases 1/2/4 reached 60 updates and
were interrupted in that full gate. Full counts, rejection categories,
native costs and provenance are in the
[comprehensive diagnosis](48_defog_comprehensive_diagnosis.md).

## Evidence and provenance

[Investigation capsule](../outputs/hpc-results/source079-long-run-investigation-001.tar.gz)
SHA-256 is `7c46d0425314d2ed7a93cdad35328334a847f99ef802d018b480cdb6969e4651`.
It contains **1,670 safe members / 1,669 individually verified files**, exported
**2026-10-09 08:57:40.214831 UTC / 10:57:40 DTU-local**. The
[verification receipt](../outputs/hpc-results/source079-long-run-investigation-001.tar.gz.verification.json)
records Python 3.12.14 / Windows 11 and local byte/hash/metadata-only inspection,
without numerical tests or model imports. Native numerical work was already
performed in the source079 HPC allocation; this report does not rerun it.

Direct records include the
[controller](../outputs/hpc-results/source079-long-run-investigation-001/campaign/controller/20261008T221046Z_ramped-defog-random_156be604/campaign.json),
[prior-1 execution](../outputs/hpc-results/source079-long-run-investigation-001/campaign/controller/20261008T221046Z_ramped-defog-random_156be604/prior-1-execution.json),
[prior-2 execution](../outputs/hpc-results/source079-long-run-investigation-001/campaign/controller/20261008T221046Z_ramped-defog-random_156be604/prior-2-execution.json),
[prior-4 execution](../outputs/hpc-results/source079-long-run-investigation-001/campaign/controller/20261008T221046Z_ramped-defog-random_156be604/prior-4-execution.json),
[prior-0 failed gate](../outputs/hpc-results/source079-long-run-investigation-001/campaign/prior-0/optimization/20261008T232742Z_ramped-prior-0_58408977/fixed_initial_mask/gate-0020.json),
[prior-3 failed gate](../outputs/hpc-results/source079-long-run-investigation-001/campaign/prior-3/optimization/20261009T032437Z_ramped-prior-3_96a249c9/fixed_initial_mask/gate-0050.json)
and [prior-5 failed gate](../outputs/hpc-results/source079-long-run-investigation-001/campaign/prior-5/optimization/20261009T063410Z_ramped-prior-5_90c97914/fixed_initial_mask/gate-0020.json).
Archive/manifest identities remain source079
`744862c9a310f87ff078edb2ab3185fd2468eaa00339222d024c1ffe0ceebcce` /
`21025b7bdb84ceeaf515ebe885ec0941491acf797a6ada66ffa02d73f1a5f29f`;
protocol `0ad8eb7e000d9e653ad880c17f308f0fe668878e6811626d7c3a2ed2c2b46f1a`.
Source080/job identities above supplement this preserved source079 evidence;
the original failures have not been reclassified or overwritten.
