# Matched full-model chemistry acceleration pilots

The user authorized combining exact native-fact caching, probability-cost reuse,
persistent four-worker ordered search and the GPU matrix local-risk kernel,
then checking actual full-model gradients and update timings for Protenix and
Boltz. This is a separate engineering validation, with no production adoption,
restart, saved-result modification or monitor unpause.

Protenix retains original random Dirichlet1 seed17, LR0.3, 50,000-update
configuration and delayed chemistry0.001 through5000 to100 at50000. Boltz
retains original chain/uniform0.5 seed17, LR0.0025, 30,000-update configuration
and the same schedule stretched to30000. Both use the exact original validation
driver, fixed single candidate and low/high disposable ten-update pilots. All
native hard parity, original raw FD0.01/0.003/0.001 at15percent/two adjacent,
null, independent primal, nonzero, frozen identity, coupled and runtime checks
remain required. A baseline runtime forecast failure is preserved as that
failure after its scientific checks pass; the candidate must pass all checks.

Each model runs the original baseline and combined candidate in fresh sequential
processes within one allocated L40S job. Same-point comparisons use the baseline
low/high start/end snapshots and frozen preparations. Exact initialization/key
and frozen model/context identities are required. Matrix arithmetic is checked
at the existing2e-5 tolerance and is not claimed bitwise identical. Extra paired
checks are outside authoritative update timers. Cold/compile and subsequent
updates are reported separately; component speedups are not multiplied.

Protenix keeps its original GPU JAX optimizer. Boltz keeps its original CPU JAX
optimizer and CUDA frozen PyTorch model worker; only chemistry local-risk
arithmetic uses a separate GPU service with a fresh custom VJP evaluation.
Native oracle decisions remain native CPU checks. Four search workers remain
persistent, ordered and bounded; no probability-dependent scientific result is
reused by the transport.

First an allocated CPU suite repeats the original143 adapter fixtures plus new
full-model policy/startup/bridge contracts. Actual same-job DONE/hpc/LSF success
and exact raw XML counts are required before two independent GPU submissions.
Each GPU allocation has8CPU slots,4GiB per slot, one exclusive48GB L40S,2h;
an internal6600s bound and separate output caps remain mandatory. Each entry
reauthenticates all original sources/configs/CPU certificates/input proofs and
fresh unchanged whole-campaign storage plus2GiB for this study, retaining the
original4GiB and100000-file reserves. No cleanup is part of this task.

DBMol compatibility was inspected read-only: its current VAE already uses12
persistent chemistry workers, cost/template reuse, ordered CPU/GPU overlap and
GPU target NLL. Its active frozen-reference objective has no Mosaic local-risk
DP to replace. An exact native-fact cache would require a separate validated
study preserving fresh classification and all VAE health/RNG/identity checks;
no DBMol runtime, release, job or saved result is changed by these pilots.

Preparation is local standard-library AST/byte/JSON only. Numerical tests,
model evaluations and pilots run only on allocated DTU HPC. Actual source,
plan, submission, raw scheduler/LSF and assessment hashes will be recorded here.
No allocated full-model pass or speedup is established at preparation time.

Actual25October10:18Paris/CET: immutable26-file capsule staged and authenticated,
manifest347e753d481ec3d584f3b0b05cc2801ee745036ee17638d7c3672fc2ca2cba16;
archive71952bytes SHA4f5ec7f81f39a67667fa254321e47795d215c26ee537cac5ed280bade320366a.
CPU29811875 RUN and own-ended callback29811876 PEND at10:18:45. Planned284cases;
no raw suite pass or full-model GPU job is established yet. Fresh unchanged
whole-campaign plus2GiB preflight passed33643657204required/40125878917available
bytes; this is not a reservation. Both8-slot GPU pilots wait for actual CPU
completion and fresh dispatch/entry storage. A local unsubmitted first package
is retained; its only replacement change fixed a synthetic tuple/list receipt
comparison before any staging or numerical execution. The monitor stays paused.
