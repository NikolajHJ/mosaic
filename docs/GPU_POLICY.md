# GPU distribution policy for exploratory work

Default for new Mosaic exploratory batches, adopted 1 November 2026: **spread
independent runs across compatible GPU queues to get useful results sooner**.
V100 is one option among V100, L40S, A10, A100 and A40. Avoid making a busy queue
the bottleneck for the entire batch.

## Required GPU compute mode

DTU support ticket43078 (2 November2026) requires **exclusive compute mode**.
Every new GPU request must explicitly include
`num=1:mode=exclusive_process:j_exclusive=yes`. `j_exclusive=yes` alone only
prevents another job from sharing the allocation; it does not replace compute
mode. The shared helper now rejects shared-mode or duplicate/conflicting
overrides. Default single-client jobs use `mps=no`; a multi-client workload may
explicitly use per-job `mps=yes` while retaining exclusive_process, after checking
its actual startup path. Never change NVIDIA compute mode/daemons manually.

The device/memory profile IDs remain the same; the current execution default
is corrected. Old sealed sources, submitted vectors and results retain their
original bytes and historical meaning. Stopped jobs need fresh corrected
snapshots and reconciled checkpoint continuations, not edits to sealed/live
files or duplicate submissions. Do not restart completed runs.

## Placement

- Check current scheduler limits, eligible hosts, per-account usage and available
  capacity. Distinct queue allowances can increase concurrency. Do not invent a
  global five-job cap or treat an old observed limit as permanent.
- Assign each arm to a compatible device using memory, software, precision and
  measured runtime for the requested remaining updates. Aim for the shortest
  expected queue wait plus execution time across the batch. Queue lengths and
  idle GPUs outside eligible host groups are not reliable wait estimates; when
  waits are uncertain, spread suitable arms across independent queue allowances.
- Prefer available compatible capacity over accumulating every arm behind a
  saturated queue. Keep submissions within actual scheduler/user limits and
  use appropriately sized CPU/memory and measured walltime with reasonable
  setup/cleanup margin. Keep the largest justified remaining update budget that
  fits; do not introduce arbitrary short segments solely to spread work.
- Distribute separate arms/seeds across devices. Request multiple GPUs for one
  job only if its runner already uses them. Never submit duplicate copies of an
  arm as a queue race. Record the actual device/profile, source and timings so
  hardware effects remain visible in comparisons.

Shared profiles live in [gpu_policy.py](../scripts/hpc/gpu_policy.py):

| Profile | Queue | Device memory | GPU memory request |
| --- | --- | --- | --- |
| `v100-32gb-v1` | `gpuv100` | 32 GB V100 |`gmem=30G`|
| `l40s-48gb-v1` | `gpul40s` | 48 GB L40S |profile's existing resource string|
| `a10-24gb-v1` | `gpua10` | 24 GB A10 |`gmem=22G`|
| `a100-40gb-v1` | `gpua100` | 40 GB or 80 GB A100 |`gmem=38G`|
| `a40-48gb-v1` | `gpua40` | 48 GB A40 |`gmem=44G`|

New batch plans must set an explicit per-arm `gpu_profile`/queue before `bsub`.
The low-level `profile()` fallback remains V100 for older callers; it is not a
new-batch placement recommendation or an automatic queue balancer. Workers use
the submitted profile, rather than making a fresh hardware choice at startup.
Check current queue walltime limits; an unsuitable runtime or memory footprint
is a reason to use another compatible queue. Preserve required CUDA/software
and scientific precision; these profiles retain CUDA 12 compatibility.

## Proportional checks

Reuse relevant device evidence and the actual scorer/backward path. For
exploratory work, finite values, attached required paths, shapes, input/frozen
preservation, hardware identity and output ownership can be checked during
normal production startup and the first saved updates. Keep those updates as
ordinary results. A separate queued probe/controller chain is optional, not a
routine prerequisite for using another compatible GPU family. Use a short
bounded check only when a real compatibility uncertainty requires it; do not
add a native/FD/FP64 qualification campaign just to select hardware.

Keep objective, LR, precision, seed, chemistry schedule, model/context inputs
and optimizer settings unchanged. Record floating-point differences honestly;
GPU placement does not certify exact replay, derivatives or chemical quality.
A genuine finite/input/frozen/ownership failure still stops the affected run.
Explicit historical qualification contracts retain their original scope.

## Existing jobs

During authorized run management, reconsider never-started queued arms as
compatible capacity becomes available. Prepare an immutable replacement plan
and fresh outputs, authenticate the exact predecessor's state/history and
absence of saved progress, then hold/recheck before retiring that owned job.
Persist the accepted successor receipt before releasing it. If the predecessor
starts, keep it running. Reconcile every unknown outcome before retrying.

Include any owned, unstarted dispatch callbacks in the same reconciliation so
an old callback cannot create a second producer. Preserve unrelated callbacks.
Leave running jobs and their live dependencies, sealed sources, checkpoints and
saved results untouched. Do not change a live job's profile or create duplicate
writers. This policy update itself submits, cancels or migrates no existing job.
