# GPU preference for new exploratory work

Prefer a **32 GB V100 (`gpuv100`)** for new Mosaic exploratory jobs. The shared
defaults live in `scripts/hpc/gpu_policy.py`. Request `gmem=30G` so 16 GB V100s
cannot satisfy the request. Keep CUDA 12 for these Volta devices.

This is a preference, not a promise of the shortest queue. Use L40S when a model
does not fit, the software does not support V100, or measured execution time
makes L40S the better choice. Check eligible queue hosts rather than counting
idle GPUs outside those host groups. Pending-job counts are not wait estimates.

Explicit Ampere alternatives are available: `a10-24gb-v1` uses `gpua10` with
`gmem=22G`; `a100-40gb-v1` uses `gpua100` with `gmem=38G` and accepts40GB or80GB
A100s. They retain CUDA12, the requested scientific precision and job-exclusive
ownership with shared compute/noMPS. These profiles do not change the V100 default.
The `a40-48gb-v1` profile uses `gpua40` with `gmem=44G` for48GB A40s.
Check current per-queue account limits as well as available hardware; distinct
queue allowances can increase concurrency, but are not a guaranteed start time.

For a new device, use a short allocated smoke of the actual scorer and backward
path at the saved checkpoint; record memory and a few update timings. Keep the
objective, precision, seed, chemistry schedule and optimizer settings. Do not
introduce a full qualification campaign just to select hardware. Finite values,
attached required paths, input/frozen-model preservation and exclusive output
ownership still matter. Record floating-point replay differences honestly.

Changing this default does not migrate existing jobs. With explicit migration
authorization, prepare the replacement first and replace only an authenticated
unstarted job. Leave running jobs alone. If the old job starts during a device
test, keep it running. Preserve receipts, old outputs and saved checkpoints.
Never submit two active producers for the same continuation.
