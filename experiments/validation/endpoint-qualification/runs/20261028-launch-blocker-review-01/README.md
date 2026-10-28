# Launch blocker and queue review, 28 October2026

The user asked why requested runs have not started, whether another GPU queue
could help, and whether submission scripts have mistakes. Investigate current
queue/resource metadata, exact failed Protenix/DBMol source15 evidence, chemistry
and plain-Boltz continuation blockers, without changing jobs or saved results.

Three read-only parallel reviews cover Protenix CUDA process ownership, GPU
portability/resource contracts, and full-storage/chemistry prerequisites. All work
is source/standard-library metadata inspection. No model import, numerical test,
GPU probe, resubmission, cleanup or monitor resumption belongs to this review.

Current failed source15 manifest:
`5448503965365f4db726d558df4ac91a41a7f711366b6ae47ae936993fd0c8bb`.
Original failures and source remain intact. New remote resource-admission
receipts, if collected, use a fresh separate audit directory and the authenticated
unchanged source15 full-capacity guard. No old failed gate is overwritten.

Evidence is under `results/hpc/20261028-launch-blocker-review-01/`. Existing
independent learned-prior optimization and other projects were inspected through
scheduler metadata only and never changed.

## Findings

The requested continuations are not pending optimization jobs. Source15's Pro
and DB native checks obtained L40S GPUs after about56--57minutes, then EXIT1.
Their callbacks correctly submitted no child. Waiting alone will not start them.

### Protenix process-ownership integration

`scripts/precision_continuation_ops.py:235--239` requests one GPU with
`mode=exclusive_process`. Production has already initialized its CUDA context
when `precision_continuation_hooks.py:402--410` launches a second live FP64 JAX
reference process. `Session` requires production/reference co-residency, inherits
the same visible GPU, and does not request MPS. The reference dies at its first
`jax.devices("gpu")` before model loading, with CUDA_ERROR_DEVICE_UNAVAILABLE.
This is strongly consistent with the exclusive-process ownership conflict;
the actual CUDA compute mode was not independently probed in this review.
It is not an observed new gradient disagreement or OOM.

The earlier source09 `continuation-handoff.md:13--20` explicitly warned about
this integration boundary. Source15's592 CPU fixtures/startup routes passed but
the real-command checks deliberately disable CUDA (`precision_continuation_ops.py`
896--907), and the co-residency fixture substitutes a ToySession. Thus the CPU
pass never established two simultaneous real CUDA clients.

DTU documents one job-exclusive GPU serving several same-job processes with
`num=1:mode=exclusive_process:mps=yes`:
https://www.hpc.dtu.dk/?page_id=5150 . This is a candidate resource-only fix,
not a validated remedy here. An allocated two-client ownership/memory startup
probe must precede repeated full native/reference/replay/frozen/pilot gates.
Do not use inter-user shared GPU access or merely remove device/queue assertions.
The alternative is serialized CUDA children with exact graph/key/gradient transport
and pending-external-reference semantics as the source09 handoff recommends;
that needs a new validated integration policy, not a provisional pass.

### DBMol and current storage scope

The failed DB entry receipt required53,811,657,553 bytes versus53,631,344,950
available. Model evaluation never began. A fresh authenticated source15 metadata
gate now passes after later offload: the same requirement versus55,100,648,609
available, retaining4GiB/100,000-file reserves and all original forecasts. This
does not submit a job, reserve space or authorize reusing the old failed parent.

However, source15 predates the RUN Protenix+DeFoG30k campaign and does not fund
its output future. The new campaign separately retains the same source15 baseline
plus its4GiB output cap and64MiB release allowance, with130,000-file headroom.
The new reusable report cache also needs conservative stale-quota accounting.

The explicit known-scope review retains the teacher's ENTIRE4GiB cap rather than
crediting writes of unknown quota visibility, counts the baseline/reserve once,
keeps64MiB release allowance, and subtracts the entire400,081,714-byte new report
tree plus308,906 review bytes. It finds58,173,733,713 required versus54,700,257,989
available:3,473,475,724 bytes (~3.24GiB) conservative shortfall. This is a
diagnostic composition of declared scopes, not a newly qualified production
admission certificate or proof of physical quota exhaustion. Additional run-level
logs/control writes outside the teacher's pilot cap require explicit accounting.

No arithmetic bug was established in source15's inherited guard. Its cumulative
historical proof allowances are conservative and can only be retired with exact
terminal/backup/provenance evidence in a new validated accounting release.
A static risk exists in the teacher's periodic guard: it fetches quota without
establishing freshness, credits already-written pilot bytes, and excludes
run-level logs from that pilot-tree count. This is not evidence of an actual quota
breach; it reinforces the need for one current cross-campaign capacity ledger.
The running study was not changed or stopped.

### Other scientific blockers

- Plain Boltz already saved20k/30k, but its20k terminal prospective endpoint
  admission failed. No20k->30k successor exists. A queue change cannot bypass
  that saved-endpoint validation failure.
- Chemistry qualification04 deliberately has no optimizer branch. Bol's native
  component passed; Pro's native qualification failed. Both still need combined
  low/high/actual-coefficient checks, BOTH ten-update pilots, replay, runtime,
  startup and fresh whole-storage admission before conditional continuations.
  Bol chemistry can qualify independently rather than waiting for Pro chemistry.
- Source15 Pro/DB callbacks already operate independently; there is no unnecessary
  inter-arm scheduler dependency to remove.

## Queue snapshot and eligibility

Actual LSF metadata at14:09Paris,28October2026. `PJOBS` is pending job count;
slot totals and reservations are not interpreted as GPU counts or wait estimates.

| Queue | Pending jobs | GPU devices in sampled batch nodes | Free GPUs | Access |
| --- | ---: | ---: | ---: | --- |
| gpul40s |73|12|0|all users|
| gpua100 |339|20|0|all users; mixes40/80GB|
| gpuv100 |400|38|0|all users; mixes16/32GB|
| gpua10 |64|2|0|all users|
| gpua40 |70|not sampled|not established|all users|
| gpuh100 |21|8|0|restricted; this account absent from expanded allow-groups|

Empty/small departmental queues were also checked. `cabl40s`, `cabgpu`,
`gpukta100`, `stenogpu`, `fotonanogpu`, `mementogpu`, and course queue `gpu02510`
are restricted; this account is not in their expanded allowed user/group lists.
They are not available shortcuts. There is no reliable current queue-time ETA;
different capacity, fair-share priority, reservations and job shapes matter.

The current source15 resource, startup, terminal proof and FP64 worker contracts
explicitly require gpul40s/L40S. Changing only `-q` fails those contracts. A100 or
other supported GPU use needs a new declared resource profile and actual-device,
exact replay, native derivative/parity/frozen/pilot/runtime/memory validation.
Host8GiB/slot times4 is32GiB host RAM, not requested VRAM. Shorter walltime for
backfill must follow measured full runtime including reference work; do not trim
the time budget or update target speculatively.

Every source15 segment currently requests23h unconditionally, even when the
measured conservative runtime bound is shorter. `precision_continuation_driver.py`
86--97 compares the bound against82,200s; it does not size `bsub -W`. A new
validated resource profile could request the measured conservative duration with
its full margins and matching internal deadline, improving backfill eligibility.
No successful source15 full native measurement yet establishes that shorter bound.
There is no demonstrated2h-vs10,800s deadline bug:10,800 covers entry AND terminal
references for a full segment; validation executes only entry with a6,900s shared
cap and a5,400s individual Pro reference cap. Its observed failure was CUDA
initialization, not exhaustion of these time limits.

## Recommended implementation order

1. Build a current combined storage inventory/remaining-futures ledger, retaining
   every active future and reserves. Offload only inactive exact-backed-up files
   if needed; never weaken the guard or delete protected scientific inputs.
2. Validate actual two-process CUDA ownership before loading full models, then
   qualify an approved exclusive-job MPS/resource profile or serialized boundary
   integration in a new immutable release. Preserve source15 and failed receipts.
3. Revalidate/retry DBMol independently once its own full guards pass. Pro's fix
   does not need to hold DB's admission. Finish chemistry's combined admission
   independently per arm and investigate plain Boltz's saved20k endpoint.
4. Add alternate GPU profiles only after native checks establish their suitability.
   Use measured safe durations for backfilling. Do not launch duplicate competing
   optimizations, bypass original checks, or change saved runs/LR/objectives.

No candidate fix, new GPU class, CUDA/MPS startup, numerical test or optimization
was validated/submitted by this review. All current jobs/settings/results remain
unchanged and the monitor stays paused. Verification and source/evidence hashes
are in `verification.json`.
