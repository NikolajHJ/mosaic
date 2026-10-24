# Combined parallel reuse and GPU matrix study

Current status at 25 October 01:29:38 Paris: CPU 29809477 passed all
143 fixtures and the exact saved-input comparisons. Its original callback
29809478 remains failed on scheduler lag. Separate admission03 passed19
metadata fixtures/actual CPU and callback completion, then submitted unchanged
GPU benchmark29809554. GPU is PEND on physical availability, without execution
or performance results yet. See the admission03 run for its independent proof.

The user explicitly requested further optimization, more parallelism and GPU
matrix conversions. Existing saved runs, scientific sources/settings and the
paused monitor remain unchanged. This is a separate engineering experiment.

The verified source01 result (CPU29809404,79passes, exact six-input comparison)
is the parent. Its cold profile assigns most time to native RDKit diagnosis,
which is not a GPU tensor operation. New candidates:

- One parent-owned bounded cache, with persistent4/8spawn workers evaluating
  only unique native hard-graph misses, original ordered search reductions and
  refreshed probability-dependent costs. Logical checks/targets stay exact.
- JAX batched integer matrices for components, padding, symmetry and valence
  counters. These are features, never substitutes for RDKit validity or pruning.
- JAX scan and balanced log-semiring transition-matrix versions of the existing
  local valence-risk dynamic program. Matrix association changes FP32 order;
  original2e-5absolute/relative value/component/gradient tolerance and original
  FD ladder0.01/0.003/0.001,15percent/twoadjacent remain required. No weaker gate.

Allocated CPU stage:16slots/2GiBslot/2h, original79 plus26hybrid,20matrix and18
oracle-profile cases (143total), no errors/failures/skips; compare original,
source01combined and hybrid4/8 on six original saved pilot inputs. Targets,
complete receipts, scalar loss/components and gradient hashes must match
exactly for these CPU search adapters. Separately profile the actual native
DetectChemistryProblems/SanitizeMol calls with exact arguments/return/errors.

Only an authenticated actualCPU DONE/hpc/LSF success/raw143case assessment can
admit the separate GPU benchmark. Own-ended CPU callback verifies these and
fresh whole capacity before submitting one GPU job. GPU stage:oneexclusive
L40S48GB/fourCPU/8GiBslot/2h, actual JAXGPU device required (no CPU fallback).
Compare original versus scan/matrix kernels on frozen CPU-prepared references
for six saved graphs, then synchronized matrix/gradient timings at batches
1/32/256. Transfers, first compile and warm execution are reported separately.
No model inference, optimizer updates or target search is run on the GPU.

Both phases reauthenticate the original643scientific source and source01's
nine-file seal/actual passed raw proofs/input hashes. Persistent output/source
allowance is1GiB total, independently added to the unchanged whole selected
campaign gate with4GiB/100000file reserves. Exclusive source/submission markers
prohibit blind replay or duplicate unknown outcomes. CPU failure blocks GPU.
Each release is immutable; preserve failed candidates and all original results.

Commands: bundled Python `-X utf8`, Mosaic working directory; scripts
`round2_prepare.py`, `round2_launch.py`, `round2_observe.py` and
`round2_verify.py` own packaging, one-shot staging, read-only delivery and local
stdlib evidence audits. All numerical work is allocated DTU HPC only. Actual
source/input/resource/job IDs and outcomes will be recorded before claims.

Actual submission25October01:15:35Paris: CPU29809477 andownendedcallback29809478,
source19files09c0ab837bede0c5779eac0da06885cfbef6a98744f67af19e7b5a8b33ffa222,
archive35875bytes SHAe079f5a30c4e062b2d8dee04311824425a593d16278c3c378d592dd46d127432.
Whole unchangedcampaign plusjoint1GiB passed32836817040required/39325369650
availablebytes, notreservation. At01:15:54CPUactualRUN/callbackPEND, no fixture
pass, GPUjob orGPUresult yet. Exclusiveparents/receipts prohibit duplicatejobs.

Static review found two reporting/provenance limits to retain in this experiment:
the worker does not compare the mutable plan file with its initial receipt SHA
at runtime, so the independent delivery audit must authenticate that exact SHA,
all six input/reference hashes and all declared settings. Hybrid nested
`cost_reuse` counters in the final report are context-final totals because of
a shallow stats copy; they are not per-input counter snapshots. Top-level
cache/worker counts, elapsed times and exact numerical comparisons are separate.
The extra oracle profile covers only cold input 0 under baseline/combined;
it does not measure hybrid worker-internal RDKit calls. These limits do not
authorize editing the sealed source or weakening any numerical gate.

At 01:23:21 Paris the original CPU job was actual DONE/hpc/LSF success.
Independent source/archive/byte/XML/schema audit verified all 143 cases, all
24 matched rows/72 preparations and exact scalar/component/gradient hashes.
CPU assessment SHA51fccda60b2178d39fb9ae90067d1ea9062c05d0560ec494dae41272ea33ec34.
Callback 29809478 exited 1 because its first actual bjobs query still showed
RUN immediately after ended fired. Its raw query/failure remain unchanged.
The numerical CPU results passed; no GPU job followed that failed callback.
Separate metadata-only admission03 retains the passed source02 GPU code/argv,
adds bounded actual terminal waiting and adopts these same proofs after its
own allocated fixtures/terminal checks and fresh unchanged full capacity.
