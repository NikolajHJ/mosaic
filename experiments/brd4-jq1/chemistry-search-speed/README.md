# Chemistry preparation speed study

The user authorized separate tests on 25 October 2026 to substantially reduce
detached chemistry repair/search time. Current optimizations, saved results,
objectives, coefficients and gradient gates stay unchanged. This study cannot
automatically promote an implementation into an existing sealed campaign.

The difficult random N25 pilot repeatedly spent 18–19 seconds preparing a
reference, exhausting 96 original plus 192 progress checks. Valid chain inputs
usually skipped search. This motivates three independently measured approaches:

1. Cache exact hard-graph chemistry facts across overlapping searches/updates.
   Never cache a probability-dependent target under only its argmax identity.
2. Hoist invariant probability validation and repeated cost bookkeeping, keeping
   the same scalar arithmetic, candidate ordering and scientific checks.
3. Evaluate independent candidates within a depth with a persistent process
   pool, then reduce in original order. Search depths/beam decisions stay serial.

Measure each and their supported combinations against the original serial path.
Record cold setup separately from warm search. Exact preparation receipts,
target arrays, status/check counts and identity must match, excluding timing;
loss/components and gradients must match on the same frozen reference. Test
invalid input handling, cache invalidation and pool restoration. Profile baseline
and candidate paths. Speed findings apply to the sampled saved graphs, not an
entire future optimization. No reduced check budget or approximate stale target
is part of this first experiment.

The original chemistry files remain read-only inputs from authenticated
`protenix-extended-release-03` (643 members) on DTU HPC. Use exact saved pilot
snapshots from the random Protenix and Boltz readiness runs, not invented
intermediate molecules. Numerical fixtures, search and gradient comparisons run
on allocated HPC CPU; no native model inference is required for this detached
loss benchmark. GPU acceleration is not presumed useful for RDKit/Python search.

`scripts/` owns experimental adapters, benchmark and launch/evidence tools;
`tests/` owns allocated fixtures; `runs/` records immutable attempts. Commands,
input hashes, resources and outcomes are in each run record. The monitor remains
paused and existing jobs are not changed.

Parent benchmark: [20261025-chemistry-speed-01](runs/20261025-chemistry-speed-01/README.md).

Further user authorization25October: [attempt02](runs/20261025-chemistry-speed-02/README.md)
tests a combined central-cache/parallel-worker search and real GPU matrix/scan
loss kernels. Phase01's verified results and immutable source stay preserved;
phase02 requires its own allocated CPU and GPU evidence before any claims.

At 25 October 01:29 Paris, phase02 CPU passed all 143 fixtures and exact
six-input comparisons. Warm difficult preparations took about 0.28 seconds
versus 16 seconds serial; first cold search still took 7.6–8.7 seconds.
See its [measured results](runs/20261025-chemistry-speed-02/RESULTS.md).
GPU benchmark 29809554 is queued after the separate
[validated admission correction](runs/20261025-chemistry-speed-gpu-adoption-03/README.md).
At that observation no GPU speedup or production adoption was established.

GPU 29809554 subsequently ran 05:57–06:03 Paris/CET on 25 October and passed
all 19 GPU-only checks. The [verified GPU results](runs/20261025-chemistry-speed-02/GPU_RESULTS.md)
show matrix local-risk value/gradient execution 6.16 times faster at batch 1,
3.77 times at batch 32, and 24 percent slower at batch 256. Scan warm execution
was slower throughout. These are GPU component gains; CPU RDKit search remains
separate and no production run has adopted the candidates.
