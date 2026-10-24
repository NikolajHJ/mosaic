# Scheduler-lag GPU admission correction

Current status: metadata CPU/callback passed; GPU 29809554 queued.

Prepared 25 October 2026 after source02 CPU 29809477 actually completed and
passed 143 fixtures plus all six saved-input comparisons. Original callback
29809478 failed because its ended dependency fired while bjobs still returned
RUN for the CPU job. The exact raw RUN snapshot, failed callback and later
actual DONE/LSF success are retained; this is an orchestration failure.

This new immutable metadata-only capsule adopts those original allocated
checks without replaying them. Its own allocated CPU regression fixtures
exercise bounded RUN/PEND-to-DONE waiting, raw proof preservation, wrong-job,
wrong-queue, EXIT, malformed and timeout rejection. A separate own-ended
callback requires actual new CPU DONE/hpc/LSF success and raw case counts,
then repeats the unchanged full selected-campaign storage gate plus the
existing joint 1 GiB benchmark allowance/4 GiB and 100,000-file reserves.

Only then may it submit the original source02 GPU benchmark with the exact
original argv/source/plan/19 GPU-only checks. No scientific source, tolerance,
finite-difference ladder, input, search budget, saved result or existing
optimization job changes. The original failed callback remains failed.
Plan bytes are now explicitly pinned before admission; all original 143
checks and benchmark/source/input/LSF proofs are reauthenticated.

CPU: two hpc slots, 2 GiB/slot, 30 minutes. Callback: one hpc slot,
2 GiB, 20 minutes. GPU is still one exclusive L40S/4 slots/8 GiB per slot/2 h,
with no scientific optimizer or native-model execution. All numerical work
remains allocated HPC; local work is stdlib AST/bytes/packaging/evidence only.

Local entry point: scripts/gpu_adoption_launch.py (one shot; inspect actual
markers before any recovery). Remote control root is the analysis campaign's
chemistry-search-speed-gpu-adoption-03; source is the distinct read-only
chemistry-search-speed-gpu-adoption-release-03. Actual jobs/hashes/outcomes
will be recorded here. Do not replay source02's failed callback or bootstrap.

Actual outcome at 25 October 01:29:38 Paris: CPU 29809552 and callback
29809553 are DONE/hpc/LSF success; all 19 raw metadata fixtures passed.
Independent source/byte/XML/LSF/dispatch audit passed. Source5 manifest
00e223b9b453aa0e390f315951eee3406916ac8c8499284945967a72fbf38698,
archive9050 bytes SHAae2031e920ec7d40058a04215fd4c3bb767e16832672538f12e11b9a5564348b.
Initial staging/preflight stopped before any job on a concurrently replaced
active trial .partial file. That failed full observation is preserved. A
separate fresh complete unchanged gate passed before first CPU submission.
Actual GPU 29809554 is submitted under the original source02 runtime and argv;
PEND on physical GPU availability. Its dispatch storage required32,815,757,995
versus conservative39,303,877,607 bytes, retaining all original projections
plus1GiB benchmark allowance/4GiB and100,000-file reserves. No reservation
or GPU execution/pass is claimed. Exact original GPU entry,19checks,FD ladder
and tolerances remain unchanged. Never replay any of these submissions.
