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

The next explicitly authorized [full-model study](runs/20261025-chemistry-full-model-01/README.md)
combines all four candidates in matched original/combined Protenix and Boltz
validation pilots. Original model/optimizer backends, settings and gates stay
unchanged. Four persistent CPU workers handle independent native chemistry
checks; only differentiable local-risk arithmetic is converted to GPU matrices.
Integrated gradient correctness and actual update speed remain unproven until
those allocated pilots pass. Existing production studies remain untouched.

Fresh25October17:28UTC terminal evidence shows both full-model pilots failed:
Protenix's extra paired check rejected exact saved probability replay; Boltz's
combined native worker could not acquire CUDA after the separate chemistry GPU
service started. These are current integration/replay blockers, with no passed
combined comparison. The earlier CPU284 pass and Boltz serial native readiness
remain valid. Neither randomPro50k+chemistry nor Boltz30k+chemistry production
is submitted. See the full-model run's authenticated failure record; production
also requires explicit adoption and full future chemistry storage admission.

At25October18:02UTC, the Boltz startup fix passed actual two-CUDA-process admission and the full original baseline/combined native comparison (GPU29816184). Pro source02 stopped on an atomic metadata rename race; unchanged numerical source03 passed388CPU cases and ProGPU29816284 is running. See [retry02](runs/20261025-chemistry-full-model-02/README.md) and [Pro retry03](runs/20261025-chemistry-full-model-03/README.md). The [production adopter](runs/20261025-chemistry-production-01/README.md) has an actual queued CPU/callback chain that waits both pilots, all94contracts, actual startup, original readiness and complete future storage. Neither chemistry production run has been submitted yet. No saved results/settings changed.

At25October18:36UTC, Pro retry03 GPU29816284 is a preserved exact cross-process
probability replay failure after all low/high pilots and immediate canaries.
Bol retry02 full native comparison remains passed. Production01 stopped at
metadata startup admission (94 fixtures passed; overall CPU/callback EXIT1),
with no long submissions/full chemistry storage verdict. The separate
[replay-only diagnostic](runs/20261025-chemistry-replay-01/README.md) passed60
allocated CPU contracts and queued GPU29816561. It recomputes original saved
logits with predeclared import/compiler controls; no model or optimizer evaluation
and no acceptance-tolerance/probability substitution. Monitor remains paused.

At26October02:15UTC the original replay diagnostic01 is a preserved tooling
failure: its final source hashing collided with its checkpoint-loading guard.
New [replay02](runs/20261026-chemistry-replay-02/README.md) passed67allocated CPU
checks including actual audit-hook regression, then submitted GPU29818513.
Original eight saved points/class/settings remain unchanged;11predeclared
compiler/import controls each run twice. No new GPU finding or accelerated
Protenix/native/production pass yet. Bol02's successful comparison is retained.
Both chemistry production submissions still require validated Protenix replay/
full-model gates and the complete future-byte admission. Monitor stays paused.

At26October03:06UTC, [replay02](runs/20261026-chemistry-replay-02/README.md)
GPU29818513 passed all22controls with original inputs restored. Autotune-disabled
compiler context alone stabilized every tested original replay fingerprint across
four fresh processes; hashseed alone did not. This qualifies a candidate only.
[Full-model05](runs/20261026-chemistry-full-model-05/README.md) CPU408 passed;
new ProGPU29818553 PEND. Failed04 fixture-only attempt remains preserved.
[Production02](runs/20261026-chemistry-production-02/README.md) CPU29818556 and
callback29818557 are actually queued, requiring fresh Pro native pass, retained
Bol02 pass, all103fixtures/actual startups and full future storage before either
chemistry long. No new chemistry optimization has started; monitor stays paused.

26October09:20UTC: fresh Protenix full-model05 GPU29818553 PASSED its complete
original native/chemistry/paired replay+gradient/runtime/frozen checks; original
same-job LSF and141compact files verified. Bol02's native pass remains valid.
Production02 CPU29818556 and callback29818557 then EXIT1 at inherited entry
storage before103fixtures/startups:36.06GB required vs34.11GB conservative
available, at least1.95GB short BEFORE adding full new chemistry projections.
No chemistry production long was submitted or started. Failed receipts retained;
settings/saved results unchanged, no further cleanup or submission this status.

26October10:32UTC: [safe archive recovery](runs/20261026-chemistry-storage-recovery-01/README.md)
completed69,914exact backed-up files/8.912GB. Current2176input/proof bindings,
sources/models/environments/other projects preserved; original interrupted stream
remains failed, only fully verified members offloaded. [Production03](runs/20261026-chemistry-production-03/README.md)
then passed106allocated cases, both actual startup commands, both native/compiler
adoptions and complete full-future storage49.553GBrequired/50.512GBavailable.
Protenix50k LR0.3 job29820255 and Boltz+chem30k LR0.0025 job29820257 submitted
independently, both PEND; own callbacks29820256/58. First10k segments only so far,
validated serial successors target5/3segments. No optimization start established.
Settings/gates unchanged, historical values byte-preserved, monitor paused.


The user authorized a [Boltz-only startup recovery04](runs/20261026-chemistry-production-04/README.md)
on26October. Its active local source03 adds the missing acceleration parent,
actual adapter-entry regressions, complete pre-import authentication and explicit
Protenix execution/submission blocks. Existing Protenix03 completed its first10k
segment and remains independent. Both complete futures/control caps and original
scientific gates are retained. The122-case candidate is only statically prepared;
SSH timed out before input collection, so no new remote staging, validation or
submission exists. Do not replay production03 or launch another Protenix chain.


27October00:21Paris: startup recovery04 passed122allocated cases, actual
startups/native adoptions/full dispatch+entry storage, then Boltz chemistry
29825209 is actualRUN at686committed updates/687states; own callback29825210
PEND. Original gradient entry/coupled reports passed; target30k unchanged.
[Bounded storage recovery](runs/20261027-repair-storage-01/README.md) verified
and offloaded175MB of previously approved inactive trial copies, but separate
plain Protenix checker05 still cannot meet its full future budget. Existing
randomPro03 first10k remains passed; its next boundary check failed and was
not restarted. Monitor remains paused, saved scientific values/settings unchanged.

27October storage readmission: [deduplication02](runs/20261027-repair-storage-02/README.md)
backed up every unique blob/path mapping before atomically consolidating819
identical read-only source copies. All131 full manifests/paths/bytes/modes remain
exact; duplicate content totals2.727GB (not exact allocated-block measurement).
Fresh full precision05 admission passed49.855GBrequired/50.922GBavailable with
all futures/4GiB/100k reserves retained. Boltz chemistry29825209 lastRUN2445
updates at00:17UTC; its running chain was untouched. No further cleanup or
monitor resumption. Endpoint checker jobs/results belong to endpoint-qualification.
