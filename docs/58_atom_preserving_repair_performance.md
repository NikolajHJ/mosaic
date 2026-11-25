# Accelerating the unchanged atom-preserving repair teacher

**Latest status, 17 October 2026 22:57 DTU-local:** source121 attempt002 replaces
the intentionally retired pending source118 suite. CPU29747653 passed715 checks
(four CUDA-only skipped) and its frozen-parent CPU repair profile;16-slot,
12-worker native GPUbenchmark29747654 and promotion29747655 remain PEND.
The current five candidates defer20/28-worker cases and cap future allocations
at24h/23h internal. No native speed or training result is claimed. Fresh quota
must pass: latest9.85GiB free is below the10.125GiB training reserve.
[Verified evidence](../outputs/hpc-results/source121a002-first-suite-002/local-verification.json)
and [current execution plan](61_chemistry_execution_selection.md).

**Historical status, 17 October 2026 21:03 DTU-local:** source118 GPU29745548/
callback29745549 remain PEND; the completed CPU/profile proof is unchanged.
No native candidate timings, speed selection, ratio adoption or training results
exist yet. Current eligible-host capacity is full/reserved, with no start ETA.
This was read-only and changed no VAE jobs/settings.
[Verified capsule005](../outputs/hpc-results/source118-first-suite-005/local-verification.json)
and [ledger](05_experiment_ledger.md).

**Source118 CPU/profile passed, 17 October 2026 13:41 DTU-local:** CPU29745547
DONE, 669 passed/four CUDA-only skipped in70.48s and profiler returncode0 with a
persisted passing CPU gate. GPU29745548/callback29745549 PEND. The CUDA checks remain
mandatory in the GPU stage. The sole source117 fixture
setup fix retains all assertions, production science and 13 configurations.
No native timing, performance result, chosen ratio or training start is
established. Source117's blocked jobs are confirmed owner-retired, with its
failure preserved. [Verified capsule003](../outputs/hpc-results/source118-first-suite-003/local-verification.json),
[docs61](61_chemistry_execution_selection.md) and [ledger](05_experiment_ledger.md).

**Historical source117 fixture failure, 17 October 2026 13:33 DTU-local:** one failed
/668 passed/four skipped. A test object bypassed constructor directory setup;
the positive selection-provenance case raised `FileNotFoundError`. The failed
gate is retained; profiler/native speed tests and training did not run.
Pending GPU/callback cancellation was requested. Source118 test-only recovery
is being prepared, retaining the same assertions, science and checks.
[Verified capsule002](../outputs/hpc-results/source117-first-suite-002/local-verification.json)
and [ledger](05_experiment_ledger.md).

**Historical source117 submission, 17 October 2026 13:32 DTU-local:** CPU29745524
RUN, GPUbenchmark29745525/callback29745526 PEND. Only CPU resource control and
matching fixtures change from source116:16 slots/64GiB for the CPU stage;
the13 configurations/native32-slot GPU stage remain identical. All source116
jobs were retired before numerical execution and confirmed EXIT. No new check
pass, performance result, winner, ratio adoption or training start exists yet.
[Verified capsule001](../outputs/hpc-results/source117-first-suite-001/local-verification.json),
[docs61](61_chemistry_execution_selection.md) and [ledger](05_experiment_ledger.md).

**Historical source116 submission, 17 October 2026 13:27 DTU-local:**
CPU 29745514/GPU benchmark 29745515/promotion 29745516 are PEND; verified capsule001
authenticates all 13 candidate plans/source/parent/receipts. The one serial
32-slot A100 benchmark will measure overlap/workers/microbatches; no speedup,
selected ratio, native pass or training start is established. Source115's CPU
pass remains history and its unstarted GPU/callback were intentionally retired.
[Docs61](61_chemistry_execution_selection.md),
[capsule verification](../outputs/hpc-results/source116-first-suite-001/local-verification.json)
and [ledger](05_experiment_ledger.md) retain exact resources/policy/identities.

**Earlier authorized preparation, 17 October 2026:** source116 is preparing a
gated overlap/worker/microbatch suite and an explicit 1:2 fallback only when no
passing 1:3 candidate meets the ten-hour preference. No new source116 job or
performance outcome is established yet. Pending source115 GPU/callback were
intentionally retired, observed EXIT at 13:15; its CPU/profile success remains
preserved. [Docs61](61_chemistry_execution_selection.md) and the
[ledger](05_experiment_ledger.md) distinguish unchanged intended loss weighting,
changed prior sampling, hard gates and conditional promotion.

**Runtime-only recovery submitted, 17 October 2026:** source115 changes the
internal segment budget 22h→71h, mapped to 72h allocation with 1h shutdown reserve.
The 1:3 mixture, loss/search, model/data/parent/optimizer/RNG and all 18 gates,
including the 1.5× plus 1800s epoch-fit formula, remain unchanged. Source112's
short benchmark remains failed on its 22h policy; a fresh native benchmark must
pass the new plan before any training starts. No 1:2 fallback was used.
At 13:02 DTU-local, **CPU 29745444 DONE / CPU gate passed**; native 29745445/
callback 29745446 remain PEND. Verified capsule003 records 384 passed/four CUDA
skips in 46.68s and a passed profiler, profile SHA256
`afd3dd8c488a32e78831d528ed9c0cfb783d2890e302f70b5871684615b8e6f9`.
Native readiness remains pending. The epoch 35 parent is preserved and no training
segment is established. Quota remains strict with only a small current margin.
[Preparation](../outputs/hpc-submissions/source115-preparation.json),
[capsule003 verification](../outputs/hpc-results/source115-first-suite-003/local-verification.json)
and [ledger](05_experiment_ledger.md) retain actual bindings/receipts.

## Saved-result speed analysis, 17 October 2026

This section analyzes the existing
[source112 benchmark](../outputs/hpc-results/source112-first-suite-003/runs/source112-chemistry-search-acceleration-attempt001/benchmark_gate.json)
and [`chemistry_update`](../src/latentmol/chemistry_continuation.py).
It is not a new benchmark, implementation, changed source115 configuration or
training ETA. No new performance job was submitted for this discussion.

| Quantity | Two-update source112 mean / interpretation |
| --- | --- |
| Full update | 12.132s for 256 real + 768 prior points |
| Teacher calls | 8.097s, about 67% of the full update |
| Whole prepass | 9.040s, including the teacher calls |
| Full update minus teacher calls | 4.035s; mixed model/transfer/backward/optimizer/overhead, not isolated GPU time |
| Search checks | 74,976/update; about 0.5 billion/epoch if that workload persists |
| Throughput | 21.101 real molecules/s; 84.404 total candidates/s |

Search time is **inside** prepass time, which is inside full update time; these
three fields must not be added. With 1,688,539 real training molecules and batch 256,
there are 6,596 logical updates per real epoch. The saved projection uses
`training_real_count / real_molecules_per_second + 600s`, yielding 22.4h from
only two measured updates. A ten-hour target under that allowance needs roughly
**5.367s/update**, about 2.26× faster overall. Search difficulty and raw validity
can change during training, so this projection is not measured epoch runtime.

The following are candidate measurements, **not implemented or submitted here**:

- Overlap CPU teacher work with GPU work at unchanged model weights within an
  update. Preserve prior/posterior draws, target order, cache/search semantics,
  population-JS totals and the single optimizer step; compare complete loss,
  gradients, optimizer and RNG against reference before using the execution path.
- Benchmark 12 versus 20/28 search workers. The coordinator's live host inventory
  reports 32 CPU slots and two A100 80 GB GPUs on `n-62-18-10`: 28 workers plus four
  main threads consumes the whole host's CPU capacity. Scaling need not be linear
  and larger requests may change queue availability.
- Test microbatch 128→256 while retaining the same 256-real/768-prior logical
  update. Saved peak reserved VRAM was 21,351,104,512 bytes (about 19.9 GiB), so
  there is a reason to measure a larger microbatch; this is not proof it fits
  or improves throughput. Compare native loss/gradient/update parity and memory.

A 1:2 fallback reduces a 256-real update from 1,024 total points to 768, rather
than halving work. In the measured cohort, 472/512 real outputs were already
valid, whereas only 14/1,536 prior outputs were already valid and 1,250 prior
searches were censored. Simple unchanged-throughput cost models suggest roughly
**15–17h/epoch**, not below 10h; that range is unmeasured and excludes evolving
search difficulty and execution interactions.

Ratio changes also need explicit objective semantics. Current repair is
`0.25 * mean_real + 0.75 * mean_prior`, with failed/already-valid targets zero
in their all-attempt branch means. Keeping those coefficients with fewer prior
draws keeps the intended repair expectation but changes sampling variance;
switching to 1/3 and 2/3 changes the objective. Element/charge JS currently pools
probability-weighted atom/charge counts over both populations, so changing their
point ratio also changes that mixture unless explicitly reweighted. Reconstruction
and KL apply only to real inputs; the occupancy anchor is a prior mean.
Candidate-ID spacing and strict configuration/reporting currently assume 1:3.
Therefore 1:2 is a new explicit protocol choice, not a silent config-only speed
switch or a claim that every loss is unchanged. Source115 is still 1:3.

Source113's 19.86–28.65× figures time teacher `build_targets` only; loss/gradient
audits are outside that timer. Its 16/64/128-total-point cohorts do not establish
native 1,024-point-update speed. Exact backend loss/update regression evidence
does not by itself establish chemistry benefit or improved random generation.

**Native benchmark retrieved, 17 October 2026:** four CUDA checks passed and
the full streamed reference/optimized execution proof passed model, gradient,
optimizer, RNG and ordered-target equality. Full update times were
12.2312/12.0329s, including 8.1965/7.9982s teacher search. This is an actual short
GPU benchmark, distinct from the CPU teacher-only speedups below. All 17 other
benchmark gates passed, but projected epoch time was 80,621.44s (22.4h), so
`complete_epoch_fits` failed its unchanged 1.5× plus 1,800s reserve rule against
a 79,200s segment. The resulting required reserve is about 34.1h. Thus the
search acceleration preserves the measured loss/update semantics but does not
yet fit the current continuation allocation policy. No longer allocation or
gate change was authorized/made in this status turn. Benchmark weights were
discarded; callback stopped and **no 1:3 training began**. Short retention/
composition/sulfur checks do not establish improved generation.
[Verified capsule003](../outputs/hpc-results/source112-first-suite-003.tar.gz),
[status summary](../outputs/hpc-results/source112-first-suite-003/status-summary.json)
and [ledger](05_experiment_ledger.md) retain the actual proof, identities and commands.

**Earlier access-limited observation, 17 October 2026:** the completed CPU study below is
unchanged. A later 16 October live query observed source112 GPU 29736546 and
callback 29736547 EXIT; their logs/cause remain unreviewed. Today's SSH accepts
the loaded DTU key then requires additional password authentication, preventing
fresh status retrieval. All pending/no-training descriptions below are dated
historical snapshots, not current remote-state claims. See the
[ledger](05_experiment_ledger.md).

**Verified supplemental study completed, 16 October 2026 14:13 DTU-local:**
source113 **CPU 29736707 DONE/passed**, nine cohorts / 108 timed passes in about 18
minutes. Exact targets/work, every evaluated loss component and all compared
logit gradients agree; 1,944 gradient rows have zero measured differences.
Twelve-worker paired median speedups span **19.86–28.65×** for teacher
`build_targets` only. On the first 4-real/12-prior cohort, all 192 nonzero named
decoder parameter-gradient tensors also agree exactly across four modes. The
whole model hash is unchanged; no optimizer update occurred.
[Final capsule002](../outputs/hpc-results/source113-paired-study-002.tar.gz),
[final study JSON](../outputs/hpc-results/source113-paired-study-002/runs/source113-repair-speed-loss-parity-attempt001/study_result.json)
and the [ledger](05_experiment_ledger.md) retain exact identities, commands,
timings and scope. **At that snapshot native GPU 29736546/ callback 29736547 were
pending; no 1:3 training start had been established.** This CPU regression pass does not replace native
streamed joint-update/runtime/retention/quota checks or establish chemistry
benefit. Conditional runtime risk below remains unresolved until that benchmark.

Verified update, 16 October 2026: source112 **CPU 29736545 DONE**, 333 checks
passed / four CUDA-only skipped; mandatory same-allocation CPU profile passed.
The same 36 decoded points produced identical semantic receipts, 3,264 checks and
11 targets. Warm times were 15.880037s reference, 4.709617s optimized serial,
1.478322s optimized 4-worker and 0.602755s optimized 12-worker (**26.35×** versus
reference). This certifies that measured CPU teacher workload, **not a full
training speedup**. GPU 29736546/callback 29736547 were pending at that read; no
1:3 training start had been established. [Actual profile](../outputs/hpc-results/source112-first-suite-001/runs/source112-chemistry-search-acceleration-attempt001/repair_profile.json),
[verified capsule001](../outputs/hpc-results/source112-first-suite-001.tar.gz) and
the [ledger](05_experiment_ledger.md) retain identity/provenance and separate
storage-maintenance scope. A fresh [capsule002](../outputs/hpc-results/source112-first-suite-002.tar.gz)
reconfirms CPU/profile passed and both native/callback pending at 13:38 DTU-local.
Separate verified archival preserved 73 historical files losslessly and leaves
10.78GiB current quota headroom; [docs50](50_hpc_space_cleanup.md) records exact
restoration and all failures. All native parity/runtime/retention gates and a
fresh promotion-time quota check remain required.

Historical initial-submission snapshot, 16 October 2026: corrected source112
was submitted as **CPU 29736545**, **native GPU 29736546**, **promotion 29736547**.
At submission no tests/profile/native results existed; the CPU/profile outcome
above supersedes that pending state. Scientific protocol/execution are unchanged,
and **native GPU readiness/training remain unestablished**.
[Source112 preparation](../outputs/hpc-submissions/source112-preparation.json)
and the [ledger](05_experiment_ledger.md) retain exact identities. Source111's
blocked GPU/callback are confirmed owner-canceled; its failures remain below.

Historical source111 **CPU 29736537 failed** (nine failed / 323 passed / four
CUDA-only skipped). Its blocked GPU 29736538 and callback 29736539 were owner-
canceled; no source111 profiler/GPU result exists. Controller/RDKit fixture
assumptions and CPU optimizer snapshot aliasing in the parity helper were
corrected in source112 and passed its CPU fixtures; source111 remains failed.
[Preparation](../outputs/hpc-submissions/source111-preparation.json)
and the [experiment ledger](05_experiment_ledger.md) retain immutable identities,
actual receipts and previous failures. The
[chemistry phase](53_atom_preserving_triangle_continuation.md) remains conditional
on all native readiness, retention, runtime, quota and serial-health gates.

## Additional paired CPU study: design and historical partial evidence

**Historical verified partial results at 13:59 DTU-local:** source113 was still running.
Two of nine cohorts (seed 17001, 16/64 total points) have 24 completed timed
passes with exact ordered semantics, evaluated loss components and isolated/
weighted decoder-logit gradients. Median paired twelve-worker speedups are
**20.8125× / 25.3020×**, respectively; full study and decoder parameter-VJP
completion were not yet recorded; final capsule002 above supersedes that state.
[Partial capsule001](../outputs/hpc-results/source113-paired-study-001.tar.gz)
and [study JSON](../outputs/hpc-results/source113-paired-study-001/runs/source113-repair-speed-loss-parity-attempt001/study_result.json)
retain actual scope. These CPU timings do not certify native training runtime
or chemistry benefit; GPU 29736546/callback 29736547 remain pending.

Historical submission context: the user requested more speed and loss-equivalence
evidence while the native GPU job waits. The standalone source113 study was
submitted as **CPU job 29736707** against unchanged sealed source112. At that
submission no study result/speedup/pass existed; the partial results above now
supersede that initial pending state. Native benchmark 29736546 and
callback 29736547 remain their existing queued jobs; this study does not alter
their source, resources, gates or training protocol.

The submitted design uses three fixed seeds and cohorts of **16, 64 and 128 total
points**, respectively **4 real/12 prior, 16 real/48 prior and 32 real/96 prior**.
Every cohort has the exact 1:3 mixture. Decode each authenticated frozen-parent
cohort once, and compare those
same logits under reference serial, optimized serial, four-worker and twelve-
worker teachers. Use one persistent-pool warmup and **three unprofiled measured
repeats per backend**, with cleared logical target caches and declared rotated
execution order. Report individual repeats and paired summaries of **teacher
`build_targets` time only**, with warmup separate. Loss/gradient audits run
outside the timed interval and are not separately timed. These figures do not
measure model decoding, complete loss/backward cost or whole training updates.
These diagnostic batch sizes are not changes to training's real 256 / prior 768
logical update.
The 624 scheduled memberships are not unique independent molecules: seed reuse
across sizes can produce nested/repeated latent coordinates. The design tests
batch-size sensitivity across three seeds, not nine independent training seeds.

For each paired cohort, require exact ordered targets, full semantic search
receipts and the declared work budgets. Compare every evaluated loss component
and its gradients with respect to the same decoder logits, holding repaired
targets constant. If a small actual decoder parameter-gradient replay is
included, declare its exact cohort, model mode/dtype and parameter scope
separately; do not label that subset a full native training-update proof.
Search remains detached, and no optimizer update/new training model is created.

The separate source113 study uses 16 CPU slots / 64GB, four main Torch threads,
a one-hour allocation / 3000-second internal budget and no training promotion.
Its nine cohorts total 624 point memberships and 108 timed passes. A 32-point
probability-distinct controlled repair fixture warms the pool and verifies
nonzero atom/charge/bond-head gradients. Actual decoder parameter VJPs are
checked once per mode on the first 4-real/12-prior cohort, without optimizer
updates. Its entrypoint is
`scripts/benchmark_repair_loss_parity.py`, with the standalone audit helper
`scripts/paired_repair_loss_audit_helpers.py`. [Preparation](../outputs/hpc-submissions/source113-preparation.json)
binds the two new scripts/config and source113 archive/manifest, while preserving
all 453 source112 scientific files exactly. Explicit `PYTHONPATH` imports the
source112 package; source113 supplies the standalone script/helper only.
The submitted configuration binds the source112 archive/manifest/plan and
original checkpoint/configuration/dataset/split; recorded cohort artifacts bind
real row IDs, latent seeds and decoded-input hashes. Actual job,
environment, thread/worker allocation, RNG preservation, execution orders,
failures, commands and results belong in the ledger. CPU equivalence/performance
extends engineering evidence; it does not replace the pending native GPU
checkpoint/full-update/runtime/retention/quota gates or demonstrate chemistry
benefit.

Interpretation: applying the same repair-loss implementation to identical logits
and accepted stopped targets should produce identical losses/gradients. These
comparisons are a strong backend-substitution regression check, **not an
independent proof that the chemistry objective is correct or improves molecule
quality**. Native streamed full-update, joint encoder/decoder, optimizer and RNG
parity remain a separate gate.

## Measured problem

Source110's [native benchmark](../outputs/hpc-results/source110-first-suite-004/runs/source110-chemistry-resume-attempt001/benchmark_gate.json)
passed the four CUDA fixtures, original parent authentication and exact native
model/AdamW/RNG checkpoint replay. Sixteen of seventeen readiness gates passed;
only `complete_epoch_fits` failed. Updates took 236.534 and 232.027 seconds,
including 232.514 and 228.017 seconds of detached repair search: about 98% of
update time. The measured projection was about 17.9 days per real-data epoch.
Benchmark weights were discarded, no training segment was promoted, and the
preserved trained parent remains epoch 35 / 230,860 updates.

Conditional performance risk: the unchanged epoch-fit predicate allows roughly
**7.73 seconds per full logical update** at this corpus size and reserved safety
margin. Source110's two-update mean was about 230.27 seconds of search plus
4.02 seconds of other work. **If that decomposition carries over**, search would
need roughly 62× acceleration to fit; a 26× teacher-only speedup alone would be
insufficient. This is a projection, not a measured source112 native failure.
The pending full GPU benchmark decides practical runtime. The queued plan
still uses twelve workers; no additional execution fallback or job is implied.

This performance work retains that failure as evidence. It changes execution
of the teacher rather than reducing the real/prior ratio, changing its loss,
skipping points, shrinking search budgets or extending walltime to conceal an
impractical epoch.

## Scientific semantics that must remain unchanged

Training still uses 256 real molecules and 768 newly sampled unconditional
prior latents per full logical update, with exact 1:3 handling of the final
partial batch. The same authenticated data, parent configuration, optimizer
moments, posterior draws, separate prior RNG and composition reference apply.
Both encoder and decoder train jointly toward **100 total epochs**, including
the parent's completed35. A resulting decoder must be frozen before molecular
latent optimization.

The teacher retains atom identity, formal charge, occupied-slot identity and
slot alignment. Only unordered bond labels can change. Beam width4, at most
32 candidate checks per depth and three edits, probability floor1e-12 and
probability weight1 are unchanged. Candidate priorities, scalar logarithms,
round-robin traversal, beam ranks and tie-breaking order must match the serial
reference. Each point has its own fixed budget; faster searches cannot consume
another point's unused allowance.

The probability-aware cost ranks observed bounded witnesses. It does **not**
certify a globally nearest valid molecule. Already-valid inputs receive no
self-target, and unsuccessful searches contribute zero to the all-attempt
chemistry mean. Accepted targets still use full atom/occupied-charge/undirected-
bond categorical NLL, including unchanged identity labels. All coefficients,
three-epoch repair ramp, population composition penalties and occupancy anchor
remain the same. No maximum-allowed-valence proxy or learned scorer is added.

## Execution changes

The historical serial teacher remains the reference implementation. The new
`optimized` backend reuses immutable per-point information: supported atom/charge
labels and atom-only RDKit construction, active pairs, scalar probability-regret
tables and invariant validation. Candidate-dependent connectivity, implicated
slots and priorities are recomputed. Detailed edit receipts and authoritative
validity/representation checks remain required for accepted witnesses. This
specialized path must reproduce reference outcomes; unsupported inputs retain
the declared authoritative diagnostic path.

A persistent pool runs independent point searches in twelve **CPU-only spawned
processes**. Tasks contain detached NumPy labels/probabilities and immutable
schema/search configuration. Model weights and CUDA tensors never enter the
pool. Results return in original input order, irrespective of completion order.
Worker exceptions propagate; there is no silent serial fallback or substituted
target. Workers use one CPU-library thread each and consume no training RNG.

The teacher simulates the original LRU cache order before dispatching misses,
preserving hit/eviction/duplicate-point decisions and logical work accounting.
The pool persists across microbatches and updates. An `ExitStack` closes it on
normal completion or exceptions at the allocation boundary. Disposable replay
clears the logical target cache afterward while retaining warmed processes, so
measured training begins with the original empty-cache scientific state.

The new plan declares resources outside the unchanged scientific `protocol`:

```json
"execution": {
  "backend": "optimized",
  "workers": 12,
  "torch_threads": 4,
  "gpu_cpus": 16
}
```

CPU preflight/profile and GPU benchmark/training request sixteen slots on one
host with4GB per slot, 64GB total requested host memory. Metadata callbacks
retain two slots. The main process retains four Torch/OMP/MKL threads by setting
its environment **after** the shared HPC wrapper; the other twelve slots serve
search workers. Allocated slots and native thread counts are checked. Historical
plans without this stanza keep their original serial execution. Execution
settings enter the new source/plan/checkpoint identity; they cannot be switched
silently while resuming a sealed chemistry phase.

## Required proof before promotion

1. **Allocated CPU fixtures and profiler.** Run the existing chemistry/controller
   fixtures plus exact diagnostic, fast-search, worker-pool and full-update-parity
   fixtures. CUDA-only fixtures remain deferred to the mandatory GPU precheck.
   `scripts/benchmark_atom_preserving_search.py` decodes an authenticated frozen
   parent on CPU, then compares identical outputs under reference serial,
   optimized serial, optimized 4-worker and optimized 12-worker execution. Its
   small diagnostic cohort is four real/posterior and 32 prior points; this is
   explicitly a profiler cohort, not the training1:3 ratio. Record exact semantic
   receipts, logical search checks, accepted/status counts, cold profiled timing,
   warm unprofiled timing, process startup and cumulative hotspots. A same-job
   identity-bound `repair_profile.json` must pass before the CPU gate can pass.
2. **Actual native full-update equivalence.** In the authenticated A100 runtime,
   compare one real batch256/prior768 update from exactly restored parent
   model/AdamW/RNG state under reference and optimized teachers. Require exact
   ordered target/provenance semantics, loss components, gradients, optimizer
   outcome and RNG lineage. Timing/cache observation fields are explicitly
   separated from scientific parity inputs. Both executions are disposable;
   failure restores parent state and prevents promotion.
3. **Native readiness and practical throughput.** Run the four mandatory CUDA
   fixtures and actual checkpoint replay, then benchmark the accelerated full
   updates. Reapply every existing finite/nonzero-gradient, coverage, memory,
   reconstruction, charge/composition/sulfur/size/occupancy retention and
   complete-epoch runtime gate. Report real molecules/second and total
   candidates/second separately. CPU-profile speedups do not replace the native
   end-to-end epoch projection.
4. **Gated serial continuation.** Only actual passed CPU/profile and native gates,
   authenticated completed allocations, matching source/plan/checkpoint ancestry
   and available quota may submit the next training segment. The optimized
   callback checks an already persisted passing scientific gate before allowing
   up to120 seconds of scheduler `RUN` propagation; it still requires genuine
   **DONE**, never `RUN` or `EXIT`. Aged-out records retain the existing unique
   persisted completion requirement. Stop on failure, preserve all artifacts,
   and retain final held-out evaluation at the authenticated absolute epoch100.

All numerical fixtures, RDKit searches, profiling, model decoding, native parity,
training and evaluation run on allocated DTU HPC. Local work is restricted to
editing, AST review, byte/JSON/hash verification, packaging and transfer. Actual
receipts, commands, environment, scope, failures and promotion outcomes must be
recorded before any claim that the accelerated chemistry continuation works.
