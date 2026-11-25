# Can we expand the molecular graph corpus beyond 50 million?

The [PubChem build config](pubchem-large-corpus.json) pins the 4 November
2026 CID–SMILES export by byte length and the publisher's checksum.
[PubChem](https://pubchem.ncbi.nlm.nih.gov/docs/statistics/) lists about
125 million compound records. The build measures how many are distinct
and representable after conversion. The target is at least 50 million
unique training graphs, retaining all eligible unique records above that
threshold as storage permits.

Ingest applies the same canonical, stereo-free graph schema as the VAE,
with lossless atom, formal charge, and bond round-trip checks. It records
rejection reasons and heavy-atom counts. Deduplication compares full
canonical SMILES across shards and against existing training membership.
The 70,000 validation and 70,000 test molecules keep their split identities.
Added graphs are screened against both holdouts and the reserved ChEMBL
scaffold families.

The CPU job downloads the 1.39 GiB compressed snapshot to node-local
scratch. It commits verified chunks of at most five million input records.
The final `corpus/manifest.json` records source attribution, counts,
dataset identity and per-file SHA-256. Its compressed `train/part-*.tsv.gz`,
`val.tsv.gz` and `test.tsv.gz` have two columns, canonical SMILES and source.
These files are graph membership for future training, with graph tensors
reproducible from the recorded schema and RDKit version. The build does
not create dense training arrays. This limits pressure on work3.

On the allocated CPU node from the sealed source directory:

```sh
PYTHONPATH=src python experiments/vae/build_large_corpus.py \
  --config experiments/vae/pubchem-large-corpus.json --output NEW_OUTPUT --workers 23
```

DTU accepted **29903880** on 5 November 2026. Both Milan hosts were
administratively closed to new jobs. The original was verified terminal
without starting before releasing replacement **29903886** on the general
`hpc` CPU queue, with 16 cores and a 24-hour allocation. The replacement
uses the same sealed code, source snapshot and output path, with 15 ingest
workers. Four focused local tests passed, including round-trip ingest,
duplicate removal, holdout exclusion, and atomic publication. The run writes to
`/work3/s234818/latentmol/data/pubchem-large-corpus-20261105a`.
The completed `corpus/manifest.json` establishes the accepted unique count.
[Handoff receipt](../../outputs/hpc-submissions/pubchem-large-corpus-20261105a/hpc-handoff.json).

### Corpus result

Job **29903886** completed successfully on 6 November 2026 in 5 hours 35
minutes. It read 124,860,638 PubChem rows and retained 96,237,427 before
deduplication and holdout screening. The final corpus contains 81,409,264
unique training graphs, including 77,738,566 new PubChem graphs. The
70,000 validation and 70,000 test graphs bring the total to 81,549,264.
Compressed corpus files occupy about 0.86 GiB.
[Verified manifest](../../outputs/hpc-results/pubchem-large-corpus-20261105a/corpus-manifest.json).

## Sparse training cache

The [cache config](pubchem-large-sparse-cache.json) binds the completed
PubChem job and its existing MOSES–ChEMBL holdout counts. A separate CPU
job reads the finalized corpus after scheduler success and verifies its
manifest hash against the parent completion marker. It encodes the full
validation and test sets first, then streams the unique training set in
seeded order. Capacity is measured on the allocation and rechecked during
writes, with 4 GiB reserved for other work3 activity. If storage limits the
build, the cache includes the fitting prefix of that order and records
the retained and omitted counts. The corpus remains the complete identity
inventory.

The cache packs occupied atom and charge labels, sparse undirected edges,
canonical identities and row offsets. `SparseGraphDataset` in
`latentmol.datasets.bulk_cache` opens the cache with memory-mapped arrays
and returns ordinary `atoms`, `charges`, and `bonds` tensors. Padding and
the bond matrix are reconstructed per row. This is a separately published
dataset for future training.

On the allocated CPU node from the sealed source directory:

```sh
PYTHONPATH=src python experiments/vae/build_large_sparse_cache.py \
  --config experiments/vae/pubchem-large-sparse-cache.json \
  --output NEW_CACHE --wall-seconds 82800
```

DTU accepted cache build **29903998** on 5 November 2026 with the dependency
`done(29903886)`. It requests one 16-core CPU allocation, 24 hours on `hpc`,
and stops admitting rows after 23 hours to leave time for publication and
readback checks. Its output path is
`/work3/s234818/latentmol/data/pubchem-large-sparse-cache-20261105a`.
Five focused local tests and the actual launch entry point on a small
fixture passed. After publication the job rehashes the cache and reads a
bounded canonical graph sample from each split. Native results are pending.
[Submission receipt](../../outputs/hpc-submissions/pubchem-large-sparse-cache-20261105a/submission.json).

The first cache job started at 03:25 CET on 6 November. It packed both
holdouts and at least 4,110,000 training rows before one stored SMILES
failed a representation round-trip when re-canonicalized. The job exited
at 03:35 CET without a published cache. Its last log reported about
0.71 GiB staged against a 22.9 GiB effective capacity.
[Cache traceback](../../outputs/hpc-results/pubchem-large-corpus-20261105a/sparse-29903998.err).

The exact failure was training row 4,125,035. RDKit 2024.09.6 changed the
stored aromatic spelling on a second canonicalization pass. The retry
parses the stored spelling directly and still requires the graph to decode
back to that exact identity. Canonical-order verification recognizes
already-matching labels before attempting atom remapping. The original
sealed code reproduced the failure. The corrected path preserved exact
graph identities across the failing row and 64 neighbors. The focused
cache and ordering suite passed 21 cases.

Retry **29904885** started at 06:36 CET on 6 November with the same corpus,
holdouts, output location, and 4 GiB quota reserve. The previous job was
terminal and the final cache path was absent before the retry was submitted.
[Retry receipt](../../outputs/hpc-submissions/pubchem-large-sparse-cache-20261105a/retry-1106a/submission.json).

## Heavy-atom-balanced 60 million split

The [view config](pubchem-hac60m-view.json) selects exactly 60,000,000
distinct training graphs from the complete packed PubChem corpus. It
censuses the actual unique availability for each of 1–50 heavy atoms
directly from the packed graph headers. A uniform allocation would need
1.2 million graphs per count. Even before deduplication the one-atom bin
has at most 39 eligible graphs. The quota calculation uses all available
rows in short bins and equalizes the remaining quotas as far as possible
at the fixed total. The manifest records original and selected histograms.

Both original 70,000-molecule holdouts retain their identities and order.
Each gains 100,000 PubChem-only graphs from bins with spare capacity, for
170,000 validation and 170,000 test graphs. The added molecules are
identity-disjoint from each other, training and both original holdouts.
Added examples may share scaffolds with other splits. The original
ChEMBL scaffold exclusions are preserved.

The new split publishes row indices and a manifest bound to the sparse
parent. This uses roughly 230 MiB of additional storage and reuses its
already packed graphs. `latentmol.datasets.bulk_view.BalancedSparseCorpus`
provides the training and holdout datasets. Run on an allocated CPU node:

```sh
PYTHONPATH=src python experiments/vae/build_hac_balanced_view.py \
  --config experiments/vae/pubchem-hac60m-view.json --output NEW_VIEW
```

DTU accepted **29904892** on 6 November 2026 with `done(29904885)`.
It uses one two-core CPU allocation with a two-hour limit and publishes to
`/work3/s234818/latentmol/data/pubchem-hac60m-view-20261106a`.
Ten focused local tests and the launch fixture passed. The job verifies the
published view and reads canonical graphs from all three splits. Exact
per-size counts are pending the native build.
[Submission receipt](../../outputs/hpc-submissions/pubchem-hac60m-view-20261106a/submission.json).

### Published cache and balanced split

On 6 November, cache retry **29904885** completed at 08:39 CET with all
81,409,264 training graphs and both original 70,000-graph holdouts. It
used 13.56 GiB, reached neither its storage nor time limit, and passed
canonical graph readback on 16 rows from each split.

View job **29904892** completed at 08:41 CET. It published 60,000,000
training graphs, 170,000 validation graphs, and 170,000 test graphs.
The view uses about 230 MiB of indices in addition to the shared cache.
All 50 heavy-atom counts appear in training. Counts 15–31 are level at
2,285,763 or 2,285,764 graphs per size. The other bins used all available
graphs, including 29 one-heavy-atom and 241,482 50-heavy-atom graphs.
The added PubChem holdout rows have 15–31 heavy atoms, where the cache
had surplus rows. The original holdouts cover a wider size range.

[Final manifests, per-size counts and scheduler output](../../outputs/hpc-results/pubchem-large-corpus-20261105a/cache-and-view-20261106T094305211844Z.json).

### Prior-only chemistry training from epoch 45

The [15:1 configuration](epoch45-pubchem60m-prior15.json) starts a separate
branch from the saved epoch-45 checkpoint, including its optimizer state.
It trains on all 60,000,000 real graphs with reconstruction and KL and
4,000,000 fresh priors with the existing atom-preserving repair NLL and
structural proposals. Each update has 240 real and 16 random points.
The repair loss averages over the random points and retains its 0.375
coefficient. This branch uses those three loss terms.

CPU job **29906086** is queued with `done(29899165)`. It reads the
combined epoch's saved cursor and resumes the original recipe in a fresh
V100 allocation if the first segment reached its time limit. The 80 GB
A100 benchmark is released only after a completed combined epoch covering
all 3,670,698 real rows. It checks 64 warmed updates, repair coverage,
finite gradients and VRAM. Training restarts from epoch 45 after the
benchmark and uses serial 24-hour allocations with a 23-hour save budget.
The GPU result is pending. Six focused CPU checks passed.
[Queue receipt and exact stage plan](../../outputs/hpc-submissions/vae-pubchem60m-prior15-20261106a/queued.json).

### Retry from the completed epoch 46

The [epoch-46 config](epoch46-pubchem60m-prior15.json) starts epoch 47
from the completed combined-data checkpoint. Both benchmarks and training
use its saved model, AdamW state and RNG. The 60M branch resets the data
cursor and retains 240 real graphs plus 16 fresh priors per update. Real
graphs use reconstruction and KL. The prior branch uses the same
atom-preserving proposal repair loss, averaged over priors and weighted
by 0.375.

The prior-only runner constructs the proposal teacher with pooled
classification directly. The previous config inherited a marker that
requires the complete historical chemistry recipe and caused both A100
jobs to fail before their first update. Eight focused CPU checks passed,
including construction with the actual recipe, an epoch-47 checkpoint
resume, and the two-report training handoff.

A100 timing job **29938680** and training benchmark **29938681** were
queued on 9 November. Both use the existing 64 measured updates per
arm after eight warmup updates. Handoff **29938682** requires scheduler
success and matching reports from both benchmarks before submitting the
first 24-hour epoch-47 segment.
[Verified submission and dependency](../../outputs/hpc-submissions/vae-pubchem60m-prior15-epoch46-20261109a/submitted.json).

### V100 benchmark transfer, 11 November 2026

Question: can the two waiting benchmarks use available 32 GB V100s while
keeping the conditional full training on an 80 GB A100?

The [V100 benchmark variant](epoch46-pubchem60m-prior15-v100-benchmark.json)
changes only the benchmark hardware requirement. The training requirement
remains A100, and the epoch-46 checkpoint, dataset, loss, 15:1 sampling ratio,
64 measured updates, eight warmup updates and logical batch of 240 real plus
16 prior points are unchanged. The existing 240/120 microbatch fallback
handles the smaller GPU. Each benchmark requests one 32 GB V100, 16 CPU
cores and two hours. Its timing estimates describe V100 performance.

Timing **29992871** and throughput benchmark **29992872** were submitted
on hold. Original A100 jobs **29938680**, **29938681** and callback
**29938682** were canceled and verified terminal with no start time or
execution host before releasing the replacements. DTU disables `bstop`,
so the transfer kept replacements held until cancellation was confirmed.
Scheduler release visibility lagged; a fresh read confirmed both V100 jobs
pending and callback **29992873** waiting on `done(29992872) && done(29992871)`.
No replacement was resubmitted during reconciliation.

Seven focused CPU tests passed, including separate benchmark/training GPU
requirements and the two-report training handoff. An initial local test run
hit a OneDrive temporary-file rename permission error; using temporary files
outside OneDrive passed all seven cases. GPU outcomes remain pending.
[Transfer plan and verified scheduler receipt](../../outputs/hpc-submissions/vae-pubchem60m-prior15-v100-20261111a/handoff.json).

### Canonical-parent resume correction, 11 November 2026

Both V100 jobs started, then exited with code 1 before any optimizer update.
`_fresh` unconditionally called the ordered-to-canonical warm-start converter,
which rejects an already-canonical encoder. Read-only checkpoint pickle
metadata confirmed that the pinned epoch-46 parent has `encoder_pooling`
set to `canonical`. This was a startup implementation error; there are no
throughput measurements or chemistry outcomes from those jobs. The failed
reports and source snapshot are preserved.

`latentmol.training.combined_continuation._fresh` now deep-copies an already
canonical parent, preserving its architecture and all parameters. An ordered
parent still uses the original lossless wrapper. Optimizer and RNG restoration
are unchanged. Nine focused CPU regressions passed, including both parent
modes through benchmark, timing, training and checkpoint resume; a dedicated
test verifies independent parameter/AdamW storage and exact saved prior RNG.
Earlier CPU coverage exercised only the ordered-parent startup.

Submitted fresh V100 timing **30004285**, throughput **30004286** and CPU
handoff **30004287** using the same config and two-hour benchmark budgets.
The failed jobs' exit receipts and absence of any active benchmark or sparse
training job were checked before submission. Existing snapshots, checkpoint,
dataset and blocked historical callbacks were left untouched. The replacement
handoff requires both new jobs to succeed before starting A100 training.
Both GPU jobs were pending at submission; native checks remain pending.
[Failure logs](../../outputs/hpc-results/vae-pubchem60m-prior15-v100-20261111a/status-20261111T144940268931Z.json)
and [retry receipt](../../outputs/hpc-submissions/vae-pubchem60m-prior15-v100-20261111b/submitted.json).

### Smaller benchmark allocation, 12 November 2026

Prepared six repair workers plus four PyTorch threads on ten CPU cores.
The earlier 1:1 worker comparison found a 4.99% throughput gain for twelve
workers over six; its six-worker arm still had a 16-core allocation, so the
result is an indication rather than a measured ten-core/15:1 timing.
Benchmark-only `--benchmark-workers 6 --benchmark-cpus 10` overrides update
the allocation check after timing-study defaults. Reports record effective
execution settings separately from the shared loss/data config digest.
The training recipe retains its existing CPU/worker settings. Eleven focused
CPU checks passed; no ten-core GPU measurement has run.

Attempted to modify pending **30004285** in place to preserve its job ID,
queue and dependencies. DTU's esub rejected `bmod`: only administrators may
use it, and the job was not modified. The second job was not edited. Both
original 16-core requests were verified afterwards. No jobs were canceled
or submitted, preserving the user's request to wait rather than move queues.
The prepared source snapshot is unused by the existing jobs.
[Rejected-edit record and unchanged job requests](../../outputs/hpc-submissions/vae-pubchem60m-prior15-v100-10core-20261112a/unchanged.json).

### Benchmarks completed; epoch 47 queued

Collected 16 November 2026. Both V100 jobs completed successfully on
14 November: timing **30004285** and throughput **30004286**. The
already-canonical parent startup fix worked in both allocated GPU jobs.
Microbatch 240 exceeded 32 GB VRAM; the configured fallback to 120 passed,
preserving the logical batch of 240 real and 16 fresh prior points.

The throughput benchmark measured **217.09 real molecules/s**, or 1.106 s
per update over 64 measured updates after eight warmups. It reserved 59.35%
of device memory and accepted 235 prior repair targets across 1,024 measured
prior points. These are short benchmark observations, not held-out validity
or reconstruction results. Instrumented and uninstrumented timing arms ran
at 213.70 and 206.64 real molecules/s, with identical ordered repair traces
and final weight hashes. Their order/cache effects prevent interpreting the
small timing difference as an instrumentation speedup. In the instrumented
arm, real forward/backward averaged 0.790 s and guarded AdamW/checks 0.242 s;
exposed repair wait averaged about 10 microseconds because search overlapped
the real branch.

At the throughput benchmark's V100 rate, 60M real molecules extrapolate to
76.77 GPU hours, excluding checkpointing and allocation waits. This is not
an A100 measurement or a guarantee that throughput stays constant all epoch.

Successful handoff **30004287** submitted A100 epoch-47 segment **30053692**
and continuation callback **30053693** on 14 November. At 09:16 CET on
16 November the training job remained pending with `Job slot limit reached:
4 hosts`. It requests the unchanged 16 CPU cores, one 80 GB A100 and a
24-hour allocation with a 23-hour save budget. No epoch-47 training updates
or new quality evaluations are available. No jobs were modified or resubmitted
during this status check.
[GPU results and current training queue evidence](../../outputs/hpc-results/vae-pubchem60m-prior15-v100-20261111b/benchmark-summary-20261116T081640264767Z.json).

### Epoch 47 running; random-prior validity warning, 17 November

At 20:15 CET, A100 job **30092509** was running the second allocation.
The first allocation **30053692** started on 16 November, saved after
23,611,440 real molecules at its 23-hour budget and exited successfully.
Callback **30053693** submitted the successor, which resumed the authenticated
checkpoint automatically. Current progress is about **28.05M/60M (46.8%)**;
the latest saved checkpoint covers 27,811,440 real molecules. The checkpoint
cursor continues to label completed epoch 46 until the full epoch-47 pass
finishes. Continuation callback **30092510** is pending behind the running job.

The latest 1,000 updates averaged 0.811 seconds/update, or 296.07 real
molecules/s. Approximately 30 GPU hours remain at that recent rate, excluding
checkpoint costs and the next allocation's queue wait. No runtime failure is
visible; no jobs or settings were changed during this read-only status check.

Generation quality has a material warning. Raw prior decodes counted as
`skipped_already_valid` by the repair teacher fell sharply:

| Training window | Fresh priors | Already valid | Fraction | Mean real reconstruction loss |
| --- | ---: | ---: | ---: | ---: |
| First allocation, first 1,000 updates | 16,000 | 3,271 | 20.44% | 0.18275 |
| First allocation, last 1,000 updates | 16,000 | 52 | 0.325% | 0.03998 |
| Second allocation, latest 1,000 updates | 16,000 | 503 | 3.14% | 0.03579 |

The latest window accepted 15,427 repair targets and censored 70 attempts;
accepted repairs do not establish validity of the decoder's raw outputs.
These windows use different real batches and fresh random priors. They are
training diagnostics, not a fixed-panel evaluation, reconstruction accuracy,
or proof that the new checkpoint improves on epoch 46. The declining loss and
successful runtime must not be reported as improved random generation. The
original epoch-46 parent remains unchanged, and the bounded running recipe is
unchanged pending a separate evaluation/decision.
[Saved progress, window aggregates and scheduler evidence](../../outputs/hpc-results/vae-pubchem60m-prior15-v100-20261111b/training-summary-20261117T191541146291Z.json).

### Epoch 47 after 47,682,960 real molecules, 19 November

A100 job 30092509 reached its allocation budget normally. Its saved cursor is
47,682,960 of 60,000,000 (79.47% of epoch 47, update 524177). The successor
30105550 and its callback 30105551 were pending when checked. Over the last
1,000 training updates, 430/16,000 fresh prior decodes were already valid,
14,526 received an accepted repair and 1,044 had no witness within budget.
Mean real reconstruction loss was 0.02606 over that window.

We evaluated the immutable checkpoint locally with the same Gaussian vectors,
sampling seed, temperature, chemistry checks and 64 historical MOSES validation
molecules used on 17 November. Epoch 46 and partial epoch 47 both reconstructed
64/64. Valid hard argmax draws fell from 190/256 to 6/256, categorical validity
from 157/256 to 6/256, and full 50-slot argmax graphs rose from 0/256 to
249/256. At the earlier 27.81M checkpoint these figures were 7/256, 1/256 and
229/256 respectively. [Fixed-panel results and per-checkpoint counts](../../outputs/runs/vae-review-20261119-47682960/fixed-panel.json)

We also compared the same two checkpoints on 256 randomly chosen molecules
from the preserved ChEMBL scaffold validation split. Exact reconstruction
rose from 79/256 to 173/256 (95 newly correct, one lost). Valid reconstructions
rose from 94/256 to 174/256. This measures generalization when the encoder
sees a molecule. The data mix, number of updates and prior loss also changed
between the checkpoints, so the isolated effect of more data is unknown.
[ChEMBL panel, seed and results](../../outputs/runs/large-vae-chembl-heldout-20261119a/summary.json)

On 512 additional PubChem validation molecules, exact reconstruction rose
from 395/512 to 480/512 (88 newly correct, three lost), and valid reconstructions
from 442/512 to 488/512. We sampled uniformly without replacement with seed
20261119 from the published 100,000-row added PubChem validation index, verified
its checksum and compared posterior-mean argmax reconstructions against each
original canonical SMILES. The test partition was untouched.
[PubChem comparison](../../outputs/runs/vae-review-20261119-47682960/pubchem-val-summary.json)
· [Panel membership](../../outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json)
