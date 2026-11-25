# Full combined-corpus VAE continuation

Train the canonical-order VAE from the completed epoch-45 checkpoint on
the full combined MOSES–ChEMBL training split. The first bounded run targets
one full pass through that split, called epoch 46. A fresh standard-normal
latent is drawn for each real graph, including the final partial batch.
Interpolation is an evaluation only.

The [data config](epoch45-combined-data.json) preserves the existing MOSES
train, validation and test identities. It reserves 50,000 ChEMBL molecules
each for validation and test by whole Bemis–Murcko scaffold families absent
from MOSES. The original MOSES cache is reused. ChEMBL labels are encoded
once with the same RDKit and graph contract into a compact cache.

The [training config](epoch45-combined-training.json) retains the
epoch-45 reconstruction, KL, atom-preserving repair NLL, structural proposal
search, composition weights, original-35 occupancy reference, original-36
valid-target reference, native AdamW moments and 0.125 displacement control.
It changes the corpus, enables verified canonical ordering, uses the new
combined shuffle and switches from the earlier 1:2 prior schedule to the
requested 1:1 fresh-prior draws. The composition reference remains the
original MOSES distribution. The prior multiplier in the composition term
is 3.0 at 1:1 sampling, preserving the validated effective 1:3 real-to-prior
weighting within that term.

Benchmark batches 128 and 256 with eight warmup and 40 measured updates each.
The trainer uses the fastest finite batch that meets the existing 85% VRAM
limit. It restores the original epoch-45 parent for the training job, so
benchmark updates do not become training state. A 24-hour allocation has
an internal stop budget and writes periodic resumable checkpoints.

The [evaluation config](epoch45-combined-evaluation.json) compares the
candidate with frozen epoch 45 on each source separately. It uses the full
held-out reconstruction sets, 10,000 matched prior vectors and up to
512 mapped paths per source, including the midpoint. Validation can guide
checkpoint selection. The final test has its own graph split and random
latent seed.

## 4 November 2026 launch

Data preparation job **29893160** completed in 280 seconds. The manifest
records 3,670,698 training molecules and 70,000 each for validation and test.
Each holdout includes 20,000 original MOSES molecules and 50,000 ChEMBL
molecules. The ChEMBL validation and test subsets have 19,312 and 18,477
disjoint scaffolds respectively. The compact ChEMBL cache uses 515 MiB and
reuses the original MOSES cache.

The local CPU tests covered the combined cache, the two-source evaluation
path and the 1:1 trainer, including an identical resumed update sequence.
One small CPU update with the authenticated epoch-45 recipe and its frozen
references produced finite gradients and used the expected displacement.

| LSF job | Purpose | Allocation |
| --- | --- | --- |
| 29893176 | Batch 128 vs 256 benchmark | 80 GB A100, 16 cores, 75 minutes |
| 29893177 | Full-corpus epoch 46, after the benchmark passes | 80 GB A100, 16 cores, 24 hours |

The benchmark will report measured throughput, projected epoch duration and
peak memory. If the epoch extends past the first allocation, the trainer
writes a checkpoint with the exact shuffle cursor and optimizer/RNG state
for a subsequent segment.

## 5 November 2026 V100 handoff

The [V100 profile](epoch45-combined-training-v100.json) requests one 32 GB
V100 and eight cores, split between six repair workers and two PyTorch
threads. The corpus, 1:1 sampling and validated chemistry objective are
the same. The benchmark tests logical batches 128 and 256 with a
microbatch of 128. Each arm can fall back to microbatch 64 if it exceeds
the memory limit. It measures 32 updates after eight warmup updates per
configuration and chooses the fastest finite configuration below the
85% reserved VRAM limit. Local fallback, report identity and resume
checks passed.

| LSF job | Purpose | Allocation |
| --- | --- | --- |
| 29899164 | V100 batch and microbatch benchmark | 32 GB V100, eight cores, one hour |
| 29899165 | Full-corpus epoch 46, after job 29899164 passes | 32 GB V100, eight cores, 24 hours |

Both replacements were submitted held. The original A100 jobs 29893176
and 29893177 were verified held with no execution host or start time, then
retired and verified terminal before releasing the V100 jobs. Training uses
the chosen V100 shape and authenticates the benchmark and dataset identity.
The hold transitions settled asynchronously and were reconciled against
the exact four scheduler records. At 14:31 CET the V100 benchmark was
eligible and pending scheduling. Training was pending `done(29899164)`.

### V100 benchmark and live training

Job **29899164** passed both configurations and selected batch 128 with
microbatch 128. It measured 17.95 real molecules/s versus 17.85 for batch
256, with 61.8% of V100 memory reserved at the winning shape.
[Verified benchmark](../../outputs/hpc-results/combined-vae-epoch46-v100-20261105a/benchmark-29899164.json).

Training job **29899165** started at 15:06 CET on 5 November. At 20:24 CET
it had processed 221,568 of 3,670,698 real molecules and an equal number
of random points. The latest resumable checkpoint covered 192,000 real
molecules. The last 100 updates averaged 11.25 seconds per batch of 128,
including 9.35 seconds recorded for chemistry search. This implies about
84 hours of remaining GPU work at that recent pace. The first allocation
has an internal 23-hour checkpoint stop. No subsequent combined-training
segment was in the scheduler snapshot. Held-out evaluation is still pending.

### Worker scaling and 15:1 timing

The [throughput study](throughput-study.json) runs two disposable benchmarks
from a separate source snapshot. Both measure 64 updates after eight warmup
updates per arm. The original training chain keeps its existing recipe.

- V100 job **29912430** compares six and twelve repair workers with the
  same 16-core allocation, 128 real and 128 prior points per update, and
  identical batches, weights, optimizer state and RNG. It starts at the
  saved epoch-46 cursor after 1,112,192 real molecules. It records throughput,
  real/prior/reference target waits, ordered repair traces and final weight
  hashes.
- A100 job **29912431** follows `done(29912430)`. It replays the 15:1
  recipe from epoch 45 on the balanced 60M view. The paired arms use the
  same updates with phase instrumentation on or off. The report separates
  sparse batch reads, CPU repair submission and exposed wait, CUDA stream
  intervals for real and prior training, and guarded AdamW. Stream and
  host times can overlap.

Both jobs request a 90-minute allocation and discard all benchmark updates.
Six focused CPU tests passed, including the instrumented and uninstrumented
replay. Both GPU jobs were pending on submission.
[Verified submission and exact commands](../../outputs/hpc-submissions/vae-throughput-20261106a/verified-20261106T210534618430Z.json).

The V100 comparison completed on 6 November at 22:26 CET. Over 64 timed
updates per arm, six workers processed 15.94 real molecules/s and twelve
processed 16.74 (+4.99%). Mean update time was 8.029 versus 7.648 seconds.
Mean real repair wait fell from 4.941 to 4.632 seconds and prior repair
wait from 0.632 to 0.571 seconds. Ordered repair traces and final weights
matched exactly. The six-worker arm ran first.

At 01:42 CET on 7 November, A100 job **29912431** was still pending
because the eligible hosts had reached their job-slot limit.
[Saved timings and scheduler snapshot](../../outputs/hpc-results/vae-throughput-20261106a/status-20261107T004254632224Z.json).

Epoch 46 completed on 8 November and checkpointed all 3,670,698 training
molecules. The final checkpoint is `combined-02/checkpoint-000325498.pt`
in the 60M handoff run directory. The held-out comparison is a separate
evaluation.

The 15:1 timing job **29912431** and production benchmark **29921947**
both exited during repair teacher initialization, before their first
update. The 15:1 configuration carries the old baseline marker, whose
validator expects the full historical chemistry recipe. No timing report
was produced. The 60M training handoff **29921948** remains blocked by
the failed benchmark.
[Epoch checkpoint and timing-job status](../../outputs/hpc-results/combined-vae-epoch46-v100-20261105a/handoff-status-20261109T115103274942Z.json).
[Production benchmark logs](../../outputs/hpc-results/combined-vae-epoch46-v100-20261105a/production-benchmark-logs-20261109T115116905396Z.json).
