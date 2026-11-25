# Dense real/prior chemistry coverage in decoder fine-tuning

Status, 8 October 2026: the user explicitly authorized a new comparison with
chemistry on **all 64 real-batch posterior latents and 64 fresh random-prior
latents** per full update. Configuration, reusable trainer extension, study
controller, submission helper and engineering fixtures are implemented and
sealed/uploaded as **source077 attempt001**. CPU engineering gate
**29638104 DONE**, with **113 passed/two CUDA-only skipped**; native L40S
profile **29638385 DONE/passed**, including exact counts, frozen encoder and
dedicated RNG parity. Latest live cohort status: seed 17/control
**29638858 EXIT**, seed 17/chemistry **29638859 RUN**, seed 23/control
**29638860 canceled/EXIT**, seed 23/chemistry **29638861 canceled/EXIT**
under the later single-candidate request. Seed-17 chemistry is last observed
RUN; no newer endpoint is verified. No full-case endpoint or
generation improvement is established. Numerical work remains on DTU HPC;
local editing, static review and packaging are separate from validation.

The failed control's child record still says `running`, but subsequent logs
confirm disk quota failure during its first epoch's `torch.save` checkpoint
write. No completed epoch-10 checkpoint is available. A separate source078
native submission also hit the confirmed 100 GiB user quota before `bsub`.
The new source078 attempt002 CPU and actual L40S repeated-resume/storage
gates passed. The user subsequently reduced the 100-epoch study to the seed-17
chemical candidate versus the original VAE. All sixteen former reservations
**29642812–29642827 were canceled and directly observed EXIT**. After restored
access, four replacement milestone allocations **29645583–29645586 were
submitted** for the single seed-17 chemical candidate, waiting for its active
prefix and the passed gates. The earlier failed recovery upload remains
history. No long training or quality result is established. The verified **historical 13:26
DTU-local** source078 evidence002 capsule preserves the native proof and former
queued receipts; links and current scope are in docs45 and the ledger.
See the latest
[ledger](05_experiment_ledger.md) and [continuation status](45_dense_chemistry_100_epoch_continuation.md).

This experiment uses the existing **local valence/connectivity proxy**, not
the search-target NLL or DeFoG guidance. The original 16/16 chemistry study,
its completed results and protected checkpoints remain unchanged. Existing
search and DeFoG jobs are not altered by this authorization.

## Question and matched methods

Does evaluating the same chemical regularizer on more real and fresh-prior
latents improve independent random generation beyond the historical 16/16
coverage, while retaining reconstruction? Larger batches may improve coverage
or reduce estimator variance; they also cost more compute. They do not
automatically increase the effective chemical-loss weight or guarantee better
generation.

All new methods start from the untouched source047 **epoch-89 stable VAE**,
SHA-256 `1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`.
Freeze its encoder parameters and buffers, retain the original decoder as
an evaluation reference, and train only a fresh copy of the decoder. Model,
protected dataset/split and original reconstruction identities are in
[docs39](39_decoder_chemical_distance_training.md). Use all **215,000 training
graphs for ten completed epochs**, training seeds **17 and 23**.

Each seed compares two methods:

| Method | Real/prior batch coverage | Chemical coefficient |
| --- | --- | --- |
| Dense matched control | Same full reconstruction, prior anchors and computed chemical/Monte Carlo diagnostics | 0 |
| Dense chemistry | Identical data, anchors, RNG schedules and dense coverage | 1 |

Both methods use the same fresh original weights, shuffle, posterior noise,
fresh-prior draws, connectivity sampling, optimizer and requested update
budget within each seed. The control computes the same chemical diagnostics
and consumes the same draw budget even when its coefficient is zero. It is
**reconstruction plus shared anchors**, not a pure original-loss continuation.
The inherited decoder-only AdamW settings are learning rate 3e-5, weight decay
1e-5 and gradient cap five. The implemented
[configuration](../configs/decoder_density_study.json) pins these settings;
the submission helper authenticates its exact bytes before any job.

The historical 16/16 control is not a replacement for this control: increasing
the fresh-prior batch also changes the anchor estimate and random-draw
schedule. Historical 16/16 chemistry coefficient 1 is a separate coverage
reference, with its differing compute/sample budget declared explicitly.

## Exact batch composition and mean-loss scaling

For a full update, encode 64 real graphs with the frozen encoder, sample their
posterior latents using the prescribed noise stream, and reconstruct all 64.
Apply chemistry to **all of those same 64 posterior points**, not an additional
real sample or the former first-16 subset. Independently draw **64 unfiltered
N(0,I) prior points**, decode them, and apply the same chemistry plus frozen
original-reference occupancy/aromatic anchors.

The actual final minibatch must keep the requested **1:1 ratio**: if it
contains N real graphs, chemistry covers all N posterior points and the prior
batch contains N fresh points. Do not retain 64 random points against a
smaller real tail. Do not validity-filter either cohort or replace invalid
prior points with easier samples.

`FineTuneConfig.prior_batch_size = null` is the explicit opt-in to matching
the actual real minibatch. `chemistry_reconstruction_batch = 64` covers all
available posterior points. Existing integer-prior configurations keep their
historical behavior and recorded fields; their checkpoints/protocols are not
rewritten. Full training has 3,360 updates per epoch, including its actual
partial minibatch.

The same mean-scaled objective remains:

```text
loss = original reconstruction(real batch)
     + occupancy_reference_KL(fresh prior)
     + aromatic_reference_KL(fresh prior)
     + alpha * [0.5 * mean_chemistry(all posterior points)
                + 0.5 * mean_chemistry(all fresh prior points)]
```

Both reference anchors retain weight one. Alpha is zero in the control and
one in the chemistry arm. **Increasing coverage from 16/16 to 64/64 does not
multiply the chemical loss by four.** Each cohort contributes its mean with
half weight. Batch counts also do not imply an 80/20 or 50/50 weighting of
the entire reconstruction-plus-anchor objective. Encoder KL is constant
with respect to decoder weights; this does not learn a new posterior or
introduce a new trainable VAE prior.

The unchanged proxy combines the explicit element/charge expected valence
excess, unsupported-charge/empty-risk terms and four-draw global connectivity
estimator documented in docs39. Isolation stays zero to avoid double counting
connectivity. No RDKit conversion enters autograd. This remains a
probability-aware defect proxy, not exact distance to a globally nearest
valid molecule. Bounded probability-aware repair is an independent evaluation
audit, not a search teacher used in this training objective.

## Independent evaluation and historical comparison

Use the same independent **4,096-point unfiltered final prior**, argmax plus
**four categorical draws per point**, and deterministic posterior-mean
reconstruction on the full **7,347 test graphs**. Preserve invalid outputs
and duplicates in the denominators, and never silently repair the raw output
for validity scoring. Final endpoints are ten completed epochs, not whichever
checkpoint gives the best test score. Compare these references within each
training seed:

- Untouched original epoch-89 decoder.
- New dense matched control.
- New dense chemistry coefficient 1.
- Authenticated historical 16/16 chemistry coefficient 1 endpoint, where that
  same training seed has completed its declared endpoint and evaluation.

Unavailable historical endpoints remain missing, rather than substituted
across seeds or inferred from a partial checkpoint. The historical comparison
measures a change in coverage and compute budget; it is not an identical-draw
experiment. Report the new control comparison first, then whether dense
chemistry improves upon the old coverage result. Saved model/source/data/panel
identities must match the declared reference before paired comparisons.

Report raw argmax/sample validity, unique-valid yield per attempt, novelty,
size/element/bond concentration, reconstruction validity/exactness and actual
work. Keep each training seed's paired latent-cluster confidence intervals
separate; evaluating the common 4,096-point bank twice does not create 8,192
independent latent points. Retain the existing exploratory criteria against
original/control, including diversity and the one-percentage-point
reconstruction tolerance. The historical 16/16 reference needs its own
reported effect and uncertainty, not a substituted significance claim.

Mean bounded repair cost is conditional on observed witnesses, not a
population mean or globally nearest chemical distance. Improved defect
scores cannot substitute for improved raw generation. Two fixed training
seeds do not establish universal validity, binding, synthesis or a smooth
latent manifold. Freeze any selected decoder before subsequent molecular-z
optimization with frozen scorers.

## Implemented interfaces and HPC gates

The separate [density protocol](../configs/decoder_density_study.json) is read
by [decoder_density_protocol.py](../src/latentmol/decoder_density_protocol.py),
a metadata-only contract that checks protected lineage, the 64/64 rule,
loss grid, endpoints and allocation headroom. The reusable
[decoder trainer](../src/latentmol/decoder_finetune.py) implements nullable
prior size, per-update coverage/profiling and epoch gradient-clip diagnostics.
[decoder_density_study.py](../src/latentmol/decoder_density_study.py) handles
the actual checkpoint gate, full cases and authenticated comparisons.
The historical 16/16 protocol and initializer remain untouched.

The [submission helper](../scripts/hpc/submit_decoder_density.py) has stages
`cpu`, `gate` and `cases`. Without `--submit` it prints an authenticated plan;
durable stage intents/receipts prevent blind duplicate submission. Every
allocated wrapper independently verifies the sealed archive and executable
manifest. Planned stages are:

| Stage | Resources and limit | Required evidence |
| --- | --- | --- |
| CPU engineering | `hpc`, two slots, 4 GB/slot, one host, 45 minutes | Same immutable protocol/initializer/data; dense/tail, mean scaling, frozen states and matched RNG fixtures |
| Actual native profile | `gpul40s`, four slots, 16 GB/slot, one exclusive GPU/host, two hours | Same-identity CPU DONE |
| Four full cases, serial | Same L40S request, 23 hours per case | Same-identity CPU and native-profile DONE plus authenticated `gate/passed.json` |

GPU requests use `num=1:mode=exclusive_process:j_exclusive=yes`. The native
profile runs the original full-width VAE for **one epoch on 513 real graphs**
at coefficients 0 and 1, with nine updates per arm: **eight batches of 64
and a final batch of one**. Actual posterior-chemistry and fresh-prior counts
must equal each real count. Both arms must preserve the encoder and dedicated
terminal RNG parity. Each small arm has a 1,800-second training budget; then
the original untouched decoder receives the full independent test/prior
evaluation. A passed gate establishes compatibility, finite updates, tail
handling and profiling, **not a generation benefit**.

Profiles record actual counts, synchronized update time, peak memory and
pre-clip gradient norm; epoch metrics retain the clipped-update count and
largest pre-clip norm. The profile's training-only forecast scales six warmed
full updates to 33,600 updates. It excludes epoch validation and final
evaluation and is not a wall-time guarantee.

Case order is seed 17/control, seed 17/chemistry, seed 23/control, then seed
23/chemistry. Each gets a **19-hour training budget** and a **1.5-hour
evaluation reservation** inside its 23-hour allocation. The reservation is
planning headroom, not a hard evaluation timeout. All later cases require
`done(cpu) && done(gate) && ended(previous_case)`; an earlier failed case does
not invalidate a different seed, but an unavailable matched control blocks
that seed's advantage claim. Larger chemistry coverage changes memory and
wall time; no completed endpoint is promised within the limit. Failures,
time censoring and retained checkpoints remain explicit; partial checkpoints
do not enter final comparisons.

## Launch and artifact contract

Run metadata authentication and submission from the extracted immutable HPC
release using the Boltz environment. Source077's paths and digest below define
`RELEASE`, `ARCHIVE_SHA` and `RECEIPTS`. All three submission stages have now
been executed in gate order; the actual CPU/native gates passed before the
four-case submission:

```bash
python scripts/hpc/submit_decoder_density.py --stage cpu \
  --release "$RELEASE" --archive-sha256 "$ARCHIVE_SHA" \
  --receipts "$RECEIPTS" --queue gpul40s --submit
# After the authenticated CPU receipt is DONE:
python scripts/hpc/submit_decoder_density.py --stage gate \
  --release "$RELEASE" --archive-sha256 "$ARCHIVE_SHA" \
  --receipts "$RECEIPTS" --queue gpul40s --submit
# After the native gate is DONE and its passed.json is authenticated:
python scripts/hpc/submit_decoder_density.py --stage cases \
  --release "$RELEASE" --archive-sha256 "$ARCHIVE_SHA" \
  --receipts "$RECEIPTS" --queue gpul40s --submit
```

The CPU command runs
[decoder_density_tests.sh](../scripts/hpc/decoder_density_tests.sh), including
the historical foundation followed by the new dense/resume fixtures. Native
wrappers call [decoder_density_run.sh](../scripts/hpc/decoder_density_run.sh)
and `python -m latentmol.decoder_density_study` with the protocol path/hash,
output root, `--stage gate` or `--stage case --case 0..3`. Numerical entry
requires an LSF allocation and immutable source identity.

Output is `/work3/s234818/latentmol/runs/<release-name>-density-study`, with
`gate/{protocol.json,original.json,original-prior,passed.json}` and per-case
`seed17-a0`, `seed17-a1`, `seed23-a0`, `seed23-a1` directories. Each case retains
its protocol, unique training run, checkpoints, update profile, final-prior
records and `summary.json`. Exact release/config hashes, commands, runtime
environment and job IDs are recorded from verified receipts as stages run.

## Historical source077 preparation/CPU submission snapshot, 8 October 2026

Sealed archive SHA-256 is
`3a706846f9e8ace5f409ea761b7b81c65bce55075aa237a3cf9ba1fb3b622cb8`;
executable manifest is
`8f9f1fa79ea1285df69c0127e807129a201a8fd4a72f80063b787fbd15c4797a`;
configuration SHA-256 is
`c5152f906396f4e2fe095b303088762fab2a34c74e6ecbdaabf1b916f8fd1347`.
The authenticated archive was uploaded/extracted with **405 safe regular
members**. Remote paths are:

- Release: `/work3/s234818/latentmol/releases/source-077-dense-chemistry-attempt001`.
- Receipts: `/work3/s234818/latentmol/submissions/source077-dense-chemistry-attempt001/receipts`.
- Campaign: `/work3/s234818/latentmol/runs/source-077-dense-chemistry-attempt001-density-study`.

The authenticated helper submitted CPU **29638104**, `hpc`, two slots,
4 GB/slot, one host, 45-minute limit. Last observed status is **RUN** on
n-62-30-24; no completed engineering result is claimed. Exact submission
command from the release directory:

```bash
PYTHONPATH=$PWD/src /work3/s234818/latentmol/env/boltz311/bin/python \
  scripts/hpc/submit_decoder_density.py --stage cpu \
  --archive-sha256 3a706846f9e8ace5f409ea761b7b81c65bce55075aa237a3cf9ba1fb3b622cb8 \
  --receipts /work3/s234818/latentmol/submissions/source077-dense-chemistry-attempt001/receipts \
  --queue gpul40s --submit
```

Local [static review](../outputs/hpc-submissions/source077-static-review.json)
used Python 3.12.14 on Windows 11 for five Python ASTs, JSON and line-ending
checks only, with no model imports/numerical tests. Remote `bash -n` accepted
both wrappers; shell syntax is not numerical validation. The sealed scientific
source stays unchanged after this documentation receipt append. Native
profile/full-case outcomes remain pending.

### Historical CPU completion/native-profile submission snapshot

At **10:23:50 UTC / 12:23:50 DTU-local**, CPU **29638104 DONE** on n-62-30-24.
Foundation fixtures report **88 passed/one CUDA-only skipped, 12.20 s**;
density/resume fixtures report **25 passed/one CUDA-only skipped, 13.61 s**,
giving **113 passed/two skipped**. These CPU checks do not pass the actual
CUDA profile. The same helper invocation with `--stage gate` submitted
**29638385**, `gpul40s`, four slots, 16 GB/slot, one exclusive GPU/host,
two-hour limit and `done(29638104)` dependency. Source/protocol/data bytes are
unchanged. Actual native results and the four full-case submissions are still
pending; the verified evidence download will be linked when available.

## Actual native gate and full-case submission, 8 October 2026

**29638385 DONE/passed** on n-62-13-14, NVIDIA L40S with
47,751,888,896 total device bytes, Torch 2.5.1+cu121 / CUDA 12.1. Gate runtime
was **284.932951 s**. Each arm completed exactly nine updates with real,
posterior-chemistry and fresh-prior counts `[64,64,64,64,64,64,64,64,1]`;
encoder identity and dedicated terminal RNG matched. Peak allocated memory
was 5,131,526,144 bytes (control) and 5,133,308,928 bytes (chemistry). The full
original test/prior evaluation finished. This positive receipt passes the
engineering profile, not an independent chemical-improvement criterion.

Warmed full-update means were 0.405296983 s / 0.387466032 s, projecting about
3.78 / 3.62 hours of training updates over ten epochs. The forecast excludes
validation, evaluation and remaining data/metrics overhead. The 19-hour
per-case training cap remains unchanged; completion still requires the actual
endpoint/evaluation receipts.

The same helper with `--stage cases` submitted the following 23-hour L40S
allocations, four slots and 16 GB/slot each, one host/exclusive GPU, without
MPS:

| Case | Job | Latest state | Serial dependency |
| --- | --- | --- | --- |
| Seed 17/control | 29638858 | RUN, n-62-13-14 | `done(29638104) && done(29638385)` |
| Seed 17/chemistry | 29638859 | PEND | Same gates and `ended(29638858)` |
| Seed 23/control | 29638860 | PEND | Same gates and `ended(29638859)` |
| Seed 23/chemistry | 29638861 | PEND | Same gates and `ended(29638860)` |

The source077 scientific bytes and protected assets are unchanged. Each case
trains all 215,000 graphs for its declared ten-epoch endpoint; no full endpoint
has finished at this snapshot. These observations are from authenticated HPC
receipts; the export/download verification is complete as recorded below. Earlier source077
snapshots are retained above with their original pending states.

## Verified evidence capsule, 8 October 2026 12:30 DTU-local

[Evidence001](../outputs/hpc-results/source077-dense-chemistry/evidence001.tar.gz)
SHA-256 is
`bd4b6400b28f29d253964bd4f8d22df047eaec407d7c6bb966b6523ed216b7d3`.
All **113 data-file digests / 114 safe members** were checked during safe
extraction. The [verification receipt](../outputs/hpc-results/source077-dense-chemistry/evidence001.tar.gz.verification.json)
records Python 3.12.14 / Windows 11, byte/hash/metadata scope only, no local
numerical tests. The capsule includes the six intents/receipts/identities,
protocol, CPU/native runtime environments and profiles, original evaluation
and [10:30 UTC scheduler state](../outputs/hpc-results/source077-dense-chemistry/extracted-evidence001/scheduler_snapshot.json).
Large per-latent rows are omitted with original SHA/byte metadata and no
checkpoint bytes are exported.

The [native gate receipt](../outputs/hpc-results/source077-dense-chemistry/extracted-evidence001/runs/source-077-dense-chemistry-attempt001-density-study/gate/passed.json)
and [first full-case receipt](../outputs/hpc-results/source077-dense-chemistry/extracted-evidence001/receipts/seed17-a0-receipt.json)
support the passed gate and active serial cohort. Full endpoints remain
pending. The exact stdlib-only HPC export/local verification commands and
the separately preserved source076 native failures are in the ledger.

All submitted/completed/failed stages belong in the
[ledger](05_experiment_ledger.md), with protected initializer/data identities,
immutable source including uncommitted code, unique paths and actual runtime
environments. A submission is not a passed gate or completed comparison.
