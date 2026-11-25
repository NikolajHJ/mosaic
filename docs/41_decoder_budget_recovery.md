# Decoder-study budget recovery with preserved completed work

Status, 8 October 2026: the user authorized correcting the full-data time
budget while preserving completed work. **Source073 attempt002 is sealed/uploaded;
CPU regression job 29624915 DONE: 112 passed, two CUDA-only skipped.** Seed17
GPU **29624919** passed the tiny-decoder CUDA resume fixture, accepted
the pretrained epoch 7 restore, and completed all three ten-epoch seed 17
arms and evaluations. Seed 23 **29624920 RUN** at **09:33 DTU-local** has
completed its control; coefficient 0.1 has four saved epochs and coefficient 1
has not started. Both seed 17 chemistry arms meet the saved exploratory
generation/reconstruction criteria; second-seed replication remains pending.
Old source070
is confirmed EXIT after epoch7 preservation. After `bmod` was rejected,
identical source071 replacement **29625193 PEND** was submitted with the new
serial dependency; the old held29623874 is now retired/EXIT. Existing source070
artifacts and failures remain history.
Exact source/config/parent-checkpoint identities and verified scheduler
receipts are recorded below and in the [ledger](05_experiment_ledger.md).

The **8 October 09:33 DTU-local** downloaded snapshot and endpoint records
show seed 17 argmax-valid counts 65/251/461 of 4,096 for coefficients 0/0.1/1,
and sampled-valid counts 130/536/1,126 of 16,384. The stronger arm retains
7,235 valid and 7,154 exact held-out reconstructions of 7,347. These are local
valence/connectivity results, **not search-target training**. Most prior graphs
are still invalid. Four completed checkpoint digests, including seed23's
control, match their declared bytes. See the
[verified evidence003](../outputs/hpc-results/source073-budget-recovery/extracted-status003/checkpoint_identity_checks.json)
and [docs39 results](39_decoder_chemical_distance_training.md#verified-seed17-endpoints-8-october-2026).
Earlier incomplete snapshots and the original budget failure remain history.

## Why a separate recovery is needed

Source070 requests ten epochs on all 215,000 training graphs per case, but
its sealed per-case training cap is three hours. The verified first-case
snapshot has six epochs at 8,348.42 seconds, about 23 minutes per epoch.
That pace projected about three hours 52 minutes for ten epochs. Left unchanged,
the old controller would time-censor the case before its prescribed endpoint,
retain artifacts and continue with an incomplete comparison. Instead the
coordinator retired it after preserving epoch7. An interim checkpoint is not
the declared ten-epoch endpoint.

The source071 search pilot depended on `ended(29622699)`, which releases after
either DONE or EXIT. The coordinator first successfully issued
**`bstop 29623874`**, placing it in **PSUSP**, then successfully issued
**`bkill 29622699`** after authenticating the epoch7 checkpoint. This preserves
the intended serial scheduling during recovery preparation. LSF then rejected
dependency editing as administrator-only. The coordinator instead submitted
identical source071 replacement **29625193**, dependent on
`done(29624915) && ended(29624920)`, before retiring the still-held original
29623874. Replacement is PEND; original29623874 is confirmed EXIT.

## Preserved work and comparison contract

Recovery uses a new immutable release and unique output roots. The first
seed17/alpha0 case has accepted restoration from **epoch7**; the other five
matched cases start from the unchanged
original epoch-89 VAE. The five fresh cases are **not** initialized from the
partially fine-tuned control. The exact authenticated parent is
`/work3/s234818/latentmol/runs/source-070-decoder-validity-attempt001-decoder-study/continuations/20261007T132150Z_decoder-s17-a0_b678fc9b/epoch-007.pt`,
SHA-256 `baf5838a6185a90f284518d92a8793a6dd768986e3c9563df1d2df3c18612e3a`.
**Seven completed epochs are preserved; unsaved partial epoch8 is discarded.**
The old run/metrics remain retained, rather than relabeled as a completed
ten-epoch comparison.

Epoch-boundary resumption must restore the decoder, unchanged encoder and
reference identity, optimizer state, the loader/posterior/prior/connectivity
RNG streams and actual completed epoch/update counts. Record the authenticated
parent's config and any recovery-only budget changes. Fail closed on incompatible
state; do not call loading only weights an exact optimizer/RNG resume. Preserve
old partial checkpoints and originals without overwriting their histories.

The actual continuation record confirms **23,520 restored cumulative updates**
and discarded partial epoch8. It restores decoder AdamW and the four dedicated
loader/posterior/prior/connectivity streams. **Global RNG restoration and
cross-device bitwise equivalence are not guaranteed**; the CUDA tiny-decoder
fixture validates its declared optimizer/stream path. The restored control
subsequently completed epochs8–10 and evaluation; all seed17 endpoint results
are now verified above. Earlier running snapshots remain in the ledger.

The target scientific comparison remains seeds17/23 × coefficients0/0.1/1,
ten completed epochs per case, the same reconstruction, anchors and chemistry
settings described in [docs39](39_decoder_chemical_distance_training.md).
The recovery protocol in [decoder_budget_recovery.json](../configs/decoder_budget_recovery.json)
declares **six hours per case**, **45 minutes reserved for each case's
evaluation** and a **21-hour segment controller budget**, across **two serial
23-hour GPU allocations**. Segment0 resumes seed17/control epoch7→10, then
starts seed17/alpha0.1 and alpha1 from the original VAE. Segment1 starts
seed23/alpha0, alpha0.1 and alpha1 from the original. Three full per-case
training/evaluation reservations total 20 hours 15 minutes per segment.
Oracle/evaluation times are not hard-killed; reservation gives more room,
not a guarantee of finishing all cases. Requested endpoints remain epoch10.

HPC CPU regression checks passed, followed by the actual CUDA tiny-decoder
resume-equivalence fixture in segment0 before scientific continuation.
Segment1 must verify the same immutable source/config's successful native
fixture receipt. Its dependency is `ended(29624919)`; exact submitted receipts
are linked below. These passes establish resume engineering, not efficacy.

Completed work is measured as authenticated full epochs/examples/updates,
not elapsed wall time or partial minibatch progress that lacks resume state.
New reports must distinguish restored updates from newly executed updates,
training from evaluation time, actual completed endpoints from censored
attempts, and equal requested epochs from unequal realized budgets.

## Gates and required receipts

Preserved metadata and scheduler action receipts are under
`/work3/s234818/latentmol/submissions/source073-budget-recovery-attempt001/`:
`parent_run.json`, `parent_metrics.jsonl`, `parent_latest.json`,
`hold-search-intent.json`, `hold-search-reply.json`,
`retire-original-intent.json`, `retire-original-reply.json` and
`recovery-protocol.json`. Parent record SHA-256 is
`41eddf1388ccb3d961f1bba05b2c36ae879d1a362e625dbe2206b6c23a111ab0`;
parent metrics SHA-256 is
`6c79a8742ec098e40ac7191bc39c55a1ff0273dca774c6959c07e059a3c8ac5f`.
The original cached reference evaluation is authenticated separately by
SHA-256 `0781fa5a0d659ab4eda8c88eb1077ba440cd1f66170433fbae8abe33e957763b`.
These are recorded metadata identities, not completed recovery outcomes.

HPC regression fixtures must cover resumable optimizer/RNG state, frozen
encoder/reference integrity, protected data split and parent/source identity,
duplicate-safe continuation and incomplete-case reporting. Numerical checks
and training remain exclusively on HPC. Local work is editing, static review,
packaging, metadata/file hashing and transfer.

Document every old/new scheduler observation, stop/cancel receipt, submitted
segment/dependency, checkpoint source/hash and completed/failed outcome.
Do not mark recovery successful until its prescribed cases and evaluation
complete. Keep the already verified fixed-target CPU latent gradient separate
from decoder learning and chemistry utility; [docs40](40_search_repair_gradient_study.md)
explains that diagnostic. No new quality improvement is established by this plan.

## Source073 packaging and CPU gate

The submitted release is
`/work3/s234818/latentmol/releases/source-073-decoder-budget-attempt002`.
[Attempt002 archive](../outputs/source-releases/source-073-decoder-budget-attempt002.tar.gz)
SHA-256 is `439b2beb5a26323409cba4ca9580d31a6d8bd814bde955ad78939a304ef11208`;
scientific manifest is
`c4349f3a8203fcaafc349339df7713dbffe20dff69e1af5f7cc9547a66692acf`;
recovery config SHA-256 is
`ea41bfd5d7369b1aaf8573265e9c945ca8fa87e0fbb3c737b5cf1450630f2e13`.
CPU **29624915 DONE** passed the foundation suite (88 passed/one CUDA skip,
13.51 seconds) and new fixtures (24 passed/one CUDA skip, 13.07 seconds):
112 passed/two skipped total. Stage invocation is
`python scripts/hpc/submit_decoder_budget.py --stage cpu --archive-sha256 ...
--receipts ... --submit`. The allocated wrapper authenticates source/archive,
records an ExperimentRun and invokes `decoder_budget_tests.sh`; exact intent,
identity and job receipt are in `submissions/source073-budget-recovery-attempt001`.
Seed17 **29624919 RUN** passed the **actual CUDA tiny stable-decoder
optimizer/stream resume-equivalence fixture**, one passed/three deselected in
9.48 seconds, then accepted the authenticated pretrained epoch7 restoration.
This distinguishes fixture equivalence from original-checkpoint state acceptance;
it is not proof of a completed resumed endpoint or improved chemistry.
Continuation is
`runs/source-073-decoder-budget-attempt002-recovery/segment-0/continuations/20261007T163225Z_decoder-s17-a0_a259b5ef`.
Seed23 **29624920 PEND** depends on `ended(29624919)` and authenticates the
same successful native gate receipt. No new completed epoch is yet verified.

Attempt001 archive/source were packaged and uploaded **but never submitted**;
static CLI guard/gate corrections superseded them before numerical testing.
Their archive and manifest hashes are
`a95e78bb73ce79f10518ebfa863d7ac8d5241e2fd67846d94fc734cfdc210379` and
`e4553b76a59af3d37a270c7872a36a2c615cc6f21c44ed12f48d96f101d1b327`.
Preserve this source-preparation history without labeling it a failed run.

The old source071 CPU job aged out of scheduler history. The helper verified
its saved completed-run proof and original native receipt identity.
`bmod` was rejected with return code255, `Only administrators can
run bmod!`; saved `retarget-search-intent.json`/`retarget-search-reply.json`
preserve it. Recovery instead submitted an identical sealed source071 experiment
with `done(29624915) && ended(29624920)` and retired the old held pilot only
after replacement receipt confirmation and a fresh held-state guard. Source074
attempt002 was a sealed/uploaded metadata-only operator
helper, has archive SHA
`be8ffcd01f1481a94809e83a003dd0bbf4e971a0736262058cf808d3e3b7be0f` and manifest
`edc7ebfe82bd02eae0e9177d2e795d4d53a67325008fe08000fffc6d51a923db`.
Attempt001 was local-only, superseded before upload for a `read(str)` fix;
its preparation identities remain in the ledger. No model mathematics changed.
Local syntax/metadata review does not establish numerical quality.

## Completed scheduling reconciliation

Since the preceding rejected-retarget snapshot, the operator helper submitted
**29625193 PEND** with the same sealed source071 scientific command, config
and unused output root, renamed `source071-search-budget-replacement-attempt001`.
Its dependency is **`done(29624915) && ended(29624920)`**. Only after its
durable receipt and an immediate fresh **PSUSP** check did the coordinator
retire **29623874**, now confirmed EXIT. That unexecuted cancellation is not a
failed scientific experiment. Receipts are in
`submissions/source073-budget-recovery-attempt001/search-replacement`.

Source074 **attempt003** contains the metadata operator helper with that last
held-state guard. It was sealed/uploaded and executed; archive SHA is
`934575ffeeb4b54e1a81b677a03b41aac91df5cae6ef2b6cd211217068b802ae`, manifest
`bc2254a9da8a5806d98df21a190951211b92202bd352b8c644a21928dce3c287`.
Attempt001 was local packaging only; attempt002 was packaged/uploaded/extracted
but never executed before the safety refinement. No model mathematics changed.
Latest snapshot shows old29622699/old29623874 EXIT, CPU29624915 DONE,
seed17 recovery29624919 RUN, seed23 29624920 PEND and replacement29625193 PEND.
The epoch7 parent remains authenticated, with no newly saved epoch yet verified.

## Verified downloaded recovery evidence

[Engineering recovery evidence](../outputs/hpc-results/source073-budget-recovery/engineering-recovery-evidence001.tar.gz)
SHA-256 is `c28123cf8b50f9ac14277066396c2c096e06ede663f3d8e865c280c0db6414bd`.
All **37 data-file hashes in 38 members** were checked during local
standard-library safe extraction; the remaining member is the digest manifest.
This is metadata/file verification, not a local numerical test.
The [16:42:33 UTC / 18:42:33 DTU-local scheduler snapshot](../outputs/hpc-results/source073-budget-recovery/extracted-evidence001/scheduler_snapshot.json)
records CPU DONE, seed17 RUN, seed23/replacement PEND and both old jobs EXIT.
See [CPU run/command/environment](../outputs/hpc-results/source073-budget-recovery/extracted-evidence001/cpu/run.json),
[CPU stdout](../outputs/hpc-results/source073-budget-recovery/extracted-evidence001/cpu/command.log),
[matching native gate receipt](../outputs/hpc-results/source073-budget-recovery/extracted-evidence001/native-gate.json),
[actual continuation record](../outputs/hpc-results/source073-budget-recovery/extracted-evidence001/continuation/run.json)
and [replacement command/dependency receipt](../outputs/hpc-results/source073-budget-recovery/extracted-evidence001/receipts/search-replacement/search-receipt.json).
The native wrapper stdout is explicitly an **8,192-byte prefix**, not a full
trajectory log; parent7 metrics and all hold/retirement/rejected-bmod receipts
remain in the bundle. SSH closed normally after retrieval. No new completed
epoch, trained-quality advantage or chemistry improvement is established yet.
