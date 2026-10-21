# Saved-only fixed-step JQ1 report: implementation plan

Status: **plan only; report fork unimplemented**. This document records a static
code audit. It establishes no report release, allocated report test, collector,
rendering job, publication or scientific completion. Existing reports and their
source files remain unchanged.

## Bindings to capture before sealing the report release

Use the actual receipts in these campaigns as the source of exact identities:

- `scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-02/campaign.json`:
  original scientific source02, configurations, CPU 482-case certificate and all
  three passed native readiness studies.
- `scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-storage-03/campaign.json`:
  original launcher source03, CPU 65-case certificate, adoption, existing
  Protenix submission and the two failed, scientifically unstarted Boltz entries.
- `scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-storage-04/campaign.json`:
  recovery launcher source04, its actual CPU/adoption/observer and any later
  replacement Boltz submissions. Read its submission files and exclusive markers;
  a running callback does not establish long-run IDs or a storage pass.

At the actual **21 October, 22:36:48 Paris** snapshot, recovery CPU **29778180**
is DONE/LSF success with **85 allocated contract cases**, independently verified
from bytes, raw XML and schema. Callback **29778181** is DONE/LSF success after
the **complete dispatch storage check passed on work3 and home**, with the full
remaining assigned projections plus each filesystem's unchanged 4 GiB reserve
and file-headroom gate. This does not reserve capacity or waive the repeated
GPU-entry check. The complete bounded home inventory returned 0; no partial
inventory was accepted. Exact saved status:
`results/hpc/20261021-brd4-jq1-25-fixed-storage-04/user-status-20261021T203646Z/status.json`.

| Arm | Actual current scientific job | Scientific output location |
| --- | --- | --- |
| Protenix | **29777709 DONE**, existing run; do not replace | `/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03/run-protenix/payload/study` |
| Boltz-2 | **29778203 RUN** allocation; no study, saved steps or completed entry gate observed yet | `/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2/payload/study` |
| Boltz-2 + chemistry | **29778204 PEND**; no study or saved steps | `/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2_chemistry/payload/study` |

The two replacement jobs were actually submitted independently at 22:34:20-21
Paris, preserving fixed LR **0.01** and the one-candidate/no-halving policy.
Both current IDs and roots now exist for the new report contract; scheduler RUN
alone must not be displayed as an authenticated optimization update. Preserve
29777710/29777711 as separate failed, scientifically unstarted admission attempts.
Never combine their records with a replacement trajectory.

Protenix's summary is present. Allocated observer **29778184** is DONE/LSF success
and saved `observed_success` for **10,000 updates / 10,001 states**, authenticating
same-job actual terminal scheduler/LSF and saved terminal metadata, JSONL and
proof hashes. The future report completion collector must still authenticate
every included snapshot and all full report gates. Observer metadata success is
not a full frame certificate or a claim that every original native byte was
independently downloaded locally. These are scientific-run/launcher facts, not
new report fixture results. No fixed report is published; milestone progress
remains **0/0/0** until its first verified edition.

Scientific source02 remains
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source`, manifest
`1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64`.
Readiness remains in the original fixed-02 run root, not each new output root.
Configs are `brd4_jq1_25_fixed_{arm}_10000.json`. Seal independent fields for
scientific source/config/readiness roots, output parent/study root, launcher
source/adoption, actual GPU submission and CPU/observer/collector proofs.
Never derive all of these paths from one historical `tag`.

## Create separate report files

Fork, rather than repin, the following historical files into new fixed-report
names and a new immutable report-only release:

| Historical code | New work required |
| --- | --- |
| `scripts/jq1_report_contract.py` | New fixed-step contract with exact actual IDs, per-arm roots and independent science/readiness/launcher pins |
| `scripts/render_jq1_25_interim.py` | Fixed-step live-prefix renderer and explicit empty/failed arm states |
| `scripts/render_jq1_25_report.py` | Fixed-step frame metadata, counts, overlays and strict final renderer |
| `scripts/capture_jq1_completion.py` | Fixed-step completion collector plus separate failed-entry evidence capture |
| `scripts/templates/jq1_25_report.html` | Fixed-step narrative, outcome markers and usable empty-arm selection |
| `scripts/publish_jq1_report.py` | New contract/job-list checks and a separate fixed-report viewer/latest pointer |
| `scripts/hpc/jq1_25_report_stage.lsf`, `jq1_25_interim_stage.lsf` | New pinned CPU/certificate/capture/interim/final wrappers, storage checks and exact evidence packaging |
| `tests/test_jq1_25_report.py`, `test_jq1_25_interim.py` | New allocated fixtures covering fixed-step and mixed-outcome behavior |

Keep the historical contracts, renderers, publisher, immutable editions and
`http://127.0.0.1:8776/report.html` untouched. Use a new fixed-report directory,
viewer and separately selected unused local port. Do not convert edition-0006
or its published milestone receipt into new-run progress.

## Authenticate fixed-step bookkeeping

The historical `verify_case()` and interim `progress_assets()` derive
`strict_improving = applied - forced_fallback`. That is invalid here: fixed-step
updates select the only finite candidate even when it is uphill, equal or a
probability no-op. In the new shared validator:

1. Require declared policy `fixed_step`, exactly one attempt with `trial == 0`,
   `selected_trial == 0`, exact attempt and accepted rates **0.3 / 0.01 / 0.01**,
   `fallback_used is False`, and unchanged authenticated runtime parameters
   (`max_backtracks == 1`). Reject halved rates, extra attempts and policy drift.
2. Count strict improvements from the selected attempt's saved `strict` flag,
   matching the producer's definition. Do not relabel every negative scalar
   delta as strict improvement: its saved criterion also checks the autodiff
   surrogate with its original tolerance.
3. Verify `applied`, `forced_fallback == 0` and saved `no_op` totals against all
   included update records. Verify saved `fixed_increased` against
   `fixed_after > fixed_before`; derive and label scalar increase/equality/
   decrease counts explicitly. These scalar categories and probability no-op
   are different, potentially overlapping descriptions.
4. Preserve `selection_policy`, `selected_trial`, accepted rate,
   `fixed_before`, `fixed_after`, `fixed_increased`, `no_op` and selected attempt
   `strict` in frame/curve metadata. Historical `write_frames()` currently drops
   most of these fields in `previous_update`.
5. Preserve native metric identity, detached context, pocket, no-diffusion,
   frozen before/after identities and original derivative/coupled/readiness
   checks. Validate the original chemistry schedule, including final weight 50;
   all other arms retain zero repair weight.

For terminal summary/case equality, authenticate the summary wrapper's exact
config `label` and `initialization`, remove only those wrapper fields, then
require exact case equality and exact original JSONL equality. Source04's
`pro_terminal_proofs()` implements this saved schema distinction; do not weaken
it to partial dictionary matching or replace the original records.

## Represent all arms and actual outcomes

Historical `live_status()` accepts only RUN/PEND/DONE and rejects every EXIT;
`status_or_completion()` knows only passed completion proofs. Add distinct
authenticated states for queued/preparing, running, completed, failed-unstarted
resource admission and failed-partial scientific execution. An aged-out live
record requires persisted same-job scheduler/LSF proofs of its actual outcome.
Unknown or missing evidence remains unknown; it is never converted into DONE.

Authenticate failed old Boltz attempts from their actual EXIT/LSF exit-1,
unchanged entry gate and absence of study, driver invocation and scientific log.
Display zero saved states, empty chunks/curves, `latest_update: null`, passed
original readiness and the actual resource failure. Once replacement IDs are
authenticated, pin them as the current scientific arms and retain the failed
attempts as explicit provenance. Do not merge different jobs into one trajectory.
The two actual replacement IDs/roots are now available for sealing the current
three-arm report bindings. Publication still requires the new allocated report
fixtures, actual rendering assessment and exact byte verification.

The historical template disables `states === 0` selectors and `draw()` returns
without clearing old content. Keep all three arms selectable and visible. On an
empty arm, stop playback, invalidate pending asynchronous loads, clear graph,
heatmaps, readout, metrics and snapshot evidence, show its saved status, and
disable timeline/numeric/arrows/playback/endpoints. Guard every control handler
against zero states. Never invent an initial molecule for queued or failed arms.

Replace static LR 0.03 and fallback-cross narratives. Show each arm's fixed rate,
one-candidate policy and honest strict/uphill/equality/no-op counts. Keep contact
and affinity plots separate and binder probability explicitly separate from the
optimized metric. Changing repair totals are not one fixed objective. Keep raw
invalid/disconnected graphs and clarify that valid graphs/fixed-pose scores do
not establish binding or activity.

## Retain exact saved frames and terminal gates

Reuse the audited algorithm through a separately sealed fork: newline-committed
contiguous state/update prefixes, common captured prefix, ignore only the
unfinished final line, reject corrupt complete rows. Hash the exact included
prefix bytes. Do not silently skip missing snapshots or read later scalar rows
than their matching saved probability frame.

On allocated CPU, authenticate every snapshot path/SHA and saved FP32 arrays:
X `(25, 6)`, E-upper `(300, 5)`, finite simplex, exact saved scalar/argmax and
unordered-pair correspondence. Retain every frame, every scalar point and raw
readout. Keep little-endian FP32 assets at **at most 100 frames per chunk**,
1,650 floats / 6,600 bytes per frame, with exact metadata/binary hashes and
contiguous chunk boundaries. Browser checks must not calculate a new scientific
state or repair/score a molecule.

Completion collectors require actual same-job final GPU scheduler DONE and LSF
success, original summary/case passed, 10,000 contiguous updates / 10,001 states,
exact JSONL equality, complete executed source file-map/config, hard native
parity, initial/coupled/terminal derivatives, frozen identity/readiness and each
snapshot pin. Reauthenticate source04 observer proofs if used after Protenix live
history expires; metadata observer success alone is insufficient. Strict final
publication requires three successful scientific completions and chemistry 50.

## Required allocated fixtures and delivery order

Before report publication, seal a new release and run both new report fixture
modules on allocated DTU CPU. Preserve exact raw XML, logs, actual scheduler/LSF,
test counts, report-source manifest and proof hashes in its CPU certificate.
Required fixtures include:

- Fixed-policy/rate/attempt validation; strict versus non-strict decrease,
  uphill/equal/no-op updates and honest count/metadata retention.
- Separate scientific/readiness/output/launcher roots and wrong-job/config/
  source/pocket/frozen/adoption rejection; authentic terminal wrapper equality.
- Truncated final lines, corrupt complete lines, concurrent prefix reads,
  noncontiguous rows, missing frames, escaped snapshot paths, bad FP32/simplex/
  scalar/argmax/pair correspondence and 99/100/101-frame chunk boundaries.
- Queued/preparing/failed-unstarted/failed-partial/completed mixtures, actual
  EXIT proofs, aged-out authenticated outcomes and absent/mismatched proofs.
  No failure becomes completion and no empty arm receives a fabricated frame.
- New receipt/publisher rejects old job IDs or report contract, changed bytes,
  unsafe edition paths, duplicate immutable editions and wrong current-arm
  mappings. Terminal fixtures reject incomplete budgets and chemistry endpoints.

Track milestones **1500, 2500, ..., 9500, 10000 per arm** from the latest
published *fixed-report* receipt. Initial milestone progress remains 0/0/0 until
that first publication. Do not schedule from the fastest arm alone or publish
every update; avoid duplicate active/unknown report submissions using exact
directories, IDs and exclusive receipts.

After actual report DONE/LSF success and passed assessment, verify its CPU and
input pins, download a fresh archive and sibling receipt, authenticate exact
archive path/length/SHA, safely extract and verify every file. Publish only those
verified bytes into a new immutable fixed edition, then atomically change only
its new viewer pointer. Verify served HTML/chunk bytes and perform browser QA:
actual interior slider drags, exact numeric jumps, arrows, endpoints, playback
and manual pause, all selectors including empty views, probability heatmaps,
overlays, raw graph labels and visible status/counts. Record the published
per-arm milestones and actual receipts in the fixed-report campaign and ledger.

No implementation or computational validation is claimed by this plan. All
scientific rendering and numerical fixtures remain allocated DTU work; local
work remains static code/AST, byte/XML/schema, transfer/publication and browser
presentation checks.
