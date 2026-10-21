# Saved-only fixed-step JQ1 report: implementation plan

Status: **first fixed-step edition published and browser-verified; updateable
viewer active at http://127.0.0.1:8777/report.html**. Source04 allocated fixtures
and first rendering passed; the separately validated source05 publisher delivered
the unchanged verified report bytes. The combined report remains IN PROGRESS. The separate contract, saved-frame
renderers, completion collector, template, publisher and allocated orchestration
are implemented. Historical report files and editions remain unchanged. This
implementation does not change any scientific run. The actual first rendering,
byte verification, publication and browser QA are recorded below.

The first immutable report release, `20261021-brd4-jq1-25-fixed-report-01`,
authenticated 13 files and actually submitted CPU **29778232** and callback
**29778233** at 22:53 Paris. CPU exited after **222 passes / 3 failures**, zero
errors or skips; its bounded-chunk fixtures had synthetic FP32 probabilities
outside the unchanged simplex gate. Callback also exited and correctly prevented
collectors and rendering. Raw logs/XML/scheduler evidence and byte verification
remain preserved in
`results/hpc/20261021-brd4-jq1-25-fixed-report/source01-failure/verification.json`.
This is a report fixture failure, not a native experiment failure.

A new immutable report source02 preserves all production scientific/frame
validation and corrects the synthetic fixture to bounded exactly binary FP32
values. It also declares the report root explicitly for PyTest module identities.
Its 13-file archive is **53,487 bytes**, SHA-256
`9ed7eb09c942a58f2cb8c9df5880ca7196215abcaf300551a74ee231d5e064e9`,
manifest SHA-256
`3b78673801b3875fd040dff9d814a869e5f0172fa8dea164ffb85656e6b3ad5e`.
Remote staging authenticated the report/scientific/launcher source trees and
passed the fresh report-only preflight. Actual CPU **29778238** and callback
**29778239** were submitted at **22:56:42 Paris**; their exact arguments are in
the receipts under
`scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-report-02/initial-submissions.json`.
The repeat encountered a collection error: its importlib collection mode could
not import a bare sibling helper used by an unchanged legacy fixture. It has no
report CPU pass. This failed attempt remains preserved; a further immutable
engineering correction requires the full allocated checks again. No collector,
rendering success, publication or future job is inferred here.



### 21 October - report source04 passed all allocated CPU gates; first saved report is running

Immutable report source03 removed the importlib collection mode while retaining
all required fixtures and original import behavior. Actual CPU **29778240**
completed DONE/LSF success with **225 passes**, zero failures/errors/skips.
Certification **29778241** nevertheless exited because **35 unchanged legacy
cases had empty XML class identities**; the 190 new cases were attributed.
That attribution gate correctly prevented collectors and rendering. Preserve
this rejected certification and its original raw evidence at
`results/hpc/20261021-brd4-jq1-25-fixed-report/source03-cpu-evidence/verification.json`;
a raw pass count did not become an authenticated report CPU certificate.

New immutable **report source04** selects the shared releases directory as
PyTest's common root, so the new and unchanged legacy modules all retain their
actual XML module identities. This changes collection bookkeeping only, with
no scientific, renderer, probability tolerance, required-test or gate change.
Its **13-file / 53,476-byte** archive has SHA-256
`e1886095f890058ed167f27ab648aca4561a8bf5851fa1bbf33f7ec0eeed53f5`,
manifest SHA-256
`7af7f5732842bdee8db930b71a6185039a1b77ec2c5939afccd8b2d9fc91800a`.
The report source is read-only under
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-report-04/source`;
its output root is
`/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-report-04`.

Actual CPU **29778243** and certification **29778244** both completed
**DONE/LSF success**. All **225 cases in all seven required modules** passed:
83 fixed contracts + 44 fixed rendering + 42 fixed publication + 21 fixed
orchestration + 26 Boltz report + 3 chemistry report + 6 chemistry-ramp report.
There were **zero failures, errors or skips**, and every case has its required
module attribution. Independent local standard-library byte/XML/schema and
actual scheduler/LSF verification passed at
`results/hpc/20261021-brd4-jq1-25-fixed-report/source04-cpu-evidence/verification.json`,
SHA-256 `22de220c19ca4737c715c29173d62f86197b619ba6bbb84712029289a0035ef1`.
CPU certificate SHA-256 is
`2a5bb4b09c3ded76c03e631e9929a36a0d358ff77a8fe2c53b8379d5d22d5992`.
Numerical fixtures ran only on allocated DTU CPU; local verification checked
saved bytes/XML/schema and performed no numerical rendering or model work.

Actual report-only downstream submissions are now recorded in the source04
`jobs.json`: Protenix collector **29778245 RUN**; Boltz collector
**29778246 PEND**; chemistry collector **29778247 PEND**; first full-prefix
interim **29778248 RUN**; strict final report **29778249 PEND**. Pending Boltz
collectors wait on their own scientific job endings, and the strict final waits
on successful completion collectors. Submission or RUN alone is not a collector
or rendering pass. No new fixed edition has yet passed rendering assessment,
archive/byte verification, publication and browser QA; fixed published
milestone progress remains **0/0/0**. Historical port **8776/edition-0006** and
all scientific trajectories, jobs, native gates and original failure records
remain unchanged. No additional cleanup or scientific rerun was performed.




### 21 October, 23:07 Paris - first fixed-step report published with every captured saved frame

The first fixed-step report is available at **http://127.0.0.1:8777/report.html**.
Its immutable **edition-0001** is published from the separate viewer at
`results/hpc/20261021-brd4-jq1-25-fixed-report/viewer`; the small stable wrapper
checks its latest pointer every **60 seconds**. The historical report at port
**8776** still serves its preserved edition-0006. No historical edition or
scientific trajectory was rewritten.

Report source04 interim **29778248** actually completed **DONE/LSF success** and
passed assessment. The allocated Protenix completion collector **29778245** also
completed DONE, authenticating all **10,001 original saved snapshots** in
addition to the terminal native checks. The first report captured
**2026-10-21T21:03:01.148625Z** (23:03 Paris):

| Current arm | Saved updates | Saved states | Authenticated captured status |
| --- | ---: | ---: | --- |
| Protenix 29777709, fixed LR 0.3 | 10,000 | 10,001 | Native completed; terminal and every-frame collector passed |
| Boltz-2 29778203, fixed LR 0.01 | 765 | 766 | RUN; own contiguous committed prefix |
| Boltz-2 + chemistry 29778204, fixed LR 0.01 | 99 | 100 | RUN; own contiguous committed prefix |

Every captured probability frame and scalar point is retained in bounded
100-frame chunks. There is no interpolation, new inference, resampling, repair
or fabricated molecule. Both Boltz runs keep one finite fixed-rate candidate per
update with no halving. The combined report remains **IN PROGRESS** while
Protenix is shown individually completed.

The original source04 report archive is **37,187,599 bytes**, SHA-256
`bc7bf1da47981a4638bb4d06bf7e321d3b101053e26e92737ce0aa187dc61ae5`.
All **256 archived files** passed safe extraction and exact byte verification.
The first local publication attempt was correctly blocked before viewer creation
because its CPU scheduler parser did not accept an actually line-wrapped LSF
queue field; the original report archive and scientific outcome remain valid.
A new immutable **publisher-only source05** compacts whitespace for parsing the
unique exact CPU job/status/queue, retaining original proof bytes/hashes and all
authentication gates. It uses manifest SHA-256
`72c3569e49676dc00c448137829fea6ff3abba342b9d512979920314a8845698`,
archive **53,591 bytes**, SHA-256
`ab6bc6b2b026818e8b00fb8e6408a741c030e62edb752bacf8c2c56e07de76ca`.
Its CPU-only validation **29778252** actually completed DONE/LSF success with
**225 passes across all seven required modules**, no failures, errors or skips;
independent verification SHA-256 is
`5fe35da486dcad6d7c8d7b3634c0eeb2de98967193f55bb65cfc30f69370db46`.
No source05 dispatch, collector, rerender or scientific execution was performed.
The allocated-passed source05 publisher copied the **unchanged verified source04
report bytes** into the new immutable edition.

Publication time is **2026-10-21T21:07:20.278149Z** (23:07 Paris).
Published HTML SHA-256 is
`a5aae7516495443924989e47b666bca6d1b838cec3f5253b32bf64c4c5039f3d`;
report receipt SHA-256 is
`c38ee0cf56db85f3fe0ed212aab13a33809f7889e5e06c03cd87964b21108901`.
All **230 served files** returned HTTP 200 with exact expected bytes; HTTP
verification SHA-256 is
`264e5a01c5249f0f1180660503da5e31fe13cc43df9ca07e4913d033a84796c3`.

Browser presentation QA exercised actual Protenix slider drags to **2397 / 7201**,
numeric **1500**, next **1501**, middle **5000** and latest **10000**. Boltz
playback reached **2**, manual numeric **321** paused playback, previous reached
**320**, and its own latest endpoint was **765**. Chemistry interior **37** and
latest **99**, all three selectors, exact saved atom/bond heatmaps (**25 atoms /
300 unordered pairs**), raw graph labels and the visibly invalid Protenix graph
were checked. No captured arm was empty, so empty-arm clearing is covered by the
allocated fixtures rather than claimed as an actual empty-arm browser test.

Published fixed milestones are now **10000 / 0 / 0**. The ten-minute monitor
continues using each arm's latest published milestone, submitting a new saved-only
allocated CPU edition at **1500, 2500, ..., 9500, 10000** per arm and a strict
final report after all three successful terminal completions. The viewer updates
after those verified editions are published. Contact remains distinct from
affinity; changing repair totals are not one fixed objective, and a valid graph
or fixed-pose score does not establish binding or activity. No additional cleanup,
scientific setting change, native rerun or local numerical work occurred.


## Authenticated scientific and launcher bindings

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
| Boltz-2 | **29778203 RUN**; first published capture includes 766 states / 765 updates | `/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2/payload/study` |
| Boltz-2 + chemistry | **29778204 RUN**; first published capture includes 100 states / 99 updates | `/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2_chemistry/payload/study` |

The later **22:41:32 Paris** observation authenticated Boltz-2 entry storage and
its original driver invocation, with **44 committed states / 43 updates**.
Chemistry remained PEND with no study or saved steps in that observation.

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
new report fixture results. The first verified fixed edition is recorded above;
its published milestones are **10000/0/0**, with all included Protenix frames
authenticated by the later allocated report collector.

Scientific source02 remains
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source`, manifest
`1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64`.
Readiness remains in the original fixed-02 run root, not each new output root.
Configs are `brd4_jq1_25_fixed_{arm}_10000.json`. Seal independent fields for
scientific source/config/readiness roots, output parent/study root, launcher
source/adoption, actual GPU submission and CPU/observer/collector proofs.
Never derive all of these paths from one historical `tag`.

## Separate implemented report files

The fork uses the following new files in a separate immutable report-only release.
The historical files are not repinned:

| Historical code | Implemented fixed-report fork |
| --- | --- |
| `scripts/jq1_report_contract.py` | `scripts/jq1_fixed_report_contract.py`: exact actual IDs, per-arm roots and independent science/readiness/launcher pins |
| `scripts/render_jq1_25_interim.py` | `scripts/render_jq1_fixed_interim.py`: live prefixes and explicit empty/failed arm states |
| `scripts/render_jq1_25_report.py` | `scripts/render_jq1_fixed_report.py`: fixed-step metadata, counts, overlays and strict final renderer |
| `scripts/capture_jq1_completion.py` | `scripts/capture_jq1_fixed_completion.py`: fixed-step completion collector; failed-entry provenance stays explicit |
| `scripts/templates/jq1_25_report.html` | `scripts/templates/jq1_fixed_report.html`: fixed-step narrative, outcome markers and usable empty-arm selection |
| `scripts/publish_jq1_report.py` | `scripts/publish_jq1_fixed_report.py`: exact contract/job checks and a separate viewer/latest pointer |
| `scripts/hpc/jq1_25_report_stage.lsf`, `jq1_25_interim_stage.lsf` | `scripts/jq1_fixed_report_ops.py` and `scripts/hpc/jq1_fixed_report_stage.lsf`: pinned CPU/certificate/capture/interim/final stages, storage and evidence packaging |
| `tests/test_jq1_25_report.py`, `test_jq1_25_interim.py` | New allocated `test_jq1_fixed_report_contract.py`, `test_jq1_fixed_render.py`, `test_jq1_fixed_publish.py` and `test_jq1_fixed_ops.py` fixtures |

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

The implementation, preserved failed allocated attempts, source04 allocated
225-case CPU/rendering pass and source05 validated publication are recorded
above. First edition-0001 is published; later updates and strict final delivery
remain gated by their own actual evidence. All scientific
rendering and numerical fixtures remain allocated DTU work. Local work remains
static code/AST/JavaScript syntax, byte/XML/schema, transfer/publication and
browser presentation checks. The first fixed edition passed all delivery gates; its published milestones
are **10000/0/0**. Future editions retain the same delivery requirements.
