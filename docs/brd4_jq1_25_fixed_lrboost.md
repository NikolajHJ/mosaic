# Higher fixed learning rates for all three JQ1-pocket runs

The user authorized stopping the current optimization runs and resubmitting
fresh 10,000-update runs on 21 October 2026. They explicitly clarified that
Boltz includes both normal Boltz-2 and Boltz-2 + chemistry/valid-output loss.

| Arm | Previous fixed LR | Authorized new fixed LR |
| --- | --- | --- |
| Protenix | 0.3 | 3.0 |
| Boltz-2 | 0.01 | 0.05 |
| Boltz-2 + chemistry/valid-output loss | 0.01 | 0.05 |

The declared version is `mosaic-brd4-jq1-25-fixed-step-lrboost-v1`.
Exactly one finite candidate is applied per update, including uphill/equal/no-op
outcomes. There is no halving, backtracking, clipping or gradient normalization.
These are fresh runs, not optimizer resumes. N25 occupied neutral slots,
50% chain/uniform initialization, key17, T1 and original JQ1-pocket objectives
remain unchanged. Chemistry stays0.001 through applied update5000, then rises
geometrically to50 at10000. No gradients through diffusion, fixed detached
Boltz context, zero recycles and no molecular-weight correction. No VAE,
decoder, DeFoG, temperature or other objective change is authorized.

All original hard parity, raw atom/bond FD scales0.01/0.003/0.001 with15% and
two adjacent scales, null directions, nonzero/independent primal, finite values,
frozen before/after, chemistry signal/coupled low/high, both chemistry ten-update
pilots, runtime and quota gates remain mandatory. Full CPU and all three native
readiness checks must rerun for the new source/configuration and actual rates;
old readiness is not a new-rate pass. Protenix uses the original authenticated
initial arrays/key and an actual single-candidate3.0 native probe.

At23:25:38 Paris the verified PEND/unstarted old report collectors29778246,
29778247 and final29778249 were intentionally retired before stopping their
native producers. Both Boltz jobs29778203 and29778204 actually exited at
23:25:47 Paris with TERM_OWNER/LSF exit130 after the user's stop request.
They are intentional partial runs, not spontaneous native failures or completed
10,000-update studies. Completed Protenix29777709 is unchanged. All artifacts
and old report servers/immutable editions are preserved, including8776 and8777.

The new immutable campaign is20261021-brd4-jq1-25-fixed-lrboost-01.
New Protenix and normal Boltz outputs are declared on work3; new chemistry
output is in its dedicated Mosaic home root. Each filesystem independently
requires all assigned full remaining new-rate pilot projections plus4GiB and
100000-file headroom. Complete bounded no-follow recent-home inventory,
authenticated fresh quota and honest physical-inode scope remain required.
Old outputs are retained and conservatively charged for stale quota.
No additional cleanup or old study movement is part of this restart.

The stop assessment is authenticated at
`/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-lrboost-stop-01/assessment.json`,
SHA-256 `8fc3ce7a6fffd0df8bb30a23ad1f37c9954beb5a572ee2b3ed8746206778800c`.
Exact local receipts are under
`scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-lrboost-01/`.

Preparation status: implementation and independent static review completed.
Immutable packaging and allocated CPU/native validation follow. At this
preparation checkpoint no new CPU/native/long job or scientific pass is established. The existing report remains a historical captured edition;
future high-rate reporting requires new actual job/config/readiness bindings.
All numerical validation/rendering stays on allocated DTU HPC; local work is
standard-library AST/JSON/byte/hash packaging and evidence verification only.


Execution checkpoint, 21 October 23:36 Paris: the 610-file source was sealed,
transferred, independently rehashed and authenticated read-only on HPC.
Manifest SHA-256 is
`9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323`;
archive 39,582,615 bytes, SHA-256
`e1a100aebf44698515ba161c1a001e338a902b250bf1a737f888df33f55098d1`.
All 145 scientific library files match the original source02 exactly. Only the
versioned contract and driver routing changed among existing production Python
files; the new immutable package adds the high-rate launchers/configs/fixtures.
The two pure storage helpers retain their exact historical authenticated bytes.

CPU validation **29778388** and its own-ended callback **29778391** were actually
submitted at 23:36:42–43 Paris. Initial work3 admission passed with 15,292,863,601
conservative available bytes against 5,368,709,120 required, including 4 GiB
reserve; finite user-quota file headroom was 19,668,302. This is the CPU admission
check, not the later full native/long campaign capacity check or a reservation.
Fresh all-three native readiness and the new pilot projections remain mandatory.


At **23:41:11 Paris**, CPU 29778388 and callback 29778391 were actual DONE.
The allocated full suite passed **598** cases: 266 launch/contract cases,
239 legacy cases, 34 native-mapping cases and 59 chemistry cases. Raw XML
shows zero failures, errors or skips; three slow/model legacy cases were
intentionally deselected as in the original suite. CPU certificate SHA-256:
`dc76df74f1432f59e4c5f6fb5491a1258752b639df893cf75bb5a6968e481630`.
The 56-file CPU/launch proof archive was downloaded, safely extracted and every
member checked locally using standard-library byte hashes. It is 62,340 bytes,
SHA-256 `d7125dff611fc3c49826ea45fe80419c05505a4c0ed37977178921be6d8bf174`.

Fresh native validations were actually submitted:

| New-rate arm | Native GPU validation | Own-ended callback |
| --- | --- | --- |
| Protenix, LR 3.0 | 29778424 | 29778425 |
| Boltz-2, LR 0.05 | 29778426 | 29778427 |
| Boltz-2 + chemistry, LR 0.05 | 29778428 | 29778429 |

All six remained **PEND at 23:42:21 Paris**. GPU validations were waiting for
physical GPU resources; callbacks awaited their own producers. These are
validation submissions, not readiness passes or long-run submissions. No new
10,000-update job ID has yet been established. The controller will authenticate
all three actual native results and fresh full per-filesystem storage before
submitting the three independent long runs. The monitor now follows this new
campaign and cannot resume the intentionally stopped old-rate jobs.


Independent CPU/launch evidence verification passed with SHA-256
`225ce416f5aabb9d549d4ecd066f2e062e66d3b92e30f05723562c46dc50cbca`.
It authenticated all 56 files, original CPU/callback scheduler and LSF success,
598 actual test rows and matching collected/passed logs, source/config/input
hashes, all six exact new submission/resource/dependency vectors, and the grouped
native-validation storage gate. That gate required 7,516,192,768 bytes against
15,292,576,076 available with 19,668,255 finite user-quota file headroom.
It covers all three disposable validations plus 4 GiB reserve; full long-run
per-filesystem storage still requires the actual new pilot projections.

The contract XML aggregate header says 269 while containing 266 passing test
rows; the unchanged historical suite similarly said 153 for 150 rows, alongside
three unchanged unittest subtests. Verification records this metadata distinction
and counts the actual rows and matching log totals only: 598, not 601.
At the final 23:45:52 Paris query all six native/callback jobs were still PEND.
Code, source identities, exact stop/submission records and current status were
pushed in `046430e`; the verification receipt is additive evidence.


## 22 October, 09:45 Paris: runs started; two terminal checks failed

All three higher-rate readiness jobs passed their original native parity,
raw-gradient, null, finite/frozen, pilot and runtime checks. Their persisted
original scheduler/LSF proofs and 254 referenced proof-file hashes are preserved.
The full independent work3/home dispatch gate and every GPU entry gate passed.
Actual fresh long submissions occurred at **01:23:09–10 Paris**:

| Arm | Fixed LR | Actual job | Latest saved outcome |
| --- | --- | --- | --- |
| Protenix | 3.0 | 29779177 | 10,000 updates, then terminal gradient failure |
| Boltz-2 | 0.05 | 29779178 | 10,000 updates, then terminal gradient failure |
| Boltz-2 + chemistry | 0.05 | 29779179 | Running; at least 8,677 updates in the captured common prefix |

Protenix started at 01:49:37 and its original same-job LSF log records exit1 at
04:00:35. Both its case report and committed logs contain 10,000 applied updates
and 10,001 states. The overall summary is **failed**, because the terminal raw
probe raised `ValueError: Predeclared epsilon leaves the simplex interior`.
Its declared terminal atom direction could not complete the original FD sweep.
The original finite-difference scales are retained; no smaller-scale substitute
is certified. Frozen model/context identities remain the original identities.

Boltz-2 started at 01:54:27 and its original same-job LSF log records exit1 at
08:17:00. It likewise saved 10,000 applied updates and 10,001 states, but its
overall summary is **failed**: `AssertionError: Native affinity E_upper finite
differences failed`. The terminal atom sweep passed all three scales. The bond
sweep reported agrees=false/false/true for 0.01/0.003/0.001, failing the original
two-adjacent-scale requirement. Full budget consumption is not a terminal
scientific pass, and neither failed outcome is relabeled or restarted.

Chemistry job 29779179 was actual RUN at 09:40:51 Paris and continued saving
updates during the capture around 09:45. Its JSONL files were read separately:
8,678 state rows through state8677 and 8,678 update rows through update8677.
The conservative shared prefix is therefore **8,677 updates / 8,678 states**.
This asynchronous observation is not a missing-production-record diagnosis or
an atomic capture. The running case summary is an older periodic snapshot.
Original full JSONL/NPZ correspondence remains an allocated report check.

Compact delivery evidence is 313,614 bytes, SHA-256
`6448daedff400ef8f66ea30675f9079063ab13b163d4b66b5e6214066c14786d`;
all 140 archive files were safely extracted and byte-verified locally. It
contains original readiness/terminal JSON, original LSF logs, exact submission
and quota records and compact saved metadata. Original large native summaries,
case reports, JSONL bytes and all NPZ frames remain authenticated HPC references,
not a claim that those originals have been downloaded or numerically checked
locally. The compact scan records original big-file hashes and counts only.

Live records for ended jobs have aged out. `bhist` execution is denied on the
login host; the first metadata-packaging attempt and its partial directory are
preserved. The second attempt records that limitation and uses original same-job
LSF records, never fabricated live scheduler statuses. This packaging limitation
does not change a scientific outcome or authorize a gate waiver. No source,
configuration, model, LR, historical artifact or active chemistry job changed.
The monitor's metadata paths were corrected to the actual `study/case-0/` logs;
this is a read-only observer fix, separate from sealed scientific source610.

The current campaign is not a successful final scientific result. Preserve the
two terminal failures and continue observing the independent chemistry run.
A new saved-only report must show these failures and its own exact captured
prefix; it must not reinterpret the old report editions or certify all three.


Independent delivery proof verification passed, SHA-256
`d939121ba93e71fccc03413d04ce93e3496624b8e81a9ebb0e54015b211ec267`.
It checked 140 listed evidence files plus the archive manifest, source610/config
bytes, CPU598 proofs, all 254 readiness references (64 original proof files
present locally and190 remaining remote-only), original native DONE/LSF records,
three actual long submissions/driver bindings and full dispatch/entry accounting.
The two terminal failures and conservative running chemistry prefix are confirmed.
All full-original, active-file and unavailable-history limitations remain explicit;
this evidence pass does not make either failed scientific run pass. New report
work is specified in `docs/brd4_jq1_25_lrboost_report_plan.md`; no higher-rate
edition has yet been published and all historical editions remain unchanged.
