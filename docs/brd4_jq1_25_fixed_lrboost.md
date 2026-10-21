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
