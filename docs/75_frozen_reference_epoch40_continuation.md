# Guarded frozen-reference continuation through epoch 40

## Epoch40 completed; bounded branch finished, 27 October 2026 11:44 CET

[Read-only snapshot018](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-018/status.json)
and [derived summary007](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/training-status007.json)
confirm **GPU29822674 DONE at11:34:10 CET**, followed by successful callback
29822675 at11:34:39. The
[segment completion](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-018/segment-004-completion.json)
is passed, epoch-complete, absolute263840, stop reason `bounded_horizon`.
The [branch receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-018/bounded-branch-completion.json)
is `completed_bounded_branch`: **original36 through absolute40 is finished**.
This does not complete the historical global100 target or authorize epoch41.
The final checkpoint is
`/work3/s234818/latentmol/runs/source138-vae-cli-startup-fix-attempt001/segments/004/20261026T230230Z_chemistry-train_e9f28b40/epoch_0040.pt`,
SHA256 `31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e`.

Final completed-checkpoint monitoring comparison:

| Metric | Epoch39 | Epoch40 |
| --- | --- | --- |
| Raw chemical validity, fixed1,024 priors | 724/1,024 (70.70%) | 753/1,024 (73.54%) |
| Strict teacher validity | 724/1,024 (70.70%) | 753/1,024 (73.54%) |
| Exact reconstruction, fixed256 | 256/256 | 256/256 |
| Chemically valid reconstruction, fixed256 | 256/256 | 256/256 |
| Sulfur frequency | 1.4237% | 1.4394% |
| Measured epoch duration | 11.77h | 11.48h |

Epoch40 adds **29 valid outputs/+2.83 percentage points** over epoch39;
relative to original36's641/1,024 (62.60%), it adds112/+10.94points.
The [latest epoch row](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-018/segments/004/20261026T230230Z_chemistry-train_e9f28b40/latest_epoch.json)
reports measured **41,316.285s**,1,688,539 real/3,377,078 fresh-prior points,
6,596 updates, and all19 end-epoch health checks passed. Completion's
59,011.392s field remains the conservative scheduling estimate, not actual
training time. The
[terminal early panel](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-018/segments/004/20261026T230230Z_chemistry-train_e9f28b40/early_health/update_000263840.json)
at11:32:29 has all18 checks passed and zero representation failures in that
panel; previous panels had occasional representation failures. **All268 saved
early panels and all their checks passed.** These repeatedly used fixed
monitoring points and deterministic-mean reconstructions are not fresh-seed,
categorical-generation or full-test confirmation; no monotonicity is implied.

The callback stopped at40 and submitted no successor. CUDA599, loss/1:2,
optimizer/RNG, every guard and sealed identities remain unchanged. This status
collection changed no runtime code/settings/jobs and submitted no evaluation
or further training.

## Epoch40 started; early validity above epoch39 endpoint, 27 October 2026 00:17 CET

[Read-only snapshot017](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-017/status.json)
and [derived summary006](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/training-status006.json)
confirm **GPU29822674 RUN since 00:01:55 CET on n-62-18-8**. The
[actual run record](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-017/segments/004/20261026T230230Z_chemistry-train_e9f28b40/run.json)
was created at00:02:30 and restores the exact completed epoch39 checkpoint,
SHA `e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0`.
The final authorized **epoch40 is underway**, not complete. Callback29822675
is PEND on its expected dependency; the endpoint remains absolute40.

The [latest saved health panel](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-017/segments/004/20261026T230230Z_chemistry-train_e9f28b40/early_health/update_000257356.json)
at **00:16:41 CET** is absolute257356, **112/6,596 updates (~1.70%) intoepoch40**.
This is last-persisted progress, not the exact live counter. On the same fixed
1,024 prior latents, **raw chemical validity is744/1,024 (72.66%)**, versus
epoch39's724/1,024 (70.70%): **+20 outputs/+1.95 percentage points**.
Strict teacher validity is743/1,024 (72.56%), with **one representation failure**;
its gain over epoch39's724 is19 outputs/+1.86 points. Exact and chemically valid
reconstruction remain256/256. Sulfur is1.5048%, versus original36's1.5070%.
Both new panels and all **203** saved panels passed every health check.
These are early fixed-panel observations, not a completed checkpoint result,
monotonic improvement guarantee or fresh/full-test confirmation.

The source138599-case CUDA pass, loss/1:2/optimizer/RNG settings, all guards
and endpoint40 are unchanged. This collection changed no runtime code,
settings or jobs and submitted no additional experiment. Epoch39 remains
the latest completed checkpoint.

## Epoch39 complete; final epoch40 queued, 26 October 2026 22:39 CET

[Snapshot016](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-016/status.json)
and [derived summary005](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/training-status005.json)
confirm **epoch39 completed successfully at 16:50:10 CET**. The
[completion receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-016/segment-003-completion.json)
is passed, epoch-complete, absolute update257244, checkpoint SHA256
`e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0`.
Its [measured epoch duration](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-016/segments/003/20261026T040252Z_chemistry-train_40cdfc66/latest_epoch.json)
was **42,354.638s (~11.77h)**. The 59,011.392s completion field remains the
conservative estimate; the normal whole-epoch budget stop was not a failure.

Completed-checkpoint monitoring results are:

| Checkpoint | Raw valid on fixed1,024 priors | Exact / valid reconstruction on fixed256 |
| --- | --- | --- |
| Original36 | 641 (62.60%) | 256 / 256 |
| 37 | 705 (68.85%) | 256 / 256 |
| 38 | 724 (70.70%) | 256 / 256 |
| 39 | 724 (70.70%) | 256 / 256 |

The [epoch39 terminal early-health receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-016/segments/003/20261026T040252Z_chemistry-train_40cdfc66/early_health/update_000257244.json)
also reports strict724/1,024, sulfur **1.4237%**, and eighteen passed checks.
**All201 saved early-health panels and all end-epoch checks passed.** The
final70.70% is below the earlier intermediate74.51% observation; that earlier
value is not the completed checkpoint's performance. Counts fluctuate, and
equal aggregate epoch38/39 counts do not establish identical outputs or absence
of learning. These fixed panels are not independent/full-test confirmation.

The [automatic callback decision](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-016/advance-after-segment-003.json)
submitted the final **GPU29822674** at **16:51:02 CET**, plus callback
**29822675**, after completion/health/source/resume/quota and
[exact-command checks](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-016/launch-command-004.json).
The queued resume SHA matches the completed epoch39 checkpoint exactly.
At **22:39:53**, both jobs are **PEND**. The GPU waits for enough processor
units to satisfy affinity on one host; **four of sixteen CPU slots are reserved
on n-62-18-8**. Its callback waits on its expected dependency. Reservation is
not execution, no epoch40 directory/setup exists, and no reliable start ETA
is established. Existing exclusive80GB A100/16-slot/24h/23h-internal settings,
599 CUDA pass, every scientific guard and the absolute40 endpoint remain
unchanged. The collection changed no runtime code, settings or jobs, and did
not submit an additional experiment. Epoch39 is now the latest completed
checkpoint; the authorized branch is not yet complete.

## Epoch38 complete; epoch39 running with latest validity74.5%, 26 October 2026 12:33 CET

[Snapshot015](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-015/status.json)
and [derived summary004](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/training-status004.json)
confirm **epoch38 completed successfully**. Segment002's saved LSF log ends
**25 October21:37:32 CET**, successful; its
[completion receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-015/segment-002-completion.json)
is passed, epoch-complete, absolute update250648, checkpoint SHA256
`1736ad792f244fda59d0b74dc87867880fe632db0417ddfb0c70d07269fd319d`.
Measured epoch38 duration was **42,905.556s (~11.92h)**. The receipt's
`epoch_seconds=59,011.392s` remains the conservative scheduling estimate.
Its normal whole-epoch budget stop was not a failure. Epoch38's fixed raw
prior validity was **724/1,024 (70.70%)**, with exact/valid reconstruction
**256/256**, and it is now the latest completed checkpoint.

The [automatic advance decision](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-015/advance-after-segment-002.json)
submitted **GPU29817247/callback29817248** after the existing health/source/
resume/quota and exact-CLI checks. GPU29817247 is **RUN since26 October
05:02:28 CET on n-62-18-9**. Its
[run record](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-015/segments/003/20261026T040252Z_chemistry-train_40cdfc66/run.json)
starts epoch38 and restores exactly the completed checkpoint SHA above;
**epoch39 is underway**. Callback29817248 remains PEND normally.

The latest saved panel at **12:23:50 CET** is absolute update254756:
**17,300 updates since original36, or4,108/6,596 (~62%) intoepoch39**. It
passed all18 health checks. Latest raw and strict validity are **763/1,024
(74.51%)**, versus original36's641/1,024 (62.60%); exact/valid reconstruction
remains **256/256**, latest sulfur **1.5057%** versus1.5070% baseline.
**All176 persisted early-health panels passed every check.** This does not
mean every generated graph passed: the preceding panel at254656 recorded
737 raw/736 strict valid and one representation failure. Latest representation
failures are zero. The last three raw/strict pairs are753/753,737/736,763/763;
fixed-panel improvements are not monotonic or independent/full-test evidence.

[Latest raw panel](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-015/segments/003/20261026T040252Z_chemistry-train_40cdfc66/early_health/update_000254756.json)
and the snapshot preserve these observations. Counters are last-persisted,
not exact live progress. No epoch39 completion exists yet. Existing599 CUDA
passes, loss/1:2/optimizer/RNG, all guards and the absolute40 endpoint remain
unchanged. This read-only check made no runtime edit, job submission or setting
change; independent studies retain their separately bound checkpoint identities.

## Epoch38 past halfway; latest fixed-panel validity71%, 25 October 2026 16:29 CET

[Read-only snapshot014](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-014/status.json)
and [derived summary003](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/training-status003.json)
confirm **GPU29809056 remains RUN**, with callback29809057 PEND on its expected
dependency. The latest saved panel, **16:20:42 CET**, is absolute update247556:
**10,100 updates since original36, or3,504/6,596 (~53%) into epoch38**. This is
the last persisted observation, not the exact live counter. Epoch38 has not
completed; epoch37 remains the latest completed checkpoint.

The fixed1,024-prior panel has **727 raw and strict valid outputs (71.00%)**,
versus **641/1,024 (62.60%)** at original36 and705/1,024 at the preceding status
check. Exact and chemically valid reconstruction remain **256/256** on the
fixed monitoring panel. Latest sulfur frequency is **1.4730%**, below the
1.5070% baseline, with zero representation failures on this latest panel.
**All103 saved early-health panels passed all their checks**. The last three
raw/strict counts are715,715,727/1,024; these repeated panels fluctuate and do
not establish independent-seed or full-test performance.

The [latest raw health receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-014/segments/002/20261025T084058Z_chemistry-train_71cc7095/early_health/update_000247556.json)
retains all eighteen checks. The source138 CUDA599 pass, unchanged scientific
settings/guards and absolute40 endpoint remain in force. No numerical rerun,
runtime-code edit, setting change or manual job submission occurred during
this status check.

## Epoch37 completed; automatic epoch38 continuation running, 25 October 2026 11:39 CET

The [read-only snapshot013](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/status.json)
and [derived training summary002](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/training-status002.json)
confirm that **epoch37 completed successfully**. Segment001's saved LSF log
records successful termination **24 October 23:43:24 CEST**, following its
10:11:51 start. Its [completion receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/segment-001-completion.json)
is passed, epoch-complete, absolute update244052, checkpoint SHA256
`eb6ecd25c36cff6fb158e2d24aa1888fdc17e8c8654bb100894d834510839446`.
The live scheduler records for that completed allocation/callback have aged
out; the saved successful logs and authenticated completion remain valid.

Epoch37 processed **1,688,539 real molecules and 3,377,078 fresh-prior points**
in **6,596 updates**, retaining the 1:2 policy. The measured epoch duration in
[`latest_epoch.json`](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/segments/001/20261024T081349Z_chemistry-train_f9da1714/latest_epoch.json)
is **48,502.236s, approximately13.47h**. Its completion receipt's historically
named `epoch_seconds` is **59,011.392s**, the conservative scheduling estimate
assigned by the controller, not the measured epoch duration. The allocation
ended at the normal `wall_budget_epoch_boundary` because another whole epoch
would not fit the remaining guarded budget; this was not a failed training run.

The [serial advance decision](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/advance-after-segment-001.json)
automatically submitted **GPU29809056** and callback **29809057** after passed
completion/health/source/resume/quota checks and a second-segment actual-CLI
dispatch pass. **GPU29809056 is RUN**, started **25 October09:40:30 CET** on
**n-62-18-8**, with the unchanged exclusive80GB A100/sixteen-slot/24h request
and23h internal cap. [Its actual run record](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/segments/002/20261025T084058Z_chemistry-train_71cc7095/run.json)
starts at epoch37 and restores exactly the completed checkpoint SHA above.
**Epoch38 is underway**; its callback remains PEND normally. The branch still
ends at absolute40 and retains all scientific settings and checks.

At the saved observation, **all77 persisted early-health panels passed every
check**, including the latest eighteen-check panel saved **11:32:03 CET** at
absolute update244956: **7,500 updates since original36, or904 into epoch38**.
These are last-persisted counters, not an exact live count.

| Fixed monitoring metric | Original36 baseline | Completed epoch37 | Latest epoch38 panel |
| --- | --- | --- | --- |
| Raw valid out of1,024 | 641 (62.60%) | 705 (68.85%) | 705 (68.85%) |
| Strict teacher valid out of1,024 | 641 | 705 | 705 |
| Exact reconstruction out of256 | 256 | 256 | 256 |
| Chemically valid reconstruction out of256 | 256 | 256 | 256 |
| Sulfur frequency | 1.5070% | 1.4543% | 1.4696% |

The latest panel's representation-failure count is zero; this does not assert
zero representation failures across every historical panel. Validity fluctuates
(the last three panels were736,695,705/1,024), rather than improving monotonically.
These repeated fixed panels demonstrate the required monitored retention; they
are not a new independent-seed experiment or a full held-out-test guarantee.
The **599 mandatory CUDA passes** remain established from this release's
first allocation. No new numerical run, runtime-code edit, setting change or
manual resubmission was performed by this status check. Only local read-only
collector coverage/reporting was extended: all health receipts retain their
full-file SHA and compact statistics, while raw downloads are limited to the
latest two per run and any failure; older raw receipts remain on the HPC.

Additional raw evidence:

- [Epoch38 latest saved health panel](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/segments/002/20261025T084058Z_chemistry-train_71cc7095/early_health/update_000244956.json)
- [Segment002 exact launch-command check](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/launch-command-002.json)
- [Segment002 GPU receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/receipts/segment-002-receipt.json)
- [Next callback receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-013/receipts/after-segment-002-receipt.json)

The completed epoch37 checkpoint is now the latest training checkpoint;
independent embedding/Protenix studies retain their originally bound models.

## Actual continuation started and CUDA passed, 24 October 2026 10:55 DTU-local

[Snapshot012](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-012/status.json)
and the [derived training summary](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/training-status001.json)
supersede the queued observations. **GPU 29802716 is RUN**, started
**10:11:51 on n-62-18-10**, one exclusive 80GB A100/sixteen CPU slots. All
**599 current-release mandatory CUDA fixtures passed**, zero failures,
errors or skips, **76.303s**. This is the new source138 launch result, distinct
from failed source136's earlier passed fixtures.

The actual training record was created at **10:13:49**, starting epoch36 from
the exact original checkpoint SHA
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
The live run is
`segments/001/20261024T081349Z_chemistry-train_f9da1714`, with status running
and branch endpoint40. Native update/quarter-AdamW-displacement receipts and
fixed health panels confirm actual optimization, not merely CUDA fixtures,
setup or the earlier zero-update dispatch harness.

**Epoch37 is in progress.** The last persisted health panel at the observation
is **300 updates into continuation**, absolute update237756, saved
**10:51:32**. All **18 health checks passed at every saved panel**:

| Updates since original36 | Raw / strict valid on fixed 1,024 priors | Exact / valid reconstruction on fixed 256 | Representation failures | Sulfur frequency |
| --- | --- | --- | --- | --- |
| 0, authenticated original36 baseline | 641 / 641 | 256 / 256 | 0 | 1.5070% |
| 32 | 642 / 642 | 256 / 256 | 0 | 1.4917% |
| 100 | 648 / 648 | 256 / 256 | 0 | 1.4816% |
| 200 | 659 / 659 | 256 / 256 | 0 | 1.4644% |
| 300 | 666 / 666 | 256 / 256 | 0 | 1.4665% |

Thus the fixed-panel raw/strict validity is **62.60% -> 65.04%**, with
reconstruction retained on that panel. These observations are early training
health, not an epoch-complete or full-test result, statistical generalization
claim or an independent-seed replication of source135's proof. The panel
counter is not an exact live update count. Original epoch36 remains the latest
**completed** checkpoint; epoch37 has not completed. Serial callback29802717
is still PEND on its expected dependency. The loss, 1:2 ratio, optimizer/RNG,
endpoint40 and every existing check remain unchanged. This read-only collection
changed no runtime code, settings or jobs; only collector coverage/documentation
was extended to retrieve the saved health receipts.

Evidence:

- [CUDA gate](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-012/training-launch-cuda-gate.json), SHA256 `adf8b3bbf47bf6b5c3afbec0122929d03cbff3d01357f1a2d772d0ee16f8b504`
- [599-case CUDA JUnit](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-012/training-launch-cuda-pytest.xml), SHA256 `b010fa7a5fd945afa71387fb1da8b65fe77f2c22b04abb3e39eb95ed6ba55bd6`
- [Actual running training record](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-012/segments/001/20261024T081349Z_chemistry-train_f9da1714/run.json)
- [Latest 300-update health receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-012/segments/001/20261024T081349Z_chemistry-train_f9da1714/early_health/update_000237756.json), SHA256 `5637ec895cd41d84fbab16f4e9c396b08aeb94bb74c12d46958f1c3a62f7be4a`

**Latest read-only check, 24 October 2026 01:30:28 DTU-local:**
[snapshot010](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-010/status.json)
confirms **GPU 29802716 remains PEND**, reporting job-slot limits on five
hosts. Eligible pending time is 14,365 seconds, ineligible time zero; the
serial callback **29802717** waits on its expected `ended(29802716)` condition.
There is no current-release CUDA gate/JUnit, GPU execution log, segment
directory or training progress. **The CUDA launch checks have not run and
actual continuation has not started; original epoch36 remains latest.**
No reliable start ETA is established. Earlier completed CPU/callback/targeted
job records have aged out of LSF; their saved DONE logs and passed full CPU,
repair profile/adoption and two actual-command dispatch checks remain valid.
The read-only observation changed no runtime code, settings or jobs.

**Latest read-only check, 23 October 2026 21:33:51 DTU-local:**
[snapshot009](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-009/status.json)
confirms GPU **29802716 PEND**, now reporting **job-slot limits on five hosts**;
serial callback **29802717 PEND** on the expected dependency. Full CPU,
profile, adoption and both actual-command dispatch passes remain established.
No CUDA launch artifact, actual training setup or epoch37 exists yet. This is
a resource wait, not an invalid submit argument or failed dependency. No
reliable start ETA; no code, settings or jobs were changed by the check.

## Source138 full CPU passed and GPU continuation resubmitted, 23 October 21:31

The [21:31:21 DTU-local snapshot](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/status.json)
supersedes the intermediate running states. **CPU29802660 DONE**, full gate
passed: **1,998 passes / 113 CUDA-only skips / zero failures/errors**, followed
by a passed allocated repair-execution profile and unchanged complete proof
adoption. The profile preserved identical ordered teacher semantics across
reference, optimized serial and 4-/12-worker execution; its four-real/32-prior
cohort is diagnostic only, not the training 1:2 ratio. Original epoch36 and
reference35 checkpoint bytes were authenticated.

**Callback29802661 DONE**, 21:30:13–21:31:05. It reauthenticated all gates,
the complete source135 proof, 2,500 updates and health evidence, and reran the
real CLI dispatch check on the exact command about to be queued. That command
is byte-for-byte identical as an argv list to the earlier targeted test. Its
second dispatch receipt has SHA256
`2351237631003afc2d17966e2bcb4e7097d278f9778a4d1f6c31ed346054194e`.
It intentionally stopped before training setup and made zero updates.

**GPU segment001 job29802716 submitted at 21:31:04**, `gpua100`, one exclusive
80GB A100, sixteen CPU slots, 24h wall limit/23h internal limit. Its unchanged
serial after-segment callback is **29802717**, submitted21:31:05. At the
saved observation the GPU is **PEND**, eligible and waiting for scheduling;
the callback is PEND on `ended(29802716)`. No reliable start ETA. There is
**no current-release CUDA gate, actual training setup or epoch37 yet**.
The job retains mandatory CUDA fixtures before `&&` the unmodified trainer;
only passed training completion/health permits each serial successor, stopping
at absolute40. Loss, sampling ratio, optimizer/native/fresh RNG and every
existing scientific/health/runtime/resume/quota check remain unchanged.

The [derived submission audit](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/startup-fix-submission-summary001.json)
verifies source136/new plan equality outside fresh names/identities, all six
overlay/archive hashes, actual-command equality and the recorded GPU
resources/CUDA-before-train wrapper. Raw evidence:

- [Full CPU gate](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/cpu_gate.json), SHA256 `80ca107bb7bc5b29a8210532bdb1fd1fb9293ad864c0b490027ea7d1afe35f6e`
- [Full CPU JUnit](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/guarded-cpu-pytest.xml), SHA256 `743151c158b4db93e7b26162b0242041e852d58d456bf5ec6a00954dc10245f2`
- [Repair execution profile](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/repair_profile.json), SHA256 `e34fe4b736870e1b52b1346e5c00d67b8c6de541e2f65f16badfe6f1d4bbcfe6`
- [Production callback's actual-command receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/launch-command-001.json)
- [Conditional training decision](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/advance-after-benchmark.json)
- [GPU submission receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/receipts/segment-001-receipt.json)
- [GPU exact command/resources](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-008/receipts/segment-001-intent.json)

The source136 failure and all original proof/bridge/Protenix evidence remain
untouched. This fix resolves its CLI startup rejection, not a demonstrated
new training or chemical-quality result.

## Source138 staged; actual startup command passed, 23 October 2026 21:26 DTU-local

New immutable `source138-vae-cli-startup-fix-attempt001` staged/authenticated
**648 members / 577 scientific members**, return zero. Its six explicit
overlays preserve the entire continuation AST except the one main predicate;
every other science/configuration/plan/check field matches sealed source136.
The [preparation](../outputs/hpc-submissions/source138-vae-cli-startup-fix-attempt001-preparation.json)
and [stage/submission receipt](../outputs/hpc-submissions/source138-vae-cli-startup-fix-attempt001-stage-and-queue-001-output.json)
bind:

| Identity | SHA256 |
| --- | --- |
| Archive | `3dd998a9953aa4d61dddc38af92b7779221308edd0a5a75140a4475d1273259a` |
| Scientific manifest | `c702a81e665ad8b77b19b37ba5d3c19ee1595f5877e3cc926a6811024a279a84` |
| Immutable plan | `183abba4bcd8e185b407ccdb5596a8fd6c37c0f9a82aeaf7c21e28a9cdb219c4` |

Full CPU gate **29802660** and callback **29802661** submitted at 21:22:15/16.
CPU began at **21:23:27 on n-62-21-90**, sixteen slots/one hour; the 21:26
snapshot is RUN, not yet a full pass. The earlier provisional no-sooner-than
03:33 estimate was superseded by actual execution. Callback remains dependent
on CPU completion and must independently authenticate its passed gate/profile.

To test startup promptly without weakening that chain, separate allocated
two-slot/fifteen-minute **29802690** ran **21:25:12–21:25:50**, DONE. All
**81 targeted CLI/adoption/policy fixtures passed**, no failures/errors/skips,
5.230s. After those fixtures, the targeted job deterministically materialized
the unchanged source135 proof adoption; the full CPU stage must recompute and
authenticate identical receipts. Their presence cannot satisfy `require_cpu`.

The targeted script extracts the exact first-segment command list expression
from sealed `Campaign.advance`, instantiates it with the actual new plan and
derived benchmark/readiness hashes, and executes the real module entrypoint
through the allocated dispatch harness. **Exit zero, passed dispatch**, with
start36/end40/segment1 and both required hashes accepted. Full real-plan,
source/checkpoint/proof/stability/health authentication ran; **model setup did
not run and updates=0**. This reproduces the boundary that failed in source136
without consuming GPU time or modifying the checkpoint. Evidence:

- [Targeted submission](../outputs/hpc-submissions/source138-vae-cli-startup-fix-attempt001-targeted-startup-submission001.json)
- [Targeted result](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-003/control/targeted-startup-result001.json)
- [Actual generated-command receipt](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-003/control/targeted-launch-command001.json), SHA256 `6f7dfaf7084816e2f3426265cba1e56bd5dc501d777efe24596be27a3a4eb68e`
- [81-case JUnit](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-003/control/targeted-startup-pytest001.xml)
- [Read-only scheduler/log snapshot](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/snapshot-003/status.json)

Derived benchmark SHA is
`255d0763ff0c118fe691f9e059f606584b0a8151dc8c0b4841f286b3aceea932`;
readiness SHA is
`f4e5f1cab226c7ca46f159189f4350fd6700b19405ea639e4fbdafa758c48e1b`.
These explicitly reuse the completed original proof, not a new native pass.
The callback repeats the exact-argv check before GPU submission. Mandatory
current-release CUDA fixtures, actual model/optimizer/RNG setup and training
health remain pending at this observation. No epoch37 or training-start claim.
The full 36->40 chain is submitted conditionally; no GPU training job ID yet.

Collector snapshot002 failed only its initial four-megabyte metadata download
bound on the existing adopted benchmark artifact; it changed no remote file.
Snapshot003 preserves that raw remote file's SHA/size/status without repeatedly
downloading its large diagnostic payload. It is not a scientific gate failure.

## Authorized startup correction: source138, 23 October 2026

The user requested the CLI fix, an actual launch-command check and a fresh
original-36 through absolute-40 submission. Source136 is preserved as failed;
its passed 586 CUDA fixtures do not establish successful training startup.

The correction in `chemistry_continuation.main` uses the same readiness
applicability as `Campaign.advance` and `require_failure_readiness`: explicit
failure guidance, short continuation, or valid-reference continuation. Missing
readiness/benchmark hashes, legacy overrides, benchmark overrides and unpaired
resume arguments still fail. No loss, model, optimizer or numerical training
body changes. The adoption reader admits exactly this predicate AST change;
all other edits to main remain forbidden.

[`tests/test_chemistry_launch.py`](../tests/test_chemistry_launch.py) exercises
the real CLI's dispatch and rejection paths. The campaign also invokes
[`check_chemistry_launch.py`](../scripts/hpc/check_chemistry_launch.py) from
its allocated callback on the **exact command list being submitted**. The
harness executes the real module entrypoint via `runpy`, with actual plan,
release, benchmark/readiness, original-checkpoint and complete source135
adoption/stability/health authentication. A trace boundary stops at entry to
`train_phase`, before its body, model setup or any update. This writes
`launch-command-NNN.json`, explicitly `passed_dispatch_not_training`.
Nonzero exit/missing or altered receipt blocks GPU submission. This is a
dispatch/receipt test, not a new CUDA/native or chemistry-health observation.
The GPU still runs every original mandatory launch check before unmodified
training setup. Serial successors repeat the argv check and every old guard.

Source138 is built from sealed source136, with only the startup predicate,
exact-source audit and launch regression/preflight overlays. Science and plan
fields remain identical except fresh release/campaign/archive/manifest/name.
It adopts the same completed source135 proof, restores authenticated original
epoch36 (not proof state), keeps 1:2 real/prior, full repair NLL and frozen36
reference weight0.1, quarter actual AdamW displacement/native moments/RNG,
all 18 health checks and 32/every100 fixed panels. Endpoint40 and serial
24h/23h internal limits remain unchanged.

The [read-only preflight](../outputs/hpc-results/source138-vae-cli-startup-fix-attempt001/preflight001.json)
at **21:20:01 DTU-local** verified protected checkpoint and all adopted proof
hashes, confirmed the failed source136 jobs inactive and observed quota
62.54/100 GiB used (37.46 free). Packaging and allocated CPU/GPU results are
not yet established by this preparation entry.

## Terminal launch failure after CUDA passed, 23 October 2026 21:02 DTU-local

[Status017](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-017.json)
and [additional details](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-017-details.json)
supersede the queued snapshots below. Persisted LSF logs show GPU **29792164
ran 14:02:45–14:04:39 DTU-local on n-62-18-8**, then the wrapper exited 1.
All **586 current-release CUDA fixtures passed**, zero failures/errors/skips,
in 82.95s. The saved CUDA gate is `passed`, SHA256
`912f41264f0f6a1ae8ee12b6976fbfeb1552eb030db9f12166f25a3b038931b6`.
This is fixture evidence, not successful actual training setup or an update.

The subsequent real training CLI exited 2 with:

```text
Historical chemistry protocols have no failure-readiness override
```

Source inspection identifies the mismatch: the
[campaign launcher](../scripts/hpc/chemistry_campaign.py) supplies
`--failure-readiness-sha256` for `valid_reference_continuation`, but
[`chemistry_continuation.main`](../src/latentmol/chemistry_continuation.py)
recognizes that argument only when `protocol.failure_guidance` is present.
The frozen-reference continuation intentionally has no such failure-guidance
field. Its downstream readiness function already recognizes the continuation
marker, but the earlier CLI rejects it before reaching that function.
**No actual continuation setup, segment, model update or epoch-37 checkpoint
exists; original epoch 36 remains latest.** No training chemistry-health or
retention panel was reached.

Callback **29792165 ran 14:04:40–14:04:52 and exited 1**, correctly stopping
on predecessor EXIT; no successor was submitted. The live scheduler records
have aged out, so persisted LSF/allocation logs establish these outcomes.
Missing scheduler records alone are not failure evidence. Earlier CPU,
CUDA and source135 scientific-proof passes remain preserved as scoped passes.

The [derived summary](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-017-summary.json)
SHA256 is `7dbc5d003582ccb9588b8e99100cffae416140789610d3288b55abe919093864`.
It separately records the completed [source137 bridge pilot](76_epoch36_boltz_embedding_bridge.md).
This status turn changes no runtime code, plan, gate or job and submits no
replacement. Preserve failed immutable source136/callback evidence. A future
correction needs a new authenticated release and targeted actual CLI/launch
coverage, keeping objective, state lineage and every guard intact.

Latest read-only observation, **23 October 2026 12:32 DTU-local**:
[status-016](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-016.json)
confirms **GPU 29792164 PEND**, all 16 CPU slots reserved on n-62-18-10
(latest reservation timestamp 12:17:12), job-slot limits on four hosts and
zero ineligible pending time. **Callback 29792165 PEND**. No CUDA launch
gate/JUnit, segment directory, training run or progress exists. Earlier
CPU/profile/adoption passes remain preserved; aged-out CPU scheduler entries
are not failures. Original epoch 36 remains the latest completed checkpoint.
The independent bridge GPU 29796624 is also PEND with four reserved slots
and no GPU outputs. The account has one running and 14 pending jobs across
projects; the running Mosaic L40S job is not this VAE. No reliable start
estimate is displayed and no settings or jobs changed.

Latest start-time observation, **23 October 2026 10:22 DTU-local**:
[status-015](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-015.json)
records GPU **29792164 PEND**, all **16 CPU slots reserved on n-62-18-10**
and job-slot limits on four hosts. The prior 12-slot reservation and provisional
24 October 04:19:11 forecast are superseded: the latest output displays no
start timestamp. No reliable new ETA, CUDA pass or training start is established.
Callback 29792165 is still PEND. No jobs/resources/settings were changed.

Latest observation, **23 October 2026 10:10 DTU-local**:
[status-014](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-014.json)
confirms GPU **29792164 PEND** and callback **29792165 PEND**, with no CUDA
launch gate/JUnit or actual training run/progress. Job-slot limits persist on
five hosts. LSF has reserved 12 CPU slots toward the 16-slot request and
reports a provisional **no-sooner-than 24 October 04:19:11 DTU-local**
timestamp. This forecast can change; reservation is not execution. Earlier
CPU/profile/adoption results remain passed. The
[account snapshot](../outputs/hpc-results/account-queue-20261023-001.json)
lists four queued and none running. This read-only check changed no job,
resource or scientific policy.

Latest observation, **22 October 2026 20:53 DTU-local**:
[status-013](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-013.json)
confirms GPU **29792164 PEND**, still awaiting job slots on five hosts.
Callback 29792165 is PEND. No CUDA launch results or actual training progress
are present; earlier CPU/profile/adoption passes remain unchanged. This was a
read-only check with no job or scientific changes.

Latest read-only observation, **22 October 2026 20:38 DTU-local**:
[status-012](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-012.json)
shows **GPU 29792164 PEND**, reporting job-slot limits on five hosts with
eligible pending time. Callback 29792165 is PEND. No CUDA launch gate/JUnit
or actual training run/progress exists yet. CPU/callback scheduler records
have aged out; their previously saved DONE and passed CPU/profile/adoption
evidence remains unchanged. No settings or jobs were changed by this check.

Status, 22 October 2026 19:49:43 DTU-local: **source136 CPU 29792116 and callback
29792117 DONE**, full
CPU gate/repair profile/additive adoption passed; **1,985 unit passes/113
CUDA-only skips**, zero failures/errors. Callback 29792117 submitted **training
segment 001, GPU 29792164 PEND**, after-segment **29792165 PEND**. No new CUDA
launch result or actual training start/epoch 37 yet. Release:
`source-136-valid-reference-epoch40-continuation-attempt001`; full identifier:
`source136-valid-reference-epoch40-continuation-attempt001`. Queued training is
not completed launch readiness or a training start; mandatory checks remain.

The user explicitly asked to package and submit a guarded continuation using
the now-validated source135 frozen-reference loss, from **original epoch 36
through absolute epoch 40**. This is four additional epochs, within the global
100-epoch maximum. The endpoint **40 supersedes 39 only for this new training
authorization**. Historical epoch-39 short-branch policies, sealed test plans,
stopped decisions and failed experiments remain unchanged. The separate
[Protenix study](73_protenix_soft_occupancy_study.md) is independent.

## Latest queued state: resource wait, not dependency failure

The [17:49:43 UTC / 19:49:43 DTU-local status-011](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-011.json)
confirms **CPU 29792116 and callback 29792117 DONE**. **Training segment 001,
GPU 29792164 PEND**, waits for eligible resource/core availability on an 80GB
host, **not a dependency failure**. **After-segment 29792165 PEND**. All prior
CPU/profile/adoption passes and source/plan/proof bindings remain unchanged.
There is no new CUDA launch result or actual training state/update in this
snapshot; original epoch 36 is still the latest completed checkpoint. No
training-start or epoch-37 completion claim follows from a submitted job.

## CPU/adoption passed; training segment 001 submitted

The [17:34:54 UTC / 19:34:54 DTU-local status-008](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-008.json)
records **CPU 29792116 DONE**, full CPU gate and repair profile passed, unit
JUnit **1,985 passes/113 CUDA-only skips, zero failures/errors**. The allocated
additive adoption is **passed_evidence_adoption_not_new_native_test**, verifying
source135's completed evidence and unchanged scientific bindings. Its launch
decision is **training_submitted_not_passed**: **GPU segment 001 job 29792164
PEND**, after-segment **29792165 PEND**. The callback has submitted jobs; the
snapshot does not establish completion of callback 29792117 itself.

All following SHA256 values bind new source136 records in its own campaign;
source135 records and job identities remain unchanged:

| New record | SHA256 |
| --- | --- |
| `cpu_gate.json` | `2097edf9bf62de430c2ebc6e3028fae5152305a8473a47d6d0678f3033aba316` |
| `repair_profile.json` | `73a5339c53551a1affe3c6707a739116a96ebce74c1a2461b97e1136be202fc3` |
| `adoption.json` | `5b140b37ceedf09f4fce7f4626a6ee8b77c05cf04ab8742f8e674bf4160c9c30` |
| `advance-after-benchmark.json` | `3a1a46ebc063173ac60185ed4826537861f8a03d9657cc0cc506a73676441e6c` |
| Derived `benchmark_gate.json` | `c6a5ef06cf2c39cf2f808ac23404224cfef15f290cbd9157f53f323fb04a4281` |
| Derived `failure_readiness.json` | `33f7cfcc6012a01cc99910e3f9b20f13152437f0d14616385a33a001ff6c72ba` |
| Derived `stability_gate.json` | `3cface9126100ebf7b84d768d8c03ea6489c78bbc033b8622b9be2de9f808db9` |

The first GPU allocation still requires full current-release CUDA/launch
fixtures, at least 518 cases with zero failures/errors/skips, before actual
original-36 training. **No new CUDA test or training update has started in this
snapshot**. All serial/24h/23h guards and absolute epoch-40 cap remain.

The repair profile inherits a misleading hardcoded `training ratio 1:3` note.
Preserve that artifact unchanged. Its **four-real/32-prior diagnostic cohort**
tests execution parity, not real training composition or new native chemistry
health. The authenticated plan still has **prior_ratio 2, 256 real/512 prior,
1:2**, with unchanged source135 loss. No scientific correction or rerun is
needed for this legacy text.

## CPU unit prefix passed; profile/adoption pending

The [19:32:15 DTU-local saved status-006](../outputs/hpc-results/source136-valid-reference-epoch40-continuation-attempt001/status-006.json)
records **CPU 29792116 RUN**, started **19:27:49** on **n-62-21-7**, and callback
**29792117** waiting. Unit JUnit has **1,985 passes, 113 CUDA-only skips, zero
failures/errors**. The repair-profile/adoption stages are not yet final;
unit success does not establish the full CPU gate or authenticated adoption.
No GPU training job is submitted. This actual RUN supersedes any earlier
provisional queue forecast; an attempted reservation helper aborted before
mutation when it found the CPU job had already started, so no reservation changed.

## Actual source136 seal, staging and CPU submission

The [sealed preparation](../outputs/hpc-submissions/source136-valid-reference-epoch40-continuation-attempt001-preparation.json)
binds **646 archive members, 575 scientific members, 12 Python overlays** and:

- Archive SHA256 `4833072e286c39e9233f4bd776cd169623bb1496a14e945c296116527cc0be69`.
- Manifest SHA256 `6c042554002f57bbb3cb1eac4b6280333a003a35c2ed24f7610cb8a95a718633`.
- Plan SHA256 `8edfbe52647a9ed0475c2762775bcd9d6a2c286268c92f817d2a757050b688e9`.

The [actual stage-and-queue receipt](../outputs/hpc-submissions/source136-valid-reference-epoch40-continuation-attempt001-stage-and-queue-001-output.json)
records **646 authenticated members**, stage/queue return code zero, **CPU
29792116 / callback 29792117**, `submitted_not_passed`. Its remote release is
`/work3/s234818/latentmol/releases/source-136-valid-reference-epoch40-continuation-attempt001`;
control/campaign use the full identifier under `submissions/` and `runs/`.
No CPU result, repair-profile pass, adoption pass or training job is established
by that submission receipt.

The sealed top-level policy is
`valid_reference_continuation={version:"epoch36-valid-reference-training-v1",start_epoch:36,end_epoch:40,purpose:"validated_reference_training"}`.
The separate `valid_reference_adoption` marker has version
`validated-frozen-reference-adoption-v1`, exact source135 plan SHA
`ad7533a6ce9011c17ba8c962fe939f1e3391b77b73c4ae7574bdbebaf203c609`,
terminal proof SHA `218a3c9e0318286f9122017cf9c4507885202ed7981bea61852d0c16886a39b0`
and an explicit artifact-to-SHA map. It authenticates old results in a new
additive record; old source135 jobs, decisions and seal remain unchanged.

The saved queue argv uses explicit
`/work3/s234818/latentmol/env/boltz311/bin/python`, the release's
`scripts/hpc/chemistry_valid_reference_campaign.py`, sealed plan path,
`--plan-sha256 8edfbe52647a9ed0475c2762775bcd9d6a2c286268c92f817d2a757050b688e9 --mode queue`.
CPU contracts/repair profile/adoption must pass before serial GPU segment 1;
that segment requires the full current-release CUDA/launch fixture list,
**at least 518 cases with zero failures/errors/skips**, before actual training.
Staging itself is not a contract pass; the later CPU/adoption pass is recorded
above. No fresh GPU benchmark or training start is claimed.

Local packaging/staging operators are
`outputs/hpc-submissions/prepare-source136-valid-reference-epoch40-continuation-attempt001.py`
and `outputs/hpc-submissions/operate-source136-valid-reference-epoch40-continuation-attempt001.py stage-and-queue`,
using bundled CPython for static bytes/AST/metadata and authenticated transfer/
scheduler actions. All numerical contracts remain allocated HPC work; the
documentation owner only inspected the saved preparation/operation records.

## Scientific parent and unchanged objective

Start from the authentic completed source121 epoch-36 model, native AdamW
moments/counters, native RNG and persisted fresh-prior RNG. Checkpoint SHA256:
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
No disposable probe/proof state becomes a training checkpoint. Keep frozen
epoch-35 occupancy reference and the original epoch-35 ramp lineage; the
separate validity-target model remains **frozen original epoch 36** even as
training progresses beyond it.

Use the validated source135 scientific objective:

- Original **unweighted full NLL** for accepted live bond-repair targets, with
  original search and 0.25/0.75 real/prior branch means and repair ramp.
- **Weight 0.1** full unweighted NLL to same-z, strict-valid frozen epoch-36
  prior targets, normalized by reference `N+K+K*(K-1)/2`, averaged over **all
  scheduled prior attempts**. Reference-invalid rows contribute zero; live
  invalidity does not remove an existing valid reference target.
- Unchanged reconstruction, VAE KL, composition and occupancy terms. Do not
  reintroduce source134 NONE weighting, moving current-self guidance, failed-row
  fallback objectives or a new coefficient/gradient-balancing scheme.

Preserve **1:2 sampling, 256 real/512 prior**, microbatch 128, 12 repair workers,
four main threads/16 CPU slots with overlap, and **quarter actual native AdamW
displacement**, retaining moments rather than clipping or resetting them. Keep
the authenticated data/split/configuration and all scientific health, source,
gradient/finite-value, retention, sulfur, execution, memory, resume and quota
checks unchanged.

## Completed evidence and additive adoption boundary

[Source135 terminal validation](74_frozen_valid_reference_guidance.md#terminal-2500-update-validation-passed-no-training-submitted)
passed the allocated CPU/profile, 518 mandatory CUDA fixtures, all 18 original
native/15 supplementary gates, the restored 200-update probe, then **2,500
disposable updates/all 13 final gates**, with **26 authenticated panel receipts
passing all 18 health checks**. Original state was restored after the probe;
the frozen epoch-36 anchor remained unchanged, and proof weights were discarded.
The terminal gate SHA256 is
`218a3c9e0318286f9122017cf9c4507885202ed7981bea61852d0c16886a39b0`.
The complete GPU wrapper finished returncode zero; its live LSF record aged out.

The new training route requires an **additive authenticated adoption record**
binding the completed source135 archive/manifest/plan, original-36 checkpoint,
native/probe/restoration/long-proof receipts, panel hashes and completed
allocation. It must verify unchanged scientific settings and bind the explicit
new endpoint/training purpose. Reusing this completed proof is not a new GPU
benchmark or a retroactive pass for any failed source. Do not rewrite the
source135 validation-only controller or its sealed test-policy fields.

The versioned training/adoption markers are sealed above. Protocol-only
loss-schema reading is not launch authorization. New metadata/controller guards
need allocated validation and authenticated results; local AST/byte/hash review
does not establish those checks. Any mismatch, missing/uncertain receipt,
unexpected active phase, failed contract or changed science stops launch.

## Guarded execution and endpoint

Restore **original epoch 36**, not the proof endpoint, for the first real
training segment. Authenticate the separate frozen epoch-36 target anchor and
unchanged epoch-35 occupancy anchor. A later segment may resume only an actual
authenticated training checkpoint with its native optimizer/RNG lineage; the
valid-target anchor must never drift to that later checkpoint.

Run strictly serial allocations, **at most 24h wall/23h internal runtime**.
Maintain fixed 1,024-prior/256-reconstruction early health panels after 32 updates
and every 100, including separate raw chemistry and strict representation
checks and bounded invalid examples. Preserve all runtime safety margins and
fresh-quota checks, authentic per-segment completion/health and no concurrent
training. Stop at **absolute epoch 40**, without automatically continuing to
41 or treating this as the global epoch-100 endpoint. Failed health, nonfinite
values, identity/resume errors or exhausted resources remain failures/partial
work, not fabricated epoch completion or permission to waive a gate.

Source135's proof had raw/strict panel ranges 631–725/630–725 and one
representation failure at update 500 while remaining within all declared
bounds. Passing it supports this guarded restart, not guaranteed future health,
monotone validity or independent training-seed replication. Each real segment
still needs its actual saved outcomes before its successor can be submitted.

## Storage work and actual execution record

The user also requested a storage-cleanup breakdown. The independent
[storage record](50_hpc_space_cleanup.md) must distinguish inventory, verified
local preservation, exact removed paths/bytes, failures, and official quota
before/after. That maintenance is separate from objective adoption and cannot
waive the fresh launch quota. Protect original epochs 35/36/89, source/proof
evidence and all failed raw results, active data/environments and unrelated
Protenix/Mosaic work. Do not claim freed space from an inventory or attribute
concurrent activity to cleanup. Actual maintenance receipts belong in docs50
and the ledger when supplied by its owner.

This entry records authorization, sealed byte staging, allocated CPU/profile/
adoption passes and queued segment 001, **not a training start**. No new numerical local tests, HPC
action or submission was performed by the documentation owner. Later additions
must record allocated checks and actual GPU/controller/segment receipts, clearly
separating queued, running, passed, failed and completed epochs.
