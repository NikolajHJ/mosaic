# V100 continuation handoff

The user authorized moving the existing epoch-36 learned-mapping campaign to
V100 while keeping the VAE continuation unchanged. Device qualification
source146 passed all three native gates, six start derivatives, three saved-state
derivatives and memory headroom. That result does not itself migrate state.

At30 October11:27 CET the existing A100 case002 job29851407 was PEND; its
callback29851408 waited normally. The exact saved states were case001 update1899
SHA `3c2fe29fcaa894395c590a828715aaa47faff493b7460502a06ea0e60024e7c1`,
case002 update20 SHA
`de07e84e018aa1f1fcefcd5fc8b7013bbbec49e2e0f937cf7306e39e73e4d91e`,
and case003 update20 SHA
`373cf9667bcfe171b7e3bad26cc1b7b52f7e9de203b6f1d0fd8c93602593ea54`.
Each trajectory has one authenticated hard-score cache entry and one best
checkpoint. Original hard scores must not silently be mixed with new V100
scores: the qualified GPU results differ bitwise across architectures.

Independent source147 attempts to re-score the exact three saved hard graphs on
V100, authenticating the passed source146 qualification and all original inputs.
It checks detached eligibility, exact cached graph identity and best-graph
identity, compares all finite scores within the existing native scalar tolerances,
and checks frozen model state before/after. It never writes to production paths
and its result does not itself migrate or authorize a cached-score waiver.

Source147 archive SHA256
`f9bff6a14b41f27128a5d5970de7e663bb0451e0c8b7d530b7e6f28ce44c1e8c`,
manifest SHA256
`ec19a9a2711806e8ff12bcba6ccba6811123c54c1a314fcdb37354859dbdb182`,
plan SHA256
`28f34414a615aec46d80e0a376a529288c4ff217685429215b3ddbf9ee775ffb`.
GPU **29853377** was submitted to `gpuv100` with `select[gpu32gb]`, four cores
and one hour. At11:30 CET it was PEND; no score audit pass is yet established.
The existing A100 job remains PEND and VAE job29849760 remains RUN.

A complete handoff still requires an isolated, validated continuation controller:
copy the authenticated committed latent/RNG/counts and immutable history into
separate V100-owned state, rebuild the hard-score cache and best-score baseline
from the V100 audit, preserve scientific settings/guards and per-case targets,
verify a bounded resume and serial continuation, and bind all new score and GPU
provenance. The active A100 job and callback must be reconciled with the scheduler
at cutover; never create concurrent writers or cancel a running segment on a
stale PEND observation. Only after the safe handoff is ready can the A100 pending
request be retired and a V100 continuation adopted. Do not call the run moved
until that happens. The source142/143 releases and independent VAE remain intact.

At11:37 CET GPU29853377 remained PEND: LSF reported physical GPU reservation
unavailable on four hosts. Four reserved CPU slots are not a GPU start; no audit
output exists. A draft isolated cache/state rebasing helper has been prepared
for the handoff, but has not been run, tested or adopted. The V100 continuation
controller, bounded resume and safe cutover remain outstanding. The existing
A100 job remains PEND and the VAE job RUN. [Scheduler receipt](../outputs/hpc-results/v100-cache-audit-status-1793356676636199900.json).

## Cache audit passed; isolated resume pilot queued,30 October2026 12:10 CET

GPU29853377 completed successfully. All three exact saved hard graphs remained
eligible and their V100 scores were within the original native scalar
tolerances of the authenticated A100 caches. This audit did not alter or adopt
the production caches. Allocated CPU29853481 completed nine rebase fixtures
with zero failures/errors/skips. Independent CPU29853484 then applied the tested
helper to copies of all three actual committed states: case001/002/003 remain
at updates1899/20/20 with latents, original start and counters preserved and
V100-scored cache/best baselines. The isolated clones were not adopted.

An independent one-update V100 resume pilot **29853488** was submitted on a
32 GB V100 using original source142's production optimizer/guards and an
isolated copy of case002's clone. At12:10 it was PEND, with no GPU result.
It tests actual restore, current-device hard scores, and one original-policy
update; it cannot by itself claim a full 10,000-update continuation. The
V100 continuation controller and exclusive cutover remain outstanding. A100
job29851407 remained PEND; the VAE29849760 remained RUN unchanged.

Receipts: `outputs/hpc-results/v100-handoff-progress-1793358450274591500.json`,
`outputs/hpc-results/v100-real-rebase-terminal-1793358555578355300.json`, and
`outputs/hpc-results/v100-resume-pilot-progress-1793358624488479400.json`.

At12:12 CET the isolated V100 resume pilot29853488 started on n-62-20-11.
As of12:13 it was RUN with no result; this is not a successful resume yet.
The A100 continuation remained queued and the independent VAE job remained
running. [Pilot snapshot](../outputs/hpc-results/v100-resume-pilot-status-1793358814755102200.json).

## Isolated V100 resume passed,30 October2026 12:15 CET

GPU29853488 completed a single original-policy update from the independently
rebased copy of case002: update20 to21, no no-op or stall, frozen state intact.
The source149 clone and original production state remained unchanged. This
one-update result is not adopted into the real campaign and does not establish
a full-segment or serial-controller pass. Fifteen raw files, including saved
state, journal and terminal logs, were verified against remote SHA256 and size.
[Verified result](../outputs/hpc-results/source150-v100-resume-pilot-attempt001/terminal-1793358953495143700/integrity.json).
The A100 continuation was still PEND and the VAE continuation RUN. The V100
serial controller and exclusive cutover remain to be implemented and tested.

## V100 production continuation started, 30 October 2026 12:27 CET

After the isolated resume passed, source151 started production job 29853560 on
32 GB V100 n-62-11-16. It resumed case002 at update20 and saved new updates21
and22. Case001 remains at1899 and case003 at20 in the rebased input snapshots.
These are saved counts, not live counters or completed 10,000-update runs.
The trained mapping, frozen epoch36 VAE, original objective, optimizer policy,
10-update observations and existing scientific guards remain in use. A100
continuation29851407 is intentionally held PSUSP as fallback; its original
callback29851408 is PEND. The independent VAE29849760 remains RUN unchanged.

Source152 serial callback29853587 waits for the first V100 segment to end; it
is configured to schedule the remaining cases and subsequent V100 segments,
retiring the held A100 pair after a successful first-segment transition, or to
resume the A100 worker if that segment fails. Its success and failure paths
passed two isolated CPU mock fixtures in source153 job29853600 (zero failures,
errors or skips). This is a controller simulation; the real automatic handoff
and final native follow-up have not occurred yet. No further cross-device
gradient qualification was added to the production handoff.

The saved scheduler and initial update21 receipt is
[status](../outputs/hpc-results/v100-live-status-1793359640506897400.json).
The update22 commit is recorded in the source151 trajectory journal. Source151
archive SHA256 `0266ebffdaba91960c3e6592f8fd0ac76fccaf7892cb6b7cfafaf28b99d610ae`;
source152 archive `b67ea113b011f508aba6814ed3fece6dcb54c64a32f377d22616337f350516b3`;
source153 test archive `4aa1d6e90667d937cbbb97c9a33f33c366f8d3b9a372306b51506adc306083de`.

## Single-molecule continuation, 30 October 2026 12:30 CET

The user requested stopping the two least advanced molecules and concentrating
on the one most likely to finish first. Case001 had 1,899 committed updates;
case002 had at least28 and case003 had20 at the preflight. The active case002
V100 job29853560, held A100 job29851407 and their callbacks29853587/29851408
were explicitly stopped; LSF subsequently reported EXIT for all four. Their
saved results remain available. VAE job29849760 and callback29849761 were not
changed.

New source154 single-case controller CPU29853616 completed and submitted
case001 V100 worker29853617 and callback29853618. The worker resumes the
verified V100-rebased case001 state at update1,899, using the unchanged source152
worker and source142 optimization settings. The callback schedules subsequent
segments for case001 only through 10,000 updates. At12:30 the V100 worker was
PEND for scheduling and its callback PEND; neither a new update nor a real
successor handoff had occurred. The final native follow-up remains pending.
[Queue snapshot](../outputs/hpc-results/single-case-check-1793359837772375800.json).
Controller release archive SHA256
`3e02806614c929e59ff361d2607aa1ee87340a6034391fab67bd9aba23df43aa`;
plan SHA256 `4fdc8d2c5f87842d4df9cd227ff6989eeeb4e6322fb366f98b45884fa85bea9d`.

## Single-molecule run stopped, 31 October 2026 23:35 CET

V100 job29853617 ran from update1899 to last journaled update2902. LSF
reported a run-limit termination on 31 October; its stderr also contains an
`OSError: Communication error on send` while flushing the update journal.
The error's underlying cause and effect on the last saved files have not been
established. Callback29853618 exited because the predecessor did not have a
successful terminal result; no successor was submitted and no single-case job
is currently running. Last sampled observation at2900 was eligible and strict
valid; its hard fixed-context loss was8.3726 versus the initial8.3443, so a
binding improvement is not established. The saved checkpoint needs inspection
before any resume. [Status evidence](../outputs/hpc-results/single-case-news-1793486131816232300.json).
