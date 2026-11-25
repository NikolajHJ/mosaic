# Single-molecule V100 recovery

**Question:** Can the epoch-36 VAE/Boltz embedding trajectory resume from its
last committed checkpoint after the V100 job's filesystem write failed?

On 31 October 2026, job 29853617 stopped after an I/O communication error and
LSF's time limit; its callback did not start another segment. Update 2,902 is
the last committed state. Its SHA-256 matches the commit journal, the checkpoint
loads with finite latent tensors, its three cache entries match their chain and
files, and its last observation and best-state references match their hashes.
The original state and failed output remain intact. WORK3 quota was 51.03/100
GiB and a new write/flush/read probe succeeded. The underlying cluster I/O cause
is not known and a successful probe does not rule out recurrence.

The recovery uses `continue_v100.py`, uploaded as a snapshot to
`/work3/s234818/latentmol/experiments/embedding-single-molecule/recovery-20261031a/`.
It creates a clean copy of the committed state and referenced cache, then uses
the existing source152 V100 worker, frozen epoch-36 VAE, trained mapping and
original optimization settings. It schedules segments of up to 500 updates
and resumes from the last committed update after a recognized storage or
wall-time failure, provided there was progress. Other failures stop for inspection.
The target is 10,000 updates; the actual relaxed/hard molecule is observed every
10 updates. The VAE continuation is independent and unchanged.

Allocated CPU setup job 29862388 completed the recovery preparation and
submitted V100 job 29862389 and its callback 29862390. At the latest check the
V100 job was queued, so no new GPU update or real callback handoff is established.
Two local CPU tests passed for cache recovery and the simulated successful
segment handoff. The [latest status](../../../outputs/hpc-results/single-recovery-latest-1793486303231146100.json)
records the scheduler evidence.

On 1 November at 08:56 CET, the first recovery segment completed from 2,902 to
3,402; its callback successfully submitted the next segment. V100 job 29864452
was RUN with update 3,432 last committed. The observation at 3,430 was eligible
and strictly valid. Its hard fixed-context loss was 8.3726, while the retained
best is the initial molecule at 8.3443; no binding improvement is established.
The first 500 updates took approximately 7.94 hours on the allocated GPU. These
are saved counts and do not establish the completion time of the full run.
[Status](../../../outputs/hpc-results/single-recovery-news-1793519788513907100.json).
