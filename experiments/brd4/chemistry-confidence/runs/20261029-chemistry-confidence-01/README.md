# Confidence margin attempt001

User authorized implementing/testing weak bounded-slope guidance against current
chemistry, with composition off and saved runs unchanged,29October2026.
Status: rejected before submission, no job ID or numerical execution.
Protocol: [experiment](../../README.md).

Policy is predeclared weight0.01/margin0.2/slope_cap1.0. Original production
configs/runs are not changed; the library path is opt-in. New tests include
independent primal/FD, slope/inactive guards, no target refresh within trials,
baseline parity and an actual two-update identity-change integration fixture.
Controlled paired pilots are chemistry-only, not native binding experiments.

Local commands/manifest/evidence appear here after preparation and execution.
Numerical source/output are node-local on a real allocated4CPU hpc job, then
streamed into ignored laptop `results/hpc/20261029-chemistry-confidence-01/`.
No WORK3/HOME scientific artifacts, current jobs, settings or saved states are
written, moved, deleted or restarted. Failed/unrun stages remain explicit.

Actual29October11:33:06UTC: bsub255, "Queue does not accept interactive jobs.
Job not submitted." Raw stdout, stderr and exact transport hashes are preserved
in laptop `results/hpc/20261029-chemistry-confidence-01/allocated01/`.
The queue/resource correction is a new source/attempt002; this attempt is not
restarted or relabelled as passed. Initial35fixtures/comparison never ran.
