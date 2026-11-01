# A10 and A100 capacity for the chemistry comparison

Use separate available queue allowances to accelerate the existing22-arm comparison. Current account limits observed1November: fourV100, threeL40S, twoA10 and twoA100 GPUs; no global five-job cap reported. Queue slots are not a hardware reservation.

Actual device probes: A10 **29865920**, A100 **29865922**. Own-ended metadata callbacks **29865921/29865923** conditionally migrate chemistry-first seeds17/23 toA10 and combined seeds17/23 toA100. At11:06UTC A100 probe wasRUN, A10 probe and both callbacksPEND; no GPU pass or new optimizer ID yet. Read each remote `run/jobs.json`, `probe/assessment.json`, `migration-summary.json` and actual submission/LSF before recovery. Never replay stage or accepted submissions.

Each probe runs25 disposable updates for chemistry-first and the combined path, records sampled memory/timing and checks original frozen identities/finite attached paths.16 portable metadata/fake-scheduler cases and six real pre-model CLI startups passed. Original allocated42-case controls are reused; no new qualification chain.143 library files and every scientific driver function except metadata entry are byte/AST unchanged. Same model/context/key17, initialization seeds, FP32 native precision, schedules, objectives, detached step targets, LR0.3, all20k updates and stride20 retention.

The callback authenticates its own successful GPUjob and original pending writer/source, measures a full20k budget, performs the actual replacement startup/capacity check, holds/rechecks the never-started predecessor, then retires only that exact job before submitting a successor. If the predecessor starts, keep it; if the probe fails, preserve the V100 job. Disposable probe states are never adopted. Only new immutable source/outputs are used. Device rounding differences remain exploratory findings.

Canonical code remains the existing `chemistry_screen_{control,ops,stage}` and scientific screen runner; profiles are shared in `scripts/hpc/gpu_policy.py`. Source roots and exact hashes/actual IDs are in campaign.json. Future report captures use the existing chemistry-screen adapter and canonical report command; no new renderer.

## Latest actual outcome1November11:15UTC

A10029865922 and callback29865923 bothDONE. Both25-update paths completed with unchanged frozen identities. Warm medians0.586s chemistry-first/0.629s combined versus V1000.796/0.846s in the same short probe recipes; approximately25% shorter warm update time, not a full-horizon performance guarantee. Sampled GPU memory peaked2535MiB. Combined seeds17/23 actually submitted **29865938/29865970**, bothPEND, full20k each/06:15 measured wall. Original unstarted29862997/29862998 were held, rechecked, retired with zero updates and preserved proofs before successors were released.

User additionally authorized A40: **probe29866016/callback29866017** submitted, bothPEND. It targets original severity seeds17/23 with the same settings and short-check/conditional-migration rules. A10probe29865920/callback29865921 remainPEND.18 current portable controls and nine actual CLI startups passed. All device namespaces are immutable; no running producer was stopped. Read successor_locations in the original campaign and each device run/jobs.json; the old source02 remote registry remains historical for migrated arms.
