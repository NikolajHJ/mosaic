

### 2026-10-29 — Queue audit and conservative runtime estimates prepared

User requested compatible faster queues and appropriate walltimes. Live audit
finds physicalL40S waits only, all12GPUs occupied; A100/A40/A10 also occupied.
SomeV100s free, but currentcuDNN9.13matrix excludesVolta and immutable workers
requireL40S. No blindqueue switch or changedruntime. Successful componenttiming
priors457sPro/692sDB/1012sBolchem do not measure complete new validations.
Prepare scheduling estimates60minPro/DB,45minProchem, retaining120minhardcaps
and every check/cleanup deadline, originalIDs/age/argv. Actualfutureoptimizer
walltime already uses1.5x measuredstepcost plus completeendpoint/setup/reserve,
rounded15min;23h is ceiling, not blanketrequest. No job mutation at preparation.
No tests/models/cleanup/monitor changes; reversible metadata only.
Owner: experiments/validation/endpoint-qualification/runs/20261029-queue-optimization-01/.
