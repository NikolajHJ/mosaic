

### 2026-10-29T08:49:15.795181+00:00 — Both GPU validations failed CUDA visibility before model loading

Fresh read-only query29October08:46:14UTC authenticated source21/46 files,
879 rawCPU cases/no failures/errors/skips,51 delivered proof files and26original
CPU certificate file bindings. Native Pro29845087 and DBMol29845089 each have
actual same-job gpul40s LSFexit1; both own callbacks29845088/90 also exited.
Live scheduler records aged out; persisted original LSF establishes these
terminal outcomes without fabricated live statuses.

Both native entry storage gates PASSED. Both then failed the two-client CUDA
allocation precheck: CUDA smoke requires exactly one explicitly visible GPU.
Smoke records and cleanup arrays are empty, optimizer_updates0, allocation
receipt empty; no worker/model/gradient check or driver invocation/study exists.
No optimizer job or added update exists. OriginalPro20k/DB7500 saved endpoints,
settings and879CPU pass remain unchanged. This is a startup admission failure,
not a new native gradient disagreement or storage shortage. Callbacks correctly
refused failed predecessors. The old failed sources/results remain preserved.

Static inspection finds CPU callbacks explicitly hide CUDA while the GPU branch
inherits its environment; the original rejected CUDA_VISIBLE_DEVICES value was
not captured, so exact scheduler/environment wiring is not yet authenticated.
Do not guess GPU0, bypass the one-GPU/co-residency gates, change MPS modes or
blindly resubmit. No fix, new allocation or job mutation was performed by this
status query. Monitor remains paused. Exact evidence is status-20261029-084614.json.
Owner: experiments/validation/endpoint-qualification/runs/20261028-precision-continuation-21/.
