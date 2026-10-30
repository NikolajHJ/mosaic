# Complete Boltz objective precision repair candidate01

Authorized30October2026: investigate/fix plain Boltz20k, Boltz+chemistry10k and
DBMol7500 derivative blockers without changing settings or saved results. The
current FP64 reference covers DBMol affinity only; other components/total and
fixed-teacher combined logits still rely on FP32 native responses. Tiny summed
loss intervals and genuinely sharp native derivatives must be distinguished.

This candidate adds a separate full-FP64 reference for the SAME two-trunk DBMol
objective and same fixed-pose Boltz affinity. Original production stays FP32.
Reference parameters/context are independent lossless promotions, every float
operation audited, native distance-bin partition fixed to original FP32
constants, every objective/component and total remains explicit. Original
chemistry targets are reconstructed once with the original settings from the
authenticated failed assay point and MUST match its original target identity;
they stay frozen through all probes. No optimizer update, diffusion or resampling.
Both original and predeclared fine scales are retained, no after-result scale
selection or tolerance loosening. A diagnostic pass alone cannot launch a child.
Full prospective qualification/pilots/runtime/shared storage remain required.

All numerical tests/evaluations execute on allocated HPC. Portable metadata/AST
controls alone may run locally. Output capped256MiB/2048files inside the existing
shared2GiB control allowance. Every phase authenticates and calls the CURRENT
production03 full shared capacity guard. Accepted commands explicitly name that
shared capacity owner and register before owned release, so the running Pro
periodic guard continues to account for them. No Pro run/source is modified.

New source/run namespaces and actual CPU/GPU submission identities are recorded
in campaign.json after preparation. Initial state is unsubmitted/unqualified.
Old failed outcomes, source releases and all saved trajectories remain intact.

The plain/Boltz+chemistry objective remains quantitative affinity_pred_value
alone. DBMol keeps its separately defined pred-minus-binary-logit affinity,
contact and confidence combination. Nineteen native/chemistry modules are byte
identical across the two scientific releases and are authenticated before use;
the differing trajectory implementation is not used by this diagnostic.

The plain20k raw report aborted before producing any direction rows. Its two
logit controls (atom15 1->4, bond132 1->0) are explicitly predeclared supplementary
directions, not fabricated original rows. DBMol retains all three failed total
bond directions209/195/264 plus atom1; Bolchem retains both original low-coupled
assay directions, exact reconstructed fixed-teacher identity and exact original
low-coefficient scalar/gradient replay. Coefficient0.001, actual global10000 and
100 cohorts are all retained. No comparison selects a passing scale.

Portable11 metadata controls and136 resource contracts passed locally, no
numerical evaluation. Allocated CPU39 cases (original17/newTorch8/newJAX3/
portable11) and three actual native server startup commands remain required.
GPU diagnostics are independently bounded2h allocations,8CPU/4GiBslot, one
exclusive-assignment shared-compute L40S without MPS; entry plus evaluation
deadline6600s leaves600s wrapper/cleanup reserve. Exact shared storage gates
repeat at entry and every300s, retaining original forecasts/reserves.
