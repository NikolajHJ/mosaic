# brd4 experiments

Scientific family index. Each study retains its own question, protocol, configuration and attempt history.

- [BRD4 contact optimization](contact-5000/README.md): Optimize the frozen Protenix contact objective against the authenticated BRD4 fixture over 5000 updates, with bounded pilot/native gates.
- [BRD4 uniform initialization control](uniform-5000/README.md): Test BRD4 contact optimization from the declared uniform initialization while retaining the original native and runtime gates.
- [BRD4 noisy-uniform control](noisy-uniform-5000/README.md): Compare a declared noisy-uniform start against the fixed-input BRD4 controls.
- [BRD4 canonical diffusion control](canonical-diffuse-5000/README.md): Compare the native canonical/diffused initialization under its separately declared context and validation contract.
- [BRD4 best-primal continuation](best-primal-continuation/README.md): Continue selected authenticated states under the declared best-primal acceptance protocol while preserving parent state and fixed objective identity.
- [BRD4 chemistry guidance](chemistry-5000/README.md): Compare a rule-based chemistry-guided molecular trajectory with the frozen native contact objective, retaining invalid graphs and original repair/search semantics.
- [BRD4 delayed chemistry ramp](chemistry-ramp-10000/README.md): Test the declared delayed chemistry coefficient schedule across random and mixed starts for 10000 requested updates; changing chemistry totals are not one fixed comparable objective.
- [BRD4 chemistry compatibility probe](chemistry-compatibility/README.md): Probe compatibility of continuous and repair-based chemistry components independently of successful production optimization.
- [BRD4 frozen Boltz-2 affinity optimization](boltz-affinity-10000/README.md): Optimize the frozen detached-context Boltz-2 affinity branch with its separate raw native gradients, contact/affinity interpretation and independent arms.
- [BRD4 Boltz gradient diagnostics](boltz-gradient-diagnostic/README.md): Investigate failed raw-gradient/finite-difference behavior without promoting a diagnostic or changing the historical native acceptance result.
- [BRD4 25-slot three-model comparison](three-model-25/README.md): Compare Protenix, Boltz-2 and Boltz-2 plus chemistry from the same declared 25-slot initialization, with independent native readiness and long-run outcomes.

See the [global experiment index](../INDEX.md) and [agent logbook](../../docs/experiment_ledger.md).
