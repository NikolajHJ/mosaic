# Contact precision and selected-branch derivative validation

Status, 14 October 2026: the new arithmetic and validation policy are locally
implemented and under static review. Allocated CPU fixtures, exact saved-state
GPU readiness and a fresh matched Boltz–DeFoG pilot are **pending**. This
document explains the evidence-supported mechanism and candidate remedy;
it does not approve historical failed guards or establish a working pilot.
Actual release/plan hashes, submissions and outcomes belong in the
[experiment ledger](05_experiment_ledger.md).

## What the completed diagnostic established

Source103 GPU **29706116** completed both authenticated source091 step-15
states, control and DeFoG-guided, in 674 seconds. Ten complete cuts, from
shared graph through final distogram, each reproduced the direct gradient
exactly. Cloned forwards, baseline-held input ReLUs and seed replay also
matched. This provides no evidence of a missing adapter or trunk gradient
path at these states. The guided natural contact finite difference still
failed its historical guard. The original records remain unchanged in
[control attribution](../outputs/hpc-results/source103-first-suite-004/runs/source-103-contact-boundary-diagnostic-attempt001-replay/20261014T085919Z_source103-contact-boundary-diagnostic_01c76680/boltz_only/20261014T090047Z_boltz_only_a5dbcbe1/attribution.json)
and [guided attribution](../outputs/hpc-results/source103-first-suite-004/runs/source-103-contact-boundary-diagnostic-attempt001-replay/20261014T085919Z_source103-contact-boundary-diagnostic_01c76680/boltz_defog/20261014T090526Z_boltz_defog_16d5611b/attribution.json).

The important distinction is between a derivative at the current point and
a central secant across two rounded forward evaluations. A small step reduces
curvature error but makes the difference of those forward values smaller,
so arithmetic error can dominate it. FP32 has finite precision; evaluating
an already-rounded FP32 tensor in FP64 cannot recover the information lost
earlier. PyTorch describes this limitation in its
[numerical-accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
That reference explains general arithmetic, not a validation of our pinned
Torch 2.5.1 runtime.

Here contact combines the pocket-average and strongest-pair anchor losses:
`L_contact = (L_pocket + L_anchor) / 2`. Their directional derivatives can
have opposite signs. Cancellation leaves a smaller net slope, so comparable
absolute forward rounding produces a larger relative error in the combined
contact slope. The root's reading of the authenticated source103 rows finds
about fivefold cancellation amplification in the guided direction, versus
about 1.2-fold in the control. This is a measured-state explanation, not a
claim about all targets or all molecules.

At the guided original direction, input-ReLU/contact-winner changes were absent
at the three smallest probes. Recomputing just the symmetrization and final
affine projection in FP64, then using the existing FP64 contact reduction,
gave the following **diagnostic** secants:

| Epsilon | Widened final-head secant | Original analytic contact slope |
| --- | --- | --- |
| 0.00001 | -0.0354441 | -0.0350662 |
| 0.00003 | -0.0356872 | -0.0350662 |
| 0.0001 | -0.0365146 | -0.0350662 |

Those residuals are nonzero but fall within the pre-existing 5% relative
tolerance. The earlier status correctly reported residual error, but did not
highlight that this counterfactual met that tolerance against the old slope.
**It is not yet a pass for the candidate production function:** that function
changes arithmetic and must be checked against its own new reverse derivative,
independent native forward derivative and unchanged native confidence outputs.
Upstream FP32 rounding and curvature remain possible residual sources.

The affinity issue is separate: the prior affinity-head experiment observed
ReLU branch crossings over its central probes. A secant across a kink can
differ from the selected local derivative even with correct backpropagation.
No ReLU smoothing, changed pretrained weight or clipping is proposed here.

## Explicit contact arithmetic change

The opt-in configuration value is
`objective_arithmetic: contact_head_and_assembly_float64_v1`.
[contact_projection.py](../src/latentmol/boltz/contact_projection.py) takes the
same native final pair state, widens it **before** pair symmetrization and
the frozen affine distogram projection, and returns FP64 contact logits.
[dbmol_model.py](../src/latentmol/boltz/dbmol_model.py) gives native FP32 logits
to confidence and the separately widened logits to the contact branch.
[dbmol_objective.py](../src/latentmol/boltz/dbmol_objective.py) preserves the
native FP32-defined distance-bin partition, contact minima, cutoff, epsilon,
loss signs and weights, and assembles the declared objective in FP64.

This is a new arithmetic identity. It does not mutate stored FP32 checkpoint
parameters, retrain the model or silently reinterpret previous runs. The
structural and affinity trunks remain native FP32; the repair NLL remains
native. Only the shared final pair state's contact projection and declared
scalar assembly change arithmetic. Unsupported head topology, unfrozen/training
modules, incompatible dtype or missing shared pair state fail explicitly.
Old `native_v1` and reduction-only policies keep their separate identities.

## New selected-local-branch contract

`fd_policy: branch_local_v2` explicitly validates the selected local Jacobian
of the unchanged native model. It **does not certify the old natural-secant
contract** or turn earlier `local_v1` failures into passes. Native ReLU branches
are never held during optimization, sampling, DeFoG repair or target updates.

[branch_local_fd.py](../src/latentmol/boltz/branch_local_fd.py) requires:

1. Capture all twelve ReLUs in the pinned two-member affinity readout at the
   exact baseline. For validation only, hold that branch through scoped hooks,
   with exact baseline scalar, total-gradient and component-gradient parity.
2. Use **native forward-mode automatic differentiation**, independently of
   reverse AD, to verify the directional slope. Both traverse the same decoder,
   shared graph and native model. Forward AD bypasses reverse checkpoint graphs
   under `no_grad`; unsupported operators fail rather than falling back to
   finite differences or a double-backward JVP. JVP relative/absolute
   tolerances are 0.001 / 0.000001.
3. Check finite differences on that same baseline readout branch using the
   predeclared direction/grid and existing tolerances. Natural central secants
   are recorded separately with their genuine pass/fail outcome.
4. Require **natural contact finite differences to pass as well**. Holding
   affinity readout branches cannot excuse a remaining contact mismatch.
   Check affinity, contact, confidence, fixed-target repair when present, and
   the combined objective; an earlier failure cannot hide later components.

Forward/reverse directional comparisons and central differences test different
properties; PyTorch's [gradcheck mechanics](https://docs.pytorch.org/docs/2.14/notes/gradcheck.html)
describes their Jacobian products and central-difference calculation. Our
selected-branch extension is explicitly project-defined. At an exact-zero
ReLU, the contract verifies PyTorch's chosen zero derivative; it does **not**
assert a two-sided mathematical derivative at the kink. A selected-branch pass
does not guarantee finite-step prediction accuracy across other branches.

The current supported configuration is fixed initial occupancy, bounded latent
steps, zero recycles and the explicit contact-head arithmetic. Molecule identity,
pose, pocket and repaired target remain fixed within each derivative check.
Only latent `z` is optimized; the decoder, Boltz and DeFoG remain frozen.

## Required evidence before a fresh pilot

[contact_precision_validation.py](../src/latentmol/boltz/contact_precision_validation.py)
must authenticate both completed source103 cases, replay the original shared
graph/target and bind the actual new production methods. It must record the
new source/arithmetic/policy identity and pass native hard-input parity,
repeatability, unchanged affinity/confidence outputs, exact source103
counterfactual parity, own-component/total reverse AD, independent forward AD,
exact selected-branch baseline parity, strict branch FD and natural contact FD.
Both saved states must pass. There are no optimizer updates or new DeFoG calls
in that readiness replay; every failure remains durable.

Only then may a newly sealed version-2 matched pilot consume the authenticated
readiness proof. It must retain a control and guided case, the frozen shared
decoder/molecule contract, fixed initial mask, bounded steps, same start/pose/
pocket/objective weights, declared DeFoG refresh schedule and budget, and explicit
repair-target stop-gradient. It must proceed beyond the previously failed
step 15, retain gradient guards and guidance acceptance/failure records, and
separately validate final discrete chemistry and fresh native binding outputs.
A completed diagnostic alone is not a completed pilot or useful-molecule result.

Ledger entries must distinguish prepared/uploaded/submitted/RUN/DONE/EXIT,
CPU versus native results, plan/source/config/data/checkpoint hashes, input and
target identity, arithmetic and derivative contract, exact command/environment,
allocation and per-child budget/use, every guard's pass/fail, optimizer steps,
DeFoG calls/accepted targets, final validity/repair edits and native rescoring.
No failed predecessor may be approved by a callback; no tolerance, direction,
budget or target may be changed after observing results.
