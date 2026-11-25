# Pointwise native derivative validation and targeted pilot readiness

For new exploratory work use the [shared exploratory policy](EXPLORATORY_RESEARCH_POLICY.md)
and [gradient policy](GRADIENT_VALIDATION_POLICY.md). Detailed native/independent
AD/FD gates below describe qualified tools and specific historical campaigns;
they are not blanket admission requirements for a new short pilot. Existing
sealed runs and historical outcomes keep their original contracts.

**Native readiness and engineering pilot passed, 16 October 2026:** both
source109 attempt002 saved-state checks passed under v4 after **303 CPU checks**; promotion actually
submitted **29732031**, which completed both **25-update** arms, **12 scheduled
audits / 216 stored AD rows**, independent bridge/recomposition/state checks
and **50 accepted native descent receipts**. Both branches reach the same `z`;
the guided arm used nonzero repair gradients throughout, with 19/20 eligible
native outputs and five selected targets. Historical finite-secant mismatches
remain recorded; this is a pass of the explicitly declared local-AD contract.

This does not establish a useful molecule: raw argmax states **0/26 valid in
both arms**, hard samples **1/208 control / 0/208 guided**. Fresh-pose followups
completed but attempted **zero predictions** because selected start/end argmax
graphs were invalid. Guided bounded repair witnesses **4/4 versus 1/4 control**
are an exploratory closeness signal, not a globally nearest-valid distance or
replicated chemical advantage. [Verified capsule003](../outputs/hpc-results/source109a002-pointwise-round-003.tar.gz)
and the [ledger](05_experiment_ledger.md) retain scientific scope, actual
receipts, quality results and the separate failed chemistry comparator.

**Attempt002 CPU passed, 15 October 2026 11:59 DTU-local:** **CPU 29729107
DONE / 303 checks passed**; **readiness 29729108 PEND**, **promotion 29729109
PEND**. [Verified capsule](../outputs/hpc-results/source109a002-pointwise-round-002.tar.gz)
establishes engineering/configuration success, not native v4 approval or a fresh
pilot. Attempt001's
302 CPU passes remain preserved; its unstarted native/callback were requested
canceled and are observed **EXIT**, following a static pilot-reader correction for inward moves inside the
global start-relative trust region. The native v4 contract and optimizer are
unchanged. [Ledger](05_experiment_ledger.md) records exact new source/plan
identities and cancellation outcomes. The archived definition remains immutable.

The [12:01 DTU-local detailed queue read](../outputs/hpc-results/source109a002-queue-status-001.json)
shows readiness waiting for available GPU capacity on an eligible host and
promotion waiting for that job's completion; no manual hold is reported.

**Implementation submitted, 15 October 2026 11:54 DTU-local:** source109
attempt001 has **CPU 29729073 completed**, **201 + 101 allocated checks passed**.
Native readiness **29729074 PEND**, promotion **29729075 PEND**; there is no
native v4 pass or new pilot submission yet. [Verified capsule001](../outputs/hpc-results/source109-pointwise-round-001.tar.gz)
and the [experiment ledger](05_experiment_ledger.md) preserve exact source,
plan, command and receipt identities. The archived definition was sealed before
this status update; the scientific contract below is unchanged.

The [existing-evidence review](56_existing_gradient_evidence_review.md) shows
that finite-interval accuracy, automatic-derivative consistency and useful
optimization are separate questions. Observed decoder/affinity branch changes,
FP32 rounding and cancellation can invalidate a central-secant comparison
without demonstrating a broken local backward path. The newest upstream
nonlinearity is not uniquely localized. Another broad boundary sweep is not
required to define a more appropriate, explicitly scoped validation contract.

## New contract

`fd_policy: pointwise_native_ad_v4` is an explicit opt-in. Despite the legacy
configuration key's name, its mandatory full-path property is **selected local
native forward/reverse derivative agreement**, not whole-network numerical
finite-difference verification. It does not prove every Jacobian direction,
global smoothness, molecular validity or binding efficacy. Historical v1/v2/v3
guards retain their original outcomes and are never retroactively approved.

Each audit checks the same frozen decoder, shared relaxed graph, native Boltz
contexts and explicitly stopped repair target. The model input has fixed
occupied slots. Protein features, geometry, masks, pocket and auxiliary native
features remain fixed; element expectations, expected formal charge,
connectivity and typed-bond expectations follow the shared graph.

1. **Native identity and repeatability.** Preserve hard-input/native parity,
   source and asset authentication, nonzero finite component derivatives,
   exact production/direct baseline values and gradients, and repeated native
   primal/graph identity. Repaired labels and their applied coefficient stay
   fixed within every audit and trial.
2. **Independent full-path derivative algorithms.** Use native forward-mode
   dual tensors under `no_grad`, independently of activation-checkpoint reverse
   AD. Check each DBMol component, optional fixed-target NLL and combined loss
   on the existing analytically preselected four directions. Retain the existing
   JVP tolerances, relative `0.001` and absolute `0.000001`. No finite-difference
   outcome may select replacement directions or relax these checks.
3. **Exact loss recomposition.** Rebuild the actual grouped objective from its
   live primitive tensors, retaining native dtype/order, scalar scaling and the
   native FP32 repair-coefficient product. Require exact scalar, reverse-vector
   and forward-tangent agreement. Do not approximate this proof by adding
   separately rounded FP32 gradient vectors.
4. **Independent graph-input bridge checks.** At the actual shared graph,
   compare native graph-to-feature values with explicitly calculated element,
   charge, connectivity and seven-category bond expectations. A cheap FP64
   affine-cut audit additionally checks expected values, tangents, reverse
   cotangents and finite differences at two preselected scales/directions,
   including the unchanged auxiliary context. These affine cut perturbations
   are derivative probes, not proposed molecules. This numerical check is
   confined to the bridge; it is not full Boltz numerical verification.
5. **Finite-step observations remain visible.** Save a small fixed set of
   natural secants as landscape diagnostics, with `required_for_pass: false`.
   Earlier complete curves remain immutable references. No baseline readout or
   decoder branches are held during optimization; native forward/reverse checks
   use the actual baseline program. A secant's disagreement is not silently
   relabeled a passing finite-difference result.

Forward/reverse AD can share an unintended detach, so their agreement alone
does not authenticate the intended function. The mandatory independent input
bridge checks, explicit fixed-feature scope, native parity and existing source
audit are complementary safeguards. Unsupported operators or missing tangents
fail; no double-backward JVP or learned surrogate substitutes for native AD.

## Actual-step acceptance

Keep the existing bounded Armijo backtracking and trust radius. Every accepted
update must improve the actual unchanged native combined objective at its own
fixed target and coefficient. A local derivative pass never approves an update.
The pilot reader rechecks each saved accepted trial, actual improvement,
original-gradient directional decrease, context, target, coefficient and trust
radius. The scalar-ULP observation guard is not a whole-network accuracy bound.
No monotonicity claim is made across a target refresh or changing repair weight.

## Targeted readiness and pilot

Validate only the two authenticated source105 failed step-15 states, one control
and one guided. Reuse their exact source/model/data/start/context/target assets.
The new configuration changes the derivative policy only. Bind the existing
saved-results review as immutable evidence; completion of a new broad observer
is not a prerequisite. Native readiness is limited to two serial 1,200-second
cases with at most 80 full native calls per case in a 45-minute allocation.
These are caps, not measured runtimes or passed outcomes.

After allocated CPU checks and both actual native readiness proofs pass, an
authenticated callback may submit one serial matched 25-update pilot. Preserve
the initial graph, fixed 22-slot mask, pocket, models, loss weights, temperature,
repair schedule, sampling budgets and audit steps 0/5/10/15/20/25. The cheaper
pointwise contract also runs at the existing repair-refresh boundaries. Preserve
bounded repair and fresh-pose followups as separate outcomes. Readiness failure
stops promotion; no outcome-selected retry or historical gate waiver exists.

Existing pending workflows must not launch a second, superseded pilot.
Record any requested and actual cancellation separately. Already executing
diagnostics may finish; their results remain independent evidence and do not
approve this new contract. The independent VAE chemistry campaign is unchanged.
