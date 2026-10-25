# Prospective endpoint qualification diagnostics

This experiment investigates whether a failed categorical endpoint probe is an
infeasible simplex perturbation, a resolved numerical disagreement, or an
unresolved precision measurement. It preserves every original failure and does
not qualify or restart a saved experiment.

The evaluator-independent policy is `endpoint-qualification-diagnostic-v1`.
It fixes contrasts and all perturbations from graph geometry before any native
loss or gradient call. Original scales 0.01/0.003/0.001 remain explicit; a fixed
geometry-derived ladder is supplementary evidence. It never chooses a scale in
response to a loss or a passing result, clips probabilities, projects a graph,
or changes a saved state. Every actual FP32 input, displacement and SHA is kept.

Independent primal and AD repeats, scalar parity, represented displacements,
signal resolution and frozen-identity hooks accompany each probe. An original
infeasible scale produces `boundary_inconclusive` even when supplementary scales
agree. Otherwise resolved disagreements remain `numerical_disagreement`;
insufficient representable signal is `unresolved_precision`. Diagnostic `passed`
does not certify an original gate. Optional saved-logit controls have their own
baseline and explicit scope; equality with raw saved probabilities is never
assumed.

Ownership:

- [scripts/qualification.py](scripts/qualification.py): lazy-NumPy, evaluator-
  independent diagnostic policy; no model setup, optimizer, file mutation or
  submission code.
- [tests/test_qualification.py](tests/test_qualification.py): allocated CPU toy
  positive/negative controls, not pretrained native evidence.

From the Mosaic root, **inside an allocated DTU HPC CPU job**, the fixture command
is `python -m pytest experiments/validation/endpoint-qualification/tests -q
--junitxml=<new-attempt-output>/qualification.xml`. Use the authenticated runtime
and immutable capsule declared by the owning attempt. Native Protenix/Boltz/DBMol
evaluators need separate authenticated saved inputs, source/model/context identities
and original null/component checks; these toy fixtures cannot establish them.

Current status, 25 October 2026: implementation prepared; local AST/byte inspection
only. No allocated fixture pass, native diagnostic result, production qualification,
continuation job or gate waiver follows from this preparation. Run records and
actual allocation/evidence links are maintained by the campaign coordinator.

Limitations: deterministic repeated forwards do not rule out FP32 rounding;
agreement along a finite set of directions does not prove the entire Jacobian;
finer-scale agreement does not diagnose a unique cause or erase a coarse-scale
disagreement. Changes to a future production admission protocol require separately
versioned validation and explicit scientific scope.

Allocated toy validation completed on25October2026 in the independent report
source03 CPU job29812102: all19 cases passed, with123 total report/policy cases,
zero failures/errors/skips, actual DONE/hpc/LSF and independent byte/XML/schema
authentication. These are toy diagnostic controls, not pretrained native probes.
The production checkers and all saved scientific outcomes remain unchanged.
Evidence: [report run record](../../reporting/jq1-live-progress/runs/20261025-jq1-live-report-01/README.md).

On25October2026 the user explicitly authorized a separately versioned prospective
endpoint-aware admission check before the requested non-chemistry continuations.
That work belongs to [the continuation run record](runs/20261025-continuation-admission-01/README.md)
and [its declared policy](runs/20261025-continuation-admission-01/prospective-policy.md).
`admission.py` is a new prospective API; the diagnostic-only `qualification.py`
above and all original failed runs remain unchanged. Numerical fixtures, actual
generated-command startup checks and native validations must pass on allocated
HPC before conditional child optimization. Preparation is not a pass or start.

Actual source01 native validation rejected Protenix and both Boltz workers failed
their import path before gradient testing. See the immutable failure audit in the
dated run. A [restricted worker-fix02](runs/20261025-continuation-worker-fix-02/README.md)
preserves that failed source and Protenix's execution block. It changes only
subprocess import plumbing, startup coverage and primary-error reporting; the
prospective numerical policy and all original scientific settings remain exact.

A separate [Protenix20k endpoint investigation](runs/20261025-protenix-endpoint-forensics-01/README.md) preserves the failed admission and all optimization settings. It measures the exact existing probe cohort, scalar execution paths, endpoint derivatives and contact-partner switches; it cannot certify or restart a study.

A new [separate-precision Protenix checker](runs/20261025-protenix-endpoint-fix-01/README.md)
is being validated at the immutable20k endpoint. It separates the original FP32
measurement from an authenticated full-FP64 reference and rejects wrong-gradient
controls. This validation-only capsule cannot submit optimizer work or erase any
original failure. Preparation/local AST checks are not an allocated/native pass.

The first separate-precision candidate failed its real-JAX dtype traversal
canary (266/267passes) before any native work. [Fix02](runs/20261025-protenix-endpoint-fix-02/README.md)
corrects that traversal and has passed all269 allocated CPU fixtures plus actual
metadata-only startup. Native saved-endpoint validation remains required; no
optimizer continuation or original recertification follows from the CPU pass.
