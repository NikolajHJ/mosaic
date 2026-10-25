# Measured Protenix20k endpoint findings, 25 October 2026

Diagnostic29813996 completed successfully13:27:30–13:34:23DTU-local. The live
scheduler entry aged out, while its actual same-job LSF success and raw native
assessment remain available. Independent delivery audit authenticates139 files,
108 exact graph points,110 NPZ evaluations, all54 fixed probes and all21
protected inputs. Model/context remain unchanged; zero optimizer updates.
Original29809291 and rejected29813840 remain failed. Protenix is still at20k.

## What the diagnostic establishes

No contact-partner winner changed at any evaluated point in either native
forward path. Thus the hard contact maximum did not explain the observed
failures in this tested cohort. This does not rule out internal ReLU changes
or other nonlinear behavior.

Weak bond finite differences are sensitive to the FP32 execution path. For
Erow90 at epsilon0.0003, the baseline AD displacement prediction is
5.737867453e-7. The independent primal difference is3.576278687e-7; the
AD-returned scalar difference is5.662441254e-7; endpoint-gradient trapezoid
prediction is5.736026289e-7. For Erow47 at the same epsilon, switching scalar
paths worsens the comparison: prediction−4.398857563e-7, independent primal
−4.768371582e-7, AD-returned scalar−2.980232239e-7. These are measured existing
GPU scalars, not locally recalculated derivative results.

Using float64 only for final contact reductions does not remove the path
mismatch. The controls operate on FP32 native logits and do not establish
full-model float64 accuracy. Deterministic same-input repeats do not bound
cross-input arithmetic error. The former checker labels some tiny residuals
as contradictions even when they are below its own reported scalar floor;
that floor is itself not a validated bound on native forward error.

The extra Xrow0 contrast has substantial finite-radius nonlinearity: at0.001
the AD displacement prediction is−1.363347986e-5 while both scalar differences
are+6.169080734e-6. At0.0001 the baseline and endpoint-gradient predictions are
about−1.3633e-6 and the primal difference is−1.2517e-6. Endpoint trapezoids
reuse native AD and are curvature controls, not independent derivative oracles.
The authentic original Xrow3 contrast remains much better resolved.

## What needs validation before50k

A prospective endpoint checker must distinguish measurement resolution from
resolved derivative disagreement and must avoid treating nonlocal probes as
local derivative tests. This can be investigated without changing LR0.3,
NativeContactLoss, temperature, model, checkpoint or optimization arithmetic.
The practical candidate is a predeclared, calibrated uncertainty model plus
feasible, informative atom/bond directions and explicit curvature controls.
Independent-primal observations must remain. It must reject deliberately wrong
gradients and refuse admission when evidence is unresolved; a favorable
forward path or epsilon cannot be selected after looking at results.

These results do not prove every failed derivative correct and do not yet
provide a validated replacement gate. After prospective validation, retain
exact X/E/logits/key replay, original hard parity/null/nonzero/frozen checks,
disposable pilot, runtime and whole-storage guards before any20k→50k child.
The current diagnostic cannot submit one. Original failures stay preserved.

## Allocated scalar tabulation

CPU29815099 completed all4hand-calculable scalar controls and the full54-row tabulation. The AD-returned scalar path improves26comparisons, worsens27and leaves1equal.17predicted differences are no larger than the former checker's reported floor and28are within four such floors. No partner switches were observed in any of110evaluations across all four contact-reduction paths. These descriptive counts preserve all original scales/directions and do not define admission; the old floor is not a calibrated native-network error bound. SummarySHA7c452566ab1747db72e12e6b707d0acfbeff963e83d6d02dbb530e13c242623b; independent12-file source/byte/schema/LSF audit passed.
