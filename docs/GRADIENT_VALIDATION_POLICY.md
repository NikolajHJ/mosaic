# Common gradient validation policy

Policy ID: `gradient-validation-policy-v2-exploratory`. Updated 30 October 2026
for DBMol and Mosaic after the user requested faster early-signal research.
This replaces v1's blanket qualification prerequisites for new exploratory work.
Read together with the [exploratory research policy](EXPLORATORY_RESEARCH_POLICY.md).

## Exploratory default

Use the shared exploratory entry points. Check finite inputs/objective/gradients,
compatible shapes and an attached required live path. Reject corrupt mappings or
unexpected frozen-model mutation when observed. One backward smoke is normally
enough before a small bounded run. Do not require a second AD implementation,
a finite-difference sweep, native parity, repeated forwards, monotone descent or
bitwise reproducibility as routine admission for early exploration.

A flat component can be correctly differentiated. Distinguish it from a detached
required path where observable. In JAX, a zero returned gradient alone cannot
distinguish a true plateau from `stop_gradient`; a smoke is not evidence that
this ambiguity was resolved. Use an informative fixture or qualified diagnostic
when that distinction matters. A nonzero-gradient requirement belongs to an explicitly informative
fixture or an experiment's usability question, not every component. Lack of
progress or worse molecules is an experimental result, not by itself a gradient
implementation failure. Save it and reach the bounded endpoint unless the user
or the selected study explicitly requires stopping.

Report scope honestly: a finite/attached smoke is not a derivative-correctness,
native-parity, reproducibility or chemical-improvement certificate. Record optional
FD as `passed`, `failed` or `inconclusive`. Do not suppress a resolved material
contradiction, label an unmeasured check passed, or alter an existing strict
receipt reader to accept missing evidence.

## When checking derivative accuracy

Use explicit qualified mode or call the detailed checker intentionally. Validate
the actual objective/mapping. Where supported, compare independent forward and
reverse AD with meaningful relative/absolute tolerances. Forward/reverse agreement
alone can share an incorrect mapping. Keep negative controls for wrong sign,
scaling, missing paths, nonfinite values and mutated state; include valid flat
and informative fixtures. Test the changed checker locally with small fixtures;
use native/device checks when that is the claim being made.

Finite differences measure a secant of a piecewise numerical program. Plan
feasible directions/scales from geometry and dtype, use the represented input
displacement, and account for output resolution and repeat-forward variation.
An infeasible probe, cancellation, unresolved signal or a branch-crossing interval
is inconclusive about the local derivative. Never choose a scale because it
matches AD, inflate tolerances to pass an actual error, or demand that every
probe yield a resolved nonzero difference. Qualified claims need independent
evidence for whatever their FD cannot resolve.

## Numerical comparisons and records

Use exact identity for immutable assets when identity matters. Use dtype-aware
numerical comparisons for recomputed outputs, permutation tests and cross-device
results. A changed computed-gradient digest is not model mutation. Avoid large
absolute floors that erase or reverse small informative gradients. Bitwise replay
is an explicit specialist contract, not the exploratory default.

Detailed producers/readers record and enforce their numerical policy and scope;
readers must not blindly trust a `passed` flag. Preserve explicit old contracts,
failed outcomes and sealed releases. Prospective fixes do not change live jobs,
scientific settings, caches or checkpoints.

## Maintained entry points

| Repository / entry point | Current scope |
| --- | --- |
| DBMol multimodel CLI and new active-set configs | Exploratory finite/attached checks by default; explicit qualified mode retains native/independent AD validation. |
| DBMol pointwise-v5 and embedding start-gate v2 | Detailed qualification tools allowing attached flat components with relative AD evidence; v4/v1 remain explicit legacy replay. |
| Mosaic shared graph checks and chemical updates | Exploratory finite backward and bounded updates by default; qualified mode requests detailed FD/JVP/native-primal evidence. Explicit `legacy-fixed-v1` retains its strict scope. |
| Mosaic endpoint continuation hooks | Versioned numerical replay tools; authenticated historical continuations keep their original readers and settings. |

Historical baseline: 543 scoped allocated CPU cases passed on 30 October 2026
(292 DBMol / 251 Mosaic). That evidence retains its original scope and source;
it is not an admission requirement for each new exploratory run. New tests and
native/GPU claims are recorded separately.
