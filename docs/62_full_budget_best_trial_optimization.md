# Future latent optimization with best-trial acceptance

**Final CPU engineering regression passed, 18 October 2026 13:58 DTU-local:**
unchanged source122 attempt003 **job 29749480 DONE: 423 passed, 0 failures,
0 errors, 0 skipped** across 16 modules in 36.61s. This includes the final
full-step-equality amendment. The result is allocated CPU engineering and
compatibility evidence, not native v2 Boltz validation or molecular efficacy.
No pretrained-model call, Boltz optimization rerun or existing-campaign change
occurred. [Verified result](../outputs/hpc-results/source122-best-trial-cpu-002/local-verification.json)
and [ledger](05_experiment_ledger.md) retain the source/command/job/JUnit proof;
the earlier failed transfer and intermediate source002 result remain below.

**Earlier final packaging/upload failure, 18 October 2026 13:14
DTU-local:** source122 attempt003 includes the full-step-equality rule and
passed standard-library AST parsing of the final 17 Python files, without
imports or numerical execution. Its first SFTP transfer failed before
authentication with exit 1: `Permission denied (password). Connection closed`.
At that earlier observation there was **no attempt003 job or allocated result**.
The later authorized retry and passed result are recorded above. The 411-pass attempt002 result remains
valid intermediate evidence, not validation of this final amendment.

| Final attempt003 identity | SHA256 |
| --- | --- |
| Archive | `15fe0329a228fcde5f963fe27b4f1232ce80544a003702831e72bfcf62804038` |
| Scientific manifest | `07de1fdd1b1a52af627d0364fcd641df9e9526ebef9130efc0ef0f737074d7e0` |
| Preparation | `eaee2d691429585477414c3c995c4b3742a8a75cf0074d8bc13f02533e821464` |

[Final preparation](../outputs/hpc-submissions/source122-best-trial-attempt003-preparation.json),
[exact failed transfer receipt](../outputs/hpc-submissions/source122-best-trial-attempt003-upload-failure.json)
and [ledger](05_experiment_ledger.md) retain the command/error and historical
test outcome. The sealed local archive is unchanged by this later status note.

**Full-step-equality amendment, 18 October 2026:** apply a full step immediately
when its loss is no greater than baseline, including equality/no-op. This edge
uses `full_step_nonincrease` when it is not a strict improvement. New source122
attempt003 regression subsequently passed 423 fixtures as recorded above;
the 411-pass result below remains valid intermediate evidence only.

**Intermediate post-sealing engineering result, 18 October 2026:** source122 attempt002
CPU job **29749152 DONE: 411 passed, 0 failures/errors/skips** across 16 fixture
modules in 36.82s. CUDA was hidden and Hugging Face offline. No pretrained-model
call or Boltz optimization rerun occurred. These are allocated CPU engineering
and compatibility results, not native v2 Boltz validation or chemical/binding
evidence. The immutable archive preserves its pre-test documentation.
[Verified result](../outputs/hpc-results/source122-best-trial-cpu-001/local-verification.json),
[preparation/source identities](../outputs/hpc-submissions/source122-best-trial-attempt002-preparation.json)
and [ledger](05_experiment_ledger.md) retain the actual outcome and the
held-unsubmitted first package.

Authorized on 18 October 2026. Implementation and local syntax review are
complete and intermediate allocated CPU engineering fixtures passed. The latest
full-step-equality clarification passed allocated CPU regression. No native Boltz-model
run under the new policy is established.

## Scope and historical boundary

Future `best_trial_v2` optimization attempts the requested update budget even
when bounded step trials find no lower loss. It changes step acceptance, not
the molecular identity, objective composition, frozen models, native derivative
validation, context protocol or resource accounting. Numerical fixtures and
model execution remain allocated DTU HPC work.

Existing sealed studies retain their exact source/configuration and protocol.
In particular, source120's Boltz-only control has 225 accepted/226 attempted
updates followed by an authenticated healthy Armijo stall, not a full 500-update or
convergence result. Its guided arm retains the sealed Armijo protocol. The new
policy neither rewrites those outcomes nor authorizes a rerun. See the
[comparison record](60_long_boltz_defog_comparison.md) and
[experiment ledger](05_experiment_ledger.md).

## Versioned selection rule

The new `StepControl()` default is `best_trial_v2`; historical behavior remains
available through explicit `armijo_v1`. New materialized configurations and
receipts must declare which policy they use; future long-plan generation also
defaults to v2. Existing records are not silently
reinterpreted, and a configuration without step control does not gain a v2
certificate merely because the constructor default changes.

For each current latent state, compute the proposal using the declared
gradient, clipping, learning rate and trust-radius projection. Hold its
context, shared molecular mask, repaired target and coefficient fixed while
comparing scalar losses. The compared scalar is that state's actual complete
optimization objective, not just a favorable component of it.

1. Score the full proposed step, scale `1`. If its actual loss is no greater
   than the current loss, apply it immediately, including equality/no-op.
   A strict improvement uses `first_improvement`; other full-step nonincrease
   uses `full_step_nonincrease`, without a descent or forced-fallback claim.
2. Otherwise score successive halves, `1/2, 1/4, ..., 2^-15`, applying the
   first strict improvement immediately. The default is 16 total sizes; the
   declared `max_trials` may range from 1 to 16.
3. If the configured trials finish without improvement, apply the finite
   candidate with the lowest actual loss, including a loss equal to or greater
   than the current one. Equal losses choose the earliest trial. A finite
   equal/no-op candidate caused by rounding can therefore consume an update.
4. Continue toward the requested update budget. Lack of a decreasing step is
   not a healthy-stall termination under v2.

V2 applies the full step on observed nonincrease and uses strict observed
decrease for early acceptance of later halves. Armijo and scalar-ULP
diagnostics do not impose additional v2 acceptance conditions. A forced
fallback is not an Armijo or monotone-descent pass. Trust projection still
bounds proposals; it does not prove that a selected fallback improves loss.
For future v2 runs this explicitly replaces the historical native
Armijo/monotone-decrease acceptance requirement. Native source/input/gradient,
finite-value and trust-radius requirements remain unchanged.

Nonfinite candidates/objectives cannot win. Exhausting the configured trials
with no finite candidate is a genuine failure. Nonfinite baselines/gradients,
failed native gates, unexpected exceptions and resource exhaustion also remain
failures or partial execution; none permits manufactured finite values,
fabricated update receipts or a claimed complete budget.

## Records and interpretation

Each step receipt records its baseline, every attempted scale, finite/rejected
observations, chosen trial, selected loss and:

- `acceptance_kind`: `first_improvement`, `full_step_nonincrease` or `best_finite_fallback`.
- `monotone_descent`: whether the chosen actual loss strictly decreased.
- Applied, strictly improving, forced-fallback and no-op outcomes separately.

Summary counters are `applied_updates`, `improving_updates`,
`forced_fallback_updates` and `no_op_updates`. The compatibility field
`accepted_updates` counts applied updates; its name does not certify descent.
Attempted, applied and requested counts remain distinct. A full-budget claim
requires the actual requested updates and all required native/terminal proof;
all exceptions and resource stops retain their true partial counts.
An equal full step is applied without increasing improving/fallback counters;
an unchanged point also increases `no_op_updates`.

Per-state metrics add `selection_reason`, `strictly_improved`,
`applied_loss_improvement` and `no_op`. The improvement is current minus selected
loss, so an uphill fallback has a negative value. The report retains those
observations without converting them into a strict-improvement count.

Summary `best_binding_states` retains best-observed latent state evidence
separately from the endpoint. Its key is SHA256 of the serialized JSON pair
`[context_sha256, active_slots]`; the same key is recorded as
`context_mask_sha256`. Each value contains:

| Field | Meaning |
| --- | --- |
| `step` | Index of the saved scored state |
| `loss` | Binding-only signed/scaled loss at that state |
| `state_path` | Ordinary saved scored state, `best-binding-{key}.pt` |
| `context_sha256` | Authenticated native context identity |
| `active_slots` | Occupied-slot identity used for this comparison |
| `context_mask_sha256` | Hash key for the context/occupied-slot pair |
| `scope` | Saved scored states; binding-only signed/scaled loss within this fixed native context |

A record changes only on a strictly lower binding loss within that same
context/occupied-slot pair, retaining the earlier best on ties. It does not
compare different native contexts or occupied masks. Binding-only observations
and total repair-augmented loss are distinct: changing repair targets or
coefficients do not define one comparable total-loss ranking, so no global
best combined-loss snapshot is selected across those changes.

Loss may rise or remain unchanged after a fallback, so neither the endpoint
nor the applied-update count proves improvement or convergence. Saved-only
reports retain these points and show fallback/no-op events alongside losses,
context/target/coefficient changes, invalid graphs and partial runs. They must
not substitute a best-observed snapshot for the actual trajectory endpoint.
Fixed-context loss changes remain separate from fresh native predictions,
chemical validity and efficacy claims. See the
[interactive-report contract](59_interactive_optimization_reports.md).

## Validation status

Local standard-library AST parsing of 17 Python files passed. Reviewer Python
AST and Node `--check` validation of report assets also passed syntax checks
only. There was no local unit/numerical execution. The first immutable
CPU-only unit/compatibility package attempt001 was held unsubmitted after a
remote-path formatting issue. New immutable attempt002 retained the same
scientific manifest and passed 411 fixtures in CPU job 29749152; actual source,
wrapper, job, JUnit and scheduler DONE identities were authenticated in the
[result capsule](../outputs/hpc-results/source122-best-trial-cpu-001/local-verification.json).
No existing HPC job or sealed source120 protocol changed for this release,
and it did not run Boltz optimization. This is intermediate evidence; the
subsequent equality clarification is packaged as immutable attempt003. Its
first upload failed authentication; the authorized retry passed 423 fixtures.

Implementation/static review and allocated CPU fixture results must be recorded
with immutable source/configuration identities, commands, job IDs and actual
receipts in the ledger. Tests should cover immediate full-step acceptance,
first improving half, deterministic finite fallback including uphill/tied/no-op
cases, all-nonfinite failure, real exception propagation, update counters and
explicit historical Armijo compatibility. Passing such CPU fixtures verifies
the optimizer's engineering behavior; it is not a new Boltz model run or a
chemical/binding result. The final 423-pass CPU result has that engineering
scope; it is not a native v2 Boltz or full molecular optimization certificate.
