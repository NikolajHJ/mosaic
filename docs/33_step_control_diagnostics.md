# Temporary step control and matched saved-state comparison

Implemented and one-step evaluated, 6 October 2026. Source062 CPU
**29611129 DONE** passed 159 engineering fixtures; native comparison
**29611130 DONE** completed all twelve declared comparisons, with twelve
accepted steps and no withheld cases. Nine accepted full proposals;
random seeds 4/6 accepted 1/8 and seed 8 accepted 1/128. Three old full
proposals worsened the objective; their shrunk replacements decreased it.
The three C6 comparisons also accepted full proposals. All children preserve
frozen/parent/audit hashes, pass current native parity and exactly match
cached source061 gradient hashes/norms. Source061 had passed all twelve
declared local-gate audits. Full-trajectory and chemical utility remain
untested by these one-step comparisons, and the curvature operation is
unresolved.

The actual L40S allocation on `n-62-13-19` ran 15:31:42–15:45:28
DTU-local, 826 seconds, maximum host memory 6,278 MB. Its launcher passed
58 fixtures in 8.92 seconds. Oracle receipts verify 25 distinct scored
candidate points/trials, twelve first-candidate cache hits, **73 top-level
forwards** (25 candidates, 24 baselines, 24 native/adapted parity) and
**twelve backwards**, plus checkpoint recomputation. Nine full-step cases
used one trial/five forwards, seeds 4/6 four trials/eight forwards, and
seed 8 eight trials/twelve forwards. These unequal budgets are not an
efficiency comparison.

Final reports/logs are downloaded in
`outputs/hpc-results/source062-step-control-attempt001/result-evidence-20261006.tar.gz`,
52 files/317,883 bytes, SHA-256
`1d7bbbdc288be4ab0ee218e54803d8f9236900606e6397a0472ac7dfbcca0545`.
Extracted controller root is
`runs/source062-step-control-comparison/20261006T133202Z_source062-step-control-comparison_767219f9/`
within that local directory. Its configuration SHA-256 is
`83a7cf4385c9302a0100b618d3a86da516fc8f14acbd05264bf5ca07fb67cb91`.
Each child's `comparison.json`, `oracle_cost.json` and `run.json` was
inspected. The earlier partial stdout-only snapshot remains historical in
the [experiment ledger](05_experiment_ledger.md); the final bundle supersedes
it for source062 completion, costs and identities only.

The final source061 artifacts independently verify all four required local
directions, native parity, exact repeated score/gradient checks and unchanged
frozen/parent identities for all twelve states. They consumed 1,062
top-level forwards and 24 backwards; that prior validation cost stays
separate from the subsequent step-comparison cost. Four retained legacy
directions still lack a plateau, and no sharp-curvature operation or safe
full-trajectory update is established by the new gate's passes.

## What the safeguard establishes

The old learning rate 400 and gradient-norm cap 0.000125 allow a nominal
0.05 latent displacement before the global radius projection. Clipping
limits movement; it does not establish that a move decreases the actual
frozen-pose objective. Source059/source060 showed that some saved states
have a narrow local region where derivatives agree, followed by coarse
secants with very different magnitude or sign. Source061's local gate
addresses derivative probe scale and direction conditioning. The model
operation responsible for the sharp curvature remains unexplained.

`armijo_v1` is a temporary, explicit step safeguard. It evaluates the same
objective on bounded proposals without changing the decoder, Boltz weights,
loss, chemical identity contract or gradient. It is not a root-cause fix,
a chemistry-repair procedure or a fresh-native affinity measurement.

## Configuration and proposal

The reusable implementation is [optimization.py](../src/latentmol/optimization.py),
with `StepControl`, `projected_proposal`, `backtracking_step`, `StepResult`
and the specific `NonfiniteObjective` exception. The Boltz integration is
[active_set.py](../src/latentmol/boltz/active_set.py).

The explicit opt-in configuration is:

```yaml
fd_policy: local_v1
step_control:
  version: armijo_v1
  max_trials: 16
  armijo: 0.0001
  minimum_improvement_ulps: 2.0
```

Omitting `step_control` preserves the historical update and omits this key
from recorded configuration identity. There is no silently enabled new
policy. The example does not alter optimizer settings; their historical
values remain declared in the surrounding run config.

For current point `z`, original start `z_initial`, original gradient `g`
and learning rate `eta`, the controller forms the clipped gradient, tests
the original proposal at scale 1, then scales `1/2`, `1/4`, through
`2^-15` if all sixteen trials are needed. Each proposal is projected into
the existing radius around **the original start**, not around the current
point. Scale 1 preserves the legacy operation order exactly. Every trial
records its proposal and actual representable displacement, gradient norm,
clipping/projection factors and reason for acceptance or rejection.

## Acceptance and failure semantics

Let `delta` be the actual representable projected candidate minus `z`,
and `I = L(z) - L(candidate)`. A trial must have a finite, nonzero
candidate and `g dot delta < 0`, using the **original unclipped gradient**
and a widened diagnostic dot product. It then requires all three:

```text
I > 0
I >= -0.0001 * (g dot delta)
I >= 2 * max(ULP(baseline_loss), ULP(candidate_loss)) + baseline_variation
```

The actual scalar dtype determines ULP spacing. Supplied baseline variation
is additional measured evidence; an omitted value is explicitly unmeasured.
Neither scalar ULP spacing nor the supplied variation is a calibrated
confidence bound on whole-model numerical error. Widening the dot product
does not change native model arithmetic or recover lost forward precision.

Zero/nonrepresentable moves, non-descent projected moves and nonfinite
proposals are retained without an unnecessary score call. An actual
nonfinite scalar objective is rejected through `NonfiniteObjective`.
Malformed affinity heads, integrity failures and unexpected callback
exceptions are persisted and propagated; they are not silently called
stalls. If all trials are exhausted, the unchanged `z` is returned with
`accepted=false`, `status=stalled`. It is not a successful optimization step.

The active-set runner explicitly records attempted and accepted updates,
requested budget, stalls and whether the full budget completed. Trials
hold the selected mask and prepared pose/context fixed. Any later occupancy
change is a separately audited transition; this safeguard makes **no
monotonicity claim across contexts**.

At the source062 submission, the full-budget random-start, protein-panel
and rescore readers did not yet support all new early-stall outcomes. That
evaluation used the dedicated one-step comparator. Source063 subsequently
adds a serial campaign and explicit endpoint termination metadata, described
in [docs34](34_stability_campaign.md), but remains unsubmitted/numerically
unvalidated during the access outage. A config example or passed one-step
comparison does not establish full-study integration readiness.

## Matched one-step comparison

[step_control_comparison.py](../src/latentmol/boltz/step_control_comparison.py),
[compare_saved_boltz_steps.py](../scripts/compare_saved_boltz_steps.py) and
[boltz_step_control_comparison.sh](../scripts/hpc/boltz_step_control_comparison.sh)
compare one old clipped proposal with at most sixteen backtracking trials
at each declared saved state. Both arms use the same latent, original
radius origin, current gradient, frozen decoder/scorer, objective, mask and
pose. Occupancy does not change, and no historical experiment is resumed.

This is **not an equal-oracle-budget comparison**: legacy scores one
proposal, whereas backtracking can try sixteen. The first candidate's exact
cache reuse avoids duplicate scoring, but additional trials are additional
compute. Measured oracle counts and elapsed time are required before any
efficiency claim; matching model/state/loss does not imply matching cost.

Before a comparison, the controller requires exactly one matching passed
source061 `local_v1` audit with declared immutable archive identity,
complete metadata, exact parent/arm/step and file identities, unchanged
scientific state, native-input parity, repeatability and all four required
directions passing. Failed, unfinished, missing or ambiguous audit cases
are withheld explicitly; they are not replaced with favorable cases.

The comparison then independently checks current scientific identities,
GPU family, native/adapted-input parity and grad/no-grad baseline agreement.
It computes one current backward and records the current gradient's
hash/norm against the cached audit. Matching GPU family is not exact
arithmetic certification. The legacy candidate is scored first; its exact
latent-tensor digest enables reuse of that identical first backtracking
trial. Cache hits, distinct scored points, native calls, backward calls and
the prior validation budget remain separate in the reports.

The initial declared cohort is the same twelve failed saved states as
source061: random seeds 0–6, 8 and 9, plus FKBP1A/PIM1/PTPN1 C6. Seed 2
uses step 4 and seed 6 step 1; all other declared states use step 0. Each
comparison has a 300-second child cap, at most twenty native forwards and
one backward; twelve children give at most sixty minutes of child budget.
Checkpoint recomputation and allocation/setup work are additional.

The immutable protocol is prepared locally at
`outputs/hpc-results/source062-step-control-protocol.json`, SHA-256
`0ce040b1030d2b150abe5b0db80cb0a0b4ec0fe8749339180565387f47c70745`.
Its output root is `/work3/s234818/latentmol/runs/source062-step-control-comparison`.
The subsequent source062 archive SHA-256 is
`3c2fd0f49632e67a55c8a070a04a079408f94e67d3f40bee6fcdc1a326076eb5`,
with 259-file source manifest SHA-256
`d20453a9014d9b7d6b20367501316ff9dc86c72d862f552753e2a3057e2a4eb7`.
It is uploaded at
`/work3/s234818/latentmol/releases/source-062-step-control-attempt001`.
The submitted protocol/receipts are under
`/work3/s234818/latentmol/submissions/source062-step-control-attempt001/`,
downloaded to `outputs/hpc-results/source062-step-control-attempt001/`.
Its completed engineering checks and separately completed native one-step
comparison do not establish a full-trajectory result. Numerical commands below are templates
for an authenticated immutable **DTU HPC allocation**, not local tests:

```text
python scripts/compare_saved_boltz_steps.py --protocol PROTOCOL_JSON \
  --validation-root SOURCE061_VALIDATION_ROOT \
  --validation-archive-sha256 SOURCE061_ARCHIVE_SHA256
```

Outputs retain `comparison.json`, `oracle_cost.json`, run metadata,
`comparison-start.pt`, `legacy-proposal.pt` and `backtracking-result.pt`.
The controller's `cases.json` distinguishes withheld, failed and completed
comparisons; a completed comparison may still stall. Improvements are
within-context relaxed frozen-pose proxy changes, with no inference about
chemical validity, synthesis, fresh-pose affinity or full-trajectory success.
