# Next-test protocol: latent geometry, frozen Boltz, and external-prior gates

Status: **candidate panel sealed on HPC; scientific tests remain planned**
(1 October 2026). This is a VAE-first feasibility study with a matched AE
comparator, not a claim that a VAE has a
smooth or chemically useful latent space. Existing AE/VAE outcomes and failures
remain in the [experiment ledger](05_experiment_ledger.md). Full sampling,
optimization and evaluation run on DTU HPC; the narrow local smoke-test
exception during the current outage is recorded in `AGENTS.md` and the
ledger.

## 1. Seal the candidate panel before downstream scoring

Use the same `zinc215k-seed17-v1` graph cache and one seeded **validation**
cohort for every candidate. The **provisional plumbing panel** in
`configs/next_test_panel_stable_provisional.json` names the 256-dimensional
`stable_deep_pair_refine` VAE at `best_validation_validity` as primary and
the same-family stable AE at that role as its control. The wide VAE
epoch-120→180 continuation is still pending; preserve its completed
epoch-120 result as a separate baseline and do not assume more training
improves it. A later wide candidate requires its own explicit panel revision
and rationale. An AE/VAE contrast must identify architecture and checkpoint
role, not pool unlike models.

For each candidate, declare exactly one role from
`common_epoch_60`, `best_validation_reconstruction`, or
`best_validation_validity`; record the completed parent workflow, sealed
`selection_before_test.json` hash, checkpoint path/hash/epoch, source/config
hashes, graph schema and dataset/split hashes. Epoch 60 is the equal-update
AE/VAE comparison. Later validation-selected checkpoints have different
training budgets and answer a separate best-available-under-policy question.
Each parent workflow selected its checkpoint role **before that workflow's
test evaluation**. The provisional panel itself was assembled **after
historical test results were known**, so it is not a pre-test model choice:
its plan records `historical_test_exposure=true` and a written selection
rationale. Do not use later Boltz or DeFoG outcomes to retune this panel
without a separately recorded revision. Preserve any inherited versus new
continuation winner explicitly, and keep fresh held-out assessments separate
from the already exposed historical test split.

`src/latentmol/candidate_panel.py` validates full-workflow/continuation
parent lineage, exact source and checkpoint hashes, and the plan's
historical-test-exposure flag, then seals validation indices before a
downstream scorer call. Source049's first HPC test gate failed and remains
in the ledger. Corrected source050 job **29539938** passed five panel tests
and sealed the provisional panel in
`20261001T094301Z_decoder-candidate-panel_adc34cfe` (`panel.json` SHA-256
`8daf20dae156a9407525d94bb1bf22782ede3ca8d1a886676c6136c18248764c`).
It contains the stable VAE epoch-89 primary, stable AE epoch-63 control and
128 validation indices. This verifies panel plumbing only: no decoder
forward, hard graph, frozen scorer or chemistry measurement is reported.

## 2. Measure latent behavior on one shared validation cohort

Use identical molecule indices, decoder temperature, atom-count policy,
sampling seeds and hard-draw budgets across matched AE/VAE arms. Keep all
attempts in denominators, including invalid graphs, duplicates and unsupported
external-model inputs. Report both argmax and categorical hardening, with
ordinary variable occupancy distinguished from any later fixed-occupancy
Boltz view. Freeze decoder weights throughout.
The general validation cohort can contain molecules ineligible for the
fixed-C13 Boltz adapter. Report the original cohort denominator, every
Boltz exclusion and its reason; a later C13-eligible subset must not be
presented as representative of all validation molecules.

| Question | Predeclared measurement |
|---|---|
| Can local moves change chemistry without destroying it? | At several latent radii scaled to each model's observed aggregate latent spread, count raw valid draws, valid **changed** molecules versus each starting identity, unique valid identities, and chemically invalid/unsupported attempts. Report exact duplicates and graph/standardization identity changes. |
| Is local change graded? | For valid hard neighbors, report structural similarity to their own starting molecule (for example one pinned fingerprint/Tanimoto definition) by radius, alongside atom/bond edit counts. Invalid graphs stay in the attempt denominator; do not assign them a fabricated similarity. |
| Do interpolations stay plausible and intermediate? | Preselect validation endpoint pairs and an interpolation grid. At each point report argmax and matched draws: validity, identity, similarity to **both** endpoints, and an explicit intermediate criterion against endpoint-copy and random-pair controls. Smooth latent coordinates alone do not guarantee a semantic molecular path. |
| Does the prior cover encoded molecules? | Keep standard-normal VAE draws separate from encoded-mean/aggregate-posterior draws; report valid/unique identities, atom count and charge distributions, and latent coverage diagnostics on the same declared budget. AE normal draws have no justified native prior. |
| Is a score gradient useful after hardening? | For each optimization step, compare relaxed score change with raw valid changed hard yield, standardized identity, and native recomputation. Retain trajectories whose relaxed score improves but whose hard molecule does not. |

Record expected and hard heavy-atom counts, total formal charge, charge-state
and size drift, connectedness, sanitization failures and representation
exclusions. Similarity, novelty or reconstruction is not binding, synthetic
feasibility, selectivity or proof of a useful generative manifold.

### Graded pre-projection closeness (source051 HPC panel completed)

Continue reporting **binary final RDKit validity**. Also retain every
schema-supported raw candidate for graded diagnostics before any projection;
RDKit invalidity alone does not discard it from this analysis. This does not
make an invalid graph eligible for a native model that requires valid hard
chemistry. Predeclare the following measures and count failures explicitly:

- Local violation severity from atom type, formal charge, bond order and
  neighborhood: **overvalence** where a declared atom/charge rule applies,
  plus disconnected or inconsistent neighborhoods. Do not assign a generic
  low-valence deficit: this heavy-atom graph permits implicit hydrogens.
  Aromatic and charge-specific cases need explicit applicability limits.
  Fractional expected valence can hide invalid hard draws, so inspect each
  hard graph. This is a diagnostic heuristic, not a validity probability or
  guaranteed repair gradient.
- Under a fixed atom/slot alignment, graph-label edit cost from the original
  candidate to the **best valid result actually reached** by a compatible
  external projection attempt. Compare at equal attempts and transition
  budgets; use occupied-slot inverse/native maps so canonical reordering is
  not counted as a chemical edit. Count each undirected bond once and report
  atom, bond, charge and size edits separately. This estimates achievable
  repair cost under that projector and budget, **not** the exact distance
  to the set of valid molecules. If no valid result appears, mark the
  observation censored or
  failed at the budget, not zero-cost or silently absent.
  An already valid input has zero distance to validity; report any projector
  changes to it separately rather than relabeling them necessary repairs.
- Valid-sample fraction and estimated decoder-law mass of observed valid
  unique graphs within small edit budgets, keeping all attempts and duplicate
  probabilities visible. Captured mass is not the unknown total valid mass.
  Compare against no-projection and random-edit controls at matched cost.
- Property/score retention from each candidate to **that same candidate's**
  repaired hard graph, with identity changes and a fresh native rescore where
  the frozen model supports both. Tanimoto and native scores of an invalid
  raw graph are undefined; in that case report the repaired molecule's score
  under its new identity, not a fictitious retention ratio. Unscorable inputs,
  failed repairs and native failures remain explicit; a score on a different
  selected molecule is not evidence of retention.

DeFoG is neither a guaranteed nearest-valid projector nor mandatory for a
candidate already suitable for the next scorer. The new hard-graph API and
evaluation integration can report RDKit problem burden, connectedness,
supplementary covered-atom overvalence and a configurable 0–100 heuristic
chemistry summary on reconstruction, neighborhood, prior and fixed-`z`
cohorts. Its
fixed-alignment edit-cost function measures **supplied** external outputs;
it does not run a projector. The broader valid-mass and native-retention
analyses remain proposed. Source051 tested the integrated chemistry summary
on the source050 sealed validation cohort; its exact outcome and limits are
in [docs27](27_graded_quality_diagnostics.md) and the
[ledger](05_experiment_ledger.md). External-model measures still
require their own validated input/identity contract. No new in-house scorer
training is authorized. The exact implemented formula and missing-component
rules are in [docs27](27_graded_quality_diagnostics.md).

**Version-2 scoring request (local smoke passed; HPC panel pending):**
replace the v1 binary-validity contribution with a bounded, verified
fixed-occupancy graph-label repair proximity, and give it, local
consistency and connectedness equal default weight. Keep binary raw
validity as a separate denominator. A search witness gives an observed
upper bound on edit distance; capped search without a witness leaves
the point contribution unavailable and reports distance/score bounds,
not a fabricated zero-distance success or statistical confidence
interval. The same hard candidate remains the molecular identity; no
search witness is silently substituted into Boltz or another scorer.
Search budget, checks, cache hits, supported occupancy and failures must
be explicit. Source051's v1 measurements above retain their original
profile and are not retrospectively reweighted. Source052 passed a narrow
user-authorized local smoke during the HPC outage, without real checkpoint
or panel evaluation. The exact v2 contract and pending HPC gate are in
[docs27](27_graded_quality_diagnostics.md).

**Optional alternating guidance after repair/parity gates (proposed only):**
if a compatible external projector returns a hard corrected graph `H`, freeze
and detach `H`, then use the existing
`sampling.hard_graph_log_prob(decoded(z), H, spec, ...)` to differentiate
`-log q_z(H)` through the **frozen graph decoder into `z` only**. Require `H`
to match the canonical graph schema, aligned slots and the declared common
occupancy law; structural support violations give no usable finite target.
This optional differentiable fixed-target surrogate is distinct from the
graded **hard edit measurement** above. It is a likelihood-to-corrected-target
update, **not** differentiation
through the projector and not an unbiased or complete gradient of the actual
repair process. Do not train the AE, decoder or external model. Keep an
explicit trust/movement budget and test whether the same repaired identity
retains native score and diversity. A repair that changes atom count cannot
enter the current fixed-C13 Boltz branch. No code, job or result implements
this option yet.

## 3. Frozen Boltz fixed-C13 gate, then a small matched trajectory study

The existing adapter has a **fixed 13-slot occupancy, fixed pose and partial
input gradient**; it cannot test variable atom count or gradients through
pose diffusion. For each selected decoder, first verify that a C13 anchor (or
a decoder-only selected nearby start) hardens to the same slot-ordered atom,
charge and bond labels in both ordinary and fixed-occupancy views. Select a
nearby start without querying Boltz, as in the earlier C13 rescue protocol.
Reject ineligible molecules instead of silently neutralizing, reindexing or
changing identity.

On HPC, pin the Boltz source/weights, prepared target/MSA, pose/context,
adapter, decoder checkpoint and feature-loss policy. Check one-hot native
forward parity, unchanged frozen parameters/context, finite nonzero gradients
to the supported graph channels and `z`, and multi-scale finite differences
for the **actual scalar objective**. The expected-charge mapping is lossy for
soft categorical charges and requires explicit opt-in. A passing derivative
gate is only local fixed-context evidence.

Then use matched starts, latent trust radius, update count and **actual Boltz
forward/backward call budget** for VAE, AE, frozen-start and random-direction
controls. Log unweighted objectives, gradient norms/conflicts, latent movement,
every sampled/argmax hard graph, invalid/duplicate/unchanged attempts, and
relaxed-to-hard score retention. Predeclare a native confirmation subset
without using native scores to tune the relaxed trajectory. Rebuild the same
hard candidates' ordinary ligand inputs and fresh poses; record effective
standardized identity, failed native calls and cost. Fixed-pose gains that
vanish under this rescore are not binding improvements. See the narrower
[Boltz stage](04_latent_pipeline.md) and [shared model contract](23_frozen_model_plugin_contract.md).

Only after that fixed-C13 gate, test the requested active-set extension in a
separate Boltz-only arm. The decoded probability of `NO_ATOM` selects each
slot at a predeclared `0.5` threshold; Boltz receives a fixed count during
each gradient step and may switch to another audited prepared context
between steps. Compare with the same-start fixed-count arm at matched
optimization-step and hard-confirmation budgets, while reporting **total**
Boltz calls and GPU time separately: context changes trigger extra parity,
finite-difference and transition calls. Report both atom-count and same-count
slot-identity changes, crossings that reverse on the next step, unsupported
counts, context-switch score discontinuities and the number of distinct
hard valid candidates. This probes an indirect latent signal for atom count;
it cannot establish a direct occupancy derivative. Source054 passed the
HPC CPU fixture suite and the selected stable VAE's exact C13 start audit.
The 2 October outcome check confirmed completed native contexts for counts
11–15, followed by failure of the initial Boltz finite-difference gate.
Hard-seed/native forward parity passed, but the random probe direction
passed only one of the required two epsilon scales. No optimization update
or fresh-pose follow-up occurred in source054. Source055's saved-state
diagnostic subsequently supported insufficient numerical resolution at the
smallest probes; reruns kept the original tolerances and used larger probe
sizes. Both ten-update pilots and fresh-pose follow-ups have now completed.
The original learning rate retained the C13 chain, while the separate
larger-step experiment produced valid `CCCCCCCCCC1CC1C` and higher native
binary scores at two pose seeds. The continuous head did not corroborate
improved activity. Both arms retained 13 atoms, with identical endpoints
and no context switches: variable-count behavior and an active-set advantage
still require testing. The ledger preserves all failures and exact outcomes.

## 4. DeFoG and any later frozen model remain separate gates

The tested **public** DeFoG checkpoint failed the complete soft-gradient
gate: native auxiliary features rejected the fractional input, and the edge
derivative did not pass the declared finite-difference check. It lacks our
formal-charge and NO_ATOM semantics. This is not a test of DBMol's adapted
charge-aware DeFoG. First test hard-only native loading, graph/charge
eligibility, hard-input label parity, neutral compatible cases, transition
wrapper behavior, all attempted outputs and compute cost. Do not enter a
DeFoG gradient in a combined objective on the basis of a successful hard
projection. See [the DeFoG evidence](18_stochastic_defog_projection_protocol.md)
and [alternative-model assessment](20_chemical_prior_alternatives.md).

Prefer a deterministic map from the **one shared canonical graph** to each
frozen model's native input. Train a small graph-to-native input/early-feature
bridge only if needed, using paired hard molecules and context, never reward,
binding, synthesis or validity labels. Verify held-out native feature and
forward parity, hard identity, then soft-path gradients and finite differences;
freeze bridge and external model before optimizing `z`. A `z`-assisted bridge
also needs cycle/identity consistency with the same shared graph. Failed
parity or unsupported chemistry means abstention or an explicitly separate
hard-only branch, not a learned surrogate scorer. The eligibility and
combination rules are in [docs23](23_frozen_model_plugin_contract.md).

No multi-model gradient combination is scheduled until each real branch
passes its own provenance, hard-input, derivative and hard-outcome gates.
When combination becomes eligible, measure each objective and gradient
separately before weighted sums, with matched budgets and native confirmation.

## 5. Ordered frozen-model expansion (planned, no integration claim)

The user accepted this order: **establish the frozen Boltz-2 baseline first;
then evaluate Nesso-1; then investigate Protenix**, which an advisor
recommended. For each model, pin its public checkpoint, code, license and
native input/output contract before making a capability claim. Record
whether its native output is an affinity estimate, a structure/confidence
measure, or something else. Structure confidence must not be reported as
binding affinity, and a fixed-pose differentiable proxy must not be treated
as fresh-pose native confirmation.

**Nesso-1 audit:** its [technical report](https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf)
§2.1 describes an affinity model conditioned on pair features/distances
without the atomistic diffusion stage, making a direct frozen affinity
gradient worth testing. The reported 22 Å/15 Å cropping is discrete and
needs an explicit fixed-context or eligibility policy before a derivative
claim. Its affinity data/curation overlap with Boltz-2 raises a
**correlated-error concern**, not a measured independence advantage. Pin
the [public repository](https://github.com/recursionpharma/nesso), weights
and exact native feature semantics; no Nesso input parity or gradient has
yet been tested here.

**Protenix audit:** the [repository README](https://github.com/bytedance/Protenix)
lists a public v1 and a v2 with explicit proprietary weight restrictions;
audit a pinned v1 checkpoint first and defer v2 until its specific access
terms are resolved. The inspected live-main [model code](https://raw.githubusercontent.com/bytedance/Protenix/main/protenix/model/protenix.py)
exposes a distogram head that depends directly on pairformer output,
independently of generated coordinates;
the [distance head](https://raw.githubusercontent.com/bytedance/Protenix/main/protenix/model/modules/head.py)
is a symmetric pairwise **structural** prediction, not affinity. A
pre-diffusion distance/contact objective is therefore a candidate gradient
path, subject to hard-native parity and finite differences. Predeclare a
pocket-specific distance/contact-consistency target or another justified
structural scalar; normalize over eligible pairs so raw contact count or
molecule size is not rewarded. Maximize-all-contacts or confidence is not an
affinity objective, and hard candidates need fresh native-pose checks. Its confidence
path instead consumes predicted coordinates and distance features; the
[confidence module](https://raw.githubusercontent.com/bytedance/Protenix/main/protenix/model/modules/confidence.py)
defaults to stopping gradients, while recycling in the model gates gradients
on training mode/final cycle. An optional fixed-pose confidence gradient
would be partial and must handle these controls without silently changing
native forward behavior. The ordinary native pose/confidence evaluator can
still serve as a hard-only check if no pathwise branch passes. These are
code-reading hypotheses from **unpinned live main**, not a working relaxed
graph adapter.

Each branch receives the **same canonical molecular graph and biological
target**, with its own native crop/MSA/pose context pinned and explicitly
mapped. First establish hard-input identity and native-forward parity,
then test whether its supported relaxed input has a finite, nonzero and
finite-difference-consistent gradient to the same molecular `z`. A small
paired graph-to-native bridge is allowed only when the frozen model's input
requires one; verify held-out feature/forward parity and freeze the bridge
before latent optimization. Unsupported graphs, lost stereochemistry,
different occupancy rules, changed molecular identity or unavailable
gradients remain explicit. Keep hard-only native outcomes separate when a
soft gradient path fails. Do not train a new in-house activity, validity,
synthesis or reward scorer.

Only after **each** chosen branch passes its own identity, derivative and
hard-outcome gates should a combined-gradient study be designed. Log each
unweighted objective and `z` gradient, scale/angle/conflict, eligibility
intersection, per-model call budget, trajectory and native retained outcome
before selecting weights or an update rule. Set component scales from the
validation cohort, and keep an independent held-out native evaluator for
branches used during optimization. Initial matched arms are Boltz-only,
Nesso-only and Boltz+Nesso. If a Protenix structural gradient later passes
its own gates, add Protenix-only, Boltz+Protenix, Nesso+Protenix and the
three-model combination on the same eligible cohort. Keep matched starts,
latent trust radius, update count, hard-draw and native-confirmation budgets,
plus no-update and random-direction controls. Also compare against
single-model controls given the same total compute budget; that is a
separate comparison from equal-update arms. Report per-model oracle calls
and measured GPU time, since model costs differ. This is a planned evaluation
sequence, **not** a claim that Nesso-1 or Protenix is loaded, compatible,
differentiable or useful; no job or new numerical result follows from it.
The local config-driven runner and extension contract are documented in
[docs28](28_multimodel_experiments.md). Source053 passed 65 local
mock/fixture tests from an immutable archive, without native weights,
genuine prepared bundles or HPC execution. This does not replace any
branch-specific scientific gate.
