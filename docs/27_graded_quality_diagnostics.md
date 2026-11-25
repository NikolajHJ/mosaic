# Graded hard-graph diagnostics and quality summary

Status: **source051 version 1 passed DTU HPC tests and its matched panel
completed; source052 version 2 passed a light local smoke but has not run on HPC**
(1 October 2026). **6 October extension:** an evaluation-only,
probability-aware repair diagnostic is implemented in source059. HPC CPU
retry 29609935 passed 112 fixtures and 13 actual historical-parent readers;
its 124-membership chemistry audit completed on 6 October: 77 already
valid, three bounded repair witnesses, 12 censored and 32 missing states.
Historical
v1/v2 mixtures remain unchanged. The code measures
hard graphs after sampling. It changes neither AE/VAE training nor latent
optimization, and it trains no validity, repair, binding or synthesis scorer.
See the [next-test protocol](26_next_test_protocol.md) and
[experiment ledger](05_experiment_ledger.md).

## Attempt-level contract

`diagnose_graph(atoms, charges, bonds, spec)` accepts the shared slot-ordered
integer atom, formal-charge and bond labels. It retains RDKit validity,
SMILES or failure reason, a slot-label graph identity even for invalid
chemistry, heavy-atom count, total charge, connected-component sizes,
structural label defects and graded diagnostics. Schema/label failures are
reported as unsupported or unavailable; RDKit-invalid but schema-supported
graphs still receive applicable diagnostics. The caller must retain every
raw attempt, checkpoint, cohort/index, argmax or categorical mode, occupancy
policy, temperature and seed. It must distinguish missing, ineligible and
not-run results instead of interpreting a failed SMILES conversion as a
zero-valued molecule.

Standalone evaluation enables `graded=True, retain_graphs=True` for
`reconstruction_metrics`, recording `graded_chemistry` and the raw
`predicted_graph` for each graph in its configured `chemistry_limit` subset
in `examples.json`; it aggregates only over that subset. The reusable
function defaults both switches to `False` so existing training probes
retain their previous cost and output. It still evaluates reconstruction
losses over the full selected loader; do not confuse those denominators.
Neighborhood and VAE-prior examples now retain raw graph labels
and graded records, including invalid outputs, in `neighbors.json` and
`prior_examples.json`; their summaries stay separate by scale and prior.
The fixed-`z` architecture study retains all raw graphs in
`attempted_graphs.pt`, adds per-attempt `graded_attempts.json` and summarizes
argmax and categorical cohorts separately. Both evaluator CLIs accept an
optional JSON `--quality-config` of `QualityScoreConfig` weights and record
the chosen weights in the run configuration. The source051 selected suite
passed 66 HPC tests; external-model and repair branches remain separate.

`quality_panel.py` is a local driver for the source050 sealed validation
panel. It verifies the panel, data-manifest and **all** candidate checkpoint
hashes before invoking the fixed-`z` study. Its intended two arms use the
same 128 validation indices, seed, graph schema, temperature and 16
categorical draws per molecule, plus separate argmax attempts, on CPU. It
records separate scores for argmax and categorical and the original unequal
training budgets; it makes no final architecture selection. Its repair and
native-scorer call counts are zero **for external models**. In v2, bounded
diagnostic search may still make RDKit candidate-verification calls; each
child's `graded_compute.physical_candidate_checks` and `cache_hits` account
for those separately. Source051 job 29540604 completed this
panel: the VAE's 2,048 categorical attempts had 1,963 RDKit-valid outputs
and default mixture 97.1829/100; the same-family AE control had 2,007
valid outputs and 98.6604/100. Their separate argmax scores were 97.4175
and 98.4301. The completed
[comparison artifact](../outputs/hpc-results/graded-quality-source051/runs/20261001T100922Z_graded-quality-panel_c8775663/comparison.json)
retains exact component values, denominators and profiles; the
[ledger](05_experiment_ledger.md) records full provenance. These are
matched **evaluation** attempts, while source047 VAE/AE checkpoints are
from epochs 89/63 and panel membership had historical test exposure.
The default score only measures heuristic hard-graph chemistry readiness;
it does not establish an overall AE winner or a useful interpolatable VAE
space. No repair or native scorer was called.

## Version 2: equal thirds and bounded repair proximity (pending HPC)

The new default `QualityScoreConfig` gives **one equal weight each** to
`repair_proximity`, `local_consistency` and `connectedness`, divided by their
total and reported on 0–100. Binary RDKit validity remains an explicit raw
count/fraction but has **zero default mixture weight**. The latter two
components retain `1/(1+problem_burden)` and `1/component_count` (zero for
empty graphs), so all three terms have the same 0–1 scale. This is a
transparent heuristic, not a calibrated probability of validity. A
caller-supplied v1 JSON profile is loaded through
`QualityScoreConfig.from_dict`, preserving its original binary weights and
profile identity; source051 v1 results above are never retroactively scored
as v2.

`repair_proximity` is `1/(1+d)` **only when** a valid hard-graph witness is
verified at observed edit cost `d`. The evaluation-only
`RepairSearchConfig(max_edits=3, checks_per_depth=256, beam_width=4)` uses a
deterministic, capped label beam search. It may edit occupied atom types,
occupied formal charges and undirected bond categories, one label per edit;
it keeps occupancy and every slot fixed and counts each bond once. Search
prioritizes diagnosed problem slots and disconnected-component links, then
other labels. Candidate chemistry is checked by the same `diagnose_graph`
validity contract. A found witness stores its graph labels, SMILES, graph
ID, edit counts and actual checks. The raw candidate is unchanged; the
witness is **diagnostic evidence**, never substituted as an input to a
frozen model or optimized molecule. This is neither an external DeFoG
projection nor a trained chemistry prior.

The first found witness is an **observed upper bound** on the nearest valid
graph in this fixed-occupancy label space; beam/check caps can hide a
closer witness. Distance is exact for already-valid input (`d=0`), a
one-edit repair (`d=1`), or a two-edit repair only when the complete
one-edit layer was exhausted. If no witness is found, `d` and the point
proximity are `None` (censored), not zero or infinity. Each attempt retains
distance/proximity bounds and search status; the cohort reports a
`score_interval` from those search-space bounds when all attempts have
search records. **This interval is not a confidence interval.** A censored
graph makes the default point mixture unavailable under the unchanged
positive-weight missing-term rule; the interval and binary valid count
remain reportable. Report the search policy, actual physical candidate
checks and cache hits. `GradedEvaluator` caches up to 512 exact slot-label
graph IDs to avoid repeating search for duplicates while retaining each
draw in the attempt denominator.

Version 2 leaves the external supplied-output `measure_repairs` API below
separate; its `repair_quality` component is **not** the bounded-search
`repair_proximity`. Search does not support variable occupancy or establish
the nearest valid molecule globally. Source052 archive SHA-256
`4c477d4573a6079c04ad4ef1501669d3af05de5ef1f9c04b443ca558990b1b85`
passed a user-authorized **local** 32-test smoke in 3.99 seconds from its
extracted immutable source. The full selected tests, small preflight and
same sealed 128 × 16 panel with matched CPU search budgets remain planned
for HPC. At this documentation snapshot, **no source052 HPC job or v2
checkpoint/panel result** is established; see the [ledger](05_experiment_ledger.md)
for the local command and environment.

## Probability-aware repair diagnostic (source059; completed HPC audit)

The user requested a distance-like diagnostic that accounts for how much
the decoder supports the edits needed to make a hard graph valid. This
extension is **supplemental**: it does not replace the historical v2
`repair_proximity`, change the equal-thirds mixture, train a scorer or add
an optimization loss. Chemistry conversion and witness verification remain
outside autograd. The witness is never substituted as a shared molecule,
input to Boltz or an optimized endpoint.

For the raw hard graph and its same-state decoder distributions, keep
occupancy and slot identity fixed. A changed occupied atom type, occupied
formal charge or unordered bond category counts as **one label edit**;
each changed bond counts once. Atom probabilities are conditional on the
occupied slot being a real atom, excluding `NO_ATOM`. The initial policy
permits at most **three edits**, **256 candidate checks per depth** and a
**beam of four**; these are configurable, recorded search limits rather
than a claim to enumerate all valid molecules.

The supplemental [probability-repair module](../src/latentmol/probability_repair.py)
provides `ProbabilityRepairConfig`, `measure_probability_edits`,
`search_probability_repair` and `ProbabilityRepairEvaluator.diagnose(...,
probabilities=...)`. The probabilities must have the shared padded atom,
charge and bond categorical shapes, finite normalized rows and symmetric
bond distributions. Raw logits or guessed probabilities are rejected.
The wrapper cache has default capacity 128 and returns independent result
records; physical search checks and cache hits stay separate from attempt
denominators. These interfaces have been statically reviewed; their
HPC CPU retry 29609935 passed the combined 112-test regression suite and
all 13 prescribed real historical-parent reader checks, and the subsequent
chemistry audit completed. This validates the diagnostic engineering and
bounded witness evidence, not a gradient gate or binding comparison.

For each edit `old → new`, retain its two decoder probabilities and compute
the nonnegative probability regret:

```text
regret = max(0, ln(max(p_old, epsilon) / max(p_new, epsilon)))
weighted_cost = edit_count + probability_weight * sum(regret)
epsilon = 1e-12
probability_weight = 1 by default
proximity = 1 / (1 + best_observed_weighted_cost)
```

The logarithm is natural and clipping is explicit in the recorded policy.
Every edit costs at least one, so multiple changes do not become free when
probabilities are nearly tied. At an argmax label, regret describes lost
factorized categorical likelihood: changing a nearly tied bond category
costs less than replacing it with a very unlikely category. For a sampled
raw label, its replacement may have greater decoder probability; the
regret floor then leaves the unit edit cost rather than rewarding a change
with negative cost. The score depends on the probability/temperature
contract and weight, so different policies cannot be compared as if they
were the same measure.

Search continues after the first valid witness to seek a better
probability-supported repair within the same capped budget. It separately
retains the **best observed weighted-cost witness** and **best observed
label-edit count** among verified repairs. They may come from different
graphs. Each is an **observed upper bound** on its respective nearest-valid
cost in this fixed-occupancy, slot-labeled space; a one-edit weighted
witness does not by itself prove minimum weighted cost. Beam/check caps
can hide a cheaper valid witness. Already-valid input has zero cost and
proximity one. Without a witness, the point cost/proximity remains `None`
with explicit `censored` status, rather than a fabricated zero or infinity.
Empty/unsupported graphs or missing/mismatched probability evidence retain
their separate status rather than receiving a made-up repair.

For each verified witness, retain the raw and corrected slot labels,
graph identities, SMILES, edit type/count, old/new categories and their
probabilities, per-edit regrets, weighted total, policy, actual candidate
checks and capped-search status. Also report replacement surprisal
`sum(-ln(max(p_new, epsilon)))`, with clipping counted, and separately the
product of the actual replacement probabilities through their unclipped
log sum. A replacement with exactly zero support records mass zero and
an unavailable finite log-mass; the clipped cost remains explicitly labeled.
Very small nonzero products may underflow in floating point, so retain the
log-mass too. These involve only the **changed labels**;
they are neither the full witness's graph likelihood nor a probability that
the molecule is chemically valid. Decoder outputs share hidden context;
the categorical product is a factorized diagnostic, not a calibrated joint
chemistry distribution or empirical validity probability.

Probability-aware caching must identify **the exact conditional probability
arrays, raw hard graph and full policy**. The same hard graph decoded with
different confidence can require different weighted costs and cannot reuse
a graph-only cached result. Frozen decoder logits can be reevaluated on HPC
from a saved latent state, retaining checkpoint/state/schema/occupancy/
temperature identity and hard-label parity. Missing evidence is unavailable,
not inferred from the hard labels alone.

The [saved-state audit driver](../src/latentmol/probability_repair_audit.py)
accepts a JSON protocol listing unique terminal parent pilots, their run-file
hashes and prescribed state indices, plus the common search policy. It
replays the frozen graph decoder on CPU HPC using each saved active mask
and historical temperature, checks the same conditional argmax against
saved graph labels/hashes when available, verifies parent/checkpoint/schema
identity and confirms that the decoder remains unchanged. Missing states
stay explicit, including a failed parent's absent state 100; missing hard
candidate parity is labeled unavailable. It loads no Boltz model and makes
zero optimization updates. The invocation is:

```bash
python -m latentmol.probability_repair_audit \
  --protocol PROTOCOL.json --sha256 PROTOCOL_SHA256
```

The first sealed source059 protocol evaluates predeclared **argmax starts
and endpoints** from **49 terminal parents**: all ten random pilots,
35 completed and three failed panel pilots in the 09:41 snapshot, and the
historical C13 control. It prescribes **124 state memberships**: both
random/C13 arms at states 0 and 100, one panel arm at those states, plus
state 4 for random seed 2 and state 1 for random seed 6 in both declared
arms. Missing states are retained rather than replaced. The selected audit
uses **three edits, 2,048 checks per depth and beam width eight**, with
probability weight one and floor `1e-12`; this is a larger declared search
budget than the reusable primitive's 256/four defaults. These are witness
limits, not an exact-search guarantee.

Remote protocol is
`/work3/s234818/latentmol/submissions/source059-diagnostics-attempt001/probability-repair.json`,
SHA-256
`ee1210bbda613049ba53b5c9f5e94b288a0509b776f23a5042b5fe58d86b9a7b`.
Its explicit terminal-parent/state cohort is separate from the original
100-case panel and ten-seed workflow denominators; unstarted parent studies
remain visible in their original reports and are not fabricated as
terminal-pilot artifacts. All prescribed missing states and invalid graphs
remain in this audit's denominator, without score-driven replacement or
new random draws. Exact cohort snapshots, source/config hashes, commands,
fixture results and diagnostics must be retained in the ledger before
scientific claims. CPU retry **29609935** completed the audit at
`/work3/s234818/latentmol/runs/source059-probability-repair/20261006T080709Z_source059-probability-repair-audit_3c132de6`.
Initial job **29609924** failed before tests because a manually transcribed
protocol digest had 65 characters; the same immutable code was retried with
the digest read directly from the file. Downloaded protocol identity above
was verified with `Get-FileHash`; the malformed initial command and failure
remain in the ledger. The audit completed at **10:12:36 DTU-local** with
**77 already-valid**, **three witness-found**, **12 censored** and
**32 missing-state** records, totaling the predeclared 124 memberships.
It retained **75,600 physical candidate checks**, **37 cache hits**, frozen
decoder parity and zero Boltz calls/optimizer updates. Repeated state
identities are not independent molecules; absent/censored observations
retain their status rather than a fabricated zero cost.

Seed 7's invalid active endpoint has a verified **two-edit witness**:
slot 16 **N→C** (probabilities 0.62586981→0.37359190) and bond [1,2]
**DOUBLE→SINGLE** (0.98988360→0.00999735). Weighted cost is
**7.11124518**, proximity **0.12328563**. This illustrates how a chemically
valid witness can be nearby in label edits but poorly supported at one
required bond change. It is an observed upper bound, **not an exact minimum
or a validity probability**, and the raw optimized graph remains invalid.
Other witnesses repaired random-0 active state 0 in three bond edits
(weighted cost 9.18098549) and random-6 active state 1 in three label edits
(7.88729229). No witness was substituted into native scoring.

Downloaded `repair-audit-run.json`, `repair-audit-summary.json` and
`repair-audit-repairs.jsonl` are under
`outputs/hpc-results/source059-diagnostics-attempt001/`. The numerical
gradient diagnosis remains pending: queued A40 replay 29609925 was
cancelled without execution and the same protocol is pending as **29609958
in gpul40s**, requesting L40S hardware. Original failed gates remain failed.

## Implemented graded components

| Component | Current computation | Interpretation and limit |
|---|---|---|
| `validity` | Binary RDKit/`inspect_graph` valid flag. | Keep valid count and **all-attempt** denominator; the mixture never replaces binary validity. |
| `local_consistency` | `1 / (1 + problem_burden)`. Burden counts distinct slots implicated by RDKit `DetectChemistryProblems`, radical slots and otherwise unlocated problems; explicit asymmetric-bond/self-bond defects also contribute. Empty graphs receive burden 1. | Intermediate credit for invalid graphs, **not** a probability, graph-edit distance or guaranteed smooth repair direction. Unknown construction failures make the term unavailable. |
| `connectedness` | `1 / component_count` for occupied heavy-atom components; zero for an empty graph. | Descriptive graph topology, not native scorer eligibility by itself. |
| Supplementary overvalence | Bond-order excess above a finite allowed maximum for covered neutral elements or a small explicit charged-element table; report checked slots and coverage. | **Not in the default mixture.** Aromatic bond environments are skipped; unsupported charged/valence cases remain uncovered. There is no generic low-valence deficit because implicit hydrogens are allowed. |

The historical **source051 v1** `QualityScoreConfig` used weights
**0.4 validity, 0.4 local consistency, 0.2 connectedness**; external repair
and native-retention weights were zero. The current code's **v2 default**
instead uses the equal thirds defined above. `combine_quality` divides
each bounded component's weighted value by
the configured total weight and multiplies by 100, yielding a 0–100
heuristic chemistry-diagnostic score when all positively weighted terms are
available. Weights must be finite/nonnegative with positive sum; custom
weights are permitted. A missing positively weighted term makes `score=None`
and status `unavailable`—there is no renormalization over observed terms.
The output retains every component, contribution, missing term, weights,
version and profile hash. Different weight profiles cannot be ranked as if
they were the same metric. The score is **not** calibrated chemical validity,
overall model quality, binding, synthesis feasibility or selectivity.
A model that repeatedly emits valid methane could score 100: retain diversity,
size, latent-neighborhood behavior and later native Boltz useful-yield as
separate axes, and never select an architecture solely from this mixture.
Likewise, two overvalent graphs with the same implicated RDKit atom can have
the same default local-consistency score even if one exceeds the covered
valence cap by one bond order and the other by three. The supplementary
`overvalence_excess` reports that magnitude where its rule applies.

`summarize_quality` averages attempt-level components over the **entire**
cohort only when each attempt has that component; otherwise that component
is unavailable, with observation coverage and schema-unsupported/failure
counts reported. Keep argmax, categorical, prior and neighborhood cohorts
separate. Supplementary size/charge drift, novelty, uniqueness and captured
valid decoder-law mass remain separate planned analyses, not hidden terms in
this implemented mixture.

## Supplied external repair evidence

`graph_edit_cost(raw, corrected, spec, corrected_to_raw=...)` compares
shared-slot labels under an explicit native-to-shared permutation if needed.
No canonical-SMILES reorder is inferred. It counts occupancy, atom and
charge edits plus each undirected changed bond **once**, including bonds
incident to occupancy changes. The inputs must have symmetric bonds without
self-loops. This is a fixed-alignment label cost, not the mathematical
distance to the set of valid molecules.

`measure_repairs` **does not run a projector**. It accepts actual graph
outputs or `None` failures from an identified external method and exact
positive budget (`projector`, checkpoint SHA-256, number of attempts,
transitions per attempt, integer seed and nonempty policy object). It records
every attempt and rejects mismatched
budgets or slot maps. For an invalid raw graph, the best observed valid
output's edit count is retained; if none appears, distance is `None` and
status is `censored` at that budget. The reported repair utility is
`exp(-best_observed_edit_cost / edit_scale)`, with default `edit_scale=3`;
a censored observation has utility zero **without** claiming zero edit cost.
An already valid raw input has zero distance-to-validity and utility one;
any changes made by its projector remain in the separate attempt records.
Fixed-occupancy mode excludes size-changing outputs from the valid repair
set. `summarize_quality` accepts optional per-graph repair reports only if
raw graph identities and budget/policy match; absent reports mean `not_run`.
Repair utility enters the mixture only under a custom positive weight.

`native_retention` is an explicit unavailable placeholder until a frozen
model's identity, eligibility and native-rescore normalization contract is
implemented. A repaired molecule has a new identity; its native score is
not a score for an invalid raw graph. Tanimoto and native scores of invalid
raw graphs may be undefined. DeFoG is neither assumed to be a valid
projector nor required for already suitable molecules.

## Verification before scientific claims

Numerical tests, chemical examples, aggregation checks and integrated
evaluations must run on DTU HPC from an immutable source/config release and
unique run directory. Test coverage should include valid and invalid hard
graphs, implicit-H/aromatic/charged boundaries, radicals, component counts,
symmetry and upper-triangle bond counting, zero versus censored repair,
budget/identity mismatch, all-attempt denominators and custom-weight
arithmetic. Record job status, source/config/data/checkpoint hashes, raw
attempt artifacts and failures in the ledger. No quality ranking or effect
on downstream Boltz performance follows from this API alone.
