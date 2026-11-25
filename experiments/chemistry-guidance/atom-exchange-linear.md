# Additive atom costs with a 1-p repair objective

11 November 2026. Follow-up to the [first exchange pilot](atom-exchange.md).
Question: with each changed label costing exactly `1 - p(replacement)`, do
additive atom and substructure surcharges improve validity and preservation
without sulfur accumulation or diversity collapse?

## Implemented semantics and tests

Set `edit_cost: one_minus_probability` inside `exchange_guidance` to select the
new search cost. A changed atom costs `1-p(new)+atom_extra_cost`; a changed
charge costs `1-p(new)+charge_extra_cost`; a changed unordered bond costs
`1-p(new)`. Unchanged labels add no base/edit surcharge. Candidate differences
from protected reference decisions add `substructure_extra_cost` per decision.
This structural term is relative to the fixed reference, so it can remain
nonzero even when a candidate leaves the current raw label unchanged. The
candidate-cost calculation and proposal ordering use the same base formula.
`probability_weight` must be 1 in this mode. The historical `log_regret` mode
remains the default for previously written configurations; it is not silently
reinterpreted. Existing runs and sealed sources are untouched.

The decoder's fixed-target NLL and separate expected-mismatch reference penalty
are unchanged. A search surcharge affects target choice, not the derivative of
a fixed target's constant. The reference penalty supplies ongoing gradients.
Its scale remains 0.05; the search-only arm sets that scale to zero.

Protected motifs are rings (including fused/aromatic and nonaromatic rings),
amide N-C=O cores, acid/ester O=C-O cores and sulfonyl O=S=O cores. Atom identity,
formal charge, internal bonds and absent internal chords are protected. Shared
decisions in overlapping motifs count once. Functional-group detection is a
small explicit SMARTS set, not a comprehensive ontology or learned motif model.
Attachments outside a detected core are not protected unless another motif
includes them. The cost is per changed protected decision, not one fixed fee
per whole group. Full-graph RDKit/schema validation still decides acceptability.

**65 focused tests passed** ([63 core/runner checks](../../outputs/exchange-linear-20261111/tests-final.xml),
[two summary checks](../../outputs/exchange-linear-20261111/summary-tests.xml)):
probabilities 0/0.2/0.8/1 for atom/charge/bond cost arithmetic; unchanged-label
zero cost; all declared functional groups and both ring types; fused overlap;
undirected-bond counting; permutation and padding invariance; one double-precision
directional AD/FD preservation check; invalid/nonfinite/asymmetric inputs;
exhaustive two-edit, three-node comparisons at atom costs 0/0.5/2; both cost
modes through generation/resume; element-independent surcharges; schema-reordering
and ambiguous-mode rejection; summary failure/sulfur denominators and duplicate
cohort rejection; previous frozen-model/runner regressions.
The exhaustive result is specific to that fully covered tiny space, not a
global guarantee for the bounded pilot beam. A reference-schema fingerprint
guard was added locally after the three snapshots were submitted and passed
these tests. The allocated runs use their original snapshots with the same
decoder/spec throughout; their executed sources were not modified.

## Expanded matched comparison

[Config](exchange-linear.json), using the existing `benchmark_exchange.py`.
Three seeds **111127, 111128, 111129**, each with 16 distinct valid prior
references from the same frozen epoch-45 VAE. Each seed alternates perturbation
norms 3 and 5 (eight starts each), followed by **32 chemistry-only latent updates**.
Thus 48 starts, nine arms and **13,824 updates**, compared with 16 starts and
16 updates in the first pilot. No native binding scorer or decoder training.
All arms within a seed reuse identical latents, masks, references and budgets.

- Existing structural bond-only baseline.
- Log-regret control: atom/charge/motif surcharges each 0.5.
- New 1-p cost with extra atom costs 0, 0.5 and 2, each with motif cost 0 or 0.5.
- New 1-p atom 0.5/motif 0.5 with continuous anchoring disabled.

All mixed arms use extra charge cost 0.5, two edit layers, 48 candidate checks
per layer, beam 4. LR 3, displacement cap 0.5, temperature 1 and fixed occupied
slots. A cost of 2 already exceeds the base cost of any single bond edit; it
can also dominate a two-bond alternative. These are preferences within the
bounded search, not a guarantee against all sulfur shortcuts. No parameter
selection or subsequent run is automatically authorized by the outcome.

```sh
PYTHONPATH=src python experiments/chemistry-guidance/benchmark_exchange.py \
  --config experiments/chemistry-guidance/exchange-linear.json --seed 111127 \
  --checkpoint /work3/s234818/latentmol/runs/source156-vae-epoch45-structural-attempt002/epoch0045/epoch_0045.pt \
  --output NEW_OUTPUT
```

Budget per cohort: four allocated CPU cores, 2,700-second internal cap and
50-minute allocation. Jobs **30000226, 30000227, 30000228**. All use source SHA
`02554efe36376bc3535ce9a00c22d16c1a25a90ad2fa5c135482eb48a3714e98` and new output
directories. [Submission receipts](../../outputs/exchange-linear-20261111/submissions.json).

Aggregate with `summarize_exchange.py`: all-start endpoint/trajectory validity,
recovery and newly broken valid starts, reference changes, target edit counts,
search censorship/compute, hard and expected sulfur drift including individual
changes, and valid-only fingerprint/scaffold diversity. Common-valid subsets
are reported to expose survivorship effects. Paired exploratory intervals
resample matched starts within each seed; per-seed and perturbation strata
remain visible. They do not establish native binding performance.

A separate [16-case sulfur cost sweep](../../outputs/exchange-linear-20261111/sulfur-stress.json)
varies proposed sulfur probability (0.05/0.2/0.5/0.8) and extra atom cost
(0/0.5/1/2) on an artificial overbonded carbon star. Costs 0 and 0.5 selected
sulfur in all four probability scenarios. Cost 1 selected bond repair at
0.05/0.2 but sulfur at 0.5/0.8; cost 2 selected bond repair in all four. This
is a target-cost probe: its probabilities are hypothetical categorical inputs,
not constrained to have the raw stress graph as their argmax (especially at
sulfur probability 0.8). It is not molecular stability or trajectory evidence.

## Completed expanded results

All three jobs are **DONE**: 584.47, 469.34 and 367.63 seconds respectively
(1,421.44 summed run-seconds). All 13,824 updates completed, with finite attached
required latent gradients and unchanged frozen decoder weights/parameter grads.
The completed epoch-45 checkpoint hash matched
`5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be` in every run.
Result, trajectory, starts and metadata downloads were SHA-256 verified.

There were 48 distinct reference molecules across seeds and 41/48 valid common
perturbed starts (21/24 at radius 3, 20/24 at radius 5). Reference coverage was
46 molecules with rings and 9 with amide cores; categories overlap. No acid/ester
or sulfonyl references happened to enter this panel. Those branches have direct
fixture tests, but no trajectory-level evidence here.

All rows except the log control use the new 1-p cost. Surcharges are additive;
continuous anchor scale is 0.05 except the explicitly disabled arm.

| Atom / motif surcharge | Final valid | Mean atom changes | Mean protected changes | Morgan diversity | Mean hard S drift | Mean expected S drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bond-only baseline | 48/48 | 2.333 | 4.646 | 0.9053 | −0.0208 | −0.0158 |
| Log control 0.5 / 0.5 | 47/48 | 1.042 | 2.708 | 0.8988 | −0.1250 | −0.1305 |
| 0 / 0 | 47/48 | 2.313 | 4.625 | 0.9055 | −0.0208 | −0.0161 |
| 0.5 / 0 | 47/48 | 1.042 | 3.542 | 0.8973 | −0.1250 | −0.1280 |
| 2 / 0 | 46/48 | 0.667 | 3.625 | 0.8940 | −0.1458 | −0.1477 |
| 0 / 0.5 | 48/48 | 1.625 | 2.771 | 0.9009 | −0.1042 | −0.0840 |
| 0.5 / 0.5 | 48/48 | 0.979 | 2.500 | 0.9003 | −0.1458 | −0.1464 |
| 2 / 0.5 | 48/48 | 0.646 | 2.250 | 0.8965 | −0.1458 | −0.1480 |
| 0.5 / 0.5, search only | 47/48 | 2.313 | 4.500 | 0.9034 | −0.0208 | −0.0158 |

All valid endpoints were distinct. The 0.5/0.5 and 2/0.5 arms each retained
24/24 valid endpoints at both perturbation radii and 16/16 in every seed.
Their 46 distinct nonempty Murcko scaffolds compare with 44 for baseline;
fingerprint diversity nevertheless decreased slightly. Common-valid subset
diversity is also available in the full summary, including for the arms with
invalid endpoints. Distinct outputs do not imply unchanged exploration freedom.

Adding motif cost 0.5 at atom cost 0.5 reduced protected changes **3.542→2.500**
and final invalid outputs **1→0** in this sample. Compared with baseline,
0.5/0.5 reduced protected changes by **46.2%** (paired mean difference −2.146,
within-seed paired-start bootstrap 95% interval −2.688 to −1.625). Trajectory
validity was 99.09% versus 99.02% baseline. These are exploratory results, not
evidence of a general validity improvement: 48/48 in both arms does not prove
equal failure rates. The direct log-control difference is also only one endpoint.

At atom 2/motif 0.5, protected changes fell further to 2.250 (51.6% below
baseline), with lower Morgan diversity 0.8965 and trajectory validity 98.76%.
Atom 2 without the motif cost ended with 46/48 valid, including one initially
valid start that became invalid. More atom pressure alone was not uniformly
beneficial.

Mean initial hard/expected sulfur counts were 0.4167/0.4218. Average hard and
soft sulfur decreased in every arm. Nevertheless, two 0.5/0.5 endpoints and
one 2/0.5 endpoint gained sulfur relative to their own starts. One invalid
intermediate state in the 2/0.5 arm gained **two** sulfur atoms absent from both
its start and reference (seed 111129, index 0, step 2). No persistent population
collapse occurred in this short chemistry-only run, but the surcharge is clearly
not a hard sulfur constraint. Shared latent changes can alter atom identities
even when accepted repair targets change only bonds.

The 0.5/0.5 arm selected **zero atom edits**, zero charge edits and 14 bond edits
across accepted repair targets. The motif-only arm selected four atom edits;
the zero-cost arm selected one. Thus the sulfur stress fixture is the direct
evidence of atom-cost ranking; most trajectory preservation here comes from
the ongoing reference penalty and its interaction with the decoder. Search-only
guidance ended at 47/48 valid and 4.5 protected changes, much closer to baseline.

Bounded search censoring is retained: zero-cost and search-only arms each had
32 censored attempts, representing one persistently unresolved trajectory;
0.5/0.5 had one. Structural baseline used 870 native candidate checks versus
1,344 for 0.5/0.5 and 1,824 for 2/0.5. Different repair spaces/strategies remain
a confound against the baseline. No claim of a globally nearest repair follows.

Conclusion for the next experiment: retain both **0.5/0.5** (moderate preservation)
and **2/0.5** (stronger substitution aversion) as candidates, since the former
fails the sulfur stress shortcut and the latter reduces fingerprint diversity
and can still transiently gain sulfur through the latent. Neither is adopted
as a production default. Native binding optimization and targeted acid/ester/
sulfonyl trajectory panels remain untested.

- [Pooled results, per-seed/perturbation strata and paired intervals](../../outputs/exchange-linear-20261111/summary.json)
- [Reference coverage and all-state sulfur audit](../../outputs/exchange-linear-20261111/audit.json)
- [Seed 111127](../../outputs/exchange-linear-20261111/111127/result.json),
  [111128](../../outputs/exchange-linear-20261111/111128/result.json),
  [111129](../../outputs/exchange-linear-20261111/111129/result.json)
- Each seed directory retains scheduler completion, verified download hashes,
  starts and all-step measurement records. Historical results remain unchanged.
