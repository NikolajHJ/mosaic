# Atom exchange costs and structure preservation

The [1-p cost follow-up](atom-exchange-linear.md) tests the user's clarified
additive formula. Results below used the earlier log-regret search cost.

11 November 2026. Question: can mixed atom/charge/bond repair recover invalid
decodes without encouraging sulfur drift or erasing established substructures?
This is an opt-in experiment; existing jobs, callbacks, checkpoints and default
losses are unchanged. See the [sulfur mechanism audit](../../docs/84_sulfur_bias_in_validity_guidance.md).

## Implementation

[`guidance/exchange.py`](../../src/latentmol/guidance/exchange.py) searches
fixed occupied slots with atom, charge and bond edits. The cost is the existing
unit edit count plus decoder probability regret, an extra cost per atom/charge
exchange, and a cost for candidate differences from a fixed valid substructure
reference. The bounded search retains its best verified witness, not a proven
minimum. Acceptance requires full connected, radical-free RDKit validity and
schema roundtrip. Chemistry conversion and target selection stay outside autograd.

The reference protects all rings and three declared motif families: amides,
carboxylic acid/ester groups, and sulfonyl groups. Within each group it protects
atom identities, charges and every unordered pair, including absent chords.
Overlapping groups are counted once per decision. It is a selected motif policy,
not a complete functional-group ontology or a learned molecular prior. References
must be valid and losslessly representable in the canonical aromatic schema;
alternative Kekule spellings requiring normalization are rejected explicitly.

The differentiable loss is repaired-target mean categorical NLL plus
`anchor_scale * (atom_extra_cost * mean_atom_mismatch + charge_extra_cost *
mean_charge_mismatch + substructure_extra_cost * mean_motif_mismatch)`.
Mismatch means `1 - P(reference label)`. Atom/charge means use occupied slots;
the motif mean uses its atoms, charges and unordered pairs. The anchor continues
on already-valid graphs and censored repairs. It never refreshes automatically.
It is a preference, not a ban on sulfur, a valence change or a population prior.
Saturated categorical probabilities can still weaken its gradient.

The existing latent-generation runner accepts an optional `exchange_guidance`
config object. Absent this object, the old loss and config identity are retained.
An enabled new run needs a valid initial graph, or explicit slot-aligned
`start_state['exchange_reference']` labels (`atoms`, `charges`, `bonds`). An
invalid random start without a reference is rejected. The reference is saved
and restored across segments. The usual chemistry coefficient multiplies the
combined repair and preservation loss. Custom step-search policies are not yet
supported with this option. Binding always receives the current decoded graph.

## Small comparison

Command (allocated CPU):

```sh
PYTHONPATH=src python experiments/chemistry-guidance/benchmark_exchange.py \
  --config experiments/chemistry-guidance/exchange-pilot.json \
  --checkpoint /work3/s234818/latentmol/runs/source156-vae-epoch45-structural-attempt002/epoch0045/epoch_0045.pt \
  --output /work3/s234818/latentmol/runs/exchange-20261111b
```

Frozen epoch-45 VAE; seed 111126. Select 16 distinct valid prior decodes with
supported references from at most 512 draws, keeping each draw's occupied slots.
Perturb each latent by a shared independent Gaussian direction of norm 3.
Every arm starts at the identical perturbed latent, without filtering failures.
Run 16 chemistry-only SGD updates, learning rate 3, displacement cap 0.5,
temperature 1, no binding objective, no training and no noise after initialization.
This is recovery near valid molecules, not de novo generation or a sulfur-driving
native optimization. Reference selection restricts the starting population.

Eight arms: existing bond-only structural guidance; extra atom costs 0, 2 and 5
with motif cost 0 or 2; and atom 5/motif 2 with its continuous anchor disabled.
All mixed arms use extra charge cost 1 and anchor scale 0.05 unless disabled.
Thus the search-only arm separates persistent anchoring from search selection;
atom cost 0 still has charge anchoring. Search uses two edit layers, 48 checks
per layer, beam 4. Round-robin bond/atom/charge streams reserve opportunities
for every edit family within the small budget. Bond-only retains its original structural proposals with 96
checks and potentially larger coordinated bond repairs; report actual checks.
The budgets match maximum checks, not identical search spaces or wall time.

Measure raw validity and element counts on all outputs, both hard and expected
sulfur drift from the common perturbed start, reference atom/charge/motif changes,
recovery, target coverage and censoring. Report valid-endpoint uniqueness, mean
pairwise Morgan distance (radius 2, 2048 bits), and distinct nonempty Murcko
scaffolds. Diversity excludes invalid graphs; acyclic outputs are counted
separately. All-state metrics and latent/reference starts are retained.

Budget: 2,048 latent updates, four CPU threads, 1,800-second internal cap,
40-minute allocation, no GPU/native model calls. Corrected DTU job **30000111**.
[Source/submission receipts](../../outputs/exchange-20261111b/) bind the isolated
archive; source SHA `6752870632311f092893764a70d06cd52d0d5bf2aac2d37734680caf22bd9de0`.
The initial isolated job **30000094** used inherited bond-first search ordering,
which could exhaust its budget before trying atom edits. It was left to finish
unchanged and retained as diagnostic evidence in
[`exchange-20261111a`](../../outputs/exchange-20261111a/). The corrected comparison
uses the same seeds, inputs, arm settings and budgets, with fair edit-family
opportunities and explicit target-edit counts.

Local checks: **29 passed**, including the existing latent runner tests, an
expanded-valence sulfur counterexample, ring/side-chain cost distinction, finite
live gradients, occupancy/input invariants, legacy config identity and opt-in
reference persistence across resume. Initial test invocations exposed local
PYTHONPATH/temp-directory setup issues and two corrected fixture defects; the
successful command used `PYTHONPATH=src` and a fresh explicit test directory.

An additional [bounded sulfur stress case](../../outputs/exchange-20261111b/sulfur-stress.json)
used an artificial overbonded four-carbon star and uniform probabilities. At
extra atom cost 0, the selected native-valid witness was `C=S(=C)=C`: one atom
edit, total cost 1. At extra costs 2 and 5, both selected `C=C=CC`: two bond
edits, total cost 2, no introduced sulfur. Each search checked 200 candidates
in about 0.7 seconds. This demonstrates target-selection behavior, not molecular
stability or a measured prevention of long-run collapse.

## Completed results

DTU **30000111 DONE**, 262.04 seconds, all 2,048 updates completed. The completed
checkpoint hash matched `5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be`.
Required latent derivatives were finite/attached and frozen decoder identity and
parameter gradients were unchanged. Downloads of the result, run metadata,
trajectory and starts were SHA-256 verified. The original diagnostic run also
completed (270.89 seconds); the table below uses only the corrected search.

The common perturbed starts were **11/16 valid**. All arms recovered those five
invalid starts. The strongest combined anchor additionally broke one initially
valid start at the endpoint (index 9, sanitization failure).

| Extra atom cost | Motif cost | Continuous anchor | Final valid | Mean atom changes vs reference | Mean protected decisions changed | Morgan diversity | Mean expected sulfur drift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bond-only baseline | — | Off | 16/16 | 1.750 | 3.500 | 0.9033 | −0.01050 |
| 0 | 0 | On | 16/16 | 1.625 | 3.438 | 0.9038 | −0.00960 |
| 2 | 0 | On | 16/16 | 0.500 | 2.125 | 0.8884 | −0.00642 |
| 5 | 0 | On | 16/16 | 0.438 | 2.125 | 0.8921 | −0.00899 |
| 0 | 2 | On | 16/16 | 0.938 | 1.250 | 0.8865 | +0.01296 |
| 2 | 2 | On | 16/16 | 0.438 | 1.125 | 0.9021 | −0.00741 |
| 5 | 2 | On | 15/16 | 0.188 | 0.875 | 0.8880 | −0.00889 |
| 5 | 2 | Off (search only) | 16/16 | 1.625 | 3.438 | 0.9038 | −0.00960 |

All valid endpoints were unique within each arm. Each arm had 15–16 distinct
nonempty Murcko scaffolds; the other valid endpoint in the 15-scaffold arms was
acyclic. Diversity in the 15/16 arm excludes its invalid endpoint and is therefore
not a comparison on an identical eligible set. At atom 2/motif 2, four endpoints
returned to their reference identity, versus none for bond-only: preservation
reduces per-start exploration even though across-start uniqueness stayed high.

Mean hard sulfur count stayed **0.3125 atoms/molecule** in every arm, so mean
hard-count drift was zero. This hides offsetting individual changes: anchored
arms can add one sulfur to one start and remove one from another. For atom
2/motif 2 and atom 5/motif 2, both changes restored reference counts (index 5:
reference 1, perturbed 2, final 1; index 9: reference 1, perturbed 0, final 1).
No sampled state introduced more than one extra sulfur relative to its own
perturbed start. The motif-only arm's positive *expected* sulfur drift is retained
as a warning, not rounded to zero. This cohort did not reproduce runaway sulfur
accumulation, including at zero extra atom cost, so it does not establish that
the new penalty prevents that failure under native binding optimization.

Mixed arms selected 1–2 atom edits and 10–19 bond edits across accepted repair
targets; no charge edits were selected. No invalid-source search was censored.
The search-only arm and zero-extra-cost arm reached the same endpoint identities
on this cohort. Most of the observed preservation benefit therefore came from
the continuous reference term, not search reweighting alone. Search used
960–1,824 actual checks per mixed arm versus 296 for structural bond-only; this
is a cost increase. Trajectory validity was 92.58% for atom 2/motif 2 versus
95.70% for bond-only, despite equal final validity. Stronger preservation is
not uniformly better throughout the path.

Atom cost **2**, motif cost **2**, anchor scale **0.05** is a reasonable next
pilot candidate: 68% fewer protected-decision changes than baseline, equal
endpoint validity/uniqueness and similar Morgan diversity here. It is not a
selected production default. A short chemistry-only pilot cannot establish
binding benefit or rule out long-run collapse. Existing runs remain unchanged.

- [Complete metrics](../../outputs/exchange-20261111b/results/result.json)
- [All-state measurements](../../outputs/exchange-20261111b/results/trajectory.jsonl)
- [Per-start sulfur/invalid-endpoint audit](../../outputs/exchange-20261111b/results/analysis.json)
- [Scheduler completion](../../outputs/exchange-20261111b/results/terminal.json)
- [Verified downloads](../../outputs/exchange-20261111b/results/download-verification.json)

Reporting correction after the run: the legacy structural wrapper does not
expose per-field target edit counts. Its zero-filled `target_edits` in the saved
raw result means **unmeasured**, not zero bond repairs. The current benchmark
now emits `null` for that unavailable field; raw executed results are retained.
