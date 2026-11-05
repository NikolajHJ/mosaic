# Penalized atom exchange in structural repair

This opt-in repair audit replays the 192 raw proposals from the [masked denoising comparison](../20261104-masked-denoising-01/README.md), using the exact saved probabilities at its three checkpoints. DTU CPU job **29897994** finished on 5 November 2026 at 11:30 CET. All 32 allocated fixtures passed, including the legacy structural suite. The paired audit performed zero optimization updates and zero scorer forwards. [Campaign and evidence](campaign.json)

## Comparison

- Both methods use the existing connected-backbone builder and the same native chemistry, schema round trip, and [common-organic environment policy](../../../../../src/mosaic/small_molecule/chemistry/environment_policy.py). The policy checks local charge-aware bond-order caps and limits sulfur to four non-fluorine heavy neighbors. Fluorine still counts toward total valence. The scope keeps sulfur hexafluoride, pentafluorosulfanyl motifs and common sulfones.
- The exchange method tries up to four supported one-site replacements, prioritizing sites implicated by native diagnostics. It preserves charge and occupancy and may also repair bonds. Already-valid in-scope inputs stay unchanged.
- Score each complete candidate against the original graph and saved probabilities. Use the existing edit count plus probability regret, with **one additional edit unit per atom substitution**. A substitution must beat the scoped bond-only witness on cost when one exists. If a newly introduced sulfur would need more bond-order capacity than the original atom, reject that candidate.
- Compare validity coverage, distinct valid molecules, bond edits, selected element transitions, and proposal cost. Repeat selected exchanges with identical inputs and under two support-preserving ±0.001 log-probability perturbations. This measures repeatability and local sensitivity of repair selection on fixed raw draws. Optimizer stability and contact scores require separate tests.

## Result

| Checkpoint | Bond only valid / distinct | Atom exchange valid / distinct | Selected swaps |
| --- | ---: | ---: | ---: |
| Sparse seed 23 | 64 / 4 | 64 / 4 | 4 |
| Sparse seed 41 | 6 / 4 | 56 / 17 | 55 |
| Shared repaired | 64 / 46 | 64 / 44 | 11 |

Under the common-organic gate, validity coverage rose from **134/192 to 184/192**. All 50 additional successes came from the sulfur-bearing seed 41 checkpoint. The old native-only structural benchmark accepted 64/64 there. The matched result here uses the stricter scope for both methods. Twenty other swaps beat a valid bond-only repair on the penalized cost. Diversity fell slightly at the shared repaired checkpoint, so a larger set of allowed edits does not imply more distinct selected targets.

No selected exchange introduced sulfur. Thirty changed sulfur to nitrogen. All 70 selected exchanges replayed exactly. In 140 small probability perturbations, every target remained valid and retained the same atom substitution. Four targets changed their bond edits. The controlled fluorine-to-oxygen case retained both bonds and produced a valid ether. The complete allocation ran for 76 seconds. [Terminal audit](terminal-verification.json)
