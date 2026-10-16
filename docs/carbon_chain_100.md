# 100 projected updates from a carbon chain

Completed 2026-10-16 on allocated DTU HPC. The user requested 100 updates
for each previous relaxation strength, now initialized from a carbon chain,
with loss and molecular changes shown over time.

The hard seed is neutral decane, `CCCCCCCCCC`: ten occupied carbon slots and
nine single bonds. The two starts mix this graph with uniform categorical
probabilities at 0.15 and 0.30, keeping the earlier seed 17, synthetic 20-residue
protein, fixed five-residue engineering pocket, geometry-off native Tiny trunk,
one recycle, frozen weights and fixed forward RNG. The [configuration](../examples/configs/carbon_chain_100.json)
requests exactly 100 accepted projected updates per start. This is a fresh
carbon-chain study, not a continuation from the earlier acetanilide graphs.

Use highest FP32 matrix precision. Initial atom and existence directional
finite differences, same-primal null controls and scalar AD/primal parity must
pass with the previous thresholds. Each update uses initial `eta_X=eta_E=0.01`
and up to ten halving trials, requiring both the AD-evaluated loss and a
standalone primal loss to decrease by more than `1e-7`. A failed numerical
gate stops the attempt; no tolerance or requested update count is silently
changed. Rejected trials are separate from accepted updates.

Every state, from 0 through 100, records graph probabilities, raw derivatives,
both losses, probability diagnostics and a raw argmax RDKit readout. Invalid
or disconnected readouts are retained and labelled, with no repair or fragment
removal. These readouts never enter the differentiated model. Occupancy and
charge stay fixed; native bond features observe existence, so this study
does not supply bond-order-specific gradients.

Allocated-HPC plotting produces a standalone interactive `report.html`, loss
and change plots, fixed-slot raw molecule depictions at updates 0/10/25/50/75/100,
and initial/final probability heatmaps. A hard graph can stay unchanged despite
soft probability changes; report both rather than infer molecular change from
loss alone. Plotting and RDKit analysis also run on HPC. Local work is editing,
static review, packaging, receipt checks and displaying downloaded artifacts.

The plotting dependency extension is prepared on an allocated helper using
hash-locked requirements constrained to preserve every installed package
version. Model/data assets are reused offline. New immutable source and unique
run directories, source/environment/config/fixture bindings, current finite-pool
quota checks (at least 4 GiB/10,000 files free), outcomes and downloads belong
in [the ledger](experiment_ledger.md). The synthetic protein/pocket and this
contact proxy do not establish binding or chemical usefulness.

## Completed result

GPU job **29735673 DONE**, with **128 selected unit checks passed** and
100 accepted updates per start. Each of the 202 saved states has its
unrepaired graph, derivatives, primal/AD losses and detached chemical readout.
Initial presentation-only CPU jobs **29735743 DONE** and **29735759 DONE** used the revised
[`render_protenix_trajectory.py`](../scripts/render_protenix_trajectory.py)
and its [HTML template](../scripts/templates/protenix_trajectory_report.html)
against that completed study. It did not execute the model again.

| Uniform mixture | Primal loss, initial → final | Hard readout at all 101 states |
| --- | --- | --- |
| 15% | 1.844622612 → 1.773505211 | Connected decane; sanitizes. |
| 30% | 1.609976530 → 1.119375110 | Connected decane; sanitizes. |

The hard molecule did **not** switch identity, topology or bond labels.
The continuous graph changed: final mean atom total variation from the start
was 0.03944 / 0.09065, and mean absolute bond-existence change was
0.01846 / 0.04042. The report shows those soft changes alongside the unchanged
hard chain. Every accepted update passed both decline requirements; a
sanitized decane readout is not an optimized binding result.

The offline report embeds every saved state, loss/change figures, probability
tables, endpoint heatmaps and a six-step depiction gallery per case. Case
buttons, a slider and playback control select among the saved states. Runtime
identities, input summary SHA, rendering source/template SHA and plotting
package versions are included. See [the ledger](experiment_ledger.md) for
immutable source/run/download receipts and validation boundaries.

Downloaded report: [open the offline HTML](../results/hpc/20261016-carbon-chain-100/report/report.html).
The sibling PNG/SVG figures and `analysis_summary.json` are also available.
All 699 initial evidence files and 21 final report evidence files passed local
byte-integrity checks; no local scientific computation was performed.

## Probability-weighted depiction update

User-requested presentation update, 2026-10-16: retain the same hard graph
labels, bonds, coordinates and all original numerical results. Atom discs
scale with the probability of their displayed argmax element;
drawn hard bonds scale with `1 - P(NO_BOND)`, the existence probability the
native model observes. The absolute 0..1 scale stays fixed across both cases
and every update. The detached depiction helper uses radius `0.18 + 0.16 p`
in layout units and bond line width
`0.8 + 3.2 p` points. Bold element/slot labels stay fixed at 10 points for
readability. SVG hover titles expose the exact probabilities. These
sizes are presentation cues; occupancy remains fixed, and the circles are not
physical atomic radii. Hard NO_BOND pairs remain absent from the drawing;
the full continuous edge probabilities remain in the heatmap/table.

The experiment includes the fixed synthetic sequence `ACDEFGHIKLMNPQRSTVWY`.
Its selected engineering pocket is residues 5–9 (`F5, G6, H7, I8, K9`, with
one-based residue numbering); the complete protein sequence enters the model
context. The objective combines pocket-coverage and anchor contact terms
using the conservative complete-bin `<8 Å` contact probability. Only ligand
atom and bond probabilities are optimized. No experimental protein structure
or validated binding site was supplied. The report displays this context
from the authenticated experiment fixture/configuration.

Completed: weighted rendering job **29735815 DONE** exposed an inline SVG
font-label issue during browser QA; corrected job **29735830 DONE** renders
all 202 saved states with clear, fixed-size bold element labels in both formats.
The report and figures were downloaded with all 21 declared evidence files
verified. A local CSS-only readability edit keeps the chain at a readable
minimum width with horizontal scrolling in narrow panels; its separate
`presentation_patch_receipt.json` proves the entire HPC-produced HTML body,
SVG/data and plots are unchanged. The report is open in-app. Exact identities,
the retained presentation failure, successful visual QA and final artifact
checksums are in [the ledger](experiment_ledger.md). No new model run occurred.
