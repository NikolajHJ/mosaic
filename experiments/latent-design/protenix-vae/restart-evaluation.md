# Independent chemistry evaluation of long-search winners

Question: do the strongest archived Protenix contact scores also produce
reasonable molecular properties? This bounded pilot selects one best
fresh-scored valid identity from each completed long search, rather than
the global top three or a random sample. Source runs remain unchanged.

The evaluator is Mosaic branch `codex/molecule-evaluation`, commit
`ef2cc6eeb76ce3e56147c75f59fd928be07571d6`. Its committed CPU-only package
was copied into a small source bundle; no native models were loaded.
Input records and report notes explain each molecule's job, wave, branch,
step, restart policy and historical score. SMILES were preserved without
repair, protonation changes or assigned stereochemistry.

| Source | Job | Wave / branch / step | Original fresh loss | QED | SA score | Rings |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Near-valid | 29943712 | 7 / 02 / 432 | 2.05132 | 0.66254 | 8.36914 | 13 |
| Deep | 29943705 | 15 / 04 / 2 | 2.08135 | 0.40602 | 7.49542 | 6 |
| Wide | 29943696 | 8 / 06 / 432 | 2.09213 | 0.49183 | 8.04736 | 14 |

All three source searches used LR 60, 500 updates per child, chemistry
and valence weights 1, step cap 0.25, zero optimization noise and 120,000
total updates. The near-valid winner used a screened latent restart;
the deep winner used exact continuation. A configuration label does not
attribute the winner to invalid-parent expansion.

## Command, compute and result

The existing `mosaic.small_molecule.evaluation.evaluate` API was called
with the detached JSON input, provenance in `details` and `notes`, CPU 1,
and no docking. The source/input bundle hash is
`d1b381c840d423a55aa9c359cc9a4b81a23a74228c31645d81578116d9f297c4`.
The generated command and settings are retained in the
[evaluation request](../../../outputs/restart-molecule-evaluation-20261111a/evaluation-request.json)
and [small invocation script](../../../outputs/restart-molecule-evaluation-20261111a/run-evaluation.py).

DTU `linuxsh` allocated PBS interactive job **726394.hnode41** on
**n-62-30-8**. Evaluation plus imports took **1.32 seconds**, using the
resident CPU Python environment without installing or changing dependencies.
The interactive allocation completed normally. All three inputs were
sanitized, connected, neutral, distinct and had 25 heavy atoms. Average
pairwise ECFP4 distance was 0.89345. Deep had one Lipinski violation
(six hydrogen-bond donors); the other two had none.

The SA estimates are high, especially for the highly caged near-valid and
wide winners. These results expose a distinction between valid topology,
a favorable contact objective, and practical synthesis. QED and Lipinski
properties alone do not capture that concern. No conformational strain,
synthetic route, docking, pose geometry or binding affinity was established.
No prepared BRD4 docking receptor or protein-frame ligand poses were
available for this first pass. Selected-input validity is not generation
validity; three selected winners do not establish a population comparison.

[Interactive HTML](../../../outputs/restart-molecule-evaluation-20261111a/results/Cyclic%20search%20long-run%20winners.html)
· [Editable agent notes](../../../outputs/restart-molecule-evaluation-20261111a/results/Cyclic%20search%20long-run%20winners.md)
· [Complete metrics](../../../outputs/restart-molecule-evaluation-20261111a/results/Cyclic%20search%20long-run%20winners.json)
· [Candidate provenance](../../../outputs/restart-molecule-evaluation-20261111a/candidates.json)
· [Interactive execution receipt](../../../outputs/restart-molecule-evaluation-20261111a/execution.json)

The downloaded archive hash was verified, and all three canonical identities
and provenance labels matched the selected input. Browser checks passed
three molecule depictions, source-job labels, filtering and no JavaScript
errors; the rendered layout was visually inspected. This was a data-only
use of the evaluator and changed no library code or running search.

## Pretrained synthesis-gradient candidates, 11 November

Read-only primary-source research found pretrained scalar graph scorers,
so training another predictor is not the first requirement. No weights were
loaded and no inference or integration claim was tested in this audit.

| Candidate | Verified public weights | Relaxed-input obstacle |
| --- | --- | --- |
| [SynFrag](https://github.com/simmzx/SynFrag) | `checkpoints/synfrag_default.pth`, 12,760,155 bytes; MIT code | Best first candidate: AttentiveFP consumes float node and edge features, but RDKit/DeepChem featurization and DGL edge existence are discrete. Needs soft features and message/attention gating at all layers. |
| [DFRscore](https://github.com/Hwoo-Kim/DFRscore) | `save/PubChem/DFRscore/Best_model_163.pt`, 1,936,456 bytes; MIT | Dense PyTorch model, but attention uses `A == 1`; fractional edges disappear. Needs soft attention and a defensible extension of its 49 atom/ring features. |
| [FSscore](https://github.com/schwallergroup/fsscore) | Graph GGLGGL checkpoint, 1,423,125 bytes; MIT code / CC BY 4.0 Figshare release | Discrete graph and line-graph topology; forward ignores explicit edge attributes. Larger adapter than SynFrag. Only GGLGGL weights were verified in the public model archive. |
| [RetroGNN](https://github.com/pchliu/RetroGNN) | Four legacy Chemprop checkpoints, approximately 6–36 MB; CC BY-NC 4.0 | Hard feature/topology construction and older runtime; released examples use a restricted building-block search space. |

SynFrag's output is an ease score (`1 - pred`), not a calibrated route
success probability. DFRscore and SynFrag both permit floating feature
inputs inside their networks, but a feature gradient alone does not supply
gradients for adding/removing graph edges. Frozen-weight expected features
and soft edge gating are plausible extensions, without predictor training.
Check hard-input agreement, finite connected atom/bond gradients, and whether
short optimization improves independently evaluated discrete molecules.

SCScore/RAscore use discrete hashed fingerprints. SYNC releases 3D
atom/coordinate guidance, without the bond-probability path our optimizer
needs. MOOD accepts continuous atom/adjacency tensors, but its released
scalar predictors combine other-target docking, QED and SA; extracting a
pure SA score from them is unsupported. SynFormer/ChemProjector generate
routes or analogs and provide no ready-made scalar synthesis loss.

Key audited interfaces: [SynFrag predictor](https://github.com/simmzx/SynFrag/blob/2c6cacec198c2746cade22643330d38cc19c62a4/model/synfrag.py),
[DFRscore hard mask](https://github.com/Hwoo-Kim/DFRscore/blob/main/scripts/modelScripts/layers.py),
[FSscore GNN](https://github.com/schwallergroup/fsscore/blob/a36390241f59bd48c2b7c2423efb0ca99f4e44af/src/fsscore/models/gnn.py),
[MOOD released objectives](https://github.com/SeulLee05/MOOD),
[SYNC guidance](https://github.com/XYxiyang/SYNC/blob/main/targetdiff/synsbdd/models/syn3d_lin.py).

## Rank-20 archive audit and proposed six-generation baseline

On 11 November, saved fresh-scored candidates were pooled through each
checkpoint and deduplicated by canonical isomeric SMILES. The existing
greedy output selector was used at pairwise Morgan Tanimoto caps 1.0,
0.8 and 0.6. This is detached CPU analysis of existing results, not new
optimization or native scoring. Original contact losses are retained.

| Checkpoint | Updates | Unique valid scored | Best | 20th distinct | 20th at similarity ≤0.60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Near-valid, generation 6 | 36,000 | 380 | 2.06081 | 2.13421 | 2.15057 |
| Deep, generation 6 | 36,000 | 372 | 2.10786 | 2.17615 | 2.19262 |
| Current 3×4 allocation A, generation 3 | 18,000 | 212 | 2.12226 | 2.25057 | 2.29381 |
| Current 3×4 allocation A, generation 4 | 24,000 | 283 | 2.12226 | 2.21483 | 2.22618 |
| Current 3×4 allocation B, generation 3 | 18,000 | 211 | 2.11417 | 2.19341 | 2.20267 |

All listed diversity selections filled 20 slots. The long near-valid/deep
checkpoints used 4×3 branching; the current allocation uses the proposed
3×4. Both have 12 windows per generation. Six generations at 500 updates
per window total 36,000 updates, with at most 432 fresh attempts per seed.
Generation one uses 12 independent random starts; later generations expand
three diverse parents into four children each. Three expansion parents
do not limit the final archive to three molecules.

Extra generations can improve the panel even when the best score plateaus:
allocation A's generation-four distinct rank-20 loss improved by 0.03575
without changing its best. However, median SA among the long six-generation
distinct top-20 panels was 7.86 near-valid and 7.19 deep; structural validity
and favorable contact loss do not establish practical synthesis. These
are BRD4 observations, not guaranteed cross-protein yields. A final panel
should pool all saved generations (and seeds when available), apply quality
and diversity criteria, and report the number passing rather than padding
to a requested count. Fresh rescoring, if added, needs an explicit budget.

[Full rank-20 selections, scores and SA summaries](../../../outputs/restart-top20-audit-20261111/rank20-analysis.json)
· [Current allocation candidate snapshot](../../../outputs/restart-top20-audit-20261111/allocation-candidates.json).
