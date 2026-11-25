# Aromatic rings and functional groups in the large VAE

The [epoch-48 mechanism investigation](epoch48-aromatic-mechanism.md) tests where the deficit arises using a frozen seven-round decoder, controlled bond and latent changes, the teacher gradient and eight starting distributions.

**The late epoch-47 checkpoint reconstructs aromatic inputs well and almost exclusively generates carbon–fluorine graphs from standard-normal vectors.** Among 512 held-out PubChem molecules, 460 contain an aromatic ring. The late checkpoint recovered a valid aromatic molecule for 440 of those 460 (95.7%). It reconstructed 480/512 input molecules exactly. From 2,048 fresh Gaussian vectors it generated 55 valid molecules and zero valid aromatic molecules. Sampling the categorical labels at temperature one on those vectors gave 21 valid outputs, again with zero aromatic molecules.

| Feature present | Real training panel (50,000) | Valid epoch-46 prior draws (1,559) | Valid late-47 prior draws (55) | Valid late-47 reconstructions (488) |
| --- | ---: | ---: | ---: | ---: |
| Any ring | 95.9% | 94.4% | 58.2% | 98.8% |
| Aromatic ring | 87.1% | 2.2% | 0% | 90.2% |
| Heterocycle | 74.3% | 87.1% | 0% | 74.0% |
| Amide-type carbonyl | 48.0% | 6.7% | 0% | 44.1% |
| Amine | 44.4% | 90.3% | 0% | 48.6% |
| Ether | 39.0% | 66.3% | 0% | 38.7% |
| Carbon-bound halogen | 40.1% | 22.9% | 100% | 41.2% |

At epoch 46, random argmax produced 1,559 valid molecules, including 1,471 with rings and 35 with aromatic rings. Size-matching the training population to the valid epoch-46 outputs gives an expected aromatic prevalence of **92.5%**. The same comparison for late epoch 47 gives **88.9%**. An independent archived 256-vector panel also showed this aromatic deficit before epoch 47. Five of 191 valid outputs had an aromatic ring at epoch 45, and one of 190 did at epoch 46.

The late checkpoint produced only carbon and fluorine on **2,018 of all 2,048 argmax draws**, including invalid outputs. The 55 valid structures were all distinct, all occupied 50 slots, and each contained 22–28 fluorines. The 50,000 real graphs contained 74 carbon–fluorine-only structures, and just one real molecule contained at least 22 fluorines. The concentration is visible before filtering for validity. [Element counts](../../../outputs/runs/vae-generation-motifs-20261120a/composition.json) · [First two vectors yielding valid outputs at both endpoints](../../../outputs/runs/vae-generation-motifs-20261120a/matched-prior-examples.png)

At the earlier epoch-47 checkpoint (27,811,440 training rows), 1,984/2,048 random outputs already contained only carbon and fluorine. Sixty passed validity, including two aromatic molecules. These snapshots place the onset of the strong composition shift between the end of epoch 46 and that earlier checkpoint. The aromatic deficit itself predates epoch 47.

There is also evidence upstream of sanitization. Out of 2,048 prior draws, 157 at epoch 46 had at least one explicitly aromatic bond, and 48 had a cycle of such bonds. At late epoch 47 those counts fell to 11 and 6. The decoder can still recover aromatic chemistry in the regions reached by encoding real graphs. These endpoint observations implicate the prior-to-decoder mapping. They leave open which part of the training objective drove the change.

One posterior draw per real molecule at the late checkpoint produced 461/512 valid outputs, including 424 exact copies. Of the 37 valid outputs that differed from their own input, 30 contained an aromatic ring and all 30 came from aromatic inputs. This supports scaffold-seeded generation as a near-term option for exploring local changes. This experiment checked identity against the source molecule only.

On 21 November we checked whether the 30 aromatic edits retained the source scaffold. On the saved `posterior_sample` records from update 524177, we compared RDKit `MurckoScaffoldSmiles(mol=...)` for the source and decoded molecule after excluding invalid outputs and exact source copies. Eight of the 30 aromatic edits retained the same scaffold string and 22 changed it. Across all 37 valid edits, nine retained the scaffold and 28 changed it. A changed scaffold can reflect a single bond or heteroatom edit. Identity against the complete training set and binding performance were not evaluated.

![Motif frequency on the valid outputs in each population](../../../outputs/runs/vae-generation-motifs-20261120a/motif-comparison.png)

## Question and measurements

Does random generation reproduce the chemistry learned during reconstruction? We compared the completed epoch-46 VAE with two saved points in epoch 47. The late checkpoint is update 524177, after 47,682,960 of the 60M training graphs. The subsequent occupancy-guided dropout leg had zero GPU runtime when the audit began.

The reference contains 50,000 uniformly sampled rows from the 60M training view. Epoch 46 itself trained on the earlier combined MOSES–ChEMBL set. For reconstruction we used the same 512 held-out PubChem graphs at every checkpoint. This additional validation panel covers 15–31 heavy atoms. We decoded both the posterior mean and one sample from each input's posterior. For random generation we decoded 2,048 shared standard-normal vectors per checkpoint in eval mode. The primary readout is raw argmax. A separate readout samples all atom, charge and bond labels once per vector at temperature one. Both readouts use the unmodified output. [Corpus and reference sampling](experiment.md) · [Evaluation script](generation_motifs.py)

An output counts as valid when the project's hard-graph inspection yields a single connected, radical-free, RDKit-sanitized molecule. A motif counts once per valid molecule, even when it appears several times. The 20 functional-group families can overlap. Aromatic rings are identified by RDKit after sanitization, including aromaticity perceived from alternating bonds. The raw graph diagnostics separately count explicit aromatic-class bonds and cycles made entirely from those bonds before sanitization. For an invalid graph these are topological observations only. [Molecular definitions](chemistry_profile.py) · [Hard graph inspection](../../../src/latentmol/chemistry.py)

The frequency export gives the number of valid outputs with each motif, the percentage among valid outputs, and the percentage among all attempts. Size matching reweights the 50,000 training rows to the exact heavy-atom count distribution of each panel's valid outputs. It controls for molecular size only. The raw compressed records retain each output's atom, charge and bond labels, SMILES or first failure, and source SMILES for reconstructions. [Full frequency table](../../../outputs/runs/vae-generation-motifs-20261120a/frequencies.csv) · [Per-checkpoint counts](../../../outputs/runs/vae-generation-motifs-20261120a/summary.json)

## Decoder representation

The decoder predicts a categorical element and charge for each slot. Its pair states exchange information with the atoms and other pairs over four triangle/attention rounds. The final bond classifier chooses among NONE, SINGLE, DOUBLE, TRIPLE and AROMATIC on each occupied unordered pair. A recognizable group emerges when the local decisions agree on a subgraph. Forming an entire aromatic ring or amide therefore requires coordinated atom and bond predictions. In this representation benzene, pyridine, furan, thiophene and acetamide round-trip. The tested neutral pyrrole and imidazole examples fail the existing round-trip check because their pyrrolic hydrogen is not explicitly encoded. This limitation also applies to the eligible training population. [Decoder](../../../src/latentmol/models/refined_decoder.py) · [Molecular schema](../../../src/latentmol/representation.py)

## Inference-only aromatic readout probe

On the first 512 shared prior vectors, we added 0, 1, 2 or 4 to the final aromatic bond logit before hardening. Atom choices, charges, hidden states and weights were identical in each paired comparison. At epoch 46 a +4 shift increased valid aromatic products from 5 to 32, but valid products overall fell from 399 to 343. At late epoch 47 the same shift changed explicit aromatic-edge proposals from 2 to 5 graphs, while valid aromatic products remained 0 and overall validity remained 14/512. This probes a uniform output bias. Attributing the shift to a particular training term requires a training comparison. [Probe script](aromatic_readout_probe.py) · [Paired outcomes](../../../outputs/runs/vae-generation-motifs-20261120a/aromatic-logit-shift.json)

## Chemistry after the epoch-47 restart

**21 November, after the occupancy-guided training resumed.** We compared update 541677 with the two earlier endpoints on the first 512 of the original 2,048 Gaussian vectors (same seed 2026112047, CPU eval, batch 12, raw argmax). We used `inspect_batch` and `motifs` from the evaluation script and kept the earlier records for the same indices. At updates 325498, 524177 and 541677, respectively, valid outputs numbered 399, 14 and 269. Valid aromatic outputs numbered 5, 0 and 8, and valid amides numbered 34, 0 and 19. Carbon–fluorine-only outputs among all draws numbered 0, 506 and 0. This measures a partial epoch and one matched panel. [Counts and checkpoint identity](../../../outputs/runs/vae-generation-motifs-20261121a/matched-motifs.json) · [Raw new outputs](../../../outputs/runs/vae-generation-motifs-20261121a/prior-argmax-512.jsonl.gz)

## Reproduction

Use the project environment with Torch and RDKit, with `PYTHONPATH=src`. The CPU experiment used seed 2026112047, batch size 12 and five Torch threads. The checkpoint directory below contains the three cached checkpoint files. Use an unused output directory.

```sh
python experiments/vae/training_data_eda/generation_motifs.py \
  --checkpoint "$CHECKPOINTS/checkpoint-000325498.pt" \
  --checkpoint "$CHECKPOINTS/checkpoint-000524177.pt" \
  --checkpoint "$CHECKPOINTS/checkpoint-000441379.pt" \
  --panel outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json \
  --profiles outputs/runs/vae-training-data-eda-20261120a/analysis/profiles.csv.gz \
  --output "$RESULTS" --prior 2048 --threads 5
python experiments/vae/training_data_eda/aromatic_readout_probe.py \
  --checkpoint "$CHECKPOINTS/checkpoint-000325498.pt" \
  --checkpoint "$CHECKPOINTS/checkpoint-000524177.pt" \
  --output "$RESULTS/aromatic-logit-shift.json" --count 512
python experiments/vae/training_data_eda/plot_generation_motifs.py "$RESULTS/summary.json"
```
