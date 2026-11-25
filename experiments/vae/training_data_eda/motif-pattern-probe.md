# Joint aromatic pattern loss on the frozen epoch-48 decoder

The previous bond-order pilots sometimes raised the desired class on unrelated edges. This candidate combines a small set of verified positive targets with an expected count of off-site aromatic bonds and a probability of partial ring labeling. A CPU replay at update 588498 measured which terms activate and where their gradients point.

For each batch of Gaussian vectors, the existing sampled motif readout gives the number of hits missing from each configured floor. On the argmax skeletons the detector can propose at most one valid six-membered aromatic completion and one amide per molecule. The candidate ranks the proposals using the mean conditional bond-type cross-entropy over the edges in each site. It chooses the lowest-loss sites up to the missing counts. The ranking is detached. Each selected site's live mean cross-entropy is divided by the full batch size before summing.

Three additional terms use ordinary softmax probabilities including NO_BOND. Graphs must sanitize, be connected and radical-free, and produce readable SMILES to participate.

* The expected count of off-site AROMATIC labels sums their probabilities once per occupied pair. Existing RDKit-perceived aromatic bonds and all verified candidate rings are protected, including candidates skipped once the quota is full. Perceived aromatic bonds are protected even if the decoder wrote alternating single and double labels.
* For each protected ring, the probability of some but not all edges being labeled AROMATIC is $1-\prod_e p_e-\prod_e(1-p_e)$ under factorized sampling with the occupied skeleton fixed. The mean over rings gives one value per graph. A fully Kekulé-labeled ring with no explicit AROMATIC labels incurs no partial-label penalty.
* For an existing neutral C–O single bond, a DOUBLE probability contributes if that bond upgrade alone would overfill the carbon or oxygen. Implicit hydrogen replacement is allowed where chemically available.

The objective is $0.5(C_{\mathrm{ring}}+0.10C_{\mathrm{amide}})+0.5E_{\mathrm{offsite}}+0.25P_{\mathrm{partial}}+0.10E_{\mathrm{blocked\ CO}}$. Every term uses the full batch denominator. The calling experiment sets the overall multiplier and update cadence.

| Two batches of 48, same frozen checkpoint | Batch 1 | Batch 2 |
|---|---:|---:|
| Strictly valid argmax skeletons | 41 | 45 |
| Fresh sampled aromatic / amide hits | 0 / 3 | 1 / 1 |
| Ring sites selected / eligible | 4 / 10 | 3 / 18 |
| Amide sites selected / eligible | 3 / 5 | 5 / 11 |
| Protected rings | 12 | 19 |
| Off-site aromatic expected count per draw | 0.121 | 0.057 |
| Mean partial-label probability per draw | 0.021 | 0.044 |
| Impossible C–O DOUBLE expected count per draw | 0.0052 | 0.0012 |
| Full candidate loss | 0.395 | 0.163 |

The off-site term had a positive derivative for a uniform increase in the AROMATIC logit in both batches (+0.0427, +0.0219). Other terms nearly canceled this on the full loss (+0.00157, −0.00386). Every gradient was finite and stayed within the formulas' known support. Site checks took roughly 0.20 s per batch and the CPU logit forward/backward another 0.06 s. These times exclude model evaluation and data transfer.

**Remaining issue.** We measured each edge's directional response along a normalized descent step of the final linear readout with features held fixed. AROMATIC-minus-SINGLE still rose on all 2,009 retained off-site single bonds, and DOUBLE-minus-SINGLE rose on all 222 chemically blocked C–O bonds. The desired localized effect on the explicit expected penalty does not establish that a shared decoder weight update improves raw generation. A matched run and fresh sampled outcomes are needed.

Set `PYTHONPATH=src` and choose a new output path to reproduce the frozen probe.

```sh
python experiments/vae/training_data_eda/motif_pattern_probe.py \
  --checkpoint outputs/runs/vae-e48-r7-complete-20261122a/checkpoint-000588498.pt \
  --out outputs/runs/vae-e48-motif-pattern-20261122a
python -m pytest tests/guidance/test_motif_patterns.py
```

[Loss implementation](../../../src/latentmol/guidance/motif_patterns.py) · [Per-batch results](../../../outputs/runs/vae-e48-motif-pattern-20261122a/summary.json)
