# Does context ranking confine bond-order gradients to the right sites?

This CPU replay uses 96 fresh Gaussian inputs in two batches of 48 at epoch-48 update 588498 (seed 2026112215). It compares two objectives on exactly the same verified near misses. Conditional cross-entropy rewards the desired bond class at a positive site. The alternative ranks the target-minus-single logit gap at that site above the same gap at a chemically incompatible negative site, with a one-logit margin. All output rows, including failed target searches, stay in the input population. The production recipe does not call the ranking module.

**Site selection.** For an aromatic target, the entire six-membered ring must pass the existing chemical completion check. Every ring edge receives a control matched on element pair (C–C or C–N), chosen from actual neutral single bonds outside rings. For an amide, the existing alcohol-adjacent-to-nitrogen proposal must produce a strictly valid amide. Its control is a neutral C–O single bond where raising bond order alone would overfill the carbon or oxygen. We prefer within-molecule controls, then use a matching control from the same batch. Every selected edge is read symmetrically. The NO_BOND logit receives no direct derivative from either objective.

| Across 96 Gaussian decodes | Aromatic | Amide |
|---|---:|---:|
| Eligible sites with matched negatives | 28 / 28 rings | 16 / 16 sites |
| Within-molecule pairs | 90 / 168 edges | 11 / 16 edges |
| Negative edges available | 219 | 222 |
| Median target-minus-single logit at the positive edge | −8.96 | −4.31 |
| Median target-minus-single logit at the matched negative | −16.79 | −16.57 |
| Median paired positive-minus-negative gap | +7.17 | +9.27 |

The relative ordering is already partly learned, even though many positive sites remain far from a hard bond-type change.

**Direct vs shared effects.** In the pure ranking objective, the same additive category bias contributes equally to each positive and negative gap and cancels. A regression test checks that cancellation. Only participating positives and controls receive nonzero gradients with respect to the raw bond logits. The seven-round decoder reuses the same bond head during three internal refinement rounds, so changing an actual head parameter can also affect the features entering the final layer. Replaying all four uses, the average maximum absolute shared-bias gradient fell from 0.503 to 0.00112 for aromatics and from 0.438 to 0.000381 for amides when replacing CE with ranking.

For a separate readout of feature sharing, we computed the induced change in each edge's gap along a unit gradient-descent direction of the **final linear readout with its features held fixed**. Values below describe one local direction. The end-to-end AdamW training update and novel hard molecules were outside this test.

| Median gap change or fraction increasing | Aromatic CE | Aromatic ranking | Amide CE | Amide ranking |
|---|---:|---:|---:|---:|
| Selected positive edges | +14.80 | +8.02 | +15.44 | +0.56 |
| Selected negative edges | +14.57 | +7.44 | +11.80 | −3.68 |
| Other single bonds of matched element types | +14.44 | +7.85 | +12.93 | −2.67 |
| Matched off-site edges increasing | 100% | 99.9% | 100% | 17.8% |

At the same scalar loss coefficient, the median final-readout gradient norm fell from 7.43 to 0.257 for aromatic sites and from 7.18 to 0.084 for amide sites. Thus the aromatic intervention supplied much less overall pressure, although the direction it supplied remained broad. Target and negative selection took 0.39 seconds for 96 graphs on this CPU, excluding decoding and backpropagation.

**Implication.** Removing direct class-bias pressure was insufficient for aromatic ranking. Its gradient direction still raised the aromatic score at almost every other C–C and C–N single bond. Amide ranking separated eligible from incompatible C–O positions, although the median absolute positive gap was already well ordered relative to its negative. It is a potential bounded amide-only training arm. No training or generation benefit was established in this replay. Mixing in absolute CE is configurable for future experiments and would reintroduce direct class pressure.

### Reproduce

Set `PYTHONPATH=src` and choose a fresh output directory.

```sh
python experiments/vae/training_data_eda/motif_contrast_probe.py \
  --checkpoint outputs/runs/vae-e48-r7-complete-20261122a/checkpoint-000588498.pt \
  --out outputs/runs/vae-e48-motif-contrast-20261122c
```

[Standalone ranking loss](../../../src/latentmol/guidance/motif_contrast.py) · [Probe results](../../../outputs/runs/vae-e48-motif-contrast-20261122c/summary.json) · [Absolute bond-completion objective](epoch48-motif-completion.md)
