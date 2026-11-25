# How often does the motif gradient disappear in a 48-draw batch?

We used the 8,192 previously saved raw categorical draws from the seven-round control after its matched 128-update pilot. The first 8,160 form 170 consecutive groups of 48 and 34 groups of 240. Both groupings use identical graphs, including all invalid draws. The corpus reference comprises 50,000 training rows with 43,528 aromatic and 23,994 amide molecules. [Panel source and endpoint comparison](epoch48-motif-completion.md)

| Motif | Total positive draws | Zero-hit batches of 48 | Zero-hit batches of 240 | Batch prevalence standard deviation, 48 vs. 240 |
|---|---:|---:|---:|---:|
| Aromatic ring | 134 | 76/170 (44.7%) | 0/34 | 1.83 vs. 1.02 percentage points |
| Amide | 309 | 27/170 (15.9%) | 0/34 | 2.73 vs. 1.35 percentage points |

At 240 draws per scored batch, raising the floors from 8% and 12% to 87.056% and 47.988% increased the mean sum of absolute score coefficients by 14.7-fold for aromatics and 5.6-fold for amides. These numbers describe the coefficients multiplying per-molecule gradients. A decoder parameter's actual update also depends on the individual gradients, the other training terms and the optimizer. Positive counts remove one cause of a zero gradient. They do not establish that a training step improves either motif.

The script also re-evaluates a deterministic prefix of the frozen endpoint on CPU and differentiates each saved graph's log probability with respect to shared atom-, charge- and bond-category logit shifts. Invalid draws retain their likelihood score. It checks the decoded argmax against the stored GPU result as a numerical replay control. Every group uses the same graphs and the same score vectors for all four policy combinations. [Script](pooled_motif_gradient_probe.py) · [Label-panel counts and coefficients](../../../outputs/runs/vae-pooled-motif-gradients-20261122c/label-panel.json) · [Directional derivatives](../../../outputs/runs/vae-pooled-motif-gradients-20261122c/summary.json)

The 480-draw gradient prefix contained seven valid aromatics and 15 amides. Every replayed argmax graph agreed with its original GPU counterpart. Aromatic gradients were exactly zero on five of ten 48-draw groups, and amide gradients were zero on three. On both 240-draw groups each motif had a finite nonzero gradient. At fixed batch size 240, raising the floors increased the norm of the **mean gradient in the 20 measured category directions** from 0.000368 to 0.004677 (12.7-fold) for aromatics and from 0.000563 to 0.002758 (4.9-fold) for amides, before the outer loss weight. The CPU score replay took 172 seconds. This tests the likelihood estimator on one trained endpoint. Two pooled observations cannot establish variance across training steps or the full-model update. The larger 8,160-label panel supplies the hit-rate comparison.

```sh
base=outputs/runs/vae-aromatic-mechanism-20261121a
python -m experiments.vae.training_data_eda.pooled_motif_gradient_probe \
    --endpoint "$base/completion-20261122b/control/endpoint.pt" \
    --panel "$base/pattern-fresh8192-20261122a" --draws 480 \
    --training-frequencies outputs/runs/vae-generation-motifs-20261120a/frequencies.csv \
    --output "$OUT"
```
