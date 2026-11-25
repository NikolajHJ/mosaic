# How sharp outputs affect molecular design at update 600116

On 23 November 2026 we froze the seven-round epoch-48 VAE at its segment-2 checkpoint. The question was whether an optimiser can leave an encoded molecule, and what is gained or lost by increasing the final softmax temperature. We measured gradients, edits to the discrete molecule and validity on separate panels. A fourth test uses the Protenix contact score in the BRD4 pocket.

At an encoded mean, the atom logits still responded to the latent vector. For an identical linear contrast on logits versus output probabilities, the respective median latent gradient norms were 2.38 versus 0.000017 at temperature 1.0. The same test gave 5.68 versus 0.480 on Gaussian starts. Warming to 1.5 multiplied the atom and present-bond probability gradient norms by median paired factors of 27.9 and 38.5 on encoded means, versus 0.85 and 0.90 on Gaussian starts. These are sensitivity tests on 24 starts per family, independent of any protein objective. [Gradient sweep](landscape.md#locating-the-output-softmax-bottleneck-at-update-600116)

![Effect of temperature on occupied atom gradients and the validity of sampled graphs](../../../outputs/runs/vae-e48-600116-sharp-sampling-20261123a/temperature-effects.png)

## Sampling from one point

The production hard decoder picks each winning conditional label and then masks absent atoms and their bonds. For a fixed vector, dividing all logits in a row by a common positive temperature preserves that winner. During search, temperature can still change the gradient direction and magnitude and thereby move the *next* vector across a different boundary. The final temperature also leaves the earlier message-passing softmax operations inside the decoder intact. [Hard and relaxed decoding](../../../src/latentmol/representation.py) · [Decoder refinement](../../../src/latentmol/models/refined_decoder.py) · [Temperature scaling in classification](https://proceedings.mlr.press/v70/guo17a.html)

We drew 96 held-out SMILES without replacement from the saved 512-row PubChem validation panel and encoded each to its posterior mean. These inputs mostly had 11–30 heavy atoms (95/96). Separately, we drew 96 unfiltered standard Gaussian vectors. At each temperature we retained eight independent categorical graph draws per center using the production variable-occupancy sampler. We reset its RNG to reuse randomness across arms. Chemical validity requires a single connected, sanitized graph without radicals. Canonical chemical identity permits equivalent atom orderings. All invalid draws count in the denominator. [Sampling implementation](../../../src/latentmol/sampling.py) · [Inspection](../../../src/latentmol/chemistry.py)

| Output temperature | Encoded valid / 768 | Encoded exact source / 768 | Gaussian valid / 768 |
|---|---:|---:|---:|
| 0.4 | 710 | 690 | 621 |
| 0.7 | 705 | 690 | 576 |
| 1.0 | 700 | 686 | 521 |
| 1.5 | 647 | 622 | 354 |
| 2.0 | 451 | 379 | 160 |

The ordinary raw argmax yielded 89/96 valid molecules in each cohort and 86/96 exact source identities in the encoded cohort. Changing a common positive temperature after decoding preserves every categorical winner at a fixed latent. Sampling does change. Between 1.0 and 1.5, Gaussian validity fell 21.7 percentage points (approximate 95% paired bootstrap interval −25.8 to −17.7, resampling 96 centers). Encoded validity fell 6.9 points (−9.0 to −4.9). Mean negative log probability of the source’s exact labeled graph in canonical order, in nats per molecule, was 1.33, 0.88, 0.75, 0.82 and 1.42 in ascending temperature order. Temperature 1.0 gave the best mean among these five choices on this panel.

We also warmed just the distribution over elements given an occupied slot or bond order given an existing bond. The probability of an atom or bond existing and the charge distribution retained their temperature-1 values. Warming only the element conditional to 1.5 yielded 700 valid encoded draws, matching the 700 at baseline. Distinct valid alternatives appeared at 7 encoded centers versus 4 at baseline. On Gaussian starts, validity was 478/768 with element-only warming, 496/768 with bond-order-only warming and 455/768 with both. Global warming gave 354/768. These are draws with fixed latent vectors. They do not evaluate a protein gradient or establish a search benefit.

From the repository root, set `PYTHONPATH` to include `src` and the checkout. Run `temperature_sampling_audit.py --checkpoint CHECKPOINT --panel PANEL --output NEW_DIRECTORY`. To reproduce the conditional comparison add `--conditional-types --temperatures 1.0 1.5`. Both runs use seed 2026112351. Their repeated baseline and global 1.5 draws agree. [Script](temperature_sampling_audit.py) · [Full sampling results](../../../outputs/runs/vae-e48-600116-sharp-sampling-20261123a/results.json) · [Conditional results](../../../outputs/runs/vae-e48-600116-selective-sampling-20261123a/results.json)

## Same molecule at two latent points

We drew 256 new Gaussian vectors at checkpoint 600116. Of 225 losslessly canonicalizable hard graphs, 34 reconstructed to the same canonical molecule at their re-encoded posterior mean. We took the first 32 such pairs in seed order and used the same four random unit directions on both representations. All 32 parents were distinct. Each cell below counts different, valid raw argmax molecules among 128 displaced decodes. The center itself was always the same molecular identity on both sides.

| Latent L2 displacement | Original Gaussian point | Re-encoded mean |
|---:|---:|---:|
| 0.1 | 17/128 | 0/128 |
| 0.5 | 67/128 | 8/128 |
| 1.0 | 90/128 | 11/128 |

At radius 1.0, 31 Gaussian neighbors retained the parent and 7 were invalid. For the re-encoded points, 116 retained the parent and 1 was invalid. At least one of the four directions found a different valid molecule for 30/32 original points and 5/32 re-encoded means. The median log probability of the complete modal slot assignment at temperature 1 was −3.00 at the original point and −0.216 at the re-encoded point. Their median latent norms also differed (15.77 and 8.03 respectively). Conditioning on a successful round trip restricts this result to the eligible pairs. These are geometric probes with random directions, independent of any optimization objective. Retain the original latent when optimizing a generated molecule where possible.

Reproduce with `matched_mode_sharpness.py --checkpoint CHECKPOINT --output NEW_DIRECTORY` using seed 2026112352. [Probe](matched_mode_sharpness.py) · [Results and raw neighbor identities](../../../outputs/runs/vae-e48-600116-matched-mode-sharpness-20261123a/results.json)

## Optimisation

In the local sensitivity panel, the most responsive existing bond accounted for a median **86.8%** of the squared softmax sensitivity at an encoded mean at temperature 1.0. Near such points, an optimizer can see a narrow set of promising changes even when the molecular score would reward changes elsewhere. Across the 24 encoded means, warming to 1.5 increased the local softmax sensitivity on 23/24 occupied-atom panels and 23/24 existing-bond panels. At the Gaussian centers the corresponding counts were 4/24 and 2/24. These measure a property of the decoder’s categorical distributions. A protein scorer supplies its own gradient over those distributions. [Mechanism, definitions and measurements](landscape.md#locating-the-output-softmax-bottleneck-at-update-600116)

On a separate panel, we asked for one additional nitrogen or oxygen atom at fixed heavy-atom count. Three wider proposals at temperature 1.0 produced four selected hard target hits among 23 valid encoded starts, compared with zero for three small proposals. The initial relaxed score fell with small steps at every one of the 23 starts. The corresponding Gaussian starts yielded 12/23 hard hits on either step ladder. Warming to 1.5 or 2.0 yielded three encoded and ten Gaussian hits on the wide ladder. For the binding question, job 30276244 compares the normalized Protenix and repair direction at temperatures 1.0 and 1.5, with the same step lengths and 31 search calls per start. Fresh native scores for the decoded edits are needed to rank the temperature settings for this checkpoint. [Descriptor test, including random and warm-backward controls](editability.md#can-a-warmer-decoder-reach-a-specified-hard-edit-at-update-600116) · [Native study and status](editability.md#does-warming-change-the-useful-protenix-gradient-direction-at-update-600116)

## Recommendation for the next comparison

Use the saved original latent for generated parents. For encoded starts, begin at temperature 1.0 and search farther only while the hard identity is unchanged. Prefer the smallest successful step and rescore distinct valid molecules under a common native scoring procedure. Use a smaller initial radius for Gaussian starts. Raising temperature can be useful as a proposal when an encoded gradient is numerically flat, with acceptance checked under the original objective. The warmed-gradient control gave little advantage on the composition test. Monitor the current encoder round-trip training with both reconstruction and the paired, same-molecule displacement test. A higher round-trip success rate and accessible valid edits are separate goals.
