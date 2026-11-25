# Compact graph VAE on the 32 GB V100

The timing question is whether logical batches of 256 or 512 are faster on the same sparse 60M training split. The model starts from scratch. The benchmark runs complete optimizer updates and discards the temporary weights.

DTU accepted job **30099959** at 21:20 CET on 17 November 2026. One job runs all four conditions. At submission it was eligible and waiting in `gpuv100`. [Submission receipt](../../outputs/hpc-submissions/vae435k-benchmark-20261117a/submitted.json)

The combined local regression run passed all 54 tests across the compact benchmark, prior guidance, graph matching, evaluation, input ordering and checkpoint comparison. The full-width model also completed a CPU optimizer update with finite gradients.

### V100 results, 19 November 2026

Job **30099959** completed all four conditions. Each timed arm used the same 4,096 real molecules. Both batches fit natively on the 32 GB V100.

| Real batch | Training objective | Real molecules/s | Peak allocated GiB | Projected hours per 60M real molecules |
| ---: | --- | ---: | ---: | ---: |
| 256 | Reconstruction + KL | 242.84 | 13.31 | 68.63 |
| 512 | Reconstruction + KL | 258.24 | 26.56 | 64.54 |
| 256 | With 15:1 prior chemistry guidance | 90.96 | 13.31 | 183.24 |
| 512 | With 15:1 prior chemistry guidance | 74.41 | 26.56 | 223.97 |

Batch 512 was 6.3% faster for real-only throughput. With guidance, batch 256 processed 22.2% more real molecules per second than batch 512. Each condition has a single timed panel on an early model. [Scheduler receipt and raw measurements](../../outputs/hpc-submissions/vae435k-benchmark-20261117a/status-20261118T230752882044Z.json)

## First complete 60M training pass

**Current production run, 19 November 2026.** V100 worker **30113896** uses the corrected recipe from its first optimizer update. Callback **30113899** continues the same run across allocations. DTU history confirmed that the superseded worker and callback ended with zero running seconds and no checkpoint. At 12:36 CET the corrected worker was eligible and pending a physical GPU. [Replacement receipt](../../outputs/hpc-submissions/vae435k-count-v100-20261119a/submitted.json) · [Scheduler check](../../outputs/hpc-submissions/vae435k-count-v100-20261119a/status-20261119T113553Z.json)

DTU accepted V100 worker **30113803** and its CPU continuation callback **30113804** on 19 November 2026. The first scheduler check showed the worker pending and the callback waiting for its success. The callback creates another allocation only if the pass is still incomplete. [Submission receipt](../../outputs/hpc-submissions/vae435k-real-v100-20261119a/submitted.json) · [Scheduler state](../../outputs/hpc-submissions/vae435k-real-v100-20261119a/status-20261119T112948Z.json)

The compact model trains from scratch on each of the 60,000,000 training rows once in shuffled order, for 234,375 native batches of 256. The real-molecule objective is graph-matched reconstruction of elements, charges and bonds, plus the Gaussian posterior KL. Its coefficient rises linearly to 0.001 over the first two million real molecules. The decoder has no dropout. The AdamW learning rate is 0.001, weight decay is 0.0001, and gradient norm clipping is 5. The run config records the weight-decay override explicitly. In a local float32 AdamW check with zero gradient and a starting parameter of 1, the previous decay of 0.00001 left it at 1; the selected decay took it to 0.99999988 in one update. [Original real-only configuration](compact435k-real-v100.json) · [Model and base loss](../../configs/vae_compact_triangle.yaml)

The [corrected production config](compact435k-count-v100.json) also trains the decoder on one fresh standard-normal latent per 15 real molecules. Carrying the fractional count across updates gives 4,000,000 training priors over the pass, usually 17 per batch and occasionally 18. Each prior batch contributes the training-histogram count KL and the element-vs-NO_ATOM argmax loss described below, each with weight 1. Both prior losses use only atom logits. The production recipe omits the pilot's bond repair and CDF terms. An independent RNG stream supplies training priors and resumes from the saved checkpoint. The decoder determines size at inference.

In a matched local optimizer check using the full model and 256 PubChem training rows, the first batch of 17 prior vectors gave a count KL of 0.76692 and an argmax loss of 0.70513. The total prior contribution was 1.47205. Its isolated gradient norm on the atom head was 1.22788. A second batch with 18 priors also produced a finite, nonzero gradient. Exact CPU checkpoint replay reproduced the model and optimizer state and the subsequent training priors. The production ratio is 15:1; the earlier size-recovery comparison used 4:1 with structural repair. Longer-term coverage and validity under this recipe remain open. [Optimizer check and reproduction](../../outputs/runs/vae435k-count-only-smoke-20261119a/result.json)

At the start and after 10, 100, 1,000 and then every 5,000 updates, the run records reconstruction on a fixed 256 validation graphs and generation on 256 independent fixed Gaussian vectors. Panels also run at segment endpoints. They record graph exactness under atom renumbering, validity and the size distribution for both argmax and categorical prior draws. The saved posterior KL by dimension helps distinguish how much information the encoder uses from whether the decoder fills all 50 slots. [Evaluation](../../src/latentmol/evaluation/compact_training_panel.py)

The original real-only V100 benchmark implies about 69 GPU-hours for one pass, with additional time for evaluations, checkpoints and queue waits between allocations. Full-pass V100 timing with the two occupancy terms is unmeasured. The trainer checkpoints every 1,000 updates and retains selected intermediate weights. Each successful allocation that reaches its time budget hands its checkpoint to the next segment of the same run. [Training loop](../../src/latentmol/training/compact_train.py)

## Model and objective

| Component | Configuration | Trainable parameters |
| --- | --- | ---: |
| Encoder | Four bond-aware message layers, canonical input order, 128-wide pooled readout with two residual updates, Gaussian mean and log variance | 238,360 |
| Decoder | 128-dimensional latent, 50 atom slots, width 24 per atom, 18 per pair, 12 in each triangle update, four attention heads and four independent refinement rounds | 197,040 |
| Total | Same element, formal-charge and bond vocabulary as the larger VAE | 435,400 |

The real-molecule objective uses atom, active-atom charge and unordered-pair bond cross-entropy, with a linear KL ramp to 0.001 over 2M real examples. The compact model uses ordinary training temperature. The control uses only real molecules. The additional arm draws one independent Gaussian latent per 15 real molecules, carrying the fractional count across steps. For these prior draws the structural teacher can propose a bond repair while retaining every occupied atom and formal charge. The decoder receives conditional bond cross-entropy averaged over occupied pairs for each accepted graph, then over all draws (weight 0.375). A second term compares the predicted population atom-count CDF with the training-split CDF using mean squared error (weight 1). It uses up to four prior batches, with past distributions detached. Training includes no interpolation targets. [Training YAML](../../configs/vae_compact_triangle.yaml) · [Benchmark settings](compact435k-v100.json) · [Prior guidance](../../src/latentmol/training/compact_prior.py)

## Preventing 50-slot collapse on unseen latent vectors

In the completed V100 benchmark, 63/273 (batch 256) and 89/273 (batch 512) prior draws filled all 50 slots under raw atom argmax. Those draws were used during training. The 60M training split contains 241,482 molecules with 50 heavy atoms (0.4025%) and has a mean of 25.51. The original count-CDF loss compares probabilities after softmax. Once a slot is very confidently occupied, a substantial logit change can have almost no effect on its probability or the CDF loss gradient.

The [candidate settings](compact435k-occupancy-v100.json) add two terms on fresh standard-normal latent vectors. The original CDF and bond-repair terms, real reconstruction and latent KL retain their weights. Each latent can decode to a different size, including 50. [Loss implementation](../../src/latentmol/training/compact_prior.py) · [Enumeration, saturation and argmax tests](../../tests/training/test_compact_prior.py)

1. **Categorical count distribution.** For each prior vector, compute the log probability of every count from 0 to 50 by combining the odds of a slot being empty or containing any element. Mix these per-vector distributions with equal weights. The term is the forward KL from the empirical training count histogram to the mixture, divided by 50, with weight 1. Computing the recurrence in log space retains a corrective gradient even when the predicted probability of a common size underflows in ordinary arithmetic.
2. **Argmax occupancy.** Compute the margin between the highest real-element logit and NO_ATOM in every slot. Rank the graphs by their sum of sigmoid margins and assign ascending quantiles of the training count distribution to them. Within each graph, apply binary cross-entropy with logits to make the best K margins positive and the rest negative (weight 1, mean over slots and graphs). The within-quantile offset rotates between updates so infrequent sizes remain represented. The graph assignment and targets are detached for that update. This gives a gradient for the hard decision made by generation, including when probability is split between several elements. Inference uses the original decoder with no target K.

The V100 candidate compares real-only, original prior guidance, and original guidance plus both new terms at both batch sizes. Run the command below from the repository root in an allocation, using an unused output directory. All arms evaluate the same 128 unseen Gaussian vectors before and after the short timed window. A second panel scales those same vectors by 1.5 to test extrapolation along identical directions. The report includes the full size histograms from argmax and one categorical sample per vector, expected size, connected validity and unique valid structures. The objective acts on probabilities, so these hard-output measurements matter. These probes sample latent directions independently of a downstream optimizer. [Panel code](../../src/latentmol/evaluation/compact_prior_panel.py)

```sh
python -m latentmol.training.compact_benchmark --config experiments/vae/compact435k-occupancy-v100.json --output "$NEW_OUTPUT"
```

### Matched local occupancy pilot, 19 November 2026

Both arms used the full 435,400-parameter architecture. After eight shared real-only warmup updates, each update used 24 real molecules sampled from a 512-row training panel plus six fresh independent Gaussian latents. A separate fixed set of 64 Gaussian vectors measured generation, and 64 disjoint PubChem validation molecules measured reconstruction. This small CPU experiment used a 4:1 real/prior ratio to exercise the new loss. The V100 comparison specifies 15:1. [Reproduction commands](compact_occupancy_pilot.py) · [16-update measurements](../../outputs/runs/vae435k-prior-occupancy-20261119b/comparison.json)

| Start and endpoint | Original prior loss | With count KL |
| --- | ---: | ---: |
| Ordinary start, initially full under argmax | 22/64 | 22/64 |
| Ordinary start, full under argmax after 16 updates | 2/64 | 0/64 |
| Ordinary start, mean argmax count after 16 updates | 47.03 | 33.98 |
| Ordinary start, empty under argmax after 16 updates | 0/64 | 0/64 |
| Saturated start, full under argmax after 16 updates | 64/64 | 64/64 |
| Saturated start, full under categorical decoding after 16 updates | 43/64 | 30/64 |

The saturation challenge temporarily lowered the learned initial NO_ATOM bias by 3 before training and left it trainable. All other starting parameters and optimizer state matched. At update 16 the ordinary-start arms both had 0/64 valid prior argmax graphs, 0/64 exact real reconstructions, and a held-out atom-count mean absolute error of 13.625. The measured early benefit concerns occupancy. Long-run size diversity, chemical validity and reconstruction require a trained model.

A longer replay revealed that the count KL alone drives down sampled occupancy while the argmax can overshoot to empty. With several plausible elements, NO_ATOM can win the single-category argmax even though all elements taken together carry most probability. We therefore also trained the actual argmax margin. All three results below use the same saturated start, real batches, fresh training latents and 64 held-out Gaussian vectors. [Original and count-KL arms](../../outputs/runs/vae435k-prior-occupancy-20261119c/comparison.json) · [Combined loss](../../outputs/runs/vae435k-prior-occupancy-20261119d/comparison.json)

| After 64 updates | Original | Add count KL | Add count KL and argmax term |
| --- | ---: | ---: | ---: |
| Exactly 50 atoms, argmax | 64/64 | 0/64 | 0/64 |
| Empty, argmax | 0/64 | 60/64 | 1/64 |
| Mean atoms, argmax | 50.00 | 0.17 | 21.25 |
| Mean atoms, categorical sample | 48.64 | 31.03 | 38.42 |
| Exactly 50 / empty at 1.5× radius, argmax (out of 64 each) | 64 / 0 | 0 / 55 | 0 / 2 |
| Valid argmax graphs | 0/64 | 2/64 | 1/64 |
| Exact held-out reconstructions | 0/64 | 0/64 | 0/64 |
| Held-out reconstruction atom-count mean absolute error | 18.58 | 8.58 | 11.44 |

The combined objective recovered a range of hard sizes (0–45) from this single challenged initialization. In the separate ordinary-start replay, 16 updates gave hard sizes of 17–39 (mean 28.55), with zero full and zero empty among the same 64 unseen vectors. Categorical draws still skew large against the 25.51-atom training mean. This CPU ablation addresses size collapse and does not establish long-run chemistry quality or behavior along optimized latent trajectories.

## What counts as a correct graph

**Full-pass training paused on 21 November 2026.** At the user's request, V100 continuation 30134846 and its automatic continuation callback 30134847 were canceled while pending. LSF confirmed EXIT and zero runtime for both. The SHA-256 of `sparse-01/checkpoint-000077159.pt` matched `latest.json` in `/work3/s234818/latentmol/runs/vae435k-count-v100-20261119a`. The restart point is update 77,159, after 19,752,704 of the 60M training molecules.

**Full-pass update, 20 November 2026 at 08:51 CET.** The tiny VAE had processed 9,088,000 of 60M real molecules (35,500 updates). At the latest fixed evaluation, update 35,000, it recovered 4,732 of 6,767 target bonds (69.9% recall) while predicting 30,381 bonded pairs (15.6% precision). All 256 held-out reconstructions failed RDKit valence checks. The 256 fixed Gaussian argmax draws averaged 25.49 heavy atoms, with zero empty graphs, zero 50-atom graphs and zero valid molecules. The count correction was working on this panel while excess predicted bonds remained a problem. [Saved measurements](../../outputs/hpc-results/vae435k-count-v100-20261119a/snapshot-20261120T075137369Z.json)

Input canonicalization gives the encoder a consistent view of the 2D graph. Output matching solves a distinct problem. If the hard output is the same labeled graph under a permutation, the training loss uses a verified correspondence that moves the elements, formal charges and **both** bond axes together. Padding can appear in different slots. In ambiguous symmetric graphs, any verified correspondence is sufficient for hard correctness. For an incorrect graph the loss uses a bounded approximation to choose a useful correspondence for the cross-entropy gradient. Discrete correspondence selection is detached. A correct argmax still receives a confidence-dependent cross-entropy loss. [Graph matcher](../../src/latentmol/training/matching.py) · [Loss implementation](../../src/latentmol/training/losses.py)

Graph-mode evaluation reports exact labeled-graph matches on the entire evaluation panel, along with the existing canonical-molecule equality and connected RDKit validity on the designated chemistry subset. Isomers with the same atom inventory and different connectivity stay distinct. This schema does not encode stereochemistry. [Evaluation](../../src/latentmol/evaluation/evaluate.py) · [Renumbering and constitutional-isomer tests](../../tests/evaluation/test_evaluation.py)

## Reproduce the timing comparison

From the repository root in the allocated DTU environment, use a fresh output directory.

```sh
python -m latentmol.training.compact_benchmark --config experiments/vae/compact435k-v100.json --output "$NEW_OUTPUT"
```

One exclusive V100 and four host cores run both batch sizes in sequence. Both sizes and both objectives start from the same weights and AdamW state after 2,048 shared warmup examples. Each timed trial discards a separate 512-example shape warmup, then processes the same 4,096 real examples. The benchmark includes loading training rows, graph matching, forward/backward passes, gradient clipping and the optimizer. The guided arms also include the CPU repair and count objective. Each arm attempts its native batch first. On CUDA out-of-memory it retries from the shared state with smaller microbatches, maintaining the specified effective batch size. The report records actual microbatch, CUDA memory and real molecules per second. A projected 60M epoch excludes validation, checkpoints and scheduler waits. Early-run repair acceptance and timing can differ from a trained decoder. [Benchmark runner](../../src/latentmol/training/compact_benchmark.py)
