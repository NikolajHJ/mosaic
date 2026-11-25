# Can predicted bonds guide molecular VAE decoding?

A relaxed bond can carry information before it becomes the most likely bond. This pilot asks whether using that signal inside the decoder makes molecules easier to reconstruct or edit. Both VAEs start from scratch with a 128-dimensional latent vector, 50 output slots, 24 features per atom, 18 per pair and four refinement rounds. The 512-update comparison covers early learning. It spans about 0.22% of a 60M training pass and 6.55% of the 2M-example KL warmup. If both arms yield few scorable molecules, the edit test has little power to distinguish their latent spaces.

| Architecture | Encoder | Decoder | Total |
| --- | ---: | ---: | ---: |
| Tiny triangle with atom attention | 238,360 | 197,040 | 435,400 |
| Soft bond messages | 238,360 | 190,736 | 429,096 |

In the new decoder each round updates the symmetric pair states and predicts a distribution over bond types using the same classifier as the final output. A neighbor sends a different learned vector for each real bond type. A message is weighted by its conditional bond probability and by the two slots’ occupancy probabilities. Self-bonds and NO_BOND carry zero message. Division by the square root of one plus the expected degree limits the scale when many pairs have moderate bond probability. The atom update and pair update retain residual paths. The comparison changes both atom attention and triangle updates, so a difference between the two VAEs alone cannot identify which of those operations mattered. [Model](../../src/latentmol/models/refined_decoder.py) · [Model checks](../../tests/training/test_bond_message_decoder.py)

## Training and evaluation

1. Train each arm for 512 updates on exactly the same 131,072 distinct training rows from the 60M balanced PubChem view. Use batch 256, microbatch 128 and one Gaussian prior draw per 15 real examples (8,738 total per arm). Copy the encoder and all shared tensors outside the refinement rounds at initialization. Both arms see identical posterior noise and prior draws. The objective uses graph-matched reconstruction and KL on real molecules, both atom-count terms on prior draws, AdamW at 0.001 with weight decay 0.0001, and zero dropout. [Run config](compact-decoder-comparison.json) · [Base recipe](compact435k-count-v100.json) · [Training loop](../../src/latentmol/training/compact_decoder_comparison.py)
2. At initialization and updates 16, 128 and 512, evaluate the same 256 held-out molecules and 256 Gaussian latent vectors. Report exact graph reconstruction under atom renumbering, decoded validity, random argmax and sampled validity, size distribution and 50-slot saturation. Also measure training throughput and peak allocated GPU memory. The endpoint topology check splits cyclic from acyclic targets and records connected reconstructions with the right independent cycle count. An incorrect structure can have the correct cycle count. [Fixed panel](../../src/latentmol/evaluation/compact_training_panel.py) · [Topology check](../../src/latentmol/evaluation/decoder_topology.py)
3. Freeze both endpoints and encode the same four held-out molecules with 25 heavy atoms. For a shared BRD4 pocket, follow the actual negative Protenix-Tiny contact gradient at temperature 1 and a random direction in both signs. Each direction gets radii 0.25, 1, 2 and 4 in latent L2 units. The native context holds each start’s 25 selected slots fixed. Record both this fixed-slot readout and the decoder’s raw graph, including invalid and unchanged outcomes. [Edit config](../latent-design/protenix-vae/compact-edits.json) · [Latent sweep](../../src/latentmol/design/decoder_comparison.py)
4. Select the first two distinct valid edits per start, direction and readout in radius order. Freshly construct discrete ligand features and a conformer, then score the pocket contacts again. A return to the original input is recorded separately. The primary outcome is the fraction of all four starts where a distinct valid edit along the gradient beats its own decoded center on this fresh contact score. Report how many centers were valid and scorable alongside the headline, plus rescues of invalid centers. Four starts and one protein give an exploratory comparison. The contact score measures a model prediction. [Fresh scorer](../../src/latentmol/design/temperature_fresh.py)

## Execution and outputs

At the bond-message endpoint, temporarily zero the learned messages in all four rounds and decode the same held-out posterior means again. This measures whether the trained model uses that path. Report the shifts in atom probabilities and in bond probabilities on true bonds and active nonbonded pairs separately, plus hard graph changes. Restore the original path and verify its logits reproduce exactly. This inference-only intervention uses up to 64 held-out molecules. [Ablation](../../src/latentmol/evaluation/bond_message_ablation.py)

The relaxed Protenix adapter consumes element probabilities, expected formal charge and bond presence. Redistributing mass between single and double bond while holding presence and all other inputs fixed leaves this contact input unchanged. Internally the decoder can still use the difference between those bond types to change subsequent atom and pair predictions. The fresh discrete evaluation rebuilds each selected molecule from its labeled graph. [Native feature mapping](../../src/latentmol/guidance/mosaic_protenix_worker.py)

The [single-allocation entry point](run_compact_decoder_pilot.sh) runs training first, releases its process, then runs the [native edit comparison](../latent-design/protenix-vae/run_decoder_comparison.sh). Use a fresh output directory. The scheduler plan identifies the installed CUDA routing helper.

```bash
bash experiments/vae/run_compact_decoder_pilot.sh NEW_OUTPUT GPU_PLAN_JSON GPU_PLAN_SHA256
```

`training/comparison.json` contains arm identities, endpoint quality, topology and timing. Each arm has a `quality.json`, update trace and ordinary loadable `final.pt`. The four held-out inputs, decoded centers and all attempted edit radii are in `edits/val-*.json`. `edits/fresh/summary.json` records per-candidate native score or failure. `edits/comparison.json` reports comparisons against both the original input and each architecture’s center.

DTU accepted job **30116009** on 19 November 2026 for one exclusive 32 GB V100, four CPU slots and four hours. The first scheduler check showed PEND and zero running time. The destination is `/work3/s234818/latentmol/runs/vae-bond-message-20261119a`. [Submission receipt](../../outputs/hpc-submissions/vae-bond-message-20261119a/submitted.json)

## First local results

The matched CPU run trained on 480 distinct PubChem training molecules and 32 Gaussian draws per arm over 16 steps. Held-out structures and Gaussian evaluation draws were disjoint from the training draws. The 64 held-out molecules had 15–31 heavy atoms and 62 contained a ring. The two networks shared the initial encoder and all common weights outside the refinement rounds. [Training summaries](../../outputs/runs/compact-decoder-cpu-20261119a/matched/comparison.json) · [Local run](../../outputs/runs/compact-decoder-cpu-20261119a/check.py)

| Held-out measurement | Triangle | Bond messages |
| --- | ---: | ---: |
| Reconstruction loss, initial → update 16 | 6.890 → 3.850 | 6.677 → 3.864 |
| Active atom type accuracy, update 16 | 85.54% | 85.74% |
| Correct bonded pairs out of 1,586 target bonds | 0 | 0 |
| Connected reconstructions out of 64 | 0 | 0 |
| Valid random argmax / categorical draws out of 64 each | 0 / 0 | 0 / 0 |
| Prior argmax size, update 16, mean heavy atoms | 33.16 | 32.06 |
| Prior argmax draws occupying all 50 slots, initial → update 16 | 38 → 0 | 27 → 0 |

Both learned atom identities and reduced reconstruction loss, while every held-out argmax reconstruction remained disconnected. This identifies bond recovery and connectivity as the first measurements to inspect in the larger run. No protein model contributed to these local figures.

Switching off the four message rounds at the bond-message endpoint changed hard labels for 64/64 held-out graphs. Mean total variation was 0.0773 on occupied target atoms, 0.0192 on true bonds and 0.0216 on nonbonded pairs of target atoms. This establishes that the trained decoder was using its messages. Neither setting produced a valid held-out reconstruction. Re-enabling messages reproduced the original logits exactly. [Saved ablation](../../outputs/runs/compact-decoder-cpu-20261119a/message-ablation.json)

A separate RTX 4060 Laptop GPU timing measured a decoder forward pass and backward pass to the latent through the same soft readout, using frozen untrained weights and FP32. After six warmups per arm, 16 interleaved measurements per arm gave median times of 36.69 ms (triangle) and 27.02 ms (messages) for a single molecule, and 182.91 ms and 115.99 ms for batch 16. This measures 26% and 37% lower decoder latency on that device. External scoring, whole-VAE training, and data loading fall outside the timed region. [Raw timings](../../outputs/runs/compact-decoder-cuda-timing-20261119a/timing.json) · [Timing script](../../outputs/runs/compact-decoder-cuda-timing-20261119a/run.py)

## Completed V100 comparison

Job **30116009** finished successfully in 24 minutes 28 seconds. At update 16 neither decoder recovered any target bond. By update 128, bond recall reached 54.9% for triangle and 50.0% for bond messages. The endpoint used 512 updates per arm on 131,072 real molecules and the same 256 held-out targets. [Launch receipt](../../outputs/hpc-results/vae-bond-message-20261119a/launch.json) · [Triangle validation](../../outputs/hpc-results/vae-bond-message-20261119a/training/triangle_refine/quality.json) · [Bond-message validation](../../outputs/hpc-results/vae-bond-message-20261119a/training/bond_message/quality.json)

| Update 512 | Triangle | Bond messages |
| --- | ---: | ---: |
| Recovered bonds out of 6,767 | 4,588 (67.8%) | 4,514 (66.7%) |
| Predicted bonds | 32,678 | 31,653 |
| Bond precision | 14.0% | 14.3% |
| Connected held-out reconstructions | 241/256 | 231/256 |
| Valid held-out reconstructions | 0/256 | 0/256 |
| Valid random argmax draws | 1/256 | 0/256 |
| Valid categorical draws | 0/256 | 0/256 |
| Real training molecules per second | 198 | 246 |

Each decoder predicted over four times the reference bond count. All 256 held-out reconstructions in each arm failed the valence check, and zero had both the correct independent cycle count and a connected graph. Switching off the learned messages changed hard labels on 64/64 held-out examples and restoring them reproduced the original logits. [Completed training comparison](../../outputs/hpc-results/vae-bond-message-20261119a/training/comparison.json)

The contact assay ran 104 relaxed Protenix evaluations. Moving 0.25 units along the negative gradient reduced the relaxed contact loss on all four inputs for each architecture. All decoded centers were invalid, and neither descent nor the two random controls found a distinct valid edit at any radius. The four original source molecules received fresh scores. There was no scorable center–edit pair to measure improvement on discrete molecules. [Edit results](../../outputs/hpc-results/vae-bond-message-20261119a/edits/comparison.json) · [Fresh scores](../../outputs/hpc-results/vae-bond-message-20261119a/edits/fresh/summary.json)

## Hybrid with triangles and typed bond messages

Do triangle updates help a bond-message decoder suppress false bonds? Each of the four rounds first updates symmetric pair states through intermediate atom slots, then uses that round’s predicted bond probabilities to send typed messages between atom slots. The same head produces the final bond logits. Total trainable parameters are 435,288 (238,360 encoder, 196,928 decoder), adding 6,192 to the bond-message VAE. The small model keeps its 128-dimensional latent and 50 output slots. [Model](../../src/latentmol/models/refined_decoder.py) · [Hybrid benchmark settings](compact-hybrid-comparison.json)

The single new arm starts from scratch and copies the reconstructed time-zero bond-message weights into every shared parameter. Only the new triangle branches have separate parameters. Their output projections start at zero so the initial outputs reproduce the function of the bond-message control. Initialization, row order, fixed validation panel and noise streams are checked against the saved comparison from job 30116009. This reuses both completed controls. [Training procedure](../../src/latentmol/training/compact_decoder_comparison.py)

The V100 comparison uses the same 512 updates on 131,072 real training molecules with a logical batch of 256, the original prior losses and 256 held-out molecules. Compare predicted and correct bond counts, precision and recall together, connected reconstructions, validity and prior generation. A lower predicted bond count can come from simply missing more true bonds. The original bond-message endpoint reached 14.3% precision at 66.7% recall. Also compare training throughput on the same V100 class. This experiment tests reconstruction and generation.

DTU accepted hybrid-only job **30128096** on 20 November. It requests one 32 GB V100 and four CPU slots for up to one hour. It writes to `/work3/s234818/latentmol/runs/vae-hybrid-20261120a/training`. The source archive starts from the original comparison release and adds the model, paired trainer and config changes. Nine regression tests passed on a fresh extraction of the submitted archive. [Submission receipt](../../outputs/hpc-submissions/vae-hybrid-20261120a/submitted.json)

**Local 16-update result.** The hybrid saw the same 480 training molecules and 32 prior draws as the saved local controls. All three produced 0/64 connected reconstructions and 0/64 valid random argmax molecules. In particular, none of the three hard decoders predicted any bond at this point. [Hybrid endpoint](../../outputs/runs/compact-hybrid-cpu-20261120a/matched/comparison.json)

| On the same 64 held-out graphs | Triangle + attention | Bond messages | Hybrid |
| --- | ---: | ---: | ---: |
| Reconstruction loss | 3.850 | 3.864 | 3.860 |
| Bond precision at 66.71% recall | 7.44% | 6.55% | 6.85% |
| Average precision for bond presence | 7.11% | 6.89% | 7.08% |

For the ranking diagnostic, score all 78,400 unordered slot pairs by marginal bond presence, which includes both endpoint occupancies. Apply the usual detached reconstruction alignment separately for each model. Selecting each model's threshold on this panel recovered the same 1,058 of 1,586 bonds, so these are descriptive measurements at matched recall. The small differences give no clear winner at 16 updates. [Scoring procedure](../../outputs/runs/compact-hybrid-cpu-20261120a/bond_ranking.py) · [Hybrid bond ranking](../../outputs/runs/compact-hybrid-cpu-20261120a/bond-ranking-triangle_bond_message.json)

The four hybrid triangle output projections moved from zero to weight norms of 0.140–0.147. Silencing just those projections on 16 held-out molecules shifted conditional bond probabilities by a mean total variation of 0.000827 and changed zero hard graphs. Restoring the triangles reproduced all original logits. The branch had begun learning but had little influence on this early readout. [Triangle intervention](../../outputs/runs/compact-hybrid-cpu-20261120a/triangle_effect.json)

## Longer three-decoder comparison

Does the hybrid learn to choose the right bonds once training passes the two-million-example KL warmup? Each decoder starts from scratch and receives **8,192 updates** (2,097,152 distinct real PubChem molecules and 139,810 Gaussian training draws per arm). The three architectures share the same ordered real inputs, noise, losses, batch of 256 and initial tensors wherever the architectures agree. The hybrid begins with the same predictions as the bond-message control. This is one new training seed; the original experiment used a different seed and saved model weights without AdamW state. [Long config](compact-decoder-long.json) · [Training code](../../src/latentmol/training/compact_decoder_comparison.py)

The fixed panel has 1,024 held-out molecules and 1,024 Gaussian vectors. At updates 0, 512, 2,048, 4,096 and 8,192, inspect bond precision and recall together, predicted versus true bond counts, connectivity, chemically valid and exact reconstructions, random validity and the distribution of atom counts. Intermediate weights at 2,048 and 4,096 and the final weights are loadable for further evaluation.

The endpoint also measures **bond precision at two-thirds recall** and average precision over all score thresholds. The first calculation ranks all unordered slot pairs by marginal bond probability, including endpoint occupancy. The second looks only at pairs of occupied target atoms after the reconstruction alignment and ranks them by conditional bond probability. Improvement on the second readout is evidence for learning which atoms bond even when their presence is given. These thresholds are chosen on the panel for descriptive comparison. [Ranking calculation](../../src/latentmol/evaluation/decoder_bond_ranking.py)

Print measured outcomes from saved files with `python experiments/vae/summarize_decoder_comparison.py TRAINING_DIRECTORY --all-updates`. The report distinguishes false bonds between two occupied target slots from false bonds involving padding. [Result reader](summarize_decoder_comparison.py)

DTU accepted job **30128180** on 20 November 2026. It runs the three arms sequentially on one 32 GB V100 with four CPU slots and a 16-hour wall limit. Based on the previous 198 and 246 real molecules/s for the two controls, the three-arm estimate is roughly 8–10 hours after dispatch. The hybrid and larger panels introduce uncertainty. Output is `/work3/s234818/latentmol/runs/vae-decoder-long-20261120a/training`. Twelve tests passed on an extraction of the submitted archive, including a small three-arm optimization with comparison of initial predictions, noise streams and checkpoint reloads. At the first scheduler check (09:27 CET), the job was pending. [Experiment submission](../../outputs/hpc-submissions/vae-decoder-long-20261120a/submitted.json)

## Wider hybrid

Do richer pair representations improve bond selection? The wider hybrid increases the pair state from 18 to 48 features and the triangle bottleneck from 12 to 32. Both hybrids have four rounds, 24 atom features, 50 slots, a 128-dimensional latent vector and a 238,360-parameter encoder. The wider decoder has 243,308 parameters versus 196,928 for the original hybrid, bringing the full VAE from 435,288 to 481,668. At initialization we copy every tensor of matching shape from the reproduced original hybrid. Resized tensors use fresh initialization, so initial predictions may differ. This comparison tests the two width changes together.

The [capacity config](compact-hybrid-wide.json) inherits the [8,192-update comparison](compact-decoder-long.json), including the real examples, posterior noise, Gaussian prior draws, loss, optimizer, batch sizes and fixed 1,024-molecule / 1,024-prior panel. Only the additional arm trains. Use the same launcher with `--config experiments/vae/compact-hybrid-wide.json`. Compare its bond precision at two-thirds recall, false bond counts, held-out reconstruction, prior validity, GPU throughput and peak allocated memory with the narrow hybrid from job 30128180.

**Local result, 20 November 2026.** After 16 updates on the same 480 real training molecules and 32 prior draws, reconstruction loss was 3.752 for the wider hybrid versus 3.860 for the original. At exactly 1,058 of 1,586 held-out bonds recovered by thresholding (66.71% recall), precision was 7.67% versus 6.85%. Both models still chose zero bonds at argmax, with zero valid reconstructions and zero valid prior argmax draws out of 64 each. This is one small panel and one seed. The training-row and noise digests and panel metadata matched. [Wide results](../../outputs/runs/compact-wide-cpu-20261120a/matched/comparison.json) · [Narrow results](../../outputs/runs/compact-hybrid-cpu-20261120a/matched/comparison.json) · [Narrow bond ranking](../../outputs/runs/compact-hybrid-cpu-20261120a/bond-ranking-triangle_bond_message.json)

DTU accepted **30129115** for one 32 GB V100, four CPU slots and a 10-hour wall limit. Its scheduler state was PEND at 10:26 CET on 20 November. Twelve checks passed on the sealed source, including a short training run with the widened decoder and checkpoint reload. [Submission receipt](../../outputs/hpc-submissions/vae-wide-hybrid-20261120a/submitted.json)
