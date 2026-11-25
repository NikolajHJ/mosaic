# Weak functional-group loss for random VAE draws

Can a small training signal increase the yield of **distinct valid aromatic molecules and amides** without reducing reconstruction, scaffold diversity or throughput? The recent epoch-47 checkpoint produced eight aromatic molecules and 19 amides among 512 random argmax decodes, including 269 valid molecules. [Earlier matched motif counts](training_data_eda/generation-motifs.md)

## Loss

Every fourth optimizer update, reuse the first 48 independent Gaussian vectors already decoded for the prior branch. Sample one categorical graph per vector at temperature one using a dedicated generator seeded by the absolute update. Inspect the raw graph with the project's strict chemistry check. Each connected, sanitized, radical-free molecule can contribute once to each of two indicators. RDKit perceives aromatic rings, including supported Kekulé representations. The amide pattern also accepts lactams, ureas and carbamates. Invalid draws remain in the denominator with zero hits.

For the aromatic and amide indicators separately, compute the fraction of **all attempts** that yield a valid molecule with that motif. Average the two penalties `0.5 * max(floor - fraction, 0)^2`, then multiply by **0.3** on the scheduled updates. The exploratory floors are 5% aromatic and 8% amide. These are gentle targets for useful product yield. A molecule with neither group is a legitimate output. The coefficient was chosen by measuring its actual contribution to the training gradient.

The sampled labels, RDKit checks and counts are detached. The gradient comes from the log probability of each complete sampled graph under the live decoder. For each graph, compute how much the batch penalty would change if its indicator switched from zero to one with the other draws held fixed. A leave-one-out baseline reduces variance. This estimates the derivative of the expected **finite-batch** penalty. A graph that just took the batch across its floor can still give a gradient; if enough others already meet the floor, its coefficient is zero. When a batch has zero examples of a motif, its observed shortfall is positive but that batch supplies no gradient for that motif. The trainer logs both conditions. [Loss and estimator](../../src/latentmol/guidance/functional_groups.py) · [Training integration](../../src/latentmol/training/projected_continuation.py)

## Matched pilot

One 32 GB V100 runs the four-round control and weak-loss arm in sequence for 256 updates each from the same completed epoch-47 checkpoint. Both use the epoch-48 data order, graph-matched reconstruction, KL, projected valence loss, occupancy guidance and 240 real graphs plus 240 priors per update. Across an arm, the extra check covers 3,072 of 61,440 prior draws (5%) and needs no additional decoder pass. The separate epoch-48 four-versus-seven-round first-leg jobs keep their sealed recipes.

Compare the arms on the same 2,048 fresh latent vectors, using both raw argmax and one categorical sample. Report valid attempts, unique valid structures with each motif, distinct nonempty Murcko scaffolds, paired gains and losses, exact reconstruction and warm optimizer-step time. Save the raw graphs, including failures. The training signal acts on categorical samples; argmax yield is an independent outcome. One short paired run can detect a substantial change but has limited power for rare groups.

Run `python experiments/vae/submit_motif_pilot.py seal` then `python experiments/vae/submit_motif_pilot.py submit` once per new experiment package. Preserve the existing scheduler receipt when checking this pilot. [Settings](epoch48-motif-pilot.json) · [Matched evaluations](motif_pilot.py) · [Local benchmark](benchmark_functional_groups.py)

**Submission, 21 November 2026.** Job **30147253** is queued on one 32 GB V100 with eight CPU cores and a six-hour limit. Its scheduler dependency is `started(30131204)`, so the common epoch-47 checkpoint will have been released before the pilot starts. The coordinator waits for each training child to exit before using the exclusive GPU for scoring. Earlier submission 30147239 was withdrawn before starting to correct that launch order and calibrate the coefficient. LSF recorded zero runtime for the superseded job. Eleven focused CPU tests passed. [Scheduler verification](../../outputs/hpc-submissions/vae-e48-weak-motifs-20261121b/verified.json)

## Local benchmark

On 21 November 2026 we probed the frozen four-round VAE at update 541677 on a GeForce RTX 4060 Laptop GPU with PyTorch 2.11.0+cu128. Across 32 fresh 48-vector batches (1,536 draws in total), raw argmax produced 850 valid molecules, including 26 aromatics and 49 amides. Sampling once at temperature one on the same vectors produced 663 valid molecules, including 14 aromatics and 30 amides. Twenty-five of 32 batches gave a nonzero gradient to the logits. Twelve gave an aromatic signal and 20 an amide signal. The timed region from existing logits through sampling, strict chemistry checks and backward to those logits took a median 21.5 ms per active batch. [Draws and gradient norms](../../outputs/runs/vae-functional-group-loss-20261121a/probe.json)

For timing and weight selection, each arm restored the same weights, optimizer state and matched random draws. Each step trained on 240 real graphs and 240 priors, with microbatches of 48, epoch-46 guidance on 16 priors, projection, graph-matched reconstruction, KL, 5% atom-refinement dropout and an AdamW update. Execution order alternated between pairs and we excluded one warm-up pair. The real graph tensors were preloaded from the saved PubChem panel. These measurements exclude storage, checkpoint and evaluation time. The restored models were disposable.

| Motif weight | Matched full-step pairs | Median baseline step | Median scheduled step minus its paired control | Projected extra time averaged over all steps |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 6 | 4.723 s | +15.7 ms | 0.08% |
| 0.30 | 4 | 4.725 s | +63.5 ms | 0.34% |

Timing variation exceeded either estimated overhead. At the selected weight 0.30 individual scheduled-step differences ranged from −55 to +192 ms. The gradient comparison uses the norm of the *difference between parameter gradients* divided by the control gradient norm, restoring each vector's pre-clipping scale. At 0.01 the median added norm across four matched steps was 0.016% of the base. At 0.30 it was 0.47% (range 0.23–0.82%). The added gradient's cosine with the base ranged from −0.44 to +0.45 across those four steps. These results measure immediate signal and local overhead; learning outcomes and V100 timing require the matched pilot. [Six-pair timing](../../outputs/runs/vae-functional-group-loss-20261121a/timing.json) · [Weight 0.01](../../outputs/runs/vae-functional-group-loss-20261121a/gradient-scale.json) · [Weight 0.30](../../outputs/runs/vae-functional-group-loss-20261121a/gradient-scale-weight030.json)

To reproduce with a CUDA PyTorch and RDKit environment, set `PYTHONPATH=src`. Point `STUDENT` to update 541677, `REFERENCE` to the epoch-46 checkpoint at update 325498, `PANEL` to the saved PubChem JSON validation panel, and `DEST` to a new output directory.

```sh
python experiments/vae/benchmark_functional_groups.py --mode probe \
  --config experiments/vae/epoch48-paired-leg1.json --checkpoint "$STUDENT" \
  --panels 32 --draws 48 --microbatch 12 --weight 0.30 --output "$DEST/probe.json"
python experiments/vae/benchmark_functional_groups.py --mode timing \
  --config experiments/vae/epoch48-paired-leg1.json --checkpoint "$STUDENT" \
  --reference "$REFERENCE" --panel "$PANEL" --batch-size 240 --microbatch 48 \
  --pairs 4 --compare-gradients --weight 0.30 --every 4 --draws 48 \
  --aromatic-floor 0.05 --amide-floor 0.08 --output "$DEST/weight030.json"
```

The linked initial frozen probe used weight 0.01. Using 0.30 scales the logit gradients by 30 with the same samples, hits and batches with a signal.

## Higher floors at the same weight

The [seven-round epoch-48 training leg](epoch47-occupancy-continuation.md#functional-group-loss-in-the-seven-round-arm) now uses the 8%/12% floors at weight 0.30. The short four-round pilot retains the original 5%/8% floors.

We repeated the local commands above with `--weight 0.30 --aromatic-floor 0.08 --amide-floor 0.12` on the same checkpoint, latents, categorical draws and real batches. At 8%/12% the auxiliary parameter gradient was 0.59–1.33% of the base norm across four scheduled steps (median 0.78%). At 5%/8% on the same steps it was 0.23–0.82% (median 0.47%). In the separate 32-batch probe, both choices gave a signal on exactly the same 25 batches, including 12 informative for aromatics and 20 for amides. With fixed draws the auxiliary gradient scales linearly with its weight: lowering it to 0.20 at the higher floors would imply a median of 0.52%. This replay measured the training signal, without testing learning at the higher floors. The submitted pilot uses 5%/8%. [Matched full-gradient replay](../../outputs/runs/vae-functional-group-loss-20261121a/gradient-scale-weight030-floors0812.json) · [32-batch replay](../../outputs/runs/vae-functional-group-loss-20261121a/probe-floor-08-12.json)
