# Teaching aromatic rings and amides on random VAE decodes

The seven-round first leg frequently produced connected rings with the wrong bond orders. Its original auxiliary measures how often a categorical sample already contains a valid aromatic ring or amide. When every sample misses the group, the batch supplies zero gradient for it. A higher weight on that estimator still gives zero on those batches. A training-data audit found aromatic rings in 87.1% of 50,000 real training molecules. On generated ring edges that could support an aromatic edit, the frozen teacher predominantly requested single bonds. [Earlier mechanism tests](epoch48-aromatic-mechanism.md)

## Loss used in the comparison

The candidate keeps the existing sampled yield loss and adds local supervision for a missing group. It uses the same latent draws and decoder forward pass. Given a batch below the 8% aromatic or 12% amide floor, it hardens those priors and inspects their occupied graphs. The proposals preserve atom identities, charges, occupancy, and which pairs are bonded.

- For an aromatic target, examine an existing neutral six-membered C/N ring. Prefer one that already contains double bonds. Try assigning aromatic type to its six edges. The whole candidate must sanitize and round-trip, all six selected bonds must be perceived as aromatic, and the molecule must be connected and radical-free. An existing valid aromatic already satisfies this group.
- For an amide target, look for a carbon bearing both a hydroxyl and a directly bonded nitrogen. Try changing that carbon–oxygen bond to a double bond. The resulting molecule must pass the same chemistry checks and match the amide definition used in the corpus analysis. An existing amide already satisfies this group.
- Each group can select one compatible site per prior and examine at most three proposals. If both groups are selected on one graph, their sites have disjoint atoms. Each group's accepted edges supply mean cross-entropy over the bond types conditional on an existing edge. The mean over the two groups gives the auxiliary. A missing suitable site contributes zero for that group. Model evaluation always counts the raw decode.

With completion weight 1.0 inside the existing coefficient of 0.30, both the old score term and the new completion term run every fourth optimizer update. Production uses 48 of the 240 already decoded priors. The original real reconstruction, KL, projection and reference losses stay in effect. The candidate has no ring-size requirement for individual molecules. A saturated six-membered ring is a possible target on a subset of under-target prior draws, and remains a legitimate real reconstruction target. Five-membered aromatics and other common functional groups are outside this first local teacher. The chemical perception rules are RDKit's. [Loss implementation](../../../src/latentmol/guidance/motif_completion.py) · [Batch objective](../../../src/latentmol/guidance/functional_groups.py) · [Training config](../epoch48-seven-round-completion.json) · [RDKit aromaticity model](https://rdkit.org/docs/RDKit_Book.html#aromaticity)

## Frozen checkpoint checks

At update 588498 two production-sized groups of 48 Gaussian priors supplied no sampled aromatic hits. The old aromatic estimator was zero on both. On the argmax graphs, the local teacher found 25 aromatic and 15 amide targets. Applying those hypothetical edits gave 25 aromatic and 27 amide outputs, compared with zero and 12 originally. Both sets had 84 valid graphs. The added motif loss supplied finite symmetric bond gradients. The extra loss computation took 139–145 ms per scheduled call in this two-thread CPU comparison. These counts describe target availability. Training effectiveness is measured on independent, unedited molecules. [Checkpoint benchmark](../../../outputs/runs/vae-e48-r7-complete-20261122a/isolated-motif-cpu.json)

## Ring encoding and ring size

Ring size and aromaticity are different outcomes. A six-membered cyclohexane ring is a correct reconstruction target. The first candidate above used aromatic bond labels only after testing the whole edited molecule. Even that chemical check tells us only that the *target* works. Independent pair predictions can move on a subset of the six edges or on unrelated bonds. All model comparisons below therefore evaluate complete, unedited draws and count invalid attempts in the denominator. Ring histograms count the fraction of molecules containing at least one ring of each size. One molecule may contribute to multiple sizes.

A second candidate sets `functional_groups.ring_encoding` to `kekule`. RDKit assigns alternating single and double bonds to the verified aromatic endpoint. Those six categorical labels supply the bond-type targets. We check that applying them to their original slots with the rest of the molecule frozen returns the same canonical SMILES as the explicit aromatic endpoint. This candidate uses one RDKit Kekulé assignment per site.

We also tested what happens if the six edge predictions cross their decision boundaries at different times. On the 25 eligible rings from update 588498, we applied one required edit at a time, easiest logit margin first. Explicit aromatic targets led to 45 invalid intermediate graphs in 124 attempts. Alternating single/double targets needed three changes on each ring and gave zero invalid intermediates in 50 attempts. All 25 final molecules agreed between encodings. The median margin per changed bond was larger for the alternating targets (16.84 vs 9.89), so the valid intermediates alone cannot predict which signal the VAE learns more readily. [Paired edit paths](../../../outputs/runs/vae-e48-r7-complete-20261122a/aromatic-edit-paths.json)

Bond-order supervision preserves adjacency. A separate CPU test on 223 valid prior decodes found that one chemically verified two-edge rewire per eligible graph could reduce the number with an eight-membered or larger ring from 165 to 51, keeping all seven aromatic molecules and all 26 amides. Those are edited proposals. A comparable experiment with training would be needed to measure changes in raw ring-size frequencies. Fixed-panel monitoring in the new trainer records aromatic and amide counts alongside ring-size frequencies. [Topology pilot and matched real-data rates](ring-topology-probe.md)

## What the V100 learned in 128 updates

Interactive DTU job **30202270** trained the original objective and full-strength Kekulé completion sequentially from seven-round checkpoint 580498. Each optimizer step used 64 real graphs and 64 Gaussian priors. The auxiliary used 16 decodes every fourth update. The two arms shared rows, initialization, optimizer state, training randomness and evaluation seeds.

| Raw outcome | Original loss | Kekulé completion |
|---|---:|---:|
| Valid on 1,024 fresh categorical draws | 665 | 506 |
| Aromatic on those draws | 17 | 13 |
| Amide on those draws | 40 | 81 |
| Valid on the same 1,024 latent vectors by argmax | 893 | 870 |
| Aromatic by argmax | 25 | 25 |
| Amide by argmax | 52 | 148 |
| Exact reconstructions on 128 held-out graphs | 116 | 117 |
| Wall time in the training loop | 197.4 s | 198.5 s |

Among 208 categorical draws that changed from valid to invalid, RDKit first reported overvalence on 192. Of 300 newly introduced double bonds on the analyzed sanitization failures, 40 matched a targeted edge in the paired control graph. Common failures included a neutral oxygen with one single and one double bond, and a carbon with three singles and a double. This arm shows why valid local target molecules and finite gradients are insufficient evidence of better generation. [Paired outcomes](../../../outputs/runs/vae-e48-r7-complete-20261122a/kekule-128-v100/short-summary.json) · [Chemical failure analysis](../../../outputs/runs/vae-e48-r7-complete-20261122a/kekule-128-v100/paired-sanitization.json)

A second V100 comparison used explicit aromatic targets and reduced only the relative amide weight to 0.10. After 512 updates, 681/1,024 fresh categorical control draws were valid versus 425 for treatment. Aromatics went from 13 to 10, amides from 40 to 30, and both arms reconstructed 117/128 exactly. Of 294 formerly valid draws that became invalid, 214 failed kekulization. The treatment put an explicit aromatic label somewhere on 705 graphs, against 100 for control. An aromatic bond count alone would reverse the interpretation of this experiment. The training loops took 782 and 781 seconds. [Job 30203823, matched endpoints](../../../outputs/runs/vae-e48-r7-complete-20261122a/balanced-512-v100/arms/paired-512.json) · [First chemical failure on each paired draw](../../../outputs/runs/vae-e48-r7-complete-20261122a/balanced-512-v100/full-paired-sanitization.json)

## The complete-ring objective

The pattern variant ranks verified local edits by their present bond-label probabilities, selecting only enough candidates to cover the observed batch shortfall. Each positive site's bond loss is divided by the full batch size. It also estimates three kinds of error directly from live bond probabilities. These are aromatic bonds outside protected ring sites, partly aromatic assignments within a protected ring, and double bonds on neutral C–O edges whose endpoints already use their available valence. For ring-edge probabilities $p_i$, the probability of a partial assignment under independent bond sampling is $1-\prod_i p_i-\prod_i(1-p_i)$. The model may keep a saturated ring. Eligibility and the scaffold are held fixed during differentiation. [Implementation](../../../src/latentmol/guidance/motif_patterns.py) · [CPU loss and gradient probe](motif-pattern-probe.md)

## What six explicit aromatic targets changed

The matched RTX 4060 pilot started at seven-round checkpoint 580498. Both arms retained the same AdamW state, frozen teacher, 8,192 real graphs, 8,192 Gaussian priors, and training random streams. The control reproduced the historical control's endpoint weights bit for bit. The treatment added conditional bond-type supervision on verified six-edge aromatic and carbonyl edits, both at full relative strength. The 1,024 extra Gaussian vectors and categorical sampler used separately fixed seeds. Each count below uses all attempts as its denominator. [Paired draws, scaffold counts and timing](../../../outputs/runs/vae-aromatic-mechanism-20261121a/completion-20261122b/paired-summary.json)

| Outcome | Current objective | Full-strength explicit completion |
|---|---:|---:|
| Valid / aromatic / amide, independent categorical samples (1,024) | 642 / 15 / 43 | 335 / 8 / 65 |
| Valid / aromatic / amide, argmax on those same codes (1,024) | 885 / 24 / 46 | 829 / 28 / 172 |
| Valid / aromatic / amide, original argmax panel (1,024) | 866 / 24 / 50 | 824 / 35 / 192 |
| Exact reconstructions (128) | 116 | 115 |
| Unique nonempty scaffolds among valid independent categorical draws | 636 | 332 |
| Eight-membered or larger ring among valid independent argmax decodes | 628/885 (71.0%) | 613/829 (73.9%) |
| Training seconds (128 updates) | 946.6 | 982.7 |

On the independent categorical panel, the treatment made 50 previously invalid draws valid and made 357 previously valid draws invalid. Across all 1,024 graphs, the number containing at least one explicit aromatic bond rose from 104 to 656. The number that contained such a bond and failed validity rose from 65 to 499. RDKit's first reported problem on the 357 lost graphs was excess valence in 200 cases and a kekulization failure in 150. Seven graphs were disconnected. New aromatic labels often failed to make a complete valid aromatic ring. [Failure classification](../../../outputs/runs/vae-aromatic-mechanism-20261121a/completion-20261122b/first-chemical-failure.json)

The treatment found 128 aromatic and 45 amide sites across its 512 scheduled prior graphs. Its median gradient norm before clipping on the 32 scheduled updates was 5.91, versus 1.85 for control. The global clipping threshold of 5 was exceeded on 21 of those updates in treatment and zero in control. We would keep this full-strength loss out of production. The higher amide yield and nearly preserved reconstruction do not compensate for the decline in valid generation.

To repeat the local comparison, set `CKPT`, `TEACHER`, `TRAIN`, `HELDOUT` and `OLD_CONTROL` to the same source checkpoint, frozen teacher, panel files and historical summary used in the [mechanism pilot](epoch48-aromatic-mechanism.md). Set `OUT` to a fresh directory and run from the repository root with `src` and the root on `PYTHONPATH`.

```sh
for arm in control treatment; do
    python -m experiments.vae.training_data_eda.teacher_motif_training_pilot \
        --checkpoint "$CKPT" --reference "$TEACHER" \
        --config experiments/vae/epoch48-seven-round-motifs.json \
        --train-panel "$TRAIN" --heldout-panel "$HELDOUT" \
        --completion-arm "$arm" --control-summary "$OLD_CONTROL" --output "$OUT/$arm"
done
python -m experiments.vae.training_data_eda.compare_completion_pilot "$OUT"
```

## Can we teach a few complete patterns while protecting other bonds?

We reused the verified local control and trained one additional treatment for 128 matched updates. It chose up to the sampled batch's motif deficit, normalized positive cross-entropy by all prior graphs, and used relative amide weight 0.10. It also penalized aromatic probability outside eligible rings, partially aromatic patterns inside protected rings, and double-bond probability at saturated C–O sites. The source checkpoint, treatment rows, corruption-free real inputs, optimizer, frozen teacher and random streams matched the control. The final model weights and raw outcomes for every draw are saved. [Paired results and component diagnostics](../../../outputs/runs/vae-aromatic-mechanism-20261121a/pattern-20261122a/paired-summary.json)

| Outcome | Current objective | Pattern treatment |
|---|---:|---:|
| Valid / aromatic / amide on the 1,024 independent categorical draws | 642 / 15 / 43 | 630 / 18 / 44 |
| Valid / aromatic / amide by argmax on the same latent vectors | 885 / 24 / 46 | 877 / 30 / 47 |
| Valid / aromatic / amide on the original 1,024 argmax draws | 866 / 24 / 50 | 869 / 31 / 53 |
| Exact reconstructions on 128 held-out graphs | 116 | 116 |
| Unique nonempty scaffolds among valid independent categorical draws | 636 | 623 |
| Eight-membered or larger ring among valid independent argmax decodes | 628/885 (71.0%) | 612/877 (69.8%) |
| Training seconds for 128 updates | 946.6 | 922.0 |

The pattern arm selected 50 of 138 available ring targets and 38 of 70 amide candidates over the 32 scheduled batches of 16 priors. It protected 159 rings. The median pre-clip gradient norm on scheduled updates was 2.24, with a maximum of 3.99 and zero clipped updates. The mean raw loss components per scheduled batch were 0.557 for aromatic positive cross-entropy, 0.294 for amide positive cross-entropy, 0.0656 for offsite aromatic probability mass, 0.0444 for partial-ring probability and 0.0040 for blocked C–O double-bond probability. The algorithm weights these components separately. Timing is a single sequential measurement on a laptop.

On categorical decoding, six aromatic molecules were gained and three were lost. Three of the gains preserved all slot identities, charges and adjacency. The same panel gained 39 valid outputs and lost 51. Among the 51 losses, the first observed failure was overvalence in 28, kekulization in 15 and disconnection in eight. The treatment avoided the large validity and scaffold losses seen with full-strength explicit supervision. Its valid aromatic yield was 1.76% and its valid amide yield was 4.30% of attempts, below the 8% and 12% training floors. This single short comparison leaves the magnitude and persistence of any motif gain uncertain. [Paired chemical failure categories](../../../outputs/runs/vae-aromatic-mechanism-20261121a/pattern-20261122a/first-chemical-failure.json)

```sh
python -m experiments.vae.training_data_eda.teacher_motif_training_pilot \
    --checkpoint "$CKPT" --reference "$TEACHER" \
    --config experiments/vae/epoch48-seven-round-motifs.json \
    --train-panel "$TRAIN" --heldout-panel "$HELDOUT" \
    --completion-arm treatment --completion-pattern \
    --control-summary "$OUT/control/summary.json" --output "$PATTERN/treatment"
python -m experiments.vae.training_data_eda.compare_completion_pilot "$PATTERN" --control-dir "$OUT/control"
```

### Independent evaluation on 8,192 more Gaussian vectors

We froze the same RTX 4060 endpoints after update 128 and selected two new seeds before evaluation. Each of 8,192 independent Gaussian vectors was decoded both ways under both models. Categorical sampling used the same noise in each arm. All 32,768 raw outcomes, including invalid graphs, retain their atom slots, charges and bond labels. Inference took 199 seconds. The table reports yields across all 8,192 attempts in each column. Intervals use paired draw-to-draw variation for these fixed checkpoints and exclude variation from retraining. [Results, paired gains and losses, and endpoint fingerprints](../../../outputs/runs/vae-aromatic-mechanism-20261121a/pattern-fresh8192-20261122a/summary.json)

| Raw outcome | Current objective | Pattern treatment | Change in percentage points (approximate 95% interval) |
|---|---:|---:|---:|
| Categorical valid | 5,254 | 5,152 | −1.245 (−1.977 to −0.513) |
| Categorical valid aromatic | 135 | 155 | +0.244 (+0.053 to +0.435) |
| Categorical valid amide | 311 | 302 | −0.110 (−0.350 to +0.131) |
| Argmax valid | 7,076 | 7,046 | −0.366 (−0.981 to +0.248) |
| Argmax valid aromatic | 183 | 228 | +0.549 (+0.326 to +0.772) |
| Argmax valid amide | 408 | 420 | +0.146 (−0.062 to +0.355) |

The pattern model gained 42 categorical aromatic molecules and lost 22. It gained 418 valid molecules and lost 520. Categorical draws with a raw aromatic bond that failed validity rose from 434 to 725. Unique nonempty scaffolds among valid categorical draws fell from 5,113 to 5,007. Rings of eight or more atoms still occurred in 3,657/5,152 valid categorical draws (71.0%), compared to 3,675/5,254 (69.9%) for control. The larger panel supports a modest rise in valid aromatic yield accompanied by a larger absolute loss of valid molecules. The amide change remains inconclusive. The interval formatter was corrected after inference and the result was rebuilt from the four saved graph files, verifying all paired draw IDs.

```sh
python -m experiments.vae.training_data_eda.compare_frozen_motif_endpoints \
    --control "$OUT/control" --treatment "$PATTERN/treatment" --output "$FRESH8192" \
    --latent-seed 2026112291 --categorical-seed 2026112292 --draws 8192 --batch 16
```

## Teaching neighborhoods of motifs the decoder already generates

Two further matched V100 tests checked whether the ring-pattern effect persisted and whether releasing conflicting frozen-teacher bond labels helped. After 512 updates, DTU job **30207795** produced 56 valid aromatic molecules on 4,096 raw categorical draws, versus 62 for the original loss. The pattern arm used 787.6 seconds in training and the original used 782.0. In job **30210075**, we retained bond-existence supervision and selectively removed the teacher's conditional SINGLE target at chemically verified six-C/N ring sites. The mask raised aromatic yield from 56 to 68 and amides from 175 to 207 compared with pattern-only, but total validity fell from 2,737 to 2,558. The original loss on the identical samples produced 2,759 valid, 62 aromatic and 185 amide molecules. The ring-label conflict is measurable; removing it at this weight cost substantially more valid samples than it gained in either motif. Both interactive jobs finished. [Pattern-only V100 comparison](../../../outputs/runs/vae-e48-r7-complete-20261122a/pattern-512-v100) · [Selective teacher comparison and raw samples](../../../outputs/runs/vae-e48-r7-complete-20261122a/ringteacher-512-v100)

A fourth local treatment started from the same seven-round checkpoint, used the original motif score and added supervision on latent neighbors of valid generated centers. When a center already contains an aromatic ring or amide, the decoder is asked to retain that motif's atom, charge and bond labels and its local connectivity on a correlated Gaussian draw. The correlation was 0.995. Each scheduled batch selected at most two centers per motif. The auxiliary weight was 3.0 inside the existing outer weight of 0.30, and the loss was divided by the full scheduled batch size. This treatment used neither of the completion losses above. [Implementation](../../../src/latentmol/guidance/motif_neighbors.py) · [Source, data and checkpoint fingerprints](../../../outputs/runs/vae-aromatic-mechanism-20261121a/neighbors-20261122a/pilot-fingerprints.json)

Across 128 matched updates, 29 of the 32 scheduled batches supplied natural motifs. Among 512 prior draws, 53 centers supplied 19 aromatic and 39 amide sites. The median pre-clip gradient norm was 2.41 on scheduled updates, with two exceeding 5. The training loop took 923.5 seconds, versus 946.6 for the saved control. On the first independent 1,024 categorical draws, validity changed from 642 to 651, valid aromatics from 15 to 17, and valid amides from 43 to 65. Exact held-out reconstruction stayed at 116/128. [Training and paired 1,024-draw results](../../../outputs/runs/vae-aromatic-mechanism-20261121a/neighbors-20261122a/paired-summary.json)

A separate RTX 4060 timing run used the production subset size of 48 prior decodes and timed only the extra work from neighbor guidance. After three warm-ups, 32 measured batches selected 35 aromatic and 54 amide sites on 81 centers. The extra chemical inspection, correlated decode and backward pass took a mean of 0.505 seconds per scheduled batch (median 0.491 seconds), equivalent to 0.126 seconds per optimizer update at the every-fourth-update cadence. The center forward pass already serves the original loss. Full V100 allocation throughput requires its own timing. [Measurements and per-batch site counts](../../../outputs/runs/vae-aromatic-mechanism-20261121a/neighbors-20261122a/production-shape-overhead.json)

We evaluated the frozen treatment on the same 8,192 larger-panel codes and categorical seed used above, reusing the saved control graphs. The intervals describe variation across these paired codes, conditional on the trained endpoints. [Larger-panel results, ring sizes and scaffold counts](../../../outputs/runs/vae-aromatic-mechanism-20261121a/neighbors-fresh8192-20261122a/summary.json)

| Outcome on 8,192 draws | Current objective | Natural-neighbor treatment | Change in percentage points (approximate 95% interval) |
|---|---:|---:|---:|
| Categorical valid | 5,254 | 5,285 | +0.378 (−0.359 to +1.115) |
| Categorical valid aromatic | 135 | 156 | +0.256 (+0.052 to +0.461) |
| Categorical valid amide | 311 | 487 | +2.148 (+1.744 to +2.553) |
| Argmax valid | 7,076 | 7,126 | +0.610 (+0.017 to +1.204) |
| Argmax valid aromatic | 183 | 232 | +0.598 (+0.381 to +0.816) |
| Argmax valid amide | 408 | 662 | +3.101 (+2.690 to +3.511) |

The paired categorical draws gained 47 valid aromatics and lost 26. They gained 233 valid amides and lost 57. Distinct valid nonempty scaffolds increased from 5,113 to 5,130. The proportion of valid categorical outputs with an eight-membered or larger ring rose from 69.9% to 71.3%, and the number of invalid draws containing an explicit aromatic bond rose from 434 to 592. These short-run results support testing persistence under a longer training allocation while following both adverse readouts. Finding the best weight and evaluating binding would require separate comparisons.

```sh
python -m experiments.vae.training_data_eda.teacher_motif_training_pilot \
    --checkpoint "$CKPT" --reference "$TEACHER" \
    --config experiments/vae/epoch48-seven-round-motifs.json \
    --train-panel "$TRAIN" --heldout-panel "$HELDOUT" \
    --completion-arm treatment --neighbor-weight 3 \
    --control-summary "$OUT/control/summary.json" --output "$NEIGHBORS/treatment"
python -m experiments.vae.training_data_eda.compare_completion_pilot "$NEIGHBORS" --control-dir "$OUT/control"
python -m experiments.vae.training_data_eda.compare_frozen_motif_endpoints \
    --control "$OUT/control" --treatment "$NEIGHBORS/treatment" --saved-control "$FRESH8192" \
    --output "$NEIGHBOR8192" --latent-seed 2026112291 --categorical-seed 2026112292 \
    --draws 8192 --batch 16
python -m experiments.vae.benchmark_functional_groups --mode neighbor-overhead \
    --checkpoint "$CKPT" --config experiments/vae/epoch48-seven-round-motifs.json \
    --draws 48 --panels 32 --every 4 --seed 2026112295 --weight 0.3 \
    --aromatic-floor 0.08 --amide-floor 0.12 --output "$NEIGHBORS/production-shape-overhead.json"
```
## Corpus-frequency and motif-batch replay at the first-leg endpoint

RDKit reread 50,000 uniform real training rows from the 60M view. Aromatic rings occurred in 43,528 molecules (87.056%), and the project's broad amide definition matched 23,994 (47.988%). These targets are molecule-level prevalence estimates. Counting individual rings whose every perimeter bond RDKit perceived as aromatic gave 20,519/34,387 five-membered rings (59.67%) and 79,117/101,947 six-membered rings (77.61%). Saturated and partially unsaturated rings contribute to the remaining examples. [Definitions and source sample](generation-motifs.md) · [Audit results](../../../outputs/runs/vae-dataset-motif-floors-20261122a/summary.json)

The completed seven-round first leg recorded 5,526 scheduled batches of 48 draws through update 597604. We pooled five consecutive logged batches as an approximation to 240 draws. We omitted one trailing batch from the 240-draw readout. The 48- and 240-draw groups with zero aromatic molecules numbered 2,832/5,526 (51.25%) and 55/1,105 (4.98%). Batches with zero amides numbered 759/5,526 (13.74%) and 0/1,105. A fixed-checkpoint cross-check using the existing 8,192 categorical samples gave 76/170 empty aromatic groups at 48 and 0/34 at 240 for the control. The neighbor-supervised checkpoint gave 61/170 and 0/34. Both fixed-checkpoint comparisons omit the same 32 trailing samples. These measurements establish how often there is a positive example for the score estimator. The larger batch's effect on learned generation requires training.

At fixed 240-draw grouping, replacing the 8%/12% floors with the corpus frequencies increased the mean summed absolute leave-one-out score coefficients **13.84-fold** for aromatics and **5.83-fold** for amides. The calculation uses the existing finite-batch formula and precedes multiplication by the graph log-probability gradients. Weighting, gradient direction and AdamW determine the eventual parameter change. [Replay code](dataset_motif_floor_audit.py)

```sh
python experiments/vae/training_data_eda/dataset_motif_floor_audit.py \
  --panel outputs/runs/vae-training-data-eda-20261120a/inputs/panel.tsv.gz \
  --training-log outputs/runs/vae-dataset-motif-floors-20261122a/first-leg-training.jsonl \
  --out "$NEW_RESULT_JSON"
```

### Testing the corpus percentages in the trainer

The [experimental recipe](../epoch48-seven-round-corpus-motifs.json) sets the floors to 0.87056 and 0.47988. Each measures the fraction of **all attempted categorical draws** that are valid and have the motif. Consequently overall validity must reach at least 87.056% before the aromatic floor is attainable. The real-data audit had valid molecules in every row. During training we also report the number of valid draws, which permits a separate comparison of motif prevalence among valid products.

On every fourth update the frequency estimator uses the 240 independent priors already assigned to that optimizer step. A detached pass draws molecules in groups of 48, computes a single leave-one-out estimate across the 240, and replays the corresponding decoder dropout masks during the regular backward pass. The fixed neighbor budget still uses the first 48 and up to two sites per motif. The outer weight remains 0.30. Raising the target by itself changes the gradient strength. Neither the generation sampler nor the amount of real reconstruction training changes. [Estimator](../../../src/latentmol/guidance/functional_groups.py) · [Training integration](../../../src/latentmol/training/projected_continuation.py)

The [local paired update test](corpus_motif_update_pilot.py) uses the actual saved optimizer and decoder state at update 597604. Each arm trains on the same 240 real panel molecules and 240 Gaussian priors per update, with 48-row microbatches and identical base random streams. The control uses the existing 8%/12% thresholds and the same neighbor term as the treatment. The two-update smoke includes one scheduled motif update and evaluates 1,024 fresh categorical draws and 128 held-out reconstructions. Sustained effects require a longer matched training comparison.

Both RTX 4060 forks completed. Their first, unscheduled updates agreed on all logged metrics, and their ending training RNG digests matched. At the scheduled update the 48-draw control saw one valid aromatic and one valid amide. The 240-draw score saw four and eight, respectively. Each arm had the same one aromatic and one amide neighbor site. The full pre-clipping parameter gradient norm was 1.295 for control and 1.312 for treatment. That scheduled training step took 114 seconds and 176 seconds respectively on this laptop in one sequential measurement. The unscheduled steps took 119 and 140 seconds, which illustrates the timing variability. V100 throughput for this recipe is unmeasured.

On the fresh categorical panel, both models returned 15 valid aromatics and 36 valid amides. Overall validity was 679/1,024 for control and 677/1,024 for treatment. Both reconstructed 118/128 held-out molecules exactly. This run establishes that the stronger guidance reaches an actual production-shaped update. It supplies one training exposure to the changed objective. [Raw panels and training steps](../../../outputs/runs/vae-e48-corpus-floors-20261122a/update-pilot) · [Summary with configuration, random streams and checkpoint digest](../../../outputs/runs/vae-e48-corpus-floors-20261122a/update-pilot/summary.json) · [Separate frozen-gradient comparison](pooled-motif-gradient.md)

```sh
python -m experiments.vae.training_data_eda.corpus_motif_update_pilot \
  --checkpoint "$LEG1_CHECKPOINT" --receipt "$LEG1_RECEIPT" --teacher "$TEACHER" \
  --config experiments/vae/epoch48-seven-round-corpus-motifs.json \
  --control-config experiments/vae/epoch48-seven-round-motif-followon.json \
  --real-panel outputs/runs/vae-training-data-eda-20261120a/inputs/panel.tsv.gz \
  --heldout outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json \
  --updates 2 --output "$NEW_PILOT_DIRECTORY"
```
