# DeFoG versus structural chemistry guidance


The three optimization subprocesses and the rescoring subprocess returned success. The rescoring receipt was committed at 12:48:31 CET. Upon closing the interactive shell, LSF recorded an external termination signal at 12:48:40 CET. We verified the downloaded selection and score file hashes against the completed scientific receipt.
Can a fragmented 25-atom graph become a valid connected molecule during 500 Protenix updates, and stay valid at the endpoint? This pilot compares two guidance methods on the same BRD4 pocket, initial graphs and model keys. Each of seeds 17, 23 and 42 gets a single paired GPU allocation.

| Shared or changed | Setting |
|---|---|
| Shared | Frozen Protenix-Tiny, native weight 1, LR 60 for atom and bond probabilities, temperature 1, Stage 2 connected projection, 500 updates |
| Initial graph | A sparse random tree with four edges deleted. Both arms receive the identical initialization within seed. |
| DeFoG | One refreshed and detached soft prediction per update, probability MSE, constant weight 0.5 and teacher time 0.8 |
| Chemistry | Historical failure-aware atom-preserving repair loss at constant weight 1. The connected-backbone search proposes targets, with the original bounded beam and progress searches available. It gives zero loss when the input already qualifies. |
| Probability support | After initialization and each projected update, both arms mix in 0.0001 uniform mass per categorical row. This keeps the repair log likelihood defined at the same graph used by Protenix and DeFoG. It changes the original Stage 3 protocol slightly, so the DeFoG arm is rerun. |

The primary generation measures are the number of seeds finding a new sanitized, single-component raw argmax and the number ending with one. Also retain per-seed valid-state counts, distinct canonical identities, and the best valid hard Protenix contact score. A repaired training target never qualifies by itself. Empty scores stay missing. Adjacent observations describe retention along the same trajectory.

Weight 0.5 was selected from the earlier Stage 3 sweep. On the nine original seed/start cells it found a new valid molecule in all nine and ended valid in four. Its 2,700 valid states out of 4,509 compare with 58 for the earlier weight-0.1 soft-valence penalty, which also found a new molecule in all nine but had no valid endpoints. That penalty measures expected valence excess. The head-to-head pilot uses the structural repair loss instead.

The saved chemistry implementation comes from Mosaic research revision `fc65ee8235318a26986d1145e9088ad02bef2ac2`. The weight of 1 follows its sparse structural-repair protocol's initial weight. The pilot uses the probability-space update shared with DeFoG. Historical chemistry experiments optimized logits and used different budgets and learning rates.

**Earlier fragmented-start reference.** All three original DeFoG and valence-penalty runs found a new valid molecule. The table shows the same three seeds used in this pilot. The repeat row substitutes the recovered DeFoG run for the original seed-17 run. The three seeds are reused, and GPU models differed between some historical arms.

| Historical guidance, with connected projection | Valid connected observations | Valid endpoints | Mean best valid hard contact |
|---|---:|---:|---:|
| DeFoG 0.5, original | 754/1,503 | 1/3 | 0.448 |
| DeFoG 0.5, substituting the seed-17 repeat | 617/1,503 | 0/3 | 0.405 |
| Simple soft-valence penalty 0.1 | 13/1,503 | 0/3 | 0.287 |

The [paired plan](../../../results/hpc/20261118-defog-vs-structural-repair-01/plan.json) and [GPU receipts](../../../results/hpc/20261118-defog-vs-structural-repair-01/submission.json) identify jobs 30103605 (seed 17, L40S), 30103606 (seed 23, V100), and 30103607 (seed 42, A40). All three were accepted on 18 November. CPU job 30103616 collects results after their terminal statuses. Run `scripts/stage3_repair_pair_plan.py --output CAMPAIGN` to generate the paired settings. The collector `scripts/summarize_repair_pair.py CAMPAIGN` checks the initial tensors, native scorer, keys and hardware for each pair.
## Structural repair without Stage 2 projection

Three additional 500-update arms compare structural repair at weight 1 on its own with the original projected DeFoG arm for each fragmented seed. The runner uses Stage 2 mode `none`, retaining the shared simplex update and 0.0001 uniform interior mass. Each seed uses the same GPU queue as its original pair in a separate allocation. The projected repair arm supplies a within-method check of the effect of projection. On seed 17 that arm visited 217 distinct valid connected molecules and ended with five fragments.

[No-projection plan](../../../results/hpc/20261118-structural-repair-no-projection-01/plan.json) and [submission receipts](../../../results/hpc/20261118-structural-repair-no-projection-01/submission.json) for seeds 17, 23 and 42.

## 19 November · fresh confirmation seeds

Seeds 71 and 101 extend the fragmented-start comparison with two methods per seed. Each job runs structural repair at weight 1 without Stage 2 projection and DeFoG at weight 0.5 with connected projection and teacher time 0.8 on the same GPU. Both retain LR 60, 500 updates, the same seeded graph and Protenix key, and the shared interior mixture. Report distinct valid connected identities, best valid hard contact and endpoint validity by seed. At submission, both earlier seed-42 jobs were still pending. [Four-run plan](../../../results/hpc/20261119-defog-vs-repair-seeds71-101/plan.json) and [submission receipts](../../../results/hpc/20261119-defog-vs-repair-seeds71-101/submission.json).

## 21 November · completed five-seed comparison

At 12:24 CET, all ten primary runs had complete summaries and 501 consecutive saved states each. Each started with five components. Both methods found a sanitized, single-component raw argmax on every seed. The table orders each cell as DeFoG (0.5, connected projection) / custom structural repair (1, no Stage 2 projection). Scores use the hard Protenix contact on the common within-seed key.

| Seed | Best valid contact in 500 updates | Distinct valid SMILES | Final components | Best contact among first 20 distinct valid SMILES |
|---:|---:|---:|---:|---:|
| 17 | 0.335 / 0.347 | 47 / 63 | 4 / 1 | 0.332 / 0.340 |
| 23 | 0.404 / 0.352 | 440 / 54 | 1 / 2 | 0.383 / 0.352 |
| 42 | 0.391 / 0.359 | 36 / 53 | 4 / 3 | 0.391 / 0.333 |
| 71 | 0.363 / 0.423 | 101 / 54 | 6 / 1 | 0.317 / 0.423 |
| 101 | 0.507 / 0.349 | 241 / 59 | 6 / 3 | 0.300 / 0.349 |

The mean best valid score was 0.400 / 0.366 (three / two paired wins). Without seed 101 the means were 0.373 / 0.370. Valid-state counts were 869 / 291 out of 2,505 observations per method. Valid endpoints were 1/5 / 2/5. In the separate comparison holding projection on for both losses, the best valid scores for DeFoG / repair were 0.335 / 0.403 (17), 0.404 / 0.322 (23) and 0.391 / 0.280 (42).

The final column is a post hoc check on an equal number of early candidates. For each trajectory, read `states.jsonl.gz` in update order, retain the first occurrence of each valid canonical SMILES, take the first 20, and compute the maximum recorded hard score. The means were 0.345 / 0.359. The primary result uses the full 500-update budget. The secondary calculation adds no optimizer updates or model calls. Evidence comes from the two 18 November campaigns and the 19 November campaign linked above.

## 21 November · stronger unprojected repair and fresh Protenix keys

We reran the three original fragmented seeds at repair weight 10, with Stage 2 projection off, learning rate 60 and 500 updates. The plan reused the weight-1 configurations and the same sealed optimization source. The initial atom and bond tensors and optimization keys matched within every seed. The new runs and rescoring used one NVIDIA A100-PCIE-40GB under DTU allocation 30151731. Historical optimization trajectories used other GPU families. [Run plan](../../../results/hpc/20261121-repair10-fresh-keys-01/plan.json) · [Optimization results and record-setting steps](../../../results/hpc/20261121-repair10-fresh-keys-01/optimization-summary.json)

| Seed | Valid states / 501 | Distinct valid identities | Final components | Best valid hard contact | Best update |
|---:|---:|---:|---:|---:|---:|
| 17 | 328 | 14 | 1 | 0.295574 | 44 |
| 23 | 249 | 10 | 1 | 0.262930 | 22 |
| 42 | 242 | 13 | 1 | 0.304409 | 29 |

All three endpoints sanitized and were connected. On these same seeds both repair 1 and projected DeFoG ended valid once. Mean best valid hard scores were 0.288 for repair 10, 0.352 for repair 1 and 0.376 for projected DeFoG.

The selection was frozen before rescoring. For each of the 16 completed comparison runs we retained its five highest-scoring distinct valid canonical identities in their original ordered hard graph representation, then evaluated all 80 candidates on the same five unused keys (7001–7005) and on their original key. Preselected primary output is the mean fresh-key score of the original winner. The secondary output averages all five candidates within each run, then averages runs equally. All 400 fresh scores exactly equaled the corresponding same-device original-key replay, with maximum absolute replay-to-archive difference 8.94 × 10⁻⁷ across 80 original-key checks. Frozen model and context checks passed. These keys supply a sensitivity check within this fixed scorer and pocket. They contribute zero independent within-candidate score variation on these graphs.

Run the [shortlist rescoring script](scripts/rescore_repair_shortlists.py) with `CAMPAIGN ASSETS OUTPUT` on an allocated GPU using the original Stage 3 source and environment. [Frozen shortlist and keys](../../../results/hpc/20261121-repair10-fresh-keys-01/selection.json) · [All candidate scores](../../../results/hpc/20261121-repair10-fresh-keys-01/scores.csv) · [Native completion receipt](../../../results/hpc/20261121-repair10-fresh-keys-01/completion.json) · [Updated PDF](../../../docs/stage3_defog_report_draft.pdf)

## 21 November · DeFoG 0.5 without projection

We tested whether the selected teacher could assemble five fragments without Stage 2 graph operators. DTU A100 allocation 30152443 ran the existing sealed Stage 3 `run-bundle` command for seeds 17, 23 and 42. The prior was constant at 0.5 and teacher time 0.8. The native coefficient was 1, LR was 60, and each run completed 500 updates. The simplex normalization and uniform interior mass of 0.0001 were retained. Checked the shared configuration, exact initial tensors and keys, frozen model/teacher/context and 501 contiguous state observations per run. Each new run made 500 teacher queries. GPU type differed from the archived projected arms.

- Seed 17 visited zero valid connected states and finished with 6 components. 296 fragmented states passed sanitization.
- Seed 23 visited zero valid connected states and finished with 2 components. 265 fragmented states passed sanitization.
- Seed 42 visited zero valid connected states and finished with 7 components. 13 fragmented states passed sanitization.

Without projection, 1,501 of 1,503 graphs were fragmented. All 574 graphs that passed sanitization had multiple components. Seed 23 briefly reached a single component twice, but both graphs failed sanitization. With connected projection, all three seeds found valid molecules, accounting for 525 of 1,503 states (34.9%). Seed 23 also ended valid. All three best valid hard contact scores without projection are missing. There were no eligible molecules to add to the fresh-key shortlist. The corresponding projected best scores were 0.335, 0.404 and 0.391. [Run plan](../../../results/hpc/20261121-defog05-no-projection-01/plan.json) · [Per-seed counts and identity checks](../../../results/hpc/20261121-defog05-no-projection-01/comparison.json) · [Stage 3 report](../../../docs/stage3_defog_report_draft.pdf)
