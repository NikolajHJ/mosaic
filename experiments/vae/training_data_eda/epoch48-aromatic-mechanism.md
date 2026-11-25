# Why random draws lose aromatic rings and amides

## Checkpoint and experiment

We froze the seven-round epoch-48 VAE at update **580498**, after the first 5,000 updates of this leg (1.2M real examples). Its SHA-256 is `1a870d1e86d6edfc03d6e61e32a27fad23e7bcc0301b4b755773dd270963a9a8`. Evaluations ran locally from a verified copy. The [source receipt](../../../outputs/runs/vae-aromatic-mechanism-20261121a/source-latest.json) identifies the training checkpoint. No experiment in this note updates its weights.

The [earlier audit](generation-motifs.md) measured aromatic rings in 87.1% of a uniform sample of 50,000 graphs from the actual 60M training view. Amides occurred in 48.0%. The graph schema excludes some donor heteroaromatics with an unrepresented pyrrolic hydrogen. The remaining data is still rich in both motifs.

We also checked the part of the real-data shuffle covered by this particular checkpoint. Replaying the trainer's CPU permutation with seed 11060199 and taking the first 1,200,000 rows intersected the uniform 50k audit at 981 molecules. Of these **849/981 (86.5%) were aromatic** and **471/981 (48.0%) had amides**. Two local Torch versions agreed on every row of the prefix, verified by a digest. This is a sample from a replay of the configured order. The live trainer does not record per-example identities. [Replay counts and digest](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/shuffle-prefix-2.6.0.json)

## Graph structures under the same decoder

| Count in 256 attempted graphs | Real held-out molecules | Reconstructions of those molecules | Independent Gaussian inputs |
|---|---:|---:|---:|
| Strictly valid | 256 | 246 | 223 |
| Valid with an aromatic ring | 234 | 225 | 7 |
| Valid with an amide | 110 | 106 | 26 |

Of the random graphs, 254 contained a cycle, 199 had a five- or six-membered ring and 191 had at least one all-single five- or six-membered ring. Nitrogen was present in 246 graphs, oxygen in 243, and a carbonyl (C=O) in only 33. The model frequently supplies the atoms and neighbors needed for these motifs, then assigns saturated bonds.

We held elements, formal charges and the entire edge set fixed. Among 120 valid nonaromatic molecules with an eligible neutral six-membered carbon/nitrogen ring, relabeling only its ring bonds produced a valid aromatic molecule in **69**. Among 112 valid nonamides with a carbon already attached to nitrogen and oxygen, changing one C–O bond to C=O produced a valid amide in **35**. These are diagnostic interventions; all generation metrics use the original graph. The aromatic class was often far from winning: on 144 nonaromatic outputs with a qualifying ring the median best-ring mean winning-logit minus aromatic-logit gap was 9.82. [Structural diagnostics and strict validity](../../../outputs/runs/vae-aromatic-mechanism-20261121a/decoder/summary.json)

![Raw generated example and bond-only intervention for each motif](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/bond-order-examples.png)

The ring sizes also differ substantially. Among the 223 strictly valid Gaussian outputs, 160 (71.7%) contained a three-membered ring, 165 (74.0%) contained a ring of at least eight atoms, and 67 (30.0%) contained a ring of at least 12. Applying the same RDKit ring definition to the real 50k sample and matching heavy-atom counts predicts 5.3%, 0.51% and 0.079%, respectively. These indicators overlap. The teacher already generated rings of at least eight atoms in 116/201 valid outputs on the same latent inputs. This separate discrepancy motivates scoring ring-size distributions alongside motif yield and validity. Keeping the teacher's edge-existence targets can preserve its ring topology bias. [Paired ring sizes](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/perceived-ring-sizes.json) · [Atom-count-matched real reference](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/size-matched-ring-sizes.json)

The [valence projection](../../../src/latentmol/guidance/valence_projection.py) only selects edits that reduce excess valence. A saturated, connected macrocycle with allowable valences incurs no projection penalty for its ring size. Thus a high validity rate and an unusual ring-size distribution can persist together.

## Where the training signal comes from

The frozen epoch-46 teacher supplies targets for atom occupancy, element, charge and bond predictions on random vectors. Its strictly eligible Gaussian decodes assigned the aromatic class to **288/37,059 present bonds (0.78%)** against **23,491/56,288 (41.7%)** on 2,048 real training examples. Reweighting the real rows to the teacher's eligible sizes gave 40.5%. On 256 identical Gaussian inputs, the current VAE preserved 4,623 of 4,737 teacher-positive bonds at the same slots and with the same type on the 201 teacher-eligible graphs. Shared ancestry also contributes to that agreement. [Teacher labels and sampled training data](../../../outputs/runs/vae-aromatic-mechanism-20261121a/objective/mechanism.json) · [Paired student–teacher comparison](../../../outputs/runs/vae-aromatic-mechanism-20261121a/objective/teacher_on_student.json)

We added one scalar to all aromatic output logits in eval mode and differentiated the actual conditional teacher cross-entropy at zero. The derivative averaged **+0.00419** across eight 16-vector batches and was positive in seven. Minimization therefore pushes that score down in this intervention. Pairs on the student's existing cycles accounted for +0.00398 of the net derivative. Weighted valence projection averaged +0.00000018 in the same direction. Centered differences agreed with the analytic decomposition. This isolates one output direction, not the combined parameter update. The aromatic deficit was observed by epoch 45 before this teacher was introduced. [Direct gradient check](../../../outputs/runs/vae-aromatic-mechanism-20261121a/objective/direct_gradient.json)

We decomposed the teacher's five-class bond cross-entropy into two exact terms, one for bond existence and one for bond type given existence. Keeping the original pair weights, valid-row weights and normalization reproduced the original loss to within 2.8e-17. Removing only the conditional type term left **+0.0000163** of the original **+0.00419** derivative. Among 384 potential edits on existing rings that succeeded in the fixed-topology test and had eligible teacher targets, the teacher requested 354 single bonds, 21 absent bonds and nine aromatic bonds. This identifies a specific conflict on the same slots that could form valid aromatics. [Directional ablation](../../../outputs/runs/vae-aromatic-mechanism-20261121a/objective/split_gradient.json) · [Teacher labels on recoverable rings](../../../outputs/runs/vae-aromatic-mechanism-20261121a/objective/recoverable_cycles_teacher.json)

In the seven-round job's first 5,000 updates, the explicit motif loss examined 60,000 of 1.2M fresh Gaussian decodes. Only 785 of those contained a valid aromatic and 2,268 contained a valid amide. Its aromatic estimator had a nonzero coefficient on 566 updates (11.3% of all updates). The current loss therefore has relatively few opportunities to reward aromatics even though the logged shortfall is positive. This observation does not measure the net effect of increasing its weight.

## Intervention in the latent space

On 160 held-out molecules we sampled one latent from each encoder posterior and one independent standard-normal latent. The input to the decoder was `sqrt(1-t) * posterior + sqrt(t) * independent_normal`. Each row uses the same two endpoints across all five conditions. Percentages here refer to the mixing coefficients squared. The two populations have different joint distributions even when their Euclidean norms are similar.

| Independent noise fraction | Strictly valid | Valid aromatic | Valid amide |
|---|---:|---:|---:|
| 0% | 143/160 | 130/160 | 64/160 |
| 25% | 94/160 | 46/160 | 21/160 |
| 50% | 133/160 | 12/160 | 14/160 |
| 75% | 139/160 | 4/160 | 10/160 |
| 100% | 136/160 | 3/160 | 14/160 |

At the halfway point all 133 valid molecules contained a ring. Before sanitization, the number of graphs with a cycle of explicit aromatic bond labels had already fallen from 145 at the starting endpoint to 15. The mean latent norm at those two points was 16.26 and 16.13. Rescaling each independent Gaussian vector to its paired posterior's norm yielded two valid aromatics instead of three. [Complete latent-path results](../../../outputs/runs/vae-aromatic-mechanism-20261121a/paths/summary.json)

We ranked latent coordinates using 352 separate reference molecules, without using aromatic labels. The rank measures how well a coordinate could distinguish source molecules given the variation between their codes and uncertainty within each posterior. We exchanged 32 out of 256 coordinate values between paired Gaussian and posterior samples on 96 evaluation molecules. The original posterior samples produced 83 valid aromatics. Replacing the highest-ranked coordinates gave one, and replacing the lowest-ranked coordinates gave 82. Transferring just the top 32 in the reverse direction raised valid aromatics from one to 13. The decoder relies on specific combinations of coordinates for this chemistry. These ranks do not identify 32 exclusively aromatic dimensions. [Coordinate interventions](../../../outputs/runs/vae-aromatic-mechanism-20261121a/paths/swap/summary.json)

Matching the displacement magnitude per molecule on the low-ranked coordinates still preserved 82/96 aromatic products. Swapping 32 random coordinates retained or produced 33. Independently shuffling all 256 coordinates across the 160 actual posterior samples reduced valid aromatics from 130 to one while preserving every coordinate's empirical distribution and leaving 136 valid molecules. [Paired transitions](../../../outputs/runs/vae-aromatic-mechanism-20261121a/paths/swap/paired-with-baselines.json) · [Latent path figure](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/latent-path.png)

The ranking proxy for coordinate (j) is one half of `log(total variance of z_j) - mean(log posterior variance of z_j given molecule)`. The top 32 accounted for 62.8% of the sum of this proxy across 256 coordinates. The ranking uses no aromatic labels. It expresses how informative a coordinate *could* be about the input, and was selected independently of the molecules used for the swap test.

This effect also appears when comparing the same number of heavy atoms. Of the 26 paired outputs that retained their atom count after the high-ranked swap, 24 were aromatic before it and zero afterward. Across all 96 pairs, 70 of the original 83 aromatic molecules turned into valid nonaromatics and 13 became invalid. [Paired atom-count controls](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/atom-count-controls.json)

## Which starting distributions helped?

We fit the following samplers to 2,048 uniformly selected *training* molecules drawn from the audited 50,000-row panel. Two independent sets of 256 draws shared the same base noise across all methods. The centered Gaussian adds the empirical posterior mean to independent unit noise. The diagonal and full Gaussian fits use both between-molecule variance and within-molecule posterior variance. A 32-component mixture uses k-means centers with empirical weights and a shared within-component covariance. The seeded conditions choose a real molecule and sample its learned diagonal posterior with its standard deviation multiplied by the stated factor. All outputs use raw argmax with fixed model weights.

| Sampler (512 attempts each) | Valid | Valid aromatic | Valid amide | Distinct valid aromatics absent from the 50k panel |
|---|---:|---:|---:|---:|
| Standard Gaussian | 449 | 12 | 35 | 12 |
| Centered Gaussian | 428 | 16 | 38 | 16 |
| Fitted diagonal Gaussian | 399 | 16 | 26 | 16 |
| Fitted full Gaussian | 367 | 11 | 36 | 11 |
| Fitted 32-component mixture | 351 | 13 | 21 | 13 |
| Sample around a real encoding, 1× posterior standard deviation | 377 | 316 | 157 | 30 |
| Sample around a real encoding, 1.5× | 239 | 173 | 86 | 100 |
| Sample around a real encoding, 2× | 199 | 49 | 38 | 48 |

At 1×, 339/512 attempts yielded exact copies of the chosen input. At 1.5×, 89 were exact copies. The 100 distinct aromatic products outside the panel spanned 90 distinct Murcko scaffolds. Of the aromatic outputs outside the panel, 78 had a different scaffold from their own seed. These results support a seeded initialization option for optimization and expose its validity cost. They do not establish the best multiplier outside these three settings. [Counts by sampler](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/paired-sampler-counts.json) · [Replicate-level results](../../../outputs/runs/vae-aromatic-mechanism-20261121a/sampler-roundtrip/summary.json)

We then checked these canonical identities against the full training set and the retained packed parent with an allocated one-core DTU scan (job 30167356, 32 seconds). Of the 100 seeded aromatic structures, three were present in the 60M training set, one occurred only in the larger parent, and 96 were absent from all 81,549,264 retained records. The 12 standard-Gaussian aromatic structures were all absent from the packed parent. The two methods therefore yielded **97 vs 12 distinct valid aromatic molecules outside the entire training set per 512 attempts**. The eight identity and split controls passed. Global chemical novelty, binding and downstream search performance were outside this comparison. [Membership records and checks](../../../outputs/runs/vae-aromatic-mechanism-20261121a/membership/result.json)

One emitted SMILES out of the 4,096 attempts could not be reparsed by the same RDKit version (2024.09.6) after initial graph sanitization. The original analysis stopped on that case. The completed analysis records the raw labels and string and includes the attempt as invalid. This happened in the 1.5× seeded condition. [All sampler outcomes](../../../outputs/runs/vae-aromatic-mechanism-20261121a/sampler-roundtrip/draws.jsonl.gz)

## Does removing the teacher's bond-type target fix generation?

On 21–22 November we trained two disposable copies of update 580498 for 128 updates each on a local RTX 4060. Both restored the same AdamW state, used the same 8,192 unique training molecules and consumed identical random streams. Each update used 64 real graphs and 64 fresh priors, with 16 teacher queries and a motif batch of 16 every fourth update. Production uses 240+240 and 48 motif draws on those updates. Both pilot arms kept the production learning-rate schedule, step scaling, dropout, reconstruction, KL, occupancy and projection. Only the bond term in the random-point teacher objective differed. Source and teacher weights stayed frozen. The same 1,024 fresh Gaussian inputs and 128 held-out reconstruction inputs were scored at all three endpoints.

| Endpoint | Valid / 1,024 | Valid aromatic / 1,024 | Valid amide / 1,024 | Distinct nonempty scaffolds among valid outputs | Exact reconstructions / 128 |
|---|---:|---:|---:|---:|---:|
| Shared starting checkpoint | 885 | 28 | 70 | 876 | 116 |
| Teacher copies bond presence and type | 866 | 24 | 50 | 858 | 116 |
| Teacher copies bond presence only | 887 | 30 | 40 | 871 | 116 |

Relative to control, the existence-only arm gained eight aromatic products and lost two on paired draws. It gained two amides and lost 12. The aromatic count exceeded the common starting point by two molecules. On valid outputs, the fraction with a ring of eight or more atoms stayed high (622/866 for control and 654/887 for existence-only). This single-seed pilot identifies a tradeoff. Removing the bond-type target alone is insufficient for the desired mix of motifs and ring sizes. [Matched endpoints, costs and RNG checks](../../../outputs/runs/vae-aromatic-mechanism-20261121a/teacher-train-pilot-a/summary.json) · [Ring sizes by endpoint](../../../outputs/runs/vae-aromatic-mechanism-20261121a/overview/pilot-ring-sizes.json)

For generating optimization starts with this checkpoint, the 1.5× seeded sampler has the strongest observed yield of distinct valid aromatics outside training among the two methods checked against the full corpus. For unseeded training, the next focused comparison would relax bond-type copying specifically on ring edges, keeping the existing supervision outside rings, and check whether the amide loss seen here persists. A later comparison could teach incomplete carbonyl and aromatic arrangements from the real training split. Use held-out data and fresh prior seeds for evaluation, reporting valid aromatic and amide yields, ring sizes, scaffold diversity and exact reconstruction separately.

## Reproducing the comparisons

Set `PYTHONPATH` to include the repository root and `src`. Use the checkpoint identified above. `HELDOUT` is the [512-row held-out PubChem packet](../../../outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json); it covers 15–31 heavy atoms. `TRAIN` is the [50k training sample](../../../outputs/runs/vae-training-data-eda-20261120a/inputs/panel.tsv.gz). Use a fresh `OUT` directory. The seeded sampler used a local RTX 4060 with PyTorch 2.11.0+cu128. Independent CPU results used PyTorch 2.6.0+cpu.

```sh
python experiments/vae/training_data_eda/decoder_motif_diagnostics.py --checkpoint "$CKPT" --panel "$HELDOUT" --output "$OUT/decoder"
python experiments/vae/training_data_eda/latent_motif_paths.py --checkpoint "$CKPT" --panel "$HELDOUT" --output "$OUT/paths"
python experiments/vae/training_data_eda/latent_motif_paths.py --checkpoint "$CKPT" --panel "$HELDOUT" --mode coordinate-swap --output "$OUT/paths/swap"
python experiments/vae/training_data_eda/teacher_motif_gradient.py --checkpoint "$CKPT" --teacher-graphs outputs/runs/vae-generation-motifs-20261120a/checkpoint-000325498.jsonl.gz --teacher-summary outputs/runs/vae-generation-motifs-20261120a/summary.json --output "$OUT/teacher-split.json"
python -m experiments.vae.training_data_eda.motif_sampling_interventions --checkpoint "$CKPT" --panel "$TRAIN" --out "$OUT/sampler-roundtrip" --device cuda
python -m experiments.vae.training_data_eda.summarize_motif_interventions "$OUT"
python -m experiments.vae.training_data_eda.teacher_motif_training_pilot --checkpoint "$CKPT" --reference "$EPOCH46_TEACHER" --config experiments/vae/epoch48-seven-round-motifs.json --train-panel "$TRAIN" --heldout-panel "$HELDOUT" --output "$OUT/teacher-train-pilot"
```
