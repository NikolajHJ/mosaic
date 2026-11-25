# Can a shared ring decision increase valid aromatic yield?

25 November 2026. All experiments use an independent copy of epoch-48 checkpoint 623116. The screen, frozen-model pilot and matched training pilot use different evaluation seeds. Model outputs retain invalid draws in their denominators.

## Experiments

- **Eligibility screen.** On 256 fresh raw argmax outputs (seed 2026112531), 228 were valid and 16 contained an aromatic ring. Relabeling existing ring bonds alone yielded a valid aromatic molecule in 66 additional graphs. This screen includes fused or partly unsaturated rings that the learned head deliberately excludes. [Raw data and settings](../../../outputs/runs/vae-shared-ring-20261125a/eligibility-623116/summary.json)
- **Frozen VAE.** Fit the head for 128 batches of 64 real training molecules and keep the VAE frozen. Evaluate 1,024 Gaussian vectors with both argmax and categorical decoding and 512 held-out reconstruction targets. Seed 2026112532. [Runner](frozen_ring_readout.py) · [Results](../../../outputs/runs/vae-shared-ring-20261125a/frozen-readout/summary.json)
- **Matched V100 comparison.** Train two copies of 623116 for 128 updates on the same 8,192 real graphs and matched 1:1 prior draws. One uses the current training objectives, and one disables contradictory single-bond supervision on verified ring sites. The control reproduces the objectives at a pilot batch size of 64 and microbatch size of 16. Train the small ring head separately on the same real rows as the masked arm. Evaluate that exact VAE both with and without the head. The endpoints share 4,096 fresh Gaussian vectors, categorical randomness and 256 held-out graphs. The comparison also reports the effect of converting every eligible ring. [Training and CLI](shared_ring_pilot.py) · [Evaluation](shared_ring_eval.py)

The head sees element, charge and off-ring substituent features. Its inputs hide the six perimeter bond types. Training uses class-balanced supervision, so its scores have not been calibrated to real-world prevalence. It considers at most one isolated neutral six-membered C/N ring per graph and requires both complete assignments to pass chemical checks. A single draw sets all six bonds on an accepted conversion. Invalid starting graphs stay in the output. Reconstruction uses the same gate decision on the model output before examining the true target. The differentiable mixture is conditional on a selected cycle. This pilot measures generation and reconstruction. [Ring selection, head and relaxed readout](shared_ring_gate.py)

The primary bond-only comparison is control versus selective teacher masking. Both produce and sample the original per-edge bond probabilities. A separate directional probe uses those same logits and credits any of three verified encodings for a complete six-membered aromatic ring. The assigned probability is the sum of the probabilities for the all-aromatic labeling and the two alternating single/double labelings. It measures how a small update to the existing final bond layer affects target rings and off-site bonds. [Bond-logit probe](aromatic_encodings_probe.py)

The larger probe used 256 prior vectors to select a direction and 256 independent vectors to evaluate it, giving 16 training rings and 19 fresh rings. For equal-sized final-layer steps, crediting all three encodings raised the mean directional change in complete-pattern log probability from 48.58 to 52.67 across fresh rings, with medians 45.98 and 45.53. Both objectives also increased aromatic-vs-single and double-vs-single logits on all 903 unrelated single bonds examined. This was a local derivative measurement. It supports testing ways to confine bond updates before using this auxiliary in full training. [Independent replay](../../../outputs/runs/shared-ring-pilot-20261125a/three-forms-bond-only-replay/summary.json)

DTU job **30363149** used the standard shared V100 interactive allocation. The first allocation **30363148** exited in device-route parsing before starting any model. The replacement accepted DTU's assigned device in the comma-separated task route. The pinned model and production worker are separate from the pilot outputs.

## Matched V100 results

Job 30363149 completed on 25 November. The two arms had identical final prior, CPU and CUDA RNG digests. Selective masking removed 2,501 conflicting bond-type targets. The head used 5,494 positive and 275 negative ring sites from the same 8,192 real training molecules.

| Metric | Starting VAE | Control after 128 updates | Teacher mask after 128 updates | Teacher mask with separate ring readout |
|---|---:|---:|---:|---:|
| Valid categorical draws / 4,096 | 2,895 | 2,718 | 2,842 | 2,842 |
| Distinct valid aromatics, categorical / 4,096 | 138 | 101 | 106 | 167 |
| Distinct valid aromatics, argmax / 4,096 | 192 | 152 | 154 | 336 |
| Distinct nonempty scaffolds, categorical | 2,791 | 2,652 | 2,773 | 2,774 |
| Exact real reconstructions / 256 | 242 | 239 | 237 | 234 |
| Exact among 45 saturated-ring targets | 43 | 43 | 42 | 39 |

Compared with control, teacher masking gained 15 and lost 10 aromatic categorical draws. It gained 263 and lost 139 valid draws. The difference in aromatic yield is small for one short training seed, and both trained bond-only arms finished below the start. With deterministic decoding the extra ring readout matched blanket conversion in aromatic yield. Its gain comes from explicit changes to the output graph. [Full summary](../../../outputs/runs/vae-shared-ring-20261125a/matched-v100/summary.json) · [Raw outputs and allocation](../../../outputs/runs/vae-shared-ring-20261125a/matched-v100/)

## Frozen-model results

The head trained on 5,466 aromatic and 285 saturated sites among 8,192 training graphs. On 1,024 fresh argmax decodes it increased distinct valid aromatic molecules from 35 to 104 while preserving all 932 valid outputs. Converting every eligible ring gave the same 104. On categorical decodes, the learned head gave 50 distinct aromatics versus 27 for the original and 63 for blanket conversion, with 719 valid in each case. Exact held-out reconstructions fell from 480/512 to 469/512. All 11 losses came from the 78 target molecules with an isolated saturated six-membered C/N ring.

When we hid the perimeter labels on real held-out rings, the scores ranked aromatic sites above saturated sites with ROC AUC 0.778. At the default 0.5 threshold, however, the head called all 354 aromatic sites and all 11 saturated sites aromatic. Only sites passing the strict two-endpoint check enter this classification test, while the reconstruction denominator includes every molecule. [Frozen comparison](../../../outputs/runs/vae-shared-ring-20261125a/frozen-readout/summary.json) · [Held-out classification](../../../outputs/runs/vae-shared-ring-20261125a/frozen-readout/heldout-ring-classification.json)

We then set a conservative veto using 8,192 unused training molecules. A ring had to score strictly above 0.562299 as well as pass the existing decision. Of 303 eligible saturated sites in this calibration set, 30 passed. At the frozen endpoint the veto gave 63 distinct valid aromatic argmax outputs per 1,024 draws and 33 with categorical sampling. It retained the original 480/512 exact reconstructions, including 75/78 exact among molecules with saturated rings. These gains are due to a separate readout applied after the original bond predictions. [Post-hoc calibration and results](../../../outputs/runs/vae-shared-ring-20261125a/frozen-readout/calibrated-veto-10pct.json)

On six freshly generated eligible rings, holding the selected cycle fixed, the median norm of the latent gradient of the mean perimeter aromatic probability was 0.0235 in the base decoder and 0.301 after the shared readout. This checks a local derivative. Evaluating optimization progress or binding scores would require a separate experiment. [Real-latent gradient measurements](../../../outputs/runs/vae-shared-ring-20261125a/frozen-readout/latent-gradient-panel.json)

## Bond-only training on aromatic rings and amides

These follow-up runs use the ordinary VAE decoder, argmax and categorical sampler. All interventions act during training. DTU job **30363301** compares four forks from checkpoint 623116. Every fork uses the same 8,192 real molecules and base-prior draws across 128 updates of 64 real and 64 prior points. The three treatment arms also inspect 32 auxiliary Gaussian draws every other update. The control follows the current production objectives at the smaller pilot batch size. Separate objectives are:

- **Ring reward.** On a chemically verified ring, maximize the summed probabilities of three equivalent complete assignments, conditional on the six edges existing. Use the mean negative log probability per edge and an outer weight of 0.1.
- **Ring reward with protection.** Use the same reward and additionally penalize the difference from the frozen starting decoder on this latent. Preserve the probabilities of bond types on existing bonds outside the target ring and preserve bond presence on starting bonds in the selected molecule. Use distillation temperature 2, preservation weight 8 inside the ring coefficient. An auxiliary microbatch with no verified ring gives zero.
- **Amide-specific supervision.** Select a C–O single bond whose change to C=O yields a strictly valid amide in the presence of its neighboring nitrogen. Raise its double-bond odds relative to a matched incompatible C–O site and penalize upward drift on blocked sites. Use an outer weight of 0.1. Ring reward is off in this arm.

All arms use the same 4,096 fresh latent draws, 256 held-out molecules and an independent panel of 1,024 starting latents for tracking verified rings. The amide definition includes lactams, ureas and carbamates. Job inputs and model checks are in the [allocation receipt](../../../outputs/runs/vae-protected-ring-20261125a/matched-v100/launch.json) and [plan](../../../outputs/runs/vae-protected-ring-20261125a/matched-v100/plan.json). [Matched runner](protected_ring_pilot.py) · [Ring objective](protected_ring_objective.py) · [Amide objective](protected_amide_objective.py)

In a separate local readout-only study, 22 equal-length updates on chemically verified C–O sites produced 55 valid amides in 512 fresh argmax decodes, up from 46, with 460 valid products at both endpoints. An unprotected bond reward produced 90 valid amides and 446 valid products. On categorical draws, the protected arm gave 43 amides and 344 valid molecules versus 36 and 363 at the start. Among 1,398 incompatible C–O bonds the unprotected objective produced 23 double bonds and the protected objective produced one. These local endpoints and seeds differ from the full-VAE forks. [Readout-only amide results](../../../outputs/runs/shared-ring-pilot-20261125a/amide-bonds-only/summary.json)

The local ring readout test used 16 small steps. On 160 independent latent points with 15 verified rings, weight 8 brought average off-site total variation from 0.00965 to 0.00678 and hard changes to originally present off-ring bonds from 52 to 33 out of 3,769. Mean ring log-probability gain was +5.25 versus +5.34 without protection. Both endpoints yielded seven valid aromatics, compared with six initially. [Exact readout results](../../../outputs/runs/vae-shared-ring-20261125a/protected-readout-target-graphs-20261125.json)

Job 30363301 finished all four arms on 25 November. Each generation column below counts valid molecules with the stated property per 4,096 categorical draws, retaining invalid attempts in the denominator. Exact reconstruction uses the same 256 real molecules.

| Arm | Valid | Aromatic | Amide | Exact reconstruction |
|---|---:|---:|---:|---:|
| Starting VAE | 2,895 | 138 | 340 | 242 |
| Control | 2,718 | 101 | 283 | 239 |
| Ring reward | 2,513 | 80 | 225 | 239 |
| Ring reward with protection | 2,723 | 92 | 277 | 239 |
| Amide bond loss | 2,834 | 119 | 426 | 240 |

The amide arm gained 193 and lost 50 amides relative to control on matched categorical draws. It also yielded 604 amides on argmax versus 366 for control. Neither arm generated a false-positive amide among the 139 held-out nonamide reconstruction targets. On the independent panel of 50 eligible rings, both ring objectives raised the summed probability of a complete aromatic assignment, but zero of the 50 rings reached a complete hard assignment in either fork. Protection reduced changes on the 908 originally present off-site bonds from ten to six. The chemical composition of generated products shifted in other ways too. The amide arm yielded 181 more categorical draws containing a three-membered ring than control. This single-seed pilot provides no estimate of between-run variation. [Matched results](../../../outputs/runs/vae-protected-ring-20261125a/matched-v100/summary.json) · [Held-out amides](../../../outputs/runs/vae-protected-ring-20261125a/matched-v100/heldout-amide-subsets.json)

### Which functional groups improved?

We audited a 50,000-row uniform sample from the 60M training view and all raw endpoints. The reference column below weights the training frequencies at each heavy-atom count by the distribution among the amide arm's 2,834 valid draws. Fractional expectations are rounded. The totals count molecules, and a molecule can contain multiple functional groups. The cyclic urea definition includes cyclic ureides with a ring N–C(=O)–N center, including keto forms of hydantoins and pyrimidinediones. The five-/six-membered row requires a perceived ring of that size through that very carbonyl. The historical amide SMARTS has narrower aromatic atom coverage, so these flags need not be nested.

| Motif | Training sample / 50,000 | Start / 4,096 | Control / 4,096 | Amide arm / 4,096 | Size-matched reference for amide arm |
|---|---:|---:|---:|---:|---:|
| Cyclic urea, any ring size | 1,011 | 47 | 36 | 63 | 38 |
| Cyclic urea, five- or six-membered | 997 | 12 | 8 | 19 | 38 |
| Noncyclic urea | 1,704 | 19 | 15 | 22 | 121 |
| Lactam | 5,860 | 185 | 151 | 244 | 240 |
| Carbamate | 2,035 | 25 | 23 | 27 | 118 |
| Sulfonamide | 3,934 | 9 | 10 | 6 | 216 |
| Nitrile | 2,544 | 13 | 9 | 7 | 135 |

The size match accounts for the narrow generated heavy-atom distribution. It does not control for topology or scaffold. All gains above are measured on raw decoder outputs. In a separate editability count, 60 baseline categorical outputs lacking cyclic urea already had a ring N–C(OH)–N site where changing just the C–O single bond to a double produced a sanitized cyclic urea on the same site. These hypothetical products are excluded from raw yields. The audit also reports the subset of these edits giving a five- or six-membered cyclic urea. [Family definitions and audit CLI](motif_family_audit.py) · [Full family, precursor and size-matched counts](../../../outputs/runs/vae-protected-ring-20261125a/motif-families-final.json)

### Do native bond scores identify the site?

At frozen checkpoint 623116, 35 of 2,048 independent Gaussian starts had a verified cyclic urea precursor. Ranking by the native C=O versus C–O logit margin, six of the higher-scoring 18 sites completed at least once under four radius-one latent perturbations, versus zero of the lower 17. With four categorical samples at the original latent the corresponding counts were seven and one. A success required chemical validity and a cyclic urea on the original four atom slots. For broader amides, 14/40 high-ranked sites versus 0/39 low-ranked sites completed in nearby decodes. Aromatic completion was too rare to establish enrichment (1/16 versus 0/16). These tests measure where the generator can complete a motif. [Cyclic urea ranking](../../../outputs/runs/vae-raw-cyclic-urea-ranking-20261125a/summary.json) · [Amide and ring ranking](../../../outputs/runs/vae-raw-prior-site-ranking-20261125b/summary.json)

Recovering hidden chemistry from a real input is a different question. On a disjoint training-corpus supplement, the native score ranked 334 deliberately reduced amide sites against 31 genuine single-bond sites with AUROC 0.491. The 512-molecule holdout yielded 150 scoreable positive sites and zero comparable negatives. For held-out aromatic rings with the whole perimeter erased, the native complete-pattern score achieved AUROC 0.544 across 417 aromatic and 12 saturated sites. Future local supervision should distinguish chemical feasibility, spontaneous completion on the prior, and evidence about the identity of a real molecule. [Blinded amide test](../../../outputs/runs/shared-ring-pilot-20261125a/amide-site-localization-expanded/summary.json) · [Blinded ring test](../../../outputs/runs/vae-shared-ring-20261125a/ring-site-localization-623116.json)
