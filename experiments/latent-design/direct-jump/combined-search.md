# Does adding Nesso-1 affinity to Protenix improve a 700-update BRD4 search?

The search optimizes 25 neutral atom probability rows and 300 unordered bond probability rows directly. Each founder and child receives 700 projected gradient updates with LR 60 and a global step cap of 0.5. Five generations of up to four parents and three children per parent follow the founder, giving 42,700 search updates per arm and seed. Seeds 111310 and 111410 use the same sparse founders across arms. All arms add twice the normalized verified bond-repair loss. The existing 0.2% parent perturbation and zero noise during optimization remain part of the comparison. [Settings](combined-search.json)

| Policy | Binding update before adding the repair gradient |
|---|---|
| Protenix | The original predicted BRD4 JQ1 pocket-contact gradient |
| Fixed | A constant weighted sum of pocket contact and Nesso-1 predicted affinity, calibrated separately for each founder |
| Adaptive | Recompute both coefficients at each update to balance the two current gradients and match the current Protenix gradient magnitude |

Before each run, a temporary 48-update Protenix plus repair trajectory supplies seven pairs of gradients at updates 0, 1, 4, 8, 16, 32 and 48. Each pair uses identical relaxed inputs in the two frozen models. The search then starts again from the original founder. The fixed fit first matches the two gradient magnitudes by the median ratio of their norms. A common rescaling then matches the median magnitude of the combined gradient to Protenix alone. The fit measures global Euclidean norms over both sets of probability rows before repair, clipping and projection. Each trajectory logs model weights, cosine similarity, contribution sizes and the combined-to-Protenix ratio. Measurements after row centering and periodic projected-step comparisons show how balancing is affected by simplex constraints. Adaptive weights are held constant within a single update. [Calibration](gradient_blend.py) · [Runner](combined_search.py)

All three arms use the existing Protenix and Vina parent ranking and near-valid fallback. Every distinct valid molecule in the archive gets a discrete Protenix contact score and an independent five-recycle native Nesso affinity prediction with a new conformer. Each generation records both scores and the number of molecules that improve on the founder under both, requiring a predicted interface on the Nesso reference and candidate. Nesso selects a protein crop from the BRD4 sequence. Its affinity prediction and the Protenix JQ1-pocket contact prediction measure different things. [Shared graph mapping](combined_nesso_worker.py)

Use `python experiments/latent-design/direct-jump/submit_combined_search.py stage --name NEW_NAME` with a fresh name starting with `combined-brd4-`. The `calibrate` phase accepts `--seed`; the `submit` phase uses the staged commands. GPU jobs write `calibration.json` before search, `dual-readouts.json` in each finished generation and `result.json` on successful completion.

## GPU calibration and submissions

Interactive A100 job 30135332 completed both native backward passes on the same GPU for founder 111310. Its seven-state fit gave weights 0.61843 for Protenix contact and 0.06259 for Nesso affinity. The median combined-to-Protenix gradient norm ratio and the median Nesso-to-Protenix contribution ratio were both 1.00. The combined norm ratio ranged from 0.82 to 1.37. The ratio of the actual projected, model-only displacement to the Protenix displacement ranged from 0.87 to 1.09. Both frozen-model checks passed, and the native reference affinity prediction contained a protein interface. An earlier interactive attempt ended during GPU routing before either scorer ran. [Calibration result](../../../outputs/combined-brd4-700-native-20261120b/111310-calibration-result.json)

Interactive A100 job 30135335 completed the same checks on founder 111410. The fitted weights were 0.68926 for Protenix and 0.07326 for Nesso. The median combined-to-Protenix and relative contribution ratios were both 1.00. The combined norm ratio ranged from 0.73 to 1.16. The projected displacement ratio had a median of 0.89 and ranged from 0.76 to 1.38. The coefficients in both receipts are fits to the seven measured states. The production jobs record how well each policy maintains balance on later molecules. [Second founder calibration](../../../outputs/combined-brd4-700-native-20261120b/111410-calibration-result.json)

| Seed | GPU | Protenix control | Fixed blend | Adaptive blend |
|---:|---|---:|---:|---:|
| 111310 | L40S | 30135336 | 30135337 | 30135338 |
| 111410 | A100 80 GB | 30135339 | 30135340 | 30135341 |

The six searches have 22-hour limits and a total budget of 256,200 search updates, plus 48 calibration-path updates per job. Each production allocation fits its own coefficients on the paired founder trajectory. At 23:38 CET on 20 November 2026, the L40S control was running and the other five jobs were queued. [GPU plan](../../../outputs/combined-brd4-700-native-20261120b/gpu-plan.json)

## First completed control and partial fixed blend

On 21 November, the Protenix control for seed 111310 finished all 42,700 search updates in 2 h 28 min. It produced 64 distinct new valid molecules, 49 of which improved on the founder in both discrete Protenix contact and fresh Nesso affinity. The fixed blend for the same seed had logged 31,076 updates by 08:46 CET. The adaptive job for that seed and all three jobs for seed 111410 were queued. [Completed control](../../../outputs/combined-brd4-700-native-20261120b/protenix-111310-result.json) · [Live progress snapshot](../../../outputs/combined-brd4-700-native-20261120b/status-20261121-084623.json)

At the common completed generation (25,900 updates), the fixed blend yielded 221 distinct new valid molecules with 220 joint improvements. The control yielded 11 with 11 joint improvements. Among the joint improvements, the best fresh Nesso affinity was −0.496 for the blend and 0.808 for the control, from a founder value of 1.352. These are predictions for one seed at matched update count. The fixed blend spends more GPU time per update. [Fixed-blend generation readout](../../../outputs/combined-brd4-700-native-20261120b/fixed-111310-generation-03-dual.json) · [Control generation readout](../../../outputs/combined-brd4-700-native-20261120b/protenix-111310-generation-03-dual.json)

Among dual-score improvements, the median reductions in contact loss and Nesso affinity were 0.120 and 0.956 for the fixed blend, versus 0.072 and 0.281 for the control. Nesso's separate binary binder probability moved in the other direction for most blend molecules. Its median was 0.052, versus 0.286 for the founder, and just one of the 220 exceeded the founder on that readout. Of the 220 blend molecules, 99 were acyclic and the other 121 spanned 43 distinct RDKit Bemis–Murcko scaffolds.

At 12:23 CET on 21 November, the fixed blend for seed 111310 had logged about 39,130 of 42,700 updates. Through the last fully rescored generation (34,300 updates), it yielded 390 distinct new valid molecules, 387 that improved both contact and affinity, and one that improved all three scores. At the same budget the Protenix control yielded 24, 20 and five, respectively. Median binder probability among the fixed blend's 387 joint hits was 0.0523 against a founder probability of 0.2864. Both confidence searches were still queued. [Fixed-blend readouts](../../../outputs/combined-brd4-700-native-20261120b/fixed-111310-generation-04-dual.json) · [Control readouts](../../../outputs/combined-brd4-700-native-20261120b/protenix-111310-generation-04-dual.json)

## Adding calibrated binder probability

Does optimizing binder probability increase the number of new valid molecules that beat their own founder on all three fresh scores? Seeds 111310 and 111410 use the same founders, update rule, chemistry weight 2, restart noise, parent ranking and 700-update segments as the fixed contact/affinity comparison. The new objective minimizes contact loss, affinity and the negative natural log of Nesso's native ensemble binder probability. Its fixed coefficients target mean gradient contributions of 35%, 35% and 30%, respectively, across seven calibration states. A common scale matches the median combined model gradient magnitude to Protenix alone before the repair gradient is added. Individual molecules can have different contributions. [Confidence settings](combined-confidence.json) · [Probability loss and native binder ensemble](../../../src/latentmol/multimodel/nesso.py)

The primary comparison uses all distinct valid SMILES after the founder and five child generations (42,700 updates) and the same native rescoring protocol in both arms. A three-score hit requires better discrete Protenix pocket contact, lower fresh Nesso predicted affinity and higher fresh Nesso binder probability than the same founder, with a predicted interface for the founder and candidate. Reports also count two-score hits, record fresh-scoring attempts and summarize confidence across all new valid molecules. One additional exploratory wave in the confidence arm takes it to 51,100 updates. Width remains four parents by three children. The two seeds use L40S and 80 GB A100 allocations respectively. [Paired comparison script](compare_confidence.py)

Stage with `python experiments/latent-design/direct-jump/submit_combined_search.py stage --name NEW_NAME --config experiments/latent-design/direct-jump/combined-confidence.json --policies confidence --fixed-campaign combined-brd4-700-native-20261120b --hours 24`, choosing a new campaign name with the `combined-brd4-` prefix. The submit phase queues each seed's search and paired report. The two reports depend on both the existing fixed job and the corresponding confidence job.

| Seed | Fixed reference job | Confidence search | Comparison job | Calibrated coefficients (contact, affinity, binder loss) |
|---:|---:|---:|---:|---|
| 111310 | 30135337 | 30147216 | 30147218 | 0.42659, 0.04382, 0.02025 |
| 111410 | 30135340 | 30147217 | 30147219 | 0.53478, 0.06851, 0.05241 |

Native A100 calibration jobs 30147214 and 30147215 measured 35%, 35% and 30% mean raw gradient contributions on each founder's seven probes and a median combined model gradient norm of 1.00 times Protenix alone. At the final probe the production weighted backward pass agreed with the separately evaluated Nesso head gradients to relative L2 errors of 7.5×10⁻⁷ and 1.6×10⁻⁶, respectively. The two new searches and both comparison jobs were queued on 21 November. [Seed 111310 calibration](../../../outputs/combined-brd4-confidence-700-20261121a/111310-calibration-result.json) · [Seed 111410 calibration](../../../outputs/combined-brd4-confidence-700-20261121a/111410-calibration-result.json) · [Scheduler receipts](../../../outputs/combined-brd4-confidence-700-20261121a/submission-status.json)
## Completed fixed blend on seed 111310

The fixed contact–affinity blend and its Protenix control both finished 42,700 updates on an L40S. Counts below are distinct new valid molecules, rescored as discrete graphs against the same founder. The binder probability is Nesso’s separate prediction. [Fixed blend readout](../../../outputs/combined-brd4-700-native-20261120b/fixed-111310-generation-05-dual.json) · [Control readout](../../../outputs/combined-brd4-700-native-20261120b/protenix-111310-generation-05-dual.json)

| Search | New valid | Improved contact and affinity | Also improved binder probability | Median binder probability among contact and affinity improvements | Wall time |
|---|---:|---:|---:|---:|---:|
| Protenix | 64 | 49 | 5 | 0.125 | 2 h 28 min |
| Fixed blend | 499 | 496 | 1 | 0.055 | 13 h 19 min |

The founder’s predicted binder probability was 0.286. The fixed blend yielded about ten times as many contact-and-affinity improvements per search and about 1.9 times as many per GPU-hour on this seed. As of 23 November at 11:12 CET, the adaptive blend had logged 17,500 updates. Both confidence-weighted searches were still queued.

At the matched endpoint for seed 111310, the direct graph search that included binder probability yielded 242 valid new molecules. Of these, 241 improved predicted contact and affinity versus the founder, and none improved all three scores. The fixed contact–affinity blend yielded 499, 496 and one, respectively. Both used 42,700 updates. This comparison informed the choice to monitor binder probability as a separate native readout in the first epoch-48 latent combination. [Paired native scores](../../../outputs/combined-brd4-confidence-700-20261121a/111310-comparison.json)
