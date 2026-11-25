# Fresh binding scores for previously unscored low-SA archive states

The question is whether the completed synthesis searches already visited molecules with audited SA at most 5 and a fresh BRD4 contact loss at most 2.3 or 2.4. The 17 November synthesis audit found 509 distinct eligible molecules, of which 445 had never been selected for fresh scoring in those 16 optimization panels.

Job **30099905** was submitted to **gpuv100** with an exclusive 12 GiB GPU request on 17 November and verified pending. Its budget is **51 fresh evaluations and zero optimizer updates**. Every evaluation uses the historical native FILE BRD4 pocket-contact protocol. The nominal wall limit is one hour.

| Panel | Molecules | Geometry seeds | Calls |
| --- | ---: | --- | ---: |
| Best relaxed binding loss with structural diversity | 24 | 1201 | 24 |
| Random matched controls from the remaining pool | 24 | 1201 | 24 |
| Previously scored SA 4.91047, fresh loss 2.302478 candidate | 1 | 1201, 2203, 3209 | 3 |

For each of 445 identities, the collector recovered the lowest recorded relaxed contact loss over its valid visits and the corresponding original founder lineage, job, branch and update. There were 11 near-valid-founder, 421 deep-founder and 13 wide-founder identities. Each arm has three, eighteen and three molecules from these founders respectively. The ranked arm takes lower relaxed loss first subject to a maximum pairwise Morgan radius-2 Tanimoto similarity of 0.65. All 24 met that threshold. Random controls were drawn without replacement, disjoint from the ranked panel, matched by founder and half-point audited SA interval with seed 1711021. The control comparison estimates the joint effect of relaxed-score ranking and structural diversity under this sampling policy.

The near-miss is job 30078658, combined seed 2, Pareto cycle 4, parent 1, update 490. Its previously observed loss was not used to select any of the 48 panel members. Report its held-out two-seed mean and individual losses separately. Each group's main results are distinct hits at both loss thresholds per 24 attempts, successful and failed scores, best and median successful loss. Counts of unscreened identities are relative to the audited follow-up.

The local smoke used simulated scores and checked all 51 attempts, both 24-member groups, matching and readiness handling. Compare actual outcomes only after native HPC receipts arrive.

[Point provenance and frozen panel](../../../outputs/synthesis-archive-screen-20261117/panel.json) |
[Pool extraction](../../../outputs/synthesis-archive-screen-20261117/collect_pool.py) |
[Selection and scoring script](restart-operations/synthesis_archive_screen.py) |
[Launch receipt](../../../outputs/synthesis-archive-screen-20261117/submission.json) |
[Scheduler snapshot](../../../outputs/synthesis-archive-screen-20261117/status.json).

## Completed archive screen, 18 November 2026

Job 30099905 completed all 51 fresh scores and exited zero. The ranked and sampled groups each scored 24/24 successfully. At audited SA at most 5 and fresh loss at most 2.3, the ranked group supplied two hits and the sampled controls supplied zero. The paired SA/loss values were 4.88141/2.29945 and 4.99240/2.29507. At loss up to 2.4 the counts were 12 and 4. Median losses were 2.39564 and 2.53863. Both strict hits have only one geometry seed. The separate historical near-miss scored 2.30248, 2.30268 and 2.30439 on the three geometry seeds (held-out mean 2.30354).

[Verified completion and arm summaries](../../../outputs/synthesis-news-20261118T091637Z/snapshot.json) and [individual native scores](../../../outputs/synthesis-news-20261118T091637Z/archive-native-rows.json).
