# What changes when the restart search uses LR 30?

7 November 2026. Repeat both LR 70 experiments with a learning rate of 30.
Use the corresponding sealed source archive for each experiment. The
scientific configs differ from the original embedded configs only in
`design.learning_rate`. The staged archive comparison verified all other
files byte for byte apart from the submission plan.

The [standalone config](restart-lr30.json) runs two cycles and up to 10,000
updates on a 32 GB V100, matching the active LR 70 replacement. The
[six-arm config](restart-lr30-six-cycle.json) runs six cycles and up to
46,000 updates per arm on L40S. All runs use 1,000 updates per branch and
the corresponding LR 70 seeds, fan-out, chemistry weights, fresh scoring
budget, step caps, and selection policies.

The step cap acts after learning-rate scaling. At a shared state, both
rates can apply the same displacement when both hit the cap. Compare
actual displacement and cap hits alongside validity and fresh loss.

The [submitter](restart-operations/submit_campaign.py) stages each config
with `--source-receipt` set to its LR 70 submission receipt. The
[standalone records](../../../outputs/protenix-restart-lr30-20261107a/)
and [six-arm records](../../../outputs/protenix-restart-lr30-six-20261107a/)
hold the staged archives, exact commands, and DTU receipts. The combined
new output allowance fit the measured storage capacity.

DTU accepted all seven jobs between 23:46 and 23:47 CET. All were pending
at the 23:47 scheduler check.

| Arm | LR 30 job | Matched LR 70 job |
| --- | ---: | ---: |
| Standalone screened | 29920268 | 29920256 |
| Combined | 29920269 | 29916515 |
| Cap 0.5 | 29920270 | 29916522 |
| Repair only | 29920271 | 29916530 |
| Unchanged parent | 29920272 | 29916537 |
| Relaxed shortlist | 29920273 | 29916544 |
| Top-scoring parents | 29920274 | 29916552 |

## Queue move on 8 November 2026

The six pending comparison jobs moved from L40S to 32 GB V100 with their original scientific settings and output roots. The replacements have a 45-minute startup allowance. DTU confirmed the originals exited and all six replacements were pending at 16:10 CET while dispatch was paused. See the [queue move receipts](../../../outputs/protenix-restart-l40s-move-20261108a/release.json).

| Arm | Active LR 30 job |
| --- | ---: |
| Combined | 29926092 |
| Cap 0.5 | 29926093 |
| Repair only | 29926094 |
| Unchanged parent | 29926095 |
| Relaxed shortlist | 29926096 |
| Top-scoring parents | 29926097 |

## Completed six-cycle arms on 9 November 2026

At 15:49 CET, five arms at each learning rate had completed all 46
branches (46,000 updates per arm). Both top-parents arms were still
running. Final best fresh Protenix contact loss was:

| Arm | LR 30 | LR 70 |
| --- | ---: | ---: |
| Combined | 2.19800 | 2.14295 |
| Cap 0.5 | 2.17221 | 2.08529 |
| Repair only | 2.11412 | 2.12505 |
| Unchanged parent | 2.19157 | 2.12279 |
| Relaxed shortlist | 2.24003 | 2.22581 |

LR 70 finished lower in four of five pairs. These are single-seed
results at matched update budgets; actual fresh attempts and GPU
assignments differed across pairs.

The median improvement from the first to the final archive was 0.163 at
LR 30 and 0.179 at LR 70. Eight of ten arms had found their final best by
cycle five. The LR 70 combined and cap 0.5 arms both set new lows in
cycle six. Across the ten arms, 223 of 450 post-bootstrap branches
fresh-scored at least one candidate below their own parent's loss.

The LR 70 cap 0.5 arm produced 86 candidates below 2.20 among 665 successful
fresh scores. The LR 70 combined arm (cap 0.25) produced one among 684.
Equalizing successful scores within each cycle by sampling uniformly
from the saved pools gives expected counts of 84.92 and 0.71 below 2.20
at 626 scores each. The same retrospective check within each LR pair
retained the direction of all five best-score comparisons. This measures
sensitivity to shortlist size using the existing pools. The optimization
and selection decisions still reflect the original scoring budgets.

The repair-only arms reached their best in cycle two and had 54–58%
raw-valid observations, versus 92–98% for combined. They also found
many low-loss candidates. Chemistry guidance and candidate quality should
therefore be assessed separately. Two run winners, LR 70 combined and
relaxed shortlist, triggered the existing four-triangle review flag
when their SMILES were parsed with RDKit.

The [matched adaptive, fixed, and independent comparison](restart-lineage-feedback.md)
tests the main remaining question: whether feeding improved candidates
back into the parent pool yields more than repeatedly exploring the
same starting families. The earlier two-arm tests favored elite
restarts over prior starts in both parent cohorts. The six-cycle runs
use a single bootstrap seed and the same BRD4 pocket and scorer.
