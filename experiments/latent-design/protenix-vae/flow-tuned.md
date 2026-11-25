# Does a learned jump help after tuning latent SGD?

Follow-up to the [fixed-rate experiment](flow-followup.md), 6 November 2026.
This study tests learning rates in the VAE latent `z` under the frozen epoch-45
decoder, BRD4 context, 25 occupied slots, temperature 1.0, and L2 displacement
cap 0.5. It uses the same fresh hard-graph Protenix FILE contact loss and
conformer seed 1201. Lower loss is better.

## Tune on separate starts

The [config](flow-tuned.json) tests **0.3, 1, 3, 10, and 150** on the same 32
connected prior starts, split into two independent 16-start cohorts. All rates
use the same initial latent, fixed slots, noise seed, and seven updates. The
initial and final observations give eight relaxed scorer calls per run. Score
every distinct connected molecule observed in that budget. Save the actual
number of displacement cap hits for each rate.

Select the rate with the largest mean best fresh loss reduction per start,
allowing each start to keep its original molecule for a gain of zero. An
unscorable baseline contributes zero verified gain for every rate. Break ties
by more starts with a verified improvement over 0.005, then the lower rate.
The CPU selector requires both complete tuning cohorts and records the score
table, selected rate, and tuning identities in `selection.json`.

## Retrain and compare on fresh starts

Four new validation cohorts each use 24 separate training starts and 24 held-out
starts. The selected rate controls the 24-update teacher used to collect new
training pairs as well as all SGD steps in all evaluation arms. Each cohort
fits a fresh one-step residual model. Starting molecule identities are checked
for overlap between tuning, training, and test panels. The four-step flow and
DeFoG arms are outside this learning-rate comparison.

At the same **16 relaxed scorer calls per start**, compare ordinary SGD plus
learned jump and refinement against ordinary SGD plus an independent restart.
A same-size random jump with the same refinement noise remains a secondary
control. The outcome is any connected edit whose fresh loss improves by more
than 0.005, with paired 95% intervals and the same 15-point practical margin.
The CPU report runs after all four cohorts and checks that they used the same
selection receipt. Fewer than eight eligible training pairs leave that cohort's
learned comparison incomplete. The 7-versus-24 update comparison stays
secondary.

The [same allocated launcher](run_flow_pilot.sh) runs `tune`, `select`,
`latent`, and `report` in sequence through scheduler dependencies. The
study applies to this BRD4 pocket, starting distribution, and contact scorer.

## Submission

| Stage | DTU job | Allocation and dependency |
| --- | ---: | --- |
| Tuning cohort 701000 | 29911621 | L40S, up to 3 h |
| Tuning cohort 702000 | 29911622 | V100, up to 3 h |
| Rate selection | 29911624 | CPU, after both tuning cohorts |
| Validation cohort 711000 | 29911625 | V100, up to 4 h, after selection |
| Validation cohort 712000 | 29911626 | L40S, up to 4 h, after selection |
| Validation cohort 713000 | 29911627 | A40, up to 4 h, after selection |
| Validation cohort 714000 | 29911628 | A100, up to 4 h, after selection |
| Paired report | 29911629 | CPU, after all four validation cohorts |

The local checkpoint screen verified 224 distinct starting identities across
the six cohorts. Forty-nine focused CPU tests passed, including the tuning
selection and paired analysis. At 21:02 CET both tuning jobs were waiting for
GPUs. The [receipts and frozen source](../../../outputs/flow-lr-20261106a/)
record the exact commands and scheduler dependencies.

## Partial sweep and scoring recovery, 11 November 2026

Both cohorts completed all latent optimization. Cohort 702000 completed fresh
scoring. Cohort 701000 failed during RDKit `EmbedMolecule` with the specific
distance-bounds invariant `upper bound not greater than lower bound`. The
worker exited before the final score summary, so rate selection and all
validation jobs stayed pending on failed dependencies.

| Learning rate | Mean best fresh contact-loss reduction on 16 starts | Starts improved by more than 0.005 |
| --- | ---: | ---: |
| 0.3 | 0.05135 | 10/16 |
| 1 | 0.12365 | 12/16 |
| 3 | 0.16337 | 13/16 |
| 10 | 0.16407 | 13/16 |
| 150 | 0.16154 | 12/16 |

These are [verified partial scores](../../../outputs/flow-lr-20261106a/results-20261111a/partial-sweep.json)
from one cohort. LR 150 hit the 0.5 displacement cap on all 112 updates. The
complete 32-start tuning panel remains required for selection.

The reference geometry helper now turns this specific RDKit bounds exception
into its existing unscorable-conformer result. Other runtime errors propagate.
Eleven relevant CPU tests passed, including both exception paths. Recovery
job **29994084** scores the saved cohort 701000 candidates in a new output
directory. It retains the original optimization job ID and input hashes in
the recovery panel. The successful tuning cohort and failed evidence remain
available at their recorded paths.

DTU's submission hook rejected `bmod` command updates. The six downstream jobs
were authenticated as pending and never started, retired with terminal EXIT
evidence, and replaced using the recovered source snapshot. The scientific
settings, selected-rate rule, and cohort starts are the same. Replacement
selector **29994198** waits for recovery. Validation jobs **29994201, 29994203,
29994205, 29994208** wait for selection. Report **29994210** waits for all four
validation jobs. At 10:40 CET recovery was pending on L40S slot availability.
[Recovery, replacement and verification receipts](../../../outputs/flow-lr-20261106a/results-20261111a/).

## Monitoring through completion, 12 November 2026

Scoring recovery 29994084 completed and selector 29994198 selected **LR 150**
using all 32 tuning starts. Local recomputation of the verified downloaded
scores reproduced the selection exactly. Mean best fresh-loss reductions were
0.05595, 0.12690, 0.15701, 0.16603, and 0.16786 for rates 0.3, 1, 3, 10, and 150.
The respective verified improvement counts were 16, 21, 26, 26, and 24 of 32. Three
starting molecules were unscorable and contributed zero verified gain at every
rate. [Selection](../../../outputs/flow-lr-20261106a/results-20261112a/verified-selection.json).

Cohorts 711000, 712000, and 714000 completed with 21, 22, and 23 eligible training pairs,
respectively. The user requested monitoring until the full comparison finishes.
The never-started A40 cohort 29994205 was blocked behind allocations with at
least 8.9 hours remaining when inspected. It and dependent report 29994210 were
retired with terminal EXIT evidence. The same 713000 source snapshot, starts,
settings and output path are now queued on V100 as **30017428**, with report
**30017430** depending on the remaining cohort. Actual hardware placement is
recorded in the [relocation receipts](../../../outputs/flow-lr-20261106a/results-20261112a/relocated-713000-submission.json).
The report also retains its dependency on completed 711000. At 12:31 CET the
last cohort remained pending. The split and score checks for the three finished
cohorts passed, and their [interim audit](../../../outputs/flow-lr-20261106a/results-20261112a/three-cohort-audit.json)
keeps the primary decision incomplete until all four cohorts are available.

## Full results verified, 17 November 2026

The relocated cohort 713000 (job 30017428) and combined report (job 30017430)
both completed successfully. The downloaded report matched the local
recomputation from all four cohorts exactly. The cohort fits used 21, 22,
22, and 23 training pairs. The validation receipts confirm 120 trajectories
and 1,776 relaxed Protenix calls per cohort, including training data collection
and secondary controls.

| Comparison at LR 150 | Starts with a verified improvement | Paired difference, percentage points (95% interval) |
| --- | ---: | ---: |
| SGD plus learned jump vs. two SGD runs, 16 calls each | 86/96 vs. 84/96 | +2.1 (−3.1 to +8.3) |
| Learned jump vs. independent SGD restart | 81/96 vs. 76/96 | +5.2 (−2.1 to +13.5) |
| Learned jump vs. same-size random jump | 81/96 vs. 81/96 | 0.0 (−9.4 to +9.4) |
| 24 vs. 7 SGD updates | 89/96 vs. 78/96 | +11.5 (+5.2 to +17.7) |

The primary comparison had five wins, three losses, and 88 ties and landed
within the prespecified ±15 percentage-point margin. The result was similar
across the 94 starts with a usable baseline score (+2.1 points, 95% interval
−3.2 to +8.5). Mean best loss reduction was 0.198 for SGD plus the learned
jump and 0.185 for two SGD runs, counting zero when a start had no verified
gain. This was a secondary descriptive measure. LR 150 won the 32-start
tuning criterion by 0.00182 over LR 10 and hit the 0.5 latent step cap on
222 of 224 updates. The comparison applies to this BRD4 context and fresh
Protenix contact score.

[Verified comparison](../../../outputs/flow-lr-20261106a/results-20261112a/verified-final-comparison.json)
· [Verified rate selection](../../../outputs/flow-lr-20261106a/results-20261112a/verified-selection.json).
