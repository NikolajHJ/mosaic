# Does a learned proposal earn its scorer calls?

Follow-up to the [flow pilot](flow-pilot.md), 6 November 2026. Use the same
frozen epoch-45 VAE, BRD4 context, 25 occupied slots, temperature, and two native
Protenix contact readouts. Fresh hard-graph scoring uses the existing FILE
featurization and conformer seed 1201. Conclusions apply to this pocket and
scoring protocol. A multi-target study needs additional prepared contexts.

## Matched latent comparison

The [fixed config](flow-followup.json) declares four disjoint cohorts. Each
screens 48 distinct connected prior starts, with the first 24 for training and
24 held out. Fit on at most one independently rescored improving endpoint per
training start, with a minimum of eight eligible pairs. Each cohort fits its
own students. The original pilot's starts and outcomes are outside this split.

| Policy | Relaxed scorer calls per held-out start |
| --- | ---: |
| Ordinary SGD, using the first 7 updates of the teacher trajectory | 8 |
| Full 24-update SGD trajectory | 25 |
| Independent-noise SGD restart | 8 |
| Random direction matched to the learned jump's displacement norm, then SGD | 8 |
| Learned one-step jump, then SGD | 8 |
| Four-step flow, then SGD | 8 |

The restart, random jump, residual and flow share a noise stream within each
start. Each run uses the same fixed active slots and optimizer. The short runs
fresh-score every distinct connected candidate, including any novel proposal
at step zero. The longer teacher also scores up to six additional distinct
edits, selected by time without using scores. Deduplicate within each fresh
scoring batch. Training examples are selected by fresh score on training starts.
There are at most 1,968 relaxed calls per cohort including teacher collection.

The **primary comparison** spends 16 calls on each portfolio. Both include
ordinary 7-update SGD. The candidate adds the learned residual arm and the
control adds the independent-noise SGD restart. An independent start succeeds
if the portfolio contains any connected edit with fresh loss more than 0.005
below its starting molecule. Unscorable starts count as having no verified hit,
with a scored-baseline sensitivity analysis. Count a start once even if both
arms succeed. Compare rates using paired resampling within each cohort, with a
95% interval and a predeclared practical margin of 15 percentage points.

- Call the candidate practically better if the interval is entirely above
  +15 points. Rule out that size of benefit if its upper bound is below +15.
- Report practical equivalence if the interval lies within ±15 points.
  Report an inconclusive result if it spans the decision boundary. Missing
  cohorts or too few training pairs leave the primary comparison incomplete.
- Treat the random jump, four-step flow, and short-versus-long comparisons as
  secondary. Report paired wins/losses, score magnitudes, scored coverage and
  per-cohort results. The interval describes held-out starts for the fitted
  cohort models. It does not measure uncertainty across pockets or training
  datasets in general. With 96 held-out starts and 50% discordant pairs, the
  approximate 95% half-width near zero is 14 points.

## Local DeFoG repair

After the four latent jobs finish, use up to 24 invalid and 16 valid graphs
from distinct held-out trajectories. Interleave cohorts, select the earliest
eligible state of the requested class, and use the other class if necessary.
Compare structural search against DeFoG at model times 0.80, 0.96 and 0.99 on
the same 128-point grid, one matched random seed per input. Keep the strict
neutral GuacaMol input policy. Record unsupported inputs encountered during
selection and the number of distinct eligible starts obtained.

For an invalid input, count a local rescue when the output is connected and
within five total atom, bond or charge label changes. For a valid input, also
require the fresh loss to worsen by at most 0.02. Compare paired outcomes with
structural search and report all recovered graphs, edit counts, score changes,
and runtime. The later schedules test whether DeFoG can repair locally after
the broad rewrite observed at 0.80. Report any shortfall from the target sample
size and treat the schedule comparisons as exploratory.

## Reproduction

The existing [launcher](run_flow_pilot.sh) accepts the follow-up study and
[fresh-score config](flow-followup-fresh-dtu.json). Pass `--seed COHORT_SEED`
to a latent allocation. Pass `--repair-parents COHORT_ROOT ...` to the repair
allocation after all four cohorts finish. The analysis consumes completed
outputs, checks the split and arm coverage, and writes a new JSON result.

```text
python experiments/latent-design/protenix-vae/flow_followup.py --config experiments/latent-design/protenix-vae/flow-followup.json --latent COHORT_1 COHORT_2 COHORT_3 COHORT_4 --repair REPAIR_ROOT --output NEW_JSON
```

## Submission and checks

| Cohort or comparison | DTU job | GPU allocation |
| --- | ---: | --- |
| Seed range from 520000 | 29905200 | V100, 4 h |
| Seed range from 530000 | 29905201 | L40S, 4 h |
| Seed range from 540000 | 29905202 | A40, 4 h |
| Seed range from 550000 | 29905204 | A10, 4 h |
| Local repair | 29905205 | L40S, 2 h, after all four cohorts succeed |

The local epoch-45 screen found 192 distinct connected starting molecules
across the four cohorts. Forty-eight focused CPU checks passed, including
the paired aggregation and coverage tests. Each allocation requests one
exclusive GPU and four CPU slots. The [submission receipts and source snapshot](../../../outputs/flow-followup-20261106a/)
contain exact launch arguments and input hashes.

## Interim readout, 6 November 2026 at 20:41 CET

The V100, L40S, and A40 cohorts completed and each scored all comparison arms.
Their training sets supplied 19, 17, and 18 eligible pairs, respectively. The
A10 cohort remained pending for a GPU, with CPU slots reserved. The repair job
still depends on that cohort.

| Comparison | Held-out starts with an improvement |
| --- | ---: |
| SGD plus learned jump versus two SGD runs | 51/72 versus 47/72 |
| Learned jump versus matched random jump | 45/72 versus 46/72 |
| Four-step flow versus learned jump | 42/72 versus 45/72 |
| 24 versus 7 SGD updates | 54/72 versus 46/72 |

The portfolio comparison had five wins, one loss, and 66 ties, a difference of
+5.6 percentage points. The three-cohort paired interval was 0.0 to +12.5
points. Three of the 72 starting molecules lacked a usable fresh score and
remain in the denominator. The predeclared primary decision is incomplete
until the fourth cohort has scored. [Interim analysis and verified inputs](../../../outputs/flow-followup-20261106a/scored-20261106a/interim.json).
