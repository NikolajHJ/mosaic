# Which restart changes improve fresh Protenix scores?

7 November 2026. Eight new exploratory comparisons use the frozen epoch-45
decoder, the same BRD4 Protenix context, 25 occupied slots, temperature 1,
constant structural chemistry weight 1, and latent SGD LR 150. Each job
contains both sides of its comparison. The A and B slices use disjoint
sets of scored, independently sourced parents from three completed flow
cohorts. Each parent is the best fresh-scored molecule available on its
source trajectory, and its bare saved latent must decode to the exact saved
hard labels and canonical identity.

| Comparison | Design in each of two jobs | Total updates per job |
| --- | --- | ---: |
| Perturbation | 12 parents, 96 updates per branch. Unchanged latent, 50/50 all-row, decoder-screened latent kick, and sparse relaxed perturbation of a connected neighborhood. Two seeds for each stochastic policy. Report first-branch matched comparisons separately. | 8,064 |
| Step cap | 12 parents, two matched latent kicks and 128 updates per branch at caps 0.1 and 0.5. | 6,144 |
| Duration | Eight parents, four 124-update branches versus one 499-update branch. 500 relaxed calls and up to 16 fresh-score slots per parent and arm. The long branch shares its start with the first short branch. | 7,960 |
| Lineage | Start both arms from the same three scored molecules. Over five cycles, compare nine elite-derived branches per cycle against nine distinct valid prior starts per cycle. Each branch has 96 updates. Both arms retain their best three observed hard graphs. | 8,640 |

Latent kicks screen four normalized directions at each radius 0.03, 0.10 and
0.25 using the decoder alone. Prefer connected new molecules one to three label
edits from the parent. The sparse arm softens the existing decoder probabilities
around three connected atoms and their existing internal bonds. It preserves
the no-bond rows and the parent argmax at step zero. Sparse and all-row offsets
fade by update 25. Per-branch records include soft bond mass, edit counts,
validity, actual displacement and cap hits.

Rank archived molecules by fresh Protenix FILE contact loss with conformer
seed 1201. The perturbation test also compares two six-slot fresh shortlists
from the same trajectories. One uses relaxed rank alone. The other uses half
relaxed rank and half evenly spread by source step. Score their union and
reuse duplicate SMILES scores. The other comparisons use the mixed shortlist.
An independent branch includes its starting molecule within its fresh budget.
Fresh score attempts and unscorable molecules are separate from update counts.

## Command and outputs

The [design](restart-followup.json) and [fresh scoring config](restart-followup-fresh-dtu.json)
are used by the allocated entry point `python -m latentmol.design.protenix_restart_study`.
The [submission directory](../../../outputs/protenix-restart-followup-20261107a/)
contains the exact commands and frozen snapshot. Per-job artifacts go under
`/work3/s234818/latentmol/runs/protenix-restart-followup-20261107a-<task>`.
Paired results retain per-parent scores. The lineage jobs record separate
archives and best fresh losses after each cycle. These are exploratory
comparisons within the single prepared BRD4 pocket.

## Submission and checks

Submitted at 11:02 CET on 7 November 2026. Each job requests one exclusive GPU,
four CPU slots, and a six-hour wall limit. All eight were pending at the
initial check. V100 was at the per-user GPU limit, and L40S/A40 had host or GPU
reservation constraints. Each paired comparison runs on a single GPU.

| Task | Queue | Job |
| --- | --- | ---: |
| mutation-a | gpul40s | 29914683 |
| mutation-b | gpua40 | 29914684 |
| cap-a | gpul40s | 29914686 |
| cap-b | gpuv100 | 29914687 |
| duration-a | gpul40s | 29914688 |
| duration-b | gpua40 | 29914689 |
| lineage-a | gpul40s | 29914690 |
| lineage-b | gpuv100 | 29914691 |

Source snapshot SHA-256
`105be1388488dab8dcecfbaf31d4e93e0a5da1a13dd057b0b6e3464856dd63be`.
Local verification passed 42 focused tests, restored 12 real archived parents
with the authenticated decoder, and exercised all four proposals and two
lineage cycles with the real decoder and mock scorers. The preflight confirmed
the inputs and output capacity. Native backward checks and scientific results
are pending.

## First results at 13:58 CET

The first perturbation panel **29914683** finished all 84 branches, and the
first step-cap panel **29914686** finished all 48. Each panel has 12 parents.
Both processes passed the native startup check and closed successfully. The
other six jobs were pending at the scheduler check.

For the perturbation panel, compare the first branch per parent and the
mixed six-candidate shortlist. Mean best fresh-loss reductions were 0.234
for the latent kick, 0.221 for sparse perturbation, 0.199 for the untouched
latent, and 0.143 for the all-row 50/50 start. Latent beat all-row on 9/12
parents with two ties, and beat untouched on 7/12 with no ties. The mixed and
pure relaxed shortlists traded wins across policies.

For the step-cap panel, 0.5 produced a mean best fresh-loss reduction of
0.261 versus 0.122 at 0.1, and won on 11/12 parents. The raw-valid share of
observations was 59.6% versus 79.3%. The 0.5 arm produced 1,312 distinct valid
molecules summed within branches versus 395 at 0.1. Most updates hit the cap
in both arms. This first panel favors 0.5 for the fresh-score objective.
The second panels and both duration and lineage comparisons are pending.

[Saved results and paired analysis](../../../outputs/protenix-restart-followup-20261107a/readout-20261107-1358/).

The [interactive report](../../../outputs/protenix-restart-report-20261107/protenix-restart-report.html)
captures both campaigns at 15:05 CET on 7 November. It includes 166 branch
traces, the two completed paired panels, and four saved chemistry archives.
The player draws the parent and the selected graph side by side. It covers
8,868 of 8,869 sampled frames, including saved invalid graphs. The remaining
frame belongs to a running branch whose raw labels had yet to be written.

The Run trees view uses the later 15:25 CET snapshot. It links 40 verified
restarts to their exact saved source states across 178 branches. Both
completed jobs have full coverage. Raw topology was still unavailable for
20 observations in one running branch.

## Completed repeats on 8 November 2026

Mutation B and duration B were complete at the 22:05 CET check. All eight
matched follow-ups had finished their update budgets. On mutation B, the first
latent-kick branch averaged a best fresh-loss reduction of 0.212 across 12
parents versus 0.076 for all-row 50/50 using the mixed shortlist. On duration
B, four shorter branches averaged 0.226 versus 0.222 for one longer branch
across eight parents. [Saved comparison](../../../outputs/protenix-restart-status-20261108-2205/playback-snapshot.json).
