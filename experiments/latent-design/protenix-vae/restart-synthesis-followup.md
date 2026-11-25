# Aligned synthesis priors and search policies — 16 November 2026

The 17 November audit supersedes the historical low-SA hit counts and the interpretation of several individual molecules below. [Audited results](#audited-synthesis-scores-and-validity-17-november-2026).

Question: which changes produce more distinct, valid molecules with useful
fresh Protenix scores and lower synthesis difficulty? This campaign follows
the [one-seed audit](restart-synthesis-next.md#detailed-attraction-and-parent-selection-audit-16-november-2026),
which found useful alternative lineages, missed intermediate replay points,
and a mismatch between bank and search atom-slot masks.

The planned panel has **18 optimization jobs / 270,000 updates**, plus one
**zero-update readout job / 12 fresh calls**. Each comparison uses three seeds:
**1613113, 1613117 and 1613129**. Every child uses **LR 60, 500 updates,
displacement cap 0.25, 25 occupied atoms and temperature 1**. Optimization
noise is zero. The same three archived founders and their original slot mask
are preserved; child-initialization perturbations use matched seeds.

## Comparisons and budgets

| Group | Question and variants | Updates per seed | Three-seed updates |
| --- | --- | ---: | ---: |
| Attraction mechanism | Common SynFrag 0.05; attraction off, broad low-SA bank, broad unfiltered bank, low-SA centroid, local low-SA bank, local unfiltered bank. | 9,000 | 27,000 |
| Attraction strength | Local attraction ratios 0.025, 0.05, 0.1 and 0.2, plus ratio 0.1 with binding-conflict projection; SynFrag stays 0.05. | 7,500 | 22,500 |
| Screening and standalone priors | Unguided, local low-SA attraction alone, local unfiltered attraction alone, and combined SynFrag/local attraction with binding-only versus binding/SA screening. | 7,500 | 22,500 |
| Selection × schedule | Constant versus wave-focused synthesis guidance crossed with binding versus Pareto parent pools. | 24,000 | 72,000 |
| Selection and replay | Pareto with terminal recovery, one reserved low-SA parent slot with terminal recovery, Pareto with joint-quality intermediate recovery. | 18,000 | 54,000 |
| Selection × attraction | Constant SynFrag 0.05; binding/Pareto parents crossed with local low-SA attraction off/on at ratio 0.05. | 24,000 | 72,000 |

Related variants share the central runner and these configs:
[attraction](restart-synthesis-attraction.json),
[selection](restart-synthesis-selection.json),
[replay](restart-synthesis-replay.json),
[combined](restart-synthesis-combined.json).
Short groups have one wave. Search groups have four waves, three children per
wave and chemistry weights **1 → 1 → 1 → 2**. The focused schedule uses synthesis
ratios **0.1 → 0.1 → 0.4 → 0.05**; its constant comparison uses 0.1 throughout.
These schedules test practical policies, not equal integrated guidance exposure.

All arms have independent allowances of **six fresh attempts per child**:
18 per method in a short panel and 72 in a four-wave panel. The optimization
jobs together allow up to **3,240 fresh calls**. Actual attempts, successful
scores, failures, unused allowances and wall time remain separate. An arm
with few valid molecules cannot reduce another arm's allowance.

## Aligned frozen latent banks

The new preparation fits only latent vectors through the **frozen existing
decoder**, using reconstruction targets mapped to the original founders'
actual occupied slots **0–23 and 25**. It does not train encoder/decoder
weights, change founder molecules, edit live checkpoints or modify the
differentiable Protenix graph. Both source banks receive the same bounded
fitting procedure. Eligibility is checked on each actual masked hard decode;
the low-SA bank additionally requires decoded SA ≤3.5. The requested bank
size is 16 per group, with equal-size disjoint decoded identities and recorded
shortfalls. Fitted latents are not subsequently norm-rescaled, because doing
so can break decoded validity. Source identity and decoded identity/SA are
recorded separately; exact source reconstruction is not assumed.

Broad attraction retains temperature 0.5. Local attraction calibrates its
temperature once on the shared saved founders, targeting a mean effective
anchor count of four; logs record achieved count and any mismatch. The
centroid control tests whether broad attraction mainly acts as general latent
regularization. Source and control banks share the distance scale derived
from the aligned low-SA bank. Ratios refer to latent-gradient norms relative
to the contact gradient, not raw loss weights. The combined auxiliary cap
scales with the chosen ratios; after more than 20 consecutive invalid
observations, the guidance and its cap are suppressed to 0.2. Projection
removes a first-order conflicting component, without guaranteeing binding.

## Parent selection, screening and replay

Selection comparisons protect the best fresh binder, then use a **common
diversity-first ordering and fresh-loss ceiling 2.35**. Pareto filtering changes
the eligible pool using fresh loss and detached hard SA; it does not change
the diversity priority or ceiling. Other settings are shared: maximum Morgan
similarity 0.85 and source-step separation 100. Budget-preserving relaxations
and protected-elite exemptions are logged. The reserved-SA-slot arm spends
one nonelite slot on the easiest eligible parent, then fills by diversity.

Joint recovery selects the lowest-SA, successfully fresh-scored, replayable
point from the **same preceding child branch**, subject to loss ≤2.35. It
replays that exact latent continuously without another initialization kick.
If no eligible point exists, the original terminal/witness fallback is logged.
A repair witness is evidence only; no edited graph replaces the latent.

The shared `binding-sa` shortlist reserves approximately half of each
allowance for standard binding candidates and the rest for low-SA witnesses.
Hard chemistry measurement stays outside autograd. Only the dedicated
`sf-local-binding-screen` arm retains the old binding-focused shortlist.
**This changes selected-sample bias relative to the older one-seed panels:**
their selected mean SA should not be treated as directly comparable to these
new means. The screening comparison measures the effect at matched optimizer
settings and scoring allowances.

## Saved-molecule readout

[The readout config](restart-synthesis-readout.json) evaluates four saved
molecules: the previously unscored attraction endpoint (SA 4.363), the
Pareto-selected parent (SA 6.127), its useful update-278 intermediate
(SA 4.217), and its worse update-500 endpoint (SA 6.758). Each receives three
geometry seeds **1201, 2203 and 3209**, for 12 fresh evaluations and zero
optimizer updates. Exact source batch/result identities and provenance text
are retained. This tests the missing endpoint score and sensitivity of the
observed parent/intermediate trade-off to geometry generation.

Primary readouts are distinct successfully scored molecules at SA thresholds
4/5/6 and fresh-loss thresholds 2.2/2.3/2.4, the corresponding diverse counts
(greedy Morgan similarity ≤0.6), and the SA/contact trade-off. Also inspect
validity across all observed states, top-three binding quality, actual costs,
DFRscore (not optimized here), and hard-input SynFrag readouts. These are proxy comparisons,
not synthesis-route or affinity certification; SA is used in selection and
therefore is not an independent validation of that selection objective.

## Preparation and submission status

Aligned-bank preparation **30078294 completed** in 796s on allocated CPUs.
Both banks contain **16/16 valid actual-mask decodes and 16/16 exact source
identities**; there was no shortfall. All 32 source fits in each bank were
eligible; the first 16 disjoint sources were retained under the declared rule.
The low-SA decoded mean is **2.3794** and the unfiltered control mean **3.2504**.
The frozen encoder/decoder stayed unchanged; this was latent inversion, not
model training. No post-fit norm rescaling was applied. Old banks are untouched.
[Preparation result](../../../outputs/synthesis-followup-20261116/aligned-summary.json).

Actual-bank CPU backward smoke passed for all five configured library modes.
Broad T 0.5 uses about 15/16 effective anchors. Local temperatures calibrated on
the saved founders are **0.084806** (low-SA) and **0.025344** (unfiltered), both
averaging 4 effective anchors; centroid attraction uses all 16. These temperatures
stay fixed during optimization, while per-step logs record effective counts,
distance, latent norm, actual gradient ratio, validity suppression and projection.
[Gradient/locality smoke](../../../outputs/synthesis-followup-20261116/aligned-gradient-smoke.json).
Final focused local verification: **79 tests passed**. This establishes tested
finite/attached/frozen behavior, not precise derivative or native parity.

**All 19 GPU jobs are submitted:** 18 optimization panels plus one readout.
The first six nonanchor jobs were sealed before later attraction-only guard/log
changes; each uses its own exact source archive. Existing pending/running jobs,
checkpoints, prior outputs and original banks were preserved.

| Comparison | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: |
| Attraction mechanism | 30078617 | 30078619 | 30078622 |
| Attraction strength/projection | 30078624 | 30078626 | 30078629 |
| Anchor-only and screening | 30078631 | 30078633 | 30078636 |
| Selection × schedule | 30078310 | 30078311 | 30078313 |
| Selection/replay policy | 30078381 | 30078382 | 30078384 |
| Attraction × selection | 30078656 | 30078658 | 30078660 |

Saved-molecule readout: **30078398** (V100), four molecules × three conformer
seeds, 12 fresh calls and zero optimizer updates. At the submission snapshot,
**30078310 and 30078381 were RUN on V100; the other 17 jobs were PEND**.
The first selection job had already completed its first 500-update child.
No scientific optimization result is claimed from that partial progress.

Total planned compute is **270,000 Protenix updates**, with **3,240 maximum
fresh attempts** across the optimization comparisons plus 12 readout calls.
Each method/seed has 1,500 updates in short panels or 6,000 in four-wave panels.
All optimization jobs have six-hour scheduler limits; the readout has one hour.
These are wall limits, not runtime promises. The sum of this campaign's bounded
output caps is 8.625 GiB; observed WORK3 quota headroom before staging was about
15.84GiB. Admission is not a whole-account reservation against unrelated writers.
[All job IDs, queues, budgets and frozen hashes](../../../outputs/synthesis-followup-20261116/submission-summary.json).

Inspect a plan through the shared entry point, for example:

```powershell
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-selection.json --arm seed-1 --plan-only
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-attraction.json --arm mechanism-1 --plan-only
```


## First two completions, 16 November 2026, 16:52 CET

Fresh scheduler and terminal-file reconciliation: **2 completed, 2 running,
15 queued, no failures observed**. Replay seed 1 **30078381** completed all
four waves/18,000 updates in 3 h 10 min with launcher exit 0. Readout **30078398**
completed all 12 scores in 210s, launcher exit 0, although it has already left the
recent scheduler listing. Terminal receipts establish its completion.
Selection seed 1 **30078310** and seed 2 **30078311** are RUN on V100/L40S,
both in their fourth/final wave; snapshots show 19,937/24,000 and 23,876/24,000
optimizer updates respectively. Fresh evaluation also takes time. The aligned
attraction and combined-method panels have not started. No jobs modified.
[Full result/progress snapshot](../../../outputs/synthesis-followup-status-20261116T155016Z/remote.jsonl),
[reconciled status](../../../outputs/synthesis-followup-status-20261116T155016Z/summary.json).

Saved-molecule scores across three geometry seeds (lower loss is better):

| Point | SA | Mean fresh loss | Range |
| --- | ---: | ---: | --- |
| Selected parent |6.127|2.16654|2.16580–2.16735|
| Saved useful intermediate |4.217|2.20191|2.20113–2.20284|
| Same branch's terminal |6.758|2.23171|2.23134–2.23243|
| Older attraction endpoint |4.363|2.49877|2.49695–2.50006|

The intermediate's chemistry/binding trade-off is stable over these conformers;
it beats its later endpoint on both metrics. The previously unscored attraction
endpoint fails the illustrative binding≤2.3 cutoff, so its low SA was not enough
to call it a useful joint-quality success. These geometry seeds are not new
optimization seeds and do not establish synthesis feasibility.

Replay panel: terminal Pareto, reserved-SA-slot terminal, and joint-intermediate
replay all produce **2 distinct/diverse candidates at SA≤5/loss≤2.3**, with the
same minimum archive SA 4.1112 and historical-best binding 2.051323. Joint replay
produces 4 candidates at the looser SA≤5/loss≤2.4 cutoff versus 2 for the terminal
policies, with 99.27% versus 99.73% valid states. Actual fresh attempts/successes
are 67/62 versus 66/61. The best SA appeared before the recovery wave; recovery
improves breadth slightly in this seed, not the best strict result. Two seeds
remain queued, so no policy is established as a winner.


## Seventeen completed, 17 November 2026, 20:16 CET

Scheduler/terminal receipts reconcile **17 completed successfully, 0 running,
2 queued, no failures observed**. This is 16 optimization panels/238,500
completed Protenix updates plus the readout. The remaining A40 jobs are
**30078631** (screening seed 1, 7,500 updates) and **30078660** (combined seed 3,
24,000 updates). No jobs changed.
[Reconciled status](../../../outputs/synthesis-followup-status-20261117T191613Z/summary.json),
[complete result/launch records](../../../outputs/synthesis-followup-status-20261117T191613Z/remote.jsonl).

Corrected-bank mechanism panels are complete across three seeds. At the primary
SA≤5/fresh-loss≤2.3 cutoff, summed per-seed yields are SynFrag-only 7, broad low-SA 7,
local low-SA 6, local unfiltered 7, centroid 3. Each arm consumed 54 fresh calls over
three seeds. These sums are not globally unique molecule counts. No consistent
local/low-SA attraction benefit over the controls is established.

In the separate strength panels, attraction 0.025 yields 3/2/3 qualifying
candidates, versus 2/1/0 at 0.05, 1/2/1 at 0.1 and 1/1/2 at 0.2. Weak attraction wins
against 0.05 in each matched seed, but this panel lacks a no-attraction arm.
Increasing 0.05→0.2 lowers selected mean SA 6.65→5.85 while validity drops
99.62%→90.60%; it does not improve strict joint yield. Two screening seeds
show binding/SA shortlists finding 1/3 qualifying candidates versus 0/0 under
binding-focused screening, with 36 fresh calls in each arm. This changes selected
sample composition, not the optimization loss. Anchor-only trails the unguided
baseline on joint yield in these two seeds.
[Attraction comparison audit](../../../outputs/synthesis-followup-status-20261117T191613Z/attraction-analysis.json).

Selection/schedule counts at SA≤5/loss≤2.3 across three completed seeds:

| Policy | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: |
| Constant, binding |2|4|3|
| Constant, Pareto |2|3|4|
| Wave-focused, binding |3|7|2|
| Wave-focused, Pareto |1|5|7|

Wave focus suggests a seed-dependent yield benefit, not a consistent Pareto
winner. Actual fresh calls sometimes differ. Strict replay counts are terminal
Pareto 2/3/4, reserved-SA-slot 2/3/4, joint replay 2/3/3; the earlier loose-threshold
replay hint does not become a robust strict-cutoff win. Combined methods remain
mixed over two seeds: binding+attraction 4/8 versus binding-alone 5/2 strict joint
counts, with an improvement at the looser 2.4 cutoff in both seeds. The third
combined seed is queued. No arm materially beats the best founder binding loss.
[Selection/replay/combination audit](../../../outputs/synthesis-followup-status-20261117T191613Z/selection-analysis.json).

One projected-guidance seed reports SA 1.1607/loss 2.12854. Its exact saved SMILES
was retrieved and the SA was reproduced locally with RDKit. However, the graph
has nine rings and native hard DFR≈5.89; the low SA alone must not be presented
as a practical synthesis breakthrough. This is an outlier for independent
assessment, not an established policy advantage.
[Exact molecule/provenance](../../../outputs/synthesis-followup-status-20261117T191613Z/low-sa-outlier.json).


## Remaining A40 jobs moved to V100, 17 November 2026, 20:31 CET

At the user's request, the final two unstarted A40 jobs were replaced on
**gpuv100**, using the proven **12 GiB** exclusive GPU request, four CPU slots
and the same six-hour wall limits. V100 had idle, schedulable hosts at the queue
snapshot, whereas A40 reported a host job-slot limit. This is a queue-choice
assessment, not a promised start time.

| Experiment | Cancelled original | Current V100 job | Verified state |
| --- | ---: | ---: | --- |
| Screening seed 1 |30078631|**30099741**|PEND|
| Combined attraction/selection seed 3 |30078660|**30099744**|PEND|

In-place migration was unavailable: this site's `bstop` is disabled and `bmod`
is restricted to administrators. Replacements were therefore submitted held,
originals were rechecked as unstarted and cancelled, their EXIT/no-start states
were reconciled, then replacements were released. No output or launch receipt
existed before release. One unused held attempt 30099736 was cancelled before
starting while reconciling delayed scheduler visibility; it did no scientific
work. All three cancelled job IDs were verified EXIT with empty START_TIME.

Original source archive digests were rechecked. Normalizing only the hold,
queue, GPU request and job-name scheduler fields reproduces the original argv
exactly: config/plan hashes, seeds, parent panels, checkpoints, update/fresh-score
budgets, output paths and scientific code are unchanged. Sealed source and plans
retain their historical A40 metadata; the synthesis runtime does not enforce
that queue field. Original submission receipts remain intact, and the current
campaign manifest records the replacement mapping.
[Migration evidence and final states](../../../outputs/synthesis-queue-move-20261117T192347Z/migration-summary.json).

## Audited synthesis scores and validity, 17 November 2026

The installed RDKit 2024.09.6 SA implementation applies a discontinuous logarithm above a scaled score of 8. The nine-ring example enters that transformation at 8.00107086 and leaves at 1.16070550. Reproducing the reported value with the same package had missed this defect. The audit retains both the native output and the scaled input to the transformation. At our thresholds of 5 and 6, checking the input removes hard-side entries that the logarithm dropped below the threshold. The scaled input is a diagnostic heuristic. Route feasibility remains unmeasured. [Versioned RDKit implementation](https://github.com/rdkit/rdkit/blob/Release_2024_09_6/Contrib/SA_Score/sascorer.py#L96-L109) and [upstream issue](https://github.com/rdkit/rdkit/issues/8251).

Read-only receipt and trajectory analysis covered 16 completed optimization panels in the current follow-up, 477 branches, 238,500 updates and 2,744 selected entries. The readout had finished separately. Screening seed 1 and combined seed 3 remained queued at 20:45 CET. We recomputed both values on every distinct valid visited SMILES, verified all archived selected SA values, reproduced each original per-arm joint count, excluded the three campaign founders, and deduplicated across jobs and methods using the best observed fresh loss per identity. No new model evaluations were performed. Results describe molecules these policies actually visited. Historical shortlist and parent decisions used native SA, so their counterfactual outcomes with a revised ranking are unknown.

| Threshold | Distinct visited after audit | Received a successful fresh score | Fresh loss at most 2.3, native / audited | Fresh loss at most 2.4, audited |
| --- | ---: | ---: | ---: | ---: |
| SA at most 5 | 509 | 64 | 41 / 0 | 17 |
| SA at most 6 | 1,287 | 102 | 85 / 6 | 38 |

The denominator is 7,921 distinct, nonfounder valid molecules visited, including 694 with successful fresh scores. The audited SA 4.9105 molecule from job 30078658, cycle 4 step 490, had loss 2.30248. SA 5.55985 and loss 2.22655 co-occurred in the focused-Pareto arm of 30078310 at 99.35% valid trajectory states. An independent example at step 313 of a centroid child in mechanism seed 2 achieved SA 4.855 and loss 2.40924 with all 501 states valid. In combined seed 2 the archive also contains an audited SA 2.2391 molecule at loss 2.38417.

Increasing local attraction from 0.025 to 0.2 cut mean trajectory validity from 99.58% to 90.60%. In seed order, distinct *visited* SA-at-most-6 identities rose from 1/12/1 to 105/63/77. Both arms recorded zero audited hits at SA at most 6 and loss at most 2.3 across their 54 fresh attempts per arm. Stronger guidance generates easier valid structures while spending more time invalid. Its joint binding benefit remains unestablished. The historical preference for 0.025 on the strict SA-5 count disappears.

Across nine older short panels, 7 apparent SA-at-most-5 and loss-at-most-2.3 hits became zero, and 14 at SA at most 6 became one. Strong DFR produced true SA-at-most-5 candidates in all three seeds from the same founder, at losses 2.464 to 2.677. The earlier four-wave pilot had zero audited SA-at-most-6 hits at loss at most 2.3. Its frequently cited intermediate moved from an audited 8.154 at the parent to 8.023 at the intermediate, despite the reported 6.127 to 4.217. The independent attraction endpoint kept its 4.363 but had mean loss 2.499. In the ten-update synthesis-only diagnostic, seven of eight valid endpoints improved before smoothing; none finished at or below 6.

For a bounded follow-up, rank saved valid molecules with the audited hard-molecule threshold, fresh-score a diverse subset of unscored candidates, and compare archive-matched control selections. Then compare difficult binding-first founders against accessible founders under the same gradient search, charging founder scores to the allowance and counting only new identities. The existing aligned low-SA bank contains 16 verified candidate starts with audited SA 1.75 to 2.99; the archived near-miss offers a start closer to the joint threshold. Use a fixed-budget independent route-planning readout and report per-seed yields at both binding cutoffs, actual fresh calls, and path validity.

[Follow-up counts, per-arm means, and paired molecules](../../../outputs/synthesis-sa-audit-20261117/followup-audit.json) |
[Score comparison figure](../../../outputs/synthesis-sa-audit-20261117/synthesis-score-audit.png) |
[Read-only evidence collection](../../../outputs/synthesis-sa-audit-20261117/collect-all.py) and [local reanalysis](../../../outputs/synthesis-sa-audit-20261117/analyze-followup.py) |
[Earlier short pilots](../../../outputs/synthesis-sa-audit-20261117/old-short-summary.json) |
[Earlier waves and conformer readout](../../../outputs/synthesis-sa-audit-20261117/old-waves-readout.json) |
[Synthesis-only diagnostic](../../../outputs/synthesis-sa-audit-20261117/old-descent-diagnostic.json).

## Final screening and combined seeds, 18 November 2026

Both V100 replacement jobs completed with launcher return code 0 by 10:22 CET, closing all 19 follow-up panels. Screening seed 1 (30099741) completed 7,500 updates and 89 successful fresh scores from 90 attempts. All five arms had zero distinct nonfounder molecules at audited SA at most 5 and fresh loss at most 2.3.

Combined seed 3 (30099744) completed all four waves and 24,000 updates. The binding-selection arm with local attraction produced two distinct wave-three hits. The audited SA and fresh loss pairs were 4.75552 / 2.28433 and 4.96033 / 2.25764. Neither identity appeared among the 7,921 nonfounder graphs in the earlier trajectory audit or the 48 archive-screen candidates. Their radius-two 2,048-bit Morgan similarity was 0.516. Binding with attraction logged 98.22% valid states and 70 successful scores from 72 attempts; binding alone logged 98.99% and 57 from 62. At SA at most 6 and the same loss limit their yields were seven and one. All four methods ran 6,000 updates each. The other three arms produced zero strict SA-at-most-5 hits. This is one seed showing joint improvement. Both other combined seeds had zero strict hits under the audited metric. Route feasibility of these new molecules has not been assessed here.

[Final combined seed with audited hit counts and identities](../../../outputs/synthesis-progress-20261118-earlier/combined-final-audited.json) · [Screening seed audit](../../../outputs/synthesis-progress-20261118-earlier/audited.json).
