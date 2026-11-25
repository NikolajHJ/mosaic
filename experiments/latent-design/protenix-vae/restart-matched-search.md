# Does adaptive restart search beat continuous and independent optimization?

The [matched config](restart-matched-search.json) runs three seed panels.
Each seed puts all methods on the same GPU and uses the frozen epoch-45
decoder and BRD4 Protenix scorer. The methods share LR 60, cap 0.25, fixed
25-slot occupancy, chemistry and valence weights of 1, temperature 1, and
zero optimization noise. There are no previously scored founders.

| Method | Allocation per seed |
| --- | --- |
| Adaptive | In wave 1, optimize the same 12 raw prior starts as the independent arm. Later waves use four frontier parents and three 500-update children per parent, with two exact continuation slots per wave. Independent starts fill missing frontier slots and are logged. |
| Continuous | Start from the first common prior. Carry its exact terminal latent and active mask through 96 consecutive 500-update windows. Fresh scores never affect the next state. |
| Independent | Draw 12 new Gaussian latents per wave and fix occupancy from the decoder at each start. Run each for 500 updates with no parent feedback or rejection based on hard graph validity. |

Each method gets 6,000 optimizer updates per wave and 48,000 over eight
waves. For continuous search, window boundaries are used for reporting and
shortlisting. Fixed stateless SGD carries the same optimization state forward.
Including each window's initial and terminal observations gives 48,096
relaxed scorer calls per method.

The [three-page comparison PDF](../../../outputs/restart-matched-report-20261116/search-method-comparison.pdf)
summarizes all three completed corrected panels, with per-seed curves, final
quality and actual scoring/time accounting. It was regenerated from fresh saved
records on 16 November. See the final-three-seed result section below.

Each window contributes up to six distinct candidates under the same mixed
shortlist rule. The corrected `equal_allowance` policy gives each method up
to 72 fresh attempts per wave independently, at most 576 per method. An
empty continuous shortlist does not block the other methods' scoring or
adaptive feedback. Unused allowances and actual attempts/successes are
recorded per method; unused calls are not transferred or filled by repeated
scoring. These are equal allowed budgets, not necessarily equal consumed
fresh-scoring costs. The original submissions below used minimum-pool
matching; their scoring bottleneck is documented in the partial-status section.
The readout compares best fresh loss, mean top-three loss, and distinct
low-loss yield at 24,000 and 48,000 updates. It counts molecules at loss
2.30 or lower with a maximum pairwise Morgan similarity of 0.60. It records
actual relaxed calls, fresh attempts, successful scores, and elapsed time.
The wall forecast can stop a panel at a completed matched wave if the
next one would exceed its allocation.

```powershell
$launcher = 'experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py'
py $launcher stage --name protenix-restart-matched-search-20261110a --config restart-matched-search.json --hours 24 --max-updates 144000
py $launcher submit --name protenix-restart-matched-search-20261110a
```

The [submission directory](../../../outputs/protenix-restart-matched-search-20261110a/)
contains the sealed source, configuration, budget, and scheduler receipts.

DTU accepted all three panels on 10 November 2026. All were pending at
17:48 CET. The two-wave CPU smoke with mock scorers and the related test
suite passed 20 tests. Native results are pending.

| Panel | Seed | Queue | Job |
| --- | ---: | --- | ---: |
| A | 541101 | L40S | 29980366 |
| B | 651101 | A100 | 29980367 |
| C | 761101 | A100 | 29980368 |

At 10:24 CET on 11 November, panel A was running on L40S after starting
at 07:36. One of eight matched waves had completed: 6,000 updates and
seven successful fresh scores per method. Best fresh losses were 2.35332
for adaptive, 2.41079 for continuous, and 2.35332 for independent. Adaptive
and independent deliberately share their wave-one starts, so this is not
yet evidence for adaptive parent selection. Continuous supplied seven
eligible fresh candidates versus 72 in each other arm; matching to that
minimum reduced the scoring budget for all three. Wave two had completed
adaptive optimization and was partway through continuous optimization.
Panels B and C were still pending on A100. No completed panel result exists
yet. [Partial native evidence](../../../outputs/restart-matched-status-20261111T102428/29980366-progress.json).

## Parent and child allocation

The [allocation config](restart-matched-width.json) uses the same seeds,
model, optimizer, and fresh scorer. All three arms use adaptive frontier
selection. Each starts with the same 12 raw prior latents in wave 1.
Waves 2–6 request 6 parents with 2 children, 4 with 3, or 3 with 4.
Each arm uses 12 children of 500 updates per wave and 36,000 updates total.

The frontier uses the same eligibility and diversity rules for every arm.
The runner records realized parent counts, exact continuations, and any
random starts used to fill vacancies. Each wave gives every arm the same
number of exact slots, up to two from distinct parents. It matches actual fresh score
attempts across arms after deduplication, up to 72 attempts per arm per
wave and 432 total. Compare best loss, mean top-three loss, and distinct
low-loss yield at 18,000 and 36,000 updates.

```powershell
$launcher = 'experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py'
py $launcher stage --name protenix-restart-matched-width-20261110a --config restart-matched-width.json --hours 24 --max-updates 108000
py $launcher submit --name protenix-restart-matched-width-20261110a
```

The [submission directory](../../../outputs/protenix-restart-matched-width-20261110a/)
contains the sealed code, budget, and scheduler receipts.

DTU accepted all three allocation panels on 10 November 2026. All were
pending at 20:09 CET. The two-wave mock and related suite passed 21 tests.

| Panel | Seed | Queue | Job |
| --- | ---: | --- | ---: |
| A | 541101 | L40S | 29980890 |
| B | 651101 | A100 | 29980891 |
| C | 761101 | A100 | 29980892 |

## GPU queue replacements on 11 November

At 10:40 CET, the four pending A100 panels were replaced with V100
submissions to broaden scheduling eligibility. DTU rejects user `bmod`
requests, so the originals were cancelled and confirmed terminal before
replacement; none had started or created worker artifacts. The source,
configuration, seeds, scientific budgets, four CPU slots, and 24-hour
wall limits are unchanged. The GPU memory reservation is now 12 GB,
allowing both 16 GB and 32 GB V100s. Scalar optimization previously sampled
about 2.9 GiB, and a long search completed on a 32 GB V100; the fresh-score
peak on 16 GB remains unmeasured. The existing wall-budget forecast can
stop a slow panel at a completed matched wave; use actual completed budgets.

| Comparison | Panel | Original A100 job | Current V100 job |
| --- | --- | ---: | ---: |
| Adaptive / continuous / independent | B | 29980367 | 29994157 |
| Adaptive / continuous / independent | C | 29980368 | 29994216 |
| Parent / child allocation | B | 29980891 | 29994169 |
| Parent / child allocation | C | 29980892 | 29994218 |

All four replacements were pending at verification. Matched-search panel A
**29980366** remains running on L40S; allocation panel A **29980890** remains
pending on L40S. Queue lengths and host availability favor broadening the
pool, but do not establish a start-time guarantee.
[Queue snapshots, guarded replacement receipts, and final verification](../../../outputs/restart-queue-move-20261111T103519/).

## Partial status and scoring bottleneck, 11 November 13:06 CET

Panel A **29980366** has completed three matched waves (18,000 updates
per method) and started wave four. All five other panels remain pending:
**29994157**, **29994216**, **29994169**, **29994218** on V100 and
**29980890** on L40S. None of these six panels has finished.

The first wave gave each method seven successful fresh scores. Continuous
offered zero eligible new fresh candidates in both subsequent waves,
whereas adaptive offered 71 and independent 72 in each. The minimum-pool
matching rule therefore gave every method zero fresh attempts in waves
two and three. All cumulative best and top-three losses remain at their
wave-one values. This prevents adaptive from adding new scored descendants
to its archive; it is a design bottleneck, not evidence of failed adaptive
optimization. These records cannot establish a useful adaptive-versus-
independent comparison beyond their common bootstrap.

Recommended correction for a separate authorized continuation or repeat:
give methods equal maximum fresh-scoring allowances and record unused
allowance and actual costs, without allowing continuous shortfalls to stop
adaptive/independent feedback. This changes actual-attempt matching and must
be explicit. No active/pending job or sealed source was changed by this
status check. [Scheduler and per-wave evidence](../../../outputs/restart-matched-status-20261111T130600/status.json).

## Corrected submissions on 11 November, 13:15 CET

At the user's request, the three affected adaptive/continuous/independent
panels were cancelled and replaced from fresh initializations. All retired
jobs were confirmed `EXIT` before submission. Their partial data remain at
the original paths; the new campaign has distinct outputs and receipts.
Parent/child allocation jobs **29980890**, **29994169**, **29994218** were
left unchanged.

| Panel | Seed | Retired job | Corrected job | Queue | Verified status |
| --- | ---: | ---: | ---: | --- | --- |
| A | 541101 | 29980366 | 29998584 | L40S | PEND |
| B | 651101 | 29994157 | 29998585 | V100 | PEND |
| C | 761101 | 29994216 | 29998586 | V100 | PEND |

The corrected runs keep LR 60, 500 updates per window, eight waves,
48,000 updates per method (144,000 per seed panel), all initial/perturbation
seeds, and the same maximum 576 fresh attempts per method. Configuration
design and search sections were checked equal to the original archive.
Actual fresh-scoring costs can differ with candidate availability; compare
them explicitly alongside update counts and wall time.

The snapshot starts from original archive
`2281af2cb38a4c7fa99f05e7cbaba458992362c1d783e000a2b4b667d91e17df`.
Only the matched runner, its config and launch plan differ; unrelated current
working-tree changes were excluded. New archive:
`64213fd4e94bdeb48e4bc78128d12617456ed018f7dd3854b1eb8004f7e4f17b`.
The runner retains historical `matched_minimum` mode for the allocation
study, and records policy, allowance, unused allowance and actual scores
per method. Three CPU regression cases passed against the isolated snapshot,
including continued scoring/archive updates with zero continuous candidates
and the existing allocation behavior. No new native result is claimed yet.

[Submission receipts, source delta and verified scheduler state](../../../outputs/protenix-restart-matched-search-allowance-20261111a/).

## Progress on 11 November, 23:09 CET

Corrected adaptive/continuous/independent jobs **29998584–29998586** remain
pending. Allocation panel A **29980890** started on L40S at 15:31 CET,
and panel B **29994169** on V100 at 14:32 CET. Both completed three of
six matched waves (18,000 updates and 216 fresh attempts per allocation)
and are in wave four. Panel C **29994218** remains pending. No allocation
shortlist shortfalls occurred in the completed waves.

| Allocation | Seed A best / top-three mean | Seed B best / top-three mean |
| --- | --- | --- |
| 6 parents × 2 children | 2.18608 / 2.18774 | 2.13872 / 2.14209 |
| 4 × 3 | 2.13539 / 2.15418 | 2.11417 / 2.11869 |
| 3 × 4 | 2.12226 / 2.13542 | 2.11417 / 2.11869 |

These partial results favor more children for best score. Diverse low-loss
counts differ: A has 30/26/21 and B has 39/58/49 for 6×2/4×3/3×4.
Every requested parent slot was filled in waves two and three, with no
independent fallback starts. Successful scoring counts are slightly unequal:
A 207/211/212 and B 211/210/211 at 216 attempts each. These are two partial
seed results, not a completed allocation recommendation.

Observed wave times suggest panel A may finish around 03:30 CET and panel B
around 07:30 CET on 12 November, if the current pace holds. These are runtime
extrapolations, not scheduler start estimates for the queued jobs.
[Scheduler and per-wave snapshot](../../../outputs/restart-status-20261111T230855/).

## Progress on 12 November, 00:44 CET

Corrected method-comparison jobs **29998584/29998585/29998586** are still
pending (L40S/V100/V100), with no initialization or results yet. Allocation
panel A **29980890** has four of six waves complete (24,000 updates and
287 fresh attempts per allocation) and is optimizing wave five. Its best
fresh losses are **2.15770 / 2.13539 / 2.12226** for 6×2 / 4×3 / 3×4.
Wave four had one missing scoring candidate, so all allocations received
71 attempts under that separate study's unchanged matching rule.

Allocation panel B **29994169** is in wave four, with three completed waves
and best losses **2.13872 / 2.11417 / 2.11417**. Panel C **29994218** remains
pending. No allocation panel has finished. No jobs or sealed sources changed.
[Fresh scheduler and per-wave evidence](../../../outputs/restart-status-20261112T004451/remote.jsonl).

## Two allocation panels complete, 12 November 09:35 CET

Allocation A **29980890** and B **29994169** completed all six waves,
36,000 updates per allocation, with zero launcher return codes. A took
11 h 50 min on L40S; B took 17 h 01 min on V100. Actual fresh attempts
were matched within each panel: 419 per allocation for A and 430 for B
(432 maximum). Successful scores differ: A 404/414/415 and B 414/423/413.

| Allocation | A best / top-three mean | B best / top-three mean | Diverse low-loss count A / B |
| --- | --- | --- | --- |
| 6×2 | 2.15770 / 2.16273 | 2.09097 / 2.09312 | 102 / 134 |
| 4×3 | 2.13539 / 2.13634 | 2.10123 / 2.11055 | 78 / 157 |
| 3×4 | 2.12226 / 2.12411 | 2.11417 / 2.11563 | 83 / 96 |

The seed-dependent ranking supersedes the earlier partial preference for
more children. Seed B's 6×2 continued improving late, whereas A's 3×4 kept
its lead. Two completed seeds do not establish a winning allocation.
Allocation C **29994218** started on V100 at 09:22:46 and is in wave one.

Corrected method-comparison B **29998585** started on V100 at 09:31:46;
its frozen archive is the corrected equal-allowance source and the first
adaptive child reached step 459 at the snapshot. No matched wave is complete.
A/C **29998584/29998586** remain pending. No jobs or sealed sources changed.
[Scheduler, terminal receipts and full wave records](../../../outputs/restart-status-20261112T093551/remote.jsonl).

## Corrected method panels A/B complete, 13 November 14:50 CET

**29998584 / 29998585** completed all eight waves with zero launcher
return codes, 48,000 updates and 48,096 relaxed calls per method. Equal
fresh allowances were retained (576 maximum per method); archive feedback
continued despite continuous generating fewer new candidates.

| Seed / method | Best fresh loss ↓ | Top-three mean ↓ | Diverse low-loss count | Actual fresh attempts / successes |
| --- | ---: | ---: | ---: | --- |
| A adaptive | 2.12258 | 2.12678 | 106 | 532 / 527 |
| A continuous | 2.28249 | 2.29597 | 2 | 82 / 82 |
| A independent | 2.18616 | 2.19023 | 26 | 576 / 562 |
| B adaptive | 2.09676 | 2.09981 | 214 | 566 / 559 |
| B continuous | 2.26136 | 2.29463 | 1 | 34 / 34 |
| B independent | 2.17370 | 2.20008 | 25 | 564 / 552 |

Adaptive beats independent on all three readouts in both completed seeds,
with similar or fewer actual fresh attempts. This is encouraging evidence
for parent feedback at the tested update budget. Actual total scoring costs
are not equal, especially against continuous; this remains an equal-update,
equal-allowance comparison. Diversity uses the recorded loss≤2.30 and greedy
fingerprint similarity≤0.60 filter. Seed C **29998586** started V100 at
10:46:46 today, has one completed wave, and is optimizing wave two. Its
wave-one adaptive/independent results are their common bootstrap.

Allocation C **29994218** also completed all six waves, 36,000 updates and
392 fresh attempts per configuration. For 6×2 / 4×3 / 3×4, best losses are
**2.17456 / 2.13091 / 2.14031**, top-three means
**2.17639 / 2.15719 / 2.14309**, and diverse counts **42 / 87 / 103**.
Successful scores were 380 / 382 / 386. Each allocation wins best loss on
one seed across A/B/C; there is no consistent allocation winner.
All three allocation panels are complete. No jobs or sealed source changed.

[Fresh scheduler, terminal receipts and all wave records](../../../outputs/restart-status-20261113T145039/remote.jsonl).

## All corrected method panels complete, 16 November collection

Fresh read-only collection confirms A/B/C **29998584/29998585/29998586** all
completed eight waves, **48,000 updates and 48,096 relaxed calls per method**,
with zero launcher return codes. Final seed C adaptive/continuous/independent
best fresh losses are **2.13091 / 2.30458 / 2.20542**; top-three means
**2.15719 / 2.31640 / 2.20710**; diverse low-loss counts **97 / 0 / 16**.
Its actual fresh attempts/successes are **474/464, 54/54, 563/555**.

Adaptive wins best loss, top-three mean and diverse low-loss yield in all three
seeds. Across-seed mean best losses are adaptive **2.11675**, independent
**2.18843**, continuous **2.28281**. Actual total fresh attempts were
**1,572 / 1,703 / 170** in that order, against 1,728 allowed each. Thus this is
equal updates and allowances, not identical consumed scoring or GPU-hour costs.
Some trajectories plateau early; seed C adaptive best is unchanged after wave 3.
Results concern BRD4/Protenix, not the new Boltz2 panel, affinity or synthesis.

[PDF](../../../outputs/restart-matched-report-20261116/search-method-comparison.pdf),
[compact native records](../../../outputs/restart-matched-report-20261116/data/remote-compact.json),
[derived report data](../../../outputs/restart-matched-report-20261116/data/summary.json).
The report builder `restart-report/method_comparison.py` reads saved JSON only;
run its `figures` phase with matplotlib and `pdf` phase with reportlab, using
`--snapshot outputs/restart-matched-report-20261116/data/remote.json --output outputs/restart-matched-report-20261116`.
The PDF has page-navigation links and three bookmarks; all pages were rendered
and inspected. No model inference, new optimization, or job changes were made.


## All three corrected method panels complete, 16 November 2026

Seed C **29998586** completed all eight waves: 48,000 updates and 48,096 relaxed
calls per method, zero launcher return code, about 21.1 hours. Its adaptive /
continuous / independent best losses are **2.13091 / 2.30458 / 2.20542**;
top-three means **2.15719 / 2.31640 / 2.20710**; diverse low-loss counts
**97 / 0 / 16**. Actual fresh attempts were **474 / 54 / 563**, successes
**464 / 54 / 555**, from equal 576-call allowances. Adaptive wins all three
readouts in all three completed seeds, with fewer consumed fresh calls than
independent in each seed. This is an equal-update/equal-allowance finding,
not equal consumed scoring cost. In C, best loss was reached in wave three;
later waves grew diverse low-loss yield from 40 to 97 without improving that best.
No current account job name contains `restart`; no jobs changed.
[Compact result](../../../outputs/restart-comparison-status-20261116-100902/summary.json),
[full terminal and scheduler evidence](../../../outputs/restart-comparison-status-20261116-100902/remote.json).
