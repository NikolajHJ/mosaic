# Frozen synthesis guidance pilots — 11 November 2026

The 17 November audit supersedes the historical low-SA hit counts and the interpretation of several individual molecules below. [Audited results](restart-synthesis-followup.md#audited-synthesis-scores-and-validity-17-november-2026).

Question: can pretrained synthesis scorers improve the practical chemistry of
strong Protenix candidates while retaining their contact scores? This work uses
frozen public SynFrag and DFRscore weights, with no predictor training. It stays
on `codex/protenix-restart-search`; original runs and queued comparisons are unchanged.

## Matched Protenix panels

Three panels use identical saved founders: the best molecules from near-valid
29943712, deep 29943705 and wide 29943696. Each variant gets the same screened
latent kick for a given founder and panel seed, followed by 500 updates. LR is
60, displacement cap 0.25, chemistry/valence weights 1, temperature 1, and
optimization noise zero. Each panel contains six variants:

| Variant | Initial synthesis/contact latent-gradient norm ratio |
| --- | --- |
| Baseline | 0 |
| SynFrag weak / strong | 0.1 / 0.5 |
| DFRscore weak / strong | 0.1 / 0.5 |
| Combined | 0.25 for each model |

Each coefficient is calibrated once at the first informative point of each
child and held fixed, with cap 100; flat initial derivatives are recorded.
Contact loss remains the binding readout and archive criterion. Guidance is a
separate opt-in gradient term, not a replacement contact objective.

Each panel has **9,000 updates**, 18 children and at most 108 fresh-score
attempts, or 1,500 updates / 18 attempts per variant. Actual fresh attempts are
matched within founder after deduplication. A founder may be re-scored once
within each variant's real budget, preventing a no-new-identity arm from
forcing every arm to zero scores. Founder rescores and genuinely novel
candidates have separate result fields. All variants in a panel share one GPU.

| Panel | Perturbation seed | Queue | Job |
| --- | ---: | --- | ---: |
| A | 811111 | V100, 12 GB reservation | 29996683 |
| B | 921111 | L40S | 29996684 |
| C | 1031111 | A10, 20 GB reservation | 29996685 |

All three were **pending at 12:17 CET**. The wall limit is six hours per panel.
The batch totals 27,000 updates and at most 324 fresh attempts. Compare SA,
QED, rings, validity, native-feature synthesis scores, novel fresh contact
loss, and best contact loss under SA ceilings 4/5/6. Runtime and actual scoring
successes are logged; this is a selected hard-cage repair question, not a random
molecular-generation benchmark.

The central entry point and shared launcher run the pilot:

```powershell
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-pilot.json --arm seed-a --plan-only
py experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py stage --name protenix-restart-synthesis-pilot-20261111c --config restart-synthesis-pilot.json --hours 6 --max-updates 9000
py experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py submit --name protenix-restart-synthesis-pilot-20261111c
```

[Configuration](restart-synthesis-pilot.json) ·
[Submission/source receipts](../../../outputs/protenix-restart-synthesis-pilot-20261111c/) ·
[Shared founder identities and provenance](../../../outputs/restart-synthesis-pilot-20261111a/parents.json).

## Completed CPU diagnostic

Before the binding pilots, each founder received ten synthesis-only latent
updates with SynFrag, DFRscore, and their balanced combination: **90 updates**
total. LR 60 and cap 0.25 were retained; coefficients were normalized once by
initial synthesis-gradient norms. This deliberately omitted both contact loss
and chemistry guidance to isolate the new gradient's immediate effect.

| Founder | Initial SA | SynFrag endpoint SA | DFR endpoint SA | Combined endpoint SA |
| --- | ---: | ---: | ---: | ---: |
| Near-valid winner | 8.369 | 7.111 | 7.225 | 5.549 |
| Deep winner | 7.495 | 6.898 | 7.447 | Invalid |
| Wide winner | 8.047 | 7.179 | 5.241 | 6.941 |

Eight of nine endpoints were valid; all eight had lower independently
calculated RDKit SA than their paired founder. Both optimized surrogate
losses fell in the combined cases, including the invalid endpoint. This is
an encouraging local signal and a concrete reason to retain chemistry
guidance, not evidence that binding is preserved or routes exist. The diagnostic
took **35.5 seconds** on interactive node n-62-30-7, PBS job 726434.hnode41.

[Molecule report](../../../outputs/restart-synthesis-pilot-20261111a/report/Frozen%20synthesis%20gradient%20diagnostic.html) ·
[Full diagnostic trajectories](../../../outputs/restart-synthesis-pilot-20261111a/cpu-descent.json) ·
[Editable report notes](../../../outputs/restart-synthesis-pilot-20261111a/report/Frozen%20synthesis%20gradient%20diagnostic.md).

## Adapter scope and checks

The shared relaxed graph is consumed without RDKit, hardening or molecular
repair in the live synthesis path. Atom, charge, bond type and bond-existence
paths remain connected. Native-feature reference scores are computed separately
outside autograd. Hydrogen, hybridization, donor/acceptor, conjugation and ring
features have explicitly documented continuous approximations; matching binary
network algebra does not establish full soft-feature or derivative correctness.
Thirty-five focused tests passed; existing search/resume checks also passed.
The diagnostic report passed molecule-count, filtering and JavaScript checks,
and its rendered layout was visually inspected.
DFRscore in particular differs materially on some cage features and can give
counterintuitive scores outside its training domain. Scores are not calibrated
route-success probabilities.

Focused tests and review caught unsupported-edge attention underflow and
float32 cancellation in `1 - P(NONE)`. Existence now sums real-bond probabilities;
absent-edge boundary exponents are bounded and calibration norms use float64.
Draft source A/B were retained without submitting any jobs. The submitted
source C SHA-256 is
`37e12f63cee97aeca7fa2b7747b0a9b080d6ab27a10c1024fb82b6f5d2097fd2`.

The sealed source passed an actual-decoder finite-path smoke on all three
founders on DTU interactive PBS job 726433.hnode41 (2.41 seconds). All six
model/founder latent gradients were finite and connected; latent norms ranged
0.040–0.089 for SynFrag and 1.58–8.69 for DFRscore. DFRscore retains large finite
probability-space derivatives near tiny edge masses; per-child fixed norm
calibration keeps the declared initial contribution controlled. This is an
exploratory usability check, not numerical qualification or proof of synthesis.

[Finite-path receipt](../../../outputs/restart-synthesis-pilot-20261111a/actual-decoder-smoke-c.json) ·
[SynFrag feature audit](../../../outputs/restart-synthesis-pilot-20261111a/synfrag-hard-feature-probe-refined.json) ·
[DFR feature audit](../../../outputs/restart-synthesis-pilot-20261111a/dfr-feature-audit-v2.json) ·
[DFR numerical regression](../../../outputs/restart-synthesis-pilot-20261111a/dfr-actual-graph-stability-v2.json).

## Live progress, 11 November 13:09 CET

Panel C **29996685** started on A10 node n-62-18-5 at 12:28:12. Startup
completed in 46 seconds. Baseline and weak SynFrag each finished their three
500-update children; strong SynFrag had one complete child and a second at
step 316. About 3,816 / 9,000 updates were done. The full matched fresh scoring
stage had not run, so this is optimization progress, not a completed efficacy
comparison. No terminal error appeared in the collected logs. Panels A/B
**29996683 / 29996684** remain pending; no synthesis GPU panel has finished.
[Scheduler, launch receipt and branch evidence](../../../outputs/restart-status-20261111T130951/status.json).

## First completed GPU panel, 12 November 00:44 CET

Panel C **29996685** completed all 9,000 updates and 108 fresh attempts
successfully in 6,705 seconds (about 1 h 52 min). The launcher returned zero;
frozen decoder and synthesis models were reported unchanged. A/B
**29996683/29996684** remain pending on V100/L40S. This is one seed panel,
with three common founder molecules and 1,500 updates per arm.

| Arm | Selected mean SA ↓ | Mean QED ↑ | Valid trajectory states | Top-three new fresh loss ↓ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 7.562 | 0.467 | 100.0% | 2.08566 |
| SynFrag weak | 7.501 | 0.499 | 99.7% | 2.07702 |
| SynFrag strong | 7.598 | 0.514 | 76.8% | 2.07574 |
| DFR weak | 7.265 | 0.573 | 95.0% | 2.06730 |
| DFR strong | 7.201 | 0.563 | 58.9% | 2.06864 |
| Combined | 7.456 | 0.521 | 100.0% | 2.08214 |

Weak DFR is the most promising balance in this panel: better selected SA/QED
and top-three binding-score coverage, with a modest trajectory-validity drop.
Strong guidance causes substantial invalid stretches; the combined setting
does not clearly outperform weak DFR. These selected-set means are proxy
measurements, not demonstrated synthetic routes. No arm beats the best
founder's fresh loss **2.05132**; best novel losses are about 2.06081–2.06278.
Actual successful fresh scores differ (16–18 of 18 attempts per arm), and
some arms include marked founder rescores. No new run or job modification.

[Completed result](../../../outputs/restart-status-20261112T004451/synthesis-c-result.json) ·
[Scheduler, launcher and live comparison progress](../../../outputs/restart-status-20261112T004451/remote.jsonl).

## Progress on 12 November, 09:35 CET

Synthesis A **29996683** started on V100 at 09:23:17. The baseline's three
500-update children finished and weak SynFrag optimization began; no fresh
scores yet. B **29996684** remains pending on L40S. C remains the only
completed synthesis panel. No jobs or source changed.
[Fresh scheduler and branch progress](../../../outputs/restart-status-20261112T093551/remote.jsonl).

## All three GPU panels complete, 13 November 14:50 CET

A **29996683**, B **29996684** and the previously reported C **29996685**
all have completed results and zero launcher return codes. Each finished
9,000 updates and 108 fresh attempts, for 27,000 updates and 324 attempts
across the three seed panels. A took 5,923 seconds, B 4,226 and C 6,705;
GPU families differ, so these times do not compare guide effectiveness.

Weak SynFrag lowered selected mean SA in A/B from baseline 7.476/7.844 to
6.932/7.287 (C 7.562→7.501). Weak DFR was mixed across seeds: A 7.476→7.540,
B 7.844→7.279, C 7.562→7.265. Strong guidance and combinations have variable
validity: strong SynFrag valid-state fractions A/B/C were 23.8%/72.1%/76.8%,
strong DFR 99.7%/63.7%/58.9%, combined 70.9%/65.5%/100%. These are exploratory
chemistry proxy trade-offs, not proven route success. No guide arm beats the
best archived founder's fresh loss 2.05132. Successful novel scores and marked
founder rescores remain separate in the per-panel records.

[All results, receipts and scheduler snapshot](../../../outputs/restart-status-20261113T145039/remote.jsonl).

## Aggregate synthesis/validity trade-off, 13 November

These paired arms reuse three founders under three perturbation seeds: nine
500-update branches per setting. Each arm contributes 54 selected valid
candidate entries (not necessarily unique molecules) and 4,509 recorded
trajectory observations. Validity is hard graph validity; final endpoint
validity and validity throughout the path are different readouts.

| Guidance | Selected mean SA ↓ | SA improvement vs baseline | Valid states | Drop in percentage points | Valid endpoints |
| --- | ---: | ---: | ---: | ---: | --- |
| None | 7.627 | — | 99.80% | — | 9/9 |
| SynFrag weak | 7.240 | 0.387 | 94.68% | 5.12 | 7/9 |
| DFR weak | 7.362 | 0.265 | 98.03% | 1.77 | 7/9 |
| SynFrag strong | 7.372 | 0.255 | 57.57% | 42.23 | 4/9 |
| DFR strong | 7.069 | 0.558 | 74.10% | 25.70 | 5/9 |
| Combined | 7.288 | 0.339 | 78.80% | 21.00 | 7/9 |

The independent SA heuristic favors weak guidance as a practical trade-off.
Weak SynFrag improves selected SA in all three seeds and yields a successfully
fresh-scored SA≤5 candidate in all three, versus none for baseline. Weak DFR
preserves state validity best but slightly worsens selected SA in seed A.
Strong SynFrag has worse average SA than weak SynFrag and a much larger
validity cost. Strong DFR improves SA most, with substantial validity loss.

Detached hard-molecule reference-feature readouts also move: SynFrag raw mean
(higher easier) is 0.00023 baseline, 0.01850 weak and 0.03050 strong. DFR mean
difficulty (lower easier) is 6.06392 baseline, 5.98177 weak and 5.77096 strong.
These are frozen-model outputs, not calibrated synthesis probabilities or
verified routes; original-runtime parity is not established. SA is a separate
heuristic. Means on selected valid candidates omit invalid states and can be
affected by survivor selection, so read them alongside validity and endpoints.

[Aggregated JSON](../../../outputs/restart-synthesis-analysis-20261113T145603/aggregate-tradeoff.json) ·
[CSV](../../../outputs/restart-synthesis-analysis-20261113T145603/aggregate-tradeoff.csv) ·
[Full saved hard-score annotations](../../../outputs/restart-synthesis-analysis-20261113T145603/batches.jsonl).

## Real-compound SA anchors, 13 November

A seven-molecule local CPU calculation using PubChem structures and RDKit
SA gives benzene 1.00, paracetamol 1.41, aspirin 1.58, ibuprofen 2.19,
caffeine 2.30, quinine 4.52 and paclitaxel 5.92. These are examples, not
calibrated feasibility cutoffs. SA primarily captures common fragments and
structural complexity; it neither proposes a route nor proves impossibility.
The pilot selected average around 7.2 remains difficult on this heuristic.
A working triage target of SA≤4, followed by route/building-block evaluation,
is a recommendation, not a validated lab-success threshold or adopted setting.
No jobs or scientific settings changed.
[Structures, software version and calculations](../../../outputs/synthesis-score-benchmarks-20261113T150227/benchmark.json) ·
[Interpretation and primary sources](../../../outputs/synthesis-score-benchmarks-20261113T150227/interpretation.json).
