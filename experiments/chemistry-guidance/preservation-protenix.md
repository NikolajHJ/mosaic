# Current repair search with isolated preservation penalties

## Completed comparison, 19 November 2026

**Atom 2 / substructure 2 gave the most consistent validity improvement. Atom 2 / substructure 0 was the only tested setting with zero endpoint sulfur gain in all six fixed-search runs. No setting consistently improved validity, sulfur exposure and hard contact loss together.** The fixed-search experiment completed all 21 batch and 21 interactive trajectories. Both executions used the same three distinct starting molecules. The original mixed-search study completed nine additional trajectories, with its three control traces identical to the batch controls. All runs used frozen epoch 45, the native BRD4 Protenix contact loss, LR 70 and 500 updates. The analysis used the saved states. [Configuration](preservation-protenix.json) · [Batch measurements by start](../../outputs/preservation-protenix-20261112a/review-batch-20261119/batch-metrics.json) · [Interactive measurements by start](../../outputs/preservation-protenix-20261112a/review-interactive-20261119/loss-details.json)

### Fixed-search outcomes

Values are **batch / interactive**. Validity is the fraction of the 1,500 raw source graphs that pass the recorded validity check, steps 0–499. Endpoint counts use step 500. Sulfur drift is the change in hard-decoded sulfur count from the sulfur-free start, including invalid graphs. Unique molecules include step 500, pool all three starts within each execution and require a parseable valid SMILES. Morgan distance is mean pairwise 1 − Tanimoto over that unique set. One distinct batch baseline SMILES from a recorded-valid state failed reparsing.

| Atom / structure | Valid sources % | Valid endpoints / 3 | Mean sulfur drift | Mean endpoint sulfur drift | Unique molecules | Morgan distance |
|---|---:|---:|---:|---:|---:|---:|
| Current (0 / 0) | 78.9 / 67.0 | 0 / 1 | 0.523 / 0.746 | +0.67 / +0.67 | 715 / 506 | .8657 / .8787 |
| 0.5 / 0 | 69.7 / 45.4 | 1 / 1 | 0.231 / 1.019 | 0 / +0.67 | 412 / 306 | .8783 / .8722 |
| 0.5 / 0.5 | 72.5 / 62.2 | 3 / 1 | 0.516 / 0.309 | +1.00 / +0.67 | 593 / 516 | .8682 / .8735 |
| 2 / 0 | 76.5 / 50.8 | 1 / 2 | 0.413 / 0.088 | 0 / 0 | 714 / 390 | .8764 / .8726 |
| 2 / 0.1 | 66.0 / 55.8 | 2 / 1 | 0.611 / 0.046 | +0.67 / 0 | 668 / 311 | .8910 / .8634 |
| 2 / 0.5 | 50.5 / 78.9 | 1 / 3 | 0.354 / 0.160 | +0.33 / 0 | 326 / 610 | .8676 / .8718 |
| 2 / 2 | **86.4 / 88.2** | **3 / 3** | 0.509 / 0.551 | +1.00 / +0.67 | 625 / 733 | .8608 / .8692 |

The relaxed loss scores the final decoder probabilities even when the final hard graph is invalid. To evaluate a usable molecule for every run, select the valid intermediate with the lowest relaxed loss and rescore its discrete graph in the same fixed context. Mean initial relaxed and hard losses were −0.1633 and −0.1487. Lower is better for both. These are pocket-contact predictions. This sweep has no fresh-pose or experimental affinity measurements.

| Atom / structure | Final relaxed contact | Selected valid hard contact | Mean protected decisions changed at endpoint | Searches without a repair target on invalid sources |
|---|---:|---:|---:|---:|
| Current (0 / 0) | −.436 / −.344 | −.539 / −.436 | 14.7 / 21.3 | 7 / 8 |
| 0.5 / 0 | −.307 / −.337 | −.422 / −.404 | 20.7 / 20.0 | 10 / 5 |
| 0.5 / 0.5 | −.301 / −.364 | −.525 / −.496 | 15.0 / 21.3 | 4 / 48 |
| 2 / 0 | **−.529** / −.363 | **−.612** / −.369 | 19.7 / 21.0 | 0 / 67 |
| 2 / 0.1 | −.391 / −.336 | −.445 / −.377 | 16.7 / 21.3 | 7 / 11 |
| 2 / 0.5 | −.426 / **−.408** | −.471 / **−.537** | 22.7 / 15.7 | 75 / 6 |
| 2 / 2 | −.467 / −.345 | −.445 / −.383 | 14.7 / 12.3 | 0 / 0 |

![Three panels compare raw validity, mean sulfur drift and selected hard contact across the seven settings in batch and interactive runs](../../outputs/preservation-protenix-20261112a/review-interactive-20261119/weight-comparison.png)

**Validity and binding.** The 2 / 2 setting improved trajectory validity on all three starts in both executions, by 0.4–11.4 percentage points in batch and 7.2–43.8 points interactively. It finished valid six times compared with once for current guidance. Its mean hard contact score was weaker in both executions, and its mean endpoint sulfur gain was at least as large as the control's. At 2 / 0, all six endpoints were sulfur-free and both executions reduced average sulfur exposure. The strongest batch mean hard score at 2 / 0 came from one start, 111410. The other two batch starts scored slightly worse than their controls and the interactive mean was weaker.

**Is 0.5 a good substructure weight?** Holding atom weight at 0.5, adding structure 0.5 increased trajectory validity and improved selected hard contact in both executions relative to 0.5 / 0. Holding atom weight at 2 reversed the conclusion between executions. Adding structure 0.5 to 2 / 0 changed validity by −26.0 points in batch and +28.1 points interactively. For 2 / 0.5, the interactive hard scores beat each corresponding current control, while the batch mean was weaker than current. Pairwise diversity varied less than valid yield. At this setting the number of distinct usable molecules fell from 715 to 326 in batch and rose from 506 to 610 interactively. The observations support an interaction with the atom weight and the trajectory. They give little basis for a universal 0.5.

**Why validity fell.** In the mixed-search experiment atom replacement candidates cost `1 − p(replacement) + atom surcharge`. It also changed the edit families, search depth and target ranking, added charge cost 0.5 and applied continuous preservation. Current / mixed 0.5 / mixed 2 gave 78.9% / 51.1% / 24.9% raw validity and left 7 / 200 / 466 invalid-source searches unresolved. Most failures at each mixed weight came from a single start. On the saved-state replay, current search rescued 2/12 and 7/12 sampled failures respectively, seven of those nine through a third edit. Even after finding a valid target, only 52% / 26% / 17% of next decoded graphs were valid. A valid target steers an update through latent space and does not guarantee where the argmax lands. In fixed-search batch 2 / 0.5, 743 source graphs were invalid. Only 75 searches on invalid inputs failed to produce a target, and 740 of the 743 graphs failed sanitization. Holding search fixed removed one source of failure while leaving the optimizer free to enter and revisit invalid regions. The 0.5 step-norm cap applied on 1,289–1,493 of 1,500 updates across the fixed-search arms and executions. [Mixed-search measurements and transitions](atom-exchange-protenix.md#completed-three-start-comparison-collected-19-november) · [Same-input replay](#completed-same-input-cpu-replay)

**Why current guidance can gain sulfur.** It fixes current atom labels while selecting a repair. Between searches the binding gradient can move the latent across an atom-class boundary. In the original three controls, 53/60 stepwise increases in sulfur followed zero chemistry gradient. The other seven followed a bond-only repair target. On start 111310 the first two additions, at steps 1 and 16, both followed binding-only steps. In the fixed-search experiment the separate penalty on mismatch from the *initial* atom labels is zero for current guidance. Increasing that penalty strengthens a different objective. [Sulfur transitions](../../outputs/exchange-protenix-20261111a/collected-20261119/repair-audit.json)

### What the substructure term protects

The implementation recognizes rings, amide N–C=O, acid/ester O=C–O and sulfonyl O=S=O cores. It averages mismatch from the initial atom labels, charges and all pairs internal to each protected motif. Absent internal bonds contribute to the denominator. The three starts here contained 7, 4 and 4 nonaromatic rings and **zero recognized functional-group cores**. They had 235, 80 and 176 protected decisions, including 158, 35 and 104 absent bonds. At structure weight 0.5, completely losing the probability of a single protected bond costs just **0.00011–0.00031** in the total objective if the other probabilities stay fixed. The entire structure term is bounded by 0.025. The discrete mixed-search surcharge and this averaged differentiable penalty have different units. [Loss implementation](../../src/latentmol/guidance/exchange.py#L218) · [Reference counts and ring outcomes](../../outputs/preservation-protenix-20261112a/review-interactive-20261119/ring-retention.json)

At 2 / 2, **0/15 batch and 3/15 interactive** starting rings retained every mapped atom, charge and internal bond label at the endpoint. Even counting only survival of the original cycle edges, the counts were 0/15 and 5/15. Current guidance retained 1/15 and 0/15 rings exactly. These counts include all endpoints. Aromatic rings could form along the path, but the fraction of distinct valid molecules containing one fell from 37/715 (5.2%) to 16/625 (2.6%) in batch and from 86/506 (17.0%) to 5/733 (0.7%) interactively at 2 / 2. There was no starting aromatic ring from which to measure retention. The frozen geometry-off contact scorer represents bond presence, so changing only the bond order at fixed connectivity does not directly distinguish aromatic and single bonds in that feature. The RDKit chemistry check and the preservation term do see typed bonds. A chemically valid conversion of a reference ring to another structure can therefore pass the validity check and pay only the soft preservation cost. [Batch aromatic counts](../../outputs/preservation-protenix-20261112a/review-batch-20261119/rings-and-control.json) · [Broader aromaticity audit](../latent-design/aromatic-rings.md)

The earlier 48-start chemistry-only test found 48/48 valid endpoints for current, 0.5 / 0.5 and 2 / 0.5. Mean protected changes fell from 4.65 to 2.50 and 2.25. Under binding optimization the best setting for validity still changed 14.7 / 12.3 protected decisions per endpoint. For the intended use, the most informative next comparison would designate an aromatic ring or functional group in each start and score exact retention separately from whole-graph validity and hard contact. A penalty computed once per designated motif from its defining atoms and bonds would test the dilution effect directly. [Chemistry-only comparison](atom-exchange-linear.md)

### Reproducibility across executions

All paired initial hard graphs matched, as did the initial latents on all three controls, the reference identifiers, checkpoint, scorer identity and nominal settings. The initial relaxed losses agreed within 0.00000006. On seed 111210 the baseline binding-gradient norm was 0.5202 in batch and 0.1602 interactively before the first update, with no chemistry gradient or noise. Both first displacements had norm 0.5, but their cosine was 0.9886 and their difference in latent units was 0.0754. The first decoded molecules already differed after that step. On the other two starts the first displacements agreed to within 0.000004 in norm of the difference, and the first observed baseline molecule or validity differences appeared at steps 11 and 7. The execution environments were not experimentally isolated. These repeated trajectories show sensitivity to the numerical path. Only three starting molecules support the rankings. [First-update latent comparison](../../outputs/preservation-protenix-20261112a/review-interactive-20261119/baseline-step-replay.json) · [Complete batch report](../../outputs/preservation-protenix-20261112a/review-batch-20261119/batch-report.json) · [Complete interactive report](../../outputs/preservation-protenix-20261112a/review-interactive-20261119/comparison.json)

## Original question and launch record

12 November 2026. Question: why did validity fall in the mixed-search native
pilot, and what do atom/reference-substructure penalties do when repair discovery
is exactly the current policy? Follow-up to the
[one completed mixed-search start](atom-exchange-protenix.md#preliminary-native-result-one-completed-matched-start).

## What the recorded run establishes

The original comparison changed search space, ranking, edit depth, target choice
and continuous reference preservation together. It cannot attribute the validity
loss to the atom surcharge alone. Current search uses fixed atom/charge labels,
structural proposals and a three-edit bond fallback. Mixed search uses at most
two edits across bond/atom/charge families, a different cost ranking and continued
search after discovering a valid candidate. Both nominally allow 96 checks, but
they spend them differently.

Recorded search outcomes across 500 source states were:

| Arm | Already valid | Repair target found | Search censored |
| --- | ---: | ---: | ---: |
| Current | 414 | 83 | 3 |
| Atom 0.5 / structure 0.5 | 50 | 255 | 195 |
| Atom 2 / structure 0.5 | 188 | 298 | 14 |

These are search-status counts; baseline raw graph sanitization separately logs
415 valid sources, including the known step-72 serialization/canonicalization
failure. The 0.5 arm has a large discovery deficit. The 2 arm's smaller discovery
deficit does not explain its much lower validity by itself. Mixed targets changed
atoms in 87/255 witnesses at cost 0.5 and 10/298 at cost 2; charge changes occurred
in 1/255 and 18/298 respectively. Different targets and continuous reference
forces also change subsequent latent directions. Existing step clipping applies
to almost every update; it cannot be interpreted as an unconstrained LR-70 run.

The current trajectory's first sulfur appeared at step 1 and its second at step
16. Both preceding updates had **zero chemistry gradient**, zero added noise,
and nonzero native binding gradients. Those sulfur additions therefore arose
from binding-driven latent decoding, not atom substitutions by repair.
Once sulfur appears in a decoded graph, fixed-atom repair keeps that current
sulfur label; it does not remember the initial carbon/heteroatom identity.

An infinite atom-edit cost can describe a prohibition *inside a repair search*.
It is not an infinite penalty against changing the initial molecule during
optimization. Moreover, increasing the old mixed coefficient also strengthens
the separate fixed-reference anchor, and the bounded mixed search retains its
different proposal scheduling and radius. Its infinite-weight limit does not
recover current behavior. The new continuous-penalty experiment recovers current
behavior at **zero** penalty, which is tested below. A global atom-identity freeze
would be a different constraint and is not introduced here.

## Completed same-input CPU replay

Allocated CPU job **30016556 DONE**, 99 stratified saved latents, 105.61 seconds.
For each original arm, sample up to 12 evenly spaced observations per recorded
search status and run current, mixed-0.5 and mixed-2 search on identical decoded
probabilities. No native scoring, optimizer updates or old-run writes. All 99
hard label sets matched the saved states exactly; decoder identity and empty
parameter gradients were verified after the replay.

| Source trajectory / sampled failure stratum | Cases | Current search found a target | Current rescues using 3 edits |
| --- | ---: | ---: | ---: |
| Mixed atom 0.5, originally censored | 12 | 2 | 2 |
| Mixed atom 2, originally censored | 12 | 7 | 5 |

These rescues used the existing bond beam; seven needed three edits, outside the
mixed radius. Two needed only two edits, showing that radius alone is not the
whole difference. Both mixed weights still failed all 12 cases in each failure
stratum. Conversely, mixed search rescued one of the three original-current
failures. Neither search dominates every input.

Among 12 sampled successful-repair states per new arm, current search also found
a target on all 12, but the target identity matched only **4/12** at atom 0.5 and
**8/12** at atom 2. The anchor opposed the mixed repair gradient in 6/12 and 3/12;
median anchor/repair gradient-norm ratios were 0.0524 and 0.0711. These are
stratified local observations, not population rates or proof that gradient
opposition caused the trajectory collapse. In particular, 10/12 of the sampled
cost-0.5 failure states also defeated current search. Controlled trajectories
are still needed to separate the effects.

[Replay summary](../../outputs/preservation-protenix-20261112a/audit/result.json),
[individual cases](../../outputs/preservation-protenix-20261112a/audit/cases.jsonl),
[target/gradient comparison](../../outputs/preservation-protenix-20261112a/audit/interpretation.json),
[recorded sulfur transitions](../../outputs/preservation-protenix-20261112a/recorded-trajectory-diagnosis.json).
Downloaded audit archive SHA-256 verified:
`34a33d3efc67580e8e48755f982f7da36c88db284ac237b01436be21eb883603`.

## Controlled native comparison

The opt-in `preservation_guidance` uses **exactly the current repair search,
ranking, budgets, target validation and normalized target NLL**. It adds only:

```text
chemistry = current_repair_NLL
          + 0.05 * [atom_weight * mean_active(1 - p(initial atom))
                    + structure_weight * mean_protected(1 - p(initial label))]
```

Protected labels include atom identity, charge and internal bonds/absent chords
from the same fixed valid initial reference. The independent charge-preservation
coefficient is zero in every arm. Current charge handling inside repair remains
unchanged. This is the continuous preservation component of the previous
implementation, not a new constant cost added to atom proposals: fixed-atom
search has no atom-edit proposals to charge. Search remains detached from
autograd. The current differentiable repair term is NLL; the new mixed proposal
ranking's `1-p` base must not be confused with that NLL.

Substructure weight **0.5 is unvalidated**. Its continuous coefficient is
0.05 × 0.5 = 0.025 on the mean protected mismatch, not a 0.5 jump in total loss
per damaged motif. The previous completed reference contained only rings, so it
also provides no native functional-group-specific conclusion. A finite reference
penalty expresses preservation preference rather than chemical invalidity.

The [config](preservation-protenix.json) runs seven arms per matched start:

| Arm | Atom weight | Structure weight |
| --- | ---: | ---: |
| Current | 0 | 0 |
| Atom half, no structure | 0.5 | 0 |
| Atom half, structure half | 0.5 | 0.5 |
| Atom two, no structure | 2 | 0 |
| Atom two, structure tenth | 2 | 0.1 |
| Atom two, structure half | 2 | 0.5 |
| Atom two, structure two | 2 | 2 |

Three prior screening seeds remain 111210/111310/111410, with the same epoch-45
checkpoint, native context, selection rule, 25 fixed occupied slots, temperature
1, chemistry weight 1, no noise, **LR 70 / 500 updates / step cap 0.5**. Each
cohort reruns its current control on the same allocated device. Total 10,500
optimization updates, ceiling three one-GPU/four-CPU/one-hour allocations.
Measure sulfur, raw/parseable validity, diversity, native relaxed and selected
hard contact, protected changes, censored searches and clipping. Additional
saved diagnostics separate repair/preservation gradient norms and their cosine,
plus binding/preservation cosine.

**80 focused local tests passed.** Identical inputs yield exactly the same
current search receipts, target identity and repair NLL at all tested weights;
zero penalties reproduce the current chemistry/combined gradients exactly on
valid and invalid fixtures. Positive penalties have finite live gradients;
reference persistence across resume, mutual exclusion with mixed search and
legacy config hashes are checked. These CPU tests do not constitute native
derivative qualification.

Submitted 12 November; checked again at **12:23 CET on 12 November**: all three
new GPU jobs remain **PEND**, with no output directories or completed native arms.
A40 reports its job-slot limit, V100 insufficient processors for affinity
placement, and L40S unavailable GPU reservation capacity. The dependent report
is waiting on those jobs. No native weight comparison is available yet, and this
check changed no job or setting.
[Latest scheduler receipt](../../outputs/preservation-protenix-20261112a/queue-details-20261112T1223.json).

| Seed | Job | Queue |
| --- | --- | --- |
| 111310 | 30016566 | gpua40 |
| 111210 | 30016572 | gpuv100 |
| 111410 | 30016586 | gpul40s |

Dependent report **30016589** uses the corrected partial-cohort/SMILES-aware
summarizer. No native outcome is available yet. Source SHA-256:
`3bb44acfc2d49fd0a6e397a027b40e5d72a7721c1f4143f8aa6f0c6740f4fdba`.
[Launch scripts and receipts](../../outputs/preservation-protenix-20261112a/).
Reuse `python -m latentmol.design.exchange_protenix` with the new config inside
the existing `protenix_latent_run.sh` wrapper; the exact allocated commands and
GPU plan are saved with the receipts. Replay command:

```sh
python experiments/chemistry-guidance/audit_exchange_search.py --cohort COMPLETED_COHORT --checkpoint EPOCH45.pt --output NEW_AUDIT --per-status 12
```

No earlier job, callback, source snapshot, cache, checkpoint or default was
changed. These results currently support correcting the comparison, not adopting
a particular atom/substructure weight.

## Interactive execution requested 12 November

At the user's request, started one additional matched cohort (screening seed
111310) on the interactive V100 at 12:30 CET, job **30018947**, host
`n-62-20-1`. The first native updates ran successfully at roughly 0.6 s/update
after startup. This uses the exact source archive, config and checkpoint above:
seven arms, 500 updates each, LR 70, one selected GPU and four allocated CPU
slots, one-hour limit, no GPU memory preallocation. It repeats seed 111310;
it must not be counted as an additional independent start when the queued cohort
finishes. Other jobs and dependent reports remain unchanged.

The [official DTU interactive GPU workflow](https://www.cc.dtu.dk/?page_id=2129)
uses shared nodes. Read-only probes found all A100/SXM2/V100 GPUs busy; V100 GPU0
had the most spare utilization and over 8 GiB free at launch. No exclusive GPU
ownership is claimed. The standard `voltash` wrapper supplies a real LSF
allocation, distributing four task routes across two shared GPUs. A first
attempt (**30018940 EXIT**) was correctly rejected by the batch launcher's
single-GPU routing check before model execution. The new attempt validates and
records those shared allocation routes, then selects the observed GPU UUID for
the single application process and its resident native child. The source,
scientific objective and batch launcher are unchanged. Probe/session receipts
also retain an initial harmless CRLF shell-input failure, corrected by sending
LF bytes through the connector.

Output: `/work3/s234818/latentmol/runs/preservation-protenix-20261112a-interactive-111310-attempt2`.
[Interactive launcher and receipts](../../outputs/preservation-protenix-20261112a/interactive/run-111310-attempt2/).
This is currently execution evidence, not a completed comparison.

### All three starts on interactive nodes

The user subsequently requested all starts interactively. At **12:38 CET on
12 November**, all three cohorts were running, each with its seven arms and
unchanged LR-70/500-update configuration:

| Seed | Interactive job | Host / selected GPU | Verified progress |
| --- | --- | --- | --- |
| 111210 | 30018978 | n-62-12-19 / A100 GPU1 | Current arm step 15 |
| 111310 | 30018947 | n-62-20-1 / V100 GPU0 | Current arm completed 500; atom 0.5 / structure 0 at step 51 |
| 111410 | 30018977 | n-62-20-9 / V100-SXM2 GPU2 | Current arm step 28 |

The two new GPUs had zero instantaneous utilization and ample free memory at
the read-only pre-launch check, with other users' resident processes retained.
Both were started through the official `a100sh`/`sxm2sh` wrappers with four
allocated CPU slots, a one-hour limit and one explicitly selected shared GPU.
Native gradients are finite in the saved progress rows. Runtime can be affected
by shared load; weight comparisons remain paired within each cohort/device.
Every arm reruns its matched control, and no source, checkpoint or loss changed.

New output suffixes are `preservation-protenix-20261112a-interactive-111210`
and `preservation-protenix-20261112a-interactive-111410`, under the same work3 run
root. CPU report **30018980** depends on all three interactive jobs ending and
writes the comparison in control directory
`preservation-protenix-20261112a-interactive-report`. It uses the same frozen
SMILES-aware summarizer and retains incomplete results explicitly.
[Status receipt](../../outputs/preservation-protenix-20261112a/interactive/all-starts/running-status.json),
[111210 launch](../../outputs/preservation-protenix-20261112a/interactive/run-111210/),
[111410 launch](../../outputs/preservation-protenix-20261112a/interactive/run-111410/),
[report receipt](../../outputs/preservation-protenix-20261112a/interactive/report/submission.json).
The older queued cohorts and their report remain untouched. Aggregate the three
interactive seeds once each; do not pool their queued repetitions as independent
starts. The full interactive comparison is still running.
