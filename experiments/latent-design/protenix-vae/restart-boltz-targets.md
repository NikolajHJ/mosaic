# Seven-target Boltz2 DBMol restart search

## Current direction, 16 November

The single-target trial is now implemented and submitted as **30082565** on a
32 GB V100, pending at 17:27 CET. It uses FP32, LR 1, 400-update children,
3 parents x 2 children, and a 22-hour working budget inside a 24-hour allocation.
The full 31-atom native BRD4/JQ1 context prepared successfully. The six-wave
nominal target can expand to a 20-wave ceiling if measured pace permits.
See the [current run record](restart-boltz-brd4.md) for the exact configuration,
receipts, validation and reference-codec warning. Existing jobs are unchanged.

The following planning rationale preceded that submission.

The user has deferred the seven-protein search campaign and requested one
substantial search on the familiar protein/ligand pair first. The original
search target is **BRD4 / 3MXF chain A, JQ1 A1**; the recent precision pilots
instead use ERK2. The seven full searches remain unsubmitted.

Requested duration is roughly 22 hours inside a 24-hour allocation, with V100
preferred. V100 has no native BF16 tensor-core support, so a V100 plan uses
FP32; native BF16 acceleration requires a newer GPU. Current VAE/full-DBMol
BRD4 timing on V100 is not measured. An older three-update Mosaic V100 smoke
used a different raw-logit/predicted-pose pipeline and cannot fix this budget.

The user specified **300-500 updates per wave and LR1** for this new long trial;
the proposed interpretation is per child within a wave, as in the earlier cycles.
Proposed structure, not yet submitted: **3 parents x 2 children**, six initial
starts, **400 updates/child**, nominally **6 waves / 14,400 updates**, one GPU
worker, and a **22-hour wave-aware time budget / 24-hour scheduler limit**.
At 5 seconds/update six waves use 20 optimization hours before scoring/setup;
at 10 seconds/update only three waves fit that optimizer budget. A maximum of
20 waves is only a ceiling for time-led depth, not a predicted completed count.
Actual completed waves must follow measured pace and the remaining-time forecast.
Exact runtime is not promised, and wave boundaries can leave unused time.
This does not alter the already submitted paired 200-update precision pilots.

BRD4 uses its own newly prepared native context. The old generated searches used
25 occupied slots, while crystal JQ1 has 31 heavy atoms; that mapping/occupancy
is explicit in the new configuration. No seven-target context or old Mosaic
serialized object was substituted.
[Target, readiness and timing evidence](../../../outputs/boltz-single-target-plan-20261116a/summary.json),
[budget arithmetic](../../../outputs/boltz-single-target-plan-20261116a/budget-proposal.json).

## Earlier seven-target campaign

12 November 2026. Generate ligands for the seven chain-A crystal pockets in
[the target catalog](restart-targets.json). The user selected **Boltz2 with the
full DBMol loss**, **learning rate 1**, and a shorter baseline sized after a
native timing pilot. The previous Protenix staging was never submitted and is
marked superseded in its local receipt directories.

Keep six generations, three parents and four children per parent: twelve random
latent starts in generation one, then twelve child windows per generation.
The update count per window is pending timing. The old 500-update value in
[the baseline template](restart-multitarget.json) is protected by
`budget_status=awaiting_boltz_timing`; it must not be submitted as the agreed
short baseline. Optimization noise is off. Initial settings retain constant
chemistry/valence weights 1 and step-norm cap 0.25.

The [timing pilot](restart-boltz-timing.json) uses one eight-update branch per
target, LR 1, one reference-fixture hard forward, and at most one generated
hard-candidate score. It records full iteration times, loss components and peak
GPU allocations. First two and terminal observations are excluded from the
steady update timing. These pilots are preparation/performance evidence, not
tests of six-generation search quality.

## Model, coordinates and selection

The generic DBMol implementation uses separate pretrained affinity and
structure/confidence Boltz2 trunks, zero recycling, fixed crystal coordinates,
and no structure diffusion. The native affinity crop retains up to 200 protein
tokens, with every requested pocket residue verified. Actual modified protein
CCD residues are retained, including CME159 in 4ZZN and CSO303 in 5L2M.
Expected formal charge is the existing explicit lossy graph channel.

Both optimization and detached hard ranking use `dbmol_total_loss`, with the
existing affinity terms and contact/confidence weights 2.5. Saved atom/charge/
bond labels preserve the coordinate correspondence. Scores are cached by exact
pose-ordered graph labels: identical SMILES with another pose-slot assignment
can be rescored. The molecule archive keeps the best scored assignment per
canonical molecule. Generated outputs must be connected, sanitized and
radical-free; final selection is the best 20 distinct molecules when available,
with the top ten marked primary. No fingerprint separation threshold is imposed.

Native reference fixtures are excluded from generated outputs. Some CCD
references use explicit hydrogen states outside the shared graph codec; these
are recorded as reference round-trip limitations while native chemistry-feature
parity remains required. The generated-molecule validity rules are not relaxed.

## Current evidence

Local selected checks: 15 passed and one optional-gemmi test skipped; the
preparer's focused tests separately passed all three with gemmi available.
These use small CPU fixtures, not native model gradients. They cover graph
packing, attached mock gradients, pose-aware cache keys, best-pose retention,
distinct bootstrap seeds, requested budgets and pocket numbering. A review
also led to releasing the prior Torch model graph before the next forward.

The first allocated CPU preparation, PBS 726693 on `n-62-30-1`, retained the
failed evidence. It found an IOK aromatic explicit-H codec limitation and a
distance-guard problem caused by FP32 `cdist` diagonal cancellation. The guard
was corrected to direct FP64 coordinate differences, with an additional
ligand-to-pocket distance audit.

The second allocated CPU preparation, PBS 726694 on `n-62-30-1`, prepared all
seven contexts in 126.87 seconds. Exact native element/charge/connectivity/
typed-bond seed feature parity passed for all seven. Every requested pocket
survived cropping; ligand-to-pocket distance error was at most 5.82e-7 Å.
IOK's reference-codec limitation remains explicit. The contexts contain
198–200 protein tokens / 215–232 total tokens and occupy about 31–34 MB each.
[Prepared paths/hashes](../../../outputs/boltz-target-preparation-20261112b/summary.json)
and [full input audits](../../../outputs/boltz-target-preparation-20261112b/audits.json)
are retained. Six further focused checks passed with gemmi available. This is
CPU feature/mapping evidence; native GPU timings and gradients are pending.

## Submitted timing pilots

Submitted 12 November, verified at 00:25 CET. All use LR 1, eight optimizer
updates, one bootstrap branch, and a one-hour allocation limit. The alpha
estrogen receptor pilot has started on an A10; the remaining six were queued.

| Target | Job | Queue |
| --- | ---: | --- |
| ER alpha / 2IOK | 30013381 | A10 |
| ER alpha Y537S / 2P15 | 30013386 | L40S |
| Glucocerebrosidase / 2V3D | 30013390 | A40 |
| ERK2 / 4ZZN | 30013396 | V100 32 GB |
| ALDH1A1 / 5L2M | 30013400 | L40S |
| p53 Y220C / 3ZME | 30013405 | V100 32 GB |
| VDR H305F / 3A2I | 30013406 | A40 |

Sealed source `8e6fa4176213173ef5046d1c3b4e0e6a5a82a7a6023ae37e08f2aa8bb3817c8e`;
[plan](../../../outputs/boltz-restart-timing-20261112a/gpu-plan.json),
[scheduler snapshot](../../../outputs/boltz-restart-timing-20261112a/scheduler-snapshot-001.json).
The final shorter baseline remains unsubmitted until timing is available.

At 00:29 CET, ER-alpha **30013381 completed successfully**: eight optimizer
updates, finite connected latent gradients and unchanged frozen weights/context.
Six measured steady iterations had median **14.8455 s/update** (mean 14.8617),
peak allocated **15.59 GiB**, peak reserved **16.04 GiB**, on an A10. Startup
took 103.45s; launcher elapsed time was 233.65s. The other six pilots were PEND.
The eight-update trajectory had no valid generated observations, so no generated
hard candidates were scored; the separate native reference-fixture forward
passed. This is a timing/finite-gradient result, not a quality/yield assessment.
At this measured A10 rate, 36,000 updates alone would take about 148.5 hours,
supporting the user's decision to size a shorter baseline after timing.
[First native result](../../../outputs/boltz-restart-timing-20261112a/er-alpha-timing-snapshot-001.json).

## Morning queue migration, 12 November

At 10:22 CET there were no further completions: the six remaining pilots were
still pending. V100 was blocked by the four-GPU user limit, while A40/L40S
reported slot/processor-affinity shortages. The A10 queue and its healthy node
were idle. Six identical pilots were therefore rerouted to A10. All 255 source
files, including the launch wrapper, are byte-identical to the original archive;
only queues/output paths/receipts changed. LR 1, seeds, eight updates, reference
and hard-scoring budgets and prepared-context hashes are unchanged.

Replacements were accepted held. Each original was checked as PEND with no
started outputs, cancelled, confirmed EXIT, and then its replacement released.
The already completed ER-alpha pilot was preserved. At 10:31 CET, ALDH1A1 and
ER-alpha Y537S were RUN on A10; the others were awaiting A10 slots/release.

| Target | Original cancelled job | A10 replacement |
| --- | ---: | ---: |
| ER alpha Y537S | 30013386 | 30016143 |
| Glucocerebrosidase | 30013390 | 30016144 |
| ERK2 | 30013396 | 30016150 |
| ALDH1A1 | 30013400 | 30016152 |
| p53 Y220C | 30013405 | 30016153 |
| VDR H305F | 30013406 | 30016188 |

[Migration receipts and plan](../../../outputs/boltz-restart-timing-20261112b-a10/gpu-plan.json),
[source/config comparison](../../../outputs/boltz-restart-timing-20261112b-a10/source-comparison.json).
The first migration's cancellation-state poll was too short; its replacement
remained held until the old job's EXIT was reconciled. One connection reset
was reconciled against submission receipts before completing the missing held
submission. No duplicate pilot execution was released.

At 14:50 CET on 13 November, all seven timing pilots have completed results
and zero launcher return codes: original ER-alpha **30013381** and A10
replacements **30016143/144/150/152/153/188**. Each completed its bounded
eight-update pilot. These timing runs do not establish search quality; the
longer baseline remains unsubmitted. No job changes were made by this check.
[Collected results and terminal launch receipts](../../../outputs/restart-status-20261113T145039/remote.jsonl).

## Measured timing and recommended baseline, 13 November

Fresh collection at 19:48 CET confirms all seven pilots completed eight
updates with zero launcher return codes. All final timings are on A10s.
Each median uses six steady iterations, excluding the first two and terminal
observations. Peak memory is allocated GPU memory, not just parameter size.

| Target | Median seconds/update | Peak GiB | Distinct valid pilot states | First valid step |
| --- | ---: | ---: | ---: | ---: |
| ER alpha | 14.85 | 15.59 | 0 | — |
| ER alpha Y537S | 15.18 | 15.75 | 0 | — |
| Glucocerebrosidase | 13.22 | 14.13 | 1 | 8 |
| ERK2 | 14.22 | 15.11 | 4 | 3 |
| ALDH1A1 | 14.69 | 15.16 | 0 | — |
| p53 Y220C | 13.80 | 14.59 | 4 | 1 |
| VDR H305F | 15.64 | 15.66 | 0 | — |

Recommend **50 updates per child per generation**, LR 1, as the initial
six-generation baseline. Twelve initial trajectories and five subsequent
3-parent × 4-child generations give 72 windows, hence **3,600 planned updates
per protein**. Measured optimization time alone is 13.2–15.6 GPU hours; allow
roughly **14–18 hours per protein** for terminal evaluations, hard scoring,
initialization and ordinary variation. Queue waiting is additional. These
are extrapolations from short pilots, not guaranteed walltimes.

| Updates per child | Total planned updates/protein | Optimization-only A10 hours |
| --- | ---: | ---: |
| 32 | 2,304 | 8.5–10.0 |
| 50 | 3,600 | 13.2–15.6 |
| 100 | 7,200 | 26.4–31.3 |

Fifty leaves useful margin under a 24-hour allocation; 100 would exceed that
limit at the observed speeds before scoring overhead. However, only three
targets produced valid states within eight updates. Each scored one generated
molecule, and the other four archives were empty. Timing success does not
establish that 50 updates will fill three parent slots or yield twenty distinct
valid molecules. Inspect validity and frontier size after the first full
generation and report actual executed budgets and output counts. The baseline
template is still unsubmitted; no existing job settings were changed here.

[Collected native results and traces](../../../outputs/boltz-target-timing-analysis-20261113/snapshot-001.json)
and [derived budgets](../../../outputs/boltz-target-timing-analysis-20261113/analysis.json).

## Selected 200-update baseline

The user rejected 50 updates as too short and selected **200 updates per child**
as the minimum for these searches. The current baseline config and target-config
generator now use **LR 1, six generations, three parents × four children**, with
twelve initial trajectories: **72 windows / 14,400 planned updates per protein**.
The constant chemistry/valence weights, zero optimization noise and six hard
scoring allowances per window are retained (432 maximum generated hard scores).
The scheduling horizon is 200; the constant-weight schedule bounds were adjusted
to match. Historical eight-update timing configs and submitted releases are
unchanged. This selection supersedes the earlier 50-update runtime compromise;
neither length has yet been validated for twenty-molecule yield.

Measured A10 rates imply **52.9–62.5 optimization hours per protein**, or roughly
**55–70 GPU hours including overhead**, excluding queue waits. One generation
requires about **8.8–10.4 optimization hours**. A full search therefore requires
multiple allocations under the current 24-hour submission limit. Keep the
23-hour segment budget and completed-generation checkpointing; resume the next
generation in a new output directory with the exact prepared context, scientific
settings and saved-state SHA256. This is existing **manual continuation** support,
not an automatic resubmission chain or mid-child checkpoint. An interrupted
incomplete generation must be replayed. A baseline starts fresh and cannot resume
an eight-update timing-pilot state because the settings are identity-bound.

Seven existing config/search/resume CPU checks passed. These establish the
resolved budget and mocked completed-wave restoration; native Boltz GPU resume
has not been exercised. The seven full-length searches remain unsubmitted.
[Per-target runtime estimates](../../../outputs/boltz-200-update-config-20261113/runtime-estimates.json).

The [parallel-branch speed study](restart-boltz-speed.md) now tests one/two/three
native workers, precision and checkpoint policies before these full searches.
The baseline config remains direct FP32 until measured GPU evidence is available.

Shared execution remains `latentmol.design.protenix_search` (historical module
name), dispatching Boltz setup through `latentmol.design.boltz_restart`. The
search engine and its trajectory/archive/checkpoint logic are reused. The
[Boltz assets](restart-boltz-dtu.json) and all seven reference CCD molecules
already exist on DTU; no model download or training is needed.

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-boltz-timing.json --arm er-alpha --plan-only
```
