# Atom-exchange costs under native Protenix optimization

## Completed three-start comparison, collected 19 November

All nine trajectories completed 500 native updates and passed their finite-gradient and frozen-model checks. Each seed used one common start and device for all three arms. All three starts contained zero sulfur. The protected reference motifs in these starts were exclusively rings. The two mixed arms both used structure 0.5 and charge 0.5. The original dependent report hit an RDKit fingerprint error on one baseline SMILES. The current summarizer rebuilt the report locally from the nine completed trajectories. [Full comparison](../../outputs/exchange-protenix-20261111a/collected-20261119/comparison.json) · [Paired measurements](../../outputs/exchange-protenix-20261111a/collected-20261119/arm-summary.json)

| Measurement | Current | Mixed, atom 0.5 | Mixed, atom 2 |
|---|---:|---:|---:|
| Valid source states, first 50 updates | 83.3% | 58.7% | 52.7% |
| Valid source states, all 1,500 per arm | 78.9% | 51.1% | 24.9% |
| Valid endpoints / 3 | 0 | 1 | 1 |
| Mean sulfur atoms across source states | 0.523 | 0.372 | 0.078 |
| Mean endpoint sulfur gain / largest transient gain | +0.667 / +3 | +0.333 / +2 | 0 / +2 |
| Mean endpoint expected sulfur gain | +0.795 | +0.324 | +0.005 |
| Unique parseable valid molecules / scaffolds | 715 / 641 | 480 / 432 | 257 / 231 |
| Morgan distance across distinct parseable valid molecules | 0.8657 | 0.8610 | 0.8697 |
| Mean final atom identities / protected decisions changed | 8.0 / 14.7 | 8.3 / 19.7 | 6.0 / 19.0 |
| Mean final relaxed contact loss | −0.4356 | −0.3699 | −0.3527 |
| Mean hard contact loss of selected valid candidate | −0.5386 | −0.4127 | −0.2680 |
| Unresolved repair searches | 7 | 200 | 466 |

Each arm started at mean relaxed loss −0.1633 and hard loss −0.1487. Lower is better. Current guidance delivered the strongest hard contact on each of the three starts when selecting the valid intermediate with the best relaxed score. The saved hard scores of the two valid step-500 endpoints were −0.2834 (mixed 0.5, seed 111410) and −0.1984 (mixed 2, seed 111310). Endpoint validity and useful intermediate yields measure different properties. One current source failed SMILES reparsing, giving 1,183/1,500 parseable valid sources and 715 of 716 distinct recorded-valid molecules eligible for fingerprints.

The sulfur reduction and the validity penalty depended on the start. Atom 0.5 actually increased average sulfur from 0.028 to 0.474 on seed 111210. Of its 200 unresolved searches, 195 came from seed 111310. At atom 2, 435 of 466 unresolved searches came from seed 111410, where only 16/500 source graphs were valid. The next step produced a valid graph after 162/310 current repair targets, 137/534 mixed-0.5 targets, and 112/661 mixed-2 targets. After an unresolved mixed search only 2/200 and 2/466 next steps were valid. These descriptive comparisons also reflect the different states visited. On the current arm, 53 of 60 stepwise increases in sulfur followed exactly zero chemistry gradient. [Repair and sulfur-transition audit](../../outputs/exchange-protenix-20261111a/collected-20261119/repair-audit.json)

The earlier chemistry-only comparison gave 48/48 valid endpoints in each of these arms and fewer protected changes with the mixed penalties. Under native optimization the mixed arms changed more protected decisions and yielded fewer valid molecules. Both search and preservation differ from baseline in this experiment. The [fixed-search comparison](preservation-protenix.md) isolates the continuous penalties. [Chemistry-only study](atom-exchange-linear.md)

11 November 2026. Question: do atom surcharges 0.5 or 2, each with substructure
cost 0.5, reduce sulfur drift while retaining validity, diversity and improvement
in the actual Protenix contact objective? This extends the
[completed chemistry-only comparison](atom-exchange-linear.md).

## Matched pilot

The [configuration](exchange-protenix.json) declares three independently screened
valid prior starts, each reused by three arms:

| Arm | Repair guidance | Atom surcharge | Substructure cost |
| --- | --- | ---: | ---: |
| current | Existing bond-only structural proposals and repair NLL | — | — |
| atom_half_structure | Mixed atom/charge/bond repair plus reference penalty | 0.5 | 0.5 |
| atom_2_structure | Mixed atom/charge/bond repair plus reference penalty | 2 | 0.5 |

All nine trajectories use frozen epoch 45, **LR 70 and 500 updates**, fixed
temperature 1, chemistry weight 1, no added noise and 25 fixed occupied slots.
The existing latent step-norm cap **0.5** remains active; clipping frequency and
mean bound factors are recorded. Consequently LR 70 does not imply displacement
70 times the gradient on a clipped step. The compute ceiling is three one-GPU,
four-CPU, one-hour allocations (4,500 optimization updates total, plus endpoint
evaluations and selected hard-graph readouts).

Screening seeds are 111210, 111310 and 111410. Each checks at most 96 successive
seeded Gaussian latents for a valid, label-preserving canonical reference, without
calling the native scorer. The selected seed, initial SMILES, fixed mask, latent
and protected motifs are saved. This conditions the experiment on valid starts;
it is not unconditional prior-generation validity. The initial structure remains
the reference for the entire trajectory.

The new arms use `1 - p(replacement) + atom_extra_cost` for changed atoms, charge
surcharge 0.5 and the declared protected-decision surcharge. The differentiable
loss retains fixed-target NLL plus expected reference mismatch with
`anchor_scale=0.05`. Thus the atom surcharge influences repair selection, while
the reference term provides continuous resistance to exchanges. Protection covers
rings, amide N-C=O, acid/ester O=C-O and sulfonyl O=S=O cores, including atom/charge
labels and internal pairs. It is a limited motif set, not all functional groups;
external attachments are generally unprotected. Full native chemistry/schema
validation still decides whether a repair target is acceptable.

This is a comparison against the **current complete guidance**, not an isolated
single-coefficient ablation. Current structural proposals can make coordinated
large bond edits and use the existing probability-regret ranking; mixed search
uses the new 1-p ranking and at most two edits. Both have a nominal 96 candidate
check budget, excluding raw/final verification. Those search-space differences
must be considered when interpreting validity or runtime changes.

## Native objective and measurements

The shared [runner](../../src/latentmol/design/exchange_protenix.py) reuses
`run_design` and `MosaicProtenixScorer`; there is no copied optimization loop.
The actual frozen native objective is the negative mean, over ligand atoms, of
the maximum pocket contact probability below 8 Å. Lower loss is better. It is
a **contact surrogate, not affinity**. Native configuration is the existing
[BRD4 geometry-off context](../latent-design/protenix-vae/native-dtu.json), with
full element probabilities, expected formal charges and bond presence. Native
bond features collapse bond order/aromaticity to presence; the chemistry branch
still receives the full typed graph. Conversion to RDKit stays outside the
differentiable scorer path.

The report compares the first 50 updates and all 500:

- Hard sulfur counts and expected sulfur, including maximum transient increase
  and invalid states; atom and protected-decision changes from the fixed start.
- Raw source validity, endpoint validity and distinct valid trajectory molecules.
- Endpoint uniqueness, Morgan distance and scaffold counts, plus pooled unique
  valid trajectory diversity (fixed-seed fingerprint sample if over 1,000).
- Initial/final relaxed native loss and same-context hard-graph scores for the
  initial state, endpoint and best relaxed-valid state. Invalid hard endpoints
  retain unknown scores. Hard readouts use the same geometry-off context, not
  fresh conformer/pose predictions, and are cached by exact step, not SMILES.
- Finite attached latent gradients, frozen decoder/native/context checks,
  clipping frequency, native timing and complete-state counts.

There are only three independent starts; trajectory observations are correlated.
Fingerprint diversity excludes invalid graphs, whose validity denominator is
reported separately. This is exploratory evidence, without a new finite-
difference/native derivative qualification or statistical-significance claim.

## Submission and current status

Checked at **10:27 CET on 12 November**: A40 job **30000913 completed** all three
500-update arms on 11 November, finishing at 23:09 CET (1,061 seconds total).
V100 **30000901** and L40S **30000930** remain **PEND**, with no output directories;
V100 reports the queue GPU-resource limit, L40S insufficient processors for
affinity placement. Report job **30001069** still waits on dependencies.
[Scheduler receipt](../../outputs/exchange-protenix-20261111a/collected-20261112/remaining-jobs.json).
The partial comparison below contains one matched start, not three.

| Screening seed | Job | Queue | Output suffix |
| --- | --- | --- | --- |
| 111210 | 30000901 | gpuv100 | exchange-protenix-20261111a-111210 |
| 111310 | 30000913 | gpua40 | exchange-protenix-20261111a-111310 |
| 111410 | 30000930 | gpul40s | exchange-protenix-20261111a-111410 |

Each cohort runs all three matched arms on its allocated GPU. GPU types differ
between cohorts, so runtime comparisons should be paired within cohort.
Outputs are under `/work3/s234818/latentmol/runs/`; control scripts and immutable
source copies are under `/zhome/95/e/206121/dbmol-control/` with the same suffix.
CPU report job **30001069** depends on all three jobs ending and writes
`comparison.json` in control directory `exchange-protenix-20261111a-report`.
If a cohort fails or is incomplete, the report records that instead of claiming
a successful comparison. Its frozen reporter predates the SMILES parsing fix
below; the full result will need the corrected local summarizer if it encounters
that failure. The pending report capsule has not been edited.
A first H100 submission was rejected because this
account cannot use that queue; it created no job. The three authorized queue
submissions above are the actual runs.

Local checks: **71 passed** (67 runner/exchange checks, including nonzero-gradient
toy optimization at LR 70, and four summary checks). These are CPU interface and
reporting checks, not native GPU results. The
[receipts and exact launch scripts](../../outputs/exchange-protenix-20261111a/)
are retained, including the rejected submission. Source archive SHA-256:
`4b94687d8d4409ee08303f7ec4372f8aba82461e85cba93508c123a5a6d8450b`.
Checkpoint SHA-256:
`5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be`.

The allocated command, wrapped by `scripts/hpc/protenix_latent_run.sh` with the
saved GPU plan, is:

```sh
python -m latentmol.design.exchange_protenix --config experiments/chemistry-guidance/exchange-protenix.json --checkpoint EPOCH45.pt --mosaic-python NATIVE_PYTHON --mosaic-config experiments/latent-design/protenix-vae/native-dtu.json --seed 111210 --output NEW_OUTPUT --plan GPU_PLAN.json --plan-sha256 PLAN_SHA256
python experiments/chemistry-guidance/summarize_exchange.py --native COHORT_111210 COHORT_111310 COHORT_111410 --output NEW_COMPARISON.json
```

No pre-existing job, callback, live source, checkpoint or default was changed.
The submitted capsules are immutable; two cohorts remain pending GPU availability.

## Preliminary native result: one completed matched start

Collected 12 November 2026. Seed 111310 selected latent seed 111311, with zero
initial sulfur atoms and four protected rings. No functional-group SMARTS core
matched this particular reference. All three arms used the same native identity
and completed finite attached gradient/frozen decoder/native checks. No new
derivative-accuracy claim follows. Source-stage counts use steps 0–499; endpoint
is step 500. Diversity uses unique valid identities across steps 0–500.

| Metric after 500 updates | Current | Atom 0.5 + structure 0.5 | Atom 2 + structure 0.5 |
| --- | ---: | ---: | ---: |
| Final hard sulfur increase | +2 | +1 | 0 |
| Maximum hard sulfur increase | +2 | +2 | +2 |
| Mean hard sulfur over source states | 1.038 | 0.158 | 0.080 |
| Final expected-sulfur change | +2.2332 | +0.9672 | −0.0057 |
| Recorded valid sources / 500 | 415 (83.0%) | 50 (10.0%) | 188 (37.6%) |
| Valid endpoint | No | No | Yes |
| Unique recorded-valid molecules | 283 | 37 | 116 |
| Unique parseable valid molecules | 282 | 37 | 116 |
| Mean pairwise Morgan distance, parseable unique set | 0.8474 | 0.8255 | 0.8395 |
| Unique parseable scaffolds | 255 | 37 | 105 |
| Final relaxed native loss | −0.3943 | −0.3454 | −0.1749 |
| Hard loss of selected best relaxed-valid state | −0.6007 | −0.4621 | −0.2372 |
| Selected state step | 185 | 454 | 42 |
| Hard endpoint loss | unavailable | unavailable | −0.1984 |
| Final atom identities changed | 10 | 6 | 4 |
| Final protected decisions changed | 8 | 15 | 9 |
| Censored repair searches / 500 sources | 3 | 195 | 14 |
| Step-cap hits / 500 | 498 | 500 | 493 |

Initial relaxed loss was −0.1496 and initial hard loss −0.1412 for all arms;
lower is better. All arms improved their relaxed final contact loss, with the
largest improvement under current guidance. The hard readout is evaluated on
the valid state selected by relaxed loss, not the best hard score over every
valid state. It uses the same fixed geometry-off native context, not a fresh
pose or affinity prediction. Invalid endpoints have no hard score.

At 50 updates, source validity was 92% / 64% / 90%, hard sulfur was 1 / 0 / 0,
and relaxed loss was −0.2702 / −0.2291 / −0.2165 (current / 0.5 / 2). Thus the
longer trajectories exposed validity losses that a short prefix would miss.
Native mean update times were 0.467 / 0.816 / 0.723 seconds on the same A40.

The costs substantially reduced average sulfur exposure in this start, and
cost 2 ended sulfur-free with a valid molecule. Neither prevented transient
sulfur additions. Both produced fewer valid diverse molecules and weaker contact
improvement than current guidance. The substructure penalty did not reduce
final protected changes here. Cost 0.5's very low validity coincided with 195
censored mixed searches, but the causal contribution of the surcharge versus
the changed search space is not isolated. Almost all steps were capped. Keep
current defaults while the other two starts remain pending.

One baseline source, step 72, was logged valid by graph sanitization but its
saved aromatic SMILES fails RDKit reparsing in both the HPC and local environment
(RDKit 2024.09.6). The tables preserve recorded validity and expose this export
limitation: requiring saved-SMILES reparsing gives **414/500 = 82.8%** baseline
validity, with the new arms unchanged. Fingerprint/scaffold metrics exclude this
one identity; no invented fingerprint or hidden denominator change is used.
The local shared summarizer now reports parse failures and partial cohort counts;
five focused tests passed. Frozen jobs/report sources were not changed. Native
stderr also logs a CPU-affinity binding warning in each arm; all runs nevertheless
finished with their required finite/frozen checks passing.

Evidence: [partial comparison](../../outputs/exchange-protenix-20261111a/collected-20261112/partial-comparison.json),
[trajectory audit](../../outputs/exchange-protenix-20261111a/collected-20261112/trajectory-audit.json),
[native cohort result](../../outputs/exchange-protenix-20261111a/collected-20261112/111310/comparison.json).
Downloaded 28 result/log files in a 511,620-byte archive and verified SHA-256
`c2701d7bec25fee0eabb0013cdfb238b722b4b26a9da38492942a34845f1ddc5`.
