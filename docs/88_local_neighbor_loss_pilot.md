# Local matched VAE pilot with neighbor-aware chemistry loss

**Prospective exploratory policy (30 October 2026):** the working pilot script now
records quality deterioration as warnings and completes its bounded budget by
default. `--research-profile qualified` preserves the earlier 10pp validity/5pp
reconstruction stop. This changes no saved pilot outcome, weights or HPC job,
runs no new pilot, and does not adopt the failed neighbor-weight0.1 arm below.


Prepared **30 October 2026**. The user explicitly authorized setting up a local
environment, checking the new loss locally, and running a small matched VAE
training pilot. This is a scoped override of the repository's usual allocated-
HPC numerical-work rule. It authorizes this local pilot; it does not change any
existing HPC continuation, model-optimization campaign, loss setting or safeguard.

**Terminal state: `safety_stopped_do_not_adopt`.** The control completed 100
updates; the added-neighbor arm stopped at its declared safety check after 10.
At the matched update10, raw fixed-panel validity was 734/1,024 for control and
611/1,024 for neighbor, from the identical 823/1,024 baseline. The added term
reduced local valence errors but greatly increased disconnected graphs. Do not
switch the HPC continuation to this configuration. All 147 local tests and the
two-update benchmark had passed; neither established training stability.
The [reviewed saved-result report](../outputs/local-neighbor-loss/20261030-attempt001/pilot-report-reviewed/report.md)
and [analysis](../outputs/local-neighbor-loss/20261030-attempt001/pilot-report-reviewed/analysis.json)
retain the actual matched prefix, failures and unavailable endpoints. No new
tests or HPC changes follow from this result.

## Question and comparison

Does adding the new fixed-context neighbor-aware event loss improve the
chemical quality of the VAE's generated samples, without degrading
reconstruction, diversity or connectivity, relative to a matched control?

Each arm starts from an independent copy of authenticated completed **epoch42**.
The source input checkpoint and dataset remain immutable. The training copies
can update their declared trainable parameters; an immutable input checkpoint
does not mean the VAE weights remain frozen throughout this training pilot.

| Arm | Objective | Shared controls |
| --- | --- | --- |
| Control | Existing guarded chemistry objective: reconstruction/KL, old repair guidance, reference guidance, composition JS and occupancy anchoring. | Same epoch42 model/AdamW moments, frozen35/36 anchors, real data order, random draws, updates and evaluation panels. |
| Added neighbor term | The same objective plus `common-organic-neighbor-v1`, weight 0.1, separate real/prior branch coefficients 0.25/0.75. | Only the declared added term differs. |

The planned main comparison is **100 updates per arm**, each with **16 real
molecules and 32 fresh prior points** (1:2 real-to-random), microbatch 8. Both
encoder and decoder train. AdamW weights/moments come from the same epoch42
checkpoint, with the existing 0.25 actual AdamW displacement policy retained.
This scales the actual parameter displacement; it does not reset the inherited
moments or silently substitute a different learning-rate schedule.

The parent config has nominal learning rate 0.001, weight decay 1e-5,
clip threshold 5 and beta 0.001; the actual checkpoint optimizer state and
continuation calculation remain authoritative. New local random streams and
smaller batches are deliberate: this is a paired local pilot, **not an exact
HPC resume or a complete epoch43**. Each 100-update arm has 1,600 real-molecule
and 3,200 prior-point exposures, with real rows sampled from its 1,024-row cohort.

The existing standard VAE building blocks are
[`reconstruction_losses` and `latent_kl`](../src/latentmol/losses.py).
Reconstruction combines categorical atom/charge/bond errors and the configured
size term; the actual weights and KL coefficient must come from the selected
configuration. The established
[`training implementation`](../src/latentmol/train.py) records model, optimizer,
RNG and checkpoint identities. The dedicated
[`local_neighbor_pilot.py`](../scripts/local_neighbor_pilot.py) calls the existing
[`chemistry_update`](../src/latentmol/chemistry_continuation.py) with explicit
`additive_guidance`; it does not present this short loop as the full guarded
HPC continuation.

## What the added loss does

The policy and allocated qualification are documented in
[neighbor-aware chemistry guidance](85_neighbor_aware_chemistry.md).
[`prepare_neighbor_environment`](../src/latentmol/chemical_environment.py)
hardens the original current decoder output and freezes elements, charges,
occupancy and the outside-bond context. The subsequent
`neighbor_environment_loss` uses live symmetric bond logits to calculate the
negative log probability of the accepted local event.

That event combines charge-aware focal valence, the remaining capacity of each
neighbor, and at most four non-fluorine heavy-atom neighbors for sulfur. Fluorine
is exempt only from the sulfur coordination counter, not valence. The event is
active for already-valid argmax graphs when unsafe probability tails remain.
Only bond logits receive its direct gradients; shared encoder/decoder parameters
can still affect atom identities indirectly.

The prepared context must stay fixed throughout one loss evaluation and its
gradient check. A fresh preparation belongs to the next declared training
evaluation, not to intermediate finite-difference perturbations. The loss is a
conditional local surrogate, not a minimum edit distance or a complete molecular
validity test. Its all-NONE event prevents impossible local mass but does not
prevent a disconnected graph. Connectivity and occupancy must therefore remain
visible in the pilot's evaluation.

Source145 attempt004 previously passed allocated CPU qualification: 138 PyTorch
and 26 JAX fixtures, six-case scalar/VJP parity, a 10,000-molecule validation
coverage audit, and actual frozen epoch40 decoder-to-latent gradients. That is
implementation evidence; it is not this pilot's local-environment check, a
training improvement, or authority to change an active HPC run.

The streamed adapter is
[`NeighborEnvironmentGuidance`](../src/latentmol/neighbor_training.py). It
freezes the original current graph during the no-grad preparation and reuses
the same context when training logits are recomputed. For each branch, its
microbatch fractions are proportional to **assessed occupied centers**, not
molecule counts. This preserves the full-branch scalar and VJP when molecule
sizes or microbatch boundaries differ. Every prepared interval must be consumed
exactly once before the optimizer can step.

The added term is
\[
 0.1\bigl(0.25\,\bar L_{\mathrm{neighbor,real}}
        +0.75\,\bar L_{\mathrm{neighbor,prior}}\bigr).
\]
These objective coefficients are separate from the 1:2 sampling ratio. The old
repair/reference/JS/occupancy terms remain in both arms. An absent guidance
object follows the historical path; explicit `None` and weight 0 behavior are
covered by tests of the actual joint AdamW parameter update.

## Required provenance

The local run record must bind the following before comparing arms:

| Item | Required receipt | Current value |
| --- | --- | --- |
| Source | Main run uses `C:/Users/nikol/DBMol_Local/neighbor-loss-v1/source001`, outside OneDrive; 448 source/test/TOML files. | Source-manifest SHA `61af99abee308a9f2cc169d28c9a1e668a2d21473b006ff6713ad5d9b4ed8d74`; per-attempt manifests preserved. |
| Starting checkpoint | Completed42, SHA `5ef851101a5f675edc8f65b31b75c23a5ec460f1c40aa5571829bf8ad6714cfd`. | Independently loaded for each arm; encoder/decoder train. |
| Frozen anchors | Epoch35 occupancy SHA `19f438222e454ef53169f6d81082335ee7e24902b1b7cf5e0f18eced4b8e92ed`; epoch36 valid-target SHA `3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`. | Frozen anchor integrity checked after each arm. |
| Local dataset | 1,024 TRAIN/256 VAL original source rows; source-selection seed 148101; no cross-split overlap. | Array bytes/indices/SMILES bound in `inputs.json`; graphs re-encoded and compared locally. |
| Environment | `C:/Users/nikol/DBMol_Local/neighbor-loss-v1/env`. | Python 3.12.14, Torch 2.5.1+cu121, RDKit 2024.9.6, NumPy 2.1.3, pytest 8.3.5, PyYAML 6.0.3. |
| Device/runtime | NVIDIA RTX4060 Laptop, 8188MiB; training FP32; neighbor event DP FP64; four CPU threads. | Deterministic checks retained; launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch. |
| Matched training seeds | Global/posterior 148102; fresh prior 148103; real-row ordering 148104. | Each update records row identities and prior/posterior RNG digests before use. |
| Evaluation panels | Fixed prior seed 148201; held-out prior seed 148202; 256 VAL reconstruction molecules. | Planned main 1,024 points per prior panel/four categorical draws; latent hashes and every raw graph saved. |
| Outputs | `C:/Users/nikol/DBMol_Local/neighbor-loss-v1`. | Separate arm/attempt directories, manifests, logs, metrics, graphs and non-promotable diagnostic checkpoints. |

Data selection must preserve the source split. Any subset exported for this
local pilot is not a new full training corpus. The local environment is
separate from HPC environments; dependency installation must not be reported
as a scientific check passing.

The authenticated input bundle is
`C:/Users/nikol/DBMol_Local/neighbor-loss-v1/inputs/inputs.json`, SHA256
`ebf15e636757d1e7e98322697caa1a4aebe136e32f448737f0821e79eac25b04`.
It includes the source manifest SHA
`f246cc93805b73a557cd0c09a78891ac8410c465192db615ce0b6709218c8c0e`
and source dataset identity
`54f37815d5901f7d76b855f69e237d97172c72b027e98616f949c08178f9cae3`.
Parent config/plan/composition-file hashes are respectively
`b6704c80c4f3308b5f20623eb0ef21fc39c4b7b1d0cd60d519a9990670b1b254`,
`a794d3f7a9b3bcbcdc357aadb6feade06806a4692b08a65e7c3a0e8e0a9420f6`,
and `5ddc4dc0eda9dc93589c8f0904433135f899cfdcab8657fc127f304d4a136ee9`.
Composition uses the saved **full TRAIN-split reference**, not the composition
of this small local sample.

[`local_neighbor_inputs.py`](../scripts/local_neighbor_inputs.py) uses the
unattended helper for read-only remote byte access and checkpoint downloads.
The complete `split.npy` is hashed and source role counts checked. The large
graph-member digests in the manifest are declarations from the authenticated
source cache, **not a fresh full 4.3GB graph-file audit**. Selected row payloads
have their own hashes and must exactly match local RDKit re-encoding of their
original source SMILES. No remote model/numerical run, scheduler submission,
output write or HPC configuration mutation occurs in input preparation.

## Evaluation and interpretation

Use the same fixed panels before training and at the declared update intervals
for both arms. Record each component loss, finite gradient/parameter checks,
elapsed time and the number of completed updates. Report reconstruction
exactness and validity on held-out real molecules alongside generation quality
from the fixed prior panel.

Hard graphs must keep the existing separation between raw chemical validity and
strict representation compatibility in
[`chemistry_classification.py`](../src/latentmol/chemistry_classification.py).
Retain invalid graphs and failure reasons rather than dropping them from the
denominator. In particular, record disconnected/empty graphs, atom counts,
sulfur and other element composition, and any unsupported local-policy centers.
Uniqueness and diversity prevent an apparent validity gain from being interpreted
as useful generation when the model repeatedly emits the same simple structure.

The directly added term is expected to change bond probabilities first. A
decreasing local loss alone does not establish improved discrete chemistry:
the model could reduce connectivity or make tiny valid structures. A small
matched pilot tests whether the implementation trains and whether there is a
promising direction under its exact budget. It does not establish convergence,
full-corpus generalization, binding quality, synthesizability or the correct
loss weight for production continuation.

The training harness records unique valid identities and atom entropy, alongside
composition/size statistics. The subsequent saved-output failure analysis also
computes bounded fingerprint/scaffold observations, described below. It reruns
no decoder/model and does not invent the failed arm's missing terminal samples.

The planned main run evaluates argmax at updates 0/10/25/50/100. Its initial
and terminal panels also use both fixed/held-out prior seeds and four categorical
draws per point. Intermediate panels do not consume training RNG streams;
evaluation saves/restores both CPU and CUDA RNG states and model mode. The two
arms' row identities and pre-update noise digests must match before producing
the final comparison receipt.

Local safety checks stop on a drop exceeding 10 percentage points in fixed-panel
raw validity or 5 points in exact reconstruction relative to that arm's initial
panel. These are bounded local pilot checks, not replacements for the existing
HPC readiness/health protocol. All output checkpoints are marked
`diagnostic_only=True`, `promotable=False`.

## Environment, execution and result history

The dedicated environment was installed using official CUDA 12.1 PyTorch wheels
plus PyPI dependencies and an editable project installation. The existing
repository `.venv` was untouched. The explicit user override applies to these
local checks and this pilot.

| Attempt/stage | Actual evidence/outcome |
| --- | --- |
| Input transfer001 | Retained `inputs/transfer-receipt-1793356993478892900.json`: exit 1 because the initial reader assumed unsigned uint8 while the existing cache uses signed int8. |
| Input transfer002 | `inputs/transfer-receipt-1793357040796503800.json`: exit 0 after checking the exact signed dtype; payload/anchors authenticated. |
| Local chemistry regressions | `cpu-tests.xml`/`cpu-tests.log`: 138 passed, zero failures/errors/skips, 8.31s. |
| New training-hook checks | `training-hook-tests.xml`/`training-hook-tests.log`: 9 passed, zero failures/errors/skips, 10.93s. Includes center-weighted scalar/VJP and absent/None/zero-weight actual AdamW parity. |
| GPU benchmark001 | Intended two updates with 64-point prior panels. `benchmark001/failed.json` and `benchmark001.log` show failure during initial decoder evaluation: deterministic CUDA needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8`. No completed benchmark training step or comparison claimed. |
| GPU benchmark002 | Completed two updates per arm. `benchmark002/result.json` reports matched real-row/noise streams and unchanged frozen anchors. Exact observations and limitations appear below. |
| Source001 first inventory | `source001/seal-attempt001-failure.json` retains the initial inventory problem: concurrent pytest generated bytecode which the first inventory accidentally included. No Python source difference; final inventory excludes bytecode. |
| Final source001 tests | `sealed-tests.xml`/`sealed-tests.log`: 147 passed, zero failures/errors/skips, 8.83s against the final frozen source. |
| Main 100-update pilot001 | Control completed 100; neighbor safety-stopped at10. Planned 100/arm, 16 real/32 prior, microbatch8, 1,024-point panels and weight0.1 remain recorded; this is not a completed 100-update matched comparison. |

### Benchmark002 actual observations

The launcher set the required CuBLAS workspace before importing Torch, keeping
deterministic algorithms enabled. Both arms completed their two-update budget
and the final comparison receipt reports
`completed_local_comparison_not_production_adoption` and
`matched_real_rows_and_noise_streams=True`.

| Observation | Control | Added neighbor term |
| --- | --- | --- |
| Completed updates | 2 | 2 |
| Mean update time | 9.765s | 4.679s |
| Peak reserved GPU memory | 1.473GiB | 1.453GiB |
| Initial fixed-panel raw validity | 51/64 | 51/64 |
| Terminal fixed-panel raw validity | 51/64 | 49/64 |
| Frozen35/36 anchors unchanged | Passed | Passed |

**These timings establish no speedup.** There are only two cold updates per
arm, run in control-then-neighbor order, with setup/cache effects confounded.
Likewise, 51 versus 49 valid molecules in a 64-point panel after two updates is
too small and early to establish training benefit or deterioration. The
benchmark verifies local execution, matched streams, budget completion and
memory feasibility, rather than the scientific advantage of the added loss.

The main pilot used the separate fixed source snapshot and larger declared
panels. Its manifest records the exact environment, checkpoint/input identity,
source hashes and unchanged-HPC flags. Its actual safety-stopped result follows.

These setup and test results do not establish training improvement. Every failed
setup, partial check and interrupted arm remains in the history; only completed
matched budgets may be compared as final pilot results. Existing HPC runs and
their immutable releases remain untouched.

## Pilot001 terminal result: local valence improves, connectivity deteriorates

The [reviewed failure-aware report](../outputs/local-neighbor-loss/20261030-attempt001/pilot-report-reviewed/report.md)
and [machine-readable analysis](../outputs/local-neighbor-loss/20261030-attempt001/pilot-report-reviewed/analysis.json)
use the latest **common saved update10**, not the control's later 100-update
endpoint. The report independently verifies identical source row identities,
prior RNG digests and posterior CUDA RNG digests throughout those first ten
updates, and identical initial raw graph classifications for both arms. All
saved training losses were finite. The predeclared stop remains enforced.

| Fixed-panel metric | Start | Control10 | Neighbor10 |
| --- | ---: | ---: | ---: |
| Raw chemically valid /1,024 | 823 | 734 | 611 |
| Strict valid /1,024 | 822 | 733 | 611 |
| Exact reconstruction /256 | 255 | 254 | 255 |
| Valid reconstruction /256 | 255 | 254 | 255 |
| Sanitization failures | 119 | 118 | 61 |
| Disconnected: native failure category | 82 | 172 | 352 |
| Disconnected: independent occupied-graph traversal | 91 | 188 | 366 |
| Local valence violations | 75 | 89 | 36 |
| Sulfur scope-excluded graphs | 1 | 1 | 1 |
| Unassessed local-rule graphs | 0 | 0 | 0 |
| Expected sulfur fraction | 1.40337% | 1.50253% | 1.47951% |
| Hard sulfur fraction | 1.39737% | 1.44265% | 1.42485% |
| Mean hard atom count | 22.2236 | 22.2031 | 22.1377 |

This supports an objective-imbalance mechanism at weight 0.1 in this small-batch
regime. The local term can improve its valence event by removing bonds; it does
not require connectivity. Here sanitization/local-cap problems decreased while
many more outputs became disconnected, so total validity fell. This is evidence
against adopting **this configuration**, not a numerical-gradient failure or
proof that every neighbor-aware formulation fails.

These two disconnected counts deliberately differ. Native classification tests
sanitization first, so its disconnected category excludes unsanitizable graphs.
The independent saved-array audit traverses all occupied hard slots, treating
any non-NONE bond as an undirected edge, regardless of RDKit acceptance. Its
91/188/366 counts confirm actual topological fragmentation rather than merely
unmasking a different failure category as sanitization improves.

Sulfur inflation was not the dominant observed failure: both endpoints were
below the full training-population sulfur fraction of 1.6879939%. A lower sulfur
fraction alone is not a chemical-quality objective for individual molecules.
Raw chemistry and representation remain distinct: start/control10 each had one
representation failure, while neighbor10 had none; improved representation
counts do not offset the connectivity loss.

### Paired changes, bond edits and bounded diversity

Relative to the identical baseline, control rescued 57 previously invalid graphs
and made 146 newly invalid; neighbor rescued 52 and made 264 newly invalid. In the
direct control10-to-neighbor10 pairing, 154 raw-valid control outputs were invalid
under neighbor, while 31 moved in the opposite direction.

Bond-edit counts include each undirected edge once and restrict the comparison
to graphs retaining every atom and charge slot label. Control had 543 such graphs,
with 186 deleted/152 added edges and 88 changed nonzero bond types. Neighbor had 491,
with 448 deleted/113 added edges and 104 changed nonzero types. The different graph
counts are retained as denominators; these are edge counts, not numbers of
molecules. They strengthen the observed bond-removal/connectivity explanation.

The [independent topology audit](../outputs/local-neighbor-loss/20261030-attempt001/reviewer-audit.json)
also counts all 1,024 graphs: net edge losses are 335 for control and 1,255 for
neighbor. Restricting to unchanged occupancy gives net losses 277 across 950
control graphs versus 1,004 across 926 neighbor graphs. In the direct endpoint
pairing, 183 connected control graphs become disconnected under neighbor, versus
five reverse transitions. Thus changed atom counts or classification order
cannot alone explain the observed bond loss. These are saved-array analyses;
no model, gradient, optimizer, training or HPC operation was rerun.

| Endpoint observation | Control10 | Neighbor10 |
| --- | ---: | ---: |
| Unique valid canonical identities | 734 | 611 |
| Unique Murcko scaffolds, including empty scaffold | 643 | 506 |
| Fingerprint cohort | 256 | 256 |
| Mean pairwise Tanimoto | 0.1068056183 | 0.1063506180 |
| Internal diversity, 1-mean similarity | 0.8931943817 | 0.8936493820 |

Fingerprints are Morgan radius 2/2048 bits on the first 256 unique valid identities
ordered by SHA256(identity), with 32,640 pairs per arm and no score selection.
Scaffold counts span each arm's valid identities, whose total counts differ.
Neither scaffold counts nor bounded fingerprint similarity establish full
population diversity, synthesis feasibility or preservation of invalid outputs.

The matched ten-update mean times were 2.591s/control and 5.708s/neighbor, medians
2.290s/5.868s. These remain short local measurements; the trajectories produce
different repair-search workloads and do not isolate pure event-DP overhead.
The earlier two-cold-update timings must not be substituted as a speedup claim.

### Completed control and missing experimental endpoints

The control's separate 100-update endpoint had raw/strict fixed validity 769/1,024,
held-out validity 767/1,024, and exact/valid reconstruction 255/256. It completed
its anchor checks and diagnostic-only checkpoint. These measurements are **not
matched comparisons against neighbor10**.

The neighbor arm stopped after its saved update10 argmax/reconstruction check,
before terminal four-draw/fresh-panel evaluation, the final full frozen-anchor
digest check and a failed-state diagnostic checkpoint. These observations and
checkpoint do not exist; they are not filled in or reported as passing. Raw
saved NPZ graphs, classifications, update receipts and the actual failure are
retained. The earlier passed benchmark's complete anchor checks do not certify
this interrupted arm's missing final check.

### Reports and additional edge-case regression

[`summarize_local_neighbor_failure.py`](../scripts/summarize_local_neighbor_failure.py)
reads existing outputs only, verifies the common prefix and preserves the
incomplete status. Its report-script SHA is
`4871e0f09be7b64c336536cc4ca7248737b64984d2d9ed507894d226b727f55f`;
the main run-manifest SHA is
`5b530e6e52025e76747beb646f2149a1d23d7435c44cf56554b2101a17c75b07`.
The complete-run
[`summarizer`](../scripts/summarize_local_neighbor_pilot.py) was separately
checked against benchmark002 and labels the actual two-update budget, rather
than hardcoding 100. It is not used to pretend the failed main arm completed.

An explicit local CPU SO3 regression at
`C:/Users/nikol/DBMol_Local/neighbor-loss-v1/so3-regression.json` retained
`O=S(=O)=O`: assessed caps and common-organic scope passed, no unassessed slots,
event loss 1.0416001394528235e-10. This checks that three S=O bonds do not trigger
the sulfur coordination exclusion. The receipt explicitly distinguishes this
local-policy assessment from native chemical validation; no training/model
mutation occurred for this regression.

The reviewed analysis binds the independent audit SHA256
`57fbf938c5d75ba0b06728448f9809a70c54b8c1abb8116a582bee6430e8a0d5`
and repeats that the interrupted arm never reached its final frozen-anchor
assertion. The [initial report](../outputs/local-neighbor-loss/20261030-attempt001/pilot-report/report.md)
and [initial analysis](../outputs/local-neighbor-loss/20261030-attempt001/pilot-report/analysis.json)
remain preserved as earlier evidence; current result links point to the reviewed
edition. Its extra topology row refines the failure mechanism without changing
the safety stop, training outputs, objective or adoption decision.

The [setup receipt](../outputs/local-neighbor-loss/20261030-attempt001/setup.json)
preserves environment/source/test hashes and the exact launcher command. Its
`100updatepilotrunning` status is a **pre-terminal setup snapshot**; current
outcome is the failure analysis, `safety_stopped_do_not_adopt`. Do not rewrite
the old receipt. The existing HPC continuation, active jobs, losses and all
safeguards remain unchanged. No additional local run, restart or HPC experiment
is authorized or started by this report.
