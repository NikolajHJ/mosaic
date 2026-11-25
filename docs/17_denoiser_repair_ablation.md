# Stage 3: combined denoiser repair ablation

Updated 28 September 2026. Immutable **source024** was uploaded and its HPC
archive hash verified. CPU preflight **29505979 is DONE**, with **four tests
passed in 4.04 seconds**. GPU job **29506002 is DONE**, with **four passed
on-node tests** and completed run
`20260928T093714Z_stage3-schema-denoiser-repair_bcd42239`. It used `gpua10`,
four CPU slots, 8 GB per slot, one exclusive GPU and a two-hour wall limit.
**Both the repair indicator and local gradient gate failed. No promotion follows.**
All numerical work remains on DTU HPC. The experiment is a bounded repair of
the schema-matched denoising AE, not DeFoG, a validated molecular prior or a
Boltz combination.

Source archive SHA256:
`223df45f366d128975e48e733be7e2df8a0ad70bad5827408ca9db1eff94e974`.
HPC source archive:
`/work3/s234818/latentmol/releases/source-024-denoiser-repair.tar.gz`.
The CPU pass establishes the selected regressions only, not a repair benefit
or useful latent gradient.

## Observed outcome

The declared exact → valid → lower-loss rule selected **epoch 1**. All counts
below use **256 attempts per condition**, without discarding invalid outputs.

| Condition / model | Valid hard outputs | Exact reconstructions | Unique valid outputs |
|---|---:|---:|---:|
| Corrupted / frozen AE | 4 | 0 | 4 |
| Corrupted / selected denoiser | 2 | 0 | 2 |
| Clean / frozen AE | 252 | 250 | 252 |
| Clean / selected denoiser | 90 | 37 | 90 |

The denoiser met the two-unique-output minimum but failed the remaining repair
requirements: corrupted validity did not exceed the control, exact recovery
did not improve, and clean validity/exactness fell far more than five molecules.
Its selected corrupted size MAE was **1.2852** atoms versus control **1.9883**;
untyped edge precision/recall were **0.7047 / 0.6777** versus **0.7293 / 0.6170**.
Those partial metrics do not compensate for failure of hard chemistry and
clean reconstruction.

| Epoch | Corrupted valid / exact | Clean valid / exact | Corrupted loss under new definition |
|---|---:|---:|---:|
| 1, selected | 2 / 0 | 90 / 37 | 4.095255 |
| 2 | 0 / 0 | 21 / 2 | 1.738109 |
| 3 | 0 / 0 | 14 / 4 | 1.035078 |
| 4 | 0 / 0 | 20 / 3 | 0.828674 |

The decreasing loss did not recover valid corrupted outputs after epoch 1.
Clean time 0 was included in this training objective, so the retention failure
here cannot be attributed simply to evaluating a previously unseen time value.
The intervention changed several factors together: these results do not isolate
the effects of its mask, class weights, clean term or checkpoint selection.
They also do not establish whether a longer or different denoising procedure
would work; this bounded configuration failed its development screen.

The separate fixed-reference gradient gate returned **`passed=false`** with
latent-gradient norm **2.6040264e-10**, below `1e-8`. At coordinate 91, the
analytic derivative was **-6.5920380e-11**, and the central finite difference
was **0.0**. The fixed-reference scalar was **0.03158913**. A small absolute
discrepancy with two near-zero derivatives is not useful-gradient evidence.
No chemical-prior guidance, DeFoG compatibility or Boltz combination was tested.

The CPU regression run is
`20260928T093543Z_tests-stage3-denoiser-repair-preflight_a6a603fc`; the GPU
regression run is `20260928T093654Z_tests-stage3-denoiser-pilot_a2a41e6b`.
The selected checkpoint SHA256 is
`e2329dd7c5c4addd0756b821eda4df80993376308fefab7e2ceffde23ce5f938`.

The complete result bundle is
`outputs/hpc-results/stage3/denoiser-repair-source024-results-20260928.tar.gz`,
exported from
`/work3/s234818/latentmol/exports/denoiser-repair-source024-results-20260928.tar.gz`.
Matching local/HPC SHA256:
`a7040543b93773efae058888ffdfc735c624b562253dd50406d17e99b5dfd399`.
The local hash, extracted summary and full four-epoch history were checked.
Run artifacts reside under
`outputs/hpc-results/stage3/20260928T093714Z_stage3-schema-denoiser-repair_bcd42239/`.
The sections below document the executed protocol.

## Motivation and scope

The [source022 pilot and source023 diagnostic](16_schema_denoiser_pilot.md)
found that much lower reconstruction loss could coexist with zero valid hard
outputs, excess predicted bonds and a failed local gradient gate. The proposed
ablation strengthens supervision of absent bonds and padding, adds clean-input
retention and changes checkpoint selection to prioritize hard outcomes.

These are **multiple changes in one intervention**. A difference from source022
would describe the combined configuration; it could not establish which change
caused the difference. The same development cohort is reused after inspecting
its previous failures, so this is exploratory development, not independent
confirmatory evidence.

## What is held fixed

Initialize again from the original validation-selected AE at epoch 108, rather
than continue training the failed source022 denoiser. The checkpoint is
`runs/20260928T002738Z_ae-train_2f61b913/best_validity.pt`, SHA256
`1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95`.
Keep its ordered 50-slot element/NO_ATOM, formal-charge and symmetric-bond schema,
128-dimensional latent/hidden architecture and time-conditioned denoiser wrapper.
The original AE control stays frozen throughout.

Keep the same seed 17, 5,000 TRAIN molecules, 256 Stage 1 pilot development
molecules, four epochs, batch size 64, AdamW learning rate 0.0002, weight decay
0.00001 and gradient clipping norm 5. The Stage 1 cohort file SHA256 is
`dd0645a76ce6074398e3d196e999116f11b134d8abbebf9726368ebc2c94d3e2`; its 200
reserved promotion members remain excluded. The dataset reader checks data,
split and schema integrity; the shell verifies starting-checkpoint and cohort
hashes. No test split is evaluated.

Training corruption still uses per-graph times uniform in [0.10, 0.35], uniform
categorical replacements within true occupied slots plus four following padding
slots, and symmetric unordered-pair bond draws. Bonds to corrupted absent atoms
are cleared and their charges neutralized. Fixed validation corruption uses
time 0.25 and the same seed protocol. The extra clean branch is evaluated at
time 0. The new clean term increases work per update; equal epoch/data budgets
are **not equal computation budgets**.

## Combined intervention

| Component | Source022 | Source024 |
|---|---|---|
| Bond-loss pairs | Clean occupied-atom pairs | Clean occupied slots plus next four padding slots, capped at 50 |
| No-bond class weight | 0.1 | 1.0 |
| NO_ATOM padding class weight | 0.2 | 0.5 |
| Clean-input reconstruction term | Absent | Added with weight 0.5 at time 0 |
| Checkpoint selection | Lowest corrupted-validation loss | Highest exact count, then valid count, then lowest new loss |

`DenoiserLossConfig` and `denoising_reconstruction_losses` in
`src/latentmol/denoiser.py` define the new loss. Its pair mask comes from the
**clean target**, never predicted occupancy. A decoder therefore cannot remove
a bond penalty simply by predicting that an endpoint is absent. Every selected
unordered pair contributes once through symmetric raw bond logits, including
NONE targets at true padding. Farther padding remains outside this bounded
bond-loss mask. This is still ordered-graph supervision, not an invariant
chemical-validity constraint.

Charge weight 1, bond-component weight 2 and normalized size weight 0.1 remain
unchanged. One update minimizes the new corrupted-input reconstruction loss
plus **0.5 times** the same loss on clean input at time 0. Time 0 is therefore
part of training here, resolving that particular extrapolation ambiguity in
the earlier clean stress test. The logged training scalar includes both terms;
the validation scalars describe their respective input conditions separately.

Raw loss values are **not comparable with source022's differently weighted
loss**. Within this run, the denoiser and frozen AE are both evaluated using
the new loss. Hard-outcome comparisons remain interpretable under the shared
decode protocol.

## Evaluation, selection and gates

`src/latentmol/denoiser_study.py` evaluates both models on identical corrupted
inputs and, separately, identical clean inputs each epoch. It counts all 256
attempts per condition, including invalid outputs. Report valid and exact hard
reconstructions, unique valid SMILES, hard atom-count MAE, predicted edge count,
and edge precision/recall, together with reconstruction loss. Precision/recall
measure an undirected edge's presence at ordered slots, not its bond order.
Unique valid identities are a collapse diagnostic, not evidence of novelty or
useful molecular diversity. Hardening and chemistry checks do not apply
post-hoc molecule repair.

Select the checkpoint lexicographically on corrupted validation:
**exact count → valid count → lower reconstruction loss**. A full tie retains
the earlier epoch. Clean-retention metrics are evaluated at that selected epoch;
the code does not choose a different checkpoint afterward to rescue its gate.

The declared **repair gate** requires all of the following:

- Corrupted-input validity exceeds the frozen AE's count.
- Corrupted-input exact reconstruction exceeds the frozen AE's count.
- At least two distinct valid output SMILES occur.
- Clean validity is no more than five molecules below the frozen AE.
- Clean exact reconstruction is no more than five molecules below the frozen AE.

This is a minimum engineering threshold on reused development data, not a
statistical efficacy claim. The five-molecule tolerance is an absolute count
out of 256, not five percentage points. A passing gate would still require
inspection of the absolute success rates, failure modes and comparisons before
deciding a next experiment.

The **local gradient gate remains separate** and unchanged from source022:
frozen original decoder → frozen selected denoiser → fixed-reference soft
consistency scalar at the first validation latent point. It uses the largest
analytic-gradient coordinate, central difference epsilon 0.01, the existing
absolute/relative tolerance and gradient norm threshold `1e-8`. It does not
test an unconditional realism score or a useful chemical optimization direction.
Report both gate booleans independently. Scheduler/workflow completion alone
does not establish either gate passing.

## Code, artifacts and next decision

The implementation reuses `denoiser.py`, `denoiser_study.py` and
`scripts/hpc/stage3_denoiser_pilot.sh`; an immutable release distinguishes it
from the earlier pilot. `tests/test_denoiser.py` adds a regression requiring
the new loss to penalize a false bond to true padding, alongside the corruption
and differentiable-input tests. The CPU preflight and GPU allocation both
passed the selected regressions.

The run name is `stage3-schema-denoiser-repair`. Save all epoch metrics,
selected indices and hashes, run provenance, gate outputs and selected checkpoint
SHA256. `best.pt` uses **format version 2**, with model/loss configurations,
graph specification, source-associated run configuration and weights. The
source023 diagnostic's format-1 reader is not automatically compatible with
this new checkpoint; any reuse must explicitly support the new loss/format.

The source024 release, both passed regression runs, completed training and
failed gates are recorded in [the experiment ledger](05_experiment_ledger.md).
Source022/023 outcomes remain unchanged. This failed combined repair is
retained without assuming that more iterations must fix it. Any further
useful-gradient or decoder-perturbation experiment requires its own declared
protocol; no Boltz integration or chemical-prior promotion follows here.
