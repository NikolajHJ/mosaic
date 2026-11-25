# Conditional native rescoring after the fixed-context pilot

Prepared 28 September 2026 for the conditional source018 follow-up.
**Both source018 native jobs completed: AE 8/8 and VAE 12/12 predictions succeeded.**
The preceding CPU preflight passed five tests. Both completed Stage 2 pilots
produced valid changed hard candidates under their fixed-context proxy, and
the coordinator adopted this bounded fresh-pose follow-up. This decision does
not establish native predicted-score improvement or observed binding.

The source018 archive SHA256 was verified on HPC:
`3427402d6e057ff7f0e53050e18d8bc65f94c5290ca9499a84f98bcec37f6b6a`.
Remote release: `/work3/s234818/latentmol/releases/source-018-native`, with the
archive at sibling `source-018-native.tar.gz`; local copy:
`outputs/source-releases/latentmol-source-018-native.tar.gz`.
CPU job **29505625** requested two cores, 2 GB per core and 30 minutes on `hpc`.
It completed the native-rescore and Boltz-pilot preflight: **five tests passed
in 7.45 seconds**, run `20260928T075217Z_tests-source018-native_9f859569`, with
reported LSF turnaround 12 seconds. This is regression evidence, not a native
prediction result. Source017 AE/VAE pilot jobs **29505584 / 29505585 subsequently
completed**, each with 19 tests passed, a passed binary-gradient gate and unchanged
frozen parameters/context.

Native jobs **AE 29505758 / VAE 29505759** each requested one gpua10 GPU, four CPU
cores, 4 GB per core and **12 hours**. They rerun regressions before selecting
candidates and predicting at seeds 17 and 23. AE passed **two tests in 7.80
seconds**, then froze a pool of **four distinct submitted SMILES including the baseline**.
Target selected a carboxylate and fused aromatic ether variant; joint selected
the carboxylate and a sulfur substitution. Trust and random had zero eligible
changed endpoints. Both completed native prediction results are recorded
below; earlier live snapshots remain in the ledger.

## Observed AE result

AE job **29505758 completed successfully**, following two passed tests. Run
`20260928T082423Z_ae-native-fresh-pose_972c2d83` took **860 seconds** and produced
**8/8 successful native predictions**, covering the frozen four-SMILES pool
at seeds 17 and 23. The completed VAE follow-up is recorded separately below.

| Selected AE identity | Selecting arm(s) | Binary difference at seed 17 | Binary difference at seed 23 | Positive versus both controls on both seeds |
|---|---|---:|---:|---|
| Aromatic ether variant `NC(Cc1ccc2c(c1)O2)C(=O)O` | Target | +0.062506 | +0.066622 | Yes |
| Carboxylate variant, standardized to baseline | Target and joint | 0 | 0 | No |
| Sulfur substitution | Joint | -0.000935 | +0.049155 | No |

The listed differences are against both the relevant perturbed start and encoded
baseline. The aromatic ether variant's mean binary increase is **+0.064564**.
Its continuous `affinity_pred_value`, however, increased from **2.521 to 2.681**
at seed 17 and **2.468 to 2.565** at seed 23. Higher values are opposite to that
head's favorable direction. This disagreement prevents treating the binary
increase as evidence of improved affinity. The later structural audit records
ring motifs and filter alerts; chemical stability and reactivity remain unmeasured.

The carboxylate's native binary and continuous predictions exactly matched the
neutral baseline at both seeds despite its earlier fixed-context improvement.
The HPC diagnostic below confirms native standardization collapses those two
inputs. The sulfur candidate's mixed signs fail the both-seeds criterion despite
its positive average.

The selected-candidate result is **target 1/2 confirmed by the binary criterion,
joint 0/2**. Trust and random had no eligible changed endpoints to select.
Retain the original **16 attempted trajectories per arm**, but do not treat this
top-two native audit as a native evaluation of every trajectory. Identical
candidate molecules across arms share predictions and are not independent
discoveries. No experimental binding, selectivity, synthetic feasibility or
general latent-model advantage is established.

The complete AE [native summary](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T082423Z_ae-native-fresh-pose_972c2d83/summary.json>) and
[per-prediction records](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T082423Z_ae-native-fresh-pose_972c2d83/native_results.json>) are now downloaded
alongside all preprocessing, prediction and pose artifacts. Archive
`outputs/hpc-results/stage2/native-ae-source018-results-20260928.tar.gz`
(approximately 10 MB) has verified matching HPC/local SHA256
`077681f9aca2dcd0b779abbcfbe1534f8ad79e5058a658e2a5496ebdd419faf4`.
The VAE transfer receipt is recorded with its completed results below.

## Observed VAE result

VAE job **29505759 completed successfully**, following two passed HPC tests.
Run `20260928T083844Z_vae-native-fresh-pose_253bb4de` produced **12/12 successful
native CLI predictions** for six submitted SMILES. Every selected endpoint has
both native seed comparisons available; there were no prediction failures.

| Selected VAE endpoint | Selecting arm(s) | Comparator | Seed 17 binary change | Seed 23 binary change |
|---|---|---|---:|---:|
| Para-aniline `Nc1ccc(CC([NH3+])C(=O)[O-])cc1` | Target and joint | Both start and baseline | -0.003112 | +0.013862 |
| Zwitterionic tyrosine form, standardized to baseline | Target and joint | Both start and baseline | 0 | 0 |
| Three-membered oxygen-ring endpoint | Trust | Both start and baseline | -0.002423 | -0.023484 |
| Brominated geminal-diol endpoint | Random | Own brominated start | -0.187988 | -0.133614 |

Target and joint selected exactly the same two molecular endpoints. Neither
passed the positive-versus-both-controls-on-both-seeds criterion: **target 0/2,
joint 0/2**. The shared para-aniline endpoint's continuous values increased
from **2.521 to 2.753** and **2.468 to 2.639**; its mixed binary signs do not
establish a consistent benefit, and the continuous head moves unfavorably.
The zwitterionic form is another submitted identity that collapses to the
baseline under native standardization.

Trust passed **0/1 selected**, and random passed **0/1 selected**. Random's
endpoint differed from the baseline by -0.064999 / +0.000262, but it was worse
than its own starting molecule on both seeds; comparing only with the encoded
baseline would miss that loss. Keep the original **16 trajectory attempts per
arm** alongside these selected-subset denominators. Shared endpoint molecules
do not constitute independent target/joint confirmations.

Evidence: [VAE native summary](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T083844Z_vae-native-fresh-pose_253bb4de/summary.json>) and
[per-prediction records](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T083844Z_vae-native-fresh-pose_253bb4de/native_results.json>).
The extracted run is beside
`outputs/hpc-results/stage2/native-vae-source018-results-20260928.tar.gz`, with
verified matching HPC/local SHA256
`a37b6cc6ee886c4683f625434e0aab7bd86eaacdb7889641c3840d0190722555`.

## Stage 2 gate decision and minimum next experiment

**Do not promote to the larger 32-seed or multiple-context Boltz study yet.**
The engineering gate established frozen differentiable scoring and some hard
changes; the broader useful-design gate remains unmet:

- Each arm had **16 correlated trajectories around one molecular context**.
  The guided arms produced only two unique changed hard endpoints per arm.
  Native rescoring retained the selected denominators: AE target **1/2**, joint
  **0/2**; VAE target **0/2**, joint **0/2**, trust **0/1**, random **0/1**.
- The [chemical audit](14_selected_chemical_audit.md) found **nine submitted
  identities but seven standardized native identities**. **Four of ten selected
  endpoint records collapsed to their start or baseline**; six remained distinct.
  These are selection records with repeated molecule memberships, not ten
  independent candidates.
- **Five submitted structures had catalog alerts and two contained an
  oxygen-containing three-membered ring.** These are structural screening flags,
  not proof of instability or synthetic failure, but the current evidence does
  not establish useful chemistry for the selected score-improving candidate.
- The only consistent AE binary gain moved the continuous head unfavorably and
  has not been replicated beyond this context. VAE's distinct guided endpoint
  also had unfavorable continuous changes and inconsistent binary signs. Neither
  two seed repeats nor agreement with one head establishes binding or a general
  AE/VAE advantage.

The minimum next experiment should be a **standalone chemical-prior interface
feasibility check**, without additional Boltz scaling. For a compatible frozen
DeFoG checkpoint, first establish the actual element/bond/charge/occupancy
contract, then test one explicit scalar objective and its gradient on a small
fixed-occupancy set. Do not silently omit unsupported charges or equate a noise
token with atom absence. A denoising-consistency loss or internal-feature prior
is a proposed objective, not native molecular likelihood.

If a meaningful differentiable interface exists, compare a small paired set of
raw hard decodes against the existing trust/no-prior controls, retaining failures,
size, changed standardized identities, diversity and the structural diagnostics.
Require hard-candidate benefit before coupling the prior to expensive Boltz
optimization. If the interface is incompatible or no benefit appears, record
that outcome and reconsider a same-schema denoiser rather than assuming the
pretrained prior works. This is a recommended feasibility experiment, **not a
submitted Stage 3 run or a claim that it will succeed**.

### Confirmed standardization collapse

On the DTU HPC login node, the coordinator inspected installed Boltz commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. The native affinity input path calls
`standardize(seq)` in `boltz.data.parse.schema` at approximately line 1242;
the function around lines 1844–1859 uses ChEMBL `standardize_mol`. Calling that
same installed function directly confirmed the following exact mapping:

```text
NC(Cc1ccc(O)cc1)C(=O)O    -> NC(Cc1ccc(O)cc1)C(=O)O
NC(Cc1ccc(O)cc1)C(=O)[O-] -> NC(Cc1ccc(O)cc1)C(=O)O
```

Exact remote Bash command after activating the pinned environment:

```bash
source /work3/s234818/latentmol/env/boltz311/bin/activate
python -c 'from boltz.data.parse.schema import standardize; print(standardize("NC(Cc1ccc(O)cc1)C(=O)O")); print(standardize("NC(Cc1ccc(O)cc1)C(=O)[O-]"))'
```

Captured stdout, exactly two lines:

```text
NC(Cc1ccc(O)cc1)C(=O)O
NC(Cc1ccc(O)cc1)C(=O)O
```

The SSH diagnostic exited **0**, with reported wall time **10.7 seconds**.

This is a deterministic HPC login diagnostic, not a new LSF job or a local
numerical test. The charged candidate and neutral baseline become the same
native affinity input; the charged form is therefore not a distinct native
candidate. Its fixed-context charge improvement does not survive preprocessing.
This explains the identical native scores without establishing a general rule
about every charged molecule.

Keep all eight already executed source018 prediction attempts in the evidence
and cost denominators. The four distinct submitted SMILES are not four verified
distinct standardized affinity inputs. Any later deduplication must distinguish
raw submission identity from the actual standardized identity rather than
retroactively erase the charged-form attempt.

The purpose is to test whether selected hard molecules retain their apparent
advantage when native Boltz rebuilds ligand features and predicts fresh poses.
The implementation is `src/latentmol/boltz/native_rescore.py`, invoked by
`scripts/hpc/boltz_native_rescore.sh`. Predictions remain model outputs, not
observed binding or calibrated selectivity.

## Parent evidence and frozen selection

Require a completed source017 `ae-boltz-latent-pilot` or
`vae-boltz-latent-pilot` whose mode matches and whose binary-gradient gate
passed. Require its complete **64 endpoints: 16 for each of target, trust,
joint and matched-displacement random arms**. The encoded baseline must be a
valid hard molecule. The runner additionally requires at least one selected
valid changed endpoint from the target or joint arm; a baseline-only selection
does not pass the execution gate.

For each arm, retain endpoints with a valid start and valid final SMILES, whose
final identity differs from both the start and encoded baseline, and whose
fixed-context binary score is available. Sort by decreasing fixed-context score,
then repeat, start index and SMILES. Select **at most two distinct SMILES per
arm**. Selection does **not** require that the fixed-context score improved over
either comparator: rescoring is allowed to test or contradict the ranking.

Add the shared encoded baseline and each selected endpoint's own perturbed-start
SMILES, then deduplicate identical SMILES across these roles and arms within
each model's follow-up. Preserve each arm's eligible-attempt count,
unique-eligible count and selected memberships. An arm with fewer than two
eligible molecules is not supplemented with invalid or unchanged molecules.
Its shortfall remains an outcome.

Write `frozen_selection.json` before the first native prediction. The maximum
is **17 unique molecules per model**: up to eight endpoints, their eight starts
and the baseline, giving at most **34 native CLI attempts** at two seeds each.
AE and VAE have their own encoded baselines;
the baseline is shared across a model's arms, not assumed identical across
models. The current code does not deduplicate across separate AE/VAE jobs.

This top-two selection is a bounded selected-candidate audit. Its success rate
is not the overall success rate of the 16 attempted trajectories per arm, and
must be shown beside those original denominators.

## Native prediction protocol

Copy the source006 example template
`boltz-preparation/source006-attempt001/input/tyrosine_smoke.yaml`, retain its
protein and MSA declarations, and replace the single ligand's SMILES. Require
exactly one ligand entry. Save each generated YAML and its hash. Each unique
molecule receives the same settings and pose seeds **17 and 23**:

| CLI setting | Value |
|---|---|
| Structure sampling steps | 100 |
| Affinity sampling steps | 100 |
| Structure diffusion samples | 1 |
| Affinity diffusion samples | 1 |
| Structure recycling, `--recycling_steps` | 0 |
| Native affinity recycling | 5, hardcoded by the pinned CLI |
| Native precision | `bf16-mixed` |
| MSA selection | `--subsample_msa --num_subsampled_msa 128` |
| Workers / preprocessing threads | 0 / 1 |
| Kernels | `--no_kernels` |

The native affinity pass therefore differs from the fixed-context pilot's
zero-recycle, FP32 affinity pass. Retain these runtime and output differences
when comparing scores; this follow-up does not test numerical parity with the
pilot.

There is one native CLI invocation per unique SMILES and seed, rebuilding ligand
features and generating new predictions. Equal seeds pair random-seed labels;
they do not force identical poses or corresponding random draws across molecules.
Native rescoring changes several aspects of the pipeline together, so any score
change cannot be attributed to pose generation alone.

The shell script first runs the native-rescore regressions. Their outcome must
be recorded when executed. No target controls, additional proteins or synthesis
oracle are introduced by this follow-up.

## Failures and paired comparisons

Journal each CLI attempt to `native_results.json` **before launch**, with its
command, input hash, log and expected affinity-output path. Update that record
with elapsed time, exit code, affinity output or failure after the call returns.
Each call has an **1,800-second timeout**. Timeouts, launch/output errors,
nonzero exit, missing affinity JSON, unreadable required fields and nonfinite
scores are retained as failures, and later candidates continue. Successful
affinity JSON files are hashed. An interrupted whole job still needs its
scheduler/experiment failure record; the journal preserves an in-flight attempt
even when no final summary can be written. The per-call ceiling is not a promise
that the whole job will finish within its eventual scheduler budget.

For each selected endpoint, compare native binary probability with **both its
own perturbed start and the encoded baseline at the same seed**. A complete
comparison requires all three predictions to succeed. Record both seed-level
triples, `successful_triples`, endpoint-minus-baseline and endpoint-minus-start
differences, and their available-triple means. Means alone are insufficient:
they may contain only one successful triple.

The confirmation indicator requires **both differences to be strictly positive
on both seeds**, with all required predictions successful. Missing endpoint,
baseline or start predictions fail that indicator, with individual success flags
retained. A baseline failure affects every candidate comparison for that seed;
it remains an explicit shared failure.

All three native continuous values are also retained, but the follow-up does
not interpret them as calibrated affinities. The endpoint-to-start association
must come from the selected trajectory row, not another start with a similar
fixed-context score.

With two seeds and few score-selected candidates, positive differences are
exploratory evidence of predicted-score persistence. They are not statistical
confirmation of better binding, semantic continuity, a superior latent model,
selectivity or synthesizability. Zero-threshold positivity on both seeds is a
diagnostic, not a biologically meaningful effect threshold.

## Provenance and reporting requirements

Run configuration records the parent run/source/checkpoint/prepared-context
identities, hashes of the parent endpoint and baseline artifacts, template hash,
referenced source MSA SHA256, both native confidence/affinity checkpoint SHA256
values, installed pinned Boltz provenance and prediction settings. Preserve the selected checkpoint roles established in
[the training audit](10_full_training_audit.md); do not select a different model
after viewing native scores.

The runner requires one available protein MSA file and hashes its actual
contents separately from the YAML. These identities make the native inputs
auditable alongside the captured environment; they do not make the new pipeline
identical to the fixed-context scorer.

Expected artifacts are `frozen_selection.json`, generated inputs, prediction
directories, per-attempt logs, `native_results.json`, `metrics.jsonl`, run
configuration and final `summary.json`. Report total attempted/successful/failed
predictions, every arm's selection shortfall, shared-candidate membership,
seed-level endpoint/start/baseline values, actual runtime and budget. Per-arm
summaries retain the **16 original attempts**, eligible changed attempts,
unique eligible molecules, selected endpoints, unselected changed attempts,
invalid/unchanged/invalid-start attempts, confirmations and incomplete triples.
Unselected-attempt counts include duplicate trajectories excluded by unique
endpoint selection; they are not counts of independently unscored molecules.
Retain
unsuccessful predictions and failed jobs in [the ledger](05_experiment_ledger.md).
Submission entries require an actual immutable release hash and scheduler ID.
The CPU preflight and subsequent native job submissions are separately recorded
in the ledger. Their eventual successes, failures and partial results must also
be recorded; a submission does not imply a completed prediction.
