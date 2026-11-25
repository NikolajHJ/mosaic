# Historical README snapshot — 30 October 2026

This is a dated reference. Start at the [current README](../../README.md).
Commands below were written for their original source and are not launch instructions.

# latentmol: molecular graph autoencoders

**Early exploration is the default.** Start with a small bounded run and a
finite, attached-gradient smoke. Local small CPU/unit tests are allowed. Record
negative quality results and inconclusive numerical diagnostics; expensive
AD/FD/native/reproducibility gates are opt-in through `research_profile="qualified"`
where supported. See the shared [exploratory research policy](../../docs/EXPLORATORY_RESEARCH_POLICY.md)
and [gradient policy](../../docs/GRADIENT_VALIDATION_POLICY.md). Keep basic computation,
output ownership and resource checks. Active jobs and sealed releases are unchanged.

The historical 543-case CPU gradient regression baseline retains its original
scope; it is not a prerequisite for each new pilot. Current instructions live in
[AGENTS.md](../../AGENTS.md); dated campaign notes are context for those campaigns.

**Current HPC connection on Nikolajs_laptop (29 October 2026):** local agents
should use the installed unattended connector for commands and file transfers:

```powershell
& 'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe' exec 'hostname'
```

This route uses the existing key and saved Windows credential for `s234818`.
Start with the [current connection guide](../../docs/HPC_CONNECTION.md) and
[agent instructions](../../AGENTS.md); plain `ssh`/`scp` examples in historical records
do not select the helper automatically. Other operators use their own account
and authorized connection setup.

**DBMol-style Boltz loss reference (23 October 2026):**
[implementation, gradient flow, validation history and pitfalls](../../docs/77_dbmol_boltz_loss_implementation.md)
is the consolidated guide, with exact reductions and code-section links.
It distinguishes the successful pointwise native AD checks from historical
finite-difference failures and from chemical/binding efficacy. The dated
experiment snapshots below remain historical records.

This project trains a molecular graph autoencoder, compares a VAE with the same
architecture, and then aims to optimize molecular latent variables through a
frozen decoder and frozen Boltz2. The AE/VAE decoder represents molecule size
through `NO_ATOM` slots. The implemented Boltz adapter fixes the seed occupancy;
atom birth/death and design-time size control are not implemented there.

**Frozen DeFoG repair guidance (7 October):** opt-in fixed-slot optimization
can alternate prescribed hard draws/native repairs with decoder cross-entropy
against an accepted, frozen repair target. Boltz still scores the current
shared relaxed graph. The exact GuacaMol mapping supports neutral atoms only;
unsupported charges/elements are recorded, without coercion. Source068 passed
**121 HPC engineering tests**, and matched six-update C13 configs with repair
weights 0 and 0.01 were prepared. Native DeFoG guidance has not been run or
shown to improve molecules. See the [pipeline and gates](../../docs/37_defog_repair_guidance.md)
and [guidance configuration](../../configs/defog_repair_guidance.example.yaml).

**Current model policy (28 September):** use frozen external pretrained
chemistry priors/scorers. No new in-house prior/scorer training; only small
supervised input bridges/decoders are allowed when needed, then frozen. Every
branch must represent the same canonical molecular graph. Source034 implements
the minimal shared-graph/frozen-branch contract and existing-Boltz facade;
Source034 test job 29508965 failed (41 passed, two facade-indexing failures).
The source035 correction passed **44 selected HPC tests in 13.16 seconds**,
job 29508977; the failed release is retained. See the
[plugin contract and validation gates](../../docs/23_frozen_model_plugin_contract.md).
This verifies interface regressions, not a new external checkpoint, trained
bridge or demonstrated chemistry improvement.

**Configurable frozen-model experiments (1 October):** the
[multi-model guide](../../docs/28_multimodel_experiments.md) explains the named
branch/weighted-gradient interface, required native asset preparation and
independent validation gates. Start from the
[Boltz baseline YAML](../../configs/multimodel_boltz.yaml) or
[ensemble YAML](../../configs/multimodel_ensemble.yaml). Source053 passed 65
local mock/fixture tests from an immutable archive; real Nesso-1 and
Protenix weights, native bundles, gradient parity and full HPC experiments
remain untested. Selecting a model name alone does not install or integrate
an arbitrary new model. The [ledger](../../docs/05_experiment_ledger.md) has the
source and local-run evidence.

**Historical training snapshot:** at 27 September 23:05:38 UTC (28 September 01:05:38 local),
AE **29503898** and VAE **29503899** are RUN at epochs **77 and 76**.
Both are in the mandatory lower-rate phase, learning rate 0.0003; the test split
has not been opened. At matched epoch 60 on all 7,347 validation molecules,
AE had **6,578 valid / 6,236 exact** reconstructions and VAE **6,028 valid / 5,571
exact**. VAE standard-normal sampling produced only **13/1,024 valid** molecules.
Current-phase results and the proposed next-experiment gates are in the
[ledger](../../docs/05_experiment_ledger.md) and [plan](../../docs/09_next_experiment_plan.md).

**Campaign setup:** full-data AE/VAE training is authorized with 215,000 training
molecules, preservation of all prior split memberships, and a staged 80–120-epoch
budget. The [full training protocol](../../docs/07_full_training_protocol.md) records
the adopted model-specific validation continuation rule and diagnostic plan.
Data preparation **29503823** completed with 215,000/7,347/7,347 molecules.
Corrected validation **29503855** passed 77 tests (two skipped) and verified zero
split overlaps and preservation of all prior memberships. GPU launches
**29503857/29503858** then failed before training because of shell path expansion.
Source014 jobs **AE 29503879 / VAE 29503880** passed GPU RNG preflights, then failed
on their first batch because a CUDA cross-entropy kernel rejects strict
determinism. Those attempts completed no optimizer update. Source015 validation
**29503897 passed 83 tests (one optional Boltz test skipped)**, including actual
CUDA AE/VAE training/resume checks, and repeated the successful data audit.
AE **29503898** and VAE **29503899** started on separate NVIDIA A10 allocations,
with identical initial model-state hashes. The first snapshot recorded epoch 1 with finite
metrics and 3,360 updates. On the fixed 1,024-molecule validation cohort, AE
produced 120 valid / 32 exact reconstructions and VAE 134 valid / 41 exact.
Those first-epoch numbers are historical; latest results appear above.
Both configurations/rules are fixed
before launch, and each model seals selection before its own test evaluation.

**Completed feasibility results:** the 60-epoch AE pilot completed, but held-out reconstruction
is weak. On 256 validation molecules, its lowest-loss checkpoint produced
**5 valid / 0 exact** reconstructions; its last checkpoint produced **26 valid /
1 exact**. Both results are retained in the ledger. The earlier smoke gate
passed with 16/16 exact, valid, unrepaired memorized molecules.

The matched VAE completed with similarly low validity: **4/256 best, 28/256 last**.
Source-004 passed 40 tests and identified predominantly valence failures. Job
**29503182** completed the stronger no-bond penalty comparison. Best-checkpoint
VAE validity rose to 25/256, but final-checkpoint validity fell for both models;
exact reconstruction remains weak. Source-006 passed 49 tests; prior sampling
yielded 0/256 valid draws for both baseline VAE checkpoints and 1/256 best,
0/256 last for the stronger no-bond penalty. The data split remains 4,000/500/500.

The frozen Boltz adapter passed the stronger source-010 audit in job **29503342**
on A10 after **53 tests passed**. Native fixed-pose outputs matched exactly,
chemical/latent gradients were finite and nonzero, four finite-difference step
sizes agreed within 2%, and frozen parameters/context remained unchanged.
Three updates lowered the continuous raw affinity objective, while relaxed
binding probability fell. The same valid hard molecule and both of its fixed-
context projection scores remained exactly unchanged. This is strong local
derivative evidence, with no demonstrated discrete molecular improvement.
Occupancy, pose and reference features remain fixed. The ledger retains earlier
setup, loader and memory failures and the successful recovery steps.

Full training, preprocessing and substantive numerical experiments run on
**DTU HPC**. The 1 October implementation task had a separate permission for
quick local fixtures and a temporary restriction on HPC testing; see `AGENTS.md`
and the dated ledger entries. The 7 October repair engineering checks ran on HPC.
Do not run the full experiment examples below on the laptop.

## Read these documents

- [Current HPC connection](../../docs/HPC_CONNECTION.md): default unattended transport
  for local agents on the configured Windows workstation; commands and limits.
- [Sulfur bias in chemical guidance](../../docs/84_sulfur_bias_in_validity_guidance.md):
  direct maximum-valence shortcuts, frozen-target distinctions, measured
  Protenix–DeFoG feedback and composition-reporting limits across experiments.
- [Full training protocol](../../docs/07_full_training_protocol.md): data reservation,
  matched configuration, convergence decisions, diagnostics and sealed test policy.
- [Active pipeline](../../docs/04_latent_pipeline.md): stages, graph contract, confirmed
  defaults, scientific questions and unresolved decisions.
- [Experiment ledger](../../docs/05_experiment_ledger.md): verified setup, source hashes,
  job history, results and failures.
- [Code reference](../../docs/06_code_reference.md): module responsibilities, configuration,
  loss definitions and current implementation limits.
- [Glossary](../../docs/00_glossary.md): background terminology.
- [Project instructions](../../AGENTS.md): current scope, execution boundary and
  experiment-recording requirements.

`docs/02_implementation_plan.md` and `docs/03_hpc_execution.md` describe an older
small-KIBA-predictor project. Their scope and local-training recommendations are
superseded; their banners link to the current plan.

## Initial representation and comparison

The initial topology-only representation has up to 50 heavy-atom slots; elements
`NO_ATOM, C, N, O, F, P, S, Cl, Br, I`; categorical charges `-2..2`; and symmetric
`NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC` bonds. Stereochemistry is removed before
deduplication. Disconnected, unsupported, oversized or non-round-trippable
molecules are rejected explicitly. Raw decoded graphs receive no chemical repair.
The source audit rejected **18,195 of 249,455 rows** because this representation
could not preserve the exact stereo-free molecular identity. The ledger retains
these exclusions; they must not be hidden when describing dataset coverage.

Both AE and VAE use the same ordered graph encoder and slot/pair decoder. The
initial hidden and latent dimensions are 128, with three encoder message layers.
The AE uses the latent mean; the VAE adds posterior sampling and a Gaussian KL
term. The initial matched pilot uses seed 17 and the prepared 5,000-molecule
ZINC250k subset. These settings have completed HPC training and evaluation;
their weak reconstruction and generation results remain the main limitation.

## HPC environment and commands

Verified account: `s234818`; personal project root:
`/work3/s234818/latentmol`. The Python environment is
`/work3/s234818/latentmol/env/ae`, using Python 3.11.9, PyTorch 2.5.1+cu121,
NumPy 2.1.3, RDKit 2024.9.6, PyYAML 6.0.2 and pytest 8.3.5.
The exact raw-data URL and SHA-256 are in the ledger.

From an allocated DTU job, activate this environment and make the project's
`src/` directory importable, or install the package into the environment. The
following are CLI forms; choose new output directories and record the exact
configuration/source identity for each execution:

```sh
# Run only inside an appropriate DTU HPC allocation.
python -m latentmol.experiments --run-root /work3/s234818/latentmol/runs \
  --name tests -- python -m pytest

python -m latentmol.data \
  --input /work3/s234818/latentmol/raw/zinc250k.csv \
  --output /work3/s234818/latentmol/data/zinc5k-seed17-v1 \
  --limit 5000 --seed 17

python -m latentmol.train --config CONFIG.yaml --data CACHE_DIRECTORY \
  --run-root /work3/s234818/latentmol/runs

python -m latentmol.evaluate --checkpoint RUN_DIRECTORY/best.pt \
  --data CACHE_DIRECTORY --run-root /work3/s234818/latentmol/runs --split val
```

Dataset preparation refuses to overwrite an existing cache. Wrap preprocessing
and other commands with `latentmol.experiments` as well so failures have durable
records. Training and evaluation create their own run records. Scheduler scripts
and final run configurations are maintained by the execution coordinator.

A successful import, scheduler exit or loss reduction is not a claim of useful
molecule design. Reconstruction, raw validity, changes in molecular size and the
score retained after discrete evaluation must be measured separately.

## Optional Boltz setup and smoke on HPC

The following are submission templates for a DTU login node. Use an immutable
release with its matching sibling archive (`source-010/` and `source-010.tar.gz`),
whose SHA-256 the job records. Installation, preparation and numerical checks run
inside LSF allocations. Start each dependent stage only after its predecessor
has completed successfully, and record the new job IDs in the ledger.

```sh
TASK_ROOT="/work3/$USER/latentmol"
cd "$TASK_ROOT/releases/source-010"
TASK_AE="$TASK_ROOT/runs/20260927T135410Z_ae-train_4b2da706/last.pt"

# Fresh optional environment and hash-verified model assets.
bsub -q hpc -n 4 -R "span[hosts=1] rusage[mem=4GB]" -W 2:00 \
  -o "$TASK_ROOT/logs/%J.out" -e "$TASK_ROOT/logs/%J.err" \
  bash -l scripts/hpc/setup_boltz_environment.sh
```

`requirements-boltz-hpc.lock.txt` is a **captured tested environment snapshot**,
not a fully enforced restoration lock. The bootstrap pins direct packages, the
Boltz Git commit and model assets; it does not install every transitive dependency
from that snapshot. A fresh bootstrap may resolve different transitive versions.
Retain its new `cache/boltz2/environment-freeze.txt`, compare it with the snapshot,
and rerun tests before attributing reproduction to an identical environment.

To make a fresh context, choose a new output directory; preparation refuses to
overwrite one. It uses the reduced-sampling official-example engineering protocol.

```sh
TASK_ATTEMPT="$TASK_ROOT/boltz-preparation/NEW_UNIQUE_ATTEMPT"
bsub -q gpua10 -n 4 -R "span[hosts=1] rusage[mem=8GB]" -W 2:00 \
  -gpu "num=1:mode=exclusive_process" \
  -o "$TASK_ROOT/logs/%J.out" -e "$TASK_ROOT/logs/%J.err" \
  bash -l scripts/hpc/boltz_run.sh python -m latentmol.experiments \
  --run-root "$TASK_ROOT/runs" --name boltz-preparation -- \
  bash scripts/hpc/boltz_prepare.sh "$TASK_ATTEMPT" "$TASK_AE"
```

After preparation succeeds, use its `prepared.pt`; to reproduce the current
fixed-context checks, reuse the existing path below instead. The recovery script
runs the release's tests before its smoke, creates new run directories, and
retains failures. Record the actual GPU; `gpua10` and `gpua40` can differ.

```sh
TASK_PREPARED="$TASK_ROOT/boltz-preparation/source006-attempt001/prepared.pt"
# For the fresh context above: TASK_PREPARED="$TASK_ATTEMPT/prepared.pt"
bsub -q gpua10 -n 4 -R "span[hosts=1] rusage[mem=8GB]" -W 2:00 \
  -gpu "num=1:mode=exclusive_process" \
  -o "$TASK_ROOT/logs/%J.out" -e "$TASK_ROOT/logs/%J.err" \
  bash -l scripts/hpc/boltz_run.sh bash scripts/hpc/resume_boltz_smoke.sh \
  "$TASK_PREPARED" "$TASK_AE"
```

These commands exercise fixed-occupancy, fixed-pose chemical gradients and a
bounded latent probe. They do not implement atom birth/death or a validated
binding-design workflow.
