# Sequential random-latent start study

Status: source056 implemented, statically reviewed and packaged locally
on 2 October 2026. On 5 October the user reported working VPN access and
explicitly authorized checking the existing results and starting this
study, lifting the previous access/submission suspension. Authenticated
access, upload and hash preflight succeeded; job **29603601** was submitted
exactly once and initially observed **PEND**. This is one 12-hour `gpua40`
allocation, not ten jobs. No fixture, preparation or random-start outcome
is established by the submission snapshot. The downloaded receipt records
submission at 23:21:58 DTU local time on 5 October; the latest check remains
queued for the GPU, with no job stdout or started seed observed. Local
receipt/config/status paths are recorded in the ledger.
Its archive is
`outputs/source-releases/source-056-random-starts.tar.gz`; exact archive,
source and config hashes are recorded in the experiment ledger. The
numerical fixtures and the real-model workflow remain **untested** and may
need refinement. The earlier implementation-only instruction and its
packaging history remain in the ledger; renewed authorization is not a
completed run or fixture result.
The user requested one long HPC job containing ten
sequential random-start optimizations to examine endpoint diversity,
atom-count changes and dependence on the carbon-chain initialization.
The existing source055 C13 job 29545141 is a separate experiment and is
not cancelled, resumed or modified by this plan. Its completed 100-update
and fresh-native results were downloaded on 5 October; the ledger records
completion evidence and the disagreement between binary and continuous
affinity heads.

## Question and scope

Does the frozen VAE/Boltz optimization behave differently when molecular
`z` starts at an unconditioned standard-normal draw instead of the encoding
of a carbon chain? Compare starting and final graph identities, atom
counts, element composition, score trajectories and distinct valid final
molecules. This is a study of initialization and search behavior with one
frozen model pair and target context. It does not establish measured
binding, synthesis, selectivity or that the VAE prior has reliable support.

Randomizing `z` removes the requirement to initialize its graph as a
carbon chain. It **does not remove carbon-template pose/context bias**:
the conditional Boltz inputs still reuse native preparations of carbon
templates. Reference geometry, protein features and other retained native
context can continue to influence the search. A claim that carbon bias
has been eliminated would require a separate context/pose experiment.

## Predeclared starts and execution

Use exactly **ten** reproducible draws from `N(0, I)`, with integer seeds
**0–9**. `sample_prior` uses a separately seeded CPU `torch.Generator`
and float32 `torch.randn((1, width))`, then transfers that vector to the
scorer's device. Width comes from the frozen VAE (256 for the selected
checkpoint); randomization never changes latent dimensionality. Freeze
the seed list, implementation and manifest before scoring, and save each
actual initial vector with its source/config/checkpoint identity. No encoding of a seed
molecule is involved. Do not redraw, repair, preoptimize, filter by
validity or select starts by affinity. If a draw has an empty or unsupported
atom selection, or fails a required gate, retain that draw as an attempted
study member with its explicit failure reason.

Run the ten members **sequentially in one GPU allocation**, without a job
array or parallel seed workers. Each seed attempts `active_set` first,
then `fixed_initial_mask` from the same saved draw; the fixed mask is
that draw's own initial selected mask. If the active arm aborts the
pilot, the fixed arm is **`not_attempted`**, not a failed or completed
control. Retain this asymmetry in the report; matched configured budgets
do not guarantee matched completed runs. The source release, manifest,
allocation and final arm configuration must be frozen in the submission
record before execution. No numerical start screening has occurred.

The allocation is one 12-hour GPU job. The original plan proposed a
scheduler dependency `ended(29545141)` so the C13 allocation would finish
or exit first. Submission on 5 October omitted that dependency after
verifying the predecessor had completed on 2 October and had aged out of
the current scheduler listing. This does not change any seed or model
budget and does not guarantee all ten seeds fit the wall limit.

The intended budget retains the focused C13 experiment's 100 latent
updates per arm, learning rate `400`, gradient norm cap `0.000125`, trust
radius `3`, temperature `1`, occupancy threshold `0.5`, one argmax plus
eight categorical samples at states 0–100, and final-argmax native pose
seeds 17 and 23. These settings explore a region whose chemical support
is unvalidated. Keep objective direction and score scale unchanged when
comparing initialization modes, and report any departure explicitly.

Before optimizing any draw, attempt native context preparation for
**counts 1–50 sequentially**, reusing the verified source054 preparations
for counts 11–15. Seal a new immutable catalog after preparation; record
every attempted count, reused identity, preparation failure and actual
supported count. A failed preparation does not make that count supported.
Never silently clamp occupancy or substitute a different start to fit
the catalog. Empty selections remain unsupported. A per-member failure should be recorded before the
sequential workflow moves on; source or checkpoint integrity failures
must stop the workflow rather than continue under unknown provenance.
Record incomplete members separately if a wall limit interrupts the job.

## Gates and confirmation

The frozen VAE and Boltz identities, shared graph schema, adapter feature
loss and native context hashes must remain explicit. This experiment
changes initialization, not the graph-to-Boltz chemistry contract.
Retain hard/native context parity checks and test the conditional latent
gradient at the actual random start with its mask held fixed. No derivative
through discrete slot selection is implied. The planned finite-difference
grid is `[0.001, 0.003, 0.01, 0.03, 0.1]`, expanded from the C13 run's
`[0.01, 0.03, 0.1]`. Relative/absolute tolerances remain `0.05`/`1e-6`,
with at least two passing scales required in each gradient/random probe
direction. This expands probe coverage and call cost; report that budget
difference when comparing with C13. Tolerances are not relaxed to obtain
ten successful trajectories.

Source056 separates responsibilities: `latentmol.initialization` defines
the scorer-independent start policy; `boltz.workflow_process` coordinates
isolated subprocesses; `boltz.random_start_study` orchestrates the serial
study; and `boltz.initialization_report` summarizes completed and partial
artifacts without evaluating a model. The active-set runner preserves its
default encoded-start behavior, while the explicit random-start mode saves
the drawn `z` before checking occupancy support. Random starts need not
decode to a valid hard molecule to enter the relaxed scorer. These are
statically reviewed implementation boundaries, not executed fixtures or
native validation.

The primary output of each completed arm is its **final step-100 argmax**,
chosen before fresh-native scores are observed. Invalid endpoints are
recorded as invalid; neither repair nor choosing another saved candidate
silently replaces them. Valid eligible starts/endpoints receive the
predeclared fresh-native confirmation. An invalid starting graph cannot
supply a valid paired start/end native comparison, even if its endpoint
is valid. Preserve unpaired endpoint scores separately where supported.
Report native failures, identity changes and all ineligible comparisons.

Optimize only the declared binary head and report both native affinity
heads separately. A higher continuous log-IC50 value is not evidence of
stronger activity; that head's intended comparisons apply to actives.
Intermediate best-proxy points remain descriptive, not a way to reselect
the final molecule. Fresh Boltz predictions remain model evaluations.

## Required report and denominators

Keep one row per seed and arm, including starts that never reach an
optimization step. Report attempted, started, completed, failed and
unfinished members separately; the original denominator remains ten
random draws. Avoid presenting a success-filtered subset as ten runs.

- Initial/final latent identity, selected count and slot identities;
  maximum count deviation, count switches and same-count slot switches.
- Initial/final argmax validity, canonical SMILES when valid, element
  composition, carbon fraction and heteroatom counts. Invalid graphs
  still retain raw atom/bond labels and failure reasons.
- Distinct valid final molecules across seeds, duplicated endpoints and
  endpoint changes relative to each start. Distinguish final-molecule
  diversity from the much larger correlated categorical-sample pool.
  The implemented comparison uses canonical SMILES identity and pairwise
  Tanimoto similarity of RDKit Morgan radius-2, 2,048-bit fingerprints,
  excluding stereochemistry under the shared graph schema. Report the
  eligible pair count; fewer than two eligible endpoints has no mean
  pairwise similarity, rather than a fabricated zero.
- Full relaxed score/displacement/gradient history, context-switch
  discontinuities, trust-boundary visits and hard validity over time.
- Per-seed fresh-native binary and continuous results, identity eligibility,
  paired differences where defined, and missing/failed predictions.
- Actual model calls, preprocessing and optimization/rescore times, with
  shared or deduplicated predictions identified. Paired active/fixed
  results from the same draw are not independent random initializations.

Compare descriptively with the separate C13-start run, keeping initialization,
context preparation and successful-evaluation denominators visible. Ten
draws can expose collapse or repeated endpoints, but cannot establish
broad search coverage or independence from initialization. A no-count-change
outcome is informative and must not be turned into an atom-count adaptation
claim. Exact execution records and all failures belong in the
[experiment ledger](05_experiment_ledger.md).

## Running and reading the study after access is restored

`configs/boltz_random_start_study.yaml` is the strict study manifest. It
names the pinned baseline config, model/cache/native assets, count list,
seed list, expanded probe grid and reference C13 pilot. It allows 900
seconds per preparation subprocess, 5,400 seconds per seed's paired
optimizer and 3,600 seconds per native-rescore subprocess. These child
limits do not extend the outer allocation's wall limit.

The following is the prepared **allocated HPC invocation**, from an
immutable release inside its LSF job. It is not a command to run locally;
the user's 5 October request now authorizes submission, while the actual
receipt and scheduler evidence still must be recorded:

```bash
bash scripts/hpc/boltz_run.sh bash scripts/hpc/boltz_random_start_study.sh \
  /absolute/path/to/protocol.yaml PROTOCOL_SHA256
```

The one-shot submission helper is:

```bash
bash scripts/hpc/submit_random_start_study.sh \
  RELEASE_DIR ARCHIVE_SHA256 CONFIG CONFIG_SHA256 RECEIPT_DIR [AFTER_JOB]
```

`RELEASE_DIR`, `CONFIG` and `RECEIPT_DIR` must be absolute paths; the
receipt directory must be new. The archive must be at
`RELEASE_DIR.tar.gz`. The optional numeric `AFTER_JOB` adds an
`ended(AFTER_JOB)` dependency; use `29545141` for the planned C13
predecessor only after checking its current scheduler state. The helper
verifies archive/config/source identities, rejects an existing study
output root, then reserves the receipt directory and requests exactly one
12-hour `gpua40` job with four cores, 8 GB per core and one exclusive GPU
on one host. It does not create a job array or submit child jobs. Inspect
`RECEIPT_DIR/bsub.txt` and scheduler state before retrying an uncertain
submission. No submission occurred during the earlier suspension; renewed
authorization was followed by verified one-shot submission **29603601**.
The exact command, paths, hashes and receipt are in the 5 October ledger
entry. The actual command had no sixth `AFTER_JOB` argument, because the
predecessor was already complete.

The current study-config SHA-256 is
`5f6b5d6378a46c632051b112b4d9f0d516991f59fed9bb362541123a87c5c8d6`.
Archive/source identities are recorded in the ledger's local packaging receipt.

The wrapper first runs the selected active-set, rescore, diagnostic,
count-preparation and random-study fixture modules. Only if they pass does
it invoke:

```bash
python -m latentmol.boltz.random_start_study \
  --config /absolute/path/to/protocol.yaml --sha256 PROTOCOL_SHA256
```

No project import, fixture execution, model load, preprocessing or numerical
evaluation has been performed as part of the current local static review.
Successful Python parsing alone does not verify dependency compatibility,
GPU behavior or correctness of the study. The exact static-check command,
release hashes and eventual submission receipt belong in the ledger.
Static review passed for 109 Python files and both shell launchers; a YAML
parse checked seeds, counts and the pinned baseline identity. These checks
do not replace the pending HPC fixture and native-model runs.

The orchestrator writes a unique `ExperimentRun` beneath the manifest's
output root. Read its `run.json`, `trials.json` and
`initialization_report.json` together: a completed orchestration run means
the declared members were processed, not ten valid molecules or ten
successful optimizations. `trajectories.csv` contains collected per-state
diagnostics when available. `contexts/preparation.json` and
`contexts/contexts.json` distinguish every attempted count from the sealed
supported catalog. Each `seeds/seed-XXXX/` retains the generated config,
optimization/rescore command logs and child experiment artifacts.

Reports distinguish verified completed endpoints from partial or
`trajectory_complete_unverified` observations. A finished arm summary
alone cannot supply a verified final molecule unless the parent completed,
frozen-state checks passed and candidate-file identities agree. Partial
JSON/JSONL after interruption remains visible through report warnings;
it is not silently interpreted as a completed candidate budget. Subprocess
timeouts/interrupts terminate their process groups before the next member
starts. The orchestrator itself holds no scorer on the GPU.
