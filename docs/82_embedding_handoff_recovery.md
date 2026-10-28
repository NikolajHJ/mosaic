# Trained mapping handoff recovery

On28 October2026 the user authorized fixing the handoff, resubmitting the pilot
with the already trained mapping, then continuing to10,000 iterations per
molecule if the checks pass. The VAE continuation must stay unchanged.

Source143 changes controller admission and callback routing only. Numerical
workers keep the immutable source142 release, original plan, original epoch36
VAE, the same three starts, frozen native models, contexts, poses, objective,
base optimizer, consistency caps and scientific gates. The successful5,000-step
mapping fit is reused; it is not repeated.

## Evidence and correction

Fit job29839755 ran16:23:47–16:39:52CET on28 October and completed successfully.
Both its exact-name outer LSF summary and `bhist` confirm success. The saved
allocated execution and published result hash also pass. The selected mapping
is step4800, checkpoint SHA256
`c8d86551849c7f7e1de5218b7121cbecacb7f9dbc9ee11a5aea5972187a81551`;
the unchanged calibration SHA256 is
`1c436618bc6dc26dfb049908b11477afc5f2e7b291132db4577a99631f0c69f8`.

Original callback29839756 exited at16:39:57CET on its scheduler equality check,
before creating a pilot or transition intent. Its raw scheduler response was
not saved, so that original mismatch's precise cause remains unproven. A later
unchanged-controller retry failed the aged-out-job branch before submission.

The captured actual cluster contract shows the demonstrated later bug:

- Formatted `bjobs` returns **exit0**, empty stdout and
  `Job <29839755> is not found` on stderr for an aged-out job. The old branch
  mistakenly required a nonzero exit code.
- JSON `bjobs` returns exit0, empty stderr, `COMMAND:bjobs`, `JOBS:1` and one
  exact `JOBID` record with that exact `ERROR`. Live JSON also supplies exact
  `STAT` and `JOB_NAME` fields.

The new controller uses this structured contract and saves full raw responses.
Live admission requires the exact job ID, full job name and DONE. An exact
aged-out record additionally requires the exact successful outer LSF log and
the authenticated allocated execution/result proof. Unknown errors, duplicate
or wrong records, empty/non-JSON output, network failures, RUN/PEND/EXIT or a
different name do not pass. This changes no numerical tolerance or cap.

Original source142 and all failed callback/retry/access receipts are retained.
Source143's configuration also pins the successful original outer LSF log's
SHA256, so a saved model file alone cannot release the pilot.

## Controller and readiness boundaries

The repair authenticates its own archive, manifest and plan, then independently
authenticates the complete original source142 release and the exact CPU/fit
results, executions, mapping and calibration. Its allocated CPU readiness
requires new handoff fixtures plus the actual generated original pilot CLI.
That CLI probe invokes the real original entrypoint and stops at the native
smoke boundary, before CUDA/model setup. The authentic package imports Torch
and NumPy on the CPU. Readiness v2 records those imports, uses explicitly empty
CUDA visibility, checks cached CUDA initialization before/after and traps CUDA
initialization and model construction/loading. It forbids parent writes and
submissions. The hashed probe and inline record must match the exact argv,
original worker source, fitted model/calibration, and zero writes, submissions,
CUDA initializations, model loads and optimization updates.

The original450-case CPU/profile/start proof is adopted with its exact hashes;
it is not represented as a new native or optimization pass. New readiness is
controller evidence only.

Scientific `cwd`, archive environment, source/manifest/plan identity, source142
run/control paths and worker arguments remain original. All future `run-stage`
and `advance` commands route through the authenticated absolute source143
controller and pinned readiness proof. Fit/CPU dispatch and `submit-initial`
are forbidden in this recovery, preventing accidental retraining. Existing
exclusive transition/submission intents, quota checks and ambiguity protection
remain in force.

After successful readiness, the original pilot must still pass its native GPU
smoke, all3 native oracles, all6 derivative/production bindings, fixed-cap
validity/consistency and nonzero progress checks. Pilot20 iterations count
toward each10,000 target. Only then do serial round-robin allocations resume
each case, with at most24h wall/23h internal and unchanged guard/stall policies.
Every10 iterations the original worker saves relaxed/hard observations. Final
native comparisons remain separate. No source140 VAE or source141 native-only
recovery job, source, setting or checkpoint is changed by this repair.

## Immutable release and initial CPU submission

Source143 attempt001 contains626 authenticated members. Archive SHA256
`6f3bcaea485793844579694d371c06c9f7c93d392444404dfce43fa6ef361754`,
manifest SHA256
`03d063ed9eeb6ffac70bf7affb7cf7151796c68b5d6efe58d185937360030c13`,
repair plan SHA256
`fe14de14231cb395bcb0781419c4d2b536772a82f9cde5c44ee097768cb0571c`.
CPU readiness29844977 was submitted with4 slots/45min wall/40min internal.
At this initial entry it was RUN; submission is not a passed readiness gate
or a new pilot submission. Final receipts are recorded below.

Entry points: `scripts/embedding_handoff_campaign.py` for local stdlib packaging,
`scripts/hpc/embedding_handoff_readiness.py` for allocated CPU validation, and
`scripts/hpc/embedding_handoff_recovery.py` for authenticated future handoffs.
The numerical experiment remains
`source-142-epoch36-guarded-embedding-long-attempt001`.

## Attempt001 retained; real preflight exposed an overstrict import check

Allocated CPU29844977 completed all75 fixtures with zero failures/errors/skips
in22.98s, then the genuine pilot entrypoint preflight failed. The original
`latentmol.boltz.__init__` imports the tensor adapter and its CPU Torch
dependency before main runs. The new v1 probe incorrectly prohibited that
existing CPU import. No readiness proof or pilot intent was produced; the
allocation exited1 and is **not promoted**. Exact raw log/XML bytes were
downloaded with remote SHA/size checks under
`outputs/hpc-results/source143-embedding-handoff-recovery-attempt001/failed-readiness-verified`.

The correction is isolated in a new immutable attempt002 and readiness v2;
it executes the authentic package without stubs or bypassing its imports.
CPU imports are distinguished from CUDA ownership/model execution as already
required by the original CPU-only coordinator policy. The source142 numerical
release and its native/derivative/validity/consistency/optimization gates remain
unchanged. No successful fit is repeated to work around this preflight failure.

## Attempt002 passed; trained-mapping pilot resubmitted,28 October2026 20:58CET

Immutable repair attempt002 archive SHA256
`6e851def4a043df774bb259fc5a16c84ca9faada85c4616c99d65b5c6a092d41`,
manifest SHA256
`57469430d6e13a8708fb593d5433b454a867cd0109c62a8355dfde3d7107077f`,
repair plan SHA256
`fe14de14231cb395bcb0781419c4d2b536772a82f9cde5c44ee097768cb0571c`.
Only the two repair modules and their two test files changed from failed001;
all original numerical worker/plan/trained input bytes remain source142.

CPU **29845008 DONE**: **85 passed**, zero failures/errors/skips, pytest29.09s.
The real original-package pilot CLI preflight also passed: authentic CPU
Torch/NumPy imports, empty GPU visibility, cached CUDAfalse before/after,
zero CUDA initializations, model loads, parent writes, job submissions and
optimization updates. Readiness v2 SHA256
`2470071046d191334a8e89bc1de63c203ae8614609c399d83d9f68736de07674`;
exact pilot CLI receipt SHA256
`3954b8b371bee65dba3777d4ddd8aefea90b1bf70c77e37bf71cadd870dc034c`.
The complete parent450-case readiness,5,000-step fit, fitted mapping/calibration
and successful outer fit LSF summary were authenticated unchanged.

The repaired original fit→pilot transition completed successfully. GPU pilot
**29845020** was submitted20:57:16CET, with serial callback **29845021**
submitted20:57:25. Both future run-stage and callback commands point to the
absolute authenticated attempt002 recovery controller and pinned readiness.
The scientific job still runs in original source142 with its original source,
manifest, plan, trained mapping and calibration. It requests one exclusive80GB
A100/four CPU slots/4h wall/3h50 shared internal deadline.

At the20:58 snapshot, pilot29845020 is **PEND**, waiting on processor-affinity
units on one host; callback29845021 waits its expected ended dependency. No
new native gate, pilot result or optimization update exists yet; no reliable
start ETA. Successful pilot gates and progress will release the original
serial10,000-iteration-per-case campaign, with observations every10 and all
existing checks intact. A failed check or scientific stall still stops it.
Source140 VAE continuation and source141 native recovery were not changed.

[Submission summary](../outputs/hpc-results/source143-embedding-handoff-recovery-attempt002/submission-summary001.json),
[verified readiness bytes](../outputs/hpc-results/source143-embedding-handoff-recovery-attempt002/readiness-verified/download-integrity.json),
and [actual pilot CLI](../outputs/hpc-results/source143-embedding-handoff-recovery-attempt002/readiness-verified/pilot-cli.json)
bind the results. Failed attempt001 and failed original callback/retry/access
receipts remain intact. Submission is not a native/pilot/optimization pass.
