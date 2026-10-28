# Prospective endpoint qualification diagnostics

Current28October10:32CET: [portable continuation15](runs/20261028-precision-continuation-15/README.md) passed all592 allocated CPU cases and seven actual startup routes, independently verified. Fresh full-model GPU validations Pro29839086 andDB29839088 plus own-ended callbacks are queued; optimizer continuations await their complete gates. Original saved20k/7500 states, settings and failed outcomes remain unchanged.

Latest 28 October 2026: [Pro09](runs/20261028-protenix-endpoint-fix-09/README.md) native29838596 and [DB13](runs/20261028-dbmol-endpoint-recovery-13/README.md) native29838712 passed independently verified separate endpoint qualification. Original failures remain failed. [Continuation14](runs/20261028-precision-continuation-14/README.md) integrates these checkers with the original full-model drivers, pilots, runtime and whole-storage guards; its 590-case CPU suite and native integration are separate required gates before any optimizer child. Read its actual receipts for submission state.

This experiment investigates whether a failed categorical endpoint probe is an
infeasible simplex perturbation, a resolved numerical disagreement, or an
unresolved precision measurement. It preserves every original failure and does
not qualify or restart a saved experiment.

The evaluator-independent policy is `endpoint-qualification-diagnostic-v1`.
It fixes contrasts and all perturbations from graph geometry before any native
loss or gradient call. Original scales 0.01/0.003/0.001 remain explicit; a fixed
geometry-derived ladder is supplementary evidence. It never chooses a scale in
response to a loss or a passing result, clips probabilities, projects a graph,
or changes a saved state. Every actual FP32 input, displacement and SHA is kept.

Independent primal and AD repeats, scalar parity, represented displacements,
signal resolution and frozen-identity hooks accompany each probe. An original
infeasible scale produces `boundary_inconclusive` even when supplementary scales
agree. Otherwise resolved disagreements remain `numerical_disagreement`;
insufficient representable signal is `unresolved_precision`. Diagnostic `passed`
does not certify an original gate. Optional saved-logit controls have their own
baseline and explicit scope; equality with raw saved probabilities is never
assumed.

Ownership:

- [scripts/qualification.py](scripts/qualification.py): lazy-NumPy, evaluator-
  independent diagnostic policy; no model setup, optimizer, file mutation or
  submission code.
- [tests/test_qualification.py](tests/test_qualification.py): allocated CPU toy
  positive/negative controls, not pretrained native evidence.

From the Mosaic root, **inside an allocated DTU HPC CPU job**, the fixture command
is `python -m pytest experiments/validation/endpoint-qualification/tests -q
--junitxml=<new-attempt-output>/qualification.xml`. Use the authenticated runtime
and immutable capsule declared by the owning attempt. Native Protenix/Boltz/DBMol
evaluators need separate authenticated saved inputs, source/model/context identities
and original null/component checks; these toy fixtures cannot establish them.

Current status, 25 October 2026: implementation prepared; local AST/byte inspection
only. No allocated fixture pass, native diagnostic result, production qualification,
continuation job or gate waiver follows from this preparation. Run records and
actual allocation/evidence links are maintained by the campaign coordinator.

Limitations: deterministic repeated forwards do not rule out FP32 rounding;
agreement along a finite set of directions does not prove the entire Jacobian;
finer-scale agreement does not diagnose a unique cause or erase a coarse-scale
disagreement. Changes to a future production admission protocol require separately
versioned validation and explicit scientific scope.

Allocated toy validation completed on25October2026 in the independent report
source03 CPU job29812102: all19 cases passed, with123 total report/policy cases,
zero failures/errors/skips, actual DONE/hpc/LSF and independent byte/XML/schema
authentication. These are toy diagnostic controls, not pretrained native probes.
The production checkers and all saved scientific outcomes remain unchanged.
Evidence: [report run record](../../reporting/jq1-live-progress/runs/20261025-jq1-live-report-01/README.md).

On25October2026 the user explicitly authorized a separately versioned prospective
endpoint-aware admission check before the requested non-chemistry continuations.
That work belongs to [the continuation run record](runs/20261025-continuation-admission-01/README.md)
and [its declared policy](runs/20261025-continuation-admission-01/prospective-policy.md).
`admission.py` is a new prospective API; the diagnostic-only `qualification.py`
above and all original failed runs remain unchanged. Numerical fixtures, actual
generated-command startup checks and native validations must pass on allocated
HPC before conditional child optimization. Preparation is not a pass or start.

Actual source01 native validation rejected Protenix and both Boltz workers failed
their import path before gradient testing. See the immutable failure audit in the
dated run. A [restricted worker-fix02](runs/20261025-continuation-worker-fix-02/README.md)
preserves that failed source and Protenix's execution block. It changes only
subprocess import plumbing, startup coverage and primary-error reporting; the
prospective numerical policy and all original scientific settings remain exact.

A separate [Protenix20k endpoint investigation](runs/20261025-protenix-endpoint-forensics-01/README.md) preserves the failed admission and all optimization settings. It measures the exact existing probe cohort, scalar execution paths, endpoint derivatives and contact-partner switches; it cannot certify or restart a study.

A new [separate-precision Protenix checker](runs/20261025-protenix-endpoint-fix-01/README.md)
is being validated at the immutable20k endpoint. It separates the original FP32
measurement from an authenticated full-FP64 reference and rejects wrong-gradient
controls. This validation-only capsule cannot submit optimizer work or erase any
original failure. Preparation/local AST checks are not an allocated/native pass.

The first separate-precision candidate failed its real-JAX dtype traversal
canary (266/267passes) before any native work. [Fix02](runs/20261025-protenix-endpoint-fix-02/README.md)
corrects that traversal and has passed all269 allocated CPU fixtures plus actual
metadata-only startup. Native saved-endpoint validation remains required; no
optimizer continuation or original recertification follows from the CPU pass.

Read-only26October10:44UTC update: worker-fix02 Boltz29813924 and DBMol29813926
both failed prospective endpoint admission; their callbacks submitted no child
optimization. Pro precision29816113 failed its initial FP32 scalar replay against
the authenticated forensic reference, before any completed FP64 comparison.
All86 compact metadata/source/own-job LSF/preservation bindings independently
verified; aged-out live history is represented only through authentic persisted
LSF, not fabricated current states. Saved progress remains Pro20k/50k,
plainBol10k/30k and DB5k/10k. Settings/results/paused monitor unchanged. See each
dated run's status-20261026-verification.json; no local scientific work or new jobs.

26October11:28UTC: [continuation06](runs/20261026-continuation-precision-06/README.md)
passed223allocated cases and actual startup; fresh Bol29820672/DB29820674 native
checks queued with own-ended callbacks. Original endpoint failures remain failed;
new prospectivev2 requires a stricter pre-forward signal budget and uncertainty
INSIDE unchanged15% tolerance, not favorable FD selection. Source03 historical
version binding,04output-root and05stale negative-fixture failures preserved.
[Pro precision03](runs/20261026-protenix-endpoint-fix-03/README.md) CPU278passed,
GPU29820643queued: fresh FP32 measurements/historical mismatch are saved before
qualification, followed by mandatory separate fullFP64 reference and negatives.
No new native pass, optimizer continuation or changed saved result/settings.
Storage now includes both submitted chemistry futures plus new512+272MiB scopes
and unchanged4GiB/100k reserves. Pro integration awaits a valid measurement;
Bol/DB callbacks remain conditional, paused monitor unchanged.


The user authorized [Protenix precision04](runs/20261026-protenix-endpoint-fix-04/README.md)
and [Boltz/DBMol investigation07](runs/20261026-endpoint-investigation-07/README.md).
Precision04 casts the authenticated integer-charge arcsinh operand to reference
weight dtype in the isolated FP64 reference only. Eight new charge controls
and full native dtype/agreement/restoration checks await allocated execution;
active local source03 declares286 cases and cannot submit optimizer work.
Investigation07 records that individual DBMol components passed but total failed;
Boltz's later failure is in a separate pilot bond-logit check. No numerical
remedy or relaxed gate is selected before reading detailed actual FD evidence.
The driver metadata now binds the actual v2 policy with four new regressions.
One read-only SSH attempt timed out; nothing newly staged, tested or submitted.


27October00:21Paris: access recovered and exact reports collected. Plain Boltz
passed raw admission but failed a separate pilot bond-logit check; DBMol total
failed one preselected atom-scale comparison despite component passes. No
scientific gate was weakened and neither continuation was resubmitted.
Precision04 CPU failed full entry capacity before tests; the numerically
identical [precision05](runs/20261027-protenix-endpoint-fix-05/README.md) is
staged but failed preflight before any CPU submission, even after bounded
verified inactive offload. Approximately1.08GB additional conservative headroom
is still needed for the checker group. No FP64/reference pass established.
Detailed evidence in investigation07; source04/05 failures preserved.

27October: exact duplicate source consolidation preserved all131 source
manifests/paths and allowed full unchanged storage admission. Precision05
CPU29826029/callback29826030 actualDONE,286 raw cases/actual startup passed;
GPU29826052 lastPEND. No native FP64 or Protenix continuation pass.
[Saved forensics01](runs/20261027-saved-gradient-forensics-01/README.md) CPU
29826033 DONE/8controls/exact17inputs preserved: correct Boltz softmax pullback
but sharp native derivative variation; DBMol recomposition matches with
underestimated assembled-scalar resolution. Original failures stay failed.
[Measurement07](runs/20261027-continuation-measurement-07/README.md) introduces
prospective rawv3/component precision and pilot-logitv1 with unchanged science.
CPU29826125 lastRUN/callback29826126PEND;258 raw cases not yet authenticated.
A stale metadata fixture expectation is corrected canonically; sealed07 stays
unchanged and [08](runs/20261027-continuation-measurement-08/README.md) is only
a planned future immutable retry. SSH255 timeout blocked terminal collection.
No new optimizer ID established, no old job replay or monitor resumption.


27October10:56UTC: [measurement08](runs/20261027-continuation-measurement-08/README.md)
passed258allocated CPU cases and actual startup/worker checks; new Boltz29829701
and DBMol29829703 native GPU checks actuallyqueued. Their continuations are
conditional on full native/pilot/runtime/frozen/startup/current-storage gates.
Original07 failed exactly one stale fixture and stays immutable. Pro05 failed
exact FP32 checkpoint replay before FP64; [precision06](runs/20261027-protenix-endpoint-fix-06/README.md)
adopts independently qualified compiler context as a candidate, requires fresh
exact replay/fullFP64, and has no optimizer branch. Saved results/settings remain
unchanged; paused monitor unchanged. Local work is static/bytes only.

Precision06 realstartup passed but4/301coordinatorfixture origin tests failed;
failedsourcepreserved. [Precision07](runs/20261027-protenix-endpoint-fix-07/README.md)
is a fixture-only retry with identicalruntime math/contextandstrictreplay.
Initialfullstorage failed by10.55MB; oneinactiveupload39.16MB was backedup
and351members verifiedbefore removal. Exactsource07 freshadmission fullgate
passed;CPU29829842/callback29829843 submitted. No nativeoroptimizerpass yet.

Latest27October11:25:42UTC: continuation09 is Boltz-only resource retry,
CPU261/startup/callback passed; GPU29829934 PEND. Continuation08DBMol29829703
PEND remainsauthoritative; no duplicate. Precision07CPU301/startup/callback
passed; GPU29829894 PEND. No blockedoptimizer started. Scientific settings and
saved results unchanged. Canonical continuation package/stage targetsONLY09.


27October: the user authorized investigation and correction of both blocked
chemistry continuations without changing settings or saved studies.
[Chemistry endpoint recovery02](runs/20261027-chemistry-endpoint-recovery-02/README.md)
first runs allocated saved-Protenix decomposition and separate native Pro/Bol
cohorts. It is explicitly diagnostic-only, preserves both original failures and
has no optimizer dispatch branch. A demonstrated correction and prospective
continuation still need their own full qualification; preparation is not a pass.


27October20:21CET: chemistry diagnostic02 CPU33 and both separate GPU jobs
29830750/29830751 completed. Independent byte/XML/LSF audit passed53compact
files; original/accelerated Pro records match exactly, all saved results
remain unchanged. These are diagnostics, not continuation qualification.
[Native qualification03](runs/20261027-chemistry-native-qualification-03/README.md)
now tests unchanged rawv3 Bol and separate fullFP64 Pro policies at the exact
chemistry10k boundaries. Its result is native-component-only: combined
low/high/actual-coefficient checks and BOTHtenpilots still precede continuations.
No optimizer branch or scientific setting change exists in this intermediate
capsule; original failures stay failed. Read actual campaign receipts.


[Qualification04](runs/20261027-chemistry-native-qualification-04/README.md)
fixes03's complete-storage inventory race without skipping files or partial
totals. Actual CPU29/LSF/startups/whole dispatch capacity passed; native Pro
29834600 and Bol29834601 are queued. Native/combined qualification and both
chemistry continuations remain pending, with all settings/results unchanged.


27October23:08Paris: [Protenix08](runs/20261027-protenix-endpoint-fix-08/README.md) fixes the separate reference geometry-wrapper charge cast;324allocated CPUcases passed, GPU29834817 queued. [DBMol10](runs/20261027-dbmol-endpoint-recovery-10/README.md) preserves the failed7500 endpoint and validates an independently promoted fullFP64 affinity reference;32allocated CPUcases passed, GPU29834830 queued. Neither native check has passed or submitted a continuation. Existing settings, saved results and original failures remain unchanged.


28October: [reference placement09](runs/20261028-protenix-endpoint-fix-09/README.md) retains source08 failedJIT/dtype/production proofs; separate reference-only GPU placement and full allocated checks precede any continuation. New preparation is not a native pass or optimizer submission.


28October: the user approved [DB finer-scale12](runs/20261028-dbmol-endpoint-recovery-12/README.md) with a fixed pre-forward8-scale ladder and unchanged15percent/two-adjacent/locality guards; the original reference10 failure remains failed. Source11 staging failed only on strict empty-lock metadata before any job and stays preserved. [Pro09](runs/20261028-protenix-endpoint-fix-09/README.md) CPU336 passed; its native common-device reference result remains required. Neither capsule can submit optimization; full reference-aware continuation integration/pilots/runtime/startup/storage remain.


Current DB finer retry is [13](runs/20261028-dbmol-endpoint-recovery-13/README.md): retry12 actualstartup rejected historicalroot10 before model/fixtures, callbackblocked GPU. Retry13 changes only own startup path, with96allocated cases/actualcommands required; source12 and all original failures remain preserved.

Current portable integration: [attempt15](runs/20261028-precision-continuation-15/README.md), CPU29839033/controller29839034 submitted after fullstorage;592 CPU cases/startups/native integration remain required. Attempt14 metadata startup failure is preserved.

Current engineering retry is [attempt16](runs/20261028-precision-continuation-16/README.md).
It preserves source15's real CUDA-ownership and storage failures, requests
same-job MPS, checks two actual CUDA clients before model loading, uses shorter
2,500/1,250-update Protenix/DBMol segments with measured full-check walltimes,
and replaces lifetime reservations with explicit rolling admission for new jobs.
CPU29841063 passed785 allocated cases; independent raw XML/LSF/byte verification
passed. New storage admission passed with45.35GB conservatively available versus
12.06GB required. Full native/reference/pilot checks still precede continuation;
no existing result or original scientific failure was changed.

[Chemistry reference repair05](runs/20261028-chemistry-native-qualification-05/README.md)
is a separate unsealed local preparation. The actual nested04 reference log
identified stale FP32 charge operations inside its FP64 reference; production
replay passed. Exact reference09 machinery and one-device placement are adopted
as a candidate, with40 allocated fixtures planned. Its staging deliberately
fails closed until a declared current cross-campaign writer/admission contract
is connected. No05 numerical pass, package, GPU or optimizer job is implied.
