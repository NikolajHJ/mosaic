# Portable metadata and allocated CPU controls

Explicit portable exception: `test_runtime_reservations.py` (28 pure metadata
cases), `test_segment_capacity.py` (60 schema/accounting cases), and ten
file-count/content-identity cases selected by `scripts/local_metadata_preflight.py` can run
locally. The runner uses no model/scientific library or fake allocation. Keep
other allocated guards intact; these 98 cases still participate in the complete
HPC release suite and cannot establish native/gradient readiness.

All 98 portable cases passed locally in 0.94 seconds on 28 October. The isolated
pytest environment had no NumPy, JAX, Torch, Equinox or Mosaic installation.
Actual CPU release and native CUDA/FP64/gradient/pilot checks remain mandatory.
See [resource candidate19](../runs/20261028-precision-continuation-19/README.md)
for exact evidence and preserved earlier Windows metadata failures.

Attempt19 adds28 explicitly portable VAE shared-quota tests and one portable
foreign-family classification case, passed locally in0.26s. They use only
metadata/AST/JSON/temporary files and synthetic command strings. They neither
load the VAE nor establish actual external-job provenance or scientific readiness.
Five additional same-job terminal-lifecycle controls passed locally, keeping all
retained files charged after an execution allowance is retired. The complete
immutable879-case allocated suite includes18 HOME-reader classification controls;
all879 passed allocated CPU before new native checks were queued. Local150
metadata cases cannot replace those native/model/FP64/gradient checks.

Current short-continuation16 suite has785 allocated cases. New metadata-only
controls cover two concurrent CUDA-client orchestration (`test_cuda_ownership`),
closed rolling ledgers (`test_segment_capacity`) and live inventory/claims
(`test_rolling_capacity`). The actual tiny CUDA calculation runs later on its
allocated GPU; fake transport fixtures cannot certify GPU/native readiness.
Short segment/runtime and owned-child periodic admission controls belong to the
precision driver/controller tests. Chemistry reference05 separately plans40 cases
and is not yet packaged or allocated-tested.

`test_qualification.py` tests smooth agreement, deliberate gradient errors,
boundary-infeasible original probes, FP32 signal loss and scale-dependent
curvature/nonsmoothness, plus planning/evidence/frozen-identity protections.
These numerical fixtures run only inside an allocated HPC CPU job; they are
not pretrained native-model gradient validation.

The new `test_admission.py`, `test_continuation_driver.py` and
`test_continuation_ops.py` cover the separately versioned prospective policy,
restricted original-checker/readiness adaptation, exact generated startup argv,
and guarded serial child dispatch. Run all four modules only in the immutable
attempt's allocated CPU job. Neither toy passes nor premodel startup receipts
establish a native model pass or optimization start.

`test_protenix_endpoint_forensics.py`, `test_protenix_forensics_native.py` and `test_protenix_forensics_ops.py` cover the separate Protenix fixed-point investigation. All fixture execution is allocated HPC; local review is AST/bytes only.

The four `test_protenix_precision_*` modules cover authenticated precision-cast
clones, lossless promotion/dtype audits, positive/adversarial endpoint policies,
metadata startup/preservation and one-shot CPU/GPU dispatch. Execute only in the
new immutable allocated CPU capsule, never locally. Toy/reference fixtures alone
cannot qualify the pretrained Protenix endpoint.

Repair06 CPU29820662 passed223 rawcases:19diagnostic/50admission/49driver/
100controller/5current-capacity. Pro03 CPU29820595 passed278:89reference/
30precision-policy/82native-harness/72controller/5capacity, all0fail/error/skip.
The old low-signal linear toy uses stronger fixture-only coefficients to remain
informative under the tighter precision rule. Original production coefficients
are unchanged. Hand-calculable saved-row intervals stay noncertifying and wrong
zero/sign/1.5x gradients remain rejected. Current baseline tests cannot certify
native correctness. Both complete suites and actual startups ran allocatedHPC;
no fixtures/scientific arrays executed locally. Native GPU results stillpending.


Precision04 adds8 charge-prefix/AST controls and extends exception-restoration
coverage to the charge encoder. Its full286-case suite is declared, not passed.
The complete native FP64 trace audit/reference comparison remains mandatory.
Four new continuation-driver tests cover actual v2 policy identity and reject
stale labels or wrong module origins; they have not been executed. Static AST
reversal confirms that this driver edit changes no numerical behavior. No
Boltz/DBMol numerical correction is selected without the current raw FD reports.

test_saved_gradient_forensics.py adds eight allocated CPU controls for exact component matching, softmax chain rules and negative recomposition controls. They are not native readiness tests and are never executed locally.

Measurement07 adds14 local-logit controls,10 component-resolution controls, actual JAX adapter-entry coverage and strict dispatch-negative fixtures. The declared full seven-module suite is258 cases, pending allocated raw XML verification. No native pass follows from toys or preparation.


`test_chemistry_endpoint_diagnostic.py` and `test_chemistry_endpoint_ops.py`
add33 allocated controls for saved-gradient decomposition, frozen-input scope,
exclusive evidence writes, declared resources and rejection of optimizer
dispatch. Numerical fixtures run only on allocated HPC, not the laptop;
actual native diagnostics are separate from these fixture outcomes.


`test_chemistry_native_qualification.py` has24allocated controls for real
startup/source/root/queue/dtype identities, unchanged policy constants, positive
raw-gradient controls and zero/reversed/scaled rejection. CPU passes do not
establish native or combined chemistry admission. No numerical tests locally.


Qualification04 now requires29allocated cases: original24 plus whole-inventory
retry, protected input failure, bounded exhaustion, capacity rejection and
failed-result rejection. Actual CPU29834574 passed all29/no skips/errors/failures.
Native/combined loss validation remains separate; no numerical tests locally.


Recovery controls `test_dbmol_reference.py`, `test_dbmol_endpoint_precision.py` and `test_recovery_capacity.py` run only on allocated HPC. They exercise full-precision operation auditing, original scalar/gradient transport, wrong-gradient rejection, preserved source inputs and complete budgets. CPU fixture passes do not certify pretrained native gradients or authorize child optimization.


Protenix reference placement09 keeps old failures and production path intact; details in the owning run20261028-protenix-endpoint-fix-09. No local numerical execution.


Continuation14 declares 590 allocated CPU cases across14 modules: original261 plus new driver64/hooks57/controller88/Pro-reference56/DB-reference17/DB-finer36/recovery11. Torch reference fixtures run in the original Torch interpreter and other fixtures in the Protenix interpreter; original raw JUnit rows from both runs must exactly equal the merged case inventory. Preparation is not a fixture/native pass. Real metadata startup commands precede GPU integration and full-model pilots.

Portable integration15 repeats the full suite with two additional PurePosixPath/PureWindowsPath declaration controls:592 allocated cases. No gate reduced; source14 failed before fixtures and remains failed.


`test_chemistry_continuation.py` contains34 explicitly portable metadata/AST cases: unchanged two-arm settings/schedules/retention, full measured horizons and fail-closed runtime, actual-coefficient cohort insertion, original optimizer count AST, exact original command preservation, authenticated empty metadata, original worker frozen-response semantics and host-memory allocations. These import no scientific array/model library. They passed locally and MUST repeat with the full944 inherited allocated cases before03 dispatch. Source02's24-case version was submitted with those944 tests; its actual outcome is unverified after SSH denial. No local pass certifies native gradients, chemistry, replay or optimization.
