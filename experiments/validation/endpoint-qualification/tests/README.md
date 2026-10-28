# Allocated CPU controls

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
