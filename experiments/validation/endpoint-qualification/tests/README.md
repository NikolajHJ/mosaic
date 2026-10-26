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
