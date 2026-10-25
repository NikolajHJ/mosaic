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
