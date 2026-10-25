# Allocated CPU controls

`test_qualification.py` tests smooth agreement, deliberate gradient errors,
boundary-infeasible original probes, FP32 signal loss and scale-dependent
curvature/nonsmoothness, plus planning/evidence/frozen-identity protections.
These numerical fixtures run only inside an allocated HPC CPU job; they are
not pretrained native-model gradient validation.
