# Confidence comparison commands

`compare.py` runs only allocated, disposable chemistry-only paired pilots and
reports raw validity/diversity/gradient/runtime measurements. `confidence_operator.py`
packages exact source bytes using stdlib, runs allocated CPU work on node-local
storage and verifies the returned evidence. See the parent protocol for scope,
limits and the distinction from native binding optimization.

`verify_evidence.py` independently audits source/archive bytes, every evidence
member, raw JUnit cases, actual same-job terminal scheduler/LSF and complete
comparison metadata using only stdlib. It performs no local array calculations.
