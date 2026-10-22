# JQ1 fixed-step comparison: tests

This folder belongs to [the experiment protocol](../README.md).

- [test_fixed25_ops.py](test_fixed25_ops.py): Allocated fixed-rate scope and real submission-vector contracts.
- [test_fixed25_recover_ops.py](test_fixed25_recover_ops.py): Allocated controller fixtures; never run a molecular model or optimization.
- [test_fixed25_recover_storage.py](test_fixed25_recover_storage.py): Allocated recovery storage fixtures; all scientific runs and old sources stay unchanged.
- [test_fixed25_split_ops.py](test_fixed25_split_ops.py): Allocated split launcher contracts; original numerical/model gates are reused unchanged.
- [test_fixed25_split_storage.py](test_fixed25_split_storage.py): Allocated all-work3 adoption contracts; no model or scientific arithmetic.
- [test_fixed25_storage.py](test_fixed25_storage.py): Allocated storage scope checks for the three-arm restart.
- [test_jq1_fixed_ops.py](test_jq1_fixed_ops.py): Allocated stdlib contracts for report scheduling, CPU proof and storage scope.
- [test_jq1_fixed_publish.py](test_jq1_fixed_publish.py): Allocated stdlib fixtures for immutable, byte-only fixed-report delivery.
- [test_jq1_fixed_render.py](test_jq1_fixed_render.py): Allocated CPU presentation fixtures; synthetic saved arrays never certify a run.
- [test_jq1_fixed_report_contract.py](test_jq1_fixed_report_contract.py): Allocated contract fixtures: fixed-step evidence and honest actual outcomes.
