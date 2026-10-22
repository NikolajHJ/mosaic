# BRD4 frozen Boltz-2 affinity optimization: tests

This folder belongs to [the experiment protocol](../README.md).

- [test_boltz_affinity_report.py](test_boltz_affinity_report.py): Allocated saved-artifact report fixtures; no molecular scorer is executed.
- [test_boltz_arm_readiness.py](test_boltz_arm_readiness.py): Evidence-only adoption contracts; run on an allocated DTU HPC CPU.
- [test_boltz_hpc_tmpdir.py](test_boltz_hpc_tmpdir.py): Allocated Linux CPU regression checks for native Boltz AF_UNIX startup.
- [test_boltz_native_mapping.py](test_boltz_native_mapping.py): Allocated CPU tests of native Torch mapping; no JAX/decoder dependencies.
- [test_boltz_submission_controller.py](test_boltz_submission_controller.py): Stdlib launch-contract checks; execute only on allocated DTU HPC CPU.
