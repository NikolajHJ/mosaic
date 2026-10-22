# Frozen model integration tests

Model interfaces, prediction/loss contracts, native parity, multi-sample behavior and explicit model smoke tests. Some tests load checkpoints; respect markers and allocated HPC requirements.

Use default pytest discovery or select this directory in an allocated DTU HPC
job. Local layout verification parses source only and is not a numerical pass.
Shared fixtures remain under [tests/data](../data/).

## Files

- [test_ablang2_loss.py](test_ablang2_loss.py)
- [test_esmfold2_multisample.py](test_esmfold2_multisample.py)
- [test_model_smoke.py](test_model_smoke.py)
- [test_native_esm.py](test_native_esm.py)
- [test_opendde.py](test_opendde.py)
- [test_opendde_contracts.py](test_opendde_contracts.py)
- [test_promera.py](test_promera.py)
- [test_structure_prediction.py](test_structure_prediction.py)

See the [test index](../README.md), [experiment index](../../experiments/INDEX.md)
and [nested layout map](../../experiments/folder-layout.json).
