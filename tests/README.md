# Tests

Select the affected area for a small local CPU check. Native/GPU tests require
their actual assets/device; a local fixture pass makes no native-model claim.

| Area | Selection |
| --- | --- |
| Models, training and continuation | `tests/training/` |
| Chemistry encoding, dataset/cache and sampling | `tests/datasets/` |
| Reconstruction and graded quality | `tests/evaluation/` |
| Chemistry environment and connectivity terms | `tests/guidance/` |
| Latent priors | `tests/design/` |
| Existing campaign and integration regressions | Existing named `tests/test_*.py` files |

`python -m pytest tests` discovers grouped and remaining campaign tests once.
`pyproject.toml` retains the unique helper-module paths used by older fixtures.
The current HPC launch scripts use the grouped test paths. Historical receipts
and sealed source trees retain their original selections and files.

New shared tests go beside their subsystem; experiment-only fixtures go in that
experiment's home. Do not create copies or migration aliases for new tests.
`test_package_organization.py` verifies the retained import/CLI compatibility.
