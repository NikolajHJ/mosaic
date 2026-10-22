# Test navigation and ownership

This root contains this index and the collection hook `conftest.py`. Shared test implementations are grouped by
subsystem; experiment-specific tests remain with their experiment.

| Folder | Purpose |
| --- | --- |
| [core/](core/README.md) | Shared cache and protein-design optimizer behavior |
| [models/](models/README.md) | Frozen-model interfaces, prediction/loss contracts and model smoke tests |
| [small_molecule/](small_molecule/README.md) | Reusable graph, chemistry, native-bridge, optimization and artifact contracts |
| [data/](data/) | Shared authenticated structures and chemistry-oracle fixtures; retained in place |
| [Experiment tests](../experiments/INDEX.md) | Tests specific to one named protocol or campaign |

In an allocated DTU HPC job, from the authenticated source root, use the default
`python -m pytest`, or select a canonical file/directory. `pyproject.toml` defines
the discovery roots `tests/core`, `tests/models`, `tests/small_molecule` and
`experiments`, with stable helper-module paths. This collects all 82 test modules
once. Existing marker selections still apply; some tests import numerical/model
libraries or require real checkpoints.

Former root adapters now live under [scripts/compat/tests/](../scripts/compat/tests/README.md),
outside discovery. The source packager emits their old names only inside **new
authenticated archives**, so legacy HPC launchers' explicit selections/globs
remain available. Do not collect those aliases together with canonical tests.
Use the configured default roots rather than scanning an entire extracted tree.
The small collection hook imports the canonical module for an explicitly selected
legacy alias while retaining pytest's argument matching and import checks.
It leaves canonical collection unchanged and imports no Mosaic/model code.

The [nested layout map](../experiments/folder-layout.json) records each old name,
current implementation and compatibility source. The
[initial migration map](../experiments/MIGRATION.md) remains historical evidence.
Keep new tests under the correct subsystem or experiment, with a folder README.
Local AST/discovery/path checks are not numerical passes; the new source still
requires its allocated CPU/native regressions before scientific execution.

See the [nesting verification](../experiments/nested-layout-verification.json)
for local static, synthetic pytest and archive-authentication results.
