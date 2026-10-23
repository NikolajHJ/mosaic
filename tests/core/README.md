# Core package tests

Cache location/override contracts and shared protein-design optimization behavior.

Use default pytest discovery or select this directory in an allocated DTU HPC
job. Local layout verification parses source only and is not a numerical pass.
Shared fixtures remain under [tests/data](../data/).

## Files

- [test_cache.py](test_cache.py)
- [test_optimizers.py](test_optimizers.py)
- [fixture_premodel_startup.py](fixture_premodel_startup.py): allocated shared startup
  controls for the Boltz comparison and DBMol pilot. The separate real callback
  smoke executes each controller-generated driver command before native dispatch.
  This allocation-only fixture is packaged as `tests/test_premodel_startup.py`;
  it is excluded from ordinary repository discovery because it requires LSF.

See the [test index](../README.md), [experiment index](../../experiments/INDEX.md)
and [nested layout map](../../experiments/folder-layout.json).
