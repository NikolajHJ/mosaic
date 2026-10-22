# Shared releases tools

Source archive packaging and authentication. Used by every experiment; include experiments/, canonical scripts and legacy adapter inputs in every future release.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [package_stage0_release.py](package_stage0_release.py) | Seal current local source bytes without importing Mosaic or running tests. | `scripts/package_stage0_release.py` |
| [verify_stage0_release.py](verify_stage0_release.py) | Authenticate an extracted source release without importing Mosaic. | `scripts/verify_stage0_release.py` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.

Former flat script/test adapters are stored under [scripts/compat/](../compat/README.md).
The packager adds their declared old names only inside a **new archive**, includes
their actual bytes in `source_manifest.json`, and records their source locations.
It never populates the working roots or changes a prior release. Default pytest
discovery selects nested canonical roots, avoiding duplicate alias collection.
