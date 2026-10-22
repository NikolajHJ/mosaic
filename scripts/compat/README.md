# Compatibility source inputs

These labeled adapters keep legacy names available without filling the working
`scripts/` and `tests/` roots. Implementations belong to the documented shared
capability or experiment. Edit the canonical implementation, not an adapter.

- [python/](python/README.md): former flat script entry points.
- [tests/](tests/README.md): test adapters for old explicit paths and shell globs.
- [assets/](assets/README.md): byte-identical legacy template copies.

The [nested layout map](../../experiments/folder-layout.json) records every
location. The source packager includes these stored files and adds their declared
old names **inside a new archive only**, authenticating both in its manifest.
No old path is created in the checkout and no existing release is rewritten.
Existing sealed/running sources and receipt paths retain their original bytes.

Use canonical commands from the experiment README. For an old Python name:

```bash
python scripts/layout/experiment_layout.py resolve scripts/package_stage0_release.py
python scripts/layout/experiment_layout.py run scripts/package_stage0_release.py --help
```

Legacy LSF scripts still require a complete authenticated release. Their explicit
old test selections work in the newly packaged release; they are not standalone
checkout commands. Do not collect compatibility tests with canonical tests.
