# Experiment layout infrastructure

This folder owns repository organization tools. It performs no numerical work.

- `experiment_layout.py` locates the repository, resolves migrated working files,
  enables the established unique script-module names and implements old-path
  compatibility entry points. It reads the preserved
  [initial migration map](../../experiments/migration-map.json) and additive
  [nested layout map](../../experiments/folder-layout.json).
- `verify_layout.py` checks AST syntax, canonical hashes, compatibility input
  equality, source-path resolution, folder/link coverage and Bash syntax. Its
  synthetic module check uses only stdlib code, never experiment/model imports.
- `audit_documentation.py` reads documentation and supported attachments,
  checks local references and reports concrete account paths. Reusable guides
  fail on personal account defaults; dated receipts and imported snapshots are
  reported separately. This does not verify remote ACLs or scientific outcomes.
- `verify_pytest_routing.py --output results/static/pytest-routing.json` exercises
  real pytest against two synthetic stdlib modules, checking default discovery,
  explicit legacy selections, cross-folder imports and monkeypatch/source identity.
  It imports no Mosaic/model code and preserves its synthetic sources/logs beside
  the output receipt. It requires pytest in the chosen interpreter.
- Canonical script modules use this runtime before their ordinary imports.
  Nested compatibility Python entry points execute canonical code in their own module
  namespace, preserving function globals and monkeypatch behavior.
- Existing external sealed-source paths fall back to their historical locations
  when canonical files do not exist in that release. The runtime never rewrites
  an archive or downloads, validates or launches an experiment.

Source releases must include this folder, the map, canonical implementations and
any retained compatibility inputs. The packager adds authenticated old flat
names to a new archive only; it never recreates loose files in the checkout.
`release_aliases()` validates the declared aliases and rejects missing inputs,
escaping paths and conflicting existing bytes. Packaging only forwarding files is
incomplete and fails explicitly. See the [policy](../../experiments/README.md).

From the repository root, run `py -3.13 scripts/layout/verify_layout.py --root .`.
Migration-time `--baseline` additionally compares protected files and scientific
numeric literals/signatures/guard messages to a pre-edit snapshot. A specifically
named live operator log can be reported as concurrently changed; that never
becomes an unchanged-byte claim or a scientific validation pass.

Initial hashes in the migration map are audit history. Normal layout checks
report later canonical source changes; `--baseline` requires the exact recorded
migration snapshot. Byte equality of declarative compatibility copies is always
required. Future authorized edits still use fresh ordinary source manifests.

To translate an old working command, use `python scripts/layout/experiment_layout.py
resolve OLD_PATH`, or execute an old Python name with
`python scripts/layout/experiment_layout.py run OLD_PYTHON_PATH [ARGS ...]`.
Shared test bootstraps opt into unique test-helper directories; pytest's explicit
canonical discovery roots exclude compatibility adapters, including archive aliases.
`test-paths PATTERN` prints the equivalent canonical selection without
importing tests. The checker validates all 82 discovery paths, helper resolution
and the absence of new loose implementation files in either root.
