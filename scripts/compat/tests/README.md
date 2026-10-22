# Legacy explicit test-path adapters

These adapters are source inputs for old test-path selections and shell globs
inside new authenticated releases. They are outside pytest's discovery roots.
Each file names its single canonical test implementation under `tests/<topic>/`
or its owning experiment. No test body is duplicated here.

Default pytest discovery collects the nested shared tests and canonical experiment
tests once. Do not add this directory to test discovery. See the
[test index](../../../tests/README.md), [compatibility contract](../README.md)
and [old-to-new map](../../../experiments/folder-layout.json).
