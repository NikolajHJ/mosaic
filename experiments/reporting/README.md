# reporting experiments

Scientific family index. Each study retains its own question, protocol, configuration and attempt history.

**Default for saved trajectory reports:** use
[`python scripts/reporting/report.py`](../../scripts/reporting/README.md) with
run paths/names or `--runs my-runs.json`. The
[reusable JQ1 reporting engine](jq1-live-progress/README.md) owns the implementation,
validated cache and update/publication workflow. Extend that owner when needed;
ordinary run-list/data updates do not require another renderer or release.
See its documented schema/profile limits before choosing a historical specialist
pipeline below.

- [Saved molecular-design milestones report](milestones/README.md): Assemble the saved six-milestone evidence without model evaluation, resampling, molecule repair or reinterpretation of failed scientific checks.
- [Saved BRD4 comparison report](brd4-comparison/README.md): Combine explicitly declared saved BRD4 studies with honest partial/failed/pending states and fixed-versus-changing loss distinctions.
- [Updating JQ1 progress report](jq1-live-progress/README.md): Default reusable reports for supported Protenix, Boltz, DBMol, noisy and chemistry trajectories; all scalar records, retained molecule matrices, immutable editions and a stable latest viewer.

See the [global experiment index](../INDEX.md) and [agent logbook](../../docs/experiment_ledger.md).
