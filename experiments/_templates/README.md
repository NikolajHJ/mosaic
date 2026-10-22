# Experiment documentation templates

This folder contains documentation, not runnable experiments.

- Copy [experiment.md](experiment.md) to
  `experiments/<family>/<experiment-id>/README.md` before implementing a new
  experiment. Replace every placeholder; use `unknown` or `not run` honestly.
- Copy [run.md](run.md) to that experiment's
  `runs/<run-id>/README.md` before an execution attempt.
- Add a short README to each family, support and run-container directory you
  create, listing its role and children. Create only needed folders.

Follow the [organization policy](../README.md). Add dated events to the existing
[global ledger](../../docs/experiment_ledger.md), using the event skeleton in the
run template. These templates grant no execution permission.
