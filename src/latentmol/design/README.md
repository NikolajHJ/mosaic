# Design

Latent update functions, prior helpers and step-control policies.

Model adapters and full frozen-model comparison execution live in `boltz/` and `multimodel/`.

Modules: `latent.py`, `priors.py`, `step_control.py`.

Keep reusable calculations here. Experiment setup and results belong in the
[experiment homes](../../../experiments/README.md). Old flat imports forward
to these same modules; edit the implementation here.
