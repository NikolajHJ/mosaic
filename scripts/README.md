# Commands

Start with the [workflow guide](../docs/workflows.md) and
[experiment index](../experiments/README.md). Reuse the package CLI for ordinary
training/evaluation/comparisons; scripts should add only necessary integration.

| Task | Entry point |
| --- | --- |
| Plan/run a configurable frozen-model comparison | `python -m latentmol.multimodel` |
| Train and evaluate | `python -m latentmol.training.workflow` |
| Evaluate a saved checkpoint | `python -m latentmol.evaluation.evaluate` |
| Render a saved optimization report | `render_optimization_report.py` |
| Bounded local neighbor comparison | `local_neighbor_pilot.py` (explicit local inputs/GPU) |
| Cluster setup and existing campaign launchers | [hpc/](hpc/) |
| Existing learned-embedding campaign controllers | `embedding_*_campaign.py`, linked from the latent-design home |

The existing campaign controllers remain compatible and at their recorded paths.
For a new arm/retry, use a config and unique output directory before adding a
new controller. Never execute a saved receipt as a launcher. Generated archives,
logs, checkpoints and reports belong in ignored outputs, not alongside code.
