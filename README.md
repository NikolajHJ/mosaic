# DBMol exploratory research

Molecular graph AE/VAE training and latent design through frozen molecular
scorers. We are exploring which representations and chemistry guidance improve
reconstruction, generation and useful molecular optimization.

Start with a small bounded experiment. Finite-gradient smoke checks are the
exploratory default; detailed qualification is explicit. See the
[research policy](docs/EXPLORATORY_RESEARCH_POLICY.md).

| I want to… | Start here |
| --- | --- |
| Find the current questions and latest saved results | [Experiment index](experiments/README.md) |
| Prepare data, train, evaluate or plan a model comparison | [Workflow commands](docs/workflows.md) |
| Find reusable code | [Package map](src/latentmol/README.md) |
| Run a small relevant test selection | [Tests](tests/README.md) |
| Use the cluster | [Current HPC connection](docs/HPC_CONNECTION.md) |
| Understand a past result or failure | [Documentation/history index](docs/README.md) |

From an environment with this checkout installed (`python -m pip install -e '.[test]'`):

```sh
python -m latentmol.training.workflow --help
python -m latentmol.multimodel plan --config configs/multimodel_boltz.yaml
python -m pytest tests/training/test_models.py -q
```

The plan command loads no model and starts no optimization. Native execution
needs actual prepared assets; the example YAML contains placeholders. Substantive
model runs use the appropriate allocated compute environment.

```text
src/latentmol/    reusable graphs, models, data, training, guidance and design
experiments/     questions, related configs/commands and result summaries
configs/         existing configurations, reused by experiment homes
scripts/         thin commands and cluster integration
tests/          grouped core tests plus existing campaign regressions
docs/           current usage and links to dated scientific records
outputs/        generated runs, reports and retained local evidence
```

Existing import/CLI paths remain compatible; use the grouped package paths for
new work. Historical campaign modules remain where their launchers expect them.
Running jobs and sealed releases use their original code and paths.

Contributor rules are in [AGENTS.md](AGENTS.md). Use one implementation, vary
configs for related arms, and keep a short question/command/result note. Earlier
README material is retained in the [dated reference](docs/history/README-20261030-before-organization.md).
