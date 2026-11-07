# Reusable package map

| Area | Owner |
| --- | --- |
| Graph tensors, schema, chemistry conversion and categorical sampling | `representation.py`, `chemistry.py`, `sampling.py` |
| AE/VAE architecture | [models/](models/) |
| Training losses, diagnostics, train/evaluate workflow | [training/](training/README.md); epoch trainer `train.py` |
| Dataset preparation/cache/split audit | [datasets/](datasets/README.md) |
| Reconstruction, generation and graded quality metrics | [evaluation/](evaluation/README.md) |
| Chemistry guidance, neighbor/valence/connectivity terms | [guidance/](guidance/README.md) |
| Latent updates and step-control primitives | [design/](design/README.md) |
| Boltz integration and existing native/embedding studies | [boltz/](boltz/) |
| Configurable frozen-model comparisons | [multimodel/](multimodel/) |
| DeFoG integration / repair-guidance studies | [defog/](defog/), [repair_guidance/](repair_guidance/) |
| Saved-result presentation | [reporting/](reporting/) |

The grouped modules contain the implementation, except the dataset data API
which forwards to the original file retained for cache provenance. The corresponding old flat
module names are small import aliases, preserving class/module identity,
monkeypatches and old `python -m` commands. New code should import the grouped
paths, for example `latentmol.guidance.connectivity`.

Existing campaign-specific flat modules have not been relocated underneath
queued jobs or immutable source manifests. Find them by question in the
[experiment index](../../experiments/README.md); do not copy them as the default
for a new study. Extend a reusable function or vary a config first.
