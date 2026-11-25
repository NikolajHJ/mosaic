# Current workflow commands

Run from the repository root in an environment with `latentmol` installed.
Use existing configs and unique output directories; edit a small config for a
new arm instead of copying a driver. These commands do not resume an existing
campaign. Substantive model runs belong on appropriate allocated compute.

## Data and AE/VAE training

```sh
python -m latentmol.datasets.data --input INPUT.smi --output data/my-cache --limit 5000
python -m latentmol.training.workflow --config configs/ae_smoke.yaml --data data/my-cache --run-root outputs/runs/vae --smoke
```

`--smoke` evaluates the explicit memorization subset as well as best/last
validation checkpoints. Use a config whose `train_limit` is set. See the
[training question](../experiments/vae/README.md) for the larger existing studies.
The existing training loop remains `latentmol.train`; losses, diagnostics and
the composed train/evaluate workflow live under `latentmol.training`.

For the 435,400-parameter VAE, the [compact training and timing guide](../experiments/vae/compact-benchmark.md)
describes the graph-matched reconstruction loss and the 256/512 V100 benchmark.
The [matched bond-message experiment](../experiments/vae/bond-message-comparison.md) trains two compact decoders and compares their latent edits with fresh Protenix scores.

For sharded preparation of a large external SMILES corpus, see the
[PubChem workflow](../experiments/vae/large-corpus.md).

For canonical hard-graph input, set `encoder_pooling: canonical` under `model`
in the training YAML. Both prepared cache formats store canonical atom order.
The trainer checks their provenance, RDKit version and up to 16 graphs per
split, then uses cached labels directly for training and its validation
passes. Checkpoints record the encoder mode and canonicalize arbitrary atom
orders when encoding supported hard graphs. The shuffled-input regression is
in [test_ordering.py](../tests/datasets/test_ordering.py). Exact continuation
binds the original executable source and model config, so adopting this mode
from an older checkpoint requires a separate training branch.

## Saved checkpoint evaluation

```sh
python -m latentmol.evaluation.evaluate --checkpoint CHECKPOINT.pt --data data/my-cache --run-root outputs/runs/evaluation --split val --device cpu
```

Use the appropriate device and limits for the selected checkpoint/panel. This
creates a new evaluation record; it does not edit the input checkpoint.

For a matched multi-checkpoint evaluation with shared fresh prior vectors,
held-out molecules and latent perturbations:

```sh
python -m latentmol.evaluation.checkpoint_comparison --config MY_COMPARISON.json --output NEW_OUTPUT --device cuda
```

The [epochs 43–45 comparison](../experiments/vae/checkpoint-comparison.md)
documents panel sizes, sampling, uncertainty and raw-output locations. This
workflow evaluates frozen checkpoints and never starts VAE training.

For similar-molecule interpolation with explicit atom maps and an encoder-order
control, use [the epoch-45 protocol](../experiments/vae/interpolation.md):

```sh
python -m latentmol.evaluation.interpolation --config MY_INTERPOLATION.json --output NEW_OUTPUT --device cuda
```

The [bounded invariant-encoder pilot](../experiments/vae/invariant-pilot.md)
creates a separate model with matching-aware reconstruction, then reuses the
shared checkpoint comparison plus atom-order evaluation:

```sh
python -m latentmol.training.invariant_pilot --config MY_PILOT.json --output NEW_OUTPUT --device cuda
```

This explicit pilot trains a new branch; it does not resume the parent VAE.

The [atom-mapped interpolation training pilot](../experiments/vae/interpolation.md)
compares the control and interpolation loss in one allocation. Use
`python -m latentmol.training.interpolation_pilot --config experiments/vae/epoch45-interpolation-pilot.json --output NEW_OUTPUT --device cuda`.

## Frozen-model latent design

```sh
python -m latentmol.multimodel plan --config configs/multimodel_boltz.yaml
python -m latentmol.multimodel pipeline --config MY_PREPARED_CONFIG.yaml --research-profile exploratory --run-root outputs/runs/design
```

`plan` needs no native model. `pipeline` needs the actual decoder, model assets
and prepared context referenced by the config. Example asset paths/hashes must
be filled in first. `--research-profile qualified` requests the detailed gates.
The [latent-design home](../experiments/latent-design/README.md) distinguishes
this reusable entry point from historical continuation controllers.

An arm using `occupancy: variable` can set an optional atom budget. For a
matched comparison, give two arms the same model and set `atom_budget` on
only one arm.

```yaml
occupancy: variable
arms:
  baseline:
    models: [binding]
  budget:
    models: [binding]
    atom_budget:
      target: 25
      tolerance: 2
      weight: 0.01
      outside_weight: 0.002
```

For expected heavy-atom count \(N=\sum_i [1-P_i(\text{NO\_ATOM})]\), the count
term is \(0.01\max(0, |N-25|-2)^2\). The optional outside term charges 0.002
per unit of occupancy outside the 25 highest-occupancy slots at the start.
Those slots remain fixed as a reference. Use `outside_weight: 0` to isolate
size pressure and let atoms swap slots without that cost. These weights are
initial settings to compare against the binding loss scale. The metrics record
both loss terms and the expected count, and candidates retain the hard count.
The chosen scorer must already support variable occupancy.

The new [VAE/Protenix experiment](../experiments/latent-design/protenix-vae/README.md)
injects a native scorer into the shared-graph latent loop. On an allocated GPU
with the prepared Mosaic context and native environment, its direct command is:

```sh
python -m latentmol.design.latent_protenix --config experiments/latent-design/protenix-vae/pilot.json --checkpoint COMPLETED_VAE.pt --mosaic-python /path/to/mosaic-native/python --mosaic-config NATIVE_ASSETS.json --device cpu --output NEW_OUTPUT
```

The Torch decoder runs on CPU; the resident JAX child owns the allocated GPU.
This does not retrain either model. The experiment's queued DTU controller
resolves epoch45 only after its existing checks and handles conditional50k
segments; do not replay its recorded submission as a generic example.

## Import a direct relaxed graph from Mosaic

The fixed-size, neutral Mosaic generator returns `x[batch, n, 6]` and `e[batch, n(n-1)/2, 5]`. Convert them into the same `RelaxedGraph` used by DBMol scorers, with the checkpoint's actual `GraphSpec`. Pass `active_slots` when the native atom order maps to nonconsecutive parent slots.

```python
from latentmol.graph_policy import graph_from_mosaic

graph = graph_from_mosaic(x, e, spec, active_slots=native_to_parent_slots)
```

The conversion maps atom and bond vocabularies by identity, keeps gradients to both inputs and fills unused slots with absent atoms, neutral charge and no bonds. [Direct generator and frozen example](../../mosaic/experiments/brd4-jq1/protenix-random-chem/aromatic-formation/local-prior/continuous-generation.md#use-the-relaxed-generator)

## Reports and tests

For a bounded native coordinate prediction from a saved soft graph, see the
[Boltz-2 soft-diffusion experiment](../experiments/latent-design/soft-diffusion.md).
This inference-only probe runs on an allocated GPU, outside existing campaigns.

```sh
python scripts/render_optimization_report.py --help
python -m pytest tests/training/test_models.py tests/datasets/test_chemistry.py -q
```

See [tests](../tests/README.md), [shared code](../src/latentmol/README.md) and the
[experiment index](../experiments/README.md). Old import/CLI paths still forward
to the same modules, including monkeypatch state; new work uses the grouped paths.
No extra qualification campaign is required just to view saved results or run a
small fixture test.
