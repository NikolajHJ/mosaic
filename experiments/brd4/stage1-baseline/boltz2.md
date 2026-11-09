# Using Boltz2 for small-molecule design

Choose `model: boltz2` in the general YAML. See
[boltz2_general.yaml](boltz2_general.yaml) for the three Boltz2 paths. The
same runner can select Stage 0 or Stage 1. Choose a new `output_dir` for
each run.

The existing Boltz2 adapter minimizes the model's quantitative affinity
output (`affinity_pred_value`, lower is better). It evaluates each candidate
with the prepared pose and protein crop held fixed. This is a model estimate,
not a new docking pose or a measured affinity. It is a separate objective
from the Protenix-Tiny contact baseline in protocol v12. Binder probability
is logged for inspection and does not contribute to the loss.

- **Stage 0** starts from the prepared carbon chain unless you provide an
  `initialization.input` folder with a saved graph of the same size. Rows
  in that graph are assigned to the prepared atom slots in order. If you
  request a specific pocket, the preparation record must include a passed
  pose audit for those residues. Leaving out `pocket` uses the context as
  prepared. Leave `losses` out of the stage YAML for the model default, or
  use only `affinity` for Boltz2.
- **Stage 1** uses the same unconstrained graph updates and diagnostics as
  Protenix-Tiny, but scores affinity instead of contacts. The slot count in
  `configs/stage1.yaml` must match the context. The worker supports 6 or 25
  slots; the included BRD4 example uses 25. A positive
  `hard_minus_soft_loss` means the raw argmax molecule scored worse.

The Boltz2 worker needs a separate Python environment with the native
Boltz source at commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c553bc`, CUDA
PyTorch, and the matching cached affinity checkpoint. The adapter checks
these identities. Keep the ordinary Mosaic environment for the runner;
the runner selects CPU JAX for Boltz2 and the worker uses the allocated GPU.

A prepared context directory contains `context.pt` and `preparation.json`.
You can reuse a compatible context you already own. To prepare the BRD4
example, pre-stage the native Boltz2 model and molecule assets in your
Boltz cache, then run this inside an allocated DTU HPC GPU job from the
repository root (adjust the worker interpreter and cache paths):

```bash
PYTHONPATH=src envs/boltz2/bin/python -m mosaic.small_molecule.boltz2.prepare \
  --fixture tests/data/brd4_3mxf/fixture_25.json --n-atoms 25 \
  --cache cache/boltz --output results/boltz2-brd4-25-context
```

After preparation, run the optimization on an allocated DTU HPC GPU:

```bash
PYTHONPATH=src python -m mosaic.small_molecule.generate_molecule \
  --general-config experiments/brd4/stage1-baseline/boltz2_general.yaml \
  --stage-config configs/stage1.yaml
```

This is configuration and implementation guidance. The new integration has
not been tested with a native Boltz2 model on a GPU.
