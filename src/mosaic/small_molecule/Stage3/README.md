# Stage 3: DeFoG guidance

`mosaic-defog` adds a frozen DeFoG molecular prior to a differentiable graph
objective. It is an independently installable package: the core does not import
Mosaic, experiment runners, or upstream DeFoG modules. Choose the optimizer and
post-update projection separately.

This folder contains the Stage 3 package, examples, tests, and benchmark
evidence. Its public Python import is `mosaic_defog`.

| Path | Contents |
| --- | --- |
| `src/mosaic_defog/` | Guidance API, graph loss, frozen teacher worker, and Mosaic integration |
| [examples/manual_jax.py](examples/manual_jax.py) | A complete optimizer loop with a replaceable projection |
| [examples/stage2-defog.yaml](examples/stage2-defog.yaml) | DeFoG combined with the shared Stage 2 projection pipeline |
| `tests/` | Model-free guidance, integration, and worker tests |
| [BENCHMARKS.md](BENCHMARKS.md), [benchmarks.json](benchmarks.json) | Historical optimization results and their limits |
| [native-validation.json](native-validation.json), [wheel-validation.json](wheel-validation.json) | Recorded native extraction and interpreter-isolation checks |
| [pyproject.toml](pyproject.toml), [requirements-teacher.txt](requirements-teacher.txt) | Package metadata and separate teacher dependencies |
| [THIRD_PARTY.md](THIRD_PARTY.md), [LICENSE](LICENSE) | Attribution and license |

At each evaluated state, `Guidance.prepare` asks the teacher for one clean graph
prediction, then returns a loss with that target held fixed:

```text
native_loss(graph) + weight * (
    mean_atoms(sum_categories((X - teacher_X)**2))
  + mean_unordered_pairs(sum_categories((E - teacher_E)**2))
)
```

The student probabilities receive gradients; teacher predictions are detached.
This is a moving-target probability-MSE regularizer, not DeFoG's sampling loop.
See [benchmark results and limits](BENCHMARKS.md).

## Install and run without model assets

From the Mosaic repository root:

```bash
python -m pip install -e './src/mosaic/small_molecule/Stage3[jax]'
python src/mosaic/small_molecule/Stage3/examples/manual_jax.py --mock
```

The example runs ten CPU-compatible JAX updates with a synthetic objective and
deterministic mock teacher. Its `projection` function and update expression are
ordinary user code. Replace either without changing the package. The mock tests
the integration pattern; it provides no evidence about pretrained DeFoG.

For a Torch student, install `./src/mosaic/small_molecule/Stage3[torch]`. Importing
`mosaic_defog` loads neither framework nor weights; NumPy is the only required
runtime dependency.

## Prepare the real teacher once

Use a separate Python 3.11 environment for the official Torch implementation.
The student can use its own Python environment and JAX version. The teacher
worker runs on CPU with two threads by default and stays alive across queries.

```bash
git clone https://github.com/manuelmlmadeira/DeFoG.git /path/to/DeFoG
git -C /path/to/DeFoG checkout 365bda9affadd5c2307014a0532ddaa244399441
python3.11 -m venv /path/to/defog-env
/path/to/defog-env/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
/path/to/defog-env/bin/python -m pip install -r src/mosaic/small_molecule/Stage3/requirements-teacher.txt
```

On Windows the environment's interpreter is `Scripts/python.exe`. The pinned
direct requirements describe the existing Linux inference environment; they
are not a complete cross-platform lockfile. The recorded environment used a
CUDA-enabled Torch wheel on CPU. Installing the CPU wheel above is a convenient
alternative, whose exact numerical equivalence has not been established.

Obtain the official **GuacaMol marginal** checkpoint from the
[upstream checkpoint collection](https://drive.switch.ch/index.php/s/MG7y2EZoithAywE),
linked by the [official source](https://github.com/manuelmlmadeira/DeFoG).
The accepted default checkpoint has SHA-256:

```text
bc97d52ccf33a20d991f98349a1d37662273ab1c3e3f9777b938f9f144bc61f5
```

The package downloads nothing. Keep the upstream Git checkout clean and provide
explicit local paths. Only load a trusted checkpoint: the upstream Lightning
checkpoint loader uses Python deserialization. Check the real assets with one
25-atom query, then run the same example with them:

```bash
python -m mosaic_defog --source /path/to/DeFoG \
  --checkpoint /path/to/guacamol.ckpt --python /path/to/defog-env/bin/python
python src/mosaic/small_molecule/Stage3/examples/manual_jax.py --source /path/to/DeFoG \
  --checkpoint /path/to/guacamol.ckpt --teacher-python /path/to/defog-env/bin/python
```

## Add it to an optimizer

The existing graph object only needs `.X` and `.E_upper` arrays. The supplied
`Graph` named tuple is a JAX pytree. Define `native(graph, *, key=None)` to return
`(scalar_loss, auxiliary_dict)` and minimize it normally:

```python
import jax
from mosaic_defog import DeFoGTeacher, Guidance, cosine, linear

with DeFoGTeacher(source, checkpoint, python=teacher_python) as teacher:
    guidance = Guidance(
        teacher,
        weight=cosine(low=0.0, high=0.1, period=1500),
        time=linear(start=0.2, end=0.95, steps=n_steps),
    )
    for step in range(n_steps):
        loss = guidance.prepare(native, graph, step)  # host IO, outside JIT/grad
        (value, aux), gradients = jax.value_and_grad(loss, has_aux=True)(graph)
        graph = project(optimizer_update(graph, gradients))
```

`graph`, `native`, `project`, and `optimizer_update` come from the caller; the
[complete example](examples/manual_jax.py) supplies runnable versions. Call
`prepare` once per state you want to evaluate, then reuse its returned loss for
that state's gradient evaluations. A prepared loss keeps its original target.
`weight=0` still makes a teacher query, which preserves matched-query accounting.
For hooks with the signature `prepare_loss(native, epochs)`, pass
`guidance.loss_factory`; the returned callable takes `(graph, step)`.

For lower-level use, `target = teacher.predict(graph, time=0.8)` exposes
`target.jax_loss(graph)` and `target.torch_loss(graph)`. With Torch, pass
`backend="torch"` to `Guidance`, call `loss(graph)`, then backpropagate the returned
scalar through the live student tensors. The worker never receives student
autograd state.

The auxiliary dictionary reports `defog_atom_mse`, `defog_bond_mse`, `defog_loss`,
`defog_weight`, `defog_time`, and `defog_native_loss`. Native metrics must avoid
these names. Use the teacher as a context manager: successful exit rechecks its
state digest and query/forward counts before terminating the worker.

## Graph and teacher contract

| Array | Meaning |
| --- | --- |
| `X[N,K]` | Floating, normalized element probabilities for occupied atom slots |
| `E_upper[N*(N-1)//2,5]` | Floating, normalized bond probabilities in `np.triu_indices(N, 1)` order |
| Default element order | `C, N, O, S, F, Cl` |
| Bond order | `NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC` |

There is no vacancy category, atom reordering, conformer generation, or chemistry
conversion. The native teacher requires at least two occupied atoms. Its
12-element vocabulary is mapped explicitly to the requested element subset and
renormalized; prediction metadata reports excluded probability mass. Alternative
subsets can be supplied through `elements=...` in the same order as `X`.

The default `feature_policy="hard_aux_soft_core"` sends the current soft graph
to the denoiser and computes upstream auxiliary features on its argmax graph.
`native_soft` computes those features on the soft graph instead; treat it as a
different experimental choice. Neither policy repairs the graph. No RDKit
conversion occurs in the differentiable student loss.

Source revision, checkpoint digest, restored tensors, graph transport, shapes,
finite normalized probabilities, and one-forward accounting are checked at the
teacher boundary. Four checkpoint evaluation-only buffers are explicitly omitted
when restoring the inference model. The constructor accepts explicit alternative
identity expectations, but other checkpoint families are not supported by this
adapter. The target is stopped, so this does not differentiate through teacher
predictions or upstream feature computation.

## Optional Mosaic Stage 2 configuration

Install this package in the Mosaic runner's environment and add a `defog` block
to the normal Stage 2 YAML. See the
[complete example](examples/stage2-defog.yaml) and
[Stage 2 guide](../Stage2/README.md).

The YAML uses `stage: stage2` to reuse that pipeline and its selectable
projection operators; the `defog` block adds Stage 3 guidance. Run it from the
repository root with your general configuration:

```bash
PYTHONPATH=src python -m mosaic.small_molecule.generate_molecule \
  --general-config /path/to/general.yaml \
  --stage-config src/mosaic/small_molecule/Stage3/examples/stage2-defog.yaml
```

```yaml
defog:
  source: /path/to/DeFoG
  checkpoint: /path/to/guacamol.ckpt
  python: /path/to/defog-env/bin/python
  weight: 0.0625
  time: 0.8
  # weight: {kind: cosine, low: 0.0, high: 0.1, period: 1500}
  # time: {kind: linear, start: 0.2, end: 0.95}
```

Relative asset paths start at the Stage 2 YAML's directory. Linear schedules
default to the optimizer's update count, include both endpoints, and hold their
final value for the terminal readout. Cosine schedules start at their minimum.
The adapter composes with an existing preparation hook such as hardening.
In that combination, DeFoG still queries the optimizer's soft graph; the native
scorer may see the explicitly chosen hardened view. This changes the objective
and has not been benchmarked as a hallucination improvement here.

Keep `operators.mode` as its own experimental choice: `none`, `valence`,
`connected`, `sparse_only`, or `sparse`. DeFoG supplies a loss; those Stage 2
operators act after the update. Valence correction and component bridging do not
guarantee a sanitized, connected hard readout.

## Validation and provenance

Run the model-free package checks with
`python -m pytest src/mosaic/small_molecule/Stage3/tests`. The real-teacher CLI above requires the
official assets and a teacher interpreter and reports its model identity.

[BENCHMARKS.md](BENCHMARKS.md) records historical pipeline outcomes separately
from validation of this package. Source/checkpoint identity alone does not prove
reproduction of those trajectories. Package code is MIT licensed; upstream
attribution and the model-asset boundary are in [THIRD_PARTY.md](THIRD_PARTY.md).
