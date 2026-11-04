# Stage 2 — explicit molecular feasibility

Stage 2 of the working protocol: keep the Stage 1 objective and the
projected-probability optimizer exactly as they are, and add the minimum
chemistry needed to stop obvious failure modes without losing topology
fluidity.

Everything runs from inside this folder. It imports the framework
(`mosaic.small_molecule.*`) and owns none of it.

| File | What it is |
| --- | --- |
| `experiment.yaml` | The experiment definition: every knob, grouped by protocol section. |
| `settings.py` | Maps that YAML onto the CLI's own options. |
| `operators.py` | The chemistry operators and `Stage2Config`. NumPy + SciPy only, no JAX. |
| `backends.py` | The two frozen models: Boltz-2 affinity (default) and Protenix-Tiny contact. |
| `run.py` | One experiment arm. Needs an allocated GPU. |
| `experiments.py` | The studies as data: the matched comparison and the improvement sweep. |
| `diagnostics.py` | Reads a finished run and checks the pitfalls the plans flag. |
| `report.py` | The results view: existing trajectory renderer plus a feasibility comparison. |
| `test_stage2.py` | 55 local tests. No GPU, no JAX, no RDKit. |

## The loop

The protocol's default Stage 2 loop is

```text
AF3 gradient step  ->  simplex projection  ->  valence projection
```

`run_trajectory` already exposes a `post_step` hook that fires after each
accepted update and before the readout, so the whole of Stage 2 is that hook.
Nothing in the objective, the model wrapper or the optimizer changes.

Within one `post_step` the order is fixed: fluidity floor (if enabled), then
valence, then bridging and valence again, then shrinkage and valence again.
Valence is always last, so the state handed back is the feasible one.

## The five steps

- **5.1 Invariants** — not an operation. One bond vector per unordered pair
  (upper triangle only), mirrored so `E_ji` is never optimized separately,
  diagonal pinned to NO_BOND, every row on its simplex, `N` and the vocabulary
  fixed, charge zero. These are construction rules, not losses: the bad state
  is unrepresentable rather than penalised. `check_invariants` asserts them on
  entry and exit of every operator set.
- **5.2 Valence projection** — unconditional. Euclidean projection onto the
  edge simplexes intersected with the soft-valence halfspaces, solved through
  the `N`-dimensional nonnegative dual rather than a 1,500-variable primal QP
  at `N=25`. KKT is verified on every active call. It removes only excess and
  never pushes an atom up toward its ceiling.
- **5.3 Connectivity** — conditional, and the condition is met. The October
  valence-only runs left 23–25 components from 25 atom slots and zero of 501
  states both sanitized and single-component. Greedy highest-confidence
  inter-component SINGLE bridges, at most `c-1`, respecting hard valence
  capacity. No chain, ring or tree topology is prescribed. A bridge the
  following valence projection removes is counted, not hidden.
- **5.4 Sparsity** — conditional, and the condition is *not* met: valence
  projection already caps local bond order and nothing so far shows diffuse
  edge mass surviving it. Implemented and available as `--mode sparse` (with
  5.3) or `--mode sparse_only` (without it, which isolates shrinkage from
  bridging and is off the protocol's ladder). Treat a result from either as
  exploratory until the gate is demonstrated.
- **5.5 Hardening** — a separate run, never mixed into the baseline loop.
  `--hardening` ramps hardness 0 → 1, feeding the model
  `h*onehot + (1-h)*probabilities` through a straight-through estimator. The
  optimized state stays the underlying probabilities, and the one-hot graph is
  rebuilt each step and never stored.

  `--hardening-gradient` picks how the backward pass crosses that step, because
  the inherited transition has a side effect worth isolating. Protocol 5.5 says
  to reuse Mosaic's existing one-hot transition, which was written for *logits*
  while we feed it *probabilities*. At `soft=0, temp=1` its Jacobian is
  `h·J_softmax(p) + (1-h)·I`, so the gradient reaching `p` is attenuated by
  roughly `(1-h) + h/K` — about 6× for atoms and 5× for bonds at full hardness.
  Hardening therefore also decays the effective step size along the ramp, which
  confounds exactly the question 5.5 asks: if the hardened arm stops moving, is
  topology locked, or is the step just smaller?

  | Mode | Behaviour |
  | --- | --- |
  | `inherited` | The existing transition, attenuation included. Faithful to the earlier runs. Default. |
  | `passthrough` | Textbook identity-Jacobian STE. Removes the attenuation at its source. |
  | `rescaled` | Keeps the existing transition and scales the gradient by `1/((1-h) + h/K)`. The value is untouched, so the recorded loss stays the true objective. |

  `rescaled` is a scalar correction to a non-scalar Jacobian: it restores
  magnitude, not direction. `--study hardening` runs all three against a
  matched relaxed control, which is what separates the two effects.

## Backends

Boltz-2 is the default. Protenix-Tiny is kept so the earlier matched runs stay
reproducible, and because the difference between them is instructive.

| | Boltz-2 (default) | Protenix-Tiny (legacy) |
| --- | --- | --- |
| Objective | `affinity_pred_value` | mean max contact probability |
| Direction | lower is a stronger predicted binder | negated score, so lower loss is a higher contact score |
| **Bond order** | **differentiable** | **invisible** |
| Pose | fixed native predicted pose | geometry off, `ref_pos = 0` |
| Atom slots | 6 or 25 | any `N` |
| Runs on | any CUDA box with `--allow-local-gpu`, or an LSF job | any CUDA box |

The bond-order row decides what Stage 2 *means*.

Protenix-Tiny receives only bond existence, `B_ij = 1 - P(NO_BOND)`, so moving
probability between SINGLE, DOUBLE, TRIPLE and AROMATIC produced exactly zero
gradient. Soft valence is a function of bond order, so on that backend the
objective exerted no pressure at all on the quantity valence projection
constrains — Stage 2 was operating in the null space of the loss.

Boltz-2 does not work that way. It mixes the relaxed bond-type distribution
into its pretrained bond-type embedding,
`bond_types @ model.token_bonds_type.weight`, so **bond order is a live
gradient direction**. The valence projection and the objective now genuinely
interact: the objective pushes on bond orders, the projection pushes back. That
makes the aromatic-order choice matter to the gradient, makes
`--valence-damping` worth sweeping rather than merely available, and means
hardening now sharpens bond types instead of being a forward no-op.

`run.py` verifies this rather than assuming it. Every Boltz-2 run does a
finite-difference preflight (`boltz2/checks.py`) that requires nonzero atom
*and* typed-bond derivatives agreeing with analytic gradients at two adjacent
step sizes, and refuses to proceed otherwise. Disable with `--no-gradient-check`.

What has not changed: Stage 2 is still judged on feasibility, not on the
objective. Components, discrete valence violations, sanitized fraction, unique
valid SMILES and the relaxed-to-discrete gap. A better affinity number on a
disconnected or over-valent graph is not a Stage 2 result.

## Defining an experiment

`experiment.yaml` holds every setting, grouped by the protocol section it comes
from. It is the intended way to define a run:

```bash
python stage2/run.py --config stage2/experiment.yaml --out runs/my-arm
```

Precedence is flag, then file, then built-in default — so a sweep is one file
plus the single thing that varies:

```bash
python stage2/run.py --config stage2/experiment.yaml --out runs/damped \
  --valence-damping 0.5
```

The file's checked-in defaults are the protocol baseline: valence projection
only, every improvement neutral, no hardening. Whatever was used is copied into
the run's `stage2_config.json`, so a finished run always carries the experiment
that produced it.

A misspelled key is an error that names what was expected at that level, rather
than a setting that silently does nothing. A test asserts that every key in the
schema maps to a real option, so the file and the CLI cannot drift apart.

`experiments.py` takes the same `--config` for a whole study and forwards any
extra flag to every arm, so `--study hardening --steps 50` works without either
file declaring `steps` twice.

## Running it

Optimization needs an allocated GPU. Everything else runs on a laptop.

Boltz-2 needs a prepared fixed-pose context once. Native prediction establishes
the pose for an N-carbon chain and then it is frozen:

```bash
python -m mosaic.small_molecule.boltz2.prepare \
  --fixture FIXTURE.json --cache BOLTZ_CACHE --output contexts/brd4-25 \
  --n-atoms 25 --pocket POCKET.json --allow-local-gpu
```

Then one arm:

```bash
python stage2/run.py --mode valence --out runs/stage2-A \
  --boltz-context contexts/brd4-25 --boltz-checkpoint affinity.ckpt --allow-local-gpu
```

The matched comparison, protocol 5.6 experiment order, as one command:

```bash
python stage2/experiments.py --study matched --out runs/stage2-matched \
  --boltz-context contexts/brd4-25 --boltz-checkpoint affinity.ckpt --allow-local-gpu
```

### Running on a workstation GPU

Upstream, both the preparation and the worker refuse to start without an LSF
job id, because they were written for DTU HPC. `--allow-local-gpu` is a local
opt-in that accepts a workstation CUDA device instead. It changes nothing else:
the device is still required to be CUDA, and the checkpoint and Boltz commit
are still verified before the model loads.

VRAM is reported rather than assumed. The worker's ready line carries the
device name, its total VRAM and what loading the model alone cost; every reply
carries that request's peak; and the run summary records the worst peak seen as
`peak_vram_bytes` / `peak_vram_gib`. So the first run answers the capacity
question directly instead of us estimating it.

Preparation is the memory-heavy half — it runs full native diffusion sampling —
while the optimization itself is a frozen trunk with gradient checkpointing, no
kernels and `recycling_steps=0`, on an affinity crop of at most 256 tokens and
2048 atoms. If something runs out of memory, expect it to be the preparation
step, and note that preparation is a one-off whose output is reusable.

The same on the legacy backend, continuing a sealed Stage 1 parent:

```bash
python stage2/experiments.py --study matched --backend protenix --target brd4 \
  --out runs/stage2-protenix \
  --parent /opt/mosaic-stage0/runs/stage1-long-20261017/brd4-canonical-seed17
```

Add `--print` to get the individual `run.py` commands instead of running them,
`--only A-valence B-connected` to run a subset, and `--compare-only` to rebuild
the comparison from arms that already finished. Finished arms are skipped, so
re-running the command resumes a study.

The tests need neither a GPU nor the framework:

```bash
uv run --no-project --python 3.12 --with numpy --with scipy --with pytest \
  python -m pytest stage2/test_stage2.py -q
```

Pitfall verdicts for a finished run or study:

```bash
python stage2/diagnostics.py runs/stage2-matched/A-valence
```

The results view:

```bash
.venv-report/bin/python stage2/report.py runs/stage2-matched --open
```

`report.py` writes `feasibility.md`, `diagnostics.json` and eight PNG/SVG panels,
then hands the saved runs to `scripts/render_small_molecule_report.py` for the
interactive `report.html` — the argmax molecule over time, the element
probability matrix, the per-atom probability curves and the sampling panels.
Pass `--skip-trajectories` for the feasibility half alone, which needs only
NumPy and matplotlib. The renderer half needs the report environment:

```bash
.venv-report/bin/python -m pip install -r scripts/requirements-report.txt
```

## Improvements

Five, each individually toggleable and every one neutral at its default, so
`Stage2Config(mode="valence")` is exactly protocol 5.2 with nothing added. Each
arm records which were active, and the sweep turns them on one at a time so
every effect is attributable.

| Flag | What it does | Why |
| --- | --- | --- |
| `--readout-caps` | Also require the argmax element's own ceiling | A state can satisfy the soft ceiling while its readout is over-valent; FKBP12 showed carbon at valence 5 from step 302 |
| `--valence-damping A` | Step a fraction `A` toward the exact projection | An exact projection every step may freeze topology changes (protocol 5.2 asks this to be monitored). Leaves a residual excess, reported as `damped_excess` |
| `--fluidity-threshold T --fluidity-strength S` | Rows above `T` are mixed back toward uniform by `S` | Protocol 4.3 C.10, premature collapse. Narrowest well-defined version: a projection-space nudge on collapsed rows only, not an entropy loss |
| `--bridge-retry` | One stronger re-bridge when valence removed a bridge | Bridging and valence projection can fight each other. Bounded to a single attempt, kept only if more bridges survive |
| `--aromatic-order 1.0` | Looser coordination bound for aromatic bonds | 1.5 is the protocol surrogate, not a complete chemical model |

```bash
python stage2/experiments.py --study improvements --target brd4 --out runs/stage2-improve \
  --parent /opt/mosaic-stage0/runs/stage1-long-20261017/brd4-canonical-seed17
python stage2/experiments.py --list   # the arms and the reason for each
```

## Pitfall checks

`diagnostics.py` turns a finished run into one verdict per documented pitfall,
each carrying its number. Ten checks, sourced to the working protocol
(sections 4.3 C and 5.2–5.4) and to the submitted plan's evaluation
requirements:

`fluidity`, `topology`, `valence`, `connectivity`, `sparsity`,
`soft_hard_gap`, `projection_aggressiveness`, `bridge_survival`,
`soft_versus_hard_feasibility`, `predictor_exploitation`.

A verdict is `ok`, `flag` or `n/a`. `flag` means the failure mode the plans told
us to watch for is visibly present — information about the run, not a defect in
the code. `n/a` means the artifact needed is absent, which is never reported as
a pass. The two that matter most for honesty are
`soft_versus_hard_feasibility`, which keeps relaxed constraint satisfaction
separate from discrete molecular validity as the plan requires, and
`predictor_exploitation`, which flags a score that improves while the chemistry
does not.

Each check is exercised against a trajectory built to fail in exactly that one
way, so the checks themselves are tested rather than merely written.

## Boundaries

No RDKit edits inside the optimization loop. No graph repair. Hard molecules
are readouts and evaluation states, never substituted back into optimization.
Charge stays at zero, `N` and the vocabulary stay fixed. The structure model
stays frozen; gradients reach only `X` and `E_upper`. A continuation from a
sealed parent reproduces that parent's final recorded loss before anything is
optimized, and stops if it cannot.

Two Boltz-2 specific limits worth stating when writing this up. The pose is
fixed: its own source calls this *"a partial chemistry derivative, not
differentiation through pose prediction"*, so gradients reach atom and bond
features, not the structure. And only the quantitative affinity is
differentiable — the binary binder metrics are explicitly stop-gradient and are
logging only.

`future/` lists what is deliberately not implemented here and why.

## Vendored files

`src/mosaic/small_molecule/boltz2/` and `pocket.py` are taken from
`NikolajHJ/mosaic`, not merged. His `losses.py` and `io.py` have diverged from
ours and are what Stage 1 and Stage 2 run on, so a merge would be disruptive;
`boltz2/` needs only `evaluate_graph` and `RelaxedGraph`, both identical across
branches, plus stdlib-only `pocket.py`.

Three of those files carry local edits, each marked `LOCAL` in the source:
`worker.py` and `prepare.py` gain `--allow-local-gpu` and no longer require an
LSF job id, and `worker.py` plus `bridge.py` report peak VRAM. `checks.py`,
`native.py` and `__init__.py` are untouched. His exploratory `dbmol.py` was not
taken. Keeping the edits small and labelled is deliberate: it keeps the eventual
manual merge readable.

## Evidence

`evidence/` holds what a reader needs to check a claim about a run, and nothing
else. Per arm: the `stage2_config.json` it ran under and its `run_summary.json`.
Per study: `comparison.json`, `study.json` with the exact command line for every
arm, and `report/diagnostics.json` with the pitfall verdicts. The per-step series
behind those verdicts is replaced by its row count, and `per_step_rows` records
how many rows it had.

Trajectories, checkpoints and evaluation arrays stay out: one 500-step arm is
roughly 25 MB of `.npz`, and a batch is a few hundred megabytes. Rerunning an arm
from its recorded command line reproduces them.

Three batches are here: `overnight-20261103` is the protocol's own experiment
order 5.6 plus the hardening and improvement studies, and `round2-20261104` and
`round3-20261104` are hyperparameter follow-ups on the best arm.
