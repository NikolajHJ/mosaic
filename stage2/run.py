"""Run one Stage 2 arm: the frozen structure model plus chemistry operators.

The objective, the model and the projected-probability optimizer are unchanged
from Stage 1. This script only selects a backend, chooses the starting graph,
attaches one ``post_step`` operator set, optionally enables forward hardening,
and writes the artifacts the diagnostics and the report read.

Boltz-2 is the default backend and needs an allocated DTU HPC job plus a
prepared fixed-pose context; Protenix-Tiny is kept for reproducing the earlier
matched runs. See :mod:`backends` for what differs between them, including the
one difference that changes Stage 2's meaning: Boltz-2 differentiates bond
order, Protenix-Tiny does not.

``operators.py`` and ``diagnostics.py`` deliberately need no model, so the
operators and their tests stay runnable anywhere. The framework imports sit
inside the functions that use them for the same reason: building the parser,
listing a study and printing its commands all work without JAX installed.

Experiment order, following protocol 5.6:

    A  --mode valence                     valence projection only
    B  --mode connected                   add component bridging
    C  --mode sparse                      add weak-edge shrinkage
    D  --parent P --hardening             against a matched relaxed arm
       plus --mode none                   the frozen Stage 1 control
"""

import argparse
from functools import partial
import json
import os
import time
from pathlib import Path

import numpy as np
from backends import METRICS, build
from operators import Stage2Config, post_step
from settings import DEFAULT, apply

ROOT = Path(__file__).resolve().parents[1]


HARDENING_GRADIENTS = ("inherited", "passthrough", "rescaled")


def attenuation(hardness, categories):
    """Scalar estimate of how much the inherited transform shrinks a gradient.

    The exact Jacobian of Mosaic's transition at ``soft=0, temp=1`` is
    ``h*J_softmax(p) + (1-h)*I``, because the term that survives
    ``stop_gradient`` is a softmax of ``p`` rather than ``p`` itself. For rows
    that are not already near one-hot, ``J_softmax = diag(s) - s s^T`` has
    eigenvalues around ``1/K`` on the simplex tangent, so the magnitude
    reaching ``p`` is about ``(1-h) + h/K``.

    This is the scalar surrogate ``rescaled`` inverts. It restores magnitude,
    not direction: the true Jacobian is not a multiple of the identity.
    """
    return (1.0 - hardness) + hardness / categories


def hardening_factory(native, total_steps, gradient="inherited"):
    """Forward hardening as a separate experiment, protocol 5.5.

    The forward representation is ``h*onehot + (1-h)*probabilities`` through a
    straight-through estimator, with ``h`` ramping linearly from 0 to 1. The
    optimized state stays the underlying probabilities, so the operators and
    the readout are unchanged, and the one-hot graph is rebuilt from ``p`` each
    step and never stored.

    ``gradient`` selects how the backward pass crosses that step; see
    ``experiment.yaml`` for what each mode is for. ``inherited`` reuses
    Mosaic's existing transition as the protocol asks, and carries its gradient
    attenuation; ``passthrough`` is the textbook identity-Jacobian estimator;
    ``rescaled`` keeps the inherited transform and compensates its magnitude.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import _colabdesign_transform
    from mosaic.small_molecule.state import RelaxedGraph

    if gradient not in HARDENING_GRADIENTS:
        raise ValueError(f"hardening gradient must be one of {HARDENING_GRADIENTS}")

    def straight_through(rows, hardness):
        """Identity on the value, identity on the gradient."""
        onehot = jax.nn.one_hot(rows.argmax(-1), rows.shape[-1], dtype=rows.dtype)
        return rows + hardness * jax.lax.stop_gradient(onehot - rows)

    def scale_gradient(rows, factor):
        """Identity on the value, ``factor`` on the gradient."""
        return rows * factor + jax.lax.stop_gradient(rows * (1.0 - factor))

    def harden(rows, hardness):
        if gradient == "passthrough":
            return straight_through(rows, hardness)
        if gradient == "rescaled":
            rows = scale_gradient(rows, 1.0 / attenuation(hardness, rows.shape[-1]))
        return _colabdesign_transform(rows, 0.0, 1.0, hardness)

    class HardenedLoss(eqx.Module):
        native: object
        hardness: jax.Array

        @eqx.filter_jit
        def __call__(self, graph, *, key):
            forward = RelaxedGraph(
                harden(graph.X, self.hardness),
                harden(graph.E_upper, self.hardness),
            )
            value, aux = self.native(forward, key=key)
            return value, {
                **aux,
                "forward_hardness": jnp.asarray(self.hardness),
                "hardening_gradient": gradient,
            }

    def prepare(graph, step):
        hardness = min(step / max(1, total_steps - 1), 1.0)
        return HardenedLoss(native, jnp.asarray(hardness))

    return prepare


def gradient_preflight(loss, graph, key, directory):
    """Confirm atom *and* bond-type derivatives against finite differences.

    Reuses ``boltz2/checks.py``, which also asserts nonzero typed-bond
    contrast. That assertion is the whole premise of Stage 2 on this backend:
    if bond order were a null direction the valence projection would be acting
    on something the objective cannot see.
    """
    from mosaic.small_molecule.boltz2.checks import check_affinity_gradients

    return check_affinity_gradients(loss, graph, key, directory)


def check_settings(args):
    """Validate the experiment before importing anything heavy.

    Deliberately ahead of the framework imports, so a missing setting is a
    clear message on any machine rather than a ``ModuleNotFoundError`` for JAX.
    Returns the operator configuration, which validates itself.
    """
    if args.backend == "boltz2" and not (args.boltz_context and args.boltz_checkpoint):
        raise SystemExit(
            "backend boltz2 needs a prepared context and a checkpoint. Set\n"
            "  boltz2.context / boltz2.checkpoint in the experiment file, or\n"
            "  --boltz-context DIR --boltz-checkpoint CKPT\n"
            "Prepare a context with\n"
            "  python -m mosaic.small_molecule.boltz2.prepare --fixture F "
            "--cache C --output DIR --n-atoms 25"
        )
    return Stage2Config(
        mode=args.mode,
        aromatic_order=args.aromatic_order,
        bridge_probability=args.bridge_probability,
        sparsity_amount=args.sparsity_amount,
        sparsity_threshold=args.sparsity_threshold,
        readout_caps=args.readout_caps,
        valence_damping=args.valence_damping,
        fluidity_threshold=args.fluidity_threshold,
        fluidity_strength=args.fluidity_strength,
        bridge_retry=args.bridge_retry,
    )


def require_callback_platform():
    """Ensure JAX has a CPU device, which the Boltz bridge needs.

    The bridge crosses into the torch worker through ``jax.pure_callback``, and
    that stages its inputs on a local CPU device. The existing WSL launchers
    export ``JAX_PLATFORMS=cuda``, which is right for the pure-JAX Protenix
    path but makes every Boltz evaluation fail with "failed to find a local CPU
    device". Appending ``cpu`` keeps CUDA first for the optimizer while giving
    the callback somewhere to land.

    Called before the first JAX import, since the variable is read at
    initialization and ignored afterwards.
    """
    platforms = os.environ.get("JAX_PLATFORMS")
    if platforms and "cpu" not in [p.strip() for p in platforms.split(",")]:
        os.environ["JAX_PLATFORMS"] = f"{platforms},cpu"
        return os.environ["JAX_PLATFORMS"]
    return platforms


def run(args, *, prepare_loss=None, prior_identity=None):
    config = check_settings(args)
    if args.backend == "boltz2":
        require_callback_platform()

    import jax

    from mosaic.small_molecule.io import (
        TrajectoryConfig,
        file_sha256,
        run_trajectory,
        save_graph,
        write_json,
    )
    from mosaic.small_molecule.optimizers import evaluate_graph
    from mosaic.small_molecule.provenance import environment_identity, source_identity

    environment = environment_identity()
    if args.backend != "boltz2" and not args.allow_cpu and not any(
        device["platform"] == "gpu" for device in environment["devices"]
    ):
        raise SystemExit(
            "no GPU visible. Stage 2 runs on an allocated GPU; use --allow-cpu "
            "only for a deliberate smoke test."
        )
    if args.hardening and prepare_loss is not None:
        raise ValueError("Compare the prior on the unhardened Stage 2 path")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)

    native, graph, provenance, finish = build(args)
    cleanup_started = False
    try:
        # Same derivation as the Stage 1 runner: its first split half seeds the
        # initialization and its second is the evaluation key. A parent
        # continuation only reproduces the parent's recorded loss if the
        # evaluation key matches, so this split must not be simplified away.
        # Boltz-2 discards the key entirely (fixed pose, no sampling).
        _, key = jax.random.split(jax.random.key(args.seed))

        if "parent_final_loss" in provenance:
            check = evaluate_graph(native, graph, key)
            difference = abs(float(check.value) - provenance["parent_final_loss"])
            if difference > 2e-6:
                raise ValueError(
                    f"parent continuity failed: loss differs by {difference:.3e}. "
                    "The backend, its context or the graph is not the parent's."
                )
            provenance["continuity_check"] = {
                "status": "passed",
                "difference": difference,
            }

        if args.gradient_check:
            provenance["gradient_check"] = gradient_preflight(
                native, graph, key, out / "gradient_check"
            )

        settings = {
            "backend": args.backend,
            "steps": args.steps,
            "eta_X": args.eta_X,
            "eta_E": args.eta_E,
            "seed": args.seed,
            "hardening": args.hardening,
            "hardening_gradient": args.hardening_gradient if args.hardening else None,
            "experiment_file": None if args.config is None else str(args.config),
            "stage2": config.record(),
            "stage2_source_sha256": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("run.py", "backends.py", "operators.py", "settings.py")
            },
            "boundary": (
                "frozen structure model; chemistry is an explicit post-step "
                "operator, never a loss; hard molecules are readouts only"
            ),
            **provenance,
        }
        if prepare_loss is not None:
            settings["prior"] = prior_identity
        identities = {
            "requested_identity": {
                "source": source_identity(ROOT),
                "environment": environment,
                "settings": settings,
            },
            "initial_graph_sha256": save_graph(out / "inputs/initial_graph.npz", graph),
        }
        write_json(out / "input_identity.json", identities)
        write_json(out / "stage2_config.json", settings)

        start = time.monotonic()
        print(
            f"START {out.name} backend={args.backend} mode={config.mode} "
            f"improvements={config.improvements or 'none'} steps={args.steps}",
            flush=True,
        )
        loss_factory = None
        if args.hardening:
            loss_factory = hardening_factory(native, args.steps, args.hardening_gradient)
        elif prepare_loss is not None:
            loss_factory = partial(prepare_loss, native)
        result = run_trajectory(
            native,
            graph,
            key,
            directory=out / "trajectory",
            config=TrajectoryConfig(
                steps=args.steps,
                eta_X=args.eta_X,
                eta_E=args.eta_E,
                readout_every=args.readout_every,
            ),
            identities=identities,
            resolved_config=settings,
            resume=args.resume,
            max_updates=args.max_updates,
            loss_factory=loss_factory,
            post_step=None if config.mode == "none" else post_step(config),
        )

        summary = {
            "run": out.name,
            "backend": args.backend,
            "score_metric": provenance["score_metric"],
            "mode": config.mode,
            "improvements": list(config.improvements),
            "hardening": args.hardening,
            "hardening_gradient": args.hardening_gradient if args.hardening else None,
            "complete": result.complete,
            "updates": result.update_step,
            "best_loss": result.best_value,
            "elapsed_seconds": time.monotonic() - start,
        }
        if result.complete:
            summary["soft_hard_gap"] = write_soft_hard_gap(
                out, native, key, provenance["score_metric"], hardened=args.hardening
            )
        cleanup_started = True
        summary.update(finish())
        write_json(out / "run_summary.json", summary)
        print(f"DONE {out.name} {json.dumps(summary)}", flush=True)
        return summary
    finally:
        if not cleanup_started:
            try:
                finish()
            except Exception as cleanup_error:
                print(f"Backend cleanup failed: {cleanup_error}", flush=True)


def write_soft_hard_gap(out, native, key, score_metric, *, hardened=False):
    """Evaluate every raw argmax readout under the same frozen model.

    A diagnostic only: a hard graph is never substituted back into the
    optimization. Recorded in *loss* space, where lower is better on both
    backends, so ``hard_minus_soft_loss`` is positive exactly when
    discretization hurt. The backend's own score is carried alongside for
    display, without assuming which direction it improves in.
    """
    import jax.numpy as jnp

    from mosaic.small_molecule.io import write_json
    from mosaic.small_molecule.state import RelaxedGraph, graph_from_hard

    trajectory = out / "trajectory"
    trace = [
        json.loads(line)
        for line in (trajectory / "trace.jsonl").read_text().splitlines()
    ]
    gaps = []
    for row in trace:
        if not row.get("readout"):
            continue
        readout = json.loads((trajectory / row["readout"]).read_text())
        hard = graph_from_hard(
            readout["element_indices"], readout["bond_upper_indices"]
        )
        hard_value, hard_aux = native(hard, key=key)
        soft_loss = row["aux_metrics"].get("native_loss", row["loss"])
        if hardened:
            # The forward graph was partly one-hot, so the arm's own loss is not
            # the relaxed state's loss. Re-evaluate the underlying probabilities.
            with np.load(trajectory / row["evaluated_arrays"]) as handle:
                relaxed = RelaxedGraph(
                    jnp.asarray(handle["X"]), jnp.asarray(handle["E_upper"])
                )
            relaxed_value, _ = native(relaxed, key=key)
            soft_loss = float(relaxed_value)
        gaps.append(
            {
                "step": row["evaluated_step"],
                "score_metric": score_metric,
                "optimization_forward_loss": row["loss"],
                "soft_loss": soft_loss,
                "hard_loss": float(hard_value),
                "hard_minus_soft_loss": float(hard_value) - soft_loss,
                "soft_score": row["aux_metrics"].get(score_metric),
                "hard_score": _scalar(hard_aux.get(score_metric)),
                "sanitized": readout.get("sanitized"),
                "fragment_count": readout.get("fragment_count"),
            }
        )
    write_json(out / "soft_hard_gap.json", gaps)
    return {
        "checkpoints": len(gaps),
        "mean_hard_minus_soft_loss": float(
            np.mean([g["hard_minus_soft_loss"] for g in gaps])
        ),
    }


def _scalar(value):
    return None if value is None else float(np.asarray(value))


def add_arguments(parser):
    parser.add_argument(
        "--config",
        type=Path,
        help=f"experiment YAML; every flag overrides it (default {DEFAULT.name})",
    )
    parser.add_argument("--backend", default="boltz2", choices=tuple(METRICS))
    parser.add_argument(
        "--mode", default="valence", choices=("none", "valence", "connected", "sparse")
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parent", type=Path, help="sealed parent run to continue")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eta-X", dest="eta_X", type=float, default=0.1)
    parser.add_argument("--eta-E", dest="eta_E", type=float, default=1.0)
    parser.add_argument("--readout-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hardening", action="store_true", help="Experiment D")
    parser.add_argument(
        "--hardening-gradient",
        default="inherited",
        choices=HARDENING_GRADIENTS,
        help="how the gradient crosses the hardening step; see experiment.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--allow-local-gpu",
        action="store_true",
        help="run the Boltz worker on a workstation GPU instead of an LSF job",
    )
    parser.add_argument(
        "--no-gradient-check",
        dest="gradient_check",
        action="store_false",
        help="skip the finite-difference preflight (Boltz-2 only)",
    )
    boltz = parser.add_argument_group("Boltz-2 backend")
    boltz.add_argument(
        "--boltz-context", type=Path, help="prepared context directory (context.pt)"
    )
    boltz.add_argument("--boltz-checkpoint", type=Path, help="pinned affinity .ckpt")
    boltz.add_argument(
        "--boltz-python",
        type=Path,
        help="interpreter for the worker when torch/boltz live in their own venv",
    )
    boltz.add_argument(
        "--seed-mixing",
        type=float,
        default=0.5,
        help="uniform mixing applied to the prepared carbon-chain seed",
    )
    protenix = parser.add_argument_group("Protenix-Tiny backend (legacy)")
    protenix.add_argument("--target", default="brd4", choices=("brd4", "t4l", "fkbp12"))
    protenix.add_argument(
        "--cache-dir", type=Path, default=Path("/opt/mosaic-stage0/cache/protenix")
    )
    chemistry = parser.add_argument_group("protocol parameters")
    chemistry.add_argument(
        "--aromatic-order", type=float, choices=(1.0, 1.5), default=1.5
    )
    chemistry.add_argument("--bridge-probability", type=float, default=0.6)
    chemistry.add_argument("--sparsity-amount", type=float, default=0.01)
    chemistry.add_argument("--sparsity-threshold", type=float, default=0.5)
    better = parser.add_argument_group("improvements (all off by default)")
    better.add_argument(
        "--readout-caps",
        action="store_true",
        help="tighten the valence ceiling to the argmax element",
    )
    better.add_argument(
        "--valence-damping",
        type=float,
        default=1.0,
        help="partial valence projection, in (0, 1]",
    )
    better.add_argument(
        "--fluidity-threshold",
        type=float,
        default=1.0,
        help="relax rows whose max probability exceeds this",
    )
    better.add_argument(
        "--fluidity-strength",
        type=float,
        default=0.0,
        help="uniform mixing applied to collapsed rows",
    )
    better.add_argument(
        "--bridge-retry",
        action="store_true",
        help="one stronger re-bridge when valence removes bridges",
    )
    return parser


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    args = apply(parser)
    try:
        run(args)
    except Exception as error:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "attempt_failure.json").write_text(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
