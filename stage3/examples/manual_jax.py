"""A runnable model-free loop, or replace its teacher with the real checkpoint.

python stage3/examples/manual_jax.py --mock
python stage3/examples/manual_jax.py --source SOURCE --checkpoint CKPT --teacher-python PYTHON
"""

import argparse
from contextlib import ExitStack
import json

import jax
import jax.numpy as jnp
import numpy as np

from mosaic_defog import DeFoGTeacher, Graph, Guidance, Prediction


def simplex(rows):
    """Example Euclidean projection. Substitute your own projection callable."""
    ordered = jnp.sort(rows, axis=-1)[:, ::-1]
    totals = jnp.cumsum(ordered, axis=-1) - 1
    ranks = jnp.arange(1, rows.shape[-1] + 1)
    active = jnp.sum(ordered - totals / ranks > 0, axis=-1)
    threshold = jnp.take_along_axis(totals, (active - 1)[:, None], axis=-1) / active[:, None]
    return jnp.maximum(rows - threshold, 0)


def projection(proposal):
    return Graph(simplex(proposal.X), simplex(proposal.E_upper))


def native(graph, *, key=None):
    # A toy differentiable score; replace with your frozen model objective.
    score = jnp.mean(graph.X[:, 1]) + 0.01 * jnp.mean(1 - graph.E_upper[:, 0])
    return -score, {"toy_score": score}


class MockTeacher:
    def __init__(self):
        self.queries = 0

    def predict(self, graph, *, time):
        self.queries += 1
        n = len(graph.X)
        i, j = np.triu_indices(n, 1)
        return Prediction(np.eye(6, dtype=np.float32)[np.zeros(n, int)],
                          np.eye(5, dtype=np.float32)[(j == i+1).astype(int)], time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-python")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not args.mock and not (args.source and args.checkpoint):
        parser.error("Choose --mock or provide --source and --checkpoint")
    if args.steps < 1:
        parser.error("--steps must be positive")
    graph = Graph(jnp.full((25, 6), 1/6), jnp.full((300, 5), 1/5))
    with ExitStack() as resources:
        teacher = MockTeacher() if args.mock else resources.enter_context(
            DeFoGTeacher(args.source, args.checkpoint, python=args.teacher_python)
        )
        guidance = Guidance(teacher, weight=0.0625, time=0.8)
        for step in range(args.steps):
            objective = guidance.prepare(native, graph, step)
            (value, aux), gradients = jax.value_and_grad(objective, has_aux=True)(graph)
            if not np.isfinite(value) or not all(
                np.isfinite(g).all() for g in gradients
            ):
                raise FloatingPointError("Nonfinite student loss or gradient")
            proposal = jax.tree.map(lambda p, g: p - 0.3*g, graph, gradients)
            graph = projection(proposal)
        print(json.dumps({"scope": "toy objective; no molecular-quality claim",
                          "updates": args.steps, "teacher_queries": teacher.queries,
                          "last_evaluated_loss": float(value),
                          "last_evaluated_native_loss": float(aux["defog_native_loss"]),
                          "final_shape": [list(graph.X.shape), list(graph.E_upper.shape)]}))


if __name__ == "__main__":
    main()
