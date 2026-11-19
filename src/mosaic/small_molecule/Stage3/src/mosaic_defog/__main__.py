"""Check a real checkpoint with one CPU query; no optimization or projection."""

import argparse
import json

import numpy as np

from . import DeFoGTeacher, Graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--python", help="Interpreter containing the upstream DeFoG dependencies"
    )
    parser.add_argument("--time", type=float, default=0.8)
    args = parser.parse_args()
    n = 25
    i, j = np.triu_indices(n, 1)
    graph = Graph(
        np.eye(6, dtype=np.float32)[np.zeros(n, dtype=int)],
        np.eye(5, dtype=np.float32)[(j == i + 1).astype(int)],
    )
    with DeFoGTeacher(args.source, args.checkpoint, python=args.python) as teacher:
        target = teacher.predict(graph, time=args.time)
        result = {
            "status": "passed",
            "scope": "one native query; no chemical-quality guarantee",
            "atoms": len(target.X),
            "pairs": len(target.E_upper),
            "time": target.time,
            "queries": teacher.queries,
            "identity": teacher.identity,
            "prediction_metrics": target.metadata,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
