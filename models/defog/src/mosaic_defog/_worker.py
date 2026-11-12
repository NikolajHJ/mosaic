"""Private CPU worker. Upstream modules are available only in this interpreter."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from .graph import digest, validate_arrays
from .teacher import PROTOCOL

NATIVE_ELEMENTS = ("C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si")
OMITTED_BUFFERS = {
    "sampling_metrics.n_target_dist",
    "sampling_metrics.node_target_dist",
    "sampling_metrics.edge_target_dist",
    "sampling_metrics.valency_target_dist",
}


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def state_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((value.shape, value.dtype)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def load_model(args):
    checkpoint_sha = file_digest(args.checkpoint)
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from the requested model")
    revision = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.source_revision and revision != args.source_revision:
        raise ValueError(
            "Upstream revision differs; checkout the documented DeFoG revision"
        )
    changed = subprocess.run(
        ["git", "-C", str(args.source), "diff", "--quiet", "HEAD", "--"], check=False
    )
    if changed.returncode:
        raise ValueError("Upstream tracked source differs from its Git revision")

    import torch
    from graph_discrete_flow_model import GraphDiscreteFlowModel

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    model = GraphDiscreteFlowModel.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
        strict=False,
        train_metrics=None,
        sampling_metrics=None,
        visualization_tools=None,
    )
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    loaded = model.state_dict()
    if set(saved) - set(loaded) != OMITTED_BUFFERS or set(loaded) - set(saved):
        raise ValueError(
            "Unexpected checkpoint omissions; only evaluation buffers may be omitted"
        )
    if any(
        value.dtype != saved[key].dtype
        or value.shape != saved[key].shape
        or not torch.equal(value, saved[key])
        for key, value in loaded.items()
    ):
        raise ValueError("Restored model tensors differ from the checkpoint")
    if model.cfg.dataset.name != "guacamol" or model.cfg.model.transition != "marginal":
        raise ValueError(
            "This adapter supports the GuacaMol marginal checkpoint family"
        )
    if model.output_dims != {"X": 12, "E": 5, "y": 0}:
        raise ValueError("Unexpected native vocabulary or output heads")
    model.eval().requires_grad_(False)
    return model, {
        "source_revision": revision,
        "checkpoint_sha256": checkpoint_sha,
        "state_digest": state_digest(model),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "input_dims": dict(model.input_dims),
        "output_dims": dict(model.output_dims),
        "omitted_evaluation_buffers": sorted(OMITTED_BUFFERS),
        "device": "cpu",
    }


def predict(model, request, args):
    import torch
    from torch.nn import functional as F

    x = np.asarray(request["X"], dtype=np.float32)
    e = np.asarray(request["E"], dtype=np.float32).reshape(-1, 5)
    validate_arrays(x, e, n_elements=len(args.elements))
    if digest({"X": x.tolist(), "E": e.tolist()}) != request["graph_sha256"]:
        raise ValueError("Graph transport digest mismatch")
    time = float(request["time"])
    if not np.isfinite(time) or not 0 <= time <= 1:
        raise ValueError("Teacher time must lie in [0,1]")
    n = len(x)
    if n < 2:
        raise ValueError("The native teacher requires at least two occupied atoms")
    indices = [NATIVE_ELEMENTS.index(element) for element in args.elements]
    nx = torch.zeros((1, n, 12), dtype=torch.float32)
    nx[0, :, indices] = torch.from_numpy(x)
    ne = torch.zeros((1, n, n, 5), dtype=torch.float32)
    upper = torch.triu_indices(n, n, 1)
    ne[0, upper[0], upper[1]] = torch.from_numpy(e)
    ne[0, upper[1], upper[0]] = torch.from_numpy(e)
    data = {
        "X_t": nx,
        "E_t": ne,
        "y_t": torch.zeros((1, 0)),
        "node_mask": torch.ones((1, n), dtype=torch.bool),
        "t": torch.full((1, 1), time),
    }
    auxiliary = data
    if args.feature_policy == "hard_aux_soft_core":
        hx = F.one_hot(nx.argmax(-1), 12).float()
        he = F.one_hot(ne.argmax(-1), 5).float()
        diagonal = torch.arange(n)
        he[:, diagonal, diagonal] = 0
        auxiliary = {**data, "X_t": hx, "E_t": he}
    with torch.inference_mode():
        extra = model.compute_extra_data(auxiliary)
        prediction = model.forward(data, extra, data["node_mask"])
        full_x, full_e = prediction.X.softmax(-1), prediction.E.softmax(-1)
        qx = full_x[0, :, indices]
        mass = qx.sum(-1, keepdim=True)
        if not torch.isfinite(mass).all() or bool((mass < 1e-8).any()):
            raise ValueError(
                "Insufficient teacher probability on the requested elements"
            )
        qx = qx / mass
        qe = full_e[0, upper[0], upper[1]]
        if not torch.allclose(full_e, full_e.transpose(1, 2), atol=1e-7, rtol=0):
            raise ValueError("Asymmetric teacher bond predictions")
        validate_arrays(qx.numpy(), qe.numpy(), n_elements=len(args.elements))
        if qx.requires_grad or qe.requires_grad:
            raise ValueError("Teacher targets must be detached")
    target = {"X": qx.tolist(), "E": qe.tolist()}
    return {
        "target": target,
        "target_sha256": digest(target),
        "graph_sha256": request["graph_sha256"],
        "time": time,
        "metrics": {
            "excluded_element_mass_mean": float((1 - mass).mean()),
            "excluded_element_mass_max": float((1 - mass).max()),
        },
    }


def emit(reply):
    print(json.dumps(reply, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--elements", required=True)
    parser.add_argument(
        "--feature-policy", choices=("hard_aux_soft_core", "native_soft"), required=True
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.elements = tuple(args.elements.split(","))
    request = {}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model, identity = load_model(args)
        emit(
            {
                "status": "ready",
                "protocol": PROTOCOL,
                **identity,
                "elements": list(args.elements),
                "feature_policy": args.feature_policy,
            }
        )
        forwards, queries = [0], 0
        hook = model.model.register_forward_pre_hook(
            lambda *_: forwards.__setitem__(0, forwards[0] + 1)
        )
        try:
            for line in sys.stdin:
                request = json.loads(line)
                if request["action"] == "predict":
                    before = forwards[0]
                    with contextlib.redirect_stdout(sys.stderr):
                        result = predict(model, request, args)
                    if forwards[0] - before != 1:
                        raise ValueError("Expected exactly one native denoiser forward")
                    queries += 1
                    emit(
                        {
                            "id": request["id"],
                            "status": "passed",
                            "forward_calls": 1,
                            **result,
                        }
                    )
                elif request["action"] == "frozen":
                    after = state_digest(model)
                    if after != identity["state_digest"] or forwards[0] != queries:
                        raise ValueError("Teacher model state or forward count changed")
                    emit(
                        {
                            "id": request["id"],
                            "status": "passed",
                            "state_digest": after,
                            "queries": queries,
                            "forwards": forwards[0],
                        }
                    )
                else:
                    raise ValueError("Unknown worker action")
        finally:
            hook.remove()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit(
            {
                "id": request.get("id"),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
