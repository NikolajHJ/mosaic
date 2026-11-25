"""Deterministic frozen DeFoG consistency critic with continuous auxiliaries.

This is an exploratory structural prior, not a synthesis classifier or a
likelihood. The original GuacaMol checkpoint owns no formal-charge channel.
The explicit charge-null projection, soft cycle clamp, and expectations of
legacy *category-index* molecular auxiliaries are part of this new policy.
No sampler, noise draw, argmax auxiliary feature, repair, or training is used.
The resident CPU process keeps the original isolated DeFoG dependencies while
returning exact first VJPs of this declared continuous scalar to Torch.

Cycle formulas derive from DeFoG, Copyright (c) 2012-2025 Manuel Madeira,
Yiming Qin, MIT license. Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
"AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import torch
from torch import nn

from latentmol.defog.assets import DEFOG_COMMIT, GUACAMOL_SHA256
from latentmol.defog.codec import GUACAMOL_ATOMIC_NUMBERS, GUACAMOL_BONDS
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer


POLICY = "defog-rrwp-softaux-attached-js-v1"


def _cycles(adjacency):
    """Native 3..6 cycle polynomials, clamped to [0,1] after /10.

    The original integer-graph identities may be negative on fractional
    adjacency. Clamping replaces their native assertions explicitly; these
    values are continuous structural features, not expected simple-cycle counts.
    """
    a = adjacency
    d = a.sum(-1)
    powers = {1: a}
    for order in range(2, 7):
        powers[order] = powers[order - 1] @ a
    diag = lambda value: value.diagonal(dim1=-2, dim2=-1)
    trace = lambda value: diag(value).sum(-1)
    triangles = diag(powers[3])
    c3 = triangles.sum(-1) / 6
    c4 = (diag(powers[4]) - d * (d - 1) - (a @ d[..., None]).sum(-1)).sum(-1) / 8
    c5 = (diag(powers[5]) - 2 * triangles * d
          - (a @ triangles[..., None]).sum(-1) + triangles).sum(-1) / 10
    d2 = diag(powers[2])
    c6 = (trace(powers[6]) - 3 * trace(powers[3].square())
          + 9 * (a * powers[2].square()).sum((-2, -1))
          - 6 * (d2 * diag(powers[4])).sum(-1)
          + 6 * trace(powers[4]) - 4 * trace(powers[3])
          + 4 * d2.pow(3).sum(-1) + 3 * powers[3].sum((-2, -1))
          - 12 * d2.square().sum(-1) + 4 * trace(powers[2])) / 12
    return (torch.stack((c3, c4, c5, c6), -1) / 10).clamp(0, 1)


def continuous_features(x, e, *, max_nodes, atom_weights, max_weight, model_time):
    """Construct the trained core's complete 34/25/7 input widths.

    X: native 12-category atom simplex; E: native five-category bonds, with
    zero diagonal. All nodes are present. RRWP matches the native policy;
    molecular category expectations replace argmax operations. In particular,
    the released feature's "valency" is its bond-category index, including
    aromatic index4, not physical bond order1.5. Preserve that trained meaning.
    """
    batch, nodes, _ = x.shape
    if x.shape != (batch, nodes, 12) or e.shape != (batch, nodes, nodes, 5):
        raise ValueError("DeFoG continuous features require native 12/5 channels")
    if nodes < 2 or max_nodes < nodes or max_weight <= 0 or not 0 < model_time < 1:
        raise ValueError("DeFoG feature metadata or graph size is incompatible")
    a = e[..., 1:].sum(-1)
    degree = a.sum(-1, keepdim=True)
    transition = a / degree.clamp_min(1e-8)
    identity = torch.eye(nodes, device=x.device, dtype=x.dtype)[None].expand(batch, -1, -1)
    walks = [identity]
    for _ in range(19):
        walks.append(walks[-1] @ transition)
    rrwp = torch.stack(walks, -1)
    index = torch.arange(nodes, device=x.device)
    rrwp_nodes = rrwp[:, index, index]
    # Expectations of the original checkpoint's index-valued auxiliaries.
    bond_index_valency = (e @ e.new_tensor(range(5))).sum(-1)
    element_index = x @ x.new_tensor(range(12))
    charge_proxy = element_index - bond_index_valency
    weight = (x @ atom_weights.to(x)).sum(-1, keepdim=True) / max_weight
    node_features = torch.cat((x, rrwp_nodes, charge_proxy[..., None], bond_index_valency[..., None]), -1)
    edge_features = torch.cat((e, rrwp), -1)
    global_features = torch.cat((x.new_full((batch, 1), nodes / max_nodes), _cycles(a),
                                 weight, x.new_full((batch, 1), model_time)), -1)
    if not all(bool(value.isfinite().all()) for value in (node_features, edge_features, global_features)):
        raise ValueError("Nonfinite continuous DeFoG auxiliary features")
    return node_features, edge_features, global_features


def _js(p, q):
    midpoint = (p + q) * 0.5
    # The epsilon explicitly bounds derivatives at exact categorical zeros.
    log_midpoint = midpoint.clamp_min(1e-8).log()
    return (0.5 * p * (p.clamp_min(1e-8).log() - log_midpoint)
            + 0.5 * q * (q.clamp_min(1e-8).log() - log_midpoint)).sum(-1)


def consistency_loss(core, x, e, *, max_nodes, atom_weights, max_weight, model_time):
    """Attached deterministic JS discrepancy: graph versus denoiser prediction."""
    node, edge, global_ = continuous_features(x, e, max_nodes=max_nodes,
        atom_weights=atom_weights, max_weight=max_weight, model_time=model_time)
    node_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    prediction = core(node, edge, global_, node_mask)
    atom_predictions, bond_predictions = prediction.X.softmax(-1), prediction.E.softmax(-1)
    upper = torch.triu_indices(x.shape[1], x.shape[1], 1, device=x.device)
    atom_js = _js(x, atom_predictions).mean(-1)
    bond_js = _js(e[:, upper[0], upper[1]], bond_predictions[:, upper[0], upper[1]]).mean(-1)
    value = atom_js + bond_js
    if not bool(value.isfinite().all()):
        raise ValueError("DeFoG consistency objective is nonfinite")
    return value, dict(atom_js=atom_js, bond_js=bond_js,
                      atom_predictions=atom_predictions, bond_predictions=bond_predictions)


class FrozenDeFoG(nn.Module):
    """CPU-isolated pretrained DeFoG with a first-derivative Torch boundary."""
    def __init__(self, spec, weights_path, *, python, source_path, stderr_path=None,
                 model_time=0.95, timeout=120, startup_timeout=180):
        super().__init__()
        path = Path(weights_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != GUACAMOL_SHA256:
            raise ValueError("DeFoG checkpoint differs from the pinned GuacaMol weights")
        if (not isinstance(python, str) or not python or not Path(source_path).is_dir()
                or not 0 < model_time < 1):
            raise ValueError("DeFoG requires an explicit runtime, source checkout, and model time")
        if (any(number not in GUACAMOL_ATOMIC_NUMBERS for number in spec.atomic_numbers[1:])
                or any(name not in GUACAMOL_BONDS for name in spec.bond_names)):
            raise ValueError("DeFoG mapping would omit an atom or bond vocabulary category")
        self.spec, self.python, self.source_path = spec, python, str(source_path)
        self.weights_path, self.model_time = str(path), float(model_time)
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self._owned_temporary = tempfile.TemporaryDirectory(prefix="latentmol-defog-") if stderr_path is None else None
        self.stderr_path = str(Path(self._owned_temporary.name) / "worker.err") if stderr_path is None else str(stderr_path)
        self._resident = None
        self._closed = False
        self.identity = None
        self.shutdown_status = None
        signature = hashlib.sha256((digest + DEFOG_COMMIT + POLICY + str(model_time)).encode()).digest()
        self.register_buffer("_asset_identity", torch.tensor(list(signature), dtype=torch.uint8))
        self.requires_grad_(False).eval()
        self.adapter_metadata = dict(model="DeFoG-GuacaMol", policy=POLICY,
            checkpoint_sha256=digest, source_commit=DEFOG_COMMIT, device="resident_cpu",
            source="https://github.com/manuelmlmadeira/DeFoG/tree/" + DEFOG_COMMIT,
            objective="fully attached atom_JS + bond_JS denoiser consistency; not likelihood or SA",
            model_time=self.model_time, input_corruption="none; exact shared mapped graph",
            charge_policy="charge-agnostic structural prior; exact zero charge VJP; no neutralization",
            feature_policy="20 RRWP powers; cycle polynomials /10 clamped[0,1]; expected legacy category-index auxiliaries and atom weights",
            teacher_output_policy="all12elements/5bonds in discrepancy; unsupported output categories not renormalized away",
            randomness="none; frozen eval mode; no sampling/noise draws",
            boundary_policy="JS logarithms and RRWP degree denominator floored1e-8")

    def forward(self, graph, active):
        if self._closed:
            raise RuntimeError("DeFoG consistency scorer is closed")
        if self._resident is None:
            self._resident = MosaicProtenixScorer(self.spec, active,
                command=[self.python, "-B", "-u", "-m", __name__, "--worker"],
                config=dict(source=self.source_path, checkpoint=self.weights_path,
                    model_time=self.model_time, policy=POLICY), stderr_path=self.stderr_path,
                timeout=self.timeout, startup_timeout=self.startup_timeout)
            self.identity = self._resident.identity
        value, _ = self._resident.score_at(graph, active)
        return dict(difficulty=value.reshape(1), raw_score=-value.reshape(1))

    def close(self):
        if self._closed:
            return
        try:
            if self._resident is not None:
                self._resident.close()
                self.shutdown_status = self._resident.shutdown_status
        finally:
            self._closed = True
            if self._owned_temporary is not None:
                self._owned_temporary.cleanup()

    def reference_score(self, mol):
        """Posthoc hard-graph consistency, not native synthesis difficulty."""
        from rdkit import Chem
        from latentmol.chemistry import encode_smiles
        from latentmol.representation import HardGraphBatch
        row = encode_smiles(Chem.MolToSmiles(mol), self.spec)
        hard = HardGraphBatch(*(torch.tensor(row[key].astype("int64"))[None]
            for key in ("atom_types", "charges", "bond_types")))
        with torch.no_grad():
            result = self(hard.to_relaxed(self.spec), hard.atoms[0] != 0)
        return float(result["difficulty"].item())


def _state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _worker():
    """Same audited first-VJP transport used by the native contact bridge."""
    import contextlib
    import os
    import sys
    import time
    import traceback
    from latentmol.guidance.mosaic_protenix import PROTOCOL

    if not (os.environ.get("LSB_JOBID") or os.environ.get("PBS_JOBID")):
        raise RuntimeError("DeFoG numerical worker requires an allocated compute node")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = None
    before = None
    for line in sys.stdin:
        request = json.loads(line)
        response = dict(protocol=PROTOCOL, request_id=request.get("request_id"), status="ok")
        try:
            # The resident transport negotiates its protocol on init; later
            # evaluate/close requests carry only their monotonically increasing ID.
            if request.get("protocol", PROTOCOL) != PROTOCOL:
                raise ValueError("Unsupported DeFoG worker transport protocol")
            if request["op"] == "init":
                if model is not None:
                    raise ValueError("DeFoG worker was already initialized")
                config = request["config"]
                if config.get("policy") != POLICY:
                    raise ValueError("Unknown DeFoG continuous feature policy")
                from latentmol.defog.interface_smoke import _verify_assets, _load_native_model
                source, checkpoint = Path(config["source"]), Path(config["checkpoint"])
                _verify_assets(source, checkpoint)
                with contextlib.redirect_stdout(sys.stderr):
                    model = _load_native_model(source, checkpoint, torch.device("cpu"))
                if model.cfg.model.extra_features != "rrwp" or model.cfg.model.rrwp_steps != 20:
                    raise ValueError("The DeFoG checkpoint's extra-feature configuration differs")
                count = request["n_atoms"]
                if type(count) is not int or count < 2 or count > int(model.extra_features.max_n_nodes):
                    raise ValueError("Incompatible DeFoG fixed graph count")
                numbers, names = request["atomic_numbers"], request["bond_names"]
                formal_charges = request["formal_charges"]
                if (any(number not in GUACAMOL_ATOMIC_NUMBERS for number in numbers)
                        or any(name not in GUACAMOL_BONDS for name in names)):
                    raise ValueError("Worker input vocabulary cannot be mapped completely")
                atom_map = torch.zeros(len(numbers), 12)
                for i, number in enumerate(numbers):
                    atom_map[i, GUACAMOL_ATOMIC_NUMBERS.index(number)] = 1
                bond_map = torch.zeros(len(names), 5)
                for i, name in enumerate(names):
                    bond_map[i, GUACAMOL_BONDS.index(name)] = 1
                meta = dict(max_nodes=int(model.extra_features.max_n_nodes),
                    atom_weights=model.domain_features.weight.atom_weight_list.detach().cpu(),
                    max_weight=float(model.domain_features.weight.max_weight), model_time=float(config["model_time"]))
                before = _state_digest(model)
                identity = dict(model="DeFoG-GuacaMol", checkpoint_sha256=GUACAMOL_SHA256,
                    source_commit=DEFOG_COMMIT, frozen_state_sha256=before, policy=POLICY,
                    parameter_count=sum(p.numel() for p in model.parameters()), device="cpu")
                response.update(identity=identity, n_atoms=count, devices=["cpu"])
            elif request["op"] == "close":
                if model is None:
                    raise ValueError("DeFoG worker has not initialized")
                unchanged = before == _state_digest(model) and all(p.grad is None for p in model.parameters())
                response.update(frozen_unchanged=unchanged)
                print(json.dumps(response, allow_nan=False), flush=True)
                break
            elif request["op"] == "evaluate":
                if model is None:
                    raise ValueError("DeFoG worker has not initialized")
                started = time.monotonic()
                arrays = request["arrays"]
                fields = [torch.tensor(arrays[name], dtype=torch.float32, requires_grad=True)
                          for name in ("atoms", "charges", "bonds")]
                expected = ((count, len(numbers)), (count, len(formal_charges)),
                            (count, count, len(names)))
                if any(tuple(field.shape) != shape for field, shape in zip(fields, expected)):
                    raise ValueError("DeFoG probability request shape differs")
                for field in fields:
                    if (not bool(field.isfinite().all()) or bool((field < 0).any())
                            or not torch.allclose(field.sum(-1), torch.ones_like(field[..., 0]), atol=2e-5, rtol=2e-5)):
                        raise ValueError("DeFoG request must contain finite categorical simplex rows")
                if not torch.allclose(fields[2], fields[2].transpose(0, 1), atol=1e-6, rtol=0):
                    raise ValueError("DeFoG request bonds are asymmetric")
                x = (fields[0] @ atom_map)[None]
                e = (fields[2] @ bond_map)[None]
                off_diagonal = 1 - torch.eye(count)[None, :, :, None]
                e = e * off_diagonal
                value, aux = consistency_loss(model.model, x, e, **meta)
                gradients = torch.autograd.grad(value.sum(), fields, allow_unused=True)
                if gradients[0] is None or gradients[2] is None:
                    raise ValueError("Required DeFoG atom/bond input path detached")
                gradients = [torch.zeros_like(field) if grad is None else grad
                             for field, grad in zip(fields, gradients)]
                if any(not bool(grad.isfinite().all()) for grad in gradients):
                    raise ValueError("DeFoG first input derivative is nonfinite")
                if any(p.grad is not None for p in model.parameters()):
                    raise ValueError("Frozen DeFoG parameter gradients changed")
                unsupported = [i for i, number in enumerate(GUACAMOL_ATOMIC_NUMBERS) if number not in numbers]
                excluded = aux["atom_predictions"][..., unsupported].sum(-1).mean() if unsupported else value.new_zeros(())
                response.update(loss=float(value.item()), gradients={name: grad.tolist()
                    for name, grad in zip(("atoms", "charges", "bonds"), gradients)},
                    aux=dict(policy=POLICY, atom_js=float(aux["atom_js"].item()),
                        bond_js=float(aux["bond_js"].item()), unsupported_teacher_atom_mass=float(excluded),
                        charge_policy="exact_null_direction", no_random_noise=True),
                    native_seconds=time.monotonic() - started)
            else:
                raise ValueError("Unknown DeFoG worker operation")
        except Exception as error:
            response.update(status="error", error=str(error), traceback=traceback.format_exc())
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    _worker()
