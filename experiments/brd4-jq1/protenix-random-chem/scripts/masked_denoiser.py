"""Read-only DeFoG patch proposals in an isolated, CPU-only model worker."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys


# Mosaic atom codes 1..6 are C, N, O, S, F, Cl. Bond codes are shared.
NATIVE_ELEMENTS = (0, 1, 2, 9, 3, 6)


class MaskedDenoiser:
    def __init__(self, config):
        command = [
            str(config["python"]), "-u", str(Path(__file__).resolve()),
            "--probe", str(config["probe"]),
            "--source", str(config["source"]),
            "--source-commit", str(config["source_commit"]),
            "--checkpoint", str(config["checkpoint"]),
            "--checkpoint-sha256", str(config["checkpoint_sha256"]),
            "--steps", str(config.get("steps", 8)),
            "--start-time", str(config.get("start_time", 0.55)),
        ]
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        )
        environment.pop("PYTHONHOME", None)
        self.log = Path(config["log"]).open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, env=environment, text=True, encoding="utf-8",
            )
            self.audit = self._read(300)
            if (
                self.audit.get("status") != "ready"
                or self.audit.get("source_commit") != config["source_commit"]
                or self.audit.get("checkpoint_sha256") != config["checkpoint_sha256"]
            ):
                raise RuntimeError(f"DeFoG startup identity mismatch: {self.audit}")
            self.requests = 0
        except BaseException:
            self._stop()
            raise

    def _read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("DeFoG worker response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"DeFoG worker closed stdout (exit {self.process.poll()})")
        return json.loads(line)

    def _request(self, payload):
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        response = self._read(180)
        if response.get("status") != "passed":
            raise RuntimeError(f"DeFoG worker request failed: {response}")
        return response

    def propose(self, atoms, bonds, editable_atom_mask, editable_bond_mask, seed):
        import numpy as np

        response = self._request({
            "action": "propose", "request": self.requests,
            "atoms": np.asarray(atoms).tolist(), "bonds": np.asarray(bonds).tolist(),
            "editable_atoms": np.asarray(editable_atom_mask).tolist(),
            "editable_bonds": np.asarray(editable_bond_mask).tolist(),
            "seed": int(seed),
        })
        if (
            response.get("request") != self.requests
            or response.get("forwards") != self.audit["steps"]
        ):
            raise RuntimeError("DeFoG response identity or forward budget mismatch")
        self.requests += 1
        return (
            np.asarray(response["atoms"], dtype=np.int64),
            np.asarray(response["bonds"], dtype=np.int64),
        )

    def close(self):
        if self.log.closed:
            return
        try:
            if self.process.poll() is not None:
                raise RuntimeError(f"DeFoG worker exited early ({self.process.returncode})")
            final = self._request({"action": "finish"})
            if (
                final.get("state_digest") != self.audit["state_digest"]
                or final.get("proposals") != self.requests
                or final.get("forwards") != self.requests * self.audit["steps"]
                or self.process.wait(timeout=20) != 0
            ):
                raise RuntimeError(f"DeFoG frozen-state or budget check failed: {final}")
            self.audit["completion"] = final
        finally:
            self._stop()

    def _stop(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
        self.log.close()


def load_denoiser(config):
    return MaskedDenoiser(config)


def _validate_request(torch, request):
    atoms = torch.as_tensor(request["atoms"])
    bonds = torch.as_tensor(request["bonds"])
    atom_mask = torch.as_tensor(request["editable_atoms"])
    bond_mask = torch.as_tensor(request["editable_bonds"])
    n = atoms.numel()
    if (
        not 2 <= n <= 32 or atoms.shape != (n,) or bonds.shape != (n, n)
        or atom_mask.shape != (n,) or bond_mask.shape != (n, n)
        or atoms.dtype != torch.int64 or bonds.dtype != torch.int64
        or atom_mask.dtype != torch.bool or bond_mask.dtype != torch.bool
    ):
        raise ValueError("Malformed fixed-occupancy graph or edit masks")
    if (
        not ((1 <= atoms) & (atoms <= len(NATIVE_ELEMENTS))).all()
        or not ((0 <= bonds) & (bonds <= 4)).all()
        or not torch.equal(bonds, bonds.T) or bonds.diagonal().any()
        or not torch.equal(bond_mask, bond_mask.T) or bond_mask.diagonal().any()
        or not (atom_mask.any() or bond_mask.any())
    ):
        raise ValueError("Invalid graph labels or empty/asymmetric edit masks")
    seed = request["seed"]
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Expected an unsigned 64-bit proposal seed")
    return atoms, bonds, atom_mask, bond_mask, seed


def _sample_patch(torch, F, native_sampler, model, request, *, steps, start_time):
    atoms, bonds, atom_mask, bond_mask, seed = _validate_request(torch, request)
    native = torch.as_tensor(NATIVE_ELEMENTS)
    n = len(atoms)
    clean_x = F.one_hot(native[atoms - 1], 12).float().unsqueeze(0)
    clean_e = F.one_hot(bonds, 5).float().unsqueeze(0)
    node_mask = torch.ones((1, n), dtype=torch.bool)
    y = torch.empty((1, 0), dtype=torch.float32)
    edit_x = atom_mask[None, :, None]
    edit_e = bond_mask[None, :, :, None]
    times = torch.linspace(start_time, 1.0, steps + 1).reshape(-1, 1, 1)
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = model.apply_noise(clean_x, clean_e, y, node_mask, t=times[0])
        x, e = initial["X_t"].float(), initial["E_t"].float()
        for step, (t, s) in enumerate(zip(times[:-1], times[1:])):
            if step == steps - 1:
                # The native last step samples the predicted clean labels.
                # Condition only this terminal atom draw on Mosaic's vocabulary.
                data = {"X_t": x, "E_t": e, "y_t": y, "node_mask": node_mask, "t": t}
                pred = model.forward(data, model.compute_extra_data(data), node_mask)
                if not all(torch.isfinite(value).all() for value in (pred.X, pred.E)):
                    raise ValueError("DeFoG returned nonfinite terminal logits")
                atom_logits = torch.full_like(pred.X, -torch.inf)
                atom_logits[..., native] = pred.X[..., native]
                result = native_sampler(atom_logits.softmax(-1), pred.E.softmax(-1), node_mask)
                native_atoms = torch.where(atom_mask, result.X[0], clean_x[0].argmax(-1))
                final_bonds = torch.where(bond_mask, result.E[0], bonds)
                break
            proposal, _ = model.sample_p_zs_given_zt(t, s, x, e, y, node_mask)
            # Anchor context at the current native corruption level, rather
            # than exposing clean labels at all noisy intermediate times.
            context = model.apply_noise(clean_x, clean_e, y, node_mask, t=s)
            x = torch.where(edit_x, proposal.X, context["X_t"])
            e = torch.where(edit_e, proposal.E, context["E_t"])
        decode = torch.zeros(12, dtype=torch.long)
        decode[native] = torch.arange(1, len(native) + 1)
        final_atoms = decode[native_atoms]
        if (
            (final_atoms == 0).any() or not torch.equal(final_bonds, final_bonds.T)
            or final_bonds.diagonal().any() or not torch.equal(final_atoms[~atom_mask], atoms[~atom_mask])
            or not torch.equal(final_bonds[~bond_mask], bonds[~bond_mask])
        ):
            raise RuntimeError("Patch proposal changed its preserved context or vocabulary")
        return final_atoms.tolist(), final_bonds.tolist()


def _worker(args):
    if not os.environ.get("LSB_JOBID") or sys.flags.optimize:
        raise RuntimeError("An allocated HPC job with assertions enabled is required")
    if not 2 <= args.steps <= 24 or not 0.0 < args.start_time < 1.0:
        raise ValueError("Invalid bounded DeFoG proposal schedule")
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from omegaconf import OmegaConf
        from torch.nn import functional as F

        spec = importlib.util.spec_from_file_location("authenticated_defog_probe", args.probe)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        probe.torch, probe.OmegaConf = torch, OmegaConf
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model, loader_audit = probe.load_model(args.source, args.checkpoint, {
            "source_commit": args.source_commit,
            "checkpoint_sha256": args.checkpoint_sha256,
        })
        from flow_matching.flow_matching_utils import sample_discrete_features

        if model.conditional or model.training:
            raise ValueError("Expected the frozen unconditional GuacaMol model")
        initial_digest = probe.state_digest(model)

    def emit(response):
        print(json.dumps(response, allow_nan=False), flush=True)

    forwards = 0

    def count_forward(*_):
        nonlocal forwards
        forwards += 1

    hook = model.model.register_forward_pre_hook(count_forward)
    emit({
        "status": "ready", "device": "cpu",
        "source_commit": args.source_commit, "checkpoint_sha256": args.checkpoint_sha256,
        "state_digest": initial_digest, "loader": loader_audit,
        "steps": args.steps, "start_time": args.start_time,
        "context": "native_marginal_corruption_each_time_then_exact_clean_clamp",
        "terminal_atoms": "native_prediction_conditioned_on_Mosaic_six_elements",
        "schedule": "linear_in_native_flow_time",
    })
    proposals = 0
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["action"] == "finish":
                final_digest = probe.state_digest(model)
                if final_digest != initial_digest:
                    raise RuntimeError("Frozen DeFoG weights or buffers changed")
                emit({"status": "passed", "state_digest": final_digest,
                      "proposals": proposals, "forwards": forwards})
                return
            if request["action"] != "propose" or request["request"] != proposals:
                raise ValueError("Unknown or out-of-order proposal request")
            before = forwards
            with contextlib.redirect_stdout(sys.stderr):
                atoms, bonds = _sample_patch(
                    torch, F, sample_discrete_features, model, request,
                    steps=args.steps, start_time=args.start_time,
                )
            if forwards - before != args.steps:
                raise RuntimeError("Unexpected number of native DeFoG forwards")
            emit({"status": "passed", "request": proposals, "atoms": atoms,
                  "bonds": bonds, "forwards": forwards - before})
            proposals += 1
    finally:
        hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--start-time", type=float, default=0.55)
    _worker(parser.parse_args())
