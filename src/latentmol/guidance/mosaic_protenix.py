"""PyTorch VJP boundary for a resident frozen Mosaic Protenix process.

The child owns JAX/native weights; the parent owns the frozen Torch decoder.
JSON transports input probability values and their native derivatives, not a
hard molecule. A custom autograd function applies that VJP to the SAME Torch
probabilities. Only first derivatives are supported. No model is trained here.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import warnings

import torch

from latentmol.representation import GraphSpec, RelaxedGraph


PROTOCOL = "mosaic-protenix-probability-vjp-v1"
_SHUTDOWN_GRACE_SECONDS = 15


class _NativeVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atoms, charges, bonds, owner):
        response = owner._evaluate_arrays(atoms, charges, bonds)
        values = tuple(torch.as_tensor(response["gradients"][name], device=value.device,
                                       dtype=value.dtype) for name, value in
                       zip(("atoms", "charges", "bonds"), (atoms, charges, bonds)))
        if any(g.shape != v.shape or not bool(torch.isfinite(g).all())
               for g, v in zip(values, (atoms, charges, bonds))):
            raise ValueError("Native VJP has incompatible/nonfinite input derivatives")
        ctx.save_for_backward(*values)
        owner.last_aux = response["aux"]
        return atoms.new_tensor(response["loss"])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        return (*(upstream * value for value in ctx.saved_tensors), None)


class MosaicProtenixScorer:
    """Frozen, fixed-slot contact scorer with an explicit subprocess command.

    ``active`` maps parent graph slots to prepared native ligand atoms in order.
    Occupied atom distributions exclude NO_ATOM; inactive graph labels remain
    outside the scorer. Charges enter as expected formal charge. Native bonds
    retain only presence; typed bond exchanges at equal presence are a declared
    null direction. Raw inputs are retained by the caller, never repaired here.
    """
    def __init__(self, spec: GraphSpec, active, *, command, config,
                 stderr_path, timeout=900, startup_timeout=900):
        self.spec = spec
        self.active = torch.as_tensor(active, dtype=torch.bool).detach().cpu().clone()
        if self.active.shape != (spec.max_nodes,) or not bool(self.active.any()):
            raise ValueError("Require one nonempty fixed parent slot mask")
        if spec.atomic_numbers[0] != 0 or any(not 1 <= n <= 128 for n in spec.atomic_numbers[1:]):
            raise ValueError("Native element channels support atomic numbers1..128")
        if spec.bond_names[0] != "NONE":
            raise ValueError("Bond index zero must denote NONE")
        if not command or not all(isinstance(v, str) and v for v in command):
            raise ValueError("Provide an explicit native worker argv")
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
               for v in (timeout, startup_timeout)):
            raise ValueError("Worker deadlines must be finite and positive")
        self.timeout = float(timeout)
        self.last_aux = {}
        self._request_id = 0
        self._lock = threading.Lock()
        self._closed = False
        self.shutdown_status = None
        path = Path(stderr_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = path.open("x", encoding="utf-8")
        self._responses = queue.Queue()
        options = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                       text=True, bufsize=1)
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        self._process = subprocess.Popen(command, **options)
        self._reader = threading.Thread(target=self._read_lines, daemon=True)
        self._reader.start()
        try:
            ready = self._rpc(dict(op="init", protocol=PROTOCOL, config=config,
                n_atoms=int(self.active.sum()), atomic_numbers=list(spec.atomic_numbers[1:]),
                formal_charges=list(spec.formal_charges), bond_names=list(spec.bond_names)),
                timeout=float(startup_timeout))
            self.identity = ready["identity"]
            self.devices = ready.get("devices", [])
            if ready.get("n_atoms") != int(self.active.sum()):
                raise ValueError("Native context slot count differs")
        except BaseException:
            self._terminate()
            raise

    def _read_lines(self):
        try:
            for line in self._process.stdout:
                # Other libraries' stdout is redirected in the worker. Reject
                # stray protocol output instead of silently skipping failures.
                try:
                    value = json.loads(line)
                except ValueError as exc:
                    self._responses.put(RuntimeError("Non-JSON native worker stdout: " + line[:200]))
                    return
                self._responses.put(value)
        finally:
            self._responses.put(EOFError("Native worker stdout closed"))

    def _rpc(self, request, *, timeout=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Native scorer is closed")
            identifier = self._request_id
            self._request_id += 1
            self._process.stdin.write(json.dumps(dict(request, request_id=identifier), allow_nan=False) + "\n")
            self._process.stdin.flush()
            try:
                response = self._responses.get(timeout=self.timeout if timeout is None else timeout)
            except queue.Empty as exc:
                self._terminate()
                raise TimeoutError("Native Protenix response deadline reached") from exc
            if isinstance(response, BaseException):
                self._terminate()
                raise response
            if response.get("request_id") != identifier or response.get("protocol") != PROTOCOL:
                self._terminate()
                raise ValueError("Native worker protocol/request identity differs")
            if response.get("status") != "ok":
                self._terminate()
                raise RuntimeError("Native Protenix failed: " + response.get("error", "unknown error"))
            return response

    def _evaluate_arrays(self, atoms, charges, bonds):
        response = self._rpc(dict(op="evaluate", arrays={name:value.detach().cpu().tolist()
            for name,value in zip(("atoms","charges","bonds"),(atoms,charges,bonds))}))
        if not isinstance(response.get("loss"), (float, int)) or not math.isfinite(response["loss"]):
            raise ValueError("Native contact objective is nonfinite")
        return response

    def __call__(self, graph: RelaxedGraph):
        return self.score_at(graph, self.active)

    def score_at(self, graph: RelaxedGraph, active):
        """Score a branch's explicit slot map in the same fixed-count native context."""
        active = torch.as_tensor(active)
        if (active.dtype != torch.bool or active.shape != self.active.shape
                or int(active.sum()) != int(self.active.sum())):
            raise ValueError("Protenix branch slot mask must preserve the prepared ligand count")
        n, a, q, e = self.spec.max_nodes, len(self.spec.atomic_numbers), len(self.spec.formal_charges), len(self.spec.bond_names)
        if tuple(graph.atoms.shape)!=(1,n,a) or tuple(graph.charges.shape)!=(1,n,q) or tuple(graph.bonds.shape)!=(1,n,n,e):
            raise ValueError("Protenix requires exactly one shared-schema graph")
        mask = active.to(graph.atoms.device)
        arrays = (graph.atoms[0,mask,1:], graph.charges[0,mask], graph.bonds[0,mask][:,mask])
        if any(value.dtype != torch.float32 or value.device != arrays[0].device for value in arrays):
            raise ValueError("Native bridge requires common-device FP32 probabilities")
        if not torch.equal(mask, (graph.occupancy[0].detach() > .5)):
            raise ValueError("Protenix active slot identity changed")
        for values in arrays:
            if not values.is_floating_point() or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
                raise ValueError("Native inputs must be finite nonnegative probabilities")
            if not torch.allclose(values.sum(-1), torch.ones_like(values[...,0]), atol=2e-5, rtol=2e-5):
                raise ValueError("Native input categorical rows must sum to one")
        if not torch.allclose(arrays[2], arrays[2].transpose(0,1), atol=1e-6, rtol=0):
            raise ValueError("Native input bonds must be symmetric")
        if bool((graph.atoms[0,~mask,1:].detach().abs()>1e-6).any()):
            raise ValueError("Inactive parent slots must be exactly absent")
        excluded = ~(mask[:,None]&mask[None,:]) | torch.eye(n,dtype=torch.bool,device=mask.device)
        if not torch.allclose(graph.bonds[0,...,0][excluded],torch.ones_like(graph.bonds[0,...,0][excluded]),atol=1e-6,rtol=0):
            raise ValueError("Excluded and diagonal parent bonds must be NONE")
        loss = _NativeVJP.apply(*arrays, self)
        return loss, dict(self.last_aux)

    evaluate = __call__

    def _terminate(self):
        self._closed = True
        requested = False
        if self._process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)
                    requested = True
                except ProcessLookupError:
                    pass
            else:
                self._process.terminate()
                requested = True
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(self._process.pid, signal.SIGKILL)
                        requested = True
                    except ProcessLookupError:
                        pass
                else:
                    self._process.kill()
                    requested = True
                self._process.wait(timeout=5)
        self._process.stdin.close()
        self._stderr.close()
        return requested

    def close(self):
        if self._closed:
            return
        exit_timeout = False
        try:
            result = self._rpc(dict(op="close"))
            if result.get("frozen_unchanged") is not True:
                raise RuntimeError("Frozen Protenix/context identity changed")
            # The child has flushed its terminal result. Stop holding its input
            # open while the worker exits the request loop and tears down JAX.
            self._process.stdin.close()
            try:
                self._process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                exit_timeout = True
        finally:
            signalled = self._terminate()
        returncode = self._process.returncode
        forced = exit_timeout and signalled
        forced_returncodes = {-signal.SIGTERM, -signal.SIGKILL} if os.name == "posix" else {1}
        if returncode and not (forced and returncode in forced_returncodes):
            raise RuntimeError(f"Native worker exited unsuccessfully ({returncode})")
        self.shutdown_status = "terminated_after_close_ack" if forced else "graceful"
        if forced:
            warnings.warn(
                "Native Protenix acknowledged unchanged frozen state but timed out during process teardown; "
                f"the parent terminated it (exit code {returncode}).",
                RuntimeWarning, stacklevel=2,
            )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
