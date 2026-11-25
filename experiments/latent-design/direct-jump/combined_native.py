"""Resident Nesso worker for the direct graph runner in the Protenix environment."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np


class NessoWorker:
    """Pass the Protenix slot-ordered probability rows to the frozen PyTorch model."""

    def __init__(self, seed, founder_graph, work, assets, python=None, *, schema_founder=None):
        script = Path(__file__).resolve().with_name("combined_nesso_worker.py")
        assets = Path(assets).resolve()
        work = Path(work).resolve()
        work.mkdir(parents=True, exist_ok=False)
        self.log_path = work / "worker.log"
        self.audit = None
        env = dict(os.environ, PYTHONPATH=str(script.parents[3] / "src"),
                   PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        for key in ("PYTHONHOME", "DBMOL_LAUNCH_READY", "DBMOL_LAUNCH_NONCE"):
            env.pop(key, None)
        if "COMBINED_NESSO_LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = env["COMBINED_NESSO_LD_LIBRARY_PATH"]
        # Both runtimes inherit exactly the scheduler-owned GPU visibility and MPS routing.
        with self.log_path.open("x", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                [str(python or assets / "env/bin/python"), "-u", str(script),
                 "--assets", str(assets), "--work", str(work), "--seed", str(seed)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                encoding="utf-8", env=env, bufsize=1)
        try:
            if schema_founder is None:
                self.identity = self._request("init", **self._graph(*founder_graph))
            else:
                self.identity = self._request("init_schema", founder=schema_founder)
        except BaseException:
            self._stop()
            raise

    @staticmethod
    def _graph(atom, bond):
        for rows, shape in ((atom, (25, 6)), (bond, (300, 5))):
            if rows.shape != shape or rows.dtype != np.float32:
                raise ValueError("Expected FP32 direct atom and unordered bond rows")
        return dict(atom=atom.tolist(), bond=bond.tolist())

    def _request(self, op, **data):
        try:
            self.process.stdin.write(json.dumps(dict(op=op, **data), separators=(",", ":"), allow_nan=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Nesso worker stopped during {op}. See {self.log_path}") from exc
        if not line:
            raise RuntimeError(f"Nesso worker stopped during {op}. See {self.log_path}")
        reply = json.loads(line)
        if reply["op"] != op:
            raise RuntimeError("Nesso worker reply did not match its request")
        if "error" in reply:
            raise RuntimeError(f"Nesso {op} failed: {reply['error']}. See {self.log_path}")
        return reply["result"]

    @staticmethod
    def _gradient_reply(reply, atom, bond):
        value, da, db = reply
        ga, gb = np.asarray(da, dtype=np.float32), np.asarray(db, dtype=np.float32)
        if (ga.shape != atom.shape or gb.shape != bond.shape or not np.isfinite(value)
                or not np.isfinite(ga).all() or not np.isfinite(gb).all()):
            raise ValueError("Nesso worker returned an invalid loss or derivative")
        return float(value), ga, gb

    def gradient(self, atom, bond):
        reply = self._request("gradient", **self._graph(atom, bond))
        return self._gradient_reply(reply, atom, bond)

    def schema_gradient(self, atom, charge, bond):
        value, *arrays = self._request("schema_gradient", atom=atom.tolist(), charge=charge.tolist(),
                                       bond=bond.tolist())
        gradients = tuple(np.asarray(x, np.float32) for x in arrays)
        if (not np.isfinite(value) or len(gradients) != 3 or
                any(d.shape != x.shape or not np.isfinite(d).all()
                    for d, x in zip(gradients, (atom, charge, bond)))):
            raise ValueError("Invalid Nesso affinity value or full-schema gradient")
        return float(value), gradients

    def affinity_confidence_gradients(self, atom, bond):
        affinity, binder = self._request("affinity_confidence_gradients", **self._graph(atom, bond))
        return self._gradient_reply(affinity, atom, bond), self._gradient_reply(binder, atom, bond)

    def weighted_affinity_confidence_gradient(self, atom, bond, affinity_weight, confidence_weight):
        reply = self._request("weighted_affinity_confidence_gradient", affinity_weight=float(affinity_weight),
                              confidence_weight=float(confidence_weight), **self._graph(atom, bond))
        affinity, binder, da, db = reply
        if not np.isfinite(binder):
            raise ValueError("Nesso worker returned a nonfinite binder loss")
        affinity, ga, gb = self._gradient_reply((affinity, da, db), atom, bond)
        return affinity, float(binder), ga, gb

    def hard_value(self, atom, bond):
        return self._request("hard_value", **self._graph(atom, bond))

    def fresh(self, smiles):
        return self._request("fresh", smiles=smiles)

    def _stop(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process.wait()

    def close(self):
        if self.audit is not None:
            return self.audit
        try:
            report = self._request("close")
            code = self.process.wait(timeout=60)
            if code:
                raise RuntimeError(f"Nesso worker exited {code}. See {self.log_path}")
            self.audit = report
        finally:
            self._stop()
        return self.audit

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self._stop()
        else:
            self.close()
