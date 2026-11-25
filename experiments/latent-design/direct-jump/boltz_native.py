"""Frozen Boltz-2 affinity and its derivatives with respect to graph probabilities."""

import os
from pathlib import Path

import numpy as np
import torch
from mosaic.small_molecule.boltz2 import native


CONTEXT = Path("/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared/context.pt")
CONTEXT_SHA256 = "a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093"
CHECKPOINT = Path("/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt")


def check_probabilities(rows, shape):
    if rows.shape != shape or rows.dtype != np.float32:
        raise ValueError(f"Expected float32 probability rows with shape {shape}")
    if (not np.isfinite(rows).all() or rows.min() < -1e-6 or rows.max() > 1.000001
            or not np.allclose(rows.sum(axis=-1), 1, rtol=0, atol=2e-5)):
        raise ValueError("Invalid categorical probability rows")


class BoltzBackend:
    def __init__(self, seed):
        if not os.environ.get("LSB_JOBID") or not torch.cuda.is_available():
            raise RuntimeError("Boltz-2 gradient evaluation requires an allocated CUDA device")
        if native.ATOMIC_NUMBERS != (6, 7, 8, 16, 9, 17) or native.TYPED_BOND_IDS != (0, 2, 3, 4, 5):
            raise ValueError("The native atom or bond vocabulary differs from the comparison")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device("cuda")
        context_path = Path(os.environ.get("DIRECT_JUMP_BOLTZ_CONTEXT", CONTEXT))
        if native.file_digest(context_path) != CONTEXT_SHA256:
            raise ValueError("The prepared 25-slot BRD4 Boltz-2 context differs")
        self.context = torch.load(context_path, map_location=self.device, weights_only=True)
        native.validate_context(self.context)
        if self.context["atom_indices"].numel() != 25:
            raise ValueError("The Boltz-2 context must contain exactly 25 occupied ligand slots")
        checkpoint_path = Path(os.environ.get("DIRECT_JUMP_BOLTZ_CHECKPOINT", CHECKPOINT))
        self.model = native.load_model(checkpoint_path).to(self.device)
        self.frozen_before = self._frozen_identity()
        self.identity = {
            "objective": "affinity_pred_value",
            "pose": "fixed native BRD4 context",
            "context_file_sha256": CONTEXT_SHA256,
            "checkpoint_sha256": native.AFFINITY_SHA256,
            "model_digest": self.frozen_before[0],
            "context_digest": self.frozen_before[1],
            "seed": int(seed),
        }

    def _frozen_identity(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Boltz-2 must remain in evaluation mode with frozen parameters")
        tensors = self.context["features"] | {
            key: value for key, value in self.context.items() if isinstance(value, torch.Tensor)
        }
        return native.tensor_digest(self.model.state_dict()), native.tensor_digest(tensors)

    def gradient(self, atom, bond):
        # Native inject uses torch.triu_indices (lexicographic i < j). One
        # unordered bond row contributes to both directed model entries.
        check_probabilities(atom, (25, 6))
        check_probabilities(bond, (300, 5))
        with torch.enable_grad():
            x = torch.as_tensor(atom, device=self.device).requires_grad_()
            edges = torch.as_tensor(bond, device=self.device).requires_grad_()
            loss = native.score(self.model, self.context, x, edges)[0]
            grad_x, grad_e = torch.autograd.grad(loss, (x, edges), allow_unused=False)
        if not all(torch.isfinite(t).all().item() for t in (loss, grad_x, grad_e)):
            raise FloatingPointError("Nonfinite Boltz-2 affinity or graph-probability gradient")
        return loss.item(), grad_x.detach().cpu().numpy(), grad_e.detach().cpu().numpy()

    def close(self):
        if self.model is None:
            return
        try:
            if self._frozen_identity() != self.frozen_before:
                raise RuntimeError("Frozen Boltz-2 weights or context changed during the run")
            return {"model_digest": self.frozen_before[0], "context_digest": self.frozen_before[1]}
        finally:
            self.model = self.context = None
            torch.cuda.empty_cache()


def make_backend(seed, work: Path):
    return BoltzBackend(seed)
