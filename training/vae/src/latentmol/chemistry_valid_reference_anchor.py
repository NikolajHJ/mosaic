"""Frozen original-epoch36 target decoder, separate from reference35 occupancy.

This wrapper is deliberately not an nn.Module child of the trained model. Its
parameters never enter the optimizer, checkpoint weights or module-mode walk.
Only the exact current fresh prior coordinates are decoded; no panel points or
additional random draws are introduced. Native boundaries verify its full state.
"""
from copy import deepcopy
import hashlib
import re


START_SHA = "3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f"


def validate_anchor_identity(value, *, expected_model_sha=None):
    """Metadata-safe exact binding to the frozen original36 tensor identity."""
    if (type(value) is not dict or set(value)!={"reference_epoch","reference_sha256","model_sha256","frozen","eval_mode"}
            or type(value.get("reference_epoch")) is not int or value["reference_epoch"]!=36
            or value.get("reference_sha256")!=START_SHA
            or type(value.get("model_sha256")) is not str or re.fullmatch(r"[0-9a-f]{64}",value["model_sha256"]) is None
            or value.get("frozen") is not True or value.get("eval_mode") is not True
            or (expected_model_sha is not None and value["model_sha256"]!=expected_model_sha)):
        raise ValueError("Frozen valid-reference identity lost original36 weights or eval/frozen scope")
    return dict(value)


def latent_identity(z):
    """Bind the actual same-z tensor, including its native dtype and shape."""
    value = z.detach().cpu().contiguous()
    h = hashlib.sha256(f"{value.dtype}|{tuple(value.shape)}|".encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


class FrozenValidReferenceAnchor:
    def __init__(self, model, *, reference_epoch, checkpoint_sha256):
        from .train import state_digest
        if reference_epoch != 36 or checkpoint_sha256 != START_SHA:
            raise ValueError("The valid-target anchor must be authenticated original epoch36")
        old_anchor=getattr(model,"_valid_reference_anchor",None)
        self._model = deepcopy(model,{id(old_anchor):None} if old_anchor is not None else {}).eval().requires_grad_(False)
        # A resumed source model could already carry gradients; they are never
        # a property of the frozen target decoder.
        for parameter in self._model.parameters():
            parameter.grad = None
        self._sha = state_digest(self._model.state_dict())
        self._checkpoint_sha = checkpoint_sha256
        self.assert_unchanged()

    @property
    def reference_identity(self):
        return dict(epoch=36,checkpoint_sha256=self._checkpoint_sha,model_sha256=self._sha)

    def assert_unchanged(self):
        from .train import state_digest
        if (any(module.training for module in self._model.modules())
                or any(p.requires_grad or p.grad is not None for p in self._model.parameters())
                or state_digest(self._model.state_dict()) != self._sha):
            raise ValueError("Frozen original36 anchor weights, gradients or eval modes changed")
        return validate_anchor_identity(dict(reference_epoch=36,reference_sha256=self._checkpoint_sha,
                    model_sha256=self._sha,frozen=True,eval_mode=True))

    def decode(self, z):
        import torch
        if (any(module.training for module in self._model.modules())
                or any(p.requires_grad or p.grad is not None for p in self._model.parameters())):
            raise ValueError("Valid-reference decoding requires a frozen eval-only model")
        with torch.no_grad():
            return self._model.decode(z)

    def prepare(self, teacher, z, *, candidate_ids, update, overlap):
        """Classify frozen labels only; the ordinary live repair remains separate."""
        decoded = self.decode(z)
        method = (teacher.build_valid_reference_targets_async if overlap
                  else teacher.build_valid_reference_targets)
        return method(decoded,candidate_ids=candidate_ids,update=update,
                      reference_identity=self.reference_identity,latent_sha256=latent_identity(z))
