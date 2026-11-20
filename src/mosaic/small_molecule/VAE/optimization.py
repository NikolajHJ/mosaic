"""Update a latent using the existing scorer's atom and bond probability gradients."""

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from mosaic.small_molecule.core.state import RelaxedGraph, validate_graph
from .provider import load_provider


class LatentOptimization:
    """One frozen model, one latent vector, and a fixed set of occupied slots."""

    @classmethod
    def create(cls, general, general_path, n_atoms, *, saved=None, smiles=None):
        options = general.get("vae", {})
        if not isinstance(options, dict) or options.keys() - {"checkpoint", "temperature"}:
            raise ValueError("vae accepts checkpoint and temperature.")
        checkpoint = options.get("checkpoint")
        if checkpoint is not None:
            checkpoint = (Path(general_path).parent / Path(checkpoint).expanduser()).resolve()
        provider = load_provider(checkpoint)
        if type(n_atoms) is not int or not 1 <= n_atoms <= provider.capacity:
            raise ValueError(f"n_atoms must be between 1 and {provider.capacity} for this VAE.")
        temperature = float(options.get("temperature", 1.0))
        rate = float(general["learning_rate"])
        if not all(math.isfinite(v) and v > 0 for v in (temperature, rate)):
            raise ValueError("VAE temperature and latent learning_rate must be finite and positive.")
        identity = {"provider": provider.identity, "temperature": temperature, "n_atoms": n_atoms}
        if saved is None:
            init = general.get("initialization", {})
            smiles = smiles or init.get("smiles")
            if init.get("input") and not smiles:
                raise ValueError("VAE initialization needs a SMILES string for a prepared input.")
            z, slots = provider.initialize(n_atoms, general.get("seed", 17), smiles=smiles)
        else:
            if not {"vae_z", "vae_slots", "vae_identity", "X", "E_upper"}.issubset(saved.keys()):
                raise ValueError("Resuming VAE optimization requires a saved latent checkpoint.")
            if json.loads(saved["vae_identity"].item()) != identity:
                raise ValueError("Starting checkpoint uses a different VAE, temperature or atom count.")
            z = torch.from_numpy(saved["vae_z"])
            slots = torch.from_numpy(saved["vae_slots"])
        if (z.shape != (1, provider.latent_dim) or z.dtype != torch.float32
                or not torch.isfinite(z).all() or slots.dtype != torch.int64
                or slots.shape != (n_atoms,) or len(torch.unique(slots)) != n_atoms
                or (slots < 0).any() or (slots >= provider.capacity).any()):
            raise ValueError("Invalid VAE latent or occupied slot indices.")
        session = cls(provider, z, slots, identity, rate)
        if saved is not None:
            for field in ("X", "E_upper"):
                graph_value = np.asarray(getattr(session.graph(), field))
                if graph_value.shape != saved[field].shape or not np.allclose(
                    graph_value, saved[field], rtol=2e-5, atol=1e-6
                ):
                    raise ValueError("The decoded latent does not reproduce the saved graph.")
        return session

    def __init__(self, provider, z, slots, identity, rate):
        self.provider = provider
        self.slots = slots.detach().clone()
        self.identity = identity
        self.learning_rate = rate
        self.last_gradient_norm = None
        self._decode(z)

    def _decode(self, z):
        self.z = z.detach().requires_grad_(True)
        self._outputs = self.provider.decode(self.z, self.slots, temperature=self.identity["temperature"])
        x, edges = self._outputs
        if any(t.numel() and not t.requires_grad for t in self._outputs):
            raise ValueError("VAE decoder detached the required atom or bond probability path.")
        self._graph = RelaxedGraph(jnp.asarray(x.detach().numpy()), jnp.asarray(edges.detach().numpy()))
        validate_graph(self._graph)

    @property
    def optimizer_record(self):
        return {"name": "latent_gradient", "learning_rate": self.learning_rate, "vae": self.identity}

    def arrays(self):
        return {"vae_z": self.z.detach().numpy(), "vae_slots": self.slots.numpy(),
                "vae_identity": json.dumps(self.identity, sort_keys=True)}

    def graph(self):
        return self._graph

    def step(self, evaluation):
        if evaluation.graph is not self._graph:
            raise ValueError("VAE update requires the evaluation of the current decoded graph.")
        # JAX differentiated the scorer with respect to its two probability arrays.
        # Give those exact cotangents to the same Torch forward that produced them.
        active_outputs = []
        cotangents = []
        for output, gradient in zip(self._outputs, (evaluation.gradients.X, evaluation.gradients.E_upper)):
            if output.numel():
                active_outputs.append(output)
                cotangents.append(torch.tensor(np.asarray(gradient), dtype=output.dtype))
        grad, = torch.autograd.grad(active_outputs, self.z, grad_outputs=cotangents)
        proposed = self.z.detach() - self.learning_rate * grad
        if not torch.isfinite(grad).all() or not torch.isfinite(proposed).all():
            raise FloatingPointError("Nonfinite VAE latent gradient or update.")
        self.last_gradient_norm = float(grad.double().norm())
        self._decode(proposed)
        return self.graph()
