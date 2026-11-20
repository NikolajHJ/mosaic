"""Export a DBMol graph VAE to Mosaic's portable, differentiable inference format.

Run in the environment that supplied the original latentmol checkpoint.
Optimization with the resulting archive requires only PyTorch and Mosaic.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from latentmol.evaluation.evaluate import load_model
from latentmol.representation import HardGraphBatch


class Runtime(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.spec = model.spec

    def encode(self, atoms, charges, bonds):
        graph = HardGraphBatch(atoms, charges, bonds)
        if self.encoder.pooling == "canonical":
            # The caller has already converted the input to canonical SMILES order.
            return self.encoder(graph, from_canonical_cache=True)[0]
        return self.encoder(graph.to_relaxed(self.spec))[0]

    def decode(self, z):
        output = self.decoder(z)
        return output.atoms, output.symmetric_bond_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model, original = load_model(args.checkpoint, "cpu")
    runtime = Runtime(model).eval().requires_grad_(False)
    dims, n = original["model_config"]["latent"], model.spec.max_nodes
    atoms = torch.ones((1, n), dtype=torch.long)
    charges = torch.full_like(atoms, model.spec.formal_charges.index(0))
    bonds = torch.zeros((1, n, n), dtype=torch.long)
    example = torch.zeros((1, dims))
    traced = torch.jit.trace_module(runtime, {"encode": (atoms, charges, bonds), "decode": (example,)})
    metadata = {
        "format": "mosaic-graph-vae-v1",
        "latent_dim": dims,
        "max_nodes": n,
        "atomic_numbers": model.spec.atomic_numbers,
        "formal_charges": model.spec.formal_charges,
        "bond_names": model.spec.bond_names,
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    with args.output.open("xb") as stream:
        torch.jit.save(traced, stream, _extra_files={"vae.json": json.dumps(metadata)})


if __name__ == "__main__":
    main()
