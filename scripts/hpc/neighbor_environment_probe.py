"""Allocated-only CPU gradient/coverage/parity observations; never train models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_probe(output, data, checkpoint):
    import numpy as np
    import torch
    from latentmol.chemical_environment import prepare_neighbor_environment, neighbor_environment_loss
    from latentmol.chemical_environment_policy import assess_environment
    from latentmol.data import GraphDataset
    from latentmol.evaluate import load_model
    from latentmol.representation import DecoderOutput, GraphSpec
    from latentmol.train import state_digest
    from test_chemical_environment import labels_from_smiles, output_from_labels, POLICY

    torch.set_num_threads(4)
    cases, parity = [], []
    for index, smiles in enumerate(("CS(=O)(=O)N", "CS(C)(C)(C)(C)C", "S(F)(F)(F)(F)(F)F",
                                    "C[N+](=O)[O-]", "P(C)(C)(C)(C)C", "c1ccsc1")):
        spec, a, q, b = labels_from_smiles(smiles)
        base = output_from_labels(spec, a, q, b, strength=6.)
        # Separate deterministic LIVE distribution from the original context.
        rng = np.random.default_rng(701 + index)
        live_bonds = rng.normal(size=tuple(base.bonds.shape)).tolist()
        live = DecoderOutput(base.atoms, base.charges,
                             torch.tensor(live_bonds, dtype=torch.double, requires_grad=True))
        prepared = prepare_neighbor_environment(base, spec, POLICY)
        result = neighbor_environment_loss(live, prepared, spec)
        gradient, = torch.autograd.grad(result["loss"], (live.bonds,))
        cases.append(dict(smiles=smiles, spec=spec.to_dict(), atoms=base.atoms.detach().tolist(),
            charges=base.charges.detach().tolist(), context_bonds=base.bonds.detach().tolist(),
            live_bonds=live_bonds, policy=POLICY))
        parity.append(dict(loss=float(result["loss"]), gradient=gradient.tolist(),
                           assessments=prepared.context["assessments"]))
    write(output / "parity-inputs.json", cases)
    write(output / "torch-parity.json", parity)

    # Integrity-checked validation split only. The result is a bounded corpus
    # audit, never proof that every legitimate chemical family is retained.
    started = time.monotonic()
    if sha(Path(data) / "manifest.json") != "f246cc93805b73a557cd0c09a78891ac8410c465192db615ce0b6709218c8c0e":
        raise ValueError("Validation corpus manifest differs from the authenticated training corpus")
    dataset = GraphDataset(data, "val")
    rng = np.random.default_rng(145091)
    selected = rng.choice(len(dataset), min(10000, len(dataset)), replace=False)
    counts, element_rows, exclusions = Counter(), Counter(), []
    for index in selected.tolist():
        row = dataset[index]
        fields = [row[name].tolist() for name in ("atoms", "charges", "bonds")]
        assessment = assess_environment(*[[value] for value in fields], dataset.spec)[0]
        counts["attempts"] += 1
        counts["local_valence_ok"] += int(assessment["local_valence_ok"])
        counts["within_common_organic_scope"] += int(assessment["within_common_organic_scope"])
        counts["fully_assessed"] += int(assessment["fully_assessed"])
        elements = {dataset.spec.atomic_numbers[value] for value in fields[0] if value}
        for element in elements:
            element_rows[str(element)] += 1
        if not assessment["within_common_organic_scope"] and len(exclusions) < 32:
            exclusions.append(dict(index=index, smiles=dataset.smiles[index], assessment=assessment))
    accepted = counts["attempts"] - counts["within_common_organic_scope"]
    coverage_passed = (counts["local_valence_ok"] == counts["attempts"]
        and counts["fully_assessed"] == counts["attempts"]
        and accepted <= counts["attempts"] * .001)
    write(output / "corpus-audit.json", dict(counts=counts, element_molecule_counts=element_rows,
        excluded_examples=exclusions, selection="10000 validation rows sampled without replacement, seed145091",
        manifest_sha256=sha(Path(data) / "manifest.json"), elapsed_seconds=time.monotonic()-started,
        common_organic_coverage_gate_passed=coverage_passed,
        coverage_gate="all local valence states assessed/allowed; scope exclusions at most0.1percent",
        limitation="Existing curated validation molecules; no new model or full chemical-space guarantee"))
    if not coverage_passed:
        raise ValueError("Common-organic corpus coverage gate failed; preserve excluded examples")

    # Actual trained, frozen decoder: retain exact model and RNG state and
    # measure its real latent VJP, rather than only independent graph logits.
    expected = "31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e"
    if sha(checkpoint) != expected:
        raise ValueError("Frozen epoch40 checkpoint identity differs")
    model, payload = load_model(checkpoint, "cpu")
    if payload["epoch"] != 40:
        raise ValueError("Expected preserved completed epoch40")
    model.requires_grad_(False)
    before = state_digest(model.state_dict())
    del payload
    z = torch.randn((2, model.config.latent), generator=torch.Generator().manual_seed(145092),
                    requires_grad=True)
    started = time.monotonic()
    decoded = model.decode(z)
    prepared = prepare_neighbor_environment(decoded, model.spec, POLICY)
    result = neighbor_environment_loss(decoded, prepared, model.spec)
    gradient, = torch.autograd.grad(result["loss"], (z,))
    if not torch.isfinite(gradient).all() or not bool(gradient.norm() > 0):
        raise ValueError("Actual frozen VAE latent gradient is nonfinite or absent")
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Frozen VAE changed during latent VJP")
    write(output / "frozen-vae-vjp.json", dict(checkpoint_sha256=expected, model_sha256=before,
        loss=float(result["loss"]), gradient_norm=float(gradient.norm()),
        per_latent_gradient_norm=gradient.norm(dim=-1).tolist(), elapsed_seconds=time.monotonic()-started,
        model_unchanged=True, parameter_gradients_absent=True, context=prepared.receipt,
        scope="Actual frozen epoch40 decoder latent VJP; not training or binding efficacy"))


def jax_probe(output):
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.chemistry._reference.representation import GraphSpec
    from mosaic.small_molecule.chemistry.state import DecoderLogits
    from mosaic.small_molecule.chemistry.environment import prepare_neighbor_environment, neighbor_environment_loss
    if not jax.config.x64_enabled:
        raise ValueError("Cross-framework probe requires JAX FP64")
    cases = json.loads((output / "parity-inputs.json").read_text())
    rows = []
    for case in cases:
        spec = GraphSpec(**{key: tuple(value) if isinstance(value, list) else value
                            for key, value in case["spec"].items()})
        a, q, b, live = [jnp.asarray(case[key], dtype=jnp.float64)
                         for key in ("atoms", "charges", "context_bonds", "live_bonds")]
        prepared = prepare_neighbor_environment(DecoderLogits(a, q, b), spec, case["policy"])
        def value(bonds):
            return neighbor_environment_loss(DecoderLogits(a, q, bonds), prepared, spec)["loss"]
        loss, gradient = jax.value_and_grad(value)(live)
        rows.append(dict(loss=float(loss), gradient=gradient.tolist(),
                         assessments=prepared.context["assessments"]))
    write(output / "jax-parity.json", rows)


def main():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical probes require an allocated LSF job")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("torch", "jax"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.phase == "torch":
        torch_probe(args.output, args.data, args.checkpoint)
    else:
        jax_probe(args.output)


if __name__ == "__main__":
    main()
