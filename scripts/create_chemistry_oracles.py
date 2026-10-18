"""Allocated CPU: emit values/VJPs from the pinned, unmodified Torch source.

Run using the existing DBMol Torch environment. The production Mosaic path
needs no Torch installation. Inputs, configurations, routes and outputs are
saved so JAX tests can replay the exact fixtures in a separate process.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Reference numerical execution requires allocated DTU CPU")
    manifest = json.loads(args.manifest.read_text())
    expected = {
        name: entry["source_sha256"] for name, entry in manifest["files"].items()
    }
    expected.update(manifest["oracle_source_sha256"])
    for name, digest in expected.items():
        with (args.source / "src/latentmol" / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Original oracle source identity differs: " + name)
    sys.path.insert(0, str(args.source / "src"))
    import torch
    from latentmol.atom_preserving_chemistry import (
        AtomPreservingRepairTeacher,
        composition_statistics,
        population_js,
    )
    from latentmol.probability_repair import ProbabilityRepairConfig
    from latentmol.repair_failure_guidance import (
        prepare_failure_guidance,
        failure_aware_repair_loss,
    )
    from latentmol.repair_progress import ProgressSearchConfig
    from latentmol.representation import GraphSpec, DecoderOutput
    import latentmol.repair_failure_guidance as original_module

    if (
        Path(original_module.__file__).resolve()
        != (args.source / "src/latentmol/repair_failure_guidance.py").resolve()
    ):
        raise ValueError("Oracle imported a different latentmol package")

    def make(elements, edges, *, n=None, confidence=3.0, unsupported=False):
        spec = GraphSpec(max_nodes=n or len(elements))
        a = torch.full(
            (1, spec.max_nodes, len(spec.atomic_numbers)),
            -confidence,
            dtype=torch.float64,
        )
        q = torch.full(
            (1, spec.max_nodes, len(spec.formal_charges)),
            -confidence,
            dtype=torch.float64,
        )
        b = torch.full(
            (1, spec.max_nodes, spec.max_nodes, len(spec.bond_names)),
            -confidence,
            dtype=torch.float64,
        )
        a[..., 0] = confidence
        q[..., spec.formal_charges.index(-2 if unsupported else 0)] = confidence
        b[..., 0] = confidence
        for i, z in enumerate(elements):
            a[0, i, 0] = -confidence
            a[0, i, spec.atomic_numbers.index(z)] = confidence
        for i, j, category in edges:
            b[0, i, j, 0] = b[0, j, i, 0] = -confidence
            b[0, i, j, spec.bond_names.index(category)] = b[
                0, j, i, spec.bond_names.index(category)
            ] = confidence
        return spec, DecoderOutput(a, q, b)

    one = [(0, 1, "TRIPLE")]
    two = one + [(2, 3, "TRIPLE")]
    base = dict(
        max_edits=1,
        checks_per_depth=1,
        beam_width=4,
        probability_weight=1.0,
        probability_floor=1e-12,
    )
    cases = [
        ("valid_full_nll", make([9, 9], one), {**base, "checks_per_depth": 32}, None),
        (
            "padding_full_nll",
            make([9, 9], one, n=3),
            {**base, "checks_per_depth": 32},
            None,
        ),
        ("already_valid", make([6, 6], [(0, 1, "SINGLE")]), base, None),
        ("local_bond_risk", make([9, 9], one), base, None),
        ("extreme_local", make([9, 9], one, confidence=1000.0), base, None),
        (
            "extra_valid_nll",
            make([9, 9], one),
            base,
            dict(max_edits=1, checks_per_depth=128),
        ),
        (
            "partial_edited_nll",
            make([9, 9, 9, 9], two),
            base,
            dict(max_edits=1, checks_per_depth=128),
        ),
        ("component_cut_risk", make([9, 9, 9, 9], two), base, None),
        ("nonempty_risk", make([], [], n=3, confidence=1000.0), base, None),
        ("unsupported", make([9, 9, 9], [], unsupported=True), base, None),
    ]
    # Check all-attempt normalization, including an already-valid zero row.
    spec, failed = make([9, 9], one)
    _, valid = make([6, 6], [(0, 1, "SINGLE")])
    mixed = DecoderOutput(
        *(
            torch.cat((x, y))
            for x, y in zip(
                (failed.atoms, failed.charges, failed.bonds),
                (valid.atoms, valid.charges, valid.bonds),
            )
        )
    )
    cases.append(("mixed_all_attempts", (spec, mixed), base, None))
    spec, asymmetric = make([9, 9], one)
    asymmetric.bonds[0, 0, 1, spec.bond_names.index("DOUBLE")] += 0.7
    cases.append(
        (
            "asymmetric_logits",
            (spec, asymmetric),
            {**base, "checks_per_depth": 32},
            None,
        )
    )
    records = []
    keys = (
        "total",
        "valid_full_nll",
        "extra_valid_nll",
        "partial_edited_nll",
        "local_bond_risk",
        "component_cut_risk",
        "nonempty_risk",
    )
    for label, (spec, output), search, progress in cases:
        output = DecoderOutput(
            *(
                v.detach().requires_grad_()
                for v in (output.atoms, output.charges, output.bonds)
            )
        )
        ids = list(range(len(output.atoms)))
        with AtomPreservingRepairTeacher(
            spec, ProbabilityRepairConfig(**search)
        ) as teacher:
            targets = teacher.build_targets(output, candidate_ids=ids, update=0)
        prepared = prepare_failure_guidance(
            output,
            targets,
            spec,
            progress_search=None
            if progress is None
            else ProgressSearchConfig(**progress),
        )
        result = failure_aware_repair_loss(output, prepared, spec)
        gradients = {}
        constant_components = []
        for key in keys:
            if result[key].requires_grad:
                grads = torch.autograd.grad(
                    result[key],
                    (output.atoms, output.charges, output.bonds),
                    retain_graph=True,
                    allow_unused=True,
                )
            else:
                # The original source returns detached zeros for inactive
                # routes. allow_unused covers unused inputs, not a constant
                # output; its mathematical VJP is zero for every input.
                grads = (None, None, None)
                constant_components.append(key)
            gradients[key] = [
                (torch.zeros_like(v) if g is None else g).detach().tolist()
                for v, g in zip((output.atoms, output.charges, output.bonds), grads)
            ]
        stats = composition_statistics(output, spec)
        counts = stats.elements.detach().clone().requires_grad_()
        law = torch.arange(1, len(spec.atomic_numbers), dtype=torch.float64)
        js = population_js(counts, law)
        (js_gradient,) = torch.autograd.grad(js, counts)
        records.append(
            {
                "label": label,
                "spec": spec.to_dict(),
                "logits": [
                    v.detach().tolist()
                    for v in (output.atoms, output.charges, output.bonds)
                ],
                "search": search,
                "progress_search": progress,
                "values": {k: float(result[k].detach()) for k in keys},
                "gradients": gradients,
                "constant_components": constant_components,
                "composition": {
                    "elements": stats.elements.detach().tolist(),
                    "charges": stats.charges.detach().tolist(),
                    "occupied_mass": float(stats.occupied_mass.detach()),
                    "law": law.tolist(),
                    "js": float(js.detach()),
                    "js_gradient": js_gradient.tolist(),
                },
                "original_statuses": [v["status"] for v in targets.receipt["attempts"]],
                "witness_kinds": [
                    None if v is None or v.witness is None else v.witness.kind
                    for v in prepared.rows
                ],
            }
        )
    output = {
        "version": "dbmol-chemistry-oracles-v1",
        "job_id": os.environ["LSB_JOBID"],
        "source_sha256": expected,
        "torch": torch.__version__,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, allow_nan=False)
    print(f"Wrote {len(records)} original-source value/VJP fixtures to {args.output}")


if __name__ == "__main__":
    main()
