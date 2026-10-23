"""Independent, cheap affine bridge checks at the actual shared molecule.

No Boltz model, diffusion, molecule conversion or repair worker runs here.
The two prescribed directions preserve category sums, fixed occupancy, bond
symmetry and masked pairs. Their FP64 finite differences extend the adapter
affinely: endpoint arrays are NOT claimed to be sampled or valid molecules.
Manual value/JVP/VJP formulae avoid the production one-hot/index-copy mapping.
All numerical use and fixtures belong on allocated DTU HPC.
"""
import torch

from latentmol.representation import RelaxedGraph
from latentmol.multimodel.study import tensor_digest
from .adapter import adapt_graph

VERSION = "fixed_graph_native_affine_bridge_v1"
CHANNELS = ("ref_element", "ref_charge", "token_bonds", "bond_types")
FIELDS = ("atoms", "charges", "bonds")
RTOL, ATOL = 1e-8, 1e-10
EPSILONS = (1e-4, .001)
# Native enum is independently declared; changing the production dictionary
# cannot silently change the oracle used by this bridge check.
BOND_IDS = {"NONE": 0, "SINGLE": 2, "DOUBLE": 3, "TRIPLE": 4, "AROMATIC": 5}


def witness(value):
    raw = value.detach()
    return {"shape": list(raw.shape), "dtype": str(raw.dtype),
            "sha256": tensor_digest({"value": raw}), "finite": bool(raw.isfinite().all()),
            "max_abs": float(raw.double().abs().max()) if raw.numel() else 0.}


def comparison(actual, expected, *, rtol=RTOL, atol=ATOL):
    """Retain scale-aware error receipts without exporting whole tensors."""
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError("Bridge comparison needs equal nonempty shapes")
    difference = (actual.double() - expected.double()).abs()
    denominator = atol + rtol * expected.double().abs()
    ratio = float((difference / denominator).max())
    left, right = witness(actual), witness(expected)
    return {"actual": left, "expected": right, "max_abs_error": float(difference.max()),
            "tolerance_max": atol + rtol * right["max_abs"], "maximum_normalized_error": ratio,
            "passed": left["finite"] and right["finite"] and ratio <= 1.}


def _channels(graph, prepared):
    features, bonds = adapt_graph(graph, prepared)
    return {**{key: features[key] for key in CHANNELS[:-1]}, "bond_types": bonds}, features


def manual_values(graph, prepared, *, tangent=False):
    """Explicit native row/category correspondence, including NONE embedding."""
    f, spec = prepared.features, prepared.spec
    result = {key: torch.zeros_like(f[key], dtype=torch.float64) if tangent
              else f[key].detach().double().clone() for key in CHANNELS[:-1]}
    types = f["type_bonds"]
    native = torch.zeros((*types.shape, 7), device=types.device, dtype=torch.float64)
    if not tangent:
        for index in range(7): native[..., index] = (types == index).double()
    slots = prepared.active.nonzero().flatten().tolist()
    for slot in slots:
        atom = int(prepared.slot_to_atom[slot])
        result["ref_element"][0, atom] = 0
        for index, atomic_number in enumerate(spec.atomic_numbers):
            result["ref_element"][0, atom, atomic_number] = graph.atoms[0, slot, index]
        result["ref_charge"][0, atom] = sum(graph.charges[0, slot, index] * charge
            for index, charge in enumerate(spec.formal_charges))
    for left in slots:
        for right in slots:
            i, j = int(prepared.slot_to_token[left]), int(prepared.slot_to_token[right])
            result["token_bonds"][0, i, j, 0] = -graph.bonds[0, left, right, 0] if tangent else 1 - graph.bonds[0, left, right, 0]
            native[0, i, j] = 0
            for index, name in enumerate(spec.bond_names):
                native[0, i, j, BOND_IDS[name]] = graph.bonds[0, left, right, index]
    return {**result, "bond_types": native}


def tangent_plan(graph, prepared):
    """Two deterministic, predeclared, jointly normalized graph directions."""
    spec, active = prepared.spec, prepared.active
    if len(spec.atomic_numbers) < 3 or len(spec.formal_charges) < 2 or len(spec.bond_names) < 2:
        raise ValueError("Bridge audit requires two occupied element/charge/bond categories")
    slots = active.nonzero().flatten().tolist()
    result = {}
    for which in range(2):
        direction = RelaxedGraph(*(torch.zeros_like(getattr(graph, key), dtype=torch.float64) for key in FIELDS))
        for slot in slots:
            first = 1 + (slot + which) % (len(spec.atomic_numbers) - 1)
            second = 1 + (slot + which + 1) % (len(spec.atomic_numbers) - 1)
            direction.atoms[0, slot, first], direction.atoms[0, slot, second] = 1., -1.
            first, second = (slot + which) % len(spec.formal_charges), (slot + which + 1) % len(spec.formal_charges)
            direction.charges[0, slot, first], direction.charges[0, slot, second] = 1., -1.
        for index, left in enumerate(slots):
            for right in slots[index + 1:]:
                category = 1 + (left + right + which) % (len(spec.bond_names) - 1)
                for i, j in ((left, right), (right, left)):
                    direction.bonds[0, i, j, 0], direction.bonds[0, i, j, category] = -1., 1.
        norm = sum(getattr(direction, key).square().sum() for key in FIELDS).sqrt()
        result[f"affine_{which + 1}"] = RelaxedGraph(*(getattr(direction, key) / norm for key in FIELDS))
    if tensor_digest({key: getattr(result["affine_1"], key) for key in FIELDS}) == tensor_digest(
            {key: getattr(result["affine_2"], key) for key in FIELDS}):
        raise ValueError("Declared bridge directions must be distinct")
    return result


def manual_vjp(graph, prepared, cotangents):
    """Transpose the bridge analytically; include raw NO_ATOM occupancy term.

    The FD directions keep NO_ATOM fixed. This full VJP additionally verifies
    the occupancy factor in expected_charge, before restricting that channel.
    """
    result = {key: torch.zeros_like(getattr(graph, key), dtype=torch.float64) for key in FIELDS}
    spec, slots = prepared.spec, prepared.active.nonzero().flatten().tolist()
    for slot in slots:
        atom = int(prepared.slot_to_atom[slot])
        for index, atomic_number in enumerate(spec.atomic_numbers):
            result["atoms"][0, slot, index] += cotangents["ref_element"][0, atom, atomic_number]
        mean_charge = sum(graph.charges[0, slot, index] * charge for index, charge in enumerate(spec.formal_charges))
        result["atoms"][0, slot, 0] -= mean_charge * cotangents["ref_charge"][0, atom]
        occupancy = 1 - graph.atoms[0, slot, 0]
        for index, charge in enumerate(spec.formal_charges):
            result["charges"][0, slot, index] = occupancy * charge * cotangents["ref_charge"][0, atom]
    for left in slots:
        for right in slots:
            i, j = int(prepared.slot_to_token[left]), int(prepared.slot_to_token[right])
            for index, name in enumerate(spec.bond_names):
                result["bonds"][0, left, right, index] = cotangents["bond_types"][0, i, j, BOND_IDS[name]]
            result["bonds"][0, left, right, 0] -= cotangents["token_bonds"][0, i, j, 0]
    return result


def validate_bridge(graph, prepared, *, context_sha256, parity_rtol, parity_atol):
    """Audit actual production mapping plus independent affine double extension."""
    before = tensor_digest({**{"feature:" + key: value for key, value in prepared.features.items()},
        "coordinates": prepared.coordinates, "slot_to_atom": prepared.slot_to_atom, "slot_to_token": prepared.slot_to_token})
    original_digest = tensor_digest({key: getattr(graph, key) for key in FIELDS})
    wide = RelaxedGraph(*(getattr(graph, key).detach().double() for key in FIELDS))
    if not torch.equal(wide.occupancy[0], prepared.active.to(wide.atoms)):
        raise ValueError("Affine cut requires exact current fixed occupancy")
    with torch.no_grad():
        native, native_features = _channels(graph, prepared)
        actual, features = _channels(wide, prepared)
        expected = manual_values(wide, prepared)
    native_rows = {key: comparison(native[key], expected[key], rtol=parity_rtol, atol=parity_atol) for key in CHANNELS}
    wide_rows = {key: comparison(actual[key], expected[key]) for key in CHANNELS}
    auxiliary = all(features[key] is value and native_features[key] is value
                    for key, value in prepared.features.items() if key not in CHANNELS[:-1])
    report = {"version": VERSION, "passed": False, "graph_sha256": original_digest, "context_sha256": context_sha256,
        "native_mapping_parity": {"rtol": parity_rtol, "atol": parity_atol, "rows": native_rows,
                                  "passed": all(row["passed"] for row in native_rows.values())},
        "float64_value_parity": {"rtol": RTOL, "atol": ATOL, "rows": wide_rows,
                                  "passed": all(row["passed"] for row in wide_rows.values())},
        "auxiliary_unchanged": auxiliary, "context_unchanged": False, "directions": [],
        "affine_extension_not_molecule_samples": True, "scorer_calls": 0}
    directions = tangent_plan(wide, prepared)
    for label, direction in directions.items():
        identity = tensor_digest({key: getattr(direction, key) for key in FIELDS})
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            dual = RelaxedGraph(*(torch.autograd.forward_ad.make_dual(getattr(wide, key), getattr(direction, key)) for key in FIELDS))
            output, _ = _channels(dual, prepared)
            observed = {}
            for key, value in output.items():
                primal, tangent = torch.autograd.forward_ad.unpack_dual(value)
                if tangent is None or not torch.equal(primal, actual[key]):
                    raise ValueError("Bridge tangent missing or native primal changed: " + key)
                observed[key] = tangent.detach().clone()
        manual = manual_values(direction, prepared, tangent=True)
        jvp_rows = {key: comparison(observed[key], manual[key]) for key in CHANNELS}
        with torch.enable_grad():
            probe = RelaxedGraph(*(getattr(wide, key).clone().requires_grad_(True) for key in FIELDS))
            output, _ = _channels(probe, prepared)
            # Fixed arithmetic pattern, independent of observed AD or FD outcomes.
            cotangents = {key: ((torch.arange(value.numel(), device=value.device, dtype=torch.float64).reshape(value.shape)
                                + (1 if label == "affine_1" else 7)) % 17 - 8) / 17 for key, value in output.items()}
            scalar = sum((output[key] * cotangents[key]).sum() for key in CHANNELS)
            gradients = torch.autograd.grad(scalar, tuple(getattr(probe, key) for key in FIELDS))
        expected_gradients = manual_vjp(wide, prepared, cotangents)
        vjp_rows = {key: comparison(gradient, expected_gradients[key]) for key, gradient in zip(FIELDS, gradients)}
        row = {"name": label, "direction_sha256": identity, "passed": False,
            "jvp": {"rtol": RTOL, "atol": ATOL, "rows": jvp_rows, "passed": all(v["passed"] for v in jvp_rows.values())},
            "vjp": {"rtol": RTOL, "atol": ATOL, "rows": vjp_rows, "passed": all(v["passed"] for v in vjp_rows.values())},
            "finite_differences": []}
        for epsilon in EPSILONS:
            with torch.no_grad():
                plus, _ = _channels(RelaxedGraph(*(getattr(wide, key) + epsilon * getattr(direction, key) for key in FIELDS)), prepared)
                minus, _ = _channels(RelaxedGraph(*(getattr(wide, key) - epsilon * getattr(direction, key) for key in FIELDS)), prepared)
            rows = {key: {**comparison((plus[key] - minus[key]) / (2 * epsilon), observed[key]),
                          "analytic_sha256": witness(observed[key])["sha256"],
                          "endpoint_plus": witness(plus[key]), "endpoint_minus": witness(minus[key])} for key in CHANNELS}
            row["finite_differences"].append({"epsilon": epsilon, "rtol": RTOL, "atol": ATOL,
                "passed": all(value["passed"] for value in rows.values()), "rows": rows})
        row["passed"] = row["jvp"]["passed"] and row["vjp"]["passed"] and all(value["passed"] for value in row["finite_differences"])
        report["directions"].append(row)
        del probe, output, scalar, gradients
    after = tensor_digest({**{"feature:" + key: value for key, value in prepared.features.items()},
        "coordinates": prepared.coordinates, "slot_to_atom": prepared.slot_to_atom, "slot_to_token": prepared.slot_to_token})
    report["context_unchanged"] = before == after
    report["passed"] = (report["native_mapping_parity"]["passed"] and report["float64_value_parity"]["passed"]
        and auxiliary and report["context_unchanged"] and all(row["passed"] for row in report["directions"]))
    return report
