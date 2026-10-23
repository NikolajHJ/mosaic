"""Independent native-oracle and derivative gates for early embedding injection.

These gates certify a new boundary, never promote an older native pass, and do
not certify a learned mapping. Hard fixed-pose diagnostics do not rebuild native
poses/names or establish binding improvement. All numerical use belongs on DTU
HPC; importing this module does not execute a model or launch an experiment.
"""

from collections.abc import Mapping
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import torch
from torch import Tensor

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.representation import RelaxedGraph
from .adapter import adapt_graph
from .embedding_injection import inject_native_embeddings
from .embedding_training import conditioned_hard
from .pointwise_cuts import tangent_plan
from .provenance import context_digest


NATIVE_GATE_VERSION = "native-embedding-oracle-gate-v1"
DEFAULT_TOLERANCES = {"scalar_rtol": 1e-4, "scalar_atol": 1e-5,
                      "gradient_rtol": .01, "gradient_atol": 1e-5}
_FIELDS = ("atoms", "charges", "bonds")
# Independent from the production enum and target lookup construction.
_NATIVE_BOND_IDS = {"NONE": 0, "SINGLE": 2, "DOUBLE": 3, "TRIPLE": 4, "AROMATIC": 5}


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Native embedding validation internal deadline reached")


def _frozen_versions(**modules):
    versions = []
    for role, module in modules.items():
        if module.training or any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("Embedding validation requires frozen eval modules: " + role)
        versions.extend((role, name, value._version) for name, value in
                        [*module.named_parameters(), *module.named_buffers()])
    return tuple(versions)


def _scalars(scores):
    if not isinstance(scores, Mapping) or "dbmol_total_loss" not in scores:
        raise ValueError("Embedding native gate requires the declared DBMol total loss")
    result = {}
    for key, value in scores.items():
        if not isinstance(value, Tensor) or value.numel() != 1 or not bool(value.isfinite().all()):
            raise ValueError("Native embedding output is not a finite scalar: " + key)
        result[key] = float(value.detach())
    return result


def _comparison(actual, expected, tolerances):
    if set(actual) != set(expected):
        raise ValueError("Native embedding scalar output sets differ")
    rows = {}
    for key, value in actual.items():
        reference = expected[key]
        limit = tolerances["scalar_atol"] + tolerances["scalar_rtol"] * abs(reference)
        rows[key] = dict(actual=value, expected=reference, absolute_error=abs(value - reference),
                         allowed_error=limit, passed=abs(value - reference) <= limit)
    return dict(passed=all(row["passed"] for row in rows.values()), rows=rows)


def graph_embedding_predictions(graph, prepared, bank):
    """Differentiable independent projection composition at the same soft graph.

    No native adapter or hard target constructor is called. Algebraic affine
    extensions outside probability simplices are permitted for FD observations;
    they are explicitly not samples of valid molecules. Native-span matrices are
    cloned authenticated bank buffers; checkpoint provenance is the caller's job.
    """
    spec = prepared.spec
    atom_mask = prepared.active.to(graph.atoms.device)[None]
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    values = {}
    element_rows = torch.tensor(spec.atomic_numbers, device=graph.atoms.device)
    bond_rows = torch.tensor([_NATIVE_BOND_IDS[name] for name in spec.bond_names],
                             device=graph.bonds.device)
    for descriptor in bank.descriptors:
        prototype = bank.projection(descriptor.key)
        if prototype.device != graph.atoms.device:
            raise ValueError("Independent native bank and graph must share a device")
        if descriptor.kind == "atom_element":
            value = graph.atoms @ prototype[element_rows]
            mask = atom_mask
        elif descriptor.kind == "atom_charge":
            charges = graph.charges.new_tensor(spec.formal_charges)
            mean_charge = (graph.charges @ charges) * (1 - graph.atoms[..., 0])
            value = mean_charge[..., None] * prototype[0]
            mask = atom_mask
        elif descriptor.kind == "connectivity":
            value = (1 - graph.bonds[..., 0:1]) * prototype[1]
            mask = pair_mask
        elif descriptor.kind == "type":
            value = graph.bonds @ prototype[bond_rows]
            mask = pair_mask
        else:
            raise ValueError("Unknown independent native projection kind")
        values[descriptor.key] = value.masked_fill(~mask[..., None], 0)
    return values


def _tangent_gradients(gradients, prepared):
    """Restrict raw gradients to the actual fixed-mask categorical tangent space."""
    active = prepared.active
    atoms, charges, bonds = gradients
    real = atoms[..., 1:]
    real = real - real.mean(-1, keepdim=True)
    atoms = torch.cat((torch.zeros_like(atoms[..., :1]), real), -1)
    atoms = atoms.masked_fill(~active[None, :, None], 0)
    charges = charges - charges.mean(-1, keepdim=True)
    charges = charges.masked_fill(~active[None, :, None], 0)
    bonds = (bonds + bonds.transpose(1, 2)) * .5
    bonds = bonds - bonds.mean(-1, keepdim=True)
    pairs = active[:, None] & active[None, :]
    pairs = pairs & ~torch.eye(len(active), dtype=torch.bool, device=active.device)
    bonds = bonds.masked_fill(~pairs[None, ..., None], 0)
    return atoms, charges, bonds


def native_gate(scorer, prepared, bank, vae, output_path, tolerances=None, *, deadline=None):
    """Persist native/oracle parity and independent composition-gradient evidence.

    Gradients are compared on real-element/category-sum-zero, charge/category-
    sum-zero and symmetric off-diagonal bond directions. NO_ATOM, absent slots
    and self-bonds are excluded by the production fixed-occupancy policy. Central
    FD chords are diagnostic because downstream native operations are nonsmooth.
    No learned-bridge accuracy or optimization approval follows from this gate.
    """
    limits = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        if not isinstance(tolerances, Mapping) or set(tolerances) - set(limits):
            raise ValueError("Unsupported native embedding tolerance names")
        limits.update(tolerances)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in limits.values()):
        raise ValueError("Native embedding tolerances must be finite and nonnegative")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Native embedding gate requires a fresh result path")
    report = dict(version=NATIVE_GATE_VERSION, status="running_not_passed", passed=False,
        stage="initializing", tolerances=limits,
        claim="new exact-native projection boundary and constrained graph derivative parity",
        learned_bridge_validated=False, optimization_authorized_by_gate=False,
        finite_differences_are_diagnostic=True, calls={"forward_attempted": 0, "backward_attempted": 0})

    def save():
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    def call(callback):
        _deadline(deadline)
        report["calls"]["forward_attempted"] += 1
        save()
        return callback()

    versions = context_before = None
    save()
    try:
        prepared.validate()
        if vae.spec != prepared.spec:
            raise ValueError("Frozen VAE and prepared native graph schemas differ")
        versions = _frozen_versions(scorer=scorer, vae=vae, bank=bank)
        context_before = context_digest(prepared)
        report["context_sha256"] = context_before
        graph = prepared.seed.to_relaxed(prepared.spec)
        features, native_bonds = adapt_graph(graph, prepared)
        with torch.no_grad():
            report["stage"] = "native_seed_parity"
            native = _scalars(call(lambda: scorer.discrete_reference(prepared.features, prepared.coordinates)))
            existing = _scalars(call(lambda: scorer(features, native_bonds, prepared.coordinates)))
            report["existing_graph_vs_native"] = _comparison(existing, native, limits)
            save()
            if not report["existing_graph_vs_native"]["passed"]:
                raise ValueError("Existing fixed-graph path failed fresh native scalar parity")
            report["stage"] = "exact_projection_injection"
            exact = bank.targets(prepared.seed, prepared.spec).values
            with inject_native_embeddings(scorer, prepared, exact) as receipt:
                injected = _scalars(call(lambda: scorer(features, native_bonds, prepared.coordinates)))
            report["exact_injection_receipt"] = receipt.record()
            report["exact_injection_vs_native"] = _comparison(injected, native, limits)
            report["exact_injection_vs_existing"] = _comparison(injected, existing, limits)
            save()
            if not (report["exact_injection_vs_native"]["passed"]
                    and report["exact_injection_vs_existing"]["passed"]):
                raise ValueError("Exact native projection injection failed scalar parity")

        def evaluate(candidate, *, injected):
            if not injected:
                f, b = adapt_graph(candidate, prepared)
                return call(lambda: scorer(f, b, prepared.coordinates)), None
            predictions = graph_embedding_predictions(candidate, prepared, bank)
            return predictions, True

        report["stage"] = "independent_projection_derivative"
        derivatives = {}
        for injected in (False, True):
            probe = RelaxedGraph(*(getattr(graph, name).detach().clone().requires_grad_(True) for name in _FIELDS))
            with torch.enable_grad():
                if injected:
                    predictions, _ = evaluate(probe, injected=True)
                    manager = inject_native_embeddings(scorer, prepared, predictions)
                else:
                    manager = nullcontext()
                with manager as receipt:
                    scores = call(lambda: scorer(features, native_bonds, prepared.coordinates)) if injected else evaluate(probe, injected=False)[0]
                    _scalars(scores)
                    report["calls"]["backward_attempted"] += 1
                    save()
                    raw = torch.autograd.grad(scores["dbmol_total_loss"].reshape(()),
                                              tuple(getattr(probe, name) for name in _FIELDS))
                derivatives["injected" if injected else "existing"] = tuple(value.detach() for value in _tangent_gradients(raw, prepared))
                if injected:
                    report["derivative_injection_receipt"] = receipt.record()
            del scores, raw, probe
        a = torch.cat([value.double().flatten() for value in derivatives["existing"]])
        b = torch.cat([value.double().flatten() for value in derivatives["injected"]])
        finite = bool(a.isfinite().all() and b.isfinite().all())
        error, reference = (float((a - b).norm()), float(a.norm())) if finite else (None, None)
        limit = limits["gradient_atol"] + limits["gradient_rtol"] * reference if finite else None
        report["projection_gradient_parity"] = dict(passed=finite and reference > 1e-12 and error <= limit,
            finite=finite, existing_norm=reference, injected_norm=float(b.norm()) if finite else None,
            error_norm=error, relative_error=error / reference if finite and reference > 0 else None,
            allowed_error_norm=limit, policy="fixed-mask-symmetric-simplex-tangent-v1")
        save()
        if not report["projection_gradient_parity"]["passed"]:
            raise ValueError("Independent native embedding composition gradient parity failed")

        report["stage"] = "central_fd_observations"
        report["finite_difference_observations"] = []
        for name, direction in tangent_plan(graph, prepared).items():
            direction = RelaxedGraph(*(getattr(direction, key).to(getattr(graph, key)) for key in _FIELDS))
            analytic = sum(float((gradient.double() * getattr(direction, key).double()).sum())
                           for gradient, key in zip(derivatives["injected"], _FIELDS, strict=True))
            for epsilon in (.001, .01):
                values = {}
                with torch.no_grad():
                    for injected in (False, True):
                        endpoints = []
                        for sign in (1, -1):
                            candidate = RelaxedGraph(*(getattr(graph, key) + sign * epsilon * getattr(direction, key) for key in _FIELDS))
                            if injected:
                                predictions = graph_embedding_predictions(candidate, prepared, bank)
                                with inject_native_embeddings(scorer, prepared, predictions):
                                    scores = call(lambda: scorer(features, native_bonds, prepared.coordinates))
                            else:
                                scores, _ = evaluate(candidate, injected=False)
                            endpoints.append(_scalars(scores)["dbmol_total_loss"])
                        values["injected" if injected else "existing"] = (endpoints[0] - endpoints[1]) / (2 * epsilon)
                report["finite_difference_observations"].append(dict(direction=name, epsilon=epsilon,
                    analytic=analytic, central=values, oracle_central_difference=abs(values["existing"] - values["injected"]),
                    injected_analytic_difference=abs(values["injected"] - analytic), valid_molecule_claim=False))
                save()
        report.update(stage="complete", status="passed_native_oracle_only", passed=True)
    except Exception as error:
        report.update(status="failed_not_passed", passed=False, error_type=type(error).__name__, error=str(error))
        save()
        raise
    finally:
        if versions is not None:
            try:
                report["frozen_modules_unchanged"] = _frozen_versions(scorer=scorer, vae=vae, bank=bank) == versions
            except Exception as error:
                report.update(frozen_modules_unchanged=False, frozen_identity_detail=str(error))
        if context_before is not None:
            try:
                report["context_unchanged"] = context_digest(prepared) == context_before
            except Exception as error:
                report.update(context_unchanged=False, context_identity_detail=str(error))
        if report.get("frozen_modules_unchanged") is False or report.get("context_unchanged") is False:
            report.update(status="failed_frozen_identity", passed=False)
        save()
    if not report["passed"]:
        raise ValueError("Native embedding frozen identity gate failed")
    return report


def score_learned_bridge(scorer, prepared, bridge, vae, z, *, require_gradient=False,
                         deadline=None, include_hard_score=True):
    """Score one learned view and take any requested backward before hook removal.

    Hard labels define the shared molecular identity and detached placeholders.
    Returned scores/gradients are detached; callers cannot accidentally backprop
    through a closed injection context. This diagnostic does no optimization.
    """
    _deadline(deadline)
    prepared.validate()
    if vae.spec != prepared.spec or z.shape != (1, vae.config.latent):
        raise ValueError("Learned bridge latent/prepared VAE schema mismatch")
    versions = _frozen_versions(scorer=scorer, bridge=bridge, vae=vae)
    before = context_digest(prepared)
    with torch.enable_grad() if require_gradient else torch.no_grad():
        point = z.detach().clone().requires_grad_(require_gradient)
        decoded, hidden = vae.decode_with_features(point)
        with torch.no_grad():
            hard = conditioned_hard(decoded, prepared.active[None], prepared.spec)
            natural = decoded.harden(prepared.spec)
            features, bonds = adapt_graph(hard.to_relaxed(prepared.spec), prepared)
        predictions = bridge(hidden.atom_hidden, hidden.bond_input, prepared.active, hidden.charge_input)
        with inject_native_embeddings(scorer, prepared, predictions) as receipt:
            _deadline(deadline)
            scores = scorer(features, bonds, prepared.coordinates)
            values = _scalars(scores)
            gradient = torch.autograd.grad(scores["dbmol_total_loss"].reshape(()), point)[0] if require_gradient else None
            if gradient is not None and not bool(gradient.isfinite().all()):
                raise ValueError("Learned bridge latent derivative is nonfinite")
    assessment = classify_repair_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], prepared.spec)
    upper = torch.triu(torch.ones_like(hard.bonds, dtype=torch.bool), diagonal=1)
    policy_difference = dict(policy="fixed-prepared-occupancy-vs-natural-argmax-v1",
        natural_occupied_slots=int((natural.atoms != 0).sum()), conditioned_occupied_slots=int((hard.atoms != 0).sum()),
        occupancy_label_differences=int(((natural.atoms != 0) != (hard.atoms != 0)).sum()),
        atom_label_differences=int((natural.atoms != hard.atoms).sum()),
        charge_label_differences=int((natural.charges != hard.charges).sum()),
        unordered_bond_label_differences=int(((natural.bonds != hard.bonds) & upper).sum()))
    result = dict(scores=values, gradient=None if gradient is None else gradient.detach(), hard=hard,
        injection_receipt=receipt.record(), context_sha256=before,
        hard_classification=assessment, canonical_smiles=assessment["smiles"],
        graph_policy_difference=policy_difference,
        invalid_hard_scores_retained_as_tensor_diagnostics=True,
        gradient_norm=None if gradient is None else float(gradient.double().norm()),
        learned_bridge_validated=False, scope="same-hard-identity-fixed-context-diagnostic")
    if include_hard_score:
        _deadline(deadline)
        with torch.no_grad():
            result["hard_fixed_context_scores"] = _scalars(scorer(features, bonds, prepared.coordinates))
        result["learned_minus_hard_total_loss"] = values["dbmol_total_loss"] - result["hard_fixed_context_scores"]["dbmol_total_loss"]
        result["hard_score_scope"] = "existing-hard-chemical-path-fixed-pose-and-names; no-native-rebuild"
    result["frozen_modules_unchanged"] = _frozen_versions(scorer=scorer, bridge=bridge, vae=vae) == versions
    result["context_unchanged"] = context_digest(prepared) == before
    if not result["frozen_modules_unchanged"] or not result["context_unchanged"]:
        raise ValueError("Frozen model/context changed during learned embedding diagnostic")
    return result
