"""Allocated source091 step15 contact/confidence reduction attribution.

The original shared graph, pose, pocket, mask, repaired target and weights are
unchanged. A complete frontier contains the structural distogram and every
native confidence-head projection. Identity clones make the head projections
and distogram siblings: distogram-dependent PDE weights must not be omitted or
counted twice. Native ConfidenceHeads.forward performs cached reductions; no
alternative confidence architecture or fabricated head outputs are used.

Production arithmetic, including its contact/assembly FP64 policy, is replayed
exactly. Explicit confidence-reduction FP64 and baseline-held contact/PTM winner
controls are diagnostic counterfactuals, never production fixes. The control's
standalone structural gates passed; its combined direction failed. Guided
structural gates were not reached after the affinity failure. This observer
reports original directions, not new outcome-selected component directions.
No DeFoG inference, latent updates or historical gate reclassification occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import math
import os
from pathlib import Path
import sys


CASE_TIMEOUT_SECONDS = 3450
VARIANTS = (("production_native", "float32", False),
            ("confidence_reduction_float64", "float64", False),
            ("production_winners_held", "float32", True),
            ("float64_winners_held", "float64", True))
GEOMETRY_FIELDS = ("token_pad_mask", "atom_pad_mask", "atom_to_token", "mol_type",
                   "asym_id", "frames_idx", "token_to_rep_atom", "residue_index", "affinity_token_mask")
CONFIDENCE_KEYS = ("iptm", "complex_iplddt", "complex_ipde", "complex_pde", "confidence_score", "ptm")
TERM_KEYS = ("dbmol_contact_loss", "dbmol_contact_pocket_loss", "dbmol_contact_anchor_loss",
             "dbmol_confidence_loss", "complex_plddt", *CONFIDENCE_KEYS)
CLAIM_LIMIT = __doc__
_TOOLS = None
_CASE = None
_FULL_CALLS = 0
_CACHE_CALLS = 0
_MAX_FULL_CALLS = 0
_MAX_CACHE_CALLS = 0


def tools():
    """Load adjacent source101 instrumentation before scientific bootstrap."""
    global _TOOLS
    if _TOOLS is None:
        spec = importlib.util.spec_from_file_location("_structural_affinity_tools",
            Path(__file__).with_name("affinity_head_diagnostic.py"))
        _TOOLS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TOOLS)
    return _TOOLS


def projection_names(heads):
    """Use the installed checkpoint's actual separate/combined projection set."""
    names = ["to_plddt_logits", "to_resolved_logits"]
    if heads.use_separate_heads:
        names += ["to_pde_intra_logits", "to_pde_inter_logits", "to_pae_intra_logits", "to_pae_inter_logits"]
    else:
        names += ["to_pde_logits", "to_pae_logits"]
    actual = {name for name, _ in heads.named_children()}
    if actual != set(names):
        raise ValueError("Unexpected native confidence projection membership")
    return tuple(names)


class StructuralCapture:
    """Identity-clone hooks; the distogram bypass into PDE is part of the cut."""

    def __init__(self, structure):
        self.structure = structure
        self.heads = structure.confidence_module.confidence_heads
        self.names = projection_names(self.heads)
        self.values, self.geometry, self.raw, self.handles = {}, {}, {}, []

    def save(self, name, value):
        import torch
        if name in self.values or not isinstance(value, torch.Tensor):
            raise ValueError("Missing, repeated or nontensor structural projection: " + name)
        result = value.clone()
        self.values[name] = result
        return result

    def __enter__(self):
        self.handles.append(self.structure.distogram_module.register_forward_hook(
            lambda _module, _args, value: self.save("distogram", value)))
        for name in self.names:
            self.handles.append(getattr(self.heads, name).register_forward_hook(
                lambda _module, _args, value, name=name: self.save(name, value)))

        def geometry(_module, args, kwargs):
            # Production attached_confidence calls the original heads by keyword.
            if args or kwargs.get("multiplicity") != 1:
                raise ValueError("Require original keyword batch-one confidence call")
            self.geometry = {"feats": {key: kwargs["feats"][key].detach().clone() for key in GEOMETRY_FIELDS},
                "x_pred": kwargs["x_pred"].detach().clone(), "d": kwargs["d"].detach().clone()}
            if any(value.requires_grad for value in (kwargs["x_pred"], kwargs["d"])):
                raise ValueError("Structural observer requires detached fixed geometry")

        self.handles.append(self.heads.register_forward_pre_hook(geometry, with_kwargs=True))
        self.handles.append(self.heads.register_forward_hook(
            lambda _module, _args, value: self.raw.update({key: value[key] for key in
                ("complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde", "pae_logits")})))
        return self

    def require_complete(self):
        if set(self.values) != {"distogram", *self.names} or not self.geometry or not self.raw:
            raise ValueError("Incomplete structural frontier/geometry/raw outputs")
        from latentmol.boltz.dbmol_objective import native_bin_centers
        self.geometry["contact_bin_centers"] = native_bin_centers(self.values["distogram"]).detach().clone()

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def fixed_projection(value):
    """Native head projection stand-in: return the authenticated cached tensor.

    Dummy s/z are only used by the original head to call these stand-ins. Every
    learned projection is replaced, so no invented learned features influence
    the cached native reduction. Gradients still flow into the cached logits.
    """
    import torch
    class Projection(torch.nn.Module):
        def forward(self, _value):
            return value
    return Projection()


def cached_raw(heads, values, geometry):
    """Execute the unchanged native head reductions at cached projected logits."""
    import torch
    names = projection_names(heads)
    if set(values) != {"distogram", *names}:
        raise ValueError("Cached frontier differs from native projection membership")
    clone = deepcopy(heads).eval().requires_grad_(False)
    # A fixture may call this while capture hooks are still installed on the
    # original. Deepcopy also copies hooks; they are observer side effects,
    # not native head math. Strip them from this private cached clone only.
    for module in clone.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
    for name in names:
        setattr(clone, name, fixed_projection(values[name]))
    n = geometry["feats"]["token_pad_mask"].shape[1]
    dummy = values["distogram"].new_zeros
    return clone(s=dummy((1, n, 1)), z=dummy((1, n, n, 1)),
        x_pred=geometry["x_pred"], d=geometry["d"], feats=geometry["feats"],
        pred_distogram_logits=values["distogram"] if values["distogram"].ndim == 4
                              else values["distogram"][:, :, :, 0], multiplicity=1)


def tm_row_scores(logits, geometry):
    """Passive native PTM/IPTM row witnesses, with fixed native frame geometry.

    Constants and operation order mirror the pinned confidence_utils function.
    This witness is checked against native PTM/IPTM outputs before it is used
    for a held-winner counterfactual. It does not replace production aggregation.
    """
    import torch
    from boltz.model.layers.confidence_utils import compute_frame_pred, tm_function
    feats = geometry["feats"]
    _, frames = compute_frame_pred(geometry["x_pred"], feats["frames_idx"], feats, 1, inference=True)
    pad = feats["token_pad_mask"]
    frame = frames.reshape(-1, frames.shape[-1])
    masks = {"ptm": frame[:, :, None] * pad[:, None, :] * pad[:, :, None],
             "iptm": frame[:, :, None] * (feats["asym_id"][:, None, :] != feats["asym_id"][:, :, None])
                     * pad[:, None, :] * pad[:, :, None]}
    width = 32. / logits.shape[-1]
    bins = torch.arange(start=.5 * width, end=32., step=width, device=logits.device).unsqueeze(0)
    native_tm = tm_function(bins, pad.sum(dim=-1, keepdim=True)).unsqueeze(1).unsqueeze(2)
    expected = torch.sum(torch.nn.functional.softmax(logits, dim=-1) * native_tm, dim=-1)
    return {name: torch.sum(expected * mask, dim=-1) / (torch.sum(mask, dim=-1) + 1e-5)
            for name, mask in masks.items()}


def contact_pairs(logits, pocket, ligand, config, *, bin_centers=None):
    """Return the exact production contact pair-loss matrix before minima."""
    import torch
    from latentmol.boltz.dbmol_objective import native_bin_centers, CONTACT64_ARITHMETIC
    centers = native_bin_centers(logits) if bin_centers is None else bin_centers
    if getattr(config, "objective_arithmetic", "native_v1") == CONTACT64_ARITHMETIC:
        logits, centers = logits.double(), centers.double()
    if logits.ndim == 5:
        if logits.shape[3] != 1:
            raise ValueError("Require one distogram ensemble")
        logits = logits[:, :, :, 0]
    selected = logits[0][pocket][:, ligand]
    mass = torch.logsumexp(selected.log_softmax(-1)[..., centers < config.distance_cutoff], -1)
    return -torch.logaddexp(mass, mass.new_tensor(math.log(config.epsilon)))


def selection_state(pair_loss, tm_rows, scores):
    """Keep winners, ties and margins, including the confidence TM fallback."""
    import torch
    def winner(value, *, maximum):
        flat = value.detach().reshape(-1)
        index = int(flat.argmax() if maximum else flat.argmin())
        ranked = flat.sort(descending=maximum).values
        return {"index": index, "exact_ties": int((flat == flat[index]).sum()),
                "tie_indices": (flat == flat[index]).nonzero().flatten().tolist(),
                "best_vs_second_margin": float((ranked[0] - ranked[1]).abs()) if flat.numel() > 1 else None}
    return {"contact_per_pocket": [winner(row, maximum=False) for row in pair_loss],
        "contact_anchor": winner(pair_loss, maximum=False),
        "tm_rows": {key: winner(value, maximum=True) for key, value in tm_rows.items()},
        "confidence_tm_uses_iptm": not torch.allclose(scores["iptm"], torch.zeros_like(scores["iptm"]))}


def selection_changes(current, baseline):
    """Compare each endpoint to baseline; plus/minus agreement can hide switches."""
    return {"contact_per_pocket": [a["index"] != b["index"] for a, b in
                zip(current["contact_per_pocket"], baseline["contact_per_pocket"])],
        "contact_anchor": current["contact_anchor"]["tie_indices"] != baseline["contact_anchor"]["tie_indices"],
        "tm_rows": {key: current["tm_rows"][key]["index"] != baseline["tm_rows"][key]["index"] for key in ("ptm", "iptm")},
        "confidence_tm_fallback": current["confidence_tm_uses_iptm"] != baseline["confidence_tm_uses_iptm"]}


def held_contact(pair_loss, selection):
    """Extend the baseline selected branches; no nearest-molecule interpretation."""
    import torch
    indices = torch.tensor([row["index"] for row in selection["contact_per_pocket"]], device=pair_loss.device)
    pocket = pair_loss.gather(1, indices[:, None]).mean()
    flat = pair_loss.reshape(-1)
    anchor = flat[selection["contact_anchor"]["index"]]
    # torch.min() without dim shares its derivative across exact global ties,
    # whereas min(dim)/max(dim) chooses the first index. Retain that baseline
    # subgradient, and center the tied mean so its baseline scalar stays exact.
    ties = selection["contact_anchor"]["tie_indices"]
    anchor = anchor + (flat[ties] - anchor).mean()
    return {"dbmol_contact_loss": (pocket + anchor) / 2,
            "dbmol_contact_pocket_loss": pocket, "dbmol_contact_anchor_loss": anchor}


def cached_terms(heads, values, geometry, config, affinity, *, held=None):
    """Native FP32/declared FP64 cached reductions, plus explicit held controls."""
    global _CACHE_CALLS
    if _MAX_CACHE_CALLS:
        if _CACHE_CALLS >= _MAX_CACHE_CALLS:
            raise ValueError("Cached structural reduction budget exceeded")
        _CACHE_CALLS += 1
    import torch
    from latentmol.boltz.dbmol_model import strict_confidence_scores
    from latentmol.boltz.dbmol_objective import contact_loss, CONTACT64_ARITHMETIC, pocket_token_mask, reduce_objective
    raw = cached_raw(heads, values, geometry)
    scores = strict_confidence_scores(raw, geometry["feats"], geometry["x_pred"])
    rows = tm_row_scores(raw["pae_logits"], geometry)
    for key in rows:
        if not torch.equal(rows[key].max(dim=1).values, scores[key]):
            raise ValueError("Native PTM row witness output parity failed: " + key)
    pocket, ligand = pocket_token_mask(geometry["feats"], config)
    centers = geometry["contact_bin_centers"]
    contact_logits = values["distogram"]
    if getattr(config, "objective_arithmetic", "native_v1") == CONTACT64_ARITHMETIC:
        contact_logits, centers = contact_logits.double(), centers.double()
    pairs = contact_pairs(values["distogram"], pocket, ligand, config, bin_centers=centers)
    selections = selection_state(pairs, rows, scores)
    contact = contact_loss(contact_logits, pocket, ligand, cutoff=config.distance_cutoff,
                           epsilon=config.epsilon, bin_centers=centers)
    result = reduce_objective({**affinity, **scores, **contact}, config)
    # Check passive pair reconstruction against the exact active production
    # contact reduction before any held-branch interpretation is permitted.
    for key, value in held_contact(pairs, selections).items():
        if not torch.equal(value, result[key]):
            raise ValueError("Native contact witness scalar parity failed: " + key)
    if held is not None:
        scores = dict(scores)
        for key in rows:
            scores[key] = rows[key][:, held["tm_rows"][key]["index"]]
        tm = scores["iptm"] if held["confidence_tm_uses_iptm"] else scores["ptm"]
        scores["confidence_score"] = (4 * scores["complex_plddt"] + tm) / 5
        result = reduce_objective({**affinity, **scores, **held_contact(pairs, held)}, config)
    return {key: result[key] for key in TERM_KEYS}, selections


def full_forward(decoder, point, branch, config, counters, capture, label):
    global _FULL_CALLS
    if _FULL_CALLS >= _MAX_FULL_CALLS:
        raise ValueError("Full structural forward budget exceeded")
    _FULL_CALLS += 1
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.study import tensor_digest
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature, fixed_active=branch.prepared.active)
    outputs = counters.call("boltz", label, "forward", lambda: branch(graph), point.device)
    capture.require_complete()
    return outputs, tensor_digest({"atoms": graph.atoms, "charges": graph.charges, "bonds": graph.bonds})


def values_parity(outputs, expected, label):
    if {key: float(value.detach()) for key, value in outputs.items()} != expected:
        raise ValueError("Original native scalar parity failed: " + label)


def structural_provenance(case, model):
    """Authenticate loaded structural classes without assuming affinity heads.

    The confidence checkpoint does not have to enable affinity prediction.
    Inspect all its actual Boltz modules (including its confidence projections),
    and authenticate every exported installed-model file, not a hand-picked
    affinity-only class list. Torch-owned Linear/container classes are separate.
    """
    import inspect
    files = {path: digest for path, digest in case["parent_artifacts_sha256"].items()
             if "/site-packages/boltz/model/" in path and path.endswith(".py")}
    actual = sorted({str(Path(inspect.getfile(type(module))).resolve()) for module in model.modules()
                     if type(module).__module__.startswith("boltz.model.")})
    if len(files) != 50 or not actual or any(path not in files for path in actual):
        raise ValueError("Loaded structural native classes are not pinned installed files")
    for path, expected in files.items():
        if tools().helper()._sha(path) != expected:
            raise ValueError("Installed structural native source changed: " + path)
    return {"loaded_structural_module_files": actual, "installed_source_files_sha256": files,
            "all_installed_model_sources_authenticated": True, "affinity_members_assumed": False}


def tensor_gradients(scalar, values, *, retain_graph=True):
    """Allow genuine unused frontier fields (resolved logits), never hide None."""
    import torch
    gradients = torch.autograd.grad(scalar, tuple(values.values()), allow_unused=True, retain_graph=retain_graph)
    return {key: torch.zeros_like(value) if gradient is None else gradient
            for (key, value), gradient in zip(values.items(), gradients)}


def budget(direction_counts):
    if tuple(direction_counts) != (2, 4):
        raise ValueError("Require the declared original two/four direction cases")
    explicit = [1 + 18 * count for count in direction_counts]
    cached = [4 + 72 * count for count in direction_counts]
    return {"direction_counts": list(direction_counts), "explicit_full_forwards_per_case": explicit,
        "auxiliary_original_forward_and_seed_parity_per_case": 3,
        "maximum_total_full_forwards": sum(explicit) + 6, "cached_reductions_per_case": cached,
        "maximum_physical_trunk_forwards_excluding_checkpoint_recomputation": 2 * (sum(explicit) + 6),
        "baseline_upstream_reverse_traversals_per_case": 2 * len(TERM_KEYS),
        "reverse_scope": "Direct and complete-cut upstream backward per structural term, plus original production score_at backwards and reduction-only cotangents; forward bound excludes checkpoint recomputation",
        "maximum_child_seconds": 2 * CASE_TIMEOUT_SECONDS, "gpu_wall_seconds": 7200,
        "controller_headroom_seconds": 7200 - 2 * CASE_TIMEOUT_SECONDS,
        "new_defog_calls": 0, "optimizer_updates": 0}


def derivative_rows(point, direction, direct_gradient, chain_gradient, epsilon,
                    native_plus, native_minus, linear_plus, linear_minus, baseline,
                    *, dtype, rtol, atol, observer):
    """Use independently measured direct and frontier VJPs, never one twice."""
    from latentmol.boltz.local_fd import endpoint_row
    native = endpoint_row(point, direction, direct_gradient, epsilon,
        {"loss": native_plus, "loss_dtype": dtype}, {"loss": native_minus, "loss_dtype": dtype},
        baseline, rtol=rtol, atol=atol)
    chain = endpoint_row(point, direction, chain_gradient, epsilon,
        {"loss": linear_plus, "loss_dtype": "torch.float64"}, {"loss": linear_minus, "loss_dtype": "torch.float64"},
        0., rtol=rtol, atol=atol)
    return native, chain, observer.qualify_residual(native, chain)


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item,
               *, persist=lambda: None, report=None):
    """Authenticate original backward once, then inspect exact endpoint reductions."""
    global _FULL_CALLS, _CACHE_CALLS, _MAX_FULL_CALLS, _MAX_CACHE_CALLS
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.boltz.local_fd import EPSILONS
    from latentmol.multimodel.study import tensor_digest
    from latentmol.boltz.dbmol_objective import CONFIDENCE_SIGNS
    report = report if report is not None else {}
    observer = tools().boundary(_CASE)
    reference, receipt = observer.authenticated_reference(_CASE, report)
    forward_base, analytic_base = tools().reference_baselines(reference)
    if config.recycling_steps != 0:
        raise ValueError("Require the original zero-recycling state")
    components = {}
    measured, total = score_at(decoder, point, branch, config, counters, "structural_original_full_analytic",
        gradients=True, component_gradients=components, repair_target=target, repair_weight=weight)
    if (measured["components"] != {key: analytic_base["values"][key] for key in measured["components"]}
            or measured["loss"] != analytic_base["values"]["combined"]
            or tensor_digest({"gradient": total}) != analytic_base["sum_gradient_identity"]["combined_gradient_sha256"]):
        raise ValueError("Original source093 full scalar/backward parity failed")
    directions = tools().helper().reconstruct_directions(point, total if item_name == "combined" else components[item_name], item)
    if any(metadata["replayed_sha256"] != reference["direction_replay"][name]["replayed_sha256"]
           for name, (_, metadata) in directions.items()):
        raise ValueError("Original source093 direction identity differs")
    count = len(directions)
    _FULL_CALLS, _CACHE_CALLS, _MAX_FULL_CALLS, _MAX_CACHE_CALLS = 0, 0, 1 + 18 * count, 4 + 72 * count
    structural_direct = {key: components[key].detach() for key in ("dbmol_contact_loss", "dbmol_confidence_loss")}
    del components, total
    report.update(status="probing", diagnostic_completed=False, reference093=receipt,
        native_provenance=structural_provenance(_CASE, branch.scorer.structure),
        finite_differences=[], direction_replay={key: meta for key, (_, meta) in directions.items()},
        original_gate_not_reclassified=True, original_gate_scope={"boltz_only": "Standalone contact/confidence gates passed; combined direction failed",
            "boltz_defog": "Structural gates not reached after affinity gate failure"}[_CASE["id"]],
        arithmetic={"production_objective": getattr(config.dbmol, "objective_arithmetic", "native_v1"),
            "alternatives": "Cached confidence projection logits/reductions widened; geometry constants retain native precision; native upstream model unchanged"},
        scope=CLAIM_LIMIT)

    def baseline():
        probe = point.detach().clone().requires_grad_()
        with torch.enable_grad(), StructuralCapture(branch.scorer.structure) as capture:
            outputs, graph_sha = full_forward(decoder, probe, branch, config, counters, capture, "structural_frontier_baseline")
            values_parity(outputs, {key: forward_base["values"][key] for key in outputs}, "baseline")
            if graph_sha != forward_base["graph_sha256"]:
                raise ValueError("Original source093 baseline graph differs")
            cotangents, gradients, chain_gradients, checks = {}, {}, {}, {}
            for key in TERM_KEYS:
                direct, = torch.autograd.grad(outputs[key], probe, retain_graph=True)
                cotangent = tensor_gradients(outputs[key], capture.values)
                chain, = torch.autograd.grad(tuple(capture.values.values()), probe,
                    grad_outputs=tuple(cotangent.values()), retain_graph=True)
                checks[key] = observer.chain_completeness(direct, chain)
                if key in structural_direct:
                    checks[key]["original_source093_component"] = observer.chain_completeness(structural_direct[key], direct)
                if not checks[key]["engineering_chain_complete"] or (key in structural_direct
                        and not checks[key]["original_source093_component"]["engineering_chain_complete"]):
                    raise ValueError("Incomplete original structural frontier VJP: " + key)
                gradients[key] = direct.detach().clone()
                chain_gradients[key] = chain.detach().clone()
                cotangents[key] = {name: value.detach().clone() for name, value in cotangent.items()}
            primitive_sum = sum(config.dbmol.confidence_weights[key] * CONFIDENCE_SIGNS[key]
                * gradients[key].double() for key in CONFIDENCE_KEYS)
            primitive_check = observer.chain_completeness(gradients["dbmol_confidence_loss"], primitive_sum)
            if not primitive_check["engineering_chain_complete"]:
                raise ValueError("Confidence primitive weighted-gradient identity failed")
            affinity = {key: outputs[key].detach().clone() for key in ("affinity_pred_value", "affinity_logits_binary")}
            return {"values": {key: value.detach().clone() for key, value in capture.values.items()},
                "geometry": deepcopy(capture.geometry), "gradients": gradients, "chain_gradients": chain_gradients, "cotangents": cotangents,
                "outputs": {key: outputs[key].detach().clone() for key in TERM_KEYS}, "affinity": affinity,
                "checks": checks, "primitive_weighted_gradient_identity": primitive_check,
                "graph_sha256": graph_sha, "projection_names": list(capture.names)}
    base = observer.phase("complete_structural_frontier_baseline", baseline, point.device, report, persist)
    native_heads = branch.scorer.structure.confidence_module.confidence_heads
    variant_baselines, variant_gradients = {}, {}
    selection = None
    for variant, dtype, held in VARIANTS:
        values = {key: value.to(getattr(torch, dtype)).requires_grad_() for key, value in base["values"].items()}
        output, chosen = cached_terms(native_heads, values, base["geometry"], config.dbmol, base["affinity"],
            held=selection if held else None)
        if variant == "production_native":
            selection = chosen
        variant_gradients[variant], parity = {}, {}
        for key in TERM_KEYS:
            cotangent = tensor_gradients(output[key], values)
            variant_gradients[variant][key] = {name: gradient.detach().clone() for name, gradient in cotangent.items()}
            if dtype == "float32":
                if not torch.equal(output[key], base["outputs"][key]):
                    raise ValueError("Cached original structural scalar parity failed: " + variant + "/" + key)
                expected = torch.cat([value.reshape(-1) for value in base["cotangents"][key].values()])
                actual = torch.cat([value.reshape(-1) for value in cotangent.values()])
                parity[key] = observer.chain_completeness(expected, actual)
                if not parity[key]["engineering_chain_complete"]:
                    raise ValueError("Cached original structural cotangent parity failed: " + variant + "/" + key)
        variant_baselines[variant] = {"values": {key: float(value) for key, value in output.items()},
            "selection": chosen, "original_scalar_vjp_parity": parity,
            "scope": "Original parity" if dtype == "float32" else "Declared confidence-reduction arithmetic alternative"}
        del output, values, cotangent
    report["baseline"] = {"graph_sha256": base["graph_sha256"], "complete_frontier_vjp": base["checks"],
        "primitive_weighted_gradient_identity": base["primitive_weighted_gradient_identity"],
        "projection_names": base["projection_names"], "frontier_metadata": observer.tensor_metadata(base["values"]),
        "geometry_sha256": tensor_digest({**base["geometry"]["feats"], "x_pred": base["geometry"]["x_pred"], "d": base["geometry"]["d"],
                                           "contact_bin_centers": base["geometry"]["contact_bin_centers"]}),
        "variants": variant_baselines}
    persist()
    for label, (direction, _) in directions.items():
        for epsilon in EPSILONS:
            record = {"direction": label, "epsilon": epsilon, "endpoints": {}, "terms": {}}
            endpoint_terms, linear_readouts, variants = {}, {}, {}
            for sign, multiplier in (("plus", 1), ("minus", -1)):
                candidate = point + multiplier * epsilon * direction
                def capture_endpoint():
                    with torch.no_grad(), StructuralCapture(branch.scorer.structure) as capture:
                        outputs, graph_sha = full_forward(decoder, candidate, branch, config, counters, capture, "structural_endpoint")
                        matches = [row[sign] for row in reference["finite_differences"] if row["direction"] == label
                                   and row["epsilon"] == epsilon and row["grad_enabled"] is False]
                        if len(matches) != 1 or graph_sha != matches[0]["graph_sha256"]:
                            raise ValueError("Original source093 endpoint graph/reference differs")
                        values_parity(outputs, {key: matches[0]["values"][key] for key in outputs}, "endpoint")
                        geometry_sha = tensor_digest({**capture.geometry["feats"], "x_pred": capture.geometry["x_pred"], "d": capture.geometry["d"],
                                                      "contact_bin_centers": capture.geometry["contact_bin_centers"]})
                        if geometry_sha != report["baseline"]["geometry_sha256"]:
                            raise ValueError("Fixed structural geometry changed at an endpoint")
                        return {"values": {key: value.detach().clone() for key, value in capture.values.items()},
                            "outputs": {key: outputs[key].detach().clone() for key in TERM_KEYS}, "graph_sha256": graph_sha}
                endpoint = observer.phase("structural_" + label + "_" + str(epsilon) + "_" + sign,
                    capture_endpoint, point.device, report, persist)
                endpoint_terms[sign] = {key: float(value) for key, value in endpoint["outputs"].items()}
                linear_readouts[sign] = {key: float(sum(((endpoint["values"][name].double() - base["values"][name].double())
                    * cotangent.double()).sum() for name, cotangent in base["cotangents"][key].items())) for key in TERM_KEYS}
                variants[sign] = {}
                with torch.no_grad():
                    for variant, dtype, held in VARIANTS:
                        vals = {key: value.to(getattr(torch, dtype)) for key, value in endpoint["values"].items()}
                        output, chosen = cached_terms(native_heads, vals, base["geometry"], config.dbmol, base["affinity"],
                            held=selection if held else None)
                        if variant == "production_native" and any(not torch.equal(output[key], endpoint["outputs"][key]) for key in TERM_KEYS):
                            raise ValueError("Cached native endpoint scalar parity failed")
                        variants[sign][variant] = {"values": {key: float(value) for key, value in output.items()},
                            "selection": chosen, "baseline_to_endpoint_switches": selection_changes(chosen, selection),
                            "linear_readouts": {key: float(sum(((vals[name].double() - base["values"][name].double())
                                * gradient.double()).sum() for name, gradient in variant_gradients[variant][key].items())) for key in TERM_KEYS}}
                        del vals, output
                record["endpoints"][sign] = {"graph_sha256": endpoint["graph_sha256"], "variants": variants[sign],
                    "frontier_metadata": observer.tensor_metadata(endpoint["values"])}
                del endpoint
            for key in TERM_KEYS:
                native, chain, attribution = derivative_rows(point, direction, base["gradients"][key], base["chain_gradients"][key], epsilon,
                    endpoint_terms["plus"][key], endpoint_terms["minus"][key],
                    linear_readouts["plus"][key], linear_readouts["minus"][key], float(base["outputs"][key]),
                    dtype=str(base["outputs"][key].dtype), rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol, observer=observer)
                comparisons = {variant: {"finite_difference": (variants["plus"][variant]["values"][key]
                    - variants["minus"][variant]["values"][key]) / (2 * epsilon),
                    "frontier_linear_prediction": (variants["plus"][variant]["linear_readouts"][key]
                    - variants["minus"][variant]["linear_readouts"][key]) / (2 * epsilon)} for variant, _, _ in VARIANTS}
                record["terms"][key] = {"original_native_reference": native, "frontier_linear_readout": chain,
                    "attribution": attribution, "cached_variants": comparisons,
                    "confidence_precision_fd_change": comparisons["confidence_reduction_float64"]["finite_difference"] - comparisons["production_native"]["finite_difference"],
                    "held_selection_fd_change_fp64": comparisons["float64_winners_held"]["finite_difference"] - comparisons["confidence_reduction_float64"]["finite_difference"]}
            report["finite_differences"].append(record)
            persist()
    report.update(status="completed", diagnostic_completed=True, old_gate_certified_by_this_diagnostic=False,
        explicit_full_forwards_used=_FULL_CALLS, explicit_full_forwards_bound=_MAX_FULL_CALLS,
        cached_reductions_used=_CACHE_CALLS, cached_reductions_bound=_MAX_CACHE_CALLS,
        interpretation="Frontier readouts distinguish upstream structural-network residuals from cached reduction residuals; precision/held-winner differences alone do not prove a unique model error or authorize a production change.")
    persist()
    return report


def run_case(plan, case, output_root):
    global _CASE, _MAX_CACHE_CALLS
    _CASE = case
    replay = tools().helper()
    tools().boundary(case)
    original = replay.probe_case
    replay.probe_case = probe_case
    replay.CASE_TIMEOUT_SECONDS, replay.REPEATS, replay.CLAIM_LIMIT = CASE_TIMEOUT_SECONDS, 1, CLAIM_LIMIT
    try:
        return replay.run_case(plan, case, output_root)
    finally:
        replay.probe_case = original
        _MAX_CACHE_CALLS = 0


def validate_cases(plan):
    tools().validate_cases(plan)
    for case in plan["cases"]:
        if any(not any(path.endswith(suffix) for path in case["parent_artifacts_sha256"]) for suffix in
               ("/boltz/model/modules/confidencev2.py", "/boltz/model/layers/confidence_utils.py")):
            raise ValueError("Native confidence sources must be pinned")
    return plan


def run_study(path, expected):
    plan = validate_cases(tools().helper().load_plan(path, expected))
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    counts = []
    for case in plan["cases"]:
        gate = Path(case["parent_run"]) / case["arm"] / "gate-0015.json"
        _, item = tools().helper().failed_item(tools().helper()._read(tools().helper().authenticated_artifact(case, gate)))
        counts.append(1 + sum(name != "gradient" and row.get("passed") is False for name, row in item["direction_results"].items()))
    with ExperimentRun(Path(plan["output_root"]), plan["name"], {"plan": plan, "plan_sha256": expected,
            "diagnostic_module_sha256": tools().helper()._sha(__file__), "budget": budget(counts), "claim_limit": CLAIM_LIMIT}, Path.cwd()) as run:
        rows = []
        for case in plan["cases"]:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / (case["id"] + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
            completed, reason = tools().boundary(case).execution_completion(execution)
            rows.append({"case_id": case["id"], "execution": execution, "diagnostic_completed": completed,
                         "incomplete_reason": reason, "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        run.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
                     diagnostic_cases_requested=2, old_gates_certified=False)
        print(run.path, flush=True)
        tools().boundary(plan["cases"][0]).require_completed_cases(rows)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not tools().helper()._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Observer requires an authenticated allocated DTU HPC job")
    plan = validate_cases(tools().helper().load_plan(args.plan, args.sha256))
    if args.child:
        matches = [case for case in plan["cases"] if case["id"] == args.child]
        if len(matches) != 1 or args.child_output is None:
            raise ValueError("Unknown child or missing unique output root")
        print(run_case(plan, matches[0], args.child_output), flush=True)
    else:
        if args.child_output is not None:
            raise ValueError("Child output requires --child")
        print(run_study(args.plan, args.sha256), flush=True)


if __name__ == "__main__":
    main()
