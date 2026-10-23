"""Exact-state, HPC-only localization of native Boltz affinity FD residuals.

This observer authenticates source091's failed step-15 states and source093's
completed endpoint replay. It neither runs DeFoG nor changes an optimizer,
production arithmetic or a validation tolerance. The original full DBMol
backward selects the already-declared directions. Subsequent observations use
its identical native affinity branch, holding graph occupancy, geometry and
context fixed. Structural/contact branches are outside this localization.

Complete cuts include BOTH ensemble members and every chemical bypass. Identity
clones make cut outputs siblings: capturing an ancestor alongside its descendant
would otherwise count a reverse path twice. Clone math/derivative parity is
measured, not assumed. Every cut receives its own fresh checkpointed graph and
three reverse traversals; no native graph survives between phases. Endpoints use
the same original FP32 native operations. Only the detached diagnostic linear
readouts are widened, and neither their slopes nor chain engineering checks
replace historical gates. A first residual-bearing interval localizes a stage,
not a unique faulty operation, a valid gradient correction or a passed pilot.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import importlib.util
import math
import os
from pathlib import Path
import sys
import time


CASE_TIMEOUT_SECONDS = 3450
MAXIMUM_AFFINITY_FORWARDS = 160
REFERENCE_ARCHIVE_SHA256 = "78405ec51fb571c8e9f0ad3fc41a98a0cc1b1fbba16e67b1724239117499c0e2"
REFERENCE_MANIFEST_SHA256 = "485936effcd12fff758e5b95a578ff486cb3a3b579ea5640fdf6cd150f6679c3"
FIELDS = ("atoms", "charges", "bonds")
CLAIM_LIMIT = __doc__
_HELPER = None


def helper():
    """Load new instrumentation alone before old-source scientific bootstrap."""
    global _HELPER
    if _HELPER is None:
        spec = importlib.util.spec_from_file_location("_operation_replay_helper",
            Path(__file__).with_name("failed_gate_attribution.py"))
        _HELPER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_HELPER)
    return _HELPER


def reference_paths(case):
    paths = [Path(path) for path in case["parent_artifacts_sha256"]
             if "source-093-first-suite-attempt001-replay" in path and Path(path).name == "attribution.json"]
    if len(paths) != 1:
        raise ValueError("Require one pinned source093 case report")
    return paths[0], paths[0].with_name("run.json")


def authenticated_reference(case, report):
    """Reference identity includes point, target, actual lambda and native state."""
    replay = helper()
    path, run_path = reference_paths(case)
    previous = replay._read(replay.authenticated_artifact(case, path))
    run = replay._read(replay.authenticated_artifact(case, run_path))
    if (run.get("status") != "completed"
            or run.get("source_archive_sha256") != REFERENCE_ARCHIVE_SHA256
            or run.get("source", {}).get("sha256") != REFERENCE_MANIFEST_SHA256
            or not all(previous.get(key) is True for key in (
                "diagnostic_completed", "frozen_state_unchanged", "parent_unchanged", "native_seed_parity"))
            or previous.get("identities_before") != report["identities_before"]
            or previous.get("identities_before") != previous.get("identities_after")
            or previous.get("target") != report["target"]
            or previous.get("original_item") != report["original_item"]):
        raise ValueError("Source093 reference is incomplete or is a different scientific state")
    return previous, {"path": str(path), "sha256": replay._sha(path),
        "run_sha256": replay._sha(run_path), "source_archive_sha256": REFERENCE_ARCHIVE_SHA256,
        "historical_gate_approval": False}


def selected_indices(count):
    """Fixed quartiles for trunk, midpoint/end for each short affinity stack."""
    if type(count) is not int or count < 1:
        raise ValueError("Require a nonempty native stack")
    return sorted({max(0, math.ceil(count * fraction) - 1) for fraction in (.25, .5, .75, 1.)})


def affinity_stage_indices(counts):
    """Pair logical midpoint/end stages of each member's own native depth.

    The pretrained ensemble legitimately uses different stack lengths. Equal
    block indices do not mean equal progress through those different stacks.
    A one-block member contributes that same physical boundary to both logical
    cuts; install its hook once, while retaining both declared frontier stages.
    An empty native stack is an identity: its already-captured conditioned input
    supplies both stages, without inventing a layer or changing native math.
    """
    if len(counts) != 2 or any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Require two native affinity stacks with nonnegative depths")
    return {"midpoint": tuple(math.ceil(count / 2) - 1 if count else None for count in counts),
            "end": tuple(count - 1 if count else None for count in counts)}


def tensor_metadata(values):
    from latentmol.multimodel.study import tensor_digest
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": tensor_digest({name: value}), "norm_float64": float(value.detach().double().norm())}
        for name, value in values.items()}


def chain_completeness(direct, chain):
    """Existing diagnostic engineering bound; never an objective FD tolerance."""
    import torch
    delta = direct.double() - chain.double()
    relative = 64 * torch.finfo(direct.dtype).eps
    maximum_bound = relative * max(float(direct.double().abs().max()), 1e-12)
    norm_bound = relative * max(float(direct.double().norm()), 1e-12)
    maximum, norm = float(delta.abs().max()), float(delta.norm())
    return {"exact_equal": torch.equal(direct, chain), "max_absolute_difference": maximum,
        "difference_norm_float64": norm, "maximum_absolute_bound": maximum_bound,
        "norm_error_bound": norm_bound,
        "engineering_chain_complete": maximum <= maximum_bound and norm <= norm_bound,
        "criterion": "Maximum and L2 error <=64 native-dtype eps times direct scale",
        "historical_fd_tolerance_changed": False}


def affinity_loss(outputs, config):
    """The original source091 primitive widening and scalar order, unchanged."""
    profile = config.dbmol
    if profile.objective_arithmetic != "contact_and_assembly_float64_v1":
        raise ValueError("Observer requires source091's declared scalar precision")
    # Same expression as source091 reduce_objective. This is the affinity
    # component, before configured whole-objective scaling or repair addition.
    return (profile.affinity_regression_weight * outputs["affinity_pred_value"].double().reshape(())
            - profile.affinity_logit_weight * outputs["affinity_logits_binary"].double().reshape(()))


class BoundaryCapture:
    """Ordered complete frontiers with independent identity-clone outputs.

    Hook counters reject recycling/repeated native calls rather than silently
    invent a cut for a changed topology. Checkpoint recomputation invokes each
    block's original bound forward, so outer module hooks fire once. ReLU hooks
    are attached only to the uncheckpointed atom encoder and native head MLPs.
    """
    def __init__(self, model, *, telemetry=True):
        self.model, self.telemetry = model, telemetry
        self.values, self.relus, self.handles, self.visits = {}, {}, [], {}
        self.embedding_call = None
        self.members = (model.affinity_module1, model.affinity_module2)
        self.trunk_indices = selected_indices(len(model.pairformer_module.layers))
        counts = [len(member.pairformer_stack.layers) for member in self.members]
        self.affinity_stages = affinity_stage_indices(counts)
        self.affinity_indices_by_member = [sorted({indices[index] for indices in self.affinity_stages.values()
                                                   if indices[index] is not None})
                                           for index in range(2)]
        self.topology = {"trunk_depth": len(model.pairformer_module.layers),
            "trunk_block_indices": self.trunk_indices, "affinity_member_depths": counts,
            "affinity_stage_block_indices": self.affinity_stages,
            "affinity_hook_block_indices_by_member": self.affinity_indices_by_member,
            "logical_stage_pairing": "Each member's own midpoint and end; equal depths are not assumed"}
        self.cuts = {"graph": ["graph." + field for field in FIELDS],
            "adapter": ["adapter.ref_element", "adapter.ref_charge", "adapter.token_bonds", "adapter.bond_types"]}
        bypass = ["adapter.token_bonds", "adapter.bond_types"]
        self.cuts["atom_encoder"] = [f"atom_encoder.{mode}.{name}" for mode in ("trunk", "affinity")
                                     for name in ("q", "c", "p")] + bypass
        self.cuts["input_embedder"] = ["embedder.trunk", "embedder.affinity"] + bypass
        for index in self.trunk_indices:
            self.cuts[f"trunk_block_{index:02d}"] = [f"trunk.{index}.s", f"trunk.{index}.z", "embedder.affinity"]
        self.cuts["affinity_inputs"] = [f"member.{i}.input.{field}" for i in range(2) for field in ("s_inputs", "z")]
        self.cuts["affinity_conditioned"] = [f"member.{i}.conditioned" for i in range(2)]
        for stage, indices in self.affinity_stages.items():
            self.cuts["affinity_" + stage] = [f"member.{i}.block.{index}" if index is not None
                                            else f"member.{i}.conditioned" for i, index in enumerate(indices)]
        self.cuts["pooled"] = [f"member.{i}.pooled" for i in range(2)]
        self.cuts["shared_head_mlp"] = [f"member.{i}.shared" for i in range(2)]

    def save(self, name, value):
        if name in self.values:
            raise ValueError("Repeated boundary: " + name)
        if not value.is_floating_point():
            raise ValueError("Only floating frontiers are captured")
        clone = value.clone()
        self.values[name] = clone
        return clone

    def _embedding_pre(self, _module, _args, kwargs):
        self.embedding_call = "affinity" if kwargs.get("affinity", False) else "trunk"

    def _embedding_post(self, _module, _args, _kwargs, output):
        mode, self.embedding_call = self.embedding_call, None
        return self.save("embedder." + mode, output)

    def _atom_encoder(self, _module, _args, output):
        if self.embedding_call not in ("trunk", "affinity") or not isinstance(output, tuple) or len(output) != 4:
            raise ValueError("Unexpected native atom-encoder call/output")
        return (*[self.save(f"atom_encoder.{self.embedding_call}.{name}", value)
                  for name, value in zip(("q", "c", "p"), output[:3])], output[3])

    def _relu(self, name, _module, args):
        key = (f"{self.embedding_call}." + name) if name.startswith("atom.") else name
        if key in self.relus or len(args) != 1:
            raise ValueError("Repeated/unexpected uncheckpointed ReLU: " + key)
        self.relus[key] = args[0].detach()

    def __enter__(self):
        import torch
        from functools import partial
        if self.model.training or not self.model.affinity_ensemble:
            raise ValueError("Require the original frozen two-member eval ensemble")
        self.handles += [self.model.input_embedder.register_forward_pre_hook(self._embedding_pre, with_kwargs=True),
            self.model.input_embedder.register_forward_hook(self._embedding_post, with_kwargs=True),
            self.model.input_embedder.atom_encoder.register_forward_hook(self._atom_encoder)]
        for index in self.trunk_indices:
            def trunk_hook(_module, _args, output, index=index):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError("Unexpected native pairformer output")
                return tuple(self.save(f"trunk.{index}.{name}", value) for name, value in zip(("s", "z"), output))
            self.handles.append(self.model.pairformer_module.layers[index].register_forward_hook(trunk_hook))
        for member_index, member in enumerate(self.members):
            def input_hook(_module, args, kwargs, index=member_index):
                if args or set(kwargs) != {"s_inputs", "z", "x_pred", "feats", "multiplicity", "use_kernels"}:
                    raise ValueError("Unexpected native affinity-module interface")
                return args, {**kwargs, **{name: self.save(f"member.{index}.input.{name}", kwargs[name])
                                         for name in ("s_inputs", "z")}}
            self.handles.append(member.register_forward_pre_hook(input_hook, with_kwargs=True))
            def stack_pre(_module, args, kwargs, index=member_index):
                if len(args) != 1:
                    raise ValueError("Unexpected native affinity stack interface")
                return (self.save(f"member.{index}.conditioned", args[0]),), kwargs
            self.handles.append(member.pairformer_stack.register_forward_pre_hook(stack_pre, with_kwargs=True))
            for block_index in self.affinity_indices_by_member[member_index]:
                def stack_hook(_module, _args, output, index=member_index, block=block_index):
                    return self.save(f"member.{index}.block.{block}", output)
                self.handles.append(member.pairformer_stack.layers[block_index].register_forward_hook(stack_hook))
            def pool_pre(_module, args, index=member_index):
                if len(args) != 1:
                    raise ValueError("Unexpected native head input")
                return (self.save(f"member.{index}.pooled", args[0]),)
            def shared_hook(_module, _args, output, index=member_index):
                return self.save(f"member.{index}.shared", output)
            self.handles += [member.affinity_heads.affinity_out_mlp.register_forward_pre_hook(pool_pre),
                             member.affinity_heads.affinity_out_mlp.register_forward_hook(shared_hook)]
            if self.telemetry:
                for name, module in member.affinity_heads.named_modules():
                    if isinstance(module, torch.nn.ReLU):
                        self.handles.append(module.register_forward_pre_hook(partial(self._relu, f"head.{member_index}.{name}")))
        if self.telemetry:
            for name, module in self.model.input_embedder.atom_encoder.named_modules():
                if isinstance(module, torch.nn.ReLU):
                    self.handles.append(module.register_forward_pre_hook(partial(self._relu, "atom." + name)))
        return self

    def __exit__(self, *_args):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def require_complete_membership(self):
        expected = {name for fields in self.cuts.values() for name in fields}
        if set(self.values) != expected:
            raise ValueError("Boundary membership differs: " + repr(sorted(expected ^ set(self.values))))


def native_forward(decoder, point, branch, config, counters, capture, label):
    """Use unchanged production decoder/graph policy/adapter/native affinity."""
    from latentmol.boltz.active_set import graph_from_decoder
    from latentmol.boltz.adapter import adapt_graph
    from latentmol.representation import RelaxedGraph
    from latentmol.multimodel.study import tensor_digest
    graph = graph_from_decoder(decoder.decode(point), decoder.spec,
                               fixed_active=branch.prepared.active, temperature=config.temperature)
    if capture is not None:
        graph = RelaxedGraph(*(capture.save("graph." + name, getattr(graph, name)) for name in FIELDS))
    support = branch.assess(graph)
    if support.status == "unsupported" or (support.status == "lossy" and not branch.allow_lossy_charge):
        raise ValueError("Same native branch rejected the shared graph")
    features, bonds = adapt_graph(graph, branch.prepared)
    if capture is not None:
        features = {**features, **{name: capture.save("adapter." + name, features[name])
                                  for name in ("ref_element", "ref_charge", "token_bonds")}}
        bonds = capture.save("adapter.bond_types", bonds)
    outputs = counters.call("boltz", label, "forward", lambda:
        branch.scorer.affinity(features, bonds, branch.prepared.coordinates), point.device)
    if capture is not None:
        capture.require_complete_membership()
    return outputs, affinity_loss(outputs, config), tensor_digest({name: getattr(graph, name) for name in FIELDS})


def phase(label, operation, device, report, persist):
    """One fresh graph per cut; detached return only and durable failure progress."""
    import torch
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    row = {"phase": label, "status": "running", "allocated_before": torch.cuda.memory_allocated(device)}
    report.setdefault("phases", []).append(row)
    persist()
    try:
        result = operation()
        assert_detached(result)
        gc.collect()
        torch.cuda.synchronize(device)
        row.update(status="completed", elapsed_seconds=time.perf_counter() - start,
            allocated_after_graph_release=torch.cuda.memory_allocated(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device), no_live_graph_returned=True)
        persist()
        return result
    except Exception as error:
        row.update(status="failed", error=str(error), error_type=type(error).__name__)
        persist()
        raise


def assert_detached(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("Native autograd graph escaped diagnostic phase")
    elif isinstance(value, dict):
        for child in value.values():
            assert_detached(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_detached(child)


def same_values(observed, expected, scope):
    import torch
    if set(observed) != set(expected) or any(not torch.equal(observed[k].detach(), expected[k]) for k in expected):
        raise ValueError("Original boundary values changed: " + scope)
    # tensor digests distinguish signed zero and include shape/dtype.
    if tensor_metadata(observed) != tensor_metadata(expected):
        raise ValueError("Original boundary bytes changed: " + scope)


def relu_crossings(current, baseline):
    """Passive branch changes are evidence of kinks, not proof of their effect."""
    import torch
    if set(current) != set(baseline):
        raise ValueError("Native ReLU telemetry membership changed")
    return {name: {"elements": value.numel(),
        "positive_branch_changes": int(torch.count_nonzero((value > 0) != (baseline[name] > 0))),
        "exact_zero_at_baseline": int(torch.count_nonzero(baseline[name] == 0)),
        "baseline_min_abs": float(baseline[name].abs().min()),
        "maximum_preactivation_change": float((value - baseline[name]).abs().max())}
        for name, value in current.items()}


def qualify_residual(native, chain):
    """Require chain accumulation differences to be <1% of both FD residuals."""
    discrepancy = max(abs(native[name] - chain[name]) for name in (
        "analytic", "analytic_dot_float64", "actual_secant_linear_prediction"))
    separated = (native["informative"] and chain["informative"]
        and not native["passed_tolerance"] and not chain["passed_tolerance"]
        and discrepancy < .01 * native["absolute_error"] and discrepancy < .01 * chain["absolute_error"])
    return {"failure_localization_eligible": separated, "maximum_projected_chain_discrepancy": discrepancy,
        "residual_fraction_bound": .01, "original_fd_rows_unmodified": True}


def native_provenance(case, model):
    """Authenticate actual loaded module files against the live exported source."""
    import inspect
    files = {path: expected for path, expected in case["parent_artifacts_sha256"].items()
             if "/site-packages/boltz/model/" in path}
    if len(files) != 50:
        raise ValueError("Require all 50 freshly exported installed native model sources")
    modules = [model, model.input_embedder, model.input_embedder.atom_encoder,
               model.pairformer_module.layers[0], model.affinity_module1,
               model.affinity_module1.pairformer_stack, model.affinity_module1.affinity_heads]
    modules += [member.pairformer_stack.layers[0] for member in (model.affinity_module1, model.affinity_module2)
                if len(member.pairformer_stack.layers)]
    actual = sorted({str(Path(inspect.getfile(type(module))).resolve()) for module in modules})
    if any(path not in files for path in actual):
        raise ValueError("Loaded native model source did not resolve to pinned installed files")
    for path, expected in files.items():
        if helper()._sha(path) != expected:
            raise ValueError("Installed native source changed: " + path)
    return {"loaded_module_files": actual, "installed_source_files_sha256": files,
            "all_installed_model_sources_authenticated": True}


def first_interval(cut_rows):
    """Report the first eligible ordered crossing; ambiguity is retained."""
    previous = "latent_input"
    for name, row in cut_rows.items():
        native, chain = row["original_native_reference"], row["measured_chain_reference"]
        if row["attribution"]["failure_localization_eligible"]:
            return {"status": "first_eligible_residual_interval", "from": previous, "to": name,
                    "unique_operation_identified": False}
        if not (native["passed"] and chain["passed"]):
            return {"status": "ambiguous_before_localizable_interval", "boundary": name,
                    "unique_operation_identified": False}
        previous = name
    return {"status": "all_cut_readouts_pass", "remaining_region": "shared_head_mlp_to_native_affinity_scalar",
            "unique_operation_identified": False}


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item,
               *, persist=lambda: None, report=None):
    """Authenticate direction/full baseline, then replay one native affinity path."""
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.boltz.local_fd import EPSILONS, endpoint_row
    from latentmol.multimodel.study import tensor_digest
    report = report if report is not None else {}
    case = _CASE
    reference, reference_receipt = authenticated_reference(case, report)
    if config.recycling_steps != 0:
        raise ValueError("Complete frontier topology is declared for original zero recycling only")
    report.update(status="probing", diagnostic_completed=False, reference=reference_receipt,
        scope="Original affinity component along original failed-item directions; structural/contact paths not localized",
        finite_differences=[], cuts={}, maximum_affinity_forwards=MAXIMUM_AFFINITY_FORWARDS,
        original_gate_not_reclassified=True)
    report["native_provenance"] = native_provenance(case, branch.scorer.affinity.model)
    report["observer_module_sha256"] = helper()._sha(__file__)
    components = {}
    measured, total_gradient = score_at(decoder, point, branch, config, counters,
        "boundary_original_full_analytic", gradients=True, component_gradients=components,
        repair_target=target, repair_weight=weight)
    original_baseline = next(row for row in reference["baselines"] if row["grad_enabled"])
    if (measured["components"] != {k: original_baseline["values"][k] for k in measured["components"]}
            or measured["loss"] != original_baseline["values"]["combined"]
            or tensor_digest({"gradient": total_gradient}) != original_baseline["sum_gradient_identity"]["combined_gradient_sha256"]):
        raise ValueError("Original full native baseline/backward differs from source093")
    relevant = total_gradient if item_name == "combined" else components[item_name]
    directions = helper().reconstruct_directions(point, relevant, item)
    if any(metadata["replayed_sha256"] != reference["direction_replay"][name]["replayed_sha256"]
           for name, (_, metadata) in directions.items()):
        raise ValueError("Original direction differs from authenticated source093 replay")
    direct = components["dbmol_affinity_loss"].detach()
    report["direction_replay"] = {name: metadata for name, (_, metadata) in directions.items()}
    report["original_gradient_sha256"] = tensor_digest({"gradient": direct})
    del total_gradient, relevant, components
    persist()

    def capture_base():
        with torch.no_grad(), BoundaryCapture(branch.scorer.affinity.model) as capture:
            outputs, loss, graph_sha = native_forward(decoder, point, branch, config, counters, capture, "boundary_base")
            expected = reference["baselines"][0]
            if (float(loss) != expected["values"]["dbmol_affinity_loss"] or graph_sha != expected["graph_sha256"]
                    or any(float(value) != expected["values"][name] for name, value in outputs.items())):
                raise ValueError("Identity-clone native baseline differs from original production values")
            return ({name: value.detach().clone() for name, value in capture.values.items()},
                    {name: value.detach().clone() for name, value in capture.relus.items()},
                    deepcopy(capture.cuts), float(loss), deepcopy(capture.topology))
    base, relus, cuts, baseline, topology = phase("capture_original_boundaries", capture_base, point.device, report, persist)
    report["native_stack_topology"] = topology
    report["boundary_order"] = list(cuts)
    if len(cuts) > 14:
        raise ValueError("Native topology exceeds the declared fourteen-cut budget")
    report["boundary_tensors"] = tensor_metadata(base)
    report["baseline_relu_tensors"] = tensor_metadata(relus)
    chains, cotangents = {}, {}
    for cut_name, members in cuts.items():
        def obtain_cut():
            probe = point.detach().clone().requires_grad_()
            with torch.enable_grad(), BoundaryCapture(branch.scorer.affinity.model, telemetry=False) as capture:
                outputs, loss, _ = native_forward(decoder, probe, branch, config, counters, capture, "boundary_cut:" + cut_name)
                same_values(capture.values, base, cut_name)
                gradient, = counters.call("boltz", "boundary_direct:" + cut_name, "backward", lambda:
                    torch.autograd.grad(loss, probe, retain_graph=True), point.device)
                clone_parity = chain_completeness(direct, gradient.detach())
                if not clone_parity["engineering_chain_complete"]:
                    raise ValueError("Clone-frontier backward changed original derivative: " + cut_name)
                fields = tuple(capture.values[name] for name in members)
                raw = counters.call("boltz", "boundary_cotangent:" + cut_name, "backward", lambda:
                    torch.autograd.grad(loss, fields, retain_graph=True, allow_unused=True), point.device)
                fixed = tuple(torch.zeros_like(field) if value is None else value.detach()
                              for field, value in zip(fields, raw))
                chain, = counters.call("boltz", "boundary_chain:" + cut_name, "backward", lambda:
                    torch.autograd.grad(fields, probe, grad_outputs=fixed), point.device)
                if any(not bool(value.isfinite().all()) for value in (gradient, chain, *fixed)):
                    raise ValueError("Nonfinite complete-cut derivative")
                completeness = chain_completeness(direct, chain.detach())
                if not completeness["engineering_chain_complete"]:
                    raise ValueError("Declared frontier does not reproduce original derivative: " + cut_name)
                return chain.detach(), dict(zip(members, fixed)), {
                    "members": members, "unused_members": [name for name, value in zip(members, raw) if value is None],
                    "identity_clone_backward_parity": clone_parity, "direct_vs_chain": completeness,
                    "chain_sha256": tensor_digest({"gradient": chain}), "cotangents": tensor_metadata(dict(zip(members, fixed)))}
        chains[cut_name], cotangents[cut_name], report["cuts"][cut_name] = phase(
            "cotangent:" + cut_name, obtain_cut, point.device, report, persist)
        persist()

    def endpoint(value, label):
        with torch.no_grad(), BoundaryCapture(branch.scorer.affinity.model) as capture:
            outputs, loss, graph_sha = native_forward(decoder, value, branch, config, counters, capture, label)
            readouts = {name: float(sum(((capture.values[key].double() - base[key].double()) * fixed.double()).sum()
                        for key, fixed in cotangent.items())) for name, cotangent in cotangents.items()}
            return {"affinity_loss": float(loss), "loss_dtype": str(loss.dtype),
                "outputs": {name: float(output) for name, output in outputs.items()}, "graph_sha256": graph_sha,
                "readouts": readouts, "relu_crossings": relu_crossings(capture.relus, relus)}

    for direction_name, (direction, _) in directions.items():
        for epsilon in EPSILONS:
            plus = endpoint(point + epsilon * direction, "boundary_endpoint_plus")
            minus = endpoint(point - epsilon * direction, "boundary_endpoint_minus")
            previous = [row for row in reference["finite_differences"] if row["direction"] == direction_name
                        and row["epsilon"] == epsilon and row["grad_enabled"] is False]
            if len(previous) != 1:
                raise ValueError("Source093 does not contain declared endpoint")
            for name, observation in (("plus", plus), ("minus", minus)):
                original = previous[0][name]
                if (observation["graph_sha256"] != original["graph_sha256"]
                        or observation["affinity_loss"] != original["values"]["dbmol_affinity_loss"]
                        or any(value != original["values"][key] for key, value in observation["outputs"].items())):
                    raise ValueError("Instrumented affinity endpoint differs from original full DBMol endpoint")
            native_row = endpoint_row(point, direction, direct, epsilon,
                {"loss": plus["affinity_loss"], "loss_dtype": plus["loss_dtype"]},
                {"loss": minus["affinity_loss"], "loss_dtype": minus["loss_dtype"]}, baseline,
                rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
            cut_rows = {}
            for cut_name in cuts:
                endpoints = [{"loss": row["readouts"][cut_name], "loss_dtype": "torch.float64"} for row in (plus, minus)]
                # A cut's detached linear readout has its own measured VJP. Both
                # original-native and chain references are retained unchanged.
                native = endpoint_row(point, direction, direct, epsilon, *endpoints, 0.,
                    rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
                chain = endpoint_row(point, direction, chains[cut_name], epsilon, *endpoints, 0.,
                    rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
                cut_rows[cut_name] = {"original_native_reference": native, "measured_chain_reference": chain,
                    "attribution": qualify_residual(native, chain)}
            report["finite_differences"].append({"direction": direction_name, "epsilon": epsilon,
                "native_affinity": native_row, "cuts": cut_rows, "plus": plus, "minus": minus,
                "source093_endpoint_parity": True, "ordered_localization": first_interval(cut_rows)})
            persist()
    report.update(status="completed", diagnostic_completed=True, old_gate_certified_by_this_diagnostic=False,
        actual_affinity_forwards=1 + len(cuts) + 2 * len(directions) * len(EPSILONS),
        full_dbmol_analytic_forwards=1, new_defog_calls=0, optimizer_updates=0,
        interpretation="Compare the fixed ordered complete-cut residuals and native ReLU branch telemetry; no unique operation or remedy is assumed.")
    persist()
    return report


_CASE = None


def run_case(plan, case, output_root):
    global _CASE
    _CASE = case
    replay = helper()
    replay.probe_case = probe_case
    replay.CASE_TIMEOUT_SECONDS, replay.REPEATS, replay.CLAIM_LIMIT = CASE_TIMEOUT_SECONDS, 1, CLAIM_LIMIT
    return replay.run_case(plan, case, output_root)


def validate_cases(plan):
    if ([case["id"] for case in plan["cases"]] != ["boltz_only", "boltz_defog"]
            or any(case["step"] != 15 or case["arm"] != "fixed_initial_mask" for case in plan["cases"])):
        raise ValueError("Require the original two source091 step15 failures")
    for case in plan["cases"]:
        reference_paths(case)
    return plan


def forward_budget(direction_counts):
    """Enforce the finite endpoint/cut plan before starting either native child.

    Six auxiliary affinity traversals occur inside original full DBMol/parity
    calls (one original analytic and two hard-input parity calls per case).
    They are reported separately and included in the total upper bound.
    """
    if (len(direction_counts) != 2 or any(type(count) is not int or not 2 <= count <= 4
                                        for count in direction_counts)):
        raise ValueError("Require two original bounded direction counts")
    explicit = sum(1 + 14 + 2 * 9 * count for count in direction_counts)
    if explicit > MAXIMUM_AFFINITY_FORWARDS:
        raise ValueError("Original direction set exceeds the bounded affinity-call budget")
    return {"planned_explicit_affinity_forwards_upper_bound": explicit,
            "maximum_explicit_affinity_forwards": MAXIMUM_AFFINITY_FORWARDS,
            "maximum_auxiliary_full_dbmol_forwards": 6,
            "maximum_total_affinity_traversals": explicit + 6,
            "child_seconds_upper_bound": 2 * CASE_TIMEOUT_SECONDS,
            "gpu_wall_seconds": 7200, "controller_headroom_seconds": 7200 - 2 * CASE_TIMEOUT_SECONDS}


def execution_completion(execution):
    """A child exit code alone cannot certify persisted observer completion."""
    if execution.get("timed_out") is not False or execution.get("returncode") != 0:
        return False, "child_failed_or_timed_out"
    paths = execution.get("run_paths", [])
    if len(paths) != 1:
        return False, "missing_or_ambiguous_child_run"
    try:
        child = Path(paths[0])
        evidence = helper()._read(child / "attribution.json")
        record = helper()._read(child / "run.json")
        if not isinstance(evidence, dict) or not isinstance(record, dict):
            raise ValueError("Child completion evidence must be JSON objects")
        complete = (record.get("status") == "completed" and evidence.get("status") == "completed"
            and all(evidence.get(key) is True for key in ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged")))
        return complete, None if complete else "child_artifacts_do_not_establish_completion"
    except (OSError, ValueError, TypeError) as error:
        # Retain this failure and run the other prespecified case. The final
        # controller still raises, so incomplete artifacts cannot yield DONE.
        return False, "child_artifact_read_failed: " + str(error)


def require_completed_cases(rows):
    """Raise only after both prescribed receipts were retained by the caller."""
    if [row.get("case_id") for row in rows] != ["boltz_only", "boltz_defog"]:
        raise RuntimeError("Operation-boundary diagnostic cases missing or reordered")
    incomplete = [row["case_id"] for row in rows if row.get("diagnostic_completed") is not True
                  or row["execution"].get("returncode") != 0 or row["execution"].get("timed_out") is not False]
    if incomplete:
        raise RuntimeError("Operation-boundary diagnostics incomplete: " + ", ".join(incomplete))


def run_study(path, expected):
    plan = validate_cases(helper().load_plan(path, expected))
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    direction_counts = []
    for case in plan["cases"]:
        gate_path = Path(case["parent_run"]) / case["arm"] / "gate-0015.json"
        _, failed = helper().failed_item(helper()._read(helper().authenticated_artifact(case, gate_path)))
        randoms = sum(name != "gradient" and row.get("passed") is False
                      for name, row in failed["direction_results"].items())
        direction_counts.append(1 + randoms)
    budget = forward_budget(direction_counts)
    inputs = {"plan": plan, "plan_sha256": expected, "diagnostic_module_sha256": helper()._sha(__file__),
        "forward_budget": budget,
        "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], inputs, Path.cwd()) as run:
        rows = []
        for case in plan["cases"]:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / (case["id"] + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
            completed, incomplete_reason = execution_completion(execution)
            rows.append({"case_id": case["id"], "execution": execution, "diagnostic_completed": completed,
                "incomplete_reason": incomplete_reason, "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        run.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
            diagnostic_cases_requested=2, old_gates_certified=False)
        # Print the retained path before raising, for discovery by enclosing
        # logged-process wrappers. ExperimentRun.__exit__ records a failed
        # controller when any child is incomplete; main propagates nonzero.
        print(run.path, flush=True)
        require_completed_cases(rows)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not helper()._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Observer requires an authenticated allocated DTU HPC job")
    plan = validate_cases(helper().load_plan(args.plan, args.sha256))
    if args.child:
        cases = [case for case in plan["cases"] if case["id"] == args.child]
        if len(cases) != 1 or args.child_output is None:
            raise ValueError("Unknown child or missing unique output root")
        print(run_case(plan, cases[0], args.child_output), flush=True)
    else:
        if args.child_output is not None:
            raise ValueError("Child output requires --child")
        print(run_study(args.plan, args.sha256), flush=True)


if __name__ == "__main__":
    main()
