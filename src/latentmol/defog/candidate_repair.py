"""Frozen DeFoG late-time candidate continuation, outside every autograd path.

The pinned public sampler starts from unconditional prior noise. This wrapper
instead places the supplied hard graph at a declared late-time grid boundary
and calls the *same* native transition method on the remaining grid suffix.
It is a proposed repair procedure, not a verified projection or likelihood.

The official GuacaMol checkpoint has no formal-charge/absence output. Legacy
v1 rejects charged inputs. Explicit v2 projects charge away without changing
the candidate, and applies the pinned native charge rule to output references
only. Neither mode reindexes, crops fragments or changes sampled atom/bond
labels to pass chemistry checks. No neural charge prediction is claimed.

The JSON CLI runs in the isolated DeFoG environment inside an HPC allocation.
It provides a narrow process boundary so Boltz and DeFoG dependency versions do
not have to coexist. No backpropagation through sampling is claimed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from typing import Callable

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.defog.codec import (PROJECT_CHARGE_POLICY, STRICT_CHARGE_POLICY,
    from_guacamol_labels, from_guacamol_native_charges, projection_receipt, to_guacamol_hard)
from latentmol.defog.interface_smoke import (
    DEFOG_COMMIT, GUACAMOL_SHA256, _load_native_model, _verify_assets,
)
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec, HardGraphBatch


VERSION = 1
PROJECT_VERSION = 2
INITIALIZATIONS = {"direct_candidate_no_corruption", "checkpoint_forward_corruption"}
DISTORTIONS = {"identity", "cos", "revcos", "polyinc", "polydec"}


def request_sha256(request: dict) -> str:
    """Hash exact JSON transport content, excluding only its own receipt field."""
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


# Public transport helper used by the optimizer's subprocess provider.
request_digest = request_sha256


def graph_record(graph: HardGraphBatch) -> dict:
    """Serialize integer category labels with the canonical batch axis intact."""
    return {name: getattr(graph, name).detach().cpu().tolist()
            for name in ("atoms", "charges", "bonds")}


def graph_from_record(value: dict, spec: GraphSpec) -> HardGraphBatch:
    """Read full-capacity hard labels without rounding floats or dropping slots."""
    if not isinstance(value, dict) or set(value) != {"atoms", "charges", "bonds"}:
        raise ValueError("Candidate must contain exactly atoms, charges and bonds")
    def all_integer(row):
        return all(all_integer(item) if isinstance(item, list) else type(item) is int
                   for item in row) if isinstance(row, list) else False
    if any(not all_integer(value[name]) for name in value):
        raise ValueError("Candidate labels must be JSON integers, not floats or booleans")
    graph = HardGraphBatch(*(torch.tensor(value[name], dtype=torch.long)
                             for name in ("atoms", "charges", "bonds")))
    if (graph.atoms.shape != (1, spec.max_nodes) or graph.charges.shape != graph.atoms.shape
            or graph.bonds.shape != (1, spec.max_nodes, spec.max_nodes)):
        raise ValueError("Candidate must have one full-capacity canonical batch")
    return graph


def _validated_request(request: dict) -> tuple[GraphSpec, HardGraphBatch]:
    required = {"version", "request_sha256", "spec", "candidate", "occupied_slots",
                "seed", "t0", "steps", "source", "checkpoint"}
    allowed = required | {"initialization", "device", "sampling", "adapter_source_sha256", "charge_policy"}
    if not isinstance(request, dict) or required - set(request) or set(request) - allowed:
        raise ValueError("Unexpected or missing candidate-repair request fields")
    if type(request["version"]) is not int or request["version"] not in (VERSION, PROJECT_VERSION):
        raise ValueError("Unsupported candidate-repair request version")
    if ((request["version"] == VERSION and "charge_policy" in request)
            or (request["version"] == PROJECT_VERSION and request.get("charge_policy") != PROJECT_CHARGE_POLICY)):
        raise ValueError("Charge projection requires explicit v2 policy; legacy v1 stays unchanged")
    if request["request_sha256"] != request_sha256(request):
        raise ValueError("Candidate-repair request SHA256 mismatch")
    adapter_hash = request.get("adapter_source_sha256")
    if adapter_hash is not None and (not isinstance(adapter_hash, str) or len(adapter_hash) != 64
                                    or any(char not in "0123456789abcdef" for char in adapter_hash)):
        raise ValueError("Declared adapter source SHA256 must be lowercase hexadecimal")
    if (type(request["seed"]) is not int or not 0 <= request["seed"] < 2**63
            or type(request["steps"]) is not int or not 2 <= request["steps"] <= 10000):
        raise ValueError("Seed or total native sampling-grid resolution is out of range")
    if (isinstance(request["t0"], bool) or not isinstance(request["t0"], (int, float))
            or not math.isfinite(request["t0"]) or not 0 < request["t0"] < 1):
        raise ValueError("t0 must be a finite model time strictly between zero and one")
    if request.get("initialization", "direct_candidate_no_corruption") not in INITIALIZATIONS:
        raise ValueError("Unsupported candidate initialization policy")
    if request.get("device", "cuda") not in {"cuda", "cpu"}:
        raise ValueError("Worker device must be cuda or cpu")
    source, checkpoint = request["source"], request["checkpoint"]
    if (not isinstance(source, dict) or set(source) != {"path", "commit"}
            or source["commit"] != DEFOG_COMMIT
            or not isinstance(source["path"], str) or not source["path"]):
        raise ValueError("Only the pinned official DeFoG source is supported")
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}
            or checkpoint["sha256"] != GUACAMOL_SHA256
            or not isinstance(checkpoint["path"], str) or not checkpoint["path"]):
        raise ValueError("Only the audited public GuacaMol checkpoint is supported")
    sampling = request.get("sampling", {})
    if not isinstance(sampling, dict) or set(sampling) - {"time_distortion", "eta", "omega", "rdb", "rdb_crit"}:
        raise ValueError("Unknown native sampling settings")
    if sampling.get("time_distortion", "polydec") not in DISTORTIONS:
        raise ValueError("Unsupported native time distortion")
    for name, default in (("eta", 300.0), ("omega", 0.1)):
        number = sampling.get(name, default)
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number < 0):
            raise ValueError(f"Native {name} must be finite and nonnegative")
    # Restrict to native implemented RDB designs without their broken entry path.
    if sampling.get("rdb", "general") not in {"general", "marginal", "column"}:
        raise ValueError("Unsupported native RDB design")
    if "rdb_crit" in sampling and sampling["rdb_crit"] not in {
            "max_marginal", "x_t", "abs_state", "p_x1_g_xt", "x_1", "p_xt_g_x1", "xhat_t"}:
        raise ValueError("Unsupported native RDB criterion")
    spec = GraphSpec.from_dict(request["spec"])
    graph = graph_from_record(request["candidate"], spec)
    slots = request["occupied_slots"]
    if (not isinstance(slots, list) or not slots
            or any(type(slot) is not int or not 0 <= slot < spec.max_nodes for slot in slots)
            or slots != sorted(set(slots))):
        raise ValueError("Occupied slots must be sorted, unique canonical slot IDs")
    actual = (graph.atoms[0] != 0).nonzero(as_tuple=False).flatten().tolist()
    if actual != slots:
        raise ValueError("Candidate occupancy differs from the fixed occupied-slot contract")
    return spec, graph


def _state_sha256(model) -> str:
    """Record frozen state bytes; this is identity bookkeeping, not a model run."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def native_time_grid(model, *, steps: int, distortion: str, device) -> torch.Tensor:
    """Use the pinned sampler's float32 normalized grid and native distortion."""
    # Native sample_batch computes i*ones / steps, then sample_ft, separately at
    # every transition. This vectorized evaluation has identical float32 inputs.
    values = torch.arange(steps + 1, device=device, dtype=torch.float32)[:, None] / steps
    grid = model.time_distorter.sample_ft(values, distortion).flatten()
    if (not torch.isfinite(grid).all() or not bool((grid[1:] > grid[:-1]).all())
            or float(grid[0]) != 0 or float(grid[-1]) != 1):
        raise ValueError("Native distorted sampling grid is not strictly increasing from zero to one")
    return grid


class NativeTransitionContractError(ValueError):
    """A bounded, serializable receipt for the exact native return invariant."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("Native transition return contract failed: " + ", ".join(receipt["failed_checks"]))


def validate_native_transition(sampled, discrete, mapped) -> dict:
    """Validate pinned DeFoG's one-hot and collapsed views without modifying them.

    ``flow_matching_utils.sample_discrete_features`` keeps an upper triangle
    and reflects it, giving diagonal integer label0 (NONE). The native sampler
    then one-hot encodes EVERY pair, so the returned diagonal is NONE one-hot,
    not an all-zero vector. ``PlaceHolder.mask`` masks padding only; this codec
    has already removed padding. The initial clean codec intentionally has a
    zero diagonal, but subsequent native tensors must remain exactly native.

    Compare both returned views before casting their verified integer labels.
    This is an API/representation check, not RDKit validity or a repair claim.
    """
    expected = {"sampled_X": mapped.x.shape, "sampled_E": mapped.e.shape,
                "discrete_X": mapped.x.shape[:-1], "discrete_E": mapped.e.shape[:-1],
                "sampled_y": (1, 0), "discrete_y": (1, 0)}
    values = {"sampled_" + name: getattr(sampled, name, None) for name in ("X", "E", "y")}
    values.update({"discrete_" + name: getattr(discrete, name, None) for name in ("X", "E", "y")})
    receipt = {"version": 1, "expected_shapes": {name: list(shape) for name, shape in expected.items()},
               "observed_shapes": {name: list(value.shape) if isinstance(value, torch.Tensor) else None
                                   for name, value in values.items()},
               "observed_dtypes": {name: str(value.dtype) if isinstance(value, torch.Tensor) else None
                                   for name, value in values.items()},
               "diagonal_convention": "native_NONE_one_hot; collapsed_NONE_label_zero",
               "scope": "Unchanged native return tensors; no categorical resampling, rounding or chemical correction"}
    checks = {name + "_shape": isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(expected[name])
              for name, value in values.items()}

    def finish():
        receipt["checks"] = checks
        receipt["failed_checks"] = [name for name, passed in checks.items() if not passed]
        receipt["passed"] = not receipt["failed_checks"]
        if not receipt["passed"]:
            raise NativeTransitionContractError(receipt)
        return receipt

    if not all(checks.values()):
        return finish()  # Do not index/argmax malformed or absent output tensors.
    for name, value in values.items():
        checks[name + "_device"] = value.device == mapped.x.device
        checks[name + "_dtype"] = value.dtype == mapped.x.dtype
        checks[name + "_finite"] = bool(torch.isfinite(value).all())
    if not all(checks.values()):
        return finish()
    checks["sampled_X_one_hot"] = torch.equal(sampled.X, torch.nn.functional.one_hot(
        sampled.X.argmax(-1), mapped.x.shape[-1]).type_as(sampled.X))
    checks["sampled_E_one_hot"] = torch.equal(sampled.E, torch.nn.functional.one_hot(
        sampled.E.argmax(-1), mapped.e.shape[-1]).type_as(sampled.E))
    checks["sampled_E_symmetric"] = torch.equal(sampled.E, sampled.E.transpose(1, 2))
    diagonal = torch.arange(mapped.e.shape[1], device=mapped.x.device)
    native_diagonal = sampled.E[:, diagonal, diagonal, :]
    expected_diagonal = torch.zeros_like(native_diagonal)
    expected_diagonal[..., 0] = 1  # Audited GuacaMol category0 is NONE.
    checks["sampled_E_diagonal_NONE_one_hot"] = torch.equal(native_diagonal, expected_diagonal)
    checks["discrete_X_integer_categories"] = torch.equal(discrete.X, discrete.X.round())
    checks["discrete_E_integer_categories"] = torch.equal(discrete.E, discrete.E.round())
    checks["discrete_X_category_range"] = bool(((discrete.X >= 0) & (discrete.X < mapped.x.shape[-1])).all())
    checks["discrete_E_category_range"] = bool(((discrete.E >= 0) & (discrete.E < mapped.e.shape[-1])).all())
    checks["discrete_X_matches_one_hot"] = torch.equal(discrete.X, sampled.X.argmax(-1).type_as(discrete.X))
    checks["discrete_E_matches_one_hot"] = torch.equal(discrete.E, sampled.E.argmax(-1).type_as(discrete.E))
    checks["discrete_E_symmetric"] = torch.equal(discrete.E, discrete.E.transpose(1, 2))
    checks["discrete_E_diagonal_NONE"] = bool((discrete.E[:, diagonal, diagonal] == 0).all())
    return finish()


@torch.no_grad()
def continue_candidate(model, mapped, request: dict, progress: dict,
                       publish: Callable[[dict], None] | None = None):
    """Continue one fixed-order candidate using only native transition routines.

    The total grid resolution is distinct from the remaining transition budget.
    t0 denotes distorted *model time*, not a fraction of the 1000-step grid.
    The nearest boundary at or after t0 is recorded. No pre-input validity gate
    prevents malformed-but-representable graphs from reaching DeFoG.
    """
    device = mapped.x.device
    sampling = request.get("sampling", {})
    original = {name: getattr(model.cfg.sample, name) for name in
                ("sample_steps", "time_distortion", "eta", "omega", "rdb", "rdb_crit")}
    effective = {"sample_steps": request["steps"],
                 "time_distortion": sampling.get("time_distortion", original["time_distortion"]),
                 "eta": sampling.get("eta", original["eta"]), "omega": sampling.get("omega", original["omega"]),
                 "rdb": sampling.get("rdb", original["rdb"]),
                 "rdb_crit": sampling.get("rdb_crit", original["rdb_crit"])}
    if (effective["time_distortion"] not in DISTORTIONS
            or effective["rdb"] not in {"general", "marginal", "column"}):
        raise ValueError("Checkpoint native schedule/RDB design is unsupported by this worker")
    for name, value in effective.items():
        setattr(model.cfg.sample, name, value)
    for name in ("eta", "omega", "rdb", "rdb_crit"):
        setattr(model.rate_matrix_designer, name, effective[name])
    model.sample_T = effective["sample_steps"]
    if bool(model.conditional) or model.output_dims["y"] != 0:
        raise ValueError("This worker supports only the audited unconditional GuacaMol model")
    grid = native_time_grid(model, steps=request["steps"],
                            distortion=effective["time_distortion"], device=device)
    boundaries = (grid[:-1] >= request["t0"]).nonzero(as_tuple=False).flatten()
    if not boundaries.numel():
        raise ValueError("No native grid boundary remains at or after t0")
    first = int(boundaries[0])
    progress.update(original_checkpoint_sampling=original, effective_sampling=effective,
                    requested_model_time=request["t0"], effective_start_time=float(grid[first]),
                    grid_start_index=first, planned_transitions=request["steps"] - first,
                    native_transitions=0, attempted_native_transitions=0, native_forward_calls=0,
                    cost_observation="partial_lower_bound_until_terminal_receipt",
                    initialization=request.get("initialization", "direct_candidate_no_corruption"),
                    trajectory=[])
    x, e, mask = mapped.x.clone(), mapped.e.clone(), mapped.node_mask
    y = x.new_zeros((1, 0))
    if progress["initialization"] == "checkpoint_forward_corruption":
        # Exact checkpoint training corruption, including its categorical draw.
        # This is an explicit additional intervention, never applied silently.
        noisy = model.apply_noise(x, e, y, mask, t=grid[first].reshape(1, 1))
        x, e, y = noisy["X_t"].float(), noisy["E_t"].float(), noisy["y_t"].float()
    initial_x, initial_e = x.argmax(-1), e.argmax(-1)
    progress["initial_native_labels"] = {"atoms": initial_x.cpu().tolist(),
                                         "bonds": initial_e.cpu().tolist()}
    progress["candidate_to_initialized_edits"] = _native_edits(
        mapped.x.argmax(-1), mapped.e.argmax(-1), initial_x, initial_e)
    hook = model.model.register_forward_pre_hook(
        lambda *_: progress.update(native_forward_calls=progress["native_forward_calls"] + 1))
    try:
        for index in range(first, request["steps"]):
            progress["attempted_native_transitions"] += 1
            sampled, discrete = model.sample_p_zs_given_zt(
                grid[index].reshape(1, 1), grid[index + 1].reshape(1, 1), x, e, y, mask)
            # Keep the exact native NONE one-hot diagonal, including when it
            # becomes the next denoiser input. Only the initial codec has zeros.
            try:
                contract = validate_native_transition(sampled, discrete, mapped)
            except NativeTransitionContractError as error:
                progress["last_native_transition_contract"] = {"grid_index": index + 1, **error.receipt}
                raise
            progress["last_native_transition_contract"] = {"grid_index": index + 1, **contract}
            next_x, next_e = discrete.X.long(), discrete.E.long()
            changes = _native_edits(x.argmax(-1), e.argmax(-1), next_x, next_e)
            progress["trajectory"].append({"grid_index": index + 1,
                                           "model_time": float(grid[index + 1]), **changes})
            progress["native_transitions"] += 1
            x, e, y = sampled.X, sampled.E, sampled.y
            if publish is not None and (progress["native_transitions"] % 10 == 0
                                        or index == request["steps"] - 1):
                publish(progress)
    finally:
        hook.remove()
    # GuacaMol marginal transition has no virtual classes. Keep the native final
    # ignore_virtual_classes call for explicit parity with the public sampler.
    collapsed = sampled.mask(mask, collapse=True)
    final_x, final_e, _ = model.noise_dist.ignore_virtual_classes(collapsed.X, collapsed.E, collapsed.y)
    return final_x.long(), final_e.long()


def _native_edits(before_x, before_e, after_x, after_e) -> dict:
    upper = torch.triu_indices(before_e.shape[1], before_e.shape[2], offset=1,
                               device=before_e.device)
    return {"atom_changes": int((before_x != after_x).sum()),
            "bond_changes": int((before_e[:, upper[0], upper[1]]
                                  != after_e[:, upper[0], upper[1]]).sum())}


def repair_request(request: dict, *, publish: Callable[[dict], None] | None = None,
                   model_loader=None, asset_verifier=None) -> dict:
    """Return one durable accepted/rejected/error receipt; never fake a repair.

    Loader/verifier injection exists only for engineering unit fixtures. The CLI
    always uses audited native checkpoint/source loading. Rejected target labels
    remain in diagnostic output; only accepted targets populate ``target``.
    """
    started = time.monotonic()
    header = request if isinstance(request, dict) else {}
    result = {"version": VERSION, "request_sha256": header.get("request_sha256"),
              "status": "error", "target": None, "occupied_slots": header.get("occupied_slots"),
              "slot_alignment": "native_order_preserved", "native_order_preserved": True, "cost": {
                  "native_transitions": 0, "attempted_native_transitions": 0,
                  "native_forward_calls": 0}, "validation_status": "implemented_unvalidated"}
    model = None
    state_before = None
    stage = "request_validation"
    try:
        spec, candidate = _validated_request(request)
        policy = request.get("charge_policy", STRICT_CHARGE_POLICY)
        projected = policy == PROJECT_CHARGE_POLICY
        result["version"] = request["version"]
        source, checkpoint = Path(request["source"]["path"]), Path(request["checkpoint"]["path"])
        stage = "asset_verification"
        (asset_verifier or _verify_assets)(source, checkpoint)
        result["provenance"] = {
            "source_commit": request["source"]["commit"],
            "checkpoint_sha256": request["checkpoint"]["sha256"],
            "adapter_source_sha256": request.get("adapter_source_sha256", file_sha256(Path(__file__))),
            "adapter_module_sha256": file_sha256(Path(__file__)),
            "codec_source_sha256": file_sha256(Path(__file__).with_name("codec.py")),
            "spec_fingerprint": spec.fingerprint(), "charged_input_policy": "reject",
            "input_sanitization": "none", "gradient_scope": "none_native_sampling",
            "source_url": f"https://github.com/manuelmlmadeira/DeFoG/tree/{DEFOG_COMMIT}",
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda,
            "determinism_policy": "strict torch algorithms; process-local declared seed"}
        if projected:
            from rdkit import rdBase
            result["provenance"].update(charged_input_policy=PROJECT_CHARGE_POLICY,
                output_charge_policy="pinned_native_rule_reference_only", rdkit_version=rdBase.rdkitVersion)
        result["candidate_sha256"] = hashlib.sha256(json.dumps(
            graph_record(candidate), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result["input_inspection"] = asdict(inspect_graph(
            candidate.atoms[0], candidate.charges[0], candidate.bonds[0], spec))
        try:
            mapped = to_guacamol_hard(candidate, spec, charge_policy=policy)
        except ValueError as error:
            result.update(status="rejected", reason="input_representation_unsupported", detail=str(error))
            return result
        if mapped.occupied_slots != tuple(request["occupied_slots"]):
            raise ValueError("Input codec changed canonical occupied-slot order")
        result["native_input_labels"] = {"atoms": mapped.x.argmax(-1).tolist(),
                                          "bonds": mapped.e.argmax(-1).tolist(),
                                          "occupied_slots": list(mapped.occupied_slots)}
        if projected:
            result["input_feature_loss"] = projection_receipt(candidate, spec, mapped)
        # No RDKit validity gate applies here: invalid schema-compatible inputs
        # are the proposed repair use case. Charge rejection is a feature gate.
        stage = "worker_runtime_setup"
        device = torch.device(request.get("device", "cuda"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Native DeFoG continuation requires the requested GPU allocation")
        random.seed(request["seed"])
        np.random.seed(request["seed"] % 2**32)
        torch.manual_seed(request["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(request["seed"])
        torch.use_deterministic_algorithms(True)
        stage = "model_load"
        model = (model_loader or _load_native_model)(source, checkpoint, device)
        model.eval().requires_grad_(False)
        state_before = _state_sha256(model)
        result["provenance"]["frozen_state_sha256_before"] = state_before
        # Native weights were trained under their dataset's node-count support.
        # A matching tensor shape alone does not authorize extrapolating outside
        # that support. Retain the unsupported attempt without calling the model.
        node_probabilities = model.node_dist.prob
        count = len(mapped.occupied_slots)
        if count >= len(node_probabilities) or float(node_probabilities[count]) <= 0:
            result.update(status="rejected", reason="input_atom_count_outside_checkpoint_support",
                          detail=f"Fixed atom count {count} has no checkpoint dataset support")
            return result
        result["provenance"]["fixed_atom_count"] = count
        result["provenance"]["checkpoint_node_count_probability"] = float(node_probabilities[count])
        from latentmol.defog.codec import DefogHardGraph
        mapped = DefogHardGraph(mapped.x.to(device), mapped.e.to(device),
                                mapped.node_mask.to(device), mapped.occupied_slots)
        # Seeding after construction isolates sampling from RNG consumed by
        # checkpoint class initialization. The process boundary owns this RNG.
        random.seed(request["seed"])
        np.random.seed(request["seed"] % 2**32)
        torch.manual_seed(request["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(request["seed"])
        stage = "native_continuation"
        final_x, final_e = continue_candidate(model, mapped, request, result["cost"], publish)
        stage = "output_validation"
        result["native_output_labels"] = {"atoms": final_x.cpu().tolist(), "bonds": final_e.cpu().tolist()}
        try:
            if projected:
                target, result["output_charge_inference"] = from_guacamol_native_charges(
                    final_x, final_e, spec, mapped.occupied_slots)
            else:
                target = from_guacamol_labels(final_x, final_e, spec, mapped.occupied_slots)
        except ValueError as error:
            result.update(status="rejected", reason="output_representation_unsupported", detail=str(error))
            return result
        roundtrip = to_guacamol_hard(target, spec, charge_policy=policy)
        if (roundtrip.occupied_slots != mapped.occupied_slots
                or not torch.equal(roundtrip.x.argmax(-1), final_x)
                or not torch.equal(roundtrip.e.argmax(-1), final_e)):
            raise ValueError("Native-to-canonical roundtrip changed labels or slot identity")
        result["gates"] = {"fixed_atom_count": True, "slot_alignment": True,
                           "native_label_roundtrip": True, "neutral_charge_subset": True}
        if projected:
            # Charge-blind label roundtrip is only X/E evidence. Inferred Q is
            # separately replayed and checked by the independent parent provider.
            result["gates"].pop("neutral_charge_subset")
            result["gates"].update(declared_input_charge_loss=True, native_output_charge_rule=True)
            from latentmol.multimodel.study import hard_digest
            result["target_sha256"] = hard_digest(target)
        inspection = inspect_graph(target.atoms[0], target.charges[0], target.bonds[0], spec)
        result["output_inspection"] = asdict(inspection)
        result["proposed_target"] = graph_record(target)
        edits = _native_edits(candidate.atoms, candidate.bonds, target.atoms.cpu(), target.bonds.cpu())
        edits["charge_changes"] = int((candidate.charges != target.charges.cpu()).sum())
        result["edits"] = edits
        identity_preserved = not projected or result["output_charge_inference"]["sanitization_identity"]["passed"]
        if projected:
            result["gates"]["output_identity_preserved"] = identity_preserved
        if not identity_preserved:
            result.update(status="rejected", reason="output_sanitization_changes_or_rejects_declared_labels",
                          detail=result["output_charge_inference"]["sanitization_identity"])
            result["gates"]["chemical_validity"] = False
        elif inspection.smiles is None:
            result.update(status="rejected", reason="output_chemical_invalid", detail=inspection.failure_reason)
            result["gates"]["chemical_validity"] = False
        else:
            result.update(status="accepted", reason="chemical_and_representation_gates_passed",
                          target=graph_record(target), canonical_smiles=inspection.smiles)
            result["gates"]["chemical_validity"] = True
    except Exception as error:
        result.update(status="error", reason="native_execution_or_contract_failure",
                      failure_stage=stage,
                      error_type=type(error).__name__, detail=str(error),
                      traceback=traceback.format_exc(limit=10))
    finally:
        if model is not None and state_before is not None:
            state_after = _state_sha256(model)
            result["provenance"]["frozen_state_sha256_after"] = state_after
            result["provenance"]["frozen_state_unchanged"] = state_after == state_before
            if state_after != state_before or any(parameter.requires_grad for parameter in model.parameters()):
                result.update(status="error", reason="frozen_model_state_changed", target=None)
        result["cost"]["wall_seconds"] = time.monotonic() - started
        result["cost"]["cost_observation"] = "terminal_receipt_exact_worker_counts"
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    """Keep completed receipts parseable if a parent process times out midway."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Native DeFoG worker requires a DTU HPC allocation")
    # The direct Python subprocess invocation does not pass through the shell
    # worker. Set this before *any* CUDA initialization/model construction here.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise RuntimeError("Strict deterministic DeFoG requires a supported CUBLAS workspace configuration")
    if args.output.exists():
        raise FileExistsError("Candidate-repair receipts are immutable; use a unique output path")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_name(args.output.name + ".progress.json")
    result = repair_request(request, publish=lambda value: _atomic_json(progress_path, {
        "version": request.get("version", VERSION), "request_sha256": request.get("request_sha256"),
        "status": "running", "cost": value}))
    result["hpc_job_id"] = os.environ["LSB_JOBID"]
    _atomic_json(args.output, result)
    # A rejected chemical target is an accounted-for attempt, not a worker crash.
    # Errors receive a result receipt and a nonzero process code for the parent.
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
