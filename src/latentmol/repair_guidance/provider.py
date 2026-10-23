"""Sample the shared fixed graph and ask a frozen hard repair worker for targets.

Sampling, subprocesses, target selection and RDKit checks run outside autograd.
Every prescribed attempt is retained, even when an earlier one succeeds. The
first accepted attempt is selected without looking at any Boltz score. A failed
refresh returns no target; retaining an old target would change the stated
alternating method. This module never rewrites the graph sent to Boltz.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from latentmol.chemistry import inspect_graph
from latentmol.defog.assets import DEFOG_COMMIT, GUACAMOL_SHA256
from latentmol.defog.codec import (PROJECT_CHARGE_POLICY, STRICT_CHARGE_POLICY,
    from_guacamol_native_charges, projection_receipt, to_guacamol_hard)
from latentmol.experiments import file_sha256, write_json
from latentmol.graph_policy import graph_from_decoder
from latentmol.multimodel.study import hard_digest, tensor_digest
from latentmol.representation import HardGraphBatch, RelaxedGraph
from latentmol.sampling import sample_fixed_relaxed_graph
from latentmol.scoring import require_frozen_eval
from .config import RepairGuidanceConfig
from .loss import FrozenRepairTarget


class RepairProviderContractError(ValueError):
    """An authenticated oracle contract changed; caller must stop the experiment."""


def request_digest(request: dict) -> str:
    """Same pure JSON digest as the isolated worker, with no native imports."""
    value = {key: item for key, item in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _seed(base, step, attempt, phase):
    payload = json.dumps([base, step, attempt, phase], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _graph_record(graph):
    return {name: getattr(graph, name).detach().cpu().tolist()
            for name in ("atoms", "charges", "bonds")}


def _target_from_record(value, spec):
    if not isinstance(value, dict) or set(value) != {"atoms", "charges", "bonds"}:
        raise RepairProviderContractError("Worker target lacks exact canonical graph fields")
    def integer_lists(row):
        return (all(integer_lists(item) if isinstance(item, list) else type(item) is int
                    for item in row) if isinstance(row, list) else False)
    if any(not integer_lists(value[name]) for name in value):
        raise RepairProviderContractError("Worker target labels must be JSON integers")
    try:
        hard = HardGraphBatch(*(torch.tensor(value[name], dtype=torch.long)
                                for name in ("atoms", "charges", "bonds")))
    except (TypeError, ValueError, RuntimeError) as error:
        raise RepairProviderContractError("Worker target is not a rectangular hard graph") from error
    if (hard.atoms.shape != (1, spec.max_nodes) or hard.charges.shape != hard.atoms.shape
            or hard.bonds.shape != (1, spec.max_nodes, spec.max_nodes)):
        raise RepairProviderContractError("Worker target changed canonical capacity/batch shape")
    for name, categories in (("atoms", len(spec.atomic_numbers)),
                             ("charges", len(spec.formal_charges)), ("bonds", len(spec.bond_names))):
        if bool(((getattr(hard, name) < 0) | (getattr(hard, name) >= categories)).any()):
            raise RepairProviderContractError("Worker target contains unsupported category labels")
    return hard


class DeFoGRepairProvider:
    """Fixed-quota adapter with default neutral and explicit lossy charge modes.

    The optional runner is an engineering-fixture seam only; production uses
    subprocess.run with a distinct interpreter and inherited immutable source.
    Config/state identities are receipts, not evidence of native repair efficacy.
    """

    def __init__(self, config, source_sha256, *, runner=None):
        self.config = config
        if (not isinstance(source_sha256, str) or len(source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in source_sha256)):
            raise ValueError("Repair provider requires the immutable scientific source SHA256")
        self.source_sha256 = source_sha256
        self.runner = runner or subprocess.run
        self.module_path = Path(__file__).parents[1] / "defog" / "candidate_repair.py"
        self.codec_path = self.module_path.with_name("codec.py")
        self.expected_module_sha256 = file_sha256(self.module_path)
        self.expected_codec_sha256 = file_sha256(self.codec_path)

    def _verify_response(self, response, request, spec, candidate, active, process_returncode):
        """Authenticate common receipts before considering a target's chemistry."""
        policy = self.config.provider.get("charge_policy", STRICT_CHARGE_POLICY)
        projected = policy == PROJECT_CHARGE_POLICY
        version = 2 if projected else 1
        if (request.get("version") != version or
                (projected and request.get("charge_policy") != policy) or
                (not projected and "charge_policy" in request)):
            raise RepairProviderContractError("Request changed the configured charge policy/version")
        if (not isinstance(response, dict) or type(response.get("version")) is not int or response.get("version") != version
                or response.get("request_sha256") != request["request_sha256"]
                or response.get("status") not in {"accepted", "rejected", "error"}):
            raise RepairProviderContractError("Worker response version/request/status mismatch")
        status = response["status"]
        if (response.get("occupied_slots") != request["occupied_slots"]
                or response.get("native_order_preserved") is not True
                or response.get("slot_alignment") != "native_order_preserved"):
            raise RepairProviderContractError("Worker changed the fixed native slot-alignment contract")
        if status == "error" and response.get("failure_stage") != "native_continuation":
            raise RepairProviderContractError("Worker configuration/provenance/representation gate failed: "
                                              + str(response.get("detail")))
        provenance = response.get("provenance")
        expected = {"source_commit": DEFOG_COMMIT, "checkpoint_sha256": GUACAMOL_SHA256,
                    "adapter_source_sha256": self.source_sha256,
                    "adapter_module_sha256": self.expected_module_sha256,
                    "codec_source_sha256": self.expected_codec_sha256,
                    "spec_fingerprint": spec.fingerprint(), "charged_input_policy": "reject",
                    "input_sanitization": "none", "gradient_scope": "none_native_sampling"}
        if projected:
            expected.update(charged_input_policy=policy, output_charge_policy="pinned_native_rule_reference_only")
        if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected.items()):
            raise RepairProviderContractError("Worker source/checkpoint/feature provenance mismatch")
        candidate_hash = hashlib.sha256(json.dumps(
            _graph_record(candidate), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if response.get("candidate_sha256") != candidate_hash:
            raise RepairProviderContractError("Worker repaired a different sampled candidate")
        if projected:
            try:
                projection = to_guacamol_hard(candidate, spec, charge_policy=policy)
            except ValueError as error:
                if status != "rejected" or response.get("reason") != "input_representation_unsupported":
                    raise RepairProviderContractError("Worker accepted an unsupported canonical input") from error
            else:
                if response.get("input_feature_loss") != projection_receipt(candidate, spec, projection):
                    raise RepairProviderContractError("Worker charge-loss/projection receipt disagrees with original candidate")
                native_input = response.get("native_input_labels", {})
                if native_input != {"atoms": projection.x.argmax(-1).tolist(),
                                    "bonds": projection.e.argmax(-1).tolist(),
                                    "occupied_slots": request["occupied_slots"]}:
                    raise RepairProviderContractError("Worker changed projected native input labels")
        cost = response.get("cost")
        if not isinstance(cost, dict) or cost.get("cost_observation") != "terminal_receipt_exact_worker_counts":
            raise RepairProviderContractError("Worker lacks a terminal cost receipt")
        for key in ("native_transitions", "attempted_native_transitions", "native_forward_calls"):
            if type(cost.get(key)) is not int or cost[key] < 0:
                raise RepairProviderContractError("Worker transition/forward counters are invalid")
        if cost["native_transitions"] > cost["attempted_native_transitions"]:
            raise RepairProviderContractError("Worker completed more transitions than it attempted")
        if status != "accepted":
            if response.get("target") is not None:
                raise RepairProviderContractError("Rejected/error response carries a selectable target")
            if response.get("reason") == "frozen_model_state_changed":
                raise RepairProviderContractError("Frozen DeFoG checkpoint state changed")
            if status == "rejected" and process_returncode != 0:
                raise RepairProviderContractError("Rejected receipt disagrees with worker process status")
            if status == "error" and (type(process_returncode) is not int or process_returncode == 0):
                raise RepairProviderContractError("Error receipt disagrees with worker process status")
            return None
        if process_returncode != 0:
            raise RepairProviderContractError("Accepted worker target has nonzero process return code")
        before, after = provenance.get("frozen_state_sha256_before"), provenance.get("frozen_state_sha256_after")
        if (not isinstance(before, str) or len(before) != 64 or before != after
                or any(char not in "0123456789abcdef" for char in before)
                or provenance.get("frozen_state_unchanged") is not True):
            raise RepairProviderContractError("Accepted repair lacks unchanged frozen-state proof")
        gates = response.get("gates", {})
        charge_gates = ("declared_input_charge_loss", "native_output_charge_rule", "output_identity_preserved") if projected else ("neutral_charge_subset",)
        if any(gates.get(key) is not True for key in
               ("fixed_atom_count", "slot_alignment", "native_label_roundtrip", "chemical_validity", *charge_gates)):
            raise RepairProviderContractError("Accepted repair did not pass every declared target gate")
        if (type(cost.get("planned_transitions")) is not int or cost["planned_transitions"] < 1
                or cost["native_transitions"] != cost["planned_transitions"]
                or cost["attempted_native_transitions"] != cost["planned_transitions"]
                or cost["native_forward_calls"] != cost["planned_transitions"]):
            raise RepairProviderContractError("Accepted repair has an incomplete native suffix budget")
        original, effective = cost.get("original_checkpoint_sampling"), cost.get("effective_sampling")
        expected_fields = {"sample_steps", "time_distortion", "eta", "omega", "rdb", "rdb_crit"}
        if (not isinstance(original, dict) or set(original) != expected_fields
                or not isinstance(effective, dict) or set(effective) != expected_fields
                or effective["sample_steps"] != request["steps"]
                or cost.get("initialization") != request["initialization"]
                or cost.get("requested_model_time") != request["t0"]):
            raise RepairProviderContractError("Accepted repair executed different native initialization/settings")
        for key in expected_fields - {"sample_steps"}:
            if effective[key] != request["sampling"].get(key, original[key]):
                raise RepairProviderContractError("Accepted repair changed a native sampling setting silently")
        first = cost.get("grid_start_index")
        actual_time = cost.get("effective_start_time")
        if (type(first) is not int or not 0 <= first < request["steps"]
                or cost["planned_transitions"] != request["steps"] - first
                or isinstance(actual_time, bool) or not isinstance(actual_time, (float, int))
                or not math.isfinite(actual_time) or not request["t0"] - 1e-7 <= actual_time < 1):
            raise RepairProviderContractError("Accepted repair changed the declared late-time grid suffix")
        target = _target_from_record(response.get("target"), spec)
        if not torch.equal(target.atoms[0] != 0, active):
            raise RepairProviderContractError("Accepted repair changed fixed atom count or occupied slot IDs")
        if not projected and bool((target.charges != spec.formal_charges.index(0)).any()):
            raise RepairProviderContractError("Neutral-only DeFoG emitted a charged target")
        if projected:
            native = response.get("native_output_labels", {})
            try:
                inferred, inference = from_guacamol_native_charges(
                    torch.as_tensor(native["atoms"]), torch.as_tensor(native["bonds"]),
                    spec, tuple(request["occupied_slots"]))
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                raise RepairProviderContractError("Native output charge inference could not be independently reproduced") from error
            observed = response.get("output_charge_inference")
            # Parent and worker may use different pinned RDKit environments;
            # verify the actual inferred labels, and retain both runtime versions.
            comparable = lambda row: {key: value for key, value in row.items() if key != "rdkit_version"}
            if (not isinstance(observed, dict) or not isinstance(observed.get("rdkit_version"), str)
                    or comparable(observed) != comparable(inference)
                    or inference["sanitization_identity"]["passed"] is not True
                    or any(not torch.equal(getattr(target, key), getattr(inferred, key))
                           for key in ("atoms", "charges", "bonds"))
                    or response.get("target_sha256") != hard_digest(target)):
                raise RepairProviderContractError("Target or inferred charges differ from independent native-rule conversion")
        pair = active[:, None] & active[None, :]
        if (not torch.equal(target.bonds, target.bonds.transpose(1, 2))
                or bool(torch.diagonal(target.bonds[0]).any())
                or bool((target.bonds[0][~pair] != 0).any())):
            raise RepairProviderContractError("Accepted repair contains invalid padding/diagonal/bond symmetry")
        inspection = inspect_graph(target.atoms[0], target.charges[0], target.bonds[0], spec)
        if inspection.smiles is None or inspection.smiles != response.get("canonical_smiles"):
            raise RepairProviderContractError("Independent chemical/identity inspection disagrees with worker")
        # Require exact equality to the raw native labels in the receipt. A
        # separately valid target must not be substituted after the oracle call.
        native_candidate = to_guacamol_hard(candidate, spec, charge_policy=policy)
        native_input = response.get("native_input_labels", {})
        if (native_input.get("atoms") != native_candidate.x.argmax(-1).tolist()
                or native_input.get("bonds") != native_candidate.e.argmax(-1).tolist()
                or native_input.get("occupied_slots") != request["occupied_slots"]):
            raise RepairProviderContractError("Worker initialized from different native candidate labels")
        mapped = to_guacamol_hard(target, spec, charge_policy=policy)
        native = response.get("native_output_labels", {})
        if (native.get("atoms") != mapped.x.argmax(-1).tolist()
                or native.get("bonds") != mapped.e.argmax(-1).tolist()):
            raise RepairProviderContractError("Target differs from retained native output labels")
        return target

    def _worker(self, request, attempt_directory):
        request_path, output_path = attempt_directory / "request.json", attempt_directory / "response.json"
        write_json(request_path, request)
        command = [self.config.provider["python"], "-m", "latentmol.defog.candidate_repair",
                   "--request", str(request_path), "--output", str(output_path)]
        write_json(attempt_directory / "invocation.json", {"command": command,
            "timeout_seconds": self.config.provider["timeout_seconds"], "request_sha256": request["request_sha256"]})
        environment = dict(os.environ)
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        started = time.monotonic()
        with ((attempt_directory / "stdout.log").open("x", encoding="utf-8") as out,
              (attempt_directory / "stderr.log").open("x", encoding="utf-8") as err):
            try:
                completed = self.runner(command, stdout=out, stderr=err, timeout=self.config.provider["timeout_seconds"],
                                        check=False, env=environment)
                execution = {"process_returncode": completed.returncode, "timeout": False}
            except subprocess.TimeoutExpired:
                execution = {"process_returncode": None, "timeout": True,
                             "cost_censored": True, "cost_observation": "partial_lower_bound_if_progress_available"}
            except OSError as error:
                raise RepairProviderContractError("Repair Python process could not be launched") from error
        execution["wall_seconds"] = time.monotonic() - started
        if execution["timeout"]:
            progress_path = output_path.with_name(output_path.name + ".progress.json")
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as error:
                    raise RepairProviderContractError("Partial worker progress is unreadable") from error
                if (not isinstance(progress, dict) or progress.get("version") != request["version"]
                        or progress.get("request_sha256") != request["request_sha256"]):
                    raise RepairProviderContractError("Partial worker progress belongs to another request")
                partial_cost = progress.get("cost")
                if (not isinstance(partial_cost, dict)
                        or partial_cost.get("cost_observation") != "partial_lower_bound_until_terminal_receipt"
                        or any(type(partial_cost.get(key)) is not int or partial_cost[key] < 0
                               for key in ("native_forward_calls", "native_transitions", "attempted_native_transitions"))):
                    raise RepairProviderContractError("Partial worker progress counters are invalid")
                execution["partial_progress"] = progress
                execution["progress_sha256"] = file_sha256(progress_path)
            return None, execution, output_path
        if not output_path.exists():
            raise RepairProviderContractError("Native worker exited without its authenticated result receipt")
        try:
            response = json.loads(output_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise RepairProviderContractError("Native worker result receipt is unreadable") from error
        execution["response_sha256"] = file_sha256(output_path)
        return response, execution, output_path

    def refresh(self, decoder, point, active, temperature, step, directory, counters=None):
        """Draw all prescribed candidates and return a newly detached reference.

        Seed derivation distinguishes the candidate draw from native denoising.
        CPU sampling uses detached copies of the *actual shared GPU probabilities*;
        no CPU re-decoding, temperature changes or eligibility resampling occurs.
        """
        receipt = {"step": step, "status": "disabled", "attempts": [],
                   "planned_attempts": self.config.attempts_per_refresh,
                   "selection": self.config.selection, "fallback": self.config.fallback,
                   "guidance_weight": self.config.weight, "old_target_retained": False}
        if self.config.weight == 0:
            return None, receipt
        require_frozen_eval(decoder)
        if type(step) is not int or step < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Repair refresh needs a nonnegative step and finite positive temperature")
        if (active.dtype != torch.bool or active.shape != (decoder.spec.max_nodes,) or not bool(active.any())):
            raise ValueError("Repair guidance requires one nonempty fixed boolean occupancy mask")
        folder = Path(directory) / f"repair-refresh-{step:05d}"
        folder.mkdir(parents=True, exist_ok=False)
        receipt_path = folder / "refresh.json"
        receipt.update(status="running", directory=str(folder), receipt_path=str(receipt_path),
                       source_sha256=self.source_sha256, config=self.config.record())
        if "charge_policy" in self.config.provider:
            from rdkit import rdBase
            receipt["parent_charge_inference_rdkit_version"] = rdBase.rdkitVersion
        write_json(receipt_path, receipt)
        try:
            if (file_sha256(self.module_path) != self.expected_module_sha256
                    or file_sha256(self.codec_path) != self.expected_codec_sha256):
                raise RepairProviderContractError("Repair adapter bytes changed after provider construction")
            if not Path(self.config.provider["python"]).is_file():
                raise RepairProviderContractError("Configured DeFoG Python interpreter is missing")
            with torch.no_grad():
                function = lambda: decoder.decode(point.detach())
                decoded = (counters.call("decoder", f"repair_refresh_{step}", "forward", function, point.device)
                           if counters is not None else function())
                shared = graph_from_decoder(decoded, decoder.spec, temperature=temperature,
                                            fixed_active=active.to(point.device))
                graph = RelaxedGraph(*(getattr(shared, name).detach().cpu().clone()
                                       for name in ("atoms", "charges", "bonds")))
            mask = active.detach().cpu().clone()
            slots = mask.nonzero(as_tuple=False).flatten().tolist()
            receipt.update(temperature=temperature, occupied_slots=slots,
                latent_sha256=tensor_digest({"z": point}), shared_graph_sha256=tensor_digest(
                    {name: getattr(graph, name) for name in ("atoms", "charges", "bonds")}),
                sampling_law="factorized fixed-occupancy sharedgraph atom/charge/unordered-bond categorical draw",
                sampling_device="cpu detached copies of current shared graph", seed_policy="SHA256(base,step,attempt,phase) first8bytes & (2**63-1)")
            write_json(receipt_path, receipt)
            accepted = []
            for index in range(self.config.attempts_per_refresh):
                attempt_folder = folder / f"attempt-{index:03d}"
                attempt_folder.mkdir(exist_ok=False)
                draw_seed, native_seed = (_seed(self.config.seed, step, index, phase) for phase in ("draw", "native"))
                candidate = sample_fixed_relaxed_graph(graph, decoder.spec,
                    generator=torch.Generator(device="cpu").manual_seed(draw_seed), fixed_active=mask)
                request = {"version": 1, "spec": decoder.spec.to_dict(), "candidate": _graph_record(candidate),
                           "occupied_slots": slots, "seed": native_seed, "adapter_source_sha256": self.source_sha256,
                           **{key: self.config.provider[key] for key in
                              ("t0", "steps", "source", "checkpoint", "initialization", "device", "sampling")}}
                if "charge_policy" in self.config.provider:
                    request.update(version=2, charge_policy=self.config.provider["charge_policy"])
                request["request_sha256"] = request_digest(request)
                attempt = {"attempt": index, "status": "started", "draw_seed": draw_seed, "native_seed": native_seed,
                           "candidate_tensor_sha256": hard_digest(candidate),
                           "candidate_json_sha256": hashlib.sha256(json.dumps(
                               _graph_record(candidate), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                           "candidate": _graph_record(candidate),
                           "request_sha256": request["request_sha256"], "directory": str(attempt_folder),
                           "input_inspection": asdict(inspect_graph(candidate.atoms[0], candidate.charges[0], candidate.bonds[0], decoder.spec))}
                receipt["attempts"].append(attempt)
                write_json(receipt_path, receipt)
                if counters is not None:
                    counters.bump("defog", f"repair_refresh_{step}", "worker_attempted")
                response, execution, output = self._worker(request, attempt_folder)
                attempt.update(execution)
                if response is None:  # Declared timeout: no target, no hidden retry.
                    attempt.update(status="timeout", reason="bounded_native_worker_timeout")
                    if counters is not None:
                        counters.bump("defog", f"repair_refresh_{step}", "worker_timeout", execution["wall_seconds"])
                else:
                    # Persist the raw receipt before validating it: a provenance
                    # rejection must retain exactly which assertion failed.
                    attempt.update(response_path=str(output), response=response)
                    target = self._verify_response(response, request, decoder.spec, candidate, mask, execution["process_returncode"])
                    attempt.update(status=response["status"], reason=response.get("reason"),
                                   target_eligible=target is not None)
                    if counters is not None:
                        counters.bump("defog", f"repair_refresh_{step}", "worker_" + response["status"], execution["wall_seconds"])
                    if target is not None:
                        accepted.append((index, target, output))
                write_json(attempt_folder / "attempt.json", attempt)
                write_json(receipt_path, receipt)
            selected = accepted[0] if accepted else None
            terminal_costs = [row["response"]["cost"] for row in receipt["attempts"] if "response" in row]
            partial_costs = [row["partial_progress"]["cost"] for row in receipt["attempts"]
                             if row.get("timeout") and "partial_progress" in row]
            observed = terminal_costs + partial_costs
            receipt["oracle_cost"] = {
                "worker_attempts": len(receipt["attempts"]),
                "native_forward_calls_observed": sum(row.get("native_forward_calls", 0) for row in observed),
                "native_transitions_observed": sum(row.get("native_transitions", 0) for row in observed),
                "attempted_native_transitions_observed": sum(row.get("attempted_native_transitions", 0) for row in observed),
                "cost_censored": any(row.get("timeout") for row in receipt["attempts"]),
                "cost_observation": "lower_bound_if_any_timeout_else_exact_worker_counts",
                "wall_seconds": sum(row["wall_seconds"] for row in receipt["attempts"])}
            receipt.update(status="accepted" if selected else "no_accepted_target", attempted=len(receipt["attempts"]),
                           accepted_attempts=[entry[0] for entry in accepted], selected_attempt=selected[0] if selected else None)
            if selected:
                receipt["selected_target_sha256"] = hard_digest(selected[1])
                target = FrozenRepairTarget.create(selected[1], mask, refresh_step=step, receipt_path=selected[2])
            else:
                target = None
            write_json(receipt_path, receipt)
            return target, receipt
        except Exception as error:
            receipt.update(status="contract_failed" if isinstance(error, RepairProviderContractError) else "failed",
                           error_type=type(error).__name__, detail=str(error), unattempted=self.config.attempts_per_refresh - len(receipt["attempts"]))
            if receipt["attempts"] and receipt["attempts"][-1]["status"] == "started":
                receipt["attempts"][-1].update(status="contract_failed", detail=str(error))
                write_json(Path(receipt["attempts"][-1]["directory"]) / "attempt.json", receipt["attempts"][-1])
            write_json(receipt_path, receipt)
            raise


def make_provider(config: RepairGuidanceConfig, source_sha256: str):
    """Names map to explicit native contracts; a new checkpoint needs a new audit."""
    if config.provider["model"] != "defog_guacamol":
        raise ValueError("No registered repair provider for this model")
    return DeFoGRepairProvider(config, source_sha256)
