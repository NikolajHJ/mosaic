"""Allocated, frozen-state attribution of the case-003 relaxed AD discrepancy.

Complete sibling cuts at logits, the graph and native chemistry retain every
route into Boltz. Native FP32 cotangents and JVPs are contracted in FP64 for
diagnosis. Arithmetic controls are fixed-cotangent experiments, never selected
production fixes, revised tolerances or optimization certificates.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

import torch

from latentmol.graph_policy import fixed_occupancy_graph
from latentmol.multimodel.study import tensor_digest
from latentmol.representation import DecoderOutput, RelaxedGraph
from latentmol.train import state_digest
from .adapter import adapt_graph
from .dbmol_objective import COMPONENT_OUTPUTS
from .embedding_comparison_experiment import prepare_starts
from .embedding_experiment import cpu_stage, load_native, load_vae, verified, write
from .embedding_validation import _deadline, _frozen_versions, _scalars, native_gate
from .provenance import context_digest


VERSION = "epoch36-embedding-gradient-diagnostic-v1"
FIELDS = ("atoms", "charges", "bonds")
NATIVE = ("ref_element", "ref_charge", "token_bonds", "bond_types")
COMPONENTS = (*COMPONENT_OUTPUTS, "dbmol_total_loss")
CLAIM = "Diagnostic attribution only; no optimization, fix selection, threshold change or historical gate promotion."


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def _dot(left, right):
    if set(left) != set(right):
        raise ValueError("Diagnostic contractions require complete matching sibling cuts")
    return math.fsum(float((left[key].double() * right[key].double()).sum()) for key in left)


def _comparison(reverse, forward):
    if not math.isfinite(reverse) or not math.isfinite(forward):
        raise ValueError("Nonfinite diagnostic duality measurement")
    allowed = 1e-6 + 1e-3 * max(abs(reverse), abs(forward))
    return dict(reverse_projection=reverse, forward_projection=forward,
        absolute_error=abs(reverse - forward), inherited_allowed_error=allowed,
        within_inherited_threshold=abs(reverse - forward) <= allowed,
        threshold_used_for_promotion=False)


def _vector_comparison(actual, expected):
    if set(actual) != set(expected) or any(actual[key].shape != expected[key].shape
            or actual[key].dtype != expected[key].dtype for key in expected):
        raise ValueError("Diagnostic VJP comparison requires every captured input sibling and dtype")
    rows = {}
    for key in expected:
        difference = actual[key].double() - expected[key].double()
        if not bool(difference.isfinite().all()):
            raise ValueError("Nonfinite diagnostic input VJP difference")
        rows[key] = dict(shape=list(actual[key].shape), dtype=str(actual[key].dtype),
            bitwise_exact=torch.equal(actual[key], expected[key]),
            maximum_absolute_difference=float(difference.abs().max()),
            difference_l2=float(difference.norm()))
    return dict(complete=True, rows=rows, bitwise_exact=all(row["bitwise_exact"] for row in rows.values()),
        maximum_absolute_difference=max(row["maximum_absolute_difference"] for row in rows.values()),
        difference_l2=math.sqrt(math.fsum(row["difference_l2"] ** 2 for row in rows.values())),
        used_for_gate_promotion=False)


def _decoded(value):
    return {key: getattr(value, key) for key in FIELDS}


def independent_logits(value):
    """Expose sibling output nodes despite decoder-internal head feedback."""
    return {key: tensor.clone() for key, tensor in _decoded(value).items()}


def graph_policy(logits, active, spec, *, wide=False):
    """The wide control changes only final categorical policy arithmetic."""
    original_dtype = logits["atoms"].dtype
    view = {key: value.double() if wide else value for key, value in logits.items()}
    result = fixed_occupancy_graph(DecoderOutput(**view), active, spec, 1.)
    return {key: getattr(result, key).to(original_dtype) for key in FIELDS}


def native_channels(graph, prepared):
    features, bonds = adapt_graph(RelaxedGraph(**graph), prepared)
    return {key: features[key] for key in NATIVE[:-1]} | {"bond_types": bonds}


def _replay(function, inputs, directions, cotangents, *, deadline=None):
    """Independent VJP/JVP duality for one complete local cut and fixed seed."""
    _deadline(deadline)
    names = tuple(inputs)
    probes = {key: value.detach().clone().requires_grad_(True) for key, value in inputs.items()}
    with torch.enable_grad():
        outputs = function(probes)
        if set(outputs) != set(cotangents):
            raise ValueError("Cotangent replay dropped a sibling output")
        gradients = torch.autograd.grad(tuple(outputs.values()), tuple(probes.values()),
            grad_outputs=tuple(cotangents[key].to(outputs[key]) for key in outputs))
    primal = {key: value.detach().clone() for key, value in outputs.items()}
    del probes, outputs
    _deadline(deadline)
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        duals = {key: torch.autograd.forward_ad.make_dual(inputs[key], directions[key]) for key in names}
        observed = function(duals)
        tangent, exact = {}, True
        for key, value in observed.items():
            p, t = torch.autograd.forward_ad.unpack_dual(value)
            if t is None or not bool(t.isfinite().all()):
                raise ValueError("Missing/nonfinite local diagnostic tangent: " + key)
            exact = exact and torch.equal(p, primal[key])
            tangent[key] = t.detach().clone()
    vjp = dict(zip(names, (value.detach() for value in gradients), strict=True))
    if not exact:
        raise ValueError("Local diagnostic AD modes changed their own primal")
    result = _comparison(_dot(vjp, directions), _dot(cotangents, tangent))
    result.update(primal_exact_between_ad_modes=exact,
        contraction_dtype="torch.float64", cotangent_seed_dtype={k: str(v.dtype) for k, v in cotangents.items()})
    return result, dict(input_vjp=vjp, output_jvp=tangent, output_primal=primal)


def _equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_equal_nested(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal_nested(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def authenticated_case(config, cpu_root):
    """Bind all prescribed starts to immutable attempt003, then focus case003."""
    parent_path = verified(config["diagnostic_parent_starts"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    fresh = torch.load(Path(cpu_root) / "starts.pt", map_location="cpu", weights_only=True)
    if not _equal_nested(parent, fresh) or len(fresh["cases"]) != 3:
        raise ValueError("Fresh diagnostic starts differ from all three authenticated parent starts")
    selected = [item for item in fresh["cases"] if item["record"]["case_id"] == "case-003"]
    if len(selected) != 1:
        raise ValueError("Diagnostic requires the original case-003 without replacement")
    return selected[0]


def cpu_probe(config, output, deadline):
    case = authenticated_case(config, output)
    vae, checkpoint = load_vae(config, "cpu")
    del checkpoint
    before = state_digest(vae.state_dict())
    z, active = case["z"], case["active"]
    generator = torch.Generator().manual_seed(139036)
    direction = torch.randn(z.shape, generator=generator).to(z)
    direction = direction / direction.double().norm().to(direction)
    function = lambda values: graph_policy(_decoded(vae.decode(values["z"])), active, vae.spec)
    with torch.no_grad():
        graph = function({"z": z})
    cotangent = {key: torch.randn(value.shape, generator=generator).to(value) for key, value in graph.items()}
    report, tensors = _replay(function, {"z": z}, {"z": direction}, cotangent, deadline=deadline)
    torch.save(_cpu(dict(z=z, direction=direction, cotangent=cotangent, replay=tensors)), output / "cpu-decoder-probe.pt")
    report.update(status="completed_diagnostic_only", claim=CLAIM,
        cotangent_scope="deterministic engineering coefficients; not native Boltz cotangents",
        case_id="case-003", frozen_vae_unchanged=state_digest(vae.state_dict()) == before)
    if not report["frozen_vae_unchanged"]:
        raise ValueError("CPU diagnostic changed the original VAE")
    write(output / "cpu-decoder-probe.json", report)
    return report


def gpu_diagnostic(config, output, deadline):
    from latentmol.experiments import file_sha256
    from .active_set import bind_slots
    from .contracts import PreparedComplex
    cpu_root = output.parent / "cpu"
    cpu = json.loads((cpu_root / "result.json").read_text())
    if (cpu["status"] != "passed" or cpu["source_sha256"] != os.environ["LATENTMOL_SOURCE_SHA256"]
            or cpu["plan_sha256"] != config["_plan_sha256"] or cpu["junit"]["tests"] < 1
            or any(cpu["junit"][key] for key in ("failures", "errors", "skipped"))
            or file_sha256(cpu_root / "cpu-tests.xml") != cpu["junit_sha256"]
            or file_sha256(cpu_root / "starts.pt") != cpu["starts"]["starts_sha256"]):
        raise ValueError("Diagnostic GPU requires exact-release CPU tests and shared-start authentication")
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated diagnostic has no CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    case = authenticated_case(config, cpu_root)
    vae, checkpoint = load_vae(config, "cuda")
    del checkpoint
    scorer, base, bank = load_native(config, "cuda")
    template = PreparedComplex.load(verified(case["record"]["context_asset"]), "cpu")
    prepared = bind_slots(template, case["active"]).to("cuda")
    if context_digest(prepared) != case["record"]["bound_context_sha256"]:
        raise ValueError("Diagnostic changed the case-003 fixed native context")
    z = case["z"].to("cuda")
    versions = _frozen_versions(vae=vae, scorer=scorer, bank=bank)
    identities = dict(vae=state_digest(vae.state_dict()), scorer=state_digest(scorer.state_dict()),
        bank=state_digest(bank.state_dict()), context=context_digest(prepared), point=tensor_digest({"z": z}))
    report = dict(version=VERSION, status="running_diagnostic_only", claim=CLAIM, case=case["record"],
        identities_before=identities, calls=dict(native_forward_attempted=0, native_forward_completed=0,
            reverse_attempted=0, reverse_completed=0, forward_ad_attempted=0),
        stage="native_oracle", parent_starts=config["diagnostic_parent_starts"],
        uncheckpointed_scorer_control=dict(status="not_run", reason="Unbounded 64-block activation retention is excluded from the 80GB diagnostic; complete cut replays are used."))
    tensors = {}

    def save():
        write(output / "diagnostic.json", report)

    def chain(point):
        # Raw atom logits are ancestors of charge/bond logits through decoder
        # feedback. Identity clones expose independent output-boundary nodes;
        # downstream policy consumes only these siblings, so their cotangents
        # cannot double-count decoder-internal ancestor routes.
        logits = independent_logits(vae.decode(point))
        graph = graph_policy(logits, prepared.active, prepared.spec)
        native = native_channels(graph, prepared)
        features = dict(prepared.features)
        features.update({key: native[key] for key in NATIVE[:-1]})
        _deadline(deadline)
        report["calls"]["native_forward_attempted"] += 1
        save()
        scores = scorer(features, native["bond_types"], prepared.coordinates)
        report["calls"]["native_forward_completed"] += 1
        return scores, dict(z={"z": point}, logits=logits, graph=graph, native=native)

    save()
    try:
        oracle = native_gate(scorer, prepared, bank, vae, output / "native-oracle.json",
                             config["native_tolerances"], deadline=deadline)
        report["native_oracle"] = oracle["status"]
        report["stage"] = "baseline_reverse_all_sibling_cuts"
        with torch.enable_grad():
            probe = z.detach().clone().requires_grad_(True)
            scores, live = chain(probe)
            baseline = _scalars(scores)
            layout = [(stage, key) for stage, values in live.items() for key in values]
            flat = tuple(live[stage][key] for stage, key in layout)
            cotangents = {}
            for index, component in enumerate(COMPONENTS):
                _deadline(deadline)
                report["calls"]["reverse_attempted"] += 1
                gradients = torch.autograd.grad(scores[component].reshape(()), flat,
                                                retain_graph=index < len(COMPONENTS) - 1)
                report["calls"]["reverse_completed"] += 1
                cotangents[component] = {stage: {} for stage in live}
                for (stage, key), gradient in zip(layout, gradients, strict=True):
                    if not bool(gradient.isfinite().all()):
                        raise ValueError("Nonfinite baseline cotangent")
                    cotangents[component][stage][key] = gradient.detach().clone()
            primals = {stage: {key: value.detach().clone() for key, value in values.items()}
                       for stage, values in live.items()}
            del scores, live, flat, gradients, probe
        gradient = cotangents["dbmol_total_loss"]["z"]["z"]
        norm = float(gradient.double().norm())
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Failed-start diagnostic has an uninformative total gradient")
        direction = gradient / norm
        report.update(baseline_scores=baseline, gradient_norm=norm,
            production_gradient_sha256=tensor_digest({"gradient": gradient}),
            direction_sha256=tensor_digest({"direction": direction}))
        if "diagnostic_parent_gate" in config:
            parent = json.loads(verified(config["diagnostic_parent_gate"]).read_text())
            binding = dict(scores_exact=baseline == parent["baseline_scores"],
                gradient_exact=report["production_gradient_sha256"] == parent["production_gradient_sha256"],
                direction_exact=report["direction_sha256"] == parent["direction_bindings"]["gradient"])
            report["parent_failure_binding"] = binding
            save()
            if not all(binding.values()):
                raise ValueError("Current failed-start baseline differs from authenticated attempt003")
        report["stage"] = "full_forward_ad_all_sibling_cuts"
        tangents = {}
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            report["calls"]["forward_ad_attempted"] += 1
            dual = torch.autograd.forward_ad.make_dual(z, direction)
            scores, live = chain(dual)
            report["full_forward_primal_exact"] = _scalars(scores) == baseline
            native_jvp = {}
            for component in COMPONENTS:
                p, tangent = torch.autograd.forward_ad.unpack_dual(scores[component].reshape(()))
                if tangent is None or not bool(tangent.isfinite()):
                    raise ValueError("Nonfinite full native forward tangent")
                native_jvp[component] = float(tangent)
            stage_primal_exact = {}
            for stage, values in live.items():
                tangents[stage] = {}
                stage_primal_exact[stage] = {}
                for key, value in values.items():
                    p, tangent = torch.autograd.forward_ad.unpack_dual(value)
                    if tangent is None or not bool(tangent.isfinite().all()):
                        raise ValueError("Missing/nonfinite complete-cut tangent: " + stage + "/" + key)
                    tangents[stage][key] = tangent.detach().clone()
                    stage_primal_exact[stage][key] = torch.equal(p, primals[stage][key])
            del scores, live, dual
        report["cut_primal_exact"] = stage_primal_exact
        if not report["full_forward_primal_exact"] or not all(all(row.values()) for row in stage_primal_exact.values()):
            raise ValueError("AD modes changed a baseline primal at a diagnostic cut")
        tensors.update(z=z, direction=direction, primals=primals, tangents=tangents,
                       cotangents=cotangents)
        report["cut_contractions"] = {}
        report["local_replays"] = {}
        tensors["local_replays"] = {}
        functions = dict(decoder=lambda values: _decoded(vae.decode(values["z"])),
            graph_policy=lambda values: graph_policy(values, prepared.active, prepared.spec),
            adapter=lambda values: native_channels(values, prepared))
        cuts = dict(decoder=("z", "logits"), graph_policy=("logits", "graph"), adapter=("graph", "native"))
        for component in COMPONENTS:
            report["cut_contractions"][component] = {stage: _comparison(
                _dot(cotangents[component][stage], tangents[stage]), native_jvp[component]) for stage in primals}
            report["local_replays"][component], tensors["local_replays"][component] = {}, {}
            for name, (left, right) in cuts.items():
                row, evidence = _replay(functions[name], primals[left], tangents[left],
                                        cotangents[component][right], deadline=deadline)
                row["captured_output_primal_exact"] = all(torch.equal(
                    evidence["output_primal"][key], primals[right][key]) for key in primals[right])
                if not row["captured_output_primal_exact"]:
                    raise ValueError("Baseline local replay differs from the captured parent cut primal: " + name)
                row["captured_input_cotangent_comparison"] = _vector_comparison(
                    evidence["input_vjp"], cotangents[component][left])
                report["local_replays"][component][name] = row
                tensors["local_replays"][component][name] = evidence
        torch.save(_cpu(tensors), output / "cut-evidence.pt")
        report["cut_evidence_sha256"] = file_sha256(output / "cut-evidence.pt")
        report["stage"] = "declared_fixed_cotangent_numerical_controls"
        save()
        controls = {
            "final_policy_float64": lambda values: graph_policy(_decoded(vae.decode(values["z"])), prepared.active, prepared.spec, wide=True)}
        # Exact stored FP32 numbers are promoted in a separate frozen copy;
        # original state/weights/decoder implementation remain untouched.
        wide_vae = deepcopy(vae).double().eval().requires_grad_(False)
        if any(not torch.equal(value, wide_vae.state_dict()[key].to(value)) for key, value in vae.state_dict().items()):
            raise ValueError("FP64 diagnostic copy changed stored pretrained numbers")
        controls["decoder_and_final_policy_float64"] = lambda values: {
            key: value.to(z.dtype) for key, value in graph_policy(
                _decoded(wide_vae.decode(values["z"].double())), prepared.active, prepared.spec).items()}
        report["numerical_controls"], control_tensors = {}, {}
        for name, function in controls.items():
            report["numerical_controls"][name], control_tensors[name] = {}, {}
            for component in COMPONENTS:
                row, evidence = _replay(function, {"z": z}, {"z": direction},
                                        cotangents[component]["graph"], deadline=deadline)
                row["baseline_graph_max_absolute_change"] = {key: float((
                    evidence["output_primal"][key].double() - primals["graph"][key].double()).abs().max()) for key in FIELDS}
                row["scope"] = "frozen original native graph cotangent; no native score evaluated at the changed primal"
                report["numerical_controls"][name][component] = row
                control_tensors[name][component] = evidence
        # Reuse the existing declared stable-occupancy diagnostic instrumentation.
        from .decoder_causal_diagnostic import DecoderObserver, decoder_control
        observer = DecoderObserver()
        with decoder_control(vae, observer):
            with torch.no_grad():
                observer.begin("original_fp32", capture=True)
                vae.decode(z)
                observer.finish()
            def stable_function(values):
                observer.begin("stable_occupancy_fp32")
                try:
                    result = graph_policy(_decoded(vae.decode(values["z"])), prepared.active, prepared.spec)
                    observer.finish()
                    return result
                except BaseException:
                    observer.abort()
                    raise
            name = "stable_internal_occupancy_float32"
            report["numerical_controls"][name], control_tensors[name] = {}, {}
            for component in COMPONENTS:
                row, evidence = _replay(stable_function, {"z": z}, {"z": direction},
                                        cotangents[component]["graph"], deadline=deadline)
                row["baseline_graph_max_absolute_change"] = {key: float((
                    evidence["output_primal"][key].double() - primals["graph"][key].double()).abs().max()) for key in FIELDS}
                row["scope"] = "diagnostic stable real-category occupancy sum; frozen native cotangent only"
                report["numerical_controls"][name][component] = row
                control_tensors[name][component] = evidence
        torch.save(_cpu(control_tensors), output / "control-evidence.pt")
        report["control_evidence_sha256"] = file_sha256(output / "control-evidence.pt")
        del wide_vae
        report.update(stage="complete", status="completed_diagnostic_no_fix_selected", optimization="not_run")
    except BaseException as error:
        report.update(status="failed_diagnostic", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        after = dict(vae=state_digest(vae.state_dict()), scorer=state_digest(scorer.state_dict()),
            bank=state_digest(bank.state_dict()), context=context_digest(prepared), point=tensor_digest({"z": z}))
        report.update(identities_after=after, frozen_identity_unchanged=after == identities,
            frozen_versions_unchanged=_frozen_versions(vae=vae, scorer=scorer, bank=bank) == versions)
        if not report["frozen_identity_unchanged"] or not report["frozen_versions_unchanged"]:
            report.update(status="failed_frozen_identity")
        save()
    if report["status"] == "failed_frozen_identity":
        raise ValueError("Diagnostic changed original frozen models/context")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "gpu"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise RuntimeError("Numerical diagnostic work requires a sealed allocated LSF job")
    from latentmol.experiments import file_sha256
    torch.set_num_threads(min(4, int(os.environ.get("LSB_DJOB_NUMPROC", "4"))))
    config = json.loads(args.plan.read_text())
    if config["version"] != VERSION:
        raise ValueError("Unknown embedding gradient diagnostic plan")
    config["_plan_sha256"] = file_sha256(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config["internal_seconds"][args.stage]
    provenance = dict(version=VERSION, stage=args.stage, job_id=os.environ["LSB_JOBID"],
        hostname=socket.gethostname(), source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"],
        plan_sha256=config["_plan_sha256"], claim=CLAIM, existing_vae_continuation="untouched")
    write(args.output / "result.json", {**provenance, "status": "running_not_passed"})
    try:
        if args.stage == "cpu":
            result = cpu_stage(config, args.output, deadline)
            result["starts"] = prepare_starts(config, args.output)
            result["decoder_probe"] = cpu_probe(config, args.output, deadline)
        else:
            result = gpu_diagnostic(config, args.output, deadline)
    except BaseException as error:
        write(args.output / "result.json", {**provenance, "status": "failed_not_passed",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started})
        raise
    write(args.output / "result.json", {**provenance, **result, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
