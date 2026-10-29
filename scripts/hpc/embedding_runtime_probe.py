"""Allocated, isolated performance probe; never mutates the production campaign.

Historical timings are descriptive. Matched direct-input and VAE measurements
use the same frozen source142 models, context, dtype and chemical primals.
All diagnostic updates use one fixed LR0.0025 candidate, without line search.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import xml.etree.ElementTree as ET

VERSION = "embedding-runtime-fixed-step-probe-v2"
CONFIG = "configs/embedding_runtime_probe.json"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def verify(value):
    path = Path(value["path"])
    if sha(path) != value["sha256"]:
        raise ValueError("Probe input hash differs: " + str(path))
    return path


def write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def validate_output(release, output, stage):
    import re
    release, output = Path(release).resolve(), Path(output).resolve()
    if not re.fullmatch(r"source-144-embedding-runtime-probe-attempt[0-9]{3}",release.name):
        raise ValueError("Runtime probe must use its own immutable source144 release")
    expected = Path("/work3/s234818/latentmol/runs") / release.name.replace("source-144","source144",1) / stage
    if output != expected:
        raise ValueError("Probe output must remain in the independent source144 campaign")


def validate_config(config):
    if (config.get("version") != VERSION or config.get("fixed_learning_rate") != .0025
            or config.get("step_candidates") != 1 or config.get("line_search") is not False
            or config.get("warmups") != 2 or config.get("repeats") != 5):
        raise ValueError("Probe must retain one fixed LR0.0025 candidate and bounded repetitions")
    if config.get("gpu_profile") != "l40s48" or config.get("reference_policy") != "same-device-exact-original-A100-certificate-reported":
        raise ValueError("Probe hardware/reference policy must be explicitly recorded")
    parent = config["parent"]
    if (Path(parent["release"]).name != "source-142-epoch36-guarded-embedding-long-attempt001"
            or Path(parent["campaign"]).name != "source142-epoch36-guarded-embedding-long-attempt001"):
        raise ValueError("Probe requires the exact running scientific parent")
    for item in parent["artifacts"].values():
        verify(item)
    return config


def require_matched(torch, raw, gold, direct, relaxed, graph):
    if (raw["scores"] != gold["scores"] or direct["scores"] != gold["scores"]
            or raw["gradient"].dtype != gold["gradient"].dtype
            or not torch.equal(raw["gradient"], gold["gradient"])
            or graph["scores"] != relaxed["scores"]):
        raise ValueError("Matched raw/direct profiling primals or original learned derivative changed")


def authenticate(config):
    parent = config["parent"]
    spec = importlib.util.spec_from_file_location("_probe_operator", Path(__file__).with_name("embedding_operator.py"))
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    release = Path(parent["release"])
    operator.authenticate(release, parent["source_sha256"], parent["plan_sha256"],
                          "configs/epoch36_embedding_long.json")
    if sha(release / "embedding-release-manifest.json") != parent["manifest_sha256"]:
        raise ValueError("Original scientific manifest changed")
    validate_config(config)
    return release


def quantiles(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Timings must be finite nonnegative observations")
    ordered = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=ordered[0], p90=ordered[math.ceil(.9 * len(ordered)) - 1], samples=values)


class Spans:
    """Nested host wall spans; CUDA events are resolved after one outer sync.

    Component wall time includes synchronizations naturally performed by that
    component. Inclusive CUDA spans overlap; they must not be added together.
    Uninstrumented end-to-end medians are the primary performance measurements.
    """
    def __init__(self, torch=None):
        self.torch, self.rows, self.stack = torch, [], []

    @contextmanager
    def span(self, name):
        started = time.perf_counter()
        events = None
        if self.torch is not None:
            events = (self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True))
            events[0].record()
        entry = dict(name=name, child_wall=0., events=events)
        self.stack.append(entry)
        try:
            yield
        finally:
            if events:
                events[1].record()
            wall = time.perf_counter() - started
            self.stack.pop()
            if self.stack:
                self.stack[-1]["child_wall"] += wall
            self.rows.append(dict(name=name, wall=wall, exclusive_wall=max(0., wall-entry["child_wall"]), events=events))

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            with self.span(name):
                return function(*args, **kwargs)
        return measured

    def report(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()
        result = {}
        for row in self.rows:
            value = result.setdefault(row["name"], dict(calls=0, wall_seconds=0., exclusive_wall_seconds=0., cuda_seconds=0.))
            value["calls"] += 1
            value["wall_seconds"] += row["wall"]
            value["exclusive_wall_seconds"] += row["exclusive_wall"]
            if row["events"]:
                value["cuda_seconds"] += row["events"][0].elapsed_time(row["events"][1]) / 1000
        return result


def measure(torch, function, warmups, repeats, deadline):
    values, result = [], None
    for index in range(warmups + repeats):
        if time.monotonic() + 60 > deadline:
            raise TimeoutError("Runtime probe internal deadline")
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        if index >= warmups:
            values.append(time.perf_counter() - started)
    return quantiles(values), result


def cpu_stage(config, output, source_sha, plan_sha):
    authenticate(config)
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_embedding_runtime_probe.py",
               "--junitxml=" + str(output / "cpu-tests.xml")]
    with (output / "cpu-tests.log").open("xb") as stream:
        done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    suites = list(ET.parse(output / "cpu-tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if done.returncode or not counts["tests"] or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise RuntimeError("Allocated probe fixtures failed")
    return dict(status="passed", source_sha256=source_sha, plan_sha256=plan_sha,
                junit=counts, junit_asset=artifact(output / "cpu-tests.xml"), numerical_models="not_loaded")


def gpu_stage(config, output, deadline):
    parent_release = authenticate(config)
    sys.path.insert(0, str(parent_release / "src"))
    import torch
    from latentmol.boltz import embedding_comparison as comparison
    from latentmol.boltz import embedding_long_optimization as long
    from latentmol.boltz import dbmol_model as dbmol
    from latentmol.boltz.embedding_long_experiment import load_long_bridge, load_case
    from latentmol.boltz.embedding_experiment import load_vae, load_native
    from latentmol.boltz.embedding_training import conditioned_hard
    from latentmol.boltz.embedding_comparison_policy import comparison_graph
    from latentmol.boltz.embedding_injection import inject_native_embeddings
    from latentmol.boltz.adapter import adapt_graph
    from latentmol.representation import DecoderOutput
    from latentmol.train import state_digest
    from latentmol.multimodel.study import tensor_digest
    if not all(Path(module.__file__).resolve().is_relative_to(parent_release / "src")
               for module in (comparison,long,dbmol)):
        raise ValueError("Probe did not import the authenticated original scientific modules")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated probe has no CUDA device")
    if "L40" not in torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).total_memory < 40*1024**3:
        raise RuntimeError("This isolated probe requires the declared 48GB L40S resource profile")
    native_config = json.loads((parent_release / "configs/epoch36_embedding_long.json").read_text())
    p = config["parent"]
    fitted = json.loads(verify(p["artifacts"]["fit_result"]).read_text())
    vae, checkpoint = load_vae(native_config, "cuda")
    del checkpoint
    scorer, unused, bank = load_native(native_config, "cuda")
    del unused
    bridge, caps = load_long_bridge(native_config, vae, bank, fitted, p["source_sha256"], p["plan_sha256"], "cuda")
    starts = torch.load(verify(p["artifacts"]["starts"]), map_location="cpu", weights_only=True)
    item = starts["cases"][0]
    if item["record"]["case_id"] != "case-001":
        raise ValueError("Probe case order changed")
    prepared, z0 = load_case(item, "cuda")
    saved = torch.load(verify(p["artifacts"]["state277"]), map_location="cpu", weights_only=True)
    manifest = json.loads(verify(p["artifacts"]["trajectory_manifest"]).read_text())
    if (saved["update"] != 277 or saved["case_id"] != "case-001"
            or not torch.equal(saved["z0"], z0.cpu())
            or saved["signature_sha256"] != manifest["sha256"]
            or manifest["signature"]["binding"]["bridge_checkpoint_sha256"] != fitted["model_identity"]["checkpoint_sha256"]):
        raise ValueError("Saved representative latent does not bind the production model/start")
    binding = manifest["signature"]["binding"]
    if (binding != json.loads(verify(p["artifacts"]["gate_binding"]).read_text())
            or binding["source_sha256"] != p["source_sha256"]
            or binding["plan_sha256"] != p["plan_sha256"]
            or binding["calibration_sha256"] != fitted["model_identity"]["calibration_sha256"]):
        raise ValueError("Profiling snapshot differs from original calibrated derivative binding")
    objective = comparison.FrozenComparisonObjective(scorer, prepared, bridge, vae, bank=bank,
        deadline=deadline, comparison_graph_policy=native_config["comparison_graph_policy"])
    guard = long.GuardedEmbeddingObjective(objective, caps)
    frozen = objective.assert_unchanged(full=True)
    before = {key: state_digest(module.state_dict()) for key,module in objective.modules.items()}
    result = dict(status="running_not_passed", cases=[], fixed_learning_rate=.0025, step_candidates=1,
        line_search=False, production_mutation=False, optimization_claim=False,
        models=dict(vae=native_config["vae"], mapping=fitted["model_identity"]),
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), cpu_threads=torch.get_num_threads()),
        context=dict(sha256=objective.context_sha256, active_slots=int(prepared.active.sum()),
            features={key:dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
                      for key,value in prepared.features.items()},
            feature_total_bytes=sum(value.numel()*value.element_size() for value in prepared.features.values())),
        historical=config["historical_baselines"], frozen_before=frozen)
    write(output / "setup.json", {k:v for k,v in result.items() if k!="cases"})

    for label, point in (("original_start", z0), ("saved_update277", saved["z"].to(z0))):
        directory = output / label
        directory.mkdir()
        with torch.no_grad():
            decoded, hidden = vae.decode_with_features(point)
            hard = conditioned_hard(decoded, prepared.active[None], prepared.spec)
            features, bonds = adapt_graph(hard.to_relaxed(prepared.spec), prepared)
            chemical = bridge(hidden.atom_hidden, hidden.bond_input, prepared.active, hidden.charge_input)
            logits = {name:getattr(decoded,name).detach().clone() for name in ("atoms","charges","bonds")}
        del hidden

        def raw_learned(z, gradient=True):
            with torch.enable_grad() if gradient else torch.no_grad():
                value = z.detach().clone().requires_grad_(gradient)
                d,h = vae.decode_with_features(value)
                with torch.no_grad():
                    g = conditioned_hard(d, prepared.active[None], prepared.spec)
                    f,b = adapt_graph(g.to_relaxed(prepared.spec), prepared)
                mapped = bridge(h.atom_hidden,h.bond_input,prepared.active,h.charge_input)
                with inject_native_embeddings(scorer,prepared,mapped):
                    scores = scorer(f,b,prepared.coordinates)
                    derivative = torch.autograd.grad(scores["dbmol_total_loss"],value)[0] if gradient else None
                return dict(scores=comparison._scalars(scores),gradient=derivative)

        def direct_embeddings():
            leaves = {key:value.detach().clone().requires_grad_() for key,value in chemical.items()}
            with inject_native_embeddings(scorer,prepared,leaves):
                scores = scorer(features,bonds,prepared.coordinates)
                derivatives = torch.autograd.grad(scores["dbmol_total_loss"],tuple(leaves.values()))
            return dict(scores=comparison._scalars(scores),gradients=derivatives)

        def direct_graph(affinity_only=False, given=None, gradient=True):
            source = logits if given is None else given
            leaves = {key:value.detach().clone().requires_grad_(gradient) for key,value in source.items()}
            d = DecoderOutput(**leaves)
            graph = comparison_graph(d,prepared.active,prepared.spec,policy=native_config["comparison_graph_policy"])
            f,b = adapt_graph(graph,prepared)
            scores = scorer.affinity(f,b,prepared.coordinates) if affinity_only else scorer(f,b,prepared.coordinates)
            loss = scores["affinity_pred_value"] if affinity_only else scores["dbmol_total_loss"]
            derivatives = torch.autograd.grad(loss,tuple(leaves.values())) if gradient else None
            values = ({k:float(v.detach()) for k,v in scores.items()} if affinity_only
                      else comparison._scalars(scores))
            return dict(loss=float(loss.detach()),gradients=derivatives,scores=values)

        gold = objective.evaluate("learned",point,require_gradient=True)
        raw = raw_learned(point)
        direct = direct_embeddings()
        relaxed = objective.evaluate("relaxed",point,require_gradient=True)
        graph_reference = direct_graph()
        require_matched(torch,raw,gold,direct,relaxed,graph_reference)
        # The production certificate is from A100. Cross-device bitwise
        # equality is reported, never presented as a new production gate pass.
        original_certificate = None
        if label == "original_start":
            original_certificate = dict(hardware="production A100 reference versus diagnostic L40S",
                scores_bitwise_equal=gold["scores"]==binding["baseline_scores"],
                gradient_digest_equal=tensor_digest({"gradient":gold["gradient"]})==binding["gradient_sha256"],
                original_gradient_sha256=binding["gradient_sha256"],
                scalar_differences={key:gold["scores"][key]-value for key,value in binding["baseline_scores"].items()},
                scientific_production_validation="not_adopted; existing A100 gates unchanged")
        assessment = guard.assess(point)
        if not assessment["record"]["eligible"]:
            raise ValueError("Selected profiling point failed unchanged production eligibility")
        parity = dict(passed=True,learned_raw_scores_exact=True,learned_raw_gradient_exact=True,
            detached_embedding_primal_exact=True,direct_graph_relaxed_primal_exact=True,
            learned_gradient_sha256=tensor_digest({"gradient":gold["gradient"]}),
            original_A100_certificate=original_certificate,
            note="All causal parity is exact on the same L40S; not a new A100 or optimization efficacy certificate")
        write(directory / "parity.json",parity)
        measurements = {}
        functions = {
            "native_affinity_direct_logits_gradient":lambda:direct_graph(True),
            "native_dbmol_direct_logits_gradient":direct_graph,
            "native_dbmol_detached_embeddings_gradient":direct_embeddings,
            "vae_mapping_dbmol_raw_gradient":lambda:raw_learned(point),
            "production_learned_evaluate_gradient":lambda:objective.evaluate("learned",point,require_gradient=True),
            "production_guarded_evaluate_gradient":lambda:guard.evaluate(point,require_gradient=True),
            "guard_assess_only":lambda:guard.assess(point),
            "frozen_assert_only":lambda:objective.assert_unchanged(),
            "context_hash_only":lambda:comparison.context_digest(prepared),
            "vae_decode_mapping_only":lambda:bridge(*(lambda dh:(dh[1].atom_hidden,dh[1].bond_input,prepared.active,dh[1].charge_input))(vae.decode_with_features(point))),
        }
        for name,function in functions.items():
            timing, measured = measure(torch,function,config["warmups"],config["repeats"],deadline)
            measurements[name] = timing
            write(directory / (name+".json"),timing)
            del measured
        io_count = [0]
        def state_io():
            io_count[0] += 1
            return long._atomic(directory / ("diagnostic-state-%03d.pt" % io_count[0]),saved,tensor=True)
        timing, _ = measure(torch,state_io,2,5,deadline)
        measurements["atomic_committed_state_io_only"] = timing
        write(directory / "atomic_committed_state_io_only.json",timing)
        # A native gradient and one independent candidate, matching fixed-step
        # counts. Direct logits and latent variables differ in parameterization.
        def fixed_direct():
            current = direct_graph()
            proposed = {key:value-.0025*g for (key,value),g in zip(logits.items(),current["gradients"],strict=True)}
            with torch.no_grad(): candidate=direct_graph(given=proposed,gradient=False)
            return dict(candidate_loss=candidate["loss"],candidates=1)
        def fixed_latent(guarded, persist=False):
            current = guard.evaluate(point,require_gradient=True) if guarded else raw_learned(point)
            proposed = point-.0025*current["gradient"]
            if guarded:
                assessment = guard.assess(proposed)
                eligible = assessment["record"]["eligible"]
                candidate = guard.evaluate(proposed,assessment=assessment) if eligible else None
            else:
                eligible, candidate = True, raw_learned(proposed,False)
            if persist:
                journal = directory / "diagnostic-fixed-calls.jsonl"
                for event in ("baseline_attempted","baseline_completed","candidate_attempted","candidate_completed","update","commit"):
                    long._append(journal,dict(event=event,candidates=1,eligible=eligible,diagnostic=True,
                        latent=point.detach().cpu().tolist(), assessment=assessment["record"],
                        candidate_loss=None if candidate is None else candidate["scores"]["dbmol_total_loss"]))
            return dict(candidates=1,eligible=eligible,candidate_loss=None if candidate is None else candidate["scores"]["dbmol_total_loss"])
        for name,function in {
            "fixed_direct_logits_one_candidate":fixed_direct,
            "fixed_latent_raw_one_candidate":lambda:fixed_latent(False),
            "fixed_latent_guarded_one_candidate":lambda:fixed_latent(True),
            "fixed_latent_guarded_with_journal":lambda:fixed_latent(True,True)}.items():
            timing, last = measure(torch,function,config["warmups"],config["repeats"],deadline)
            measurements[name] = dict(**timing, last=last)
            write(directory/(name+".json"),measurements[name])

        spans = Spans(torch)
        with ExitStack() as patches:
            for owner,attr,name in [(comparison,"context_digest","context_hash"),(comparison,"_versions","model_version_checks"),
                (comparison,"_scalars","scalar_validation"),(vae,"decode_with_features","vae_decoder"),
                (bridge,"forward","mapping"),(scorer,"forward","boltz_dbmol_forward"),
                (scorer.affinity,"forward","affinity_branch"),
                (dbmol,"attached_trunk","structural_trunk"),
                (dbmol,"attached_confidence","confidence_head"),
                (scorer,"_reduce","DBMol_reduction"),
                (objective,"_gradient","backward"),(bank,"targets","feature_targets"),
                (bank,"loss","feature_mse"),(bank,"identity_metrics","feature_identity_audit"),
                (long,"classify_repair_graph","rdkit_classification"),(long,"_append","journal_fsync")]:
                patches.enter_context(patch.object(owner,attr,spans.wrap(name,getattr(owner,attr))))
            torch.cuda.synchronize()
            with spans.span("instrumented_fixed_guarded_step"):
                fixed_latent(True,True)
            components=spans.report()
        # Isolated VAE+mapping backward with captured native chemical cotangents.
        # This excludes both Boltz trunks while preserving their actual VJP.
        cotangents=direct["gradients"]
        def mapping_vjp():
            leaf=point.detach().clone().requires_grad_()
            d,h=vae.decode_with_features(leaf)
            outputs=bridge(h.atom_hidden,h.bond_input,prepared.active,h.charge_input)
            return torch.autograd.grad(tuple(outputs.values()),leaf,grad_outputs=cotangents)[0]
        timing, split_gradient=measure(torch,mapping_vjp,2,5,deadline)
        max_error=float((split_gradient-gold["gradient"]).abs().max())
        if not torch.allclose(split_gradient,gold["gradient"],atol=1e-5,rtol=1e-4):
            raise ValueError("Split native-input VJP does not recover original latent derivative")
        measurements["vae_mapping_only_native_cotangent_backward"]=timing
        record=dict(label=label,parity=parity,point_sha256=tensor_digest({"z":point}),
            timings=measurements,instrumented_components=components,split_gradient_max_error=max_error,
            component_note="Nested inclusive spans overlap; wall-exclusive values only describe instrumented host time. Uninstrumented matched medians are primary.",
            fixed_update_policy="LR0.0025; one finite proposal; no halving, clipping or normalization; diagnostic only",
            frozen_after=objective.assert_unchanged(full=True))
        write(directory/"result.json",record)
        result["cases"].append(record)
    after={key:state_digest(module.state_dict()) for key,module in objective.modules.items()}
    if before!=after:
        raise RuntimeError("Frozen model state changed during profiling")
    authenticate(config)
    result.update(status="completed_diagnostic_no_production_change",frozen_unchanged=True,
        memory_peak_bytes=torch.cuda.max_memory_allocated(),actual_model_calls=objective.calls,
        no_new_production_updates=True,existing_VAE_continuation="untouched")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("cpu","gpu"))
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--source-sha256",required=True)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpu-sha256")
    args=parser.parse_args()
    if (not os.environ.get("LSB_JOBID") or os.environ.get("LATENTMOL_SOURCE_SHA256")!=args.source_sha256
            or sha(args.config)!=args.plan_sha256):
        raise RuntimeError("Probe requires the exact allocated immutable release")
    config=validate_config(json.loads(args.config.read_text()))
    release=Path(__file__).resolve().parents[2]
    if args.config.resolve()!=release/CONFIG:
        raise ValueError("Probe must use this release's authenticated configuration")
    spec=importlib.util.spec_from_file_location("_runtime_operator",Path(__file__).with_name("embedding_operator.py"))
    operator=importlib.util.module_from_spec(spec); spec.loader.exec_module(operator)
    operator.authenticate(release,args.source_sha256,args.plan_sha256,CONFIG)
    validate_output(release,args.output,args.stage)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    header=dict(version=VERSION,stage=args.stage,source_sha256=args.source_sha256,
        plan_sha256=args.plan_sha256,job_id=os.environ["LSB_JOBID"])
    try:
        if args.stage=="cpu":
            result=cpu_stage(config,args.output,args.source_sha256,args.plan_sha256)
        else:
            cpu=json.loads((args.output.parent/"cpu/result.json").read_text())
            if (cpu["status"]!="passed" or cpu["source_sha256"]!=args.source_sha256
                    or cpu["plan_sha256"]!=args.plan_sha256 or sha(cpu["junit_asset"]["path"])!=cpu["junit_asset"]["sha256"]):
                raise ValueError("GPU profiling requires exact-release allocated CPU readiness")
            if (args.cpu_sha256 is None or sha(args.output.parent/"cpu/result.json")!=args.cpu_sha256
                    or any(cpu["junit"][key] for key in ("failures","errors","skipped"))):
                raise ValueError("GPU probe CPU dependency changed after authenticated submission")
            result=gpu_stage(config,args.output,started+3300)
    except BaseException as error:
        write(args.output/"result.json",dict(**header,status="failed_not_passed",error=str(error),
            error_type=type(error).__name__,traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        raise
    write(args.output/"result.json",{**header,**result,"elapsed_seconds":time.monotonic()-started})


if __name__=="__main__":
    main()
