"""Stage or submit one config-driven DTU restart campaign from the repository root."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import tarfile


EXPERIMENT = Path("experiments/latent-design/protenix-vae")
FRESH = EXPERIMENT / "restarts-fresh-dtu.json"
NATIVE = EXPERIMENT / "native-dtu.json"
ASSETS = EXPERIMENT / "one-atom-dtu.json"
WRAPPER = Path("scripts/hpc/protenix_latent_run.sh")
LAUNCHER = Path("scripts/hpc/hpc_launch.py")
STORAGE = Path(r"C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic\scripts\hpc\storage_accounting.py")
TRANSPORT = runpy.run_path(str(Path(__file__).with_name("protenix-restart-followup-20261107a") / "submit.py"))
remote, upload, sha = (TRANSPORT[key] for key in ("remote", "upload", "sha"))
BASE = TRANSPORT["BASE"]
GPU_RESOURCE = TRANSPORT["GPU_RESOURCE"] | {
    "gpua100": "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=36G",
    "gpua10": "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=20G",
}
BENCHMARK_DEVICE = {"a100-80gb": ("gpua100", "gpu80gb"), "v100-32gb": ("gpuv100", "gpu32gb")}
OUTPUT_MIB = 768
OUTPUT_FILES = 6000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def matched_source(receipt_path, config):
    """Verify a matched learning-rate repeat or fixed-length Boltz allocation arm."""
    receipt = read(receipt_path)
    argv = receipt["argv"]
    if receipt["returncode"] or argv[0] != "bsub":
        raise ValueError("Matched source requires a successful submission receipt")
    path = receipt_path.with_name("source.tar.gz")
    digest = sha(path)
    if digest != argv[argv.index("--archive-sha256") + 1]:
        raise ValueError("Matched source archive checksum differs from the receipt")
    old_config = argv[argv.index("--config") + 1]
    old_plan = argv[argv.index("--plan") + 1]
    with tarfile.open(path) as source:
        original = json.load(source.extractfile(old_config))
        allocation = config.get("study") == "boltz_search_methods"
        if allocation:
            if original.get("study") != config["study"] or set(config["cases"]) != {"fixed75"}:
                raise ValueError("Expected the fixed-75 follow-up to an allocation pilot")
            original["cases"] = {"fixed75": dict(original["cases"]["fixed150"], policy="fixed75", count=32)}
            original["tasks"] = {name.replace("fixed150-", "fixed75-"): dict(task, case="fixed75")
                                 for name, task in original["tasks"].items() if task["case"] == "fixed150"}
        else:
            learning_rate = config["design"]["learning_rate"]
            if (original["design"]["learning_rate"] == learning_rate
                or any("learning_rate" in arm for arm in config["arms"].values())):
                raise ValueError("Expected one changed learning rate for all matched arms")
            original["design"]["learning_rate"] = learning_rate
        if original != config:
            raise ValueError("Matched configuration differs beyond the declared change")
        raw_plan = source.extractfile(old_plan).read()
        if hashlib.sha256(raw_plan).hexdigest() != argv[argv.index("--plan-sha256") + 1]:
            raise ValueError("Matched GPU plan checksum differs from the receipt")
        files = ((EXPERIMENT / "restart-followup-fresh-dtu.json", NATIVE, ASSETS,
                  Path("scripts/hpc/restart_search_run.sh"), EXPERIMENT / config["search_config"],
                  EXPERIMENT / config["scorer"]["assets_config"]) if allocation else (FRESH, NATIVE, ASSETS, WRAPPER))
        for asset in files:
            if source.extractfile(asset.as_posix()).read() != asset.read_bytes():
                raise ValueError(f"Matched launch input changed: {asset}")
    return path, digest, {old_config, old_plan}, json.loads(raw_plan)


def stage(args, local, control, release):
    config = EXPERIMENT / args.config
    cfg = read(config)
    study = cfg.get("study")
    benchmark = study == "batch_throughput"
    restart_search = study == "restart_search"
    nesso_blend = cfg.get("nesso_blend")
    if nesso_blend and (not restart_search or cfg.get("optimization_space") != "vae_latent"
                        or not cfg.get("encoded_founder")):
        raise ValueError("The latent Nesso blend requires an encoded founder and native Protenix restart search")
    boltz_speed = study == "boltz_speed"
    boltz_precision = study == "boltz_precision"
    native_graph_search = study == "boltz_native_graph_search"
    boltz_methods = study in {"boltz_search_methods", "boltz_generations", "boltz_generation_search"} or native_graph_search
    boltz_taco = study == "boltz_tacogfn"
    boltz_direct = study == "boltz_direct_graph"
    without_vae = boltz_direct or native_graph_search
    boltz_probe = boltz_speed or boltz_precision or boltz_methods or boltz_taco or boltz_direct
    boltz_search = (restart_search or boltz_probe) and cfg.get("scorer", {}).get("kind") == "boltz2_dbmol"
    if boltz_probe and not boltz_search:
        raise ValueError("Boltz probe configs require the full Boltz2 DBMol scorer")
    if boltz_search and cfg.get("budget_status") == "awaiting_boltz_timing":
        raise ValueError("Choose the shorter baseline budget after the Boltz timing pilot before staging")
    wrapper = Path("scripts/hpc/restart_search_run.sh") if boltz_search else WRAPPER
    boltz_assets_path = EXPERIMENT / cfg["scorer"]["assets_config"] if boltz_search else None
    graph_edit = study == "graph_edit"
    near_valid = study == "near_valid"
    conformer_rescore = study == "conformer_rescore"
    matched_search = study in ("matched_search", "matched_width")
    synthesis_pilot = study == "synthesis_pilot"
    uses_tasks = study in ("initialization_noise", "lineage_feedback", "batch_throughput",
                           "step_cap", "graph_edit", "near_valid", "conformer_rescore") or matched_search or synthesis_pilot or boltz_probe
    uses_cohorts = uses_tasks or restart_search
    fresh_path = EXPERIMENT / "restart-followup-fresh-dtu.json" if uses_cohorts else FRESH
    if boltz_direct:
        module = "latentmol.design.boltz_direct_pilot"
    elif boltz_taco:
        module = "latentmol.design.boltz_tacogfn"
    elif boltz_methods:
        module = "latentmol.design.boltz_search_methods"
    elif boltz_probe:
        module = "latentmol.design.boltz_speed"
    elif restart_search or synthesis_pilot:
        module = "latentmol.design.protenix_search"
    elif benchmark:
        module = "latentmol.design.protenix_restart_benchmark"
    elif graph_edit:
        module = "latentmol.design.protenix_graph_edit_study"
    elif near_valid:
        module = "latentmol.design.protenix_near_valid"
    elif conformer_rescore:
        module = "latentmol.design.protenix_conformer_rescore"
    elif matched_search:
        module = "latentmol.design.protenix_matched_search"
    else:
        module = "latentmol.design.protenix_restart_study" if uses_tasks else "latentmol.design.protenix_restarts"
    startup_seconds = {"initialization_noise": 1800, "lineage_feedback": 2700, "step_cap": 2700,
                       "batch_throughput": 2700, "restart_search": 2700}.get(study, 900)
    matched_noise = study == "initialization_noise" and any(
        case.get("matched_fresh") for case in cfg["comparisons"].values())
    if matched_noise or graph_edit or near_valid or matched_search or synthesis_pilot:
        startup_seconds = 2700
    if boltz_probe:
        startup_seconds = 2700
    compact = study == "step_cap" or matched_noise or graph_edit or near_valid or conformer_rescore or synthesis_pilot
    output_mib, output_files = (192, 2000) if compact else (OUTPUT_MIB, OUTPUT_FILES)
    if synthesis_pilot and cfg.get("waves", 1) > 1:
        output_mib, output_files = OUTPUT_MIB, OUTPUT_FILES
    if study == "matched_width":
        output_mib = 576
    if args.a100_80gb and not restart_search:
        raise ValueError("Explicit 80 GB GPU selection requires a restart_search config")
    host_resource = "gpu80gb" if args.a100_80gb else None
    if benchmark:
        device = cfg.get("device", "a100-80gb")
        if not isinstance(device, str) or device not in BENCHMARK_DEVICE:
            raise ValueError(f"Unsupported throughput benchmark device: {device}")
        benchmark_queue, host_resource = BENCHMARK_DEVICE[device]
    host_requirements = ["-R", f"select[{host_resource}]"] if host_resource else []
    hold = getattr(args, "hold", False)
    worker_env = "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
    outputs = {}
    budgets = {}
    scoring_budgets = {}
    redocking_budgets = {}
    confirmation_budgets = {}
    queues = {}
    cpu_workers = {}
    continuations = {}
    configured = cfg["tasks" if uses_tasks else "arms"]
    selected = args.arm or list(configured)
    if len(set(selected)) != len(selected) or set(selected) - configured.keys():
        raise ValueError(f"Unknown or duplicate requested arms: {selected}")
    targeted = restart_search and any("target" in configured[arm] for arm in selected)
    if targeted and not all("target" in configured[arm] for arm in selected):
        raise ValueError("Stage target-specific and historical pocket runs separately")
    if targeted:
        from latentmol.design.protenix_search import resolve_config
        for arm in selected:
            resolve_config(cfg, arm)
        if type(cfg.get("output_cap_mib", output_mib)) is not int or cfg.get("output_cap_mib", output_mib) < 1:
            raise ValueError("Target output budget must be a positive integer MiB count")
        output_mib = cfg.get("output_cap_mib", output_mib)
    if boltz_probe:
        output_mib=cfg.get("output_cap_mib",256)
    for arm in selected:
        override = configured[arm]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", arm):
            raise ValueError(f"Invalid arm name: {arm}")
        cpu_workers[arm] = 4 if benchmark else 1
        if boltz_direct:
            from latentmol.design.boltz_direct_pilot import task_budget
            budgets[arm], scoring_budgets[arm] = task_budget(cfg, arm)
        elif boltz_methods:
            from latentmol.design.boltz_search_methods import resolve_case
            case = resolve_case(cfg, arm)
            budgets[arm] = case["updates"]
            scoring_budgets[arm] = case["fresh_budget"]
        elif boltz_precision:
            from latentmol.design.boltz_speed import precision_budget
            paired=precision_budget(cfg,arm)
            if type(cfg["cpu_threads"]) is not int or cfg["cpu_threads"] < 1:
                raise ValueError("Precision comparison requires positive CPU thread count")
            budgets[arm]=paired["updates"]
            scoring_budgets[arm]=paired["fresh_attempts"]
        elif boltz_speed:
            children,updates=cfg["children"],cfg["updates"]
            if any(type(value) is not int or value<1 for value in (children,updates,cfg["cpu_threads"])):
                raise ValueError("Boltz speed benchmark needs positive child/update/thread counts")
            cases=[cfg["cases"][name] for name in override["cases"]]
            if any(children%case["parallel"] for case in cases):
                raise ValueError("Matched child workload must divide every concurrency")
            budgets[arm]=children*updates*len(cases)
            cpu_workers[arm]=max(case["parallel"] for case in cases)
        elif benchmark:
            cases = cfg["cases"]
            batches = {case["batch_size"] for case in cases}
            measured = (cfg["design"]["updates"] * cfg["parents"] * cfg["children_per_parent"]
                        * len(cases) * cfg["repeats"])
            budgets[arm] = measured + cfg["warmup_steps"] * sum(batches)
        elif restart_search:
            options = cfg["search"] | override.get("search", {})
            design = cfg["design"] | override.get("design", {})
            counts = [options[key] for key in ("waves", "parents", "children", "cpu_workers")]
            if any(type(count) is not int or count < 1 for count in [*counts, design["updates"]]):
                raise ValueError(f"Restart search requires positive integer budget counts for {arm}")
            branches_per_wave = options["parents"] * options["children"]
            if options["initialization"] == "archived":
                branches = options["waves"] * branches_per_wave
            elif options["initialization"] == "bootstrap":
                branches = options.get("bootstrap_branches", 1) + (options["waves"] - 1) * branches_per_wave
            else:
                raise ValueError(f"Unknown restart initialization for {arm}")
            budgets[arm] = branches * design["updates"]
            cpu_workers[arm] = options["cpu_workers"]
            if resume := override.get("resume"):
                path = PurePosixPath(resume["path"])
                digest = resume["sha256"]
                if (not path.is_absolute() or ".." in path.parts
                        or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                    raise ValueError(f"Resume for {arm} requires an absolute checkpoint path and SHA-256")
                continuations[arm] = (str(path), digest)
        elif graph_edit:
            budgets[arm] = override["parents"] * cfg["design"]["updates"]
        elif near_valid:
            budgets[arm] = 2 * len(override["waves"]) * cfg["design"]["updates"]
        elif conformer_rescore:
            budgets[arm] = 0
        elif matched_search:
            search = cfg["search"]
            waves, windows, scores = (search[key] for key in ("waves", "branches_per_arm", "fresh_per_branch"))
            updates = cfg["design"]["updates"]
            if any(type(count) is not int or count < 1 for count in (waves, windows, scores, updates)):
                raise ValueError("Matched search needs positive integer update and scoring budgets")
            budgets[arm] = 3 * waves * windows * updates
            scoring_budgets[arm] = 3 * waves * windows * scores
        elif synthesis_pilot or boltz_taco:
            if boltz_taco:
                from latentmol.design.protenix_tacogfn_study import resolve_pair
                _, measured = resolve_pair(cfg, arm)
            else:
                from latentmol.design.protenix_synthesis_study import resolve_pilot
                _, measured = resolve_pilot(cfg, arm)
            budgets[arm] = measured["updates"]
            scoring_budgets[arm] = measured["fresh_attempts"]
            if "native_vina_attempts" in measured:
                redocking_budgets[arm] = measured["native_vina_attempts"]
            if "confirmation_fresh_attempts" in measured:
                confirmation_budgets[arm] = measured["confirmation_fresh_attempts"]
        elif uses_tasks:
            comparison = cfg["comparisons"][override["comparison"]]
            if study == "lineage_feedback":
                budgets[arm] = (3 * comparison["parents"] * comparison["cycles"]
                                * comparison["branches_per_parent"] * comparison["updates"])
            else:
                budgets[arm] = comparison["parents"] * sum(
                    sum(variant["updates"]) for variant in comparison["variants"].values())
        else:
            campaign = cfg["campaign"] | cfg.get("campaign_arms", {}).get(arm, {})
            design = cfg["design"] | override
            branches = 1 + (campaign["cycles"] - 1) * campaign["elite_count"] * campaign["branches_per_elite"]
            budgets[arm] = branches * design["updates"]
        queues[arm] = override.get("queue", args.queue) if uses_cohorts else args.queue
        outputs[arm] = f"{BASE}/runs/{args.name}-{arm}"
    if study == "boltz_generations":
        if len(selected) != 9 or set(selected) != set(configured):
            raise ValueError("Stage all candidate policies for the three paired generation seeds")
    if conformer_rescore:
        planned_fresh_attempts = len(cfg["panel"]) * len(cfg["geometry_seeds"])
        if (set(budgets) != {"panel"} or args.max_updates != 0 or args.max_fresh_attempts is None
                or not 0 < planned_fresh_attempts <= args.max_fresh_attempts):
            raise ValueError("Conformer rescore requires one panel, zero updates, and a sufficient fresh-attempt limit")
    elif boltz_precision or boltz_methods or boltz_taco or boltz_direct:
        if (not budgets or any(not 0 < updates <= args.max_updates for updates in budgets.values())
                or args.max_fresh_attempts is None
                or any(cap > args.max_fresh_attempts for cap in scoring_budgets.values())):
            raise ValueError("Boltz comparison needs sufficient optimizer and common FP32 scoring caps")
    elif args.max_fresh_attempts is not None or not budgets or any(
            not 0 < updates <= args.max_updates for updates in budgets.values()):
        raise ValueError(f"Arm exceeds the declared update budget: {budgets}")
    if set(queues.values()) - GPU_RESOURCE.keys():
        raise ValueError(f"Unsupported GPU queues: {queues}")
    if nesso_blend and set(queues.values()) - {"gpul40s", "gpua40", "gpua100"}:
        raise ValueError("The two resident scorers require a 44 GB or larger GPU allocation")
    if boltz_precision and set(queues.values()) - {"gpua10", "gpua40", "gpul40s", "gpua100"}:
        raise ValueError("BF16 precision comparison needs an Ampere-or-newer GPU queue")
    if benchmark and queues != {"bench": benchmark_queue}:
        raise ValueError(f"The throughput benchmark requires one bench task on {benchmark_queue}")
    if args.a100_80gb and set(queues.values()) != {"gpua100"}:
        raise ValueError("The 80 GB host requirement needs gpua100 for every selected arm")
    restart_host_resources = {}
    if restart_search or boltz_methods or boltz_taco or boltz_direct:
        for arm in selected:
            resource = configured[arm].get("host_resource")
            if resource is None:
                continue
            if (resource not in {"gpu32gb", "gpu80gb"}
                    or queues[arm] != {"gpu32gb": "gpuv100", "gpu80gb": "gpua100"}[resource]
                    or (host_resource is not None and resource != host_resource)):
                raise ValueError(f"Incompatible restart GPU host requirement for {arm}: {resource}")
            restart_host_resources[arm] = resource

    source_path = None
    if args.source_receipt:
        if uses_cohorts and study != "boltz_search_methods":
            raise ValueError("Matched archive staging supports learning-rate and allocation repeats")
        source_path, source_digest, replaced, prior = matched_source(args.source_receipt, cfg)
    else:
        prior = read(Path("outputs/protenix-restarts-20261107a/gpu-plan.json"))
    plan = {key: prior[key] for key in ("cuda_ownership_helper", "cuda_ownership_sha256", "mosaic_python")}
    plan.update(arms=outputs, queue=args.queue, planned_updates=budgets,
                output_cap_mib=output_mib, config_sha256=sha(config))
    if nesso_blend:
        plan["nesso_blend"] = nesso_blend
        plan["nesso_asset_provenance"] = cfg["nesso_asset_provenance"]
    checkpoint = cfg.get("vae_checkpoint")
    if checkpoint is not None:
        path = PurePosixPath(checkpoint["path"])
        if (without_vae or args.source_receipt or not path.is_absolute() or ".." in path.parts
                or not re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])
                or type(checkpoint["updates"]) is not int or checkpoint["updates"] < 1):
            raise ValueError("An explicit VAE checkpoint needs a pinned path, digest and training update")
        plan["vae_checkpoint"] = checkpoint
    if without_vae:
        plan.pop("mosaic_python")
        plan["optimization_space"] = cfg["optimization_space"]
    if study == "boltz_generations":
        plan["selection_required"] = True
    if conformer_rescore:
        plan["planned_fresh_attempts"] = {"panel": planned_fresh_attempts}
    elif scoring_budgets:
        plan["planned_fresh_attempts"] = scoring_budgets
    if confirmation_budgets:
        plan["planned_confirmation_fresh_attempts"] = confirmation_budgets
    if redocking_budgets:
        plan["planned_native_vina_attempts"] = redocking_budgets
    if boltz_speed:
        plan["warmup_gradient_probes"]={arm:cfg["children"]*len(configured[arm]["cases"]) for arm in selected}
        plan["managed_mps"]={arm:configured[arm].get("managed_mps",False) for arm in selected}
    if boltz_precision:
        plan["relaxed_gradient_observations"]={arm:precision_budget(cfg,arm)["relaxed_observations"] for arm in selected}
        plan["evaluation_execution"]=dict(tf32=False,autocast="none",checkpoint_every=1,checkpoint_preserve_rng_state=True)
        plan["scoring_budget_scope"]="Exhaustive common FP32 exact-pose graph union; consumed calls may be lower"
    if uses_cohorts:
        plan["queues"] = queues
        del plan["queue"]
    if restart_host_resources:
        plan["host_resources"] = restart_host_resources
    if restart_search:
        plan["a100_80gb"] = args.a100_80gb
        if targeted:
            plan["planned_fresh_attempts"] = {
                arm: resolve_config(cfg, arm)[2]["budget"]["fresh_attempts"] for arm in selected}
            if boltz_search and cfg["scorer"].get("timing_pilot"):
                plan["reference_forward_attempts"] = {arm: 1 for arm in selected}
    if continuations:
        plan["continuations"] = continuations
    if benchmark and "device" in cfg:
        plan["device"] = device
    if hold:
        plan["held_on_submission"] = True
    if source_path:
        plan["matched_source_sha256"] = source_digest
    local.mkdir()
    save(local / "gpu-plan.json", plan)
    if b"\r\n" in wrapper.read_bytes():
        raise ValueError("Unexpected CRLF in launch wrapper")
    package = local / "source.tar.gz"
    remote_archive = release + ".tar.gz"
    plan_name = EXPERIMENT / "restart-campaign-gpu-plan.json"
    with tarfile.open(package, "x:gz") as bundle:
        if source_path:
            with tarfile.open(source_path) as source:
                for member in source:
                    if member.name in replaced:
                        continue
                    if not member.isfile():
                        raise ValueError(f"Unexpected matched archive member: {member.name}")
                    with source.extractfile(member) as stream:
                        if boltz_methods and member.name == "src/latentmol/design/boltz_allocation_pilot.py":
                            # Keep the control implementation verbatim and register just the new fixed length.
                            content = stream.read() + b'\nPOLICY_LENGTHS["fixed75"] = 75\n'
                            member.size = len(content)
                            bundle.addfile(member, io.BytesIO(content))
                        else:
                            bundle.addfile(member, stream)
        else:
            extra_assets = [] if without_vae else [fresh_path, NATIVE, ASSETS]
            if nesso_blend:
                extra_assets.extend(sorted((EXPERIMENT.parent / "direct-jump").glob("*.py")))
            if boltz_direct:
                extra_assets.append(EXPERIMENT / "restart-operations/evaluate_boltz_direct.py")
            for path in [*sorted(Path("src/latentmol").rglob("*.py")), *extra_assets, wrapper]:
                bundle.add(path, arcname=path.as_posix())
            if boltz_search:
                bundle.add(boltz_assets_path, arcname=boltz_assets_path.as_posix())
                target_arms=[] if boltz_probe else selected
                if boltz_probe and not (boltz_taco or boltz_direct):
                    search_path=EXPERIMENT/cfg["search_config"]
                    bundle.add(search_path,arcname=search_path.as_posix())
                for arm in target_arms:
                    target = configured[arm]["target"]
                    pdb = target["pocket"]["pdb_id"]
                    structure = Path(configured[arm].get("structure_source",
                        Path("outputs/multitarget-setup-20261111") / pdb.lower() / f"{pdb}.cif"))
                    if sha(structure) != target["pocket"]["mmcif_sha256"]:
                        raise ValueError(f"Changed target structure: {pdb}")
                    bundle.add(structure, arcname=target["structure_path"])
        bundle.add(config, arcname=config.as_posix())
        bundle.add(local / "gpu-plan.json", arcname=plan_name.as_posix())

    expected = {plan["cuda_ownership_helper"]: plan["cuda_ownership_sha256"]}
    if not without_vae:
        assets, native, fresh = map(read, (ASSETS, NATIVE, fresh_path))
        if checkpoint is not None:
            assets = dict(assets, checkpoint=checkpoint["path"], checkpoint_sha256=checkpoint["sha256"])
        expected.update({
            assets["checkpoint"]: assets["checkpoint_sha256"],
            assets["native"]["weights_path"]: assets["native"]["weights_sha256"],
            native["context_ready"]: native["context_ready_sha256"],
            native["pocket_path"]: native["pocket_sha256"],
            fresh["pocket"]["path"]: fresh["pocket"]["sha256"],
        })
    if (targeted or boltz_probe) and not without_vae:
        # These are newly prepared in each allocation, not the BRD4 inputs.
        for path in (native["context_ready"], native["pocket_path"], fresh["pocket"]["path"]):
            expected.pop(path, None)
    if boltz_search:
        boltz_assets = read(boltz_assets_path)
        if boltz_direct and "pharmaconet_map" in cfg:
            for record in (cfg["pharmaconet_map"], cfg["vina_parent_panel"]["box"], cfg["vina_parent_panel"]["receptor"]):
                path = PurePosixPath(record["path"])
                if not path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"]):
                    raise ValueError("Expected a pinned absolute PharmacoNet or docking input")
                expected[str(path)] = record["sha256"]
        if not without_vae:
            expected.pop(assets["native"]["weights_path"], None)
        expected[boltz_assets["checkpoint"]] = boltz_assets["checkpoint_sha256"]
        confidence = boltz_assets["confidence_checkpoint"]
        expected[confidence["path"]] = confidence["sha256"]
        if boltz_probe:
            if native_graph_search or cfg.get("prior_founders"):
                history = cfg["history_exclusions"]
                expected[history["path"]] = history["sha256"]
            elif not boltz_direct:
                parent=cfg["parent_state"]
                expected[parent["path"]]=parent["sha256"]
            target_cfg = (cfg["prepared_config"] if (boltz_taco or boltz_direct) else
                          read(EXPERIMENT/cfg["search_config"])["arms"][cfg["target_arm"]]["prepared_config"])
            expected[target_cfg["path"]]=target_cfg["sha256"]
        for arm in selected:
            if prepared := configured[arm].get("prepared_config"):
                if not PurePosixPath(prepared["path"]).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", prepared["sha256"]):
                    raise ValueError("Prepared target config needs an absolute path and SHA256")
                expected[prepared["path"]] = prepared["sha256"]
    expected.update(continuations.values())
    if synthesis_pilot or boltz_taco:
        docking = cfg.get("vina_parent_panel", {})
        taco = cfg.get("tacogfn", {})
        panels = [cfg["parent_panel"]] if synthesis_pilot else []
        for source in (*panels, *cfg["synthesis_assets"].values(),
                       *[cfg[k] for k in ("anchor_bank", "anchor_control_bank") if k in cfg],
                       *cfg.get("anchor_libraries", {}).values(),
                       *[taco[k] for k in ("checkpoint", "pocket", "topologies", "openbabel_wheel") if k in taco],
                       *[docking[k] for k in ("receptor", "box") if k in docking]):
            path, digest = PurePosixPath(source["path"]), source["sha256"]
            if not path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Synthesis pilots require absolute asset paths and SHA-256 digests")
            expected[str(path)] = digest
    if conformer_rescore:
        for source in (cfg["source_result"], *cfg["sources"].values()):
            path, digest = PurePosixPath(source["path"]), source["sha256"]
            if not path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("Conformer rescore requires absolute source paths and SHA-256 digests")
            expected[str(path)] = digest
    torch_python, fresh_deps, fresh_data = (TRANSPORT[key] for key in (
        "TORCH_PYTHON", "FRESH_DEPS", "FRESH_DATA"))
    if boltz_search:
        required = [torch_python, str(PurePosixPath(boltz_assets["cache"]) / "mols")]
        if boltz_direct and "pharmaconet_map" in cfg:
            required.append(cfg["vina_parent_panel"]["python"])
    else:
        required = [native["mosaic_source"], plan["mosaic_python"], assets["native"]["source_root"],
                    torch_python, fresh_deps, fresh_data, fresh["ccd_root"]]
    if nesso_blend:
        nesso_assets = PurePosixPath(nesso_blend["assets_dir"])
        if not nesso_assets.is_absolute() or ".." in nesso_assets.parts:
            raise ValueError("Expected an absolute Nesso asset directory")
        pins = cfg["nesso_asset_provenance"]
        for key, relative in (("weights_sha256", "checkpoint/model.safetensors"),
                              ("hparams_sha256", "checkpoint/hparams.json"),
                              ("pinned_assets_manifest_sha256", "assets-ready.json")):
            expected[str(nesso_assets / relative)] = pins[key]
        required.extend(str(nesso_assets / path) for path in (
            "env/bin/python", "esm/model.safetensors", "ccd.pkl"))
    if synthesis_pilot or boltz_taco:
        for source in cfg["synthesis_assets"].values():
            required.extend(source[key] for key in ("python", "source_path") if key in source)
        if docking.get("python"):
            required.append(docking["python"])
        if taco.get("openbabel_site"):
            required.append(taco["openbabel_site"])
    if near_valid:
        source = PurePosixPath(cfg["source_run"])
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("Near-valid comparison requires an absolute saved source run")
        required.append(str(source / "run.json"))
        for wave in sorted({wave for arm in selected for wave in configured[arm]["waves"]}):
            required.extend(str(source / f"cycle-{wave:02d}" / filename)
                            for filename in ("wave-plan.json", "archive.json"))
    cohorts = cfg.get("parent_cohorts", []) if uses_cohorts else []
    probe = f"{BASE}/runs/{args.name}-capacity-probe"
    targets = [remote_archive, release, control, probe, *outputs.values()]
    if targeted:
        targets.extend(output + "-inputs" for output in outputs.values())
    cohort_geometry_seed = fresh["geometry_seed"] if cohorts else None
    save(local / "preflight.json", remote(f"""
import hashlib, json, subprocess
from pathlib import Path
for target in map(Path, {targets!r}):
    if target.exists():
        raise FileExistsError(target)
for path, expected in {expected!r}.items():
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('Changed input: ' + path)
for path in {required!r}:
    if not Path(path).exists():
        raise FileNotFoundError(path)
nesso = {nesso_blend!r}
if nesso:
    commit = subprocess.check_output(["git", "-C", str(Path(nesso["assets_dir"]) / "source"),
                                      "rev-parse", "HEAD"], text=True).strip()
    if commit != {cfg.get("nesso_asset_provenance", {}).get("source_commit")!r}:
        raise ValueError("Nesso checkout differs from the pinned commit")
for root in map(Path, {cohorts!r}):
    if not (root / 'panel.json').is_file():
        raise FileNotFoundError(root / 'panel.json')
    fresh = json.loads((root / 'fresh' / 'summary.json').read_text())
    if fresh['status'] != 'completed' or fresh['geometry_seed'] != {cohort_geometry_seed!r}:
        raise ValueError('Source cohort fresh score protocol differs: ' + str(root))
queues = {sorted(set(queues.values()))!r}
scheduler = subprocess.run(['bqueues', *queues], capture_output=True, text=True, check=True, timeout=15)
active = {{fields[0] for line in scheduler.stdout.splitlines() if len(fields := line.split()) > 2
           and fields[2] == 'Open:Active'}}
if active != set(queues):
    raise ValueError('Queue is not active: ' + scheduler.stdout)
control = Path({control!r})
control.mkdir()
for arm in {list(outputs)!r}:
    (control / arm).mkdir()
print(json.dumps(dict(inputs={expected!r}, outputs={outputs!r}, scheduler=scheduler.stdout)))
"""))
    uploads = [(package, remote_archive), (LAUNCHER, f"{control}/hpc_launch.py"),
               (STORAGE, f"{control}/storage_accounting.py")]
    for source, target in uploads:
        upload(source, target)
    save(local / "storage.json", remote(f"""
import hashlib, subprocess, tarfile
from pathlib import Path
for path, expected in {[(target, sha(source)) for source, target in uploads]!r}:
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('Upload checksum differs: ' + path)
with tarfile.open({remote_archive!r}) as bundle:
    subprocess.run(['bash', '-n'], input=bundle.extractfile({wrapper.as_posix()!r}).read(), check=True)
probe = Path({probe!r})
probe.mkdir()
try:
    result = subprocess.run(['/usr/bin/python3', {uploads[-1][1]!r}, '--output-root', str(probe),
        '--cap-mib', {str(output_mib * len(outputs))!r}, '--file-cap', {str(output_files * len(outputs))!r}],
        check=True, text=True, capture_output=True)
    print(result.stdout)
finally:
    probe.rmdir()
"""))
    archive_sha = sha(package)
    plan_sha = sha(local / "gpu-plan.json")
    commands = {}
    for arm, output in outputs.items():
        receipts = f"{control}/{arm}"
        queue = queues[arm]
        parallel_cpu = cpu_workers[arm] > 1
        cpu_slots = cpu_workers[arm] + 4 if parallel_cpu else 4
        # Override the wrapper's thread defaults after it resolves GPU ownership.
        thread_env = (["/usr/bin/env", "DBMOL_CPU_THREADS=1", "OMP_NUM_THREADS=1",
                       "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1"] if parallel_cpu else [])
        arm_requirements=host_requirements
        if arm in restart_host_resources:
            arm_requirements=["-R",f"select[{restart_host_resources[arm]}]"]
        gpu_request=GPU_RESOURCE[queue]
        arm_worker_env=worker_env
        if nesso_blend:
            arm_worker_env = "all,~CUDA_VISIBLE_DEVICES"
            gpu_request = "num=1:mode=exclusive_process:j_exclusive=yes:mps=yes:gmem=44G"
        elif boltz_precision:
            cpu_slots=max(4,cfg["cpu_threads"]+1)
            thread_env=["/usr/bin/env",f"OMP_NUM_THREADS={cfg['cpu_threads']}",f"MKL_NUM_THREADS={cfg['cpu_threads']}",f"OPENBLAS_NUM_THREADS={cfg['cpu_threads']}"]
            gpu_request="num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=20G"
        elif boltz_speed:
            task=configured[arm]
            cpu_slots=cpu_workers[arm]*cfg["cpu_threads"]+2
            thread_env=["/usr/bin/env",f"OMP_NUM_THREADS={cfg['cpu_threads']}",f"MKL_NUM_THREADS={cfg['cpu_threads']}",f"OPENBLAS_NUM_THREADS={cfg['cpu_threads']}"]
            arm_worker_env="all,~CUDA_VISIBLE_DEVICES"
            if task.get("host_resource"):
                arm_requirements=["-R",f"select[{task['host_resource']}]"]
            memory={"gpua10":"20G","gpua40":"44G","gpul40s":"44G","gpua100":"72G"}[queue]
            gpu_request=f"num=1:mode=exclusive_process:j_exclusive=yes:mps={'yes' if task.get('managed_mps') else 'no'}:gmem={memory}"
        elif boltz_search and cfg["scorer"].get("parallel_workers",1)>1:
            workers=cfg["scorer"]["parallel_workers"]
            cpu_slots=max(cpu_slots,workers*cfg["scorer"].get("worker_cpu_threads",2)+max(1,cpu_workers[arm]))
            arm_worker_env="all,~CUDA_VISIBLE_DEVICES"
            gpu_request=gpu_request.replace("mps=no","mps=yes")
        resume_args = (["--resume-state", continuations[arm][0], "--resume-sha256", continuations[arm][1]]
                       if arm in continuations else [])
        scoring_args = (["--boltz-assets", boltz_assets_path.as_posix()] if boltz_search else
            ["--mosaic-python", plan["mosaic_python"], "--native-config", NATIVE.as_posix(),
             "--fresh-config", fresh_path.as_posix(), "--fresh-deps", fresh_deps, "--fresh-data", fresh_data])
        commands[arm] = [
            "bsub", *(["-H"] if hold else []), "-J", f"dbmol-{args.name}-{arm}",
            "-q", queue, "-n", str(cpu_slots),
            "-R", "span[hosts=1]", "-R", "rusage[mem=4096]", *arm_requirements,
            "-gpu", ("num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=12G"
                      if synthesis_pilot and queue == "gpuv100" else gpu_request),
            "-env", arm_worker_env,
            "-W", f"{args.hours:02d}:00", "-oo", f"{receipts}/run.%J.out", "-eo", f"{receipts}/run.%J.err",
            "/usr/bin/python3", f"{control}/hpc_launch.py", "--archive", remote_archive,
            "--archive-sha256", archive_sha, "--release", release, "--receipt", f"{receipts}/launch.json",
            "--startup-seconds", str(startup_seconds), "--run-seconds", str(args.hours * 3600 - 900),
            "--", "bash", wrapper.as_posix(), *(["--nesso-blend"] if nesso_blend else []),
            *thread_env, torch_python, "-m", module,
            "--config", config.as_posix(), "--arm", arm, "--output", output,
            *([] if without_vae else ["--checkpoint", assets["checkpoint"], "--checkpoint-sha256", assets["checkpoint_sha256"]]),
            *scoring_args,
            "--plan", plan_name.as_posix(), "--plan-sha256", plan_sha,
            *resume_args,
        ]
    save(local / "commands.json", commands)
    print(f"Staged {len(commands)} arms from source {archive_sha}")


def redock_command(gpu, job, control, arm):
    if "latentmol.design.boltz_direct_pilot" not in gpu:
        raise ValueError("The redocking readout requires a direct-graph optimization")
    flag = lambda name: gpu[gpu.index(name) + 1]
    release, output = flag("--release"), flag("--output")
    log = f"{control}/{arm}"
    return ["bsub", "-J", f"dbmol-direct-vina-{job}", "-w", f"done({job})", "-q", "hpc",
            "-n", "1", "-R", "rusage[mem=4096]", "-W", "02:00", "-oo", f"{log}/vina.%J.out",
            "-eo", f"{log}/vina.%J.err", "/usr/bin/python3", f"{control}/hpc_launch.py",
            "--archive", flag("--archive"), "--archive-sha256", flag("--archive-sha256"),
            "--release", release, "--receipt", f"{log}/vina-launch.json", "--startup-seconds", "6800",
            "--run-seconds", "6900", "--", "/usr/bin/env", f"PYTHONPATH={release}/src", "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", TRANSPORT["TORCH_PYTHON"],
            (EXPERIMENT / "restart-operations/evaluate_boltz_direct.py").as_posix(),
            "--run", output, "--output", output + "-vina", "--vina-config", flag("--config")]


def submit(local, control, *, redock=False):
    if read(local / "gpu-plan.json").get("selection_required"):
        raise ValueError("Use the pilot-dependent generation selector to submit the selected two policies")
    if (local / "superseded.json").exists():
        raise ValueError("This staged campaign was superseded; no jobs may be submitted from it")
    if read(local / "storage.json")["status"] != "passed":
        raise ValueError("Missing successful capacity check")
    for arm, argv in read(local / "commands.json").items():
        prefix = "vina-" if redock else ""
        if redock:
            gpu = read(local / f"{arm}-submission.json")
            if gpu["argv"] != argv or gpu["returncode"] or not isinstance(gpu["job_id"], int):
                raise ValueError("Expected the matching confirmed GPU submission")
            argv = redock_command(argv, gpu["job_id"], control, arm)
        output = argv[argv.index("--output") + 1]
        queue = argv[argv.index("-q") + 1]
        receipt = remote(f"""
import json, re, subprocess
from pathlib import Path
root = Path({f"{control}/{arm}"!r})
argv = {argv!r}
receipt_path = root / {f"{prefix}submission.json"!r}
if receipt_path.exists():
    receipt = json.loads(receipt_path.read_text())
    if receipt['argv'] != argv:
        raise ValueError('Existing submission command differs')
else:
    with (root / {f"{prefix}submit-intent.json"!r}).open('x') as stream:
        json.dump(argv, stream)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    job = re.search(r'Job <([0-9]+)>', result.stdout)
    receipt = dict(job_id=int(job[1]) if job else None, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, argv=argv, queue={queue!r}, output={output!r})
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
""")
        path = local / f"{arm}-{prefix}submission.json"
        if not path.exists():
            save(path, receipt)
        elif read(path) != receipt:
            raise ValueError(f"Local submission differs for {arm}")
        if receipt["returncode"] or receipt["job_id"] is None:
            raise RuntimeError(f"Scheduler did not confirm {arm}: {receipt['stderr']}")
        print(arm, receipt["stdout"].strip(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "submit", "redock"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--config")
    parser.add_argument("--arm", action="append", help="Stage only the named arm; repeat to select several")
    parser.add_argument("--source-receipt", type=Path, help="Reuse sealed source from a matched submission")
    parser.add_argument("--queue", choices=tuple(GPU_RESOURCE), default="gpul40s")
    parser.add_argument("--a100-80gb", action="store_true", help="Require an 80 GB A100 for a restart_search config")
    parser.add_argument("--hold", action="store_true", help="Seal commands that submit jobs in held state")
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--max-updates", type=int, required=False)
    parser.add_argument("--max-fresh-attempts", type=int, help="Scoring cap for a conformer rescore or FP32 precision comparison")
    args = parser.parse_args()
    if not WRAPPER.is_file() or not re.fullmatch(r"(?:protenix|boltz)-restart-[a-z0-9-]+", args.name):
        parser.error("Run from the DBMol root with a unique Protenix restart campaign name")
    local = Path("outputs") / args.name
    control = f"/zhome/95/e/206121/dbmol-control/{args.name}"
    if args.phase == "stage":
        if (not args.config or not re.fullmatch(r"restart-[a-z0-9-]+\.json", args.config)
                or args.max_updates is None or args.max_updates < 0 or not 1 <= args.hours <= 24):
            parser.error("Staging requires a restart config, update limit, and 1–24 hour walltime")
        stage(args, local, control, f"{BASE}/releases/{args.name}")
    else:
        if args.arm or args.a100_80gb or args.hold or args.max_fresh_attempts is not None:
            parser.error("Arm, GPU, hold and fresh-budget options belong to staging; submit uses sealed commands")
        submit(local, control, redock=args.phase == "redock")


if __name__ == "__main__":
    main()
