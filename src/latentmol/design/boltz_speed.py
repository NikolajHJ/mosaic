"""Matched child-throughput and numerical probes for frozen Boltz2 DBMol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _cosine(left, right):
    dot = sum(x*y for x,y in zip(left,right,strict=True))
    a = math.sqrt(sum(x*x for x in left))
    b = math.sqrt(sum(y*y for y in right))
    return dot/(a*b) if a and b else None


def group_workers(group):
    return [group["results"][index]["result"] for index in sorted(group["results"],key=int)]


def group_device(group):
    devices=[worker["device"] for worker in group_workers(group)]
    if (any(d["device_count"]!=1 for d in devices)
            or len({d["device_name"] for d in devices})!=1
            or len({json.dumps(d["environment"],sort_keys=True) for d in devices})!=1):
        raise ValueError("Concurrent workers do not share the same one-GPU allocation")
    uuids={d["device_uuid"] for d in devices if d.get("device_uuid")}
    if len(uuids)>1:raise ValueError("Concurrent workers reported different physical GPUs")
    return dict(uuid=next(iter(uuids),None),name=devices[0]["device_name"],
        scope="matched_gpu_uuid" if all(d.get("device_uuid") for d in devices) else "inherited_visibility_only")


def sample_gpu(group_root, stop, samples):
    """Read only this allocation's worker-reported GPU UUID; no parent CUDA."""
    while not stop.wait(1.):
        try:
            path=group_root/"group.json"
            if not path.exists():continue
            group=json.loads(path.read_text())
            ready=list(group.get("ready",{}).values())
            if not ready:continue
            uuid=ready[0]["device"].get("device_uuid")
            if not uuid:continue
            result=subprocess.run(["nvidia-smi","--id="+uuid,
                "--query-gpu=uuid,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],
                capture_output=True,text=True,timeout=3)
            fields=[value.strip() for value in result.stdout.strip().split(",")]
            if result.returncode or len(fields)!=4:continue
            samples.append(dict(timestamp=time.perf_counter(),uuid=fields[0],used_mib=float(fields[1]),
                                total_mib=float(fields[2]),gpu_busy_percent=float(fields[3])))
        except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):
            continue


def compare_children(reference, result):
    """Compare identical child seeds at their common initial latent points."""
    baseline = {row["child"]: row for worker in group_workers(reference) for row in worker["children"]}
    comparisons = []
    for worker in group_workers(result):
        for row in worker["children"]:
            ref = baseline[row["child"]]
            if row.get("initial_point_sha256")!=ref.get("initial_point_sha256"):
                raise ValueError("Matched child seeds produced different initial latent/mask tensors")
            a, b = ref["initial_binding_gradient"], row["initial_binding_gradient"]
            denom = math.sqrt(sum(x*x for x in a))
            comparisons.append(dict(child=row["child"],
                initial_loss_delta=row["initial_binding_loss"]-ref["initial_binding_loss"],
                binding_gradient_cosine=_cosine(a,b),
                binding_gradient_relative_l2=math.sqrt(sum((x-y)**2 for x,y in zip(a,b,strict=True)))/denom if denom else None,
                combined_gradient_cosine=_cosine(ref["initial_combined_gradient"],row["initial_combined_gradient"]),
                observed_geometry_bins_equal=(row["geometry_bins"]==ref["geometry_bins"]
                    if row.get("geometry_bins") and ref.get("geometry_bins") else None),
                terminal_hard_graph_equal=row["terminal_hard_graph"]==ref["terminal_hard_graph"],
                terminal_latent_l2=math.sqrt(sum((x-y)**2 for x,y in zip(row["terminal_latent"],ref["terminal_latent"],strict=True)))))
    return comparisons


def load_inputs(payload):
    """Common CPU-only initial state and authenticated benchmark assets."""
    import torch
    from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity
    from latentmol.design.protenix_restarts import Candidate
    from latentmol.evaluation.evaluate import load_model
    from latentmol.experiments import file_sha256
    from latentmol.representation import HardGraphBatch

    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Native speed benchmark requires an allocated job")
    torch.set_num_threads(payload["cpu_threads"])
    if file_sha256(Path(payload["checkpoint"])) != payload["checkpoint_sha256"]:
        raise ValueError("VAE checkpoint changed")
    if file_sha256(Path(payload["parent_state"]["path"])) != payload["parent_state"]["sha256"]:
        raise ValueError("Saved parent state changed")
    decoder, _ = load_model(Path(payload["checkpoint"]), "cpu")
    decoder.eval().requires_grad_(False)
    decoder_identity=_model_identity(decoder)
    state = torch.load(payload["parent_state"]["path"], map_location="cpu", weights_only=True)
    parent = min(state["archive"].values(), key=lambda row: (row["fresh_loss"],row["smiles"]))
    if int(parent["active"].sum()) != payload["design"]["occupied_atoms"]:
        raise ValueError("Benchmark parent occupancy differs from selected target")
    candidate = Candidate(parent["smiles"], parent["latent"],
        HardGraphBatch(parent["atoms"],parent["charges"],parent["bonds"]),
        parent["relaxed_loss"],parent["branch"],parent["step"])
    design = LatentDesignConfig.for_restart(payload["design"])
    scorer_config = json.loads(Path(payload["prepared_config"]["path"]).read_text())
    if file_sha256(Path(payload["prepared_config"]["path"])) != payload["prepared_config"]["sha256"]:
        raise ValueError("Prepared Boltz config changed")
    scorer_config["execution"] = payload["execution"]
    return decoder,decoder_identity,parent,candidate,design,scorer_config


def worker(payload, context):
    """Own one model pair and run independent seeded children in this process."""
    import torch
    from latentmol.chemistry import inspect_graph
    from latentmol.design.latent_protenix import _model_identity, evaluate_point, hard_from_graph, latent_step
    from latentmol.design.protenix_restart_proposals import propose_start
    from latentmol.gpu_process_probe import cuda_operation
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.guidance.boltz_search import BoltzSearchScorer, hard_graph_record, hard_pose_key
    decoder,decoder_identity,parent,candidate,design,scorer_config=load_inputs(payload)
    scorer = None
    cleanup = dict(cuda_synchronized=False,frozen_checked=False)
    try:
        retained, device = cuda_operation()
        device["total_memory_bytes"]=torch.cuda.get_device_properties(0).total_memory
        device["mps_active_thread_percentage"]=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
        scorer = BoltzSearchScorer.load(scorer_config, decoder.spec, device="cuda",
            source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"])
        # VAE occupied slots need not be the context's contiguous CCD slots.
        def evaluate(latent, step):
            return evaluate_point(decoder,lambda graph: scorer.score_at(graph,parent["active"]),
                latent,parent["active"],design,step)
        starts, probes = {}, []
        geometry_bins={}
        hooks=[]
        for name,module in scorer.scorer.named_modules():
            if name.endswith("dist_bin_pairwise_embed"):
                def capture(_module,arguments,_name=name):
                    value=arguments[0].detach().cpu().contiguous()
                    geometry_bins[_name]=dict(shape=list(value.shape),sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
                hooks.append(module.register_forward_pre_hook(capture))
        for child in payload["children"]:
            context.check_abort()
            latent, _, proposal = propose_start(decoder, candidate, parent["active"], policy="latent",
                seed=payload["child_seed"]+child, temperature=design.temperature_initial)
            starts[child] = latent.detach().clone()
            evaluated = evaluate(latent,0)
            probes.append(dict(child=child,initial_binding_loss=float(evaluated["binding"]),
                initial_binding_gradient=evaluated["binding_gradient"].flatten().tolist(),
                initial_combined_gradient=evaluated["gradient"].flatten().tolist(),proposal=proposal,
                geometry_bins=dict(geometry_bins),initial_point_sha256=hashlib.sha256(
                    latent.detach().contiguous().numpy().tobytes()+parent["active"].contiguous().numpy().tobytes()).hexdigest()))
            del evaluated
        for hook in hooks:hook.remove()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if payload.get("ready_path"):
            write(payload["ready_path"],dict(status="models_and_inputs_ready",nonce=os.environ.get("DBMOL_LAUNCH_NONCE"),
                job_id=os.environ.get("LSB_JOBID"),scope="First serial native worker loaded; finite attached warmup gradients checked"))
        context.ready(dict(device=device, execution=scorer.execution_record,
            parent_smiles=parent["smiles"],children=payload["children"],cpu_threads=payload["cpu_threads"]))
        measured_started = time.perf_counter()
        results = []
        for probe in probes:
            child = probe["child"]
            point = starts[child].clone()
            times, trajectory = [], []
            generator = torch.Generator().manual_seed(payload["child_seed"]+child)
            for step in range(payload["updates"]):
                context.check_abort()
                tick = time.perf_counter()
                evaluated = evaluate(point,step)
                point, update = latent_step(point,evaluated["binding_gradient"],evaluated["gradient"],design,step,generator)
                times.append(time.perf_counter()-tick)
                trajectory.append(dict(step=step,loss=float(evaluated["binding"]),
                    chemistry=float(evaluated["chemistry"]),gradient_norms=evaluated["gradient_norms"],update=update))
                del evaluated
            with torch.no_grad():
                graph = graph_from_decoder(decoder.decode(point),decoder.spec,fixed_active=parent["active"],temperature=design.temperature_initial)
                hard = hard_from_graph(graph)
                checked = inspect_graph(hard.atoms[0],hard.charges[0],hard.bonds[0],decoder.spec)
            result = dict(probe,terminal_latent=point.flatten().tolist(),terminal_hard_graph=hard_pose_key(hard_graph_record(hard)),
                terminal_valid=checked.smiles is not None,terminal_smiles=checked.smiles,
                update_seconds=times,trajectory=trajectory)
            results.append(result)
        torch.cuda.synchronize()
        measured_ended = time.perf_counter()
        summary = dict(worker=context.index,pid=os.getpid(),device=device,execution=scorer.execution_record,
            measured_started=measured_started,measured_ended=measured_ended,measured_seconds=measured_ended-measured_started,
            updates=len(results)*payload["updates"],children=results,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(context.output/"measurement.json",summary)
        return summary
    finally:
        # Let outstanding work finish before leaving an LSF-managed MPS client.
        cleanup["scorer_created"]=scorer is not None
        try:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            cleanup["cuda_synchronized"]=True
        finally:
            try:
                if scorer is not None:
                    scorer.close()
                    cleanup["frozen_checked"]=True
                if _model_identity(decoder)!=decoder_identity or any(p.grad is not None for p in decoder.parameters()):
                    raise ValueError("Frozen VAE changed during child benchmark")
                cleanup["decoder_frozen_checked"]=True
            finally:
                write(context.output/"cleanup.json",cleanup)


def proxy_worker(payload,context):
    """Exercise the search scorer's CPU autograd bridge with parallel GPU VJPs."""
    import torch
    from latentmol.chemistry import inspect_graph
    from latentmol.design.latent_protenix import _model_identity,finish_point,prepare_point,hard_from_graph,latent_step
    from latentmol.design.protenix_restart_proposals import propose_start
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.guidance.boltz_search import hard_graph_record,hard_pose_key
    from latentmol.guidance.boltz_vjp import BoltzVJPScorer

    decoder,decoder_identity,parent,candidate,design,scorer_config=load_inputs(payload)
    pool=None;cleanup=dict(cuda_synchronized=False,frozen_checked=False,scorer_created=False)
    try:
        pool=BoltzVJPScorer.load(scorer_config,decoder.spec,workers=payload["parallel_workers"],
            source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"],output=context.output/"vjp-workers",
            managed_mps=True,worker_cpu_threads=payload["cpu_threads"])
        device=pool.device_identity
        starts={};probes={};peaks={}
        for child in payload["children"]:
            latent,_,proposal=propose_start(decoder,candidate,parent["active"],policy="latent",
                seed=payload["child_seed"]+child,temperature=design.temperature_initial)
            starts[child]=latent.detach().clone()
            probes[child]=dict(child=child,proposal=proposal,geometry_bins={},initial_point_sha256=hashlib.sha256(
                latent.detach().contiguous().numpy().tobytes()+parent["active"].contiguous().numpy().tobytes()).hexdigest())
        def evaluate(points,step):
            prepared=[prepare_point(decoder,p,parent["active"],design,step) for p in points]
            scores=pool.score_batch([p.graph for p in prepared],[parent["active"]]*len(prepared))
            results=[finish_point(decoder,p,s) for p,s in zip(prepared,scores,strict=True)]
            for row in pool.last_worker_metrics:
                old=peaks.setdefault(row["worker"],dict(allocated=0,reserved=0))
                old["allocated"]=max(old["allocated"],row["peak_allocated_bytes"])
                old["reserved"]=max(old["reserved"],row["peak_reserved_bytes"])
            return results
        groups=[payload["children"][i:i+pool.workers] for i in range(0,len(starts),pool.workers)]
        for children in groups:
            context.check_abort()
            values=evaluate([starts[i] for i in children],0)
            for child,value in zip(children,values,strict=True):
                probes[child].update(initial_binding_loss=float(value["binding"]),
                    initial_binding_gradient=value["binding_gradient"].flatten().tolist(),
                    initial_combined_gradient=value["gradient"].flatten().tolist())
            del values
        if payload.get("ready_path"):
            write(payload["ready_path"],dict(status="models_and_inputs_ready",nonce=os.environ.get("DBMOL_LAUNCH_NONCE"),job_id=os.environ.get("LSB_JOBID")))
        context.ready(dict(device=device,execution=scorer_config["execution"],children=payload["children"],
                           parent_smiles=parent["smiles"],native_worker_pids=[x["pid"] for x in pool.identity["workers"]]))
        measured_started=time.perf_counter();children_results=[]
        for children in groups:
            points=[starts[i].clone() for i in children]
            times={i:[] for i in children};traces={i:[] for i in children}
            generators={i:torch.Generator().manual_seed(payload["child_seed"]+i) for i in children}
            for step in range(payload["updates"]):
                context.check_abort();tick=time.perf_counter()
                values=evaluate(points,step)
                for k,(child,value) in enumerate(zip(children,values,strict=True)):
                    points[k],update=latent_step(points[k],value["binding_gradient"],value["gradient"],design,step,generators[child])
                    traces[child].append(dict(step=step,loss=float(value["binding"]),chemistry=float(value["chemistry"]),
                        gradient_norms=value["gradient_norms"],update=update))
                elapsed=time.perf_counter()-tick
                for child in children:times[child].append(elapsed)
                del values
            for child,point in zip(children,points,strict=True):
                with torch.no_grad():
                    graph=graph_from_decoder(decoder.decode(point),decoder.spec,fixed_active=parent["active"],temperature=design.temperature_initial)
                    hard=hard_from_graph(graph)
                    checked=inspect_graph(hard.atoms[0],hard.charges[0],hard.bonds[0],decoder.spec)
                children_results.append(dict(probes[child],terminal_latent=point.flatten().tolist(),
                    terminal_hard_graph=hard_pose_key(hard_graph_record(hard)),terminal_valid=checked.smiles is not None,
                    terminal_smiles=checked.smiles,update_seconds=times[child],trajectory=traces[child]))
        measured_ended=time.perf_counter()
        summary=dict(worker=context.index,pid=os.getpid(),device=device,execution=scorer_config["execution"],backend="vjp_pool",
            measured_started=measured_started,measured_ended=measured_ended,measured_seconds=measured_ended-measured_started,
            updates=len(children_results)*payload["updates"],children=children_results,
            peak_allocated_bytes=max(x["allocated"] for x in peaks.values()),peak_reserved_bytes=sum(x["reserved"] for x in peaks.values()),
            native_worker_peaks=peaks,parent_cuda_initialized=torch.cuda.is_initialized())
        if summary["parent_cuda_initialized"]:raise ValueError("VJP parent unexpectedly created CUDA state")
        write(context.output/"measurement.json",summary)
        return summary
    finally:
        try:
            if pool is not None:
                cleanup["scorer_created"]=True
                try:
                    pool.close();cleanup["frozen_checked"]=True
                finally:
                    stopped=pool.shutdown()
                    cleanup["cuda_synchronized"]=stopped.get("cooperative_cleanup_complete",False)
                    cleanup["shutdown"]=stopped
                    if stopped.get("requires_job_stop") or not cleanup["cuda_synchronized"]:
                        raise RuntimeError("VJP workers did not shut down cooperatively; stop this MPS allocation")
            if _model_identity(decoder)!=decoder_identity or any(p.grad is not None for p in decoder.parameters()):
                raise ValueError("Frozen VAE changed during proxy benchmark")
            cleanup["decoder_frozen_checked"]=True
        finally:write(context.output/"cleanup.json",cleanup)


def run_benchmark(config, task_name, runtime):
    from latentmol.guidance.boltz_parallel import ParallelWorkerError, run_spawn_group

    task = config["tasks"][task_name]
    search_path = runtime.config.parent/config["search_config"]
    search = json.loads(search_path.read_text())
    target = search["arms"][config["target_arm"]]
    design = search["design"] | target.get("design",{})
    total_children, updates = config["children"],config["updates"]
    output = runtime.output
    output.mkdir(parents=True)
    write(output/"run.json",dict(config=config,task=task_name,source_sha256=os.environ.get("LATENTMOL_SOURCE_SHA256")))
    records, baseline = [], None
    memory_profiles={}
    for case_name in task["cases"]:
        case = config["cases"][case_name]
        count = case["parallel"]
        backend=case.get("backend","independent_workers")
        if backend not in ("independent_workers","vjp_pool"):
            raise ValueError("Unknown branch execution backend")
        outer_count=1 if backend=="vjp_pool" else count
        if count>1 and not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
            raise RuntimeError("Concurrent CUDA workers require LSF-managed MPS")
        if total_children%count:
            raise ValueError("Matched child count must be divisible by concurrency")
        policy=json.dumps(case["execution"],sort_keys=True)
        if count>1 and policy in memory_profiles:
            memory=memory_profiles[policy]
            if count*memory["reserved"]>0.90*memory["total"]:
                records.append(dict(case=case_name,status="skipped_memory_estimate",parallel=count,
                    predicted_reserved_bytes=count*memory["reserved"],total_memory_bytes=memory["total"]))
                write(output/f"summary-{len(records):02d}.json",dict(task=task_name,cases=records))
                continue
        case_root=output/case_name
        jobs=[dict(children=list(range(i,total_children,outer_count)),updates=updates,cpu_threads=config["cpu_threads"],
            checkpoint=str(runtime.checkpoint),checkpoint_sha256=runtime.checkpoint_sha256,
            parent_state=config["parent_state"],design=design,prepared_config=target["prepared_config"],
            execution=case["execution"],child_seed=config["child_seed"],parallel_workers=count,
            ready_path=os.environ.get("DBMOL_LAUNCH_READY") if not records and i==0 else None) for i in range(outer_count)]
        stop=threading.Event();samples=[]
        sampler=threading.Thread(target=sample_gpu,args=(case_root,stop,samples),daemon=True)
        sampler.start()
        try:
            worker_reference="latentmol.design.boltz_speed:"+("proxy_worker" if backend=="vjp_pool" else "worker")
            result=run_spawn_group(worker_reference,jobs,case_root,
                startup_timeout=config.get("startup_timeout",1800),run_timeout=config.get("case_timeout",900))
            workers=group_workers(result)
            device=group_device(result)
            if baseline is not None and device.get("uuid") and group_device(baseline).get("uuid")!=device["uuid"]:
                raise ValueError("GPU allocation changed between matched cases")
            # Only matched optimizer intervals enter throughput; load, barrier
            # probes and final frozen hashes are reported separately.
            elapsed=max(row["measured_ended"] for row in workers)-min(row["measured_started"] for row in workers)
            measured_samples=[row for row in samples if min(w["measured_started"] for w in workers)<=row["timestamp"]<=max(w["measured_ended"] for w in workers)]
            entry=dict(case=case_name,status="completed",parallel=count,backend=backend,execution=case["execution"],device=device,
                total_updates=total_children*updates,measured_seconds=elapsed,
                updates_per_second=total_children*updates/elapsed,
                seconds_per_aggregate_update=elapsed/(total_children*updates),
                peak_worker_allocated_bytes=max(row["peak_allocated_bytes"] for row in workers),
                sum_worker_peak_reserved_bytes=sum(row["peak_reserved_bytes"] for row in workers),
                nvml_timed_samples=len(measured_samples),
                nvml_peak_used_mib=max((row["used_mib"] for row in measured_samples),default=None),
                nvml_mean_gpu_busy_percent=statistics.mean(row["gpu_busy_percent"] for row in measured_samples) if measured_samples else None,
                end_to_end_group_seconds=result.get("group_wall_seconds"))
            if baseline is None:
                if count!=1 or case["execution"].get("tf32",False) or case["execution"].get("autocast","none")!="none":
                    raise ValueError("First case must be serial FP32 reference")
                baseline=result
            if count==1:
                memory_profiles[policy]=dict(reserved=workers[0]["peak_reserved_bytes"],total=workers[0]["device"]["total_memory_bytes"],
                    updates_per_second=entry["updates_per_second"])
            if policy in memory_profiles:
                entry["parallel_speedup_same_execution"]=entry["updates_per_second"]/memory_profiles[policy]["updates_per_second"]
            entry["relative_throughput"] = entry["updates_per_second"]/records[0]["updates_per_second"] if records else 1.
            entry["comparisons"]=compare_children(baseline,result)
            records.append(entry)
        except ParallelWorkerError as error:
            records.append(dict(case=case_name,status="failed",error=str(error),group_report=error.report))
            worker_roots=[path for path in case_root.glob("worker-*") if path.is_dir()]
            cleanup_ok=bool(worker_roots) and all((path/"cleanup.json").exists()
                and json.loads((path/"cleanup.json").read_text()).get("cuda_synchronized")
                and (not json.loads((path/"cleanup.json").read_text()).get("scorer_created")
                     or json.loads((path/"cleanup.json").read_text()).get("frozen_checked")) for path in worker_roots)
            write(output/f"{case_name}-failure.json",records[-1])
            if baseline is None:
                raise RuntimeError("Serial FP32 reference failed; no valid speed comparison is available") from error
            if error.report.get("requires_job_stop") or not cleanup_ok:
                raise RuntimeError("Worker cleanup/MPS health is unresolved; stop this allocation") from error
        finally:
            stop.set();sampler.join(timeout=5)
            write(output/f"{case_name}-gpu-samples.json",samples)
        write(output/f"summary-{len(records):02d}.json",dict(task=task_name,cases=records))
        print(json.dumps(records[-1]),flush=True)
    write(output/"result.json",dict(status="completed",task=task_name,cases=records,
        scope="Matched short child throughput and initial value/gradient comparisons; no long-search quality claim"))


def precision_budget(config, task_name):
    """The paired quality test counts updates, endpoint gradients and a union cap."""
    task = config["tasks"][task_name]
    order = task.get("case_order", ["fp32", "bf16"])
    updates = config.get("updates", 200)
    checkpoints = config.get("checkpoints", [0, 50, 100, 200])
    seed = task.get("child_seed", config.get("child_seed"))
    if (config.get("children", 1) != 1 or type(updates) is not int or updates != 200
            or not isinstance(order, list) or len(order) != 2 or len(set(order)) != 2
            or "fp32" not in order or set(order) - config["cases"].keys()
            or type(seed) is not int or not 0 <= seed < 2**63
            or checkpoints != sorted(set(checkpoints)) or not {0, updates}.issubset(checkpoints)
            or any(type(step) is not int or not 0 <= step <= updates for step in checkpoints)):
        raise ValueError("Precision study requires one matched 200-update pair, explicit seed and ordered checkpoints")
    for name in order:
        if config["cases"][name].get("parallel", 1) != 1:
            raise ValueError("Precision arms run serially in separate native processes")
    fp32 = config["cases"]["fp32"]["execution"]
    if fp32.get("tf32", False) or fp32.get("autocast", "none") != "none":
        raise ValueError("Common hard scoring requires a full FP32 reference case")
    return dict(updates=2*updates, fresh_attempts=2*(updates+1), relaxed_observations=2*(updates+1),
                case_order=order, child_seed=seed, checkpoints=checkpoints)


def _precision_design(config, inherited):
    defaults = dict(updates=200, schedule_horizon=200, learning_rate=1., max_step_norm=.25,
        max_step_norm_final=None, chemistry_initial=1., chemistry_final=1., chemistry_hold=150,
        chemistry_ramp_end=200, noise_initial=0., noise_zero_at=200,
        temperature_initial=1., temperature_final=1.)
    design = inherited | defaults | config.get("design", {})
    fixed = ("updates", "schedule_horizon", "learning_rate", "max_step_norm", "max_step_norm_final",
             "chemistry_initial", "chemistry_final", "noise_initial", "temperature_initial", "temperature_final")
    if any(design[key] != defaults[key] for key in fixed):
        raise ValueError("Paired precision test uses fixed LR1, cap0.25, chemistry1, temperature1 and no update noise")
    return design


def _point_digest(latent, active):
    import torch

    digest = hashlib.sha256()
    for value in (latent, active):
        value = value.detach().cpu().contiguous()
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def collect_precision_states(branch_root, spec, active, updates):
    """Reconstruct every raw state, including alternate poses of the same SMILES.

    BranchResult.valid deliberately retains only one point per molecule. Reading
    the complete raw blocks here prevents each arm's own low-precision relaxed
    loss from deciding which hard graphs receive the common FP32 evaluation.
    """
    import numpy as np
    from latentmol.chemistry import inspect_graph
    from latentmol.guidance.boltz_search import hard_pose_key

    branch_root = Path(branch_root)
    metrics = [json.loads(line) for line in (branch_root/"metrics.jsonl").read_text().splitlines()]
    if [row["step"] for row in metrics] != list(range(updates+1)):
        raise ValueError("Precision branch did not record exactly updates+1 observations")
    states = {}
    for path in sorted((branch_root/"raw-blocks").glob("states-*.npz")):
        with np.load(path, allow_pickle=False) as block:
            for index, step in enumerate(block["steps"].tolist()):
                if step in states:
                    raise ValueError("Duplicate raw precision state writer")
                atoms, charges = block["atoms"][index], block["charges"][index]
                bonds = np.zeros((spec.max_nodes, spec.max_nodes), dtype=np.int64)
                left, right = block["upper_left"], block["upper_right"]
                bonds[left, right] = bonds[right, left] = block["bonds_upper"][index]
                if (atoms.shape != (spec.max_nodes,) or charges.shape != atoms.shape
                        or not np.array_equal(atoms != 0, active.detach().cpu().numpy())
                        or not np.isfinite(block["latent"][index]).all()):
                    raise ValueError("Raw precision state shape, occupancy or latent differs")
                states[step] = dict(hard_graph=dict(atoms=atoms.tolist(), charges=charges.tolist(), bonds=bonds.tolist()),
                                    latent=block["latent"][index].copy())
    if sorted(states) != list(range(updates+1)):
        raise ValueError("Incomplete raw precision trajectory")
    observations, candidates, elapsed = [], {}, 0.
    for metric in metrics:
        step = metric["step"]
        graph = states[step]["hard_graph"]
        checked = inspect_graph(graph["atoms"], graph["charges"], graph["bonds"], spec)
        if checked.smiles != metric["smiles"] or (checked.smiles is not None) != metric["raw_valid"]:
            raise ValueError("Raw graph and recorded chemical validity disagree")
        seconds = metric["timing"]["iteration_seconds"]
        if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(metric["relaxed_loss"]):
            raise ValueError("Nonfinite precision timing or relaxed score")
        elapsed += seconds
        key = hard_pose_key(graph)
        observations.append(dict(step=step, elapsed_seconds=elapsed, raw_valid=checked.smiles is not None,
            smiles=checked.smiles, hard_graph_key=key, scoring_key=key if checked.smiles else None,
            relaxed_loss=metric["relaxed_loss"], chemistry_loss=metric["chemistry_loss"]))
        if checked.smiles is not None:
            if key not in candidates:
                candidates[key] = dict(smiles=checked.smiles, hard_graph=graph, scoring_key=key,
                    first_step=step, first_elapsed_seconds=elapsed, observed_steps=[])
            candidates[key]["observed_steps"].append(step)
    return dict(observations=observations, candidates=list(candidates.values()),
        raw_valid_states=sum(row["raw_valid"] for row in observations),
        distinct_valid_molecules=len({row["smiles"] for row in candidates.values()}),
        distinct_valid_pose_graphs=len(candidates), initial_latent=states[0]["latent"],
        terminal_latent=states[updates]["latent"])


def run_precision_child(decoder, scorer, parent, candidate, design, seed, output, *, ready=None):
    """Use the shared branch engine and its 201 all-state observations once."""
    import torch
    from latentmol.chemistry import inspect_graph
    from latentmol.design.latent_protenix import hard_from_graph
    from latentmol.design.protenix_restart_proposals import propose_start
    from latentmol.design.protenix_restarts import Candidate
    from latentmol.design.protenix_search_branches import Branch, execute_branches
    from latentmol.design.protenix_search_frontier import SearchPoint
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key

    latent, offset, proposal = propose_start(decoder, candidate, parent["active"], policy="latent", seed=seed,
        temperature=design.temperature_initial)
    if offset is not None:
        raise ValueError("Precision child initialization must be latent-only, without fading graph offsets")
    with torch.no_grad():
        hard = hard_from_graph(graph_from_decoder(decoder.decode(latent), decoder.spec,
            fixed_active=parent["active"], temperature=design.temperature_initial))
    checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
    initial = Candidate(checked.smiles, latent.detach().clone(), hard, 0., "paired-initialization", 0)
    point = SearchPoint(initial, parent["active"], None, ("paired-initialization",))
    branch = Branch("child", point, seed, policy="continuous", noise_seed=seed)
    started = time.perf_counter()
    results = execute_branches(decoder, scorer, design, output, [branch], batch_size=1, cpu_workers=1,
        overlap=False, repair_limit=0, invalid_limit=0, ready=ready, profile_steps=True)
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    measured = time.perf_counter()-started
    if len(results) != 1 or results[0].summary["updates"] != design.updates:
        raise ValueError("Precision arm completed an unexpected update budget")
    raw = collect_precision_states(Path(output)/"child", decoder.spec, parent["active"], design.updates)
    replay = torch.as_tensor(raw.pop("initial_latent"))[None].to(latent)
    if not torch.equal(replay, latent):
        raise ValueError("Recorded precision initial latent differs from the matched proposal")
    terminal_latent = raw.pop("terminal_latent").tolist()
    initial_graph = hard_graph_record(hard)
    if raw["observations"][0]["hard_graph_key"] != hard_pose_key(initial_graph):
        raise ValueError("Recorded precision initial graph differs from the matched proposal")
    return dict(raw, initial_point_sha256=_point_digest(latent, parent["active"]),
        initial_latent_sha256=hashlib.sha256(latent.detach().contiguous().numpy().tobytes()).hexdigest(),
        active_sha256=hashlib.sha256(parent["active"].detach().contiguous().numpy().tobytes()).hexdigest(),
        initial_hard_graph=initial_graph, initial_hard_key=hard_pose_key(initial_graph), initial_smiles=checked.smiles,
        parent_smiles=candidate.smiles, parent_state_point_sha256=_point_digest(candidate.latent, parent["active"]),
        child_seed=seed, proposal=proposal, updates=design.updates, relaxed_observations=design.updates+1,
        measured_seconds=measured, seconds_per_update=measured/design.updates, terminal_latent=terminal_latent)


class _PrecisionAbortScorer:
    def __init__(self, scorer, context):
        self.scorer, self.context = scorer, context

    def score_at(self, graph, active):
        self.context.check_abort()
        return self.scorer.score_at(graph, active)

    def __getattr__(self, name):
        return getattr(self.scorer, name)


def _seed_precision_worker(seed):
    """Match native RNG streams as well as the proposal's local generator."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # Seeds CPU and visible CUDA devices, lazily if needed.


def precision_worker(payload, context):
    """A fresh native process for exactly one precision arm; no extra warmups."""
    import torch
    from latentmol.design.latent_protenix import _model_identity
    from latentmol.gpu_process_probe import cuda_operation
    from latentmol.guidance.boltz_search import BoltzSearchScorer

    _seed_precision_worker(payload["child_seed"])
    decoder, decoder_identity, parent, candidate, design, scorer_config = load_inputs(payload)
    scorer = None
    cleanup = dict(cuda_synchronized=False, frozen_checked=False, scorer_created=False)
    try:
        retained, device = cuda_operation()
        device["total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        scorer = BoltzSearchScorer.load(scorer_config, decoder.spec, device="cuda",
            source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"])
        torch.cuda.synchronize()
        context.ready(dict(device=device, execution=scorer.execution_record, readiness="Models loaded; first finite attached update follows barrier"))
        result = run_precision_child(decoder, _PrecisionAbortScorer(scorer, context), parent, candidate, design,
            payload["child_seed"], context.output/"trajectory", ready=payload.get("ready_path"))
        result.update(device=device, execution=scorer.execution_record, native_identity=scorer.identity,
            native_rng_seed=payload["child_seed"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(context.output/"measurement.json", result)
        return result
    finally:
        cleanup["scorer_created"] = scorer is not None
        try:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            cleanup["cuda_synchronized"] = True
            if scorer is not None:
                scorer.close()
                cleanup["frozen_checked"] = True
            if _model_identity(decoder) != decoder_identity or any(p.grad is not None for p in decoder.parameters()):
                raise ValueError("Frozen VAE changed during precision comparison")
            cleanup["decoder_frozen_checked"] = True
        finally:
            write(context.output/"cleanup.json", cleanup)


def precision_scoring_worker(payload, context):
    """Evaluate the shared exact pose-graph union once under the FP32 case."""
    import torch
    from latentmol.design.latent_protenix import _model_identity
    from latentmol.gpu_process_probe import cuda_operation
    from latentmol.guidance.boltz_search import BoltzSearchScorer

    _seed_precision_worker(payload["child_seed"])
    decoder, identity, parent, candidate, design, scorer_config = load_inputs(payload)
    scorer = None
    cleanup = dict(cuda_synchronized=False, frozen_checked=False, scorer_created=False)
    try:
        retained, device = cuda_operation()
        scorer = BoltzSearchScorer.load(scorer_config, decoder.spec, device="cuda",
            source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"])
        context.ready(dict(device=device, execution=scorer.execution_record, scope="Common FP32 detached hard scoring"))
        context.check_abort()
        root = context.output/"ranking"
        root.mkdir()
        started = time.perf_counter()
        def cancellable_candidates():
            for row in payload["candidates"]:
                context.check_abort()
                yield row
        rows = scorer.score_valid_records(root, cancellable_candidates())
        torch.cuda.synchronize()
        result = dict(candidates=rows, unique_pose_graphs=len(rows), elapsed_seconds=time.perf_counter()-started,
                      device=device, identity=scorer.identity, execution=scorer.execution_record)
        write(context.output/"measurement.json", result)
        return result
    finally:
        cleanup["scorer_created"] = scorer is not None
        try:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            cleanup["cuda_synchronized"] = True
            if scorer is not None:
                scorer.close()
                cleanup["frozen_checked"] = True
            if _model_identity(decoder) != identity or any(p.grad is not None for p in decoder.parameters()):
                raise ValueError("Frozen VAE changed during common FP32 scoring")
            cleanup["decoder_frozen_checked"] = True
        finally:
            write(context.output/"cleanup.json", cleanup)


def precision_union(arms):
    """Exact ordered-pose union; each arm retains its own discovery times."""
    if len(arms) != 2:
        raise ValueError("Precision comparison requires exactly two arms")
    first, second = list(arms.values())
    keys = ("initial_point_sha256", "initial_latent_sha256", "active_sha256", "initial_hard_key",
            "initial_smiles", "child_seed", "parent_state_point_sha256")
    if any(first[key] != second[key] for key in keys):
        raise ValueError("Precision arms did not use the same saved parent, initialization and seed")
    union = {}
    for name, arm in arms.items():
        for row in arm["candidates"]:
            key = row["scoring_key"]
            if key in union and union[key]["smiles"] != row["smiles"]:
                raise ValueError("One exact pose graph has inconsistent chemical identities")
            union.setdefault(key, dict(smiles=row["smiles"], scoring_key=key, hard_graph=row["hard_graph"],
                                      start="paired-precision", arm="common-fp32", role="gradient_trajectory", step=row["first_step"]))
    return list(union.values())


def summarize_precision_pair(arms, scored, checkpoints):
    """Rank only each arm's available candidates, all with common FP32 scores."""
    union = precision_union(arms)
    expected = {row["scoring_key"] for row in union}
    scores = {}
    for row in scored:
        if row["scoring_key"] in scores or row.get("loss") is None or not math.isfinite(row["loss"]):
            raise ValueError("Common FP32 scoring returned duplicate/missing/nonfinite scores")
        scores[row["scoring_key"]] = row
    if scores.keys() != expected:
        raise ValueError("Common FP32 scorer returned a different exact graph union")

    def rank(arm, observations):
        available = {row["scoring_key"] for row in observations if row["raw_valid"]}
        by_smiles = {}
        for key in sorted(available):
            row = scores[key]
            candidate = dict(smiles=row["smiles"], loss=row["loss"], scoring_key=key)
            current = by_smiles.get(row["smiles"])
            if current is None or (candidate["loss"], key) < (current["loss"], current["scoring_key"]):
                by_smiles[row["smiles"]] = candidate
        ranked = sorted(by_smiles.values(), key=lambda row: (row["loss"], row["smiles"], row["scoring_key"]))
        return dict(best_loss=ranked[0]["loss"] if ranked else None, best=ranked[0] if ranked else None,
            mean_top_three_loss=statistics.mean(row["loss"] for row in ranked[:3]) if len(ranked) >= 3 else None,
            mean_top_twenty_loss=statistics.mean(row["loss"] for row in ranked[:20]) if len(ranked) >= 20 else None,
            top_three_count=min(3, len(ranked)), top_twenty_count=min(20, len(ranked)), molecules=ranked[:20],
            distinct_valid_molecules=len(ranked), distinct_valid_pose_graphs=len(available),
            raw_valid_states=sum(row["raw_valid"] for row in observations), observations=len(observations),
            last_observed_update=max((row["step"] for row in observations), default=None))

    horizon = min(arm["observations"][-1]["elapsed_seconds"] for arm in arms.values())
    results = {}
    sets = []
    for name, arm in arms.items():
        sets.append({row["scoring_key"] for row in arm["candidates"]})
        by_update = [dict(updates=step, **rank(arm, [row for row in arm["observations"] if row["step"] <= step]))
                     for step in checkpoints]
        terminal = arm["observations"][-1]
        initial = arm["observations"][0]
        results[name] = dict(by_update=by_update, final=by_update[-1],
            terminal=dict(step=terminal["step"], raw_valid=terminal["raw_valid"], smiles=terminal["smiles"],
                          scoring_key=terminal["scoring_key"], loss=scores[terminal["scoring_key"]]["loss"] if terminal["raw_valid"] else None),
            initial_fp32_loss=scores[initial["scoring_key"]]["loss"] if initial["raw_valid"] else None,
            fixed_time_proxy=dict(seconds=horizon, **rank(arm, [row for row in arm["observations"] if row["elapsed_seconds"] <= horizon])),
            measured_seconds=arm["measured_seconds"], seconds_per_update=arm["seconds_per_update"],
            execution=arm["execution"], initial_point_sha256=arm["initial_point_sha256"], child_seed=arm["child_seed"])
    return dict(arms=results, common_fp32_pose_graphs=len(union), intersection_pose_graphs=len(sets[0] & sets[1]),
        own_pose_graphs={name: len(keys) for name, keys in zip(arms, sets)},
        cross_arm_duplicate_calls_saved=sum(map(len, sets))-len(union),
        scoring_budget_scope="One shared exact pose-graph union; actual own candidate counts/costs can differ between arms",
        fixed_time_scope="Saved trajectory observation prefixes at the shorter summed iteration clock; excludes model loading, common FP32 ranking and final hashes; a proxy, not a new fixed-wall run")


def run_precision(config, task_name, runtime):
    from latentmol.guidance.boltz_parallel import run_spawn_group

    budget = precision_budget(config, task_name)
    search = json.loads((runtime.config.parent/config["search_config"]).read_text())
    target = search["arms"][config["target_arm"]]
    design = _precision_design(config, search["design"] | target.get("design", {}))
    output = runtime.output
    output.mkdir(parents=True)
    write(output/"run.json", dict(config=config, task=task_name, budget=budget, design=design,
        source_sha256=os.environ.get("LATENTMOL_SOURCE_SHA256"), arm_order=budget["case_order"]))
    base = dict(children=[0], updates=budget["updates"]//2, cpu_threads=config["cpu_threads"],
        checkpoint=str(runtime.checkpoint), checkpoint_sha256=runtime.checkpoint_sha256,
        parent_state=config["parent_state"], design=design, prepared_config=target["prepared_config"], child_seed=budget["child_seed"])
    arms, groups = {}, []
    for index, name in enumerate(budget["case_order"]):
        payload = base | dict(execution=config["cases"][name]["execution"],
            ready_path=os.environ.get("DBMOL_LAUNCH_READY") if index == 0 else None)
        group = run_spawn_group("latentmol.design.boltz_speed:precision_worker", [payload], output/name,
            startup_timeout=config.get("startup_timeout", 1800), run_timeout=config.get("case_timeout", 7200))
        arms[name] = group_workers(group)[0]
        if groups and group_device(groups[0]).get("uuid") != group_device(group).get("uuid"):
            raise ValueError("GPU allocation changed between precision arms")
        groups.append(group)
        write(output/f"summary-{index+1:02d}.json", dict(status="partial", completed_arms=list(arms)))
    union = precision_union(arms)
    if len(union) > budget["fresh_attempts"]:
        raise ValueError("Common FP32 candidate union exceeds the all-state scoring cap")
    scoring = dict(candidates=[], unique_pose_graphs=0, elapsed_seconds=0., status="not_needed_no_valid_candidates")
    if union:
        payload = base | dict(execution=config["cases"]["fp32"]["execution"], candidates=union)
        group = run_spawn_group("latentmol.design.boltz_speed:precision_scoring_worker", [payload], output/"common-fp32",
            startup_timeout=config.get("startup_timeout", 1800), run_timeout=config.get("scoring_timeout", 3600))
        if group_device(groups[0]).get("uuid") != group_device(group).get("uuid"):
            raise ValueError("GPU allocation changed before common FP32 scoring")
        scoring = group_workers(group)[0]
        groups.append(group)
    summary = summarize_precision_pair(arms, scoring["candidates"], budget["checkpoints"])
    result = dict(status="completed", study="boltz_precision", task=task_name, child_seed=budget["child_seed"],
        budget=budget, actual_updates=sum(arm["updates"] for arm in arms.values()),
        actual_relaxed_observations=sum(arm["relaxed_observations"] for arm in arms.values()),
        actual_common_fp32_calls=len(union), common_scoring_seconds=scoring["elapsed_seconds"],
        group_wall_seconds=sum(group.get("group_wall_seconds", 0.) for group in groups), **summary)
    write(output/"result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    from latentmol.experiments import file_sha256
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","checkpoint","output","plan","boltz-assets"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("arm","checkpoint-sha256","plan-sha256"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    if not os.environ.get("LSB_JOBID","").isdigit():raise RuntimeError("Speed benchmark requires an allocated GPU job")
    if file_sha256(args.plan)!=args.plan_sha256 or file_sha256(args.checkpoint)!=args.checkpoint_sha256:
        raise ValueError("Benchmark plan/checkpoint identity differs")
    config=json.loads(args.config.read_text());plan=json.loads(args.plan.read_text())
    task=config["tasks"][args.arm]
    precision = config.get("study") == "boltz_precision"
    budget=precision_budget(config,args.arm) if precision else dict(updates=len(task["cases"])*config["children"]*config["updates"])
    if (file_sha256(args.config)!=plan["config_sha256"] or budget["updates"]!=plan["planned_updates"][args.arm]
            or str(args.output)!=plan["arms"][args.arm]):raise ValueError("Benchmark budget or config differs from staged plan")
    if precision and budget["fresh_attempts"] != plan["planned_fresh_attempts"][args.arm]:
        raise ValueError("Precision common FP32 scoring cap differs from the staged plan")
    if precision and "evaluation_execution" in plan:
        from latentmol.guidance.boltz_search import resolve_execution
        if resolve_execution(plan["evaluation_execution"]) != resolve_execution(config["cases"]["fp32"]["execution"]):
            raise ValueError("Precision common FP32 evaluation policy differs from the staged plan")
    (run_precision if precision else run_benchmark)(config,args.arm,args)


if __name__=="__main__":main()
