"""BRD4 direct graph searches with measured Protenix and Nesso-1 gradient weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from gradient_blend import blend_gradients, blend_three, calibrate, calibrate_three
from repair_frontier import ParentFrontier
from repair_search import digest_graph, finite_contact, project, run_arm, update
from study import hard_smiles, write


def projected_ratio(graph, protenix_gradient, blend, cfg):
    """Compare model-only movement under the same LR, global step cap and simplex floor."""
    _, reference = update(graph, protenix_gradient, cfg)
    _, proposed = update(graph, blend, cfg)
    baseline = reference["probability_step_norm"]
    return proposed["probability_step_norm"] / baseline if baseline else None


def calibrate_search(protenix, nesso, oracle, founder, cfg, policy="fixed"):
    # All policies probe the same Protenix + repair path and then restart at the founder.
    steps = cfg["calibration_steps"]
    if not steps or steps != sorted(set(steps)) or steps[0] != 0:
        raise ValueError("Calibration steps must increase from the founder")
    rows, components, graphs = [], [], []
    graph = founder
    for step in range(steps[-1] + 1):
        p, pg = finite_contact(protenix, graph)
        if step in steps:
            row = dict(step=step, probabilities_sha256=digest_graph(graph), protenix_contact_loss=p)
            if policy == "confidence":
                (n, na, nb), (binder, ca, cb) = nesso.affinity_confidence_gradients(*graph)
                row.update(nesso_fixed_crop_binder_probability=math.exp(-binder), binder_loss=binder)
                components.append((pg, (na, nb), (ca, cb)))
            else:
                n, ng = finite_contact(nesso, graph)
                components.append((pg, ng))
            row["nesso_fixed_crop_affinity"] = n
            rows.append(row)
            graphs.append(graph)
        if step < steps[-1]:
            repair_loss, da, db, _ = oracle.evaluate(*graph)
            if not math.isfinite(repair_loss):
                raise FloatingPointError("Nonfinite calibration repair loss")
            gradient = tuple(d + cfg["repair_weights"][0] * r for d, r in zip(pg, (da, db)))
            graph, _ = update(graph, gradient, cfg)
    if policy == "confidence":
        shares = cfg["search_versions"][policy]["gradient_shares"]
        coefficients = calibrate_three(components, tuple(shares[k] for k in ("protenix", "affinity", "confidence")))
    else:
        coefficients = calibrate(components)
    for row, graph, grads in zip(rows, graphs, components):
        if policy == "confidence":
            fixed, _ = blend_three(*grads, coefficients)
        else:
            fixed, _ = blend_gradients(*grads, coefficients, "fixed")
        row["projected_model_step_relative_to_protenix"] = projected_ratio(graph, grads[0], fixed, cfg)
    result = dict(coefficients=coefficients, probe_states=rows, probe_path_updates=steps[-1],
                  geometry="global L2 across original 25 atom and 300 bond probability rows, before repair and clipping")
    if policy == "confidence":
        # Check the single-backward production path against the independently measured head gradients.
        aw, cw = coefficients["nesso_weight"], coefficients["confidence_weight"]
        begin = perf_counter()
        a, c, da, db = nesso.weighted_affinity_confidence_gradient(*graphs[-1], aw, cw)
        _, affinity_grad, confidence_grad = components[-1]
        expected = [aw*x + cw*y for x, y in zip(affinity_grad, confidence_grad)]
        actual = (da, db)
        norm = math.sqrt(sum(float(np.vdot(x, x)) for x in expected))
        diff = math.sqrt(sum(float(np.vdot(x-y, x-y)) for x, y in zip(expected, actual)))
        result["weighted_backward_check"] = dict(step=steps[-1], warm_seconds=perf_counter()-begin,
            absolute_gradient_l2_error=diff, relative_gradient_l2_error=diff/norm if norm else None,
            affinity_value_difference=a-rows[-1]["nesso_fixed_crop_affinity"],
            binder_loss_difference=c-rows[-1]["binder_loss"])
    return result


class BindingBackend:
    def __init__(self, protenix, nesso, policy, coefficients, cfg):
        self.protenix = protenix
        self.nesso = nesso
        self.policy = policy
        self.coefficients = coefficients
        self.cfg = cfg
        self.calls = 0

    def gradient(self, atom, bond):
        graph = (atom, bond)
        p, pg = finite_contact(self.protenix, graph)
        diagnostic_step = self.calls % 50 == 0
        binder = None
        if self.policy == "confidence":
            if diagnostic_step:
                (n, na, nb), (binder, ca, cb) = self.nesso.affinity_confidence_gradients(*graph)
                blend, metrics = blend_three(pg, (na, nb), (ca, cb), self.coefficients)
            else:
                # One backward pass computes the exact fixed weighted sum of the two Nesso derivatives.
                metrics = {key: self.coefficients[key] for key in ("protenix_weight", "nesso_weight", "confidence_weight")}
                n, binder, na, nb = self.nesso.weighted_affinity_confidence_gradient(
                    *graph, metrics["nesso_weight"], metrics["confidence_weight"])
                blend = tuple((metrics["protenix_weight"] * d + e).astype(np.float32) for d, e in zip(pg, (na, nb)))
        else:
            n, ng = (None, None) if self.policy == "protenix" else finite_contact(self.nesso, graph)
            blend, metrics = blend_gradients(pg, ng, self.coefficients, self.policy)
        weighted = metrics["protenix_weight"] * p
        if n is not None:
            weighted += metrics["nesso_weight"] * n
        if binder is not None:
            weighted += metrics["confidence_weight"] * binder
            metrics.update(nesso_binder_loss=binder, nesso_fixed_crop_binder_probability=math.exp(-binder),
                           measured_component_gradients=diagnostic_step)
        self.binding_metrics = dict(metrics, policy=self.policy, protenix_contact_loss=p,
                                    nesso_fixed_crop_affinity=n, weighted_model_value=weighted)
        if diagnostic_step:
            self.binding_metrics["projected_model_step_relative_to_protenix"] = projected_ratio(
                graph, pg, blend, self.cfg)
        self.calls += 1
        # Keep the original parent ranking, including its relaxed Protenix tie-break.
        # evaluate_segment records the separate update objective and both components.
        return p, *blend


class FreshReadouts:
    def __init__(self, nesso, founder, directory):
        self.nesso = nesso
        self.founder = founder
        self.cache = {}
        self.path = directory / "fresh-nesso.jsonl"
        self.path.touch(exist_ok=False)
        self.score(founder)
        self.last_summary = None

    def score(self, smiles):
        if smiles not in self.cache:
            result = self.nesso.fresh(smiles)
            if result["status"] == "scored":
                if not math.isfinite(result["affinity_pred_value"]) or not 0 <= result["affinity_probability_binary"] <= 1:
                    raise FloatingPointError("Invalid native affinity or binder probability on a valid molecule")
            self.cache[smiles] = result
            with self.path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(dict(result, smiles=smiles), allow_nan=False) + "\n")
        return self.cache[smiles]

    def record(self, generation, candidates):
        ref = self.cache[self.founder]
        rows = []
        for smiles, contact in candidates.items():
            n = self.score(smiles)
            delta = None
            probability_change = None
            if n["status"] == ref["status"] == "scored":
                delta = n["affinity_pred_value"] - ref["affinity_pred_value"]
                probability_change = n["affinity_probability_binary"] - ref["affinity_probability_binary"]
            joint_hit = (smiles != self.founder and contact["score_change"] < 0 and delta is not None
                         and delta < 0 and ref["has_interface"] and n["has_interface"])
            rows.append(dict(contact, native_nesso=n, nesso_change=delta,
                             binder_probability_change=probability_change, improved_both=joint_hit,
                             improved_all_three=joint_hit and probability_change > 0))
        hits = [r for r in rows if r["improved_both"]]
        triple_hits = [r for r in hits if r["improved_all_three"]]
        self.last_summary = dict(identities=len(rows), distinct_new=len(rows)-1,
                                 native_scored=sum(r["native_nesso"]["status"] == "scored" for r in rows),
                                 improved_both=len(hits),
                                 improved_all_three=len(triple_hits),
                                 highest_binder_among_triple_hits=max(triple_hits, key=lambda r:r["binder_probability_change"], default=None),
                                 best_contact_among_joint_hits=min(hits, key=lambda r:r["selected_hard_contact_loss"], default=None),
                                 best_affinity_among_joint_hits=min(hits, key=lambda r:r["nesso_change"], default=None))
        write(generation / "dual-readouts.json", dict(summary=self.last_summary, molecules=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "plan", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", required=True, choices=("protenix", "fixed", "adaptive", "confidence"))
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    cfg = json.loads(args.config.read_bytes())
    t = plan["tasks"][args.task]
    budget = cfg["founder_updates"] + cfg["child_updates"] * cfg["child_generations"] * cfg["parents"] * cfg["children_per_parent"]
    if (hashlib.sha256(args.plan.read_bytes()).hexdigest() != args.plan_sha256
            or hashlib.sha256(args.config.read_bytes()).hexdigest() != plan["config_sha256"]
            or (t["seed"], t["policy"], t["output"]) != (args.seed, args.policy, str(args.output))
            or t["updates"] != (0 if args.calibration_only else budget)
            or cfg["repair_weights"] != [2.0] or cfg["update_noise"] or cfg["occupied_atoms"] != 25
            or args.seed not in cfg["seeds"] or args.policy not in cfg["search_versions"]
            or not os.getenv("LSB_JOBID", "").isdigit()):
        raise ValueError("Plan, inputs or allocation differ from requested paired search")
    from combined_native import NessoWorker
    from latentmol.design.protenix_vina_parents import VinaParentPanel
    from protenix_native import make_backend
    from repair_signal import RepairOracle
    from stage3_math import make_initial_graph

    start = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    initial = make_initial_graph(args.seed, "sparse", n_atoms=cfg["occupied_atoms"])
    graph = tuple(project(np.asarray(x), cfg["probability_floor"]) for x in (initial.X, initial.E_upper))
    smi = hard_smiles(graph)
    if not smi:
        raise ValueError("Sparse founder is invalid")
    write(args.output / "founder.json", dict(seed=args.seed, smiles=smi, start_sha256=digest_graph(graph)))
    protenix = make_backend(args.seed, args.output)
    assets = Path(os.environ.get("COMBINED_NESSO_ASSETS", "/work3/s234818/latentmol/assets/nesso1-brd4-20261120a"))
    try:
        worker = NessoWorker(args.seed, graph, args.output / "nesso", assets)
        with worker, RepairOracle(python=os.environ["DIRECT_REPAIR_PYTHON"]) as oracle:
            calibration = calibrate_search(protenix, worker, oracle, graph, cfg, args.policy)
            write(args.output / "calibration.json", calibration)
            fresh = FreshReadouts(worker, smi, args.output)
            if marker := os.getenv("DBMOL_LAUNCH_READY"):
                write(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                         nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
            result = dict(seed=args.seed, policy=args.policy, optimization_space=cfg["optimization_space"],
                          config=cfg, founder_sha256=digest_graph(graph), calibration=calibration,
                          native_identity=dict(protenix=protenix.identity, nesso=worker.identity),
                          founder_native_nesso=fresh.cache[smi], planned_search_updates=t["updates"])
            if not args.calibration_only:
                backend = BindingBackend(protenix, worker, args.policy, calibration["coefficients"], cfg)
                frontier = ParentFrontier(protenix, VinaParentPanel(cfg["vina_parent_panel"]))
                result["search"] = run_arm(backend, oracle, frontier, graph, cfg, cfg["repair_weights"][0],
                                           args.seed, args.policy, args.output / "search", score_archive=fresh.record)
                result.update(dual_readouts=fresh.last_summary, docking=frontier.docking.accounting())
    finally:
        protenix_audit = protenix.close()
    result.update(status="completed", native_audit=dict(protenix=protenix_audit, nesso=worker.audit),
                  wall_seconds=perf_counter()-start)
    write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
