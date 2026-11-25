"""Compare native fixed and binder-confidence BRD4 archives at shared search budgets."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


MATCHED_GENERATION = 5
FOUNDER_UPDATES = 700
WAVE_UPDATES = 4 * 3 * 700


def load(path):
    return json.loads(path.read_bytes())


def finite(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite score")
    return value


def median_or_none(values):
    return median(values) if values else None


def native(row):
    n = row["native_nesso"]
    if n["status"] != "scored":
        return None
    affinity = finite(n["affinity_pred_value"])
    probability = finite(n["affinity_probability_binary"])
    if not 0 <= probability <= 1 or type(n["has_interface"]) is not bool:
        raise ValueError("Invalid native binder probability or interface flag")
    return affinity, probability, n["has_interface"]


def summarize_generation(readout, founder):
    rows = {r["smiles"]: r for r in readout["molecules"]}
    if len(rows) != len(readout["molecules"]) or founder not in rows:
        raise ValueError("Expected one row per distinct identity including its founder")
    start = rows[founder]
    # An archive can retain a better-scored pose/slot ordering of the founder identity.
    baseline_contact = finite(start["selected_hard_contact_loss"]) - finite(start["score_change"])
    baseline_nesso = native(start)
    contacts, affinities, probabilities, joint_probs, triple = [], [], [], [], []
    for smi, r in rows.items():
        contact = finite(r["selected_hard_contact_loss"])
        if not math.isclose(contact - baseline_contact, finite(r["score_change"]), abs_tol=1.e-6):
            raise ValueError("Inconsistent founder contact reference")
        if smi == founder:
            continue
        contacts.append(contact)
        scores = native(r)
        if scores is None:
            continue
        affinity, prob, interface = scores
        affinities.append(affinity)
        probabilities.append(prob)
        if (baseline_nesso and interface and baseline_nesso[2] and contact < baseline_contact
                and affinity < baseline_nesso[0]):
            joint_probs.append(prob)
            if prob > baseline_nesso[1]:
                triple.append(dict(smiles=smi, contact_loss=contact, affinity=affinity,
                                   binder_probability=prob, branch=r["branch"], step=r["step"]))
    summary = readout["summary"]
    scored = len(affinities) + int(baseline_nesso is not None)
    if (summary["distinct_new"] != len(contacts) or summary["native_scored"] != scored
            or summary["improved_both"] != len(joint_probs)
            or ("improved_all_three" in summary and summary["improved_all_three"] != len(triple))):
        raise ValueError("Native readout summary disagrees with molecule records")
    return dict(distinct_new=len(contacts), fresh_attempts_including_founder=len(rows),
                fresh_scored_including_founder=scored, fresh_scored_new=len(affinities),
                founder_contact_loss=baseline_contact, founder_native=baseline_nesso,
                improved_both=len(joint_probs), improved_all_three=len(triple),
                median_binder_probability_new=median_or_none(probabilities),
                median_binder_probability_joint_hits=median_or_none(joint_probs),
                best_binder_probability_new=max(probabilities, default=None),
                median_contact_loss_new=median_or_none(contacts), best_contact_loss_new=min(contacts, default=None),
                median_affinity_new=median_or_none(affinities), best_affinity_new=min(affinities, default=None),
                highest_probability_triple_hit=max(triple, key=lambda r:r["binder_probability"], default=None))


def read_arm(root, policy, launch_path):
    launch = load(launch_path) if launch_path and launch_path.is_file() else {}
    failed = (launch.get("status") == "launch_failed" or
              (launch.get("status") == "worker_exited" and launch.get("returncode") != 0))
    final = load(root / "result.json") if (root / "result.json").is_file() else {}
    if final and final["status"] != "completed":
        failed = True
    status = "partial" if root.is_dir() else "missing"
    if failed:
        status = "failed"
    elif final:
        status = "completed"
    report = dict(directory=str(root), status=status, launch=launch or None,
                  final_status=final.get("status"), wall_seconds=final.get("wall_seconds"),
                  completed_generations=[], settings_verified=False)
    generations = {}
    founder_path = root / "founder.json"
    if not founder_path.exists():
        if final.get("status") == "completed":
            raise ValueError(f"Missing founder for completed {policy} search")
        return report, generations
    f = load(founder_path)
    digest = f["start_sha256"]
    if not f["smiles"] or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid founder record")
    report["founder"] = {k: f[k] for k in ("seed", "smiles", "start_sha256")}
    last_gen = MATCHED_GENERATION + int(policy == "confidence")
    if final.get("status") == "completed":
        cfg = final["config"]
        if (final["seed"] != f["seed"] or final["founder_sha256"] != digest or final["policy"] != policy
                or cfg["child_generations"] != last_gen or cfg["founder_updates"] != FOUNDER_UPDATES
                or cfg["parents"] != 4 or cfg["children_per_parent"] != 3 or cfg["child_updates"] != 700
                or cfg["learning_rate"] != 60 or cfg["repair_weights"] != [2.]
                or final["planned_search_updates"] != FOUNDER_UPDATES + last_gen*WAVE_UPDATES):
            raise ValueError(f"Unexpected trial identity or search budget for {policy}")
        report["settings_verified"] = True
    for gen in range(last_gen + 1):
        g = root / "search" / f"generation-{gen:02d}"
        progress, dual = g / "progress.json", g / "dual-readouts.json"
        if not progress.is_file() or not dual.is_file():
            continue
        try:
            state, readout = load(progress), load(dual)
        except json.JSONDecodeError:
            if status == "completed":
                raise
            # During a live snapshot the latest JSON document can still be being written.
            report["unreadable_generation"] = gen
            continue
        if state["generation"] != gen or state["updates"] != FOUNDER_UPDATES + gen*WAVE_UPDATES:
            raise ValueError(f"Unmatched search update budget at {g}")
        gen_summary = summarize_generation(readout, f["smiles"])
        generations[gen] = dict(gen_summary, generation=gen, updates=state["updates"],
                                recorded_elapsed_seconds=state.get("elapsed_seconds"))
    report["completed_generations"] = sorted(generations)
    if status == "completed" and len(generations) != last_gen + 1:
        raise ValueError(f"Completed {policy} search is missing generation records")
    return report, generations


def compare(fixed, confidence, fixed_launch=None, confidence_launch=None):
    left, lgens = read_arm(fixed, "fixed", fixed_launch)
    right, rgens = read_arm(confidence, "confidence", confidence_launch)
    if "founder" in left and "founder" in right and left["founder"] != right["founder"]:
        raise ValueError("Cannot pair different founder probabilities, seed or molecule")
    shared = {}
    for gen in sorted(lgens.keys() & rgens.keys()):
        if gen <= MATCHED_GENERATION:
            shared[str(gen)] = dict(updates=lgens[gen]["updates"], fixed=lgens[gen], confidence=rgens[gen])
    return dict(arms=dict(fixed=left, confidence=right),
                primary_generation=MATCHED_GENERATION, primary_updates=FOUNDER_UPDATES+MATCHED_GENERATION*WAVE_UPDATES,
                primary_available=str(MATCHED_GENERATION) in shared, matched_generations=shared,
                confidence_only_seventh_generation=rgens.get(MATCHED_GENERATION+1),
                definitions=dict(joint="New molecule with lower hard contact loss and lower native affinity than its founder; both have a predicted interface.",
                                 triple="A joint hit whose native binder probability also exceeds its founder's.",
                                 fresh="One native Nesso attempt per distinct archived identity, including the founder. Counts may differ at matched updates.",
                                 time="wall_seconds is whole-job time when final receipt exists. Per-generation elapsed time is included only when explicitly recorded."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "confidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("fixed", "confidence"):
        parser.add_argument("--" + name + "-launch-receipt", type=Path)
    args = parser.parse_args()
    report = compare(args.fixed, args.confidence, args.fixed_launch_receipt, args.confidence_launch_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(report, f, allow_nan=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
