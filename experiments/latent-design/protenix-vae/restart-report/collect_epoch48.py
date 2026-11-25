"""Build the epoch-48 overview from the paired readouts, broad lineage audit and ERK2 snapshot."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, rdBase

from latentmol.evaluation.synthesis_accessibility import audited_sa_score


def read(path):
    return json.loads(path.read_bytes())


@lru_cache(maxsize=None)
def sa(smiles):
    return audited_sa_score(Chem.MolFromSmiles(smiles))


def discovery_panels(runs_root, pair, broad):
    result = {}
    for condition in ("control", "blend"):
        root = runs_root / Path(pair[f"{condition}_path"]).name
        run = read(root / "result.json")
        result[condition] = dict(source=root.name, cycles=run["cycles"], elapsed_seconds=run["elapsed_seconds"],
                                 held_out=read(root / "novel-panel.json")["held_out"])
    for condition, summary in broad["runs"].items():
        root = runs_root / Path(summary["run_root"]).name
        top = read(root / "diversity.json")["score_only"]["molecules"]
        result[condition] = dict(source=root.name, top=[dict(smiles=r["smiles"], contact=r["fresh_loss"],
            branch=r["branch"], step=r["step"], lineage=r["ancestry"][0]) for r in top])
    return result


def m02_panel(pair, extras):
    comparison = read(pair / "comparison.json")
    molecules, arms = [], []
    for condition, label in (("control", "Protenix only"), ("blend", "Protenix + Nesso-1")):
        receipt = read(pair / f"{condition}-readout.json")
        detail = extras[condition]
        summary = comparison[condition]
        panel = []
        for index, r in enumerate([receipt["founder"], *receipt["molecules"]]):
            native = r["native_nesso"]
            held = {g: scores[r["smiles"]]["loss"] for g, scores in detail["held_out"].items()
                    if r["smiles"] in scores and scores[r["smiles"]]["status"] == "scored"}
            panel.append(dict(condition=condition, founder=index == 0, smiles=r["smiles"], wave=r["cycle"],
                branch=r["branch"], step=r["step"], contact=r["contact_loss"], contact_delta=r.get("contact_change", 0.),
                affinity=native.get("affinity_pred_value"), affinity_delta=r.get("affinity_change", 0.),
                probability=native.get("affinity_probability_binary"), probability_delta=r.get("probability_change", 0.),
                sa=sa(r["smiles"]), interface=native.get("has_interface"), nesso_status=native["status"],
                joint=r.get("improved_both"), triple=r.get("improved_all_three"), held_out=held))
        new = panel[1:]
        if len(new) != summary["distinct_new"] or receipt["search_job_id"] != summary["job_id"]:
            raise ValueError(f"Paired {condition} readout disagrees with final discovery panel")
        waves = []
        running = None
        for c in detail["cycles"]:
            scored = [r["contact"] for r in new if r["wave"] == c["cycle"]]
            if len(scored) != c["fresh_scored"]:
                raise ValueError("M02 wave counts differ from the underlying search result")
            best = min(scored, default=None)
            if best is not None:
                running = best if running is None else min(best, running)
            waves.append(dict(wave=c["cycle"], updates=c["cumulative_updates"], attempts=c["fresh_attempts"],
                              scored=len(scored), best_contact=best, running_best=running))
        arms.append(dict(condition=condition, label=label, job=summary["job_id"], updates=summary["updates"],
                         attempts=summary["protenix_attempts_new"], scored=len(new), nesso_scored=summary["nesso_scored_new"],
                         contact_hits=sum(r["contact_delta"] < 0 for r in new),
                         affinity_hits=sum(r["affinity_delta"] is not None and r["affinity_delta"] < 0 for r in new),
                         probability_hits=sum(r["probability_delta"] is not None and r["probability_delta"] > 0 for r in new),
                         joint_hits=summary["joint_hits"], triple_hits=summary["triple_hits"], best_contact=running,
                         elapsed_seconds=detail["elapsed_seconds"], founder_contact=panel[0]["contact"], waves=waves))
        molecules.extend(panel)
    return dict(arms=arms, molecules=molecules, shared_novel=comparison["shared_novel_identities"],
                evaluation_job=comparison["evaluation_job_id"], update_budget=comparison["planned_updates"],
                discovery_budget=comparison["planned_discovery_scores"])


def broad_panel(receipt, extras):
    arms = []
    for condition, run in receipt["runs"].items():
        valid_founders = {r["founder_smiles"] for r in run["lineages"].values() if r["founder_smiles"]}
        top = []
        for r in extras[condition]["top"]:
            if r["smiles"] in valid_founders:
                continue
            wave = int(r["branch"].split("/")[0].split("-")[1])
            founder_loss = run["lineages"][r["lineage"]]["founder_discovery_loss"]
            top.append(dict(r, condition=condition, founder=False, wave=wave, sa=sa(r["smiles"]),
                            contact_delta=None if founder_loss is None else r["contact"]-founder_loss))
        if top[0]["smiles"] != run["best"]["smiles"] or top[0]["contact"] != run["best"]["loss"]:
            raise ValueError("Ranked broad winner differs from independent lineage audit")
        waves = []
        for w in run["waves"]:
            previous_attempts = waves[-1]["cumulative_attempts"] if waves else 0
            waves.append(dict(wave=w["wave"], updates=w["cumulative_updates"], scored=w["scored_new_this_wave"],
                              attempts=w["cumulative_attempts"]-previous_attempts, cumulative_attempts=w["cumulative_attempts"],
                              best_contact=w["best_this_wave"], running_best=w["best_so_far"]))
        comparable = run["comparable_to_valid_founder"]
        arms.append(dict(condition=condition, label="Seed "+condition.upper(), job=run["job_id"], status=run["status"],
            updates=waves[-1]["updates"], attempts=waves[-1]["cumulative_attempts"], scored=run["distinct_new_scored"],
            best_contact=run["best"]["loss"], elapsed_seconds=run["elapsed_seconds"], comparable=comparable,
            contact_hits=run["improved_valid_founder"] if comparable else None,
            from_valid_starts=run["from_valid_starts"], from_invalid_starts=run["from_invalid_starts"],
            lineages=run["lineages"], waves=waves, top=top))
    return dict(arms=arms, checked_at=receipt["checked_at"])


def erk2_panel(receipt):
    arms = []
    for r in receipt["jobs"]:
        c = r["arm"].split("-")[-1]
        scored = [dict(s, wave=g["number"], child=child["child"], founder=False, sa=sa(s["smiles"])) for g in r["generations"]
                  for child in g["children"] for s in child["hard_scores"] if s["status"] == "scored"]
        founders = [dict(smiles=f["smiles"], arm=f["branch"], step=0, wave=0, founder=True, loss=f["fresh_loss"],
                         sa=sa(f["smiles"])) for f in r["founders"]]
        arms.append(dict(condition=c, label="Seed "+c.upper(), job=str(r["job_id"]), status=r["status"], queue=r["queue"],
                         waves=r["completed_generations"], updates=r["updates_at_last_observation"],
                         completed_child_updates=r["completed_child_updates"], founder_count=len(r["founders"]),
                         founder_best=r["best_founder_loss"], best_new=r["best_new_loss"], scored=r["distinct_new_fresh"],
                         improved=r["improved_over_best_founder"], observations=scored, founders=founders))
    return dict(checked_at=receipt["captured_at"], metric=receipt["metric"], arms=arms)


def main():
    outputs = Path("outputs")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, default=outputs/"protenix-restart-brd4-e48-m02-nesso-20261124a")
    parser.add_argument("--broad", type=Path, default=outputs/"protenix-restart-brd4-e48-20261124a/broad-six-waves-final.json")
    parser.add_argument("--panels", type=Path, default=outputs/"epoch48-interactive-report/discovery-panels.json")
    parser.add_argument("--runs-root", type=Path, help="Read result, novel-panel and diversity receipts directly from a run mirror.")
    parser.add_argument("--erk2", type=Path, default=outputs/"protenix-restart-report-e48-20261125/erk2-status.json")
    parser.add_argument("--out", type=Path, default=outputs/"epoch48-interactive-report/overview-data.json")
    args = parser.parse_args()
    broad = read(args.broad)
    extras = (discovery_panels(args.runs_root, read(args.paired/"comparison.json"), broad)
              if args.runs_root else read(args.panels))
    result = dict(m02=m02_panel(args.paired, extras), broad=broad_panel(broad, extras),
                  erk2=erk2_panel(read(args.erk2)), metadata=dict(checkpoint_update=623116, epoch=48, geometry_seed=1201,
                      nesso_seed=9312401, confirmation_geometry_seeds=[2203,3209], rdkit_version=rdBase.rdkitVersion,
                      contact="Fresh predicted BRD4 pocket contact loss; lower is better.",
                      affinity="Nesso-1 predicted affinity value; lower is better.",
                      probability="Nesso-1 predicted binder probability; higher is better.",
                      sa="Audited RDKit synthetic accessibility before the upper-tail transform; lower is easier.",
                      sources=[str(p) for p in (args.paired/"comparison.json", args.paired/"blend-readout.json",
                              args.paired/"control-readout.json", args.broad, args.runs_root or args.panels, args.erk2)]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",",":"), allow_nan=False)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
