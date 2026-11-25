"""Build the epoch-48 BRD4 tree explorer from the standard snapshots and native run metadata."""
import argparse
import math
from pathlib import Path

from build import assemble, read


def add_founder(tree, supplement):
    """Only join roots whose explicit source matches an actual saved run parent."""
    parents = {p["branch"]: p for p in supplement["run_parents"]}
    attached = set()
    for branch in tree["branches"]:
        if branch["cycle"] != 1:
            continue
        start = supplement["roots"][branch["name"]]
        first = branch["states"][0]
        if first[:3] != [0, start["step0"]["raw_valid"], start["step0"]["smiles"]]:
            raise ValueError(f"First observation disagrees with source trajectory in {branch['name']}")
        source = start["source"]
        if source["branch"] not in parents:
            if start["policy"] != "bootstrap" or parents:
                raise ValueError("Unrecognized founder")
            continue
        parent = parents[source["branch"]]
        if (start["recorded_parent_source"] != {"branch": parent["branch"], "step": parent["step"]}
                or not parent["smiles"] or parent["smiles"] != source["smiles"]
                or branch["parent_smiles"] != parent["smiles"] or parent["step"] != 0):
            raise ValueError("Archive source differs from the recorded founder")
        branch["source"] = dict(cycle=0, branch=parent["branch"], step=0,
                                smiles=parent["smiles"], fresh_loss=parent["fresh_loss"])
        attached.add(parent["branch"])
    for key in sorted(attached):
        parent = parents[key]
        # The saved archive entry has a discrete contact score. It has no relaxed objective.
        tree["branches"].append(dict(name=key, arm="Encoded M02 founder", cycle=0, source=None,
            root_type="archived-parent", states=[[0, True, parent["smiles"], None, None]],
            loss_series=[[0, None, None, None, None]]))


def build_epoch48(data, output):
    playback, trees, extra = (read(data / f"{name}.json") for name in ("snapshot","tree-snapshot","supplement"))
    playback.update(report_title="BRD4 search trees", model_tag="BRD4 · epoch-48 VAE")
    trees["layout_invalid_states"] = True
    by_id = {job["id"]: job for job in playback["jobs"]}
    for tree in trees["jobs"]:
        supplement = extra["jobs"][str(tree["id"])]
        job = by_id[tree["id"]]
        if tree["completed_updates"] != job["result"]["completed_updates"]:
            raise ValueError("Search receipt and tree update totals disagree")
        components = supplement["model_components"]
        if components:
            tree["has_blend_components"] = True
            tree["relaxed_label"] = job["relaxed_label"] = "Relaxed contact–affinity objective"
            if len(components) != len(tree["branches"]):
                raise ValueError("Missing branch model components")
            weights = supplement["calibration"]
            for branch in tree["branches"]:
                source = f"cycle-{branch['cycle']:02d}/{branch['name']}"
                values = components[source]
                for row, (step, contact, affinity) in zip(branch["loss_series"], values, strict=True):
                    mixed = weights["protenix_weight"]*contact + weights["nesso_weight"]*affinity
                    if step != row[0] or not math.isclose(mixed, row[1], rel_tol=2e-5, abs_tol=2e-5):
                        raise ValueError(f"Model components disagree at {source} step {step}")
                    if len(row) == 4:
                        row.append(None)
                    row.extend((contact, affinity))
        add_founder(tree, supplement)
    assemble(playback, trees, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Folder containing snapshot, tree-snapshot and supplement JSON")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_epoch48(args.data, args.output)
