"""Paired AE/VAE interpolation audit on the fixed Stage 1 validation paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latentmol.experiments import ExperimentRun, file_sha256, write_json


def _valid(point):
    return bool(point["decoded"]["smiles"])


def _distinct_middle(path):
    first, last = path["points"][0]["decoded"]["smiles"], path["points"][-1]["decoded"]["smiles"]
    return sum(_valid(point) and point["decoded"]["smiles"] not in (first, last)
               for point in path["points"][1:-1])


def _counts(paths):
    return {"paths": len(paths), "endpoint_valid_paths": sum(_valid(p["points"][0])
                and _valid(p["points"][-1]) for p in paths),
            "valid_intermediates": sum(_valid(point) for p in paths for point in p["points"][1:-1]),
            "valid_distinct_intermediates": sum(_distinct_middle(p) for p in paths),
            "all_five_valid_and_some_distinct": sum(all(_valid(point) for point in p["points"])
                and _distinct_middle(p) > 0 for p in paths)}


def _paired_interval(ae, vae, eligible, *, seed=20260928, draws=4000):
    """Cluster by input anchor: each anchor owns one near and one far pair."""
    groups = {}
    for left, (a, v) in enumerate(zip(ae, vae)):
        if eligible is not None and not eligible[left]:
            continue
        key = a["left_index"]
        counts = groups.setdefault(key, [0, 0, 0])
        counts[0] += sum(_valid(point) for point in a["points"][1:-1])
        counts[1] += sum(_valid(point) for point in v["points"][1:-1])
        counts[2] += 3
    if not groups:
        return None
    values = np.asarray(list(groups.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    sums = values[samples].sum(axis=1)
    differences = (sums[:, 1] - sums[:, 0]) / sums[:, 2]
    point = (values[:, 1].sum() - values[:, 0].sum()) / values[:, 2].sum()
    return {"vae_minus_ae": float(point), "anchor_clustered_95_interval":
            np.quantile(differences, [0.025, 0.975]).tolist(),
            "anchor_molecules": len(groups), "intermediate_attempts_per_model": int(values[:, 2].sum())}


def compare(ae_path: Path, vae_path: Path, run_root: Path):
    config = {"ae_interpolations": str(ae_path), "vae_interpolations": str(vae_path),
              "ae_sha256": file_sha256(ae_path), "vae_sha256": file_sha256(vae_path),
              "scope": "Stage1 pilot validation paths, no new model calls or test data"}
    with ExperimentRun(run_root, "ae-vae-interpolation-comparison", config, Path.cwd()) as run:
        manifests = [json.loads((path.parent / "run.json").read_text()) for path in (ae_path, vae_path)]
        for declared, manifest in zip(("ae", "vae"), manifests):
            if manifest.get("status") != "completed" or manifest.get("name") != f"{declared}-latent-stage1":
                raise ValueError("Interpolation artifact lacks its completed, matching model run")
            if manifest.get("config", {}).get("mode") != declared:
                raise ValueError("Interpolation artifact and declared AE/VAE mode differ")
        a_config, v_config = (manifest["config"] for manifest in manifests)
        for key in ("dataset_sha256", "split_sha256", "graph_spec", "cohort_seed", "cohort"):
            if a_config[key] != v_config[key]:
                raise ValueError(f"Paired interpolation protocol differs: {key}")
        run.annotate(parent_runs=[manifest["run_id"] for manifest in manifests],
                     dataset_sha256=a_config["dataset_sha256"],
                     split_sha256=a_config["split_sha256"])
        ae = json.loads(ae_path.read_text())
        vae = json.loads(vae_path.read_text())
        if len(ae) != len(vae) or not ae:
            raise ValueError("Interpolation paths are empty or unequal")
        for a, v in zip(ae, vae):
            identity = ("pair_class", "left_index", "right_index", "left_input", "right_input")
            grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            if (any(a[k] != v[k] for k in identity)
                    or [point["t"] for point in a["points"]] != grid
                    or [point["t"] for point in v["points"]] != grid):
                raise ValueError("AE/VAE interpolation paths do not match")
        both_endpoints_valid = [all(_valid(path["points"][i]) for path in (a, v) for i in (0, -1))
                                for a, v in zip(ae, vae)]
        both_midpoints_valid = [both_endpoints_valid[i] and _valid(a["points"][2])
                                and _valid(v["points"][2]) for i, (a, v) in enumerate(zip(ae, vae))]
        mid_similarity = []
        for i, (a, v) in enumerate(zip(ae, vae)):
            if both_midpoints_valid[i]:
                left = min(a["points"][2]["similarity_to_decoded_endpoints"])
                right = min(v["points"][2]["similarity_to_decoded_endpoints"])
                mid_similarity.append({"pair_class": a["pair_class"], "left_index": a["left_index"],
                                       "ae_min_tanimoto": left, "vae_min_tanimoto": right})
        report = {
            "ae": {label: _counts([p for p in ae if p["pair_class"] == label]) for label in ("near", "far")},
            "vae": {label: _counts([p for p in vae if p["pair_class"] == label]) for label in ("near", "far")},
            "common_endpoint_valid_paths": sum(both_endpoints_valid),
            "common_endpoint_valid_middle_comparison": _paired_interval(ae, vae, both_endpoints_valid),
            "all_path_middle_comparison": _paired_interval(ae, vae, None),
            "common_valid_midpoints": len(mid_similarity),
            "midpoint_two_sided_tanimoto_on_common_valid_midpoints": {
                "ae_mean": float(np.mean([row["ae_min_tanimoto"] for row in mid_similarity]))
                if mid_similarity else None,
                "vae_mean": float(np.mean([row["vae_min_tanimoto"] for row in mid_similarity]))
                if mid_similarity else None},
            "limitation": "Fingerprint overlap and path validity are structural proxies, not continuous property semantics or optimization success. Model-training seed is one; anchor bootstrap conditions on the fixed partner pool and does not cover dependence from reused right endpoints."
        }
        write_json(run.path / "summary.json", report)
        write_json(run.path / "common_midpoint_similarity.json", mid_similarity)
        run.log({"stage": "paired_interpolation_audit", **report})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-interpolations", type=Path, required=True)
    parser.add_argument("--vae-interpolations", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(compare(args.ae_interpolations, args.vae_interpolations, args.run_root))


if __name__ == "__main__":
    main()
