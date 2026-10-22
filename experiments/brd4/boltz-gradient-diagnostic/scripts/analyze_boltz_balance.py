"""Allocated saved-evidence analysis; never rerun a model or chemistry evaluator."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not os.environ.get("LSB_JOBID"):
        raise SystemExit("Saved numerical analysis requires an allocated HPC job")
    import numpy as np

    study, output = map(Path, sys.argv[1:])
    assert digest(study / "summary.json") == (
        "7f49087f12efc762a46691581f80bc42c43395c00b0cc4778a9e04a6679aae7c"
    )
    report_path = study / "case-0/report.json"
    assert digest(report_path) == (
        "13ca896450a8be2e298858226e9f11dac2be9e15f0d1cc092d6e4edf3707dad2"
    )
    case = json.loads(report_path.read_text())
    states, updates = case["states"], case["updates"]
    assert case["status"] == "passed" and len(states) == 10001
    assert [s["step"] for s in states] == list(range(10001))
    assert [u["step"] for u in updates] == list(range(10000))

    def valid(s):
        return s["readout"]["sanitized"] and not s["readout"]["disconnected"]

    def row(s):
        c = s["components"]
        return dict(
            step=s["step"], affinity=s["affinity_loss"],
            binder_probability=c["affinity_probability_binary"],
            repair_weight=s["repair_weight"], chemistry=c["chemistry_loss"],
            weighted_chemistry=s["repair_weight"] * c["chemistry_loss"],
            weighted_composition=0.1 * c["element_js"], valid=valid(s),
            smiles=s["readout"]["canonical_smiles"],
            error=s["readout"]["conversion_error"],
            route=s["chemistry_preparation"][0]["route"],
        )

    best = min(states, key=lambda s: s["affinity_loss"])
    valid_states = [s for s in states if valid(s)]
    selected = sorted(set([0, 1000, 1500, 2000, 5000, 6000, 7000, 8000,
                           8500, 9000, 9500, 9998, 9999, 10000, best["step"]]))
    windows = []
    for lo, hi in [(0, 5000), (5000, 8000), (8000, 9000), (9000, 9500),
                   (9500, 10000)]:
        items = states[lo:hi+1]
        us = updates[lo:hi]
        windows.append(dict(
            lo=lo, hi=hi, n_states=len(items), valid=sum(map(valid, items)),
            validity_switches=sum(valid(a) != valid(b) for a, b in zip(items, items[1:])),
            routes=dict(Counter(s["chemistry_preparation"][0]["route"] for s in items)),
            selected_trials=dict(Counter(u["selected_trial"] for u in us)),
            affinity_increase_updates=sum(u["affinity_after"] > u["affinity_before"] for u in us),
            mean_update_seconds=sum(u["elapsed_seconds"] for u in us) / len(us),
        ))
    snapshots = []
    for step in selected:
        state = states[step]
        path = study / "case-0" / state["snapshot"]
        assert digest(path) == state["snapshot_sha256"]
        with np.load(path, allow_pickle=False) as z:
            native_parts, total_parts = [], []
            for pkey, gkey, tkey in [
                ("X", "affinity_gradient_X", "gradient_atom_logits"),
                ("E_upper", "affinity_gradient_E_upper", "gradient_bond_logits"),
            ]:
                p, g = z[pkey], z[gkey]
                native_parts.append((p * (g - np.sum(p*g, axis=-1, keepdims=True))).ravel())
                total_parts.append(z[tkey].ravel())
            ng, tg = np.concatenate(native_parts), np.concatenate(total_parts)
            residual = tg - ng
            nn, rn = float(np.linalg.norm(ng)), float(np.linalg.norm(residual))
            snapshots.append(dict(
                step=step, native_logit_gradient_norm=nn,
                weighted_chemistry_plus_composition_gradient_norm=rn,
                residual_native_ratio=None if nn == 0 else rn/nn,
                residual_native_cosine=None if nn*rn == 0 else float(np.dot(ng,residual)/(nn*rn)),
                chlorine_extra_edge_1_3=z["E_upper"][1].tolist(),
            ))
    final_reference = json.loads((study / "case-0/references/step-10000.json").read_text())
    result = dict(
        status="recorded", scope="saved trajectory analysis only; no inference or chemistry evaluation",
        job_id=os.environ["LSB_JOBID"], native_job_id="29759212",
        script_sha256=digest(Path(__file__)), case_sha256=digest(report_path),
        points=[row(states[s]) for s in selected], windows=windows,
        best_affinity=row(best), best_valid_affinity=row(min(valid_states, key=lambda s:s["affinity_loss"])),
        last_valid=row(valid_states[-1]), first_invalid=row(next(s for s in states if not valid(s))),
        valid_states=len(valid_states), total_states=len(states),
        native_counts=case["counts"],
        accepted_step_sizes=dict(Counter(str(u["accepted_step_size"]) for u in updates)),
        selected_trials=dict(Counter(u["selected_trial"] for u in updates)),
        snapshots=snapshots,
        final_reference=dict(
            route=final_reference["arrays"]["route"],
            changed=final_reference["arrays"]["changed"],
            target_atoms=final_reference["arrays"]["target_atoms"],
            target_bonds=final_reference["arrays"]["target_bonds"],
        ),
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status":"recorded", "output":str(output), "sha256":digest(output)}))


if __name__ == "__main__":
    main()
