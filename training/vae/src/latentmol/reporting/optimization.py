"""Artifact-only, model-independent molecular optimization reports.

The adapters read JSON/JSONL only. They never unpickle tensors, decode a latent,
sanitize/repair a molecule, execute a scorer, or certify an experiment. Missing
historical distributions remain missing. The offline HTML and its JSON sidecar
are derived presentations; the hashed input records remain authoritative.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


REPORT_VERSION = "molecule-optimization-report-v1"
ELEMENTS = ("NO_ATOM H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
            "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac "
            "Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


class SavedArtifacts:
    """Bound every consumed file to an explicit input root and byte digest."""

    def __init__(self):
        self.sources = {}
        self.roots = {}

    def read(self, path, root, *, lines=False):
        path, root = Path(path).resolve(), Path(root).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Saved artifact escapes its declared root: " + str(path))
        raw = path.read_bytes()
        if len(raw) > 256 * 1024**2:
            raise ValueError("Report input exceeds 256 MiB: " + str(path))
        digest = _sha(raw)
        previous = self.sources.get(str(path))
        if previous is not None and previous["sha256"] != digest:
            raise ValueError("Input changed while assembling report: " + str(path))
        self.sources[str(path)] = {"path": str(path), "sha256": digest,
                                   "bytes": len(raw), "kind": "saved_experiment_artifact"}
        self.roots[str(path)] = root
        text = raw.decode("utf-8-sig")
        if lines:
            return [json.loads(line, parse_constant=_constant) for line in text.splitlines() if line.strip()]
        return json.loads(text, parse_constant=_constant)

    def verify_unchanged(self):
        """Do not combine inputs that changed during this presentation snapshot."""
        for name, receipt in self.sources.items():
            path = Path(name)
            if not path.resolve().is_relative_to(self.roots[name]) or _sha(path.read_bytes()) != receipt["sha256"]:
                raise ValueError("Input changed while assembling report: " + name)


def _spec(value):
    if not isinstance(value, dict):
        raise ValueError("A saved graph requires its explicit GraphSpec")
    for key in ("atomic_numbers", "formal_charges", "bond_names"):
        if not isinstance(value.get(key), list) or not value[key] or len(set(value[key])) != len(value[key]):
            raise ValueError("Invalid saved graph vocabulary: " + key)
    if (type(value.get("max_nodes")) is not int or not 0 < value["max_nodes"] <= 1024
            or value["atomic_numbers"][0] != 0 or value["bond_names"][0] != "NONE"):
        raise ValueError("Invalid saved graph capacity/NO_ATOM/NO_BOND")
    if (any(type(z) is not int or not 0 <= z < len(ELEMENTS) for z in value["atomic_numbers"])
            or any(type(q) is not int for q in value["formal_charges"])
            or any(not isinstance(bond, str) for bond in value["bond_names"])):
        raise ValueError("Saved graph vocabulary labels have unsupported types")
    return value


def _candidate(row):
    """Depict the recorded labels even when their stored chemical check failed."""
    labels, spec = row.get("labels"), row.get("graph_spec")
    graph = None
    if labels is not None:
        spec = _spec(spec)
        n = spec["max_nodes"]
        if (not isinstance(labels, dict) or len(labels.get("atoms", [])) != n
                or len(labels.get("charges", [])) != n or len(labels.get("bonds", [])) != n
                or any(len(line) != n for line in labels["bonds"])):
            raise ValueError("Saved hard graph does not match GraphSpec shape")
        for name, vocab in (("atoms", "atomic_numbers"), ("charges", "formal_charges"), ("bonds", "bond_names")):
            values = labels[name] if name != "bonds" else [v for line in labels[name] for v in line]
            if any(type(v) is not int or not 0 <= v < len(spec[vocab]) for v in values):
                raise ValueError("Saved hard graph has out-of-vocabulary " + name)
        atoms = []
        for slot, label in enumerate(labels["atoms"]):
            z = spec["atomic_numbers"][label]
            if z:
                atoms.append({"slot": slot, "atomic_number": z,
                              "element": ELEMENTS[z] if 0 <= z < len(ELEMENTS) else "Z=" + str(z),
                              "charge": spec["formal_charges"][labels["charges"][slot]]})
        # Full raw arrays remain in the sidecar. The schematic uses unordered
        # pairs only, and never invents bonds from a SMILES or repair target.
        occupied = {atom["slot"] for atom in atoms}
        bonds = [{"i": i, "j": j, "type": spec["bond_names"][labels["bonds"][i][j]]}
                 for i in range(n) for j in range(i + 1, n)
                 if i in occupied and j in occupied and labels["bonds"][i][j] != 0]
        graph = {"atoms": atoms, "bonds": bonds, "graph_spec": spec, "raw_labels": labels}
    smiles, failure = row.get("smiles"), row.get("failure")
    validity = "valid" if smiles and not failure else "invalid" if failure else "unknown"
    return {"id": row.get("candidate_id"), "kind": row.get("kind", "unknown"),
            "draw": row.get("draw"), "validity": validity, "smiles": smiles,
            "failure": failure, "graph": graph, "graph_sha256": row.get("graph_sha256")}


def _matrix(matrix, rows, columns, name):
    if not isinstance(matrix, list) or len(matrix) != rows:
        raise ValueError("Saved probability row count differs: " + name)
    for row in matrix:
        if (not isinstance(row, list) or len(row) != columns
                or any(_number(v) is None or not 0 <= v <= 1 for v in row)
                or abs(sum(row) - 1) > 2e-5):
            raise ValueError("Malformed saved categorical probabilities: " + name)


def validate_probabilities(value):
    """Validate representation, not chemical validity. Partial channels are explicit."""
    spec = _spec(value["graph_spec"])
    slots = value.get("row_slots", list(range(spec["max_nodes"])))
    if (len(set(slots)) != len(slots) or any(type(i) is not int or not 0 <= i < spec["max_nodes"] for i in slots)):
        raise ValueError("Probability rows require unique original slot identities")
    n = len(slots)
    for name, vocab in (("atoms", "atomic_numbers"), ("charges", "formal_charges")):
        if value.get(name) is not None:
            _matrix(value[name], n, len(spec[vocab]), name)
    if value.get("bonds") is not None:
        bonds = value["bonds"]
        if len(bonds) != n:
            raise ValueError("Malformed saved pair probability dimensions")
        for row in bonds:
            _matrix(row, n, len(spec["bond_names"]), "bonds")
        for i, row in enumerate(bonds):
            if abs(row[i][0] - 1) > 2e-6:
                raise ValueError("Saved pair diagonal must be NO_BOND")
            for j in range(n):
                if any(abs(a - b) > 2e-6 for a, b in zip(row[j], bonds[j][i])):
                    raise ValueError("Saved pair probabilities are not symmetric")
    return {**value, "row_slots": slots}


def _saved_probabilities(root, arm, step, row, candidates, artifacts):
    path = arm / f"probabilities-{step:04d}.json"
    if path.is_file():
        value = artifacts.read(path, root)
        if value.get("snapshot_version") != "shared-relaxed-graph-v1" or value.get("step") != step:
            raise ValueError("Probability snapshot does not identify this saved step")
        expected_spec = next((c["graph"]["graph_spec"] for c in candidates if c["graph"]), None)
        if expected_spec is not None and value.get("graph_spec") != expected_spec:
            raise ValueError("Probability snapshot and saved hard candidates use different GraphSpec")
        expected_slots = row.get("selection", {}).get("active_slots")
        if expected_slots is not None and value.get("active_slots") != expected_slots:
            raise ValueError("Probability snapshot and scored state use different occupied slots")
        return validate_probabilities(value)
    # Historical active-set runs already recorded the live conditioned element
    # distribution. Do not re-decode z to manufacture missing charge/bond data.
    elements = row.get("score", row).get("elements", {})
    atoms = elements.get("conditional_probabilities")
    spec = next((c["graph"]["graph_spec"] for c in candidates if c["graph"]), None)
    if atoms is not None and spec is not None:
        if elements.get("atomic_numbers") != spec["atomic_numbers"]:
            raise ValueError("Saved element diagnostics and graph vocabulary disagree")
        return validate_probabilities({"atoms": atoms, "charges": None, "bonds": None,
            "row_slots": elements["active_slots"], "active_slots": elements["active_slots"],
            "graph_spec": spec,
            "scope": "Saved shared-graph atom probabilities conditioned on the occupied mask; charge/bond distributions were not saved."})
    return None


def _compact_score(score):
    result = {k: v for k, v in score.items() if k not in ("elements",)}
    if "elements" in score:
        result["elements"] = {k: v for k, v in score["elements"].items()
                              if "gradients" not in k and "gradient_contrast" not in k}
    return result


def _frame(row, candidates, audit, probabilities, previous):
    score = row.get("score", row)
    repair, components = score.get("repair_guidance", {}), score.get("components", {})
    context, target = row.get("context_sha256", score.get("context_sha256")), repair.get("target_sha256")
    metrics = {"loss": _number(score.get("loss", score.get("weighted_loss"))),
        "binding_loss": _number(score.get("binding_loss", components.get("dbmol_total_loss"))),
        "repair_loss": _number(repair.get("normalized_nll")),
        "repair_weight": _number(repair.get("applied_weight")),
        "gradient_norm": _number(score.get("gradient_norm")),
        "affinity_loss": _number(components.get("dbmol_affinity_loss")),
        "contact_loss": _number(components.get("dbmol_contact_loss")),
        "confidence_loss": _number(components.get("dbmol_confidence_loss")),
        "active_count": _number(row.get("selection", {}).get("count")),
        "expected_atom_count": _number(row.get("expected_atoms")),
        "natural_count": _number(row.get("natural_selection", {}).get("count")),
        "latent_displacement": _number(row.get("latent_displacement"))}
    events = []
    changed = bool(previous and previous["context_sha256"] != context)
    if changed:
        events.append("Native context/mask changed; losses across this boundary need not decrease")
    if previous and previous["repair_target_sha256"] != target:
        events.append("Repair target changed; this objective jump is not an optimization improvement")
    if previous and previous["metrics"]["repair_weight"] != metrics["repair_weight"]:
        events.append("Repair coefficient changed")
    # These describe the update FROM this saved state. Do not infer acceptance
    # from adjacent scores: the next state may have refreshed its target/weight.
    selection = {key: deepcopy(row[key]) for key in
                 ("selection_reason", "strictly_improved", "applied_loss_improvement", "no_op") if key in row}
    if selection.get("selection_reason") == "best_finite_fallback":
        improvement = _number(selection.get("applied_loss_improvement"))
        outcome = (" despite a higher objective" if improvement is not None and improvement < 0 else
                   " at an equal objective" if improvement == 0 else "")
        events.append("Following update: best finite fallback applied" + outcome + ".")
    elif selection.get("selection_reason") == "first_improvement":
        events.append("Following update: first strictly lower trial applied.")
    elif selection.get("selection_reason") == "full_step_nonincrease":
        events.append("Following update: full step applied without objective increase.")
    if selection.get("no_op") is True:
        events.append("Following update: no representable latent movement (no-op).")
    kept = {k: row[k] for k in ("step", "arm", "selection", "natural_selection", "update_status",
            "candidate_count", "valid_candidates", "repair_weight_schedule", "step_control_path",
            "selection_reason", "strictly_improved", "applied_loss_improvement", "no_op") if k in row}
    kept["score"] = _compact_score(score)
    frame = {"step": row["step"], "metrics": metrics, "context_sha256": context,
            "context_changed": changed, "repair_target_sha256": target, "events": events,
            "audit": {"status": audit.get("status", "passed" if audit.get("passed") else "not_saved"),
                      "policy": audit.get("derivative_contract"), "passed": audit.get("passed"),
                      "scope": audit.get("scope")},
            "probabilities": probabilities, "candidates": candidates, "raw_metrics": kept}
    if selection:
        frame["step_selection"] = selection
    return frame


def _trajectory(root, arm, metadata, summary, artifacts):
    by_step = defaultdict(list)
    path = arm / "candidates.jsonl"
    if path.is_file():
        seen = set()
        for item in artifacts.read(path, root, lines=True):
            key = (item["step"], item.get("draw"), item.get("kind"))
            if key in seen:
                raise ValueError("Duplicate saved candidate membership")
            seen.add(key)
            by_step[item["step"]].append(_candidate(item))
    frames, steps = [], set()
    # Long trajectories can append one journal instead of creating tens of
    # thousands of tiny files. Both forms retain the same uniqueness checks.
    rows = [artifacts.read(path, root) for path in sorted(arm.glob("metrics-*.json"))]
    journal = arm / "metrics.jsonl"
    if journal.is_file():
        rows.extend(artifacts.read(journal, root, lines=True))
    for row in rows:
        step = row.get("step")
        if type(step) is not int or step < 0 or step in steps:
            raise ValueError("Saved trajectory steps must be unique nonnegative integers")
        steps.add(step)
        audit_path = arm / f"gate-{step:04d}.json"
        audit = artifacts.read(audit_path, root) if audit_path.is_file() else {}
        probabilities = _saved_probabilities(root, arm, step, row, by_step[step], artifacts)
        frames.append(_frame(row, by_step[step], audit, probabilities, None))
    # An audit or a raw draw may persist before a later scoring/persistence
    # failure. Retain such partial states, with missing scores explicitly null.
    residual_steps = set(by_step)
    for pattern in ("gate-*.json", "probabilities-*.json"):
        for path in arm.glob(pattern):
            suffix = path.stem.rsplit("-", 1)[-1]
            if suffix.isdigit():
                residual_steps.add(int(suffix))
    for step in sorted(residual_steps - steps):
        audit_path = arm / f"gate-{step:04d}.json"
        audit = artifacts.read(audit_path, root) if audit_path.is_file() else {}
        row = {"step": step}
        probabilities = _saved_probabilities(root, arm, step, row, by_step[step], artifacts)
        frame = _frame(row, by_step[step], audit, probabilities, None)
        frame["events"].append("Partial saved state: scored metrics were not saved")
        frame["raw_metrics"]["missing_scored_metrics"] = True
        frames.append(frame)
        steps.add(step)
    frames.sort(key=lambda frame: frame["step"])
    for previous, frame in zip(frames, frames[1:]):
        frame["context_changed"] = previous["context_sha256"] != frame["context_sha256"]
        if frame["context_changed"]:
            frame["events"].append("Native context/mask changed")
        if previous["repair_target_sha256"] != frame["repair_target_sha256"]:
            frame["events"].append("Repair target changed")
        if previous["metrics"]["repair_weight"] != frame["metrics"]["repair_weight"]:
            frame["events"].append("Repair coefficient changed")
    config = metadata.get("config", {})
    details = summary.get("arms", {}).get(arm.name, {})
    own = arm / "arm.json"
    if own.is_file():
        details = artifacts.read(own, root)
    # Preserve only explicitly saved counters and best-state receipts. Older
    # accepted-update counts cannot establish how many steps improved or moved.
    selection_summary = {key: deepcopy(details[key]) for key in
                         ("applied_updates", "improving_updates", "forced_fallback_updates", "no_op_updates",
                          "best_binding_states") if key in details}
    policy = details.get("step_control", config.get("step_control"))
    if isinstance(policy, dict) and isinstance(policy.get("version"), str):
        selection_summary["step_control_version"] = policy["version"]
    best_states = selection_summary.get("best_binding_states")
    if isinstance(best_states, dict):
        for frame in frames:
            for key, saved_best in best_states.items():
                if (isinstance(saved_best, dict) and saved_best.get("step") == frame["step"]
                        and saved_best.get("context_sha256", key) == frame["context_sha256"]
                        and ("active_slots" not in saved_best or saved_best["active_slots"]
                             == frame["raw_metrics"].get("selection", {}).get("active_slots"))):
                    frame["events"].append("Saved best binding state for this mask/pose context; fresh native predictions are separate.")
    argmax = [c for f in frames for c in f["candidates"] if c["kind"] == "argmax"]
    samples = [c for f in frames for c in f["candidates"] if c["kind"] == "categorical"]
    return {"id": arm.name, "label": arm.name.replace("_", " "),
        "status": details.get("optimization_status", details.get("status", metadata.get("status", "unknown"))),
        "summary": {"requested_steps": details.get("requested_steps", details.get("steps", config.get("optimizer", {}).get("steps"))),
            "accepted_updates": details.get("accepted_updates"),
            **selection_summary,
            "audits_passed": sum(f["audit"]["passed"] is True for f in frames),
            "audits_total": sum(f["audit"]["passed"] is not None for f in frames),
            "argmax_valid": sum(c["validity"] == "valid" for c in argmax), "argmax_total": len(argmax),
            "sampled_valid": sum(c["validity"] == "valid" for c in samples), "sampled_total": len(samples),
            "saved_states": len(frames), "saved_steps": [f["step"] for f in frames],
            "gaps": [i for i in range(max(steps) + 1) if i not in steps] if steps else [],
            "scientific_summary": details}, "frames": frames}


def _followups(root, artifacts):
    result = {}
    # Inspect only known sibling followup folders, never a path embedded in an
    # untrusted receipt. This also works on a downloaded evidence capsule.
    if root.parent.name == "optimization":
        case = root.parent.parent
        for name in ("fresh-pose", "repair-audit"):
            values = []
            for path in sorted((case / name).glob("*/summary.json")):
                values.append({"path": str(path), "summary": artifacts.read(path, case)})
            if values:
                result[name] = values
    return result


def _load_run(path, label, artifacts):
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError("--run must name a saved optimization run directory")
    metadata = artifacts.read(root / "run.json", root) if (root / "run.json").is_file() else {}
    summary = artifacts.read(root / "summary.json", root) if (root / "summary.json").is_file() else {}
    folders = [p for p in root.iterdir() if p.is_dir() and
               (any(p.glob("metrics-*.json")) or any(p.glob("gate-*.json"))
                or any(p.glob("probabilities-*.json")) or (p / "candidates.jsonl").is_file()
                or (p / "metrics.jsonl").is_file())]
    if (root / "arms").is_dir():
        folders.extend(p for p in (root / "arms").iterdir() if p.is_dir())
    if not folders:
        # Early failed native gates are a useful report, not a fabricated empty
        # successful trajectory. The raw failure remains in run.json/traceback.
        cfg = metadata.get("config", {})
        if (not cfg.get("prepared") and not (cfg.get("contexts") and cfg.get("decoder"))
                and not (cfg.get("stage") in ("run", "pipeline") and cfg.get("study"))):
            raise ValueError("No supported saved active-set or named-model optimization artifacts: " + str(root))
    config = metadata.get("config", {})
    identity = {k: v for k, v in metadata.items() if k.endswith("sha256") or k.endswith("_mode")
                or k in ("run_id", "gradient_scope", "loss_profile", "initial_selection", "environment")}
    if isinstance(metadata.get("source"), dict):
        identity["source_sha256"] = metadata["source"].get("sha256")
    if (root / "identity.json").is_file():
        identity["model_inputs"] = artifacts.read(root / "identity.json", root)
    initialization = config.get("start", config.get("study", {}).get("start", {}))
    if config.get("initial_latent"):
        initialization = {**initialization, "saved_latent": config["initial_latent"]}
    warnings = []
    trajectories = [_trajectory(root, arm, metadata, summary, artifacts) for arm in sorted(set(folders))]
    if any(frame["probabilities"] is None or any(frame["probabilities"].get(k) is None for k in ("atoms", "charges", "bonds"))
           for arm in trajectories for frame in arm["frames"]):
        warnings.append("Some probability channels were not saved. Missing data are not re-decoded or inferred from hard labels.")
    if any(arm["summary"]["gaps"] for arm in trajectories):
        warnings.append("Saved trajectory has missing step indices. Playback visits saved states only.")
    if metadata.get("status") != "completed":
        warnings.append("Partial, failed or unfinished run; display is not a scientific pass.")
    return {"id": metadata.get("run_id", root.name), "label": label or metadata.get("name", root.name),
        "path": str(root), "status": metadata.get("status", "unknown"), "error": metadata.get("error"),
        "claim_limit": summary.get("claim_limit", "Saved proxy objective only; chemical and fresh-native outcomes are separate."),
        "identity": identity, "config": config, "initialization": initialization,
        "followups": _followups(root, artifacts), "warnings": warnings, "trajectories": trajectories}


def _matching_config(config, allow_initialization, allow_guidance):
    config = deepcopy(config)
    # Display locations and recorded source pins remain in each native receipt,
    # but are not scientific settings. Every other undeclared setting must match.
    for key in ("name", "output_root", "source_sha256", "source_archive_sha256", "scope", "config_path", "config_file_sha256"):
        config.pop(key, None)
    # Declared wall ceilings can differ with DeFoG overhead. The native call
    # ceiling and budget version remain matching requirements, and every raw
    # configuration retains its full operational provenance in the report.
    if isinstance(config.get("execution_budget"), dict):
        config["execution_budget"].pop("wall_seconds", None)
    if allow_initialization:
        config.pop("start", None)
        config.pop("initial_latent", None)
    if allow_guidance and isinstance(config.get("repair_guidance"), dict):
        config["repair_guidance"].pop("weight", None)
        config["repair_guidance"].pop("weight_schedule", None)
    return config


def build_report(paths, *, labels=None, title="Molecular optimization trajectories", require_matched=False,
                 allow_initialization_comparison=False, allow_guidance_comparison=False):
    """Adapt one or more saved runs without approving or rerunning any science."""
    paths = list(paths)
    if not paths or len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("Require distinct saved run directories")
    labels = list(labels) if labels is not None else [None] * len(paths)
    if len(labels) != len(paths):
        raise ValueError("Give exactly one --label per --run, or omit all labels")
    artifacts = SavedArtifacts()
    runs = [_load_run(path, label, artifacts) for path, label in zip(paths, labels)]
    keys = ("decoder_state_sha256", "scorer_state_sha256", "target_sequence_sha256", "graph_spec_sha256", "msa_sha256")
    differences = [key for key in keys if len({json.dumps(r["identity"].get(key), sort_keys=True) for r in runs}) > 1]
    if require_matched:
        if differences or any(r["identity"].get(k) is None for r in runs for k in keys[:4]):
            raise ValueError("Matched overlay requires identical, explicit frozen decoder/scorer/target/schema identities")
        configs = [_matching_config(r["config"], allow_initialization_comparison, allow_guidance_comparison) for r in runs]
        if len({json.dumps(config, sort_keys=True) for config in configs}) > 1:
            raise ValueError("Matched overlay changes undeclared experiment settings, context or sampling budget")
        contexts = [{frame["context_sha256"] for arm in r["trajectories"] for frame in arm["frames"]} for r in runs]
        if any(None in context or not context for context in contexts) or any(context != contexts[0] for context in contexts[1:]):
            raise ValueError("Matched overlay requires matching saved native contexts")
        if any(r["status"] != "completed" for r in runs):
            raise ValueError("Matched overlay requires completed runs; partial runs remain available in descriptive reports")
    warnings = ["Overlaid trajectories are descriptive; this presentation does not establish causal matching, chemical benefit or experimental binding.",
                "Graph drawings preserve saved raw labels, including invalid molecules. Layout is a slot schematic, not a predicted 3D pose.",
                "Losses can jump when the context, repaired target or guidance coefficient changes. Fresh-pose predictions are separate outcomes."]
    if differences:
        warnings.append("Runs have different identities: " + ", ".join(differences))
    if any("pointwise_native_ad" in json.dumps(frame["audit"]["policy"]) for run in runs
           for arm in run["trajectories"] for frame in arm["frames"]):
        warnings.append("Passed pointwise audits certify the recorded local-AD policy; historical whole-network finite-difference failures are not certified by this report.")
    if any(arm["summary"].get("step_control_version") == "best_trial_v2" for run in runs for arm in run["trajectories"]):
        warnings.append("best_trial_v2 can apply a higher-loss or equal-loss fallback, including a no-op. Applied updates are not a count of strict improvements.")
    artifacts.verify_unchanged()
    return {"report_version": REPORT_VERSION, "title": title,
            "created_at": datetime.now(timezone.utc).isoformat(), "warnings": warnings,
            "comparison": {"policy": "matched_identity_checks" if require_matched else "descriptive_overlay",
                           "identity_differences": differences,
                           "declared_initialization_comparison": allow_initialization_comparison,
                           "declared_guidance_comparison": allow_guidance_comparison},
            "sources": list(artifacts.sources.values()), "runs": runs}


def render_report(paths, output, **options):
    """Publish a new offline HTML/JSON/manifest directory atomically.

    No LSB_JOBID requirement: this operation only presents saved JSON. Native
    models, decoding, chemical checks and probability regeneration remain HPC
    work and are deliberately absent. Existing reports are never overwritten.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    data = build_report(paths, **options)
    assets = Path(__file__).with_name("assets")
    template = (assets / "optimization_report.html").read_text(encoding="utf-8")
    css = (assets / "optimization_report.css").read_text(encoding="utf-8")
    js = (assets / "optimization_report.js").read_text(encoding="utf-8")
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    # Inert application/json can still terminate a script tag in HTML parsing.
    embedded = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    for key in ("@@REPORT_DATA@@", "@@REPORT_CSS@@", "@@REPORT_JS@@"):
        if template.count(key) != 1:
            raise ValueError("Report template must contain exactly one " + key)
    html = template.replace("@@REPORT_CSS@@", css).replace("@@REPORT_JS@@", js).replace("@@REPORT_DATA@@", embedded)
    manifest = {"report_version": REPORT_VERSION, "created_at": data["created_at"],
        "execution": "saved JSON/JSONL presentation only; no model/decoder/chemistry execution",
        "inputs": data["sources"], "renderer": {str(p.name): _sha(p.read_bytes()) for p in
            [Path(__file__), *sorted(assets.glob("optimization_report.*"))]},
        "outputs": {"index.html": _sha(html.encode("utf-8")), "report.json": _sha((payload + "\n").encode("utf-8"))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".optimization-report-", dir=output.parent))
    try:
        # Byte writes avoid platform newline translation invalidating the hashes.
        (temporary / "index.html").write_bytes(html.encode("utf-8"))
        (temporary / "report.json").write_bytes((payload + "\n").encode("utf-8"))
        (temporary / "manifest.json").write_bytes((json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        temporary.rename(output)
    except BaseException:
        # Only the newly created staging directory is removed, never input or
        # historical run artifacts; path is from mkdtemp under this output parent.
        if temporary.resolve().parent != output.parent or not temporary.name.startswith(".optimization-report-"):
            raise RuntimeError("Refuse cleanup outside this report's staging parent")
        shutil.rmtree(temporary)
        raise
    return output / "index.html"


def default_report(run):
    """Best-effort presentation after the terminal scientific record is saved.

    A presentation failure gets its own receipt and cannot replace the original
    scientific failure, change the optimizer, or approve any failed gate.
    """
    root = Path(run)
    receipt = {"report_version": REPORT_VERSION, "status": "failed"}
    try:
        path = render_report([root], root / "interactive-report")
        receipt.update(status="generated", path=str(path), manifest="interactive-report/manifest.json")
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    (root / "interactive-report-status.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, action="append", help="Saved active-set or named-model optimization run; repeat to overlay")
    parser.add_argument("--label", action="append", help="One human-readable label per --run")
    parser.add_argument("--output", type=Path, required=True, help="New report directory; never overwrite")
    parser.add_argument("--title", default="Molecular optimization trajectories")
    parser.add_argument("--require-matched", action="store_true", help="Require completed runs with matching saved frozen identities and experiment settings")
    parser.add_argument("--allow-initialization-comparison", action="store_true", help="Declare different starts when requesting matched identity checks")
    parser.add_argument("--allow-guidance-comparison", action="store_true", help="Declare different repair coefficients/schedules; other provider/budget settings must match")
    args = parser.parse_args(argv)
    print(render_report(args.run, args.output, labels=args.label, title=args.title, require_matched=args.require_matched,
        allow_initialization_comparison=args.allow_initialization_comparison,
        allow_guidance_comparison=args.allow_guidance_comparison))


if __name__ == "__main__":
    main()
