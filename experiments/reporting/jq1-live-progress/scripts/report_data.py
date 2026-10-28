"""Allocated-CPU saved-only scalar/matrix collector for current JQ1 studies.

No model, RDKit, scoring, repair, interpolation or probability reconstruction is
performed. Every committed scalar record is retained. Molecule frames come only
from authenticated actual retained NPZs, with exact FP32 serialization checks.
Presentation completeness and the original scientific verdict remain separate.
"""

import argparse
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat


VERSION = "jq1-live-saved-report-v1"
ATOMS = ("C", "N", "O", "S", "F", "Cl")
BONDS = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
MATRIX_STRIDE = 20
CHUNK_FRAMES = 100
LAYOUT_VERSION = "raw-bond-layout-v1"
_LAYOUT_CACHE = OrderedDict()
_LAYOUT_CACHE_LIMIT = 512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def load(path):
    return json.loads(Path(path).read_bytes(), parse_constant=_reject_constant)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _numeric_finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite committed scalar")
    elif isinstance(value, dict):
        for item in value.values():
            _numeric_finite(item)
    elif isinstance(value, list):
        for item in value:
            _numeric_finite(item)


def bound_file(binding):
    require(isinstance(binding, dict), "Missing saved-file binding")
    path = Path(binding.get("path", ""))
    require(path.is_absolute() and ".." not in path.parts, "Saved input must use an absolute canonical path")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "Saved input is not a regular original file")
    require(type(binding.get("bytes")) is int and binding["bytes"] >= 0
            and info.st_size == binding["bytes"], "Saved input length changed")
    require(re.fullmatch("[a-f0-9]{64}", binding.get("sha256", "")) is not None
            and sha(path) == binding["sha256"], "Saved input SHA changed")
    return path


def committed_jsonl(binding):
    """Read complete newline-committed rows; disclose an uncommitted suffix."""
    path = bound_file(binding)
    raw = path.read_bytes()
    boundary = raw.rfind(b"\n") + 1
    prefix, suffix = raw[:boundary], raw[boundary:]
    rows = []
    for number, line in enumerate(prefix.splitlines()):
        require(line.strip(), "Blank committed NDJSON line")
        row = json.loads(line, parse_constant=_reject_constant)
        require(isinstance(row, dict), "Committed record is not a JSON object")
        _numeric_finite(row)
        rows.append(row)
    return rows, {
        "path": str(path), "bytes": len(raw), "sha256": binding["sha256"],
        "committed_rows": len(rows), "committed_bytes": len(prefix),
        "committed_sha256": hashlib.sha256(prefix).hexdigest(),
        "uncommitted_suffix_bytes": len(suffix),
        "uncommitted_suffix_sha256": hashlib.sha256(suffix).hexdigest() if suffix else None,
    }


def _saved_path(anchor, relative):
    require(isinstance(relative, str) and "\\" not in relative, "Missing original retained snapshot path")
    part = PurePosixPath(relative)
    require(not part.is_absolute() and ".." not in part.parts, "Retained frame escaped original study")
    anchor = Path(anchor)
    path = anchor.joinpath(*part.parts)
    require(path.resolve().is_relative_to(anchor.resolve()) and not path.is_symlink(), "Retained frame resolves outside its study")
    for parent in path.parents:
        if parent == anchor.parent:
            break
        require(not parent.is_symlink(), "Retained frame parent is an alias")
    require(path.is_file(), "Original retained frame is missing")
    return path


def _losses(state, update=None):
    values = {}
    for key, value in state.items():
        if type(value) in (int, float) and any(term in key for term in ("loss", "surrogate", "primal", "repair_weight")):
            values[key] = value
    for key, value in state.get("components", {}).items():
        if type(value) in (int, float):
            values.setdefault(key, value)
            values["components." + key] = value

    def flatten(value, prefix):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(item, prefix + "." + str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                flatten(item, prefix + "." + str(index))
        elif type(value) in (int, float):
            values[prefix] = value
    if update is not None:
        flatten(update, "update")
    return values


def _scalar_equal(actual, expected, message):
    require(type(expected) in (int, float) and math.isfinite(expected), message + " missing saved scalar")
    require(actual.shape == () and actual.dtype.kind == "f" and float(actual) == expected, message + " scalar differs")


def layout_raw_graph(nodes, edges):
    """Return independent display geometry, reusing only exact raw identities."""
    require(isinstance(nodes, list) and nodes and isinstance(edges, list), "Missing raw graph for display layout")
    require(all(isinstance(node, dict) and type(node.get("id")) is int for node in nodes), "Raw layout node identities differ")
    canonical_nodes = sorted(nodes, key=lambda node: node["id"])
    canonical_edges = sorted(edges, key=lambda edge: json.dumps(edge, sort_keys=True, allow_nan=False))
    key = json.dumps([canonical_nodes, canonical_edges], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if key not in _LAYOUT_CACHE:
        geometry = _layout_raw_graph_uncached(canonical_nodes, canonical_edges)
        _LAYOUT_CACHE[key] = geometry
        while len(_LAYOUT_CACHE) > _LAYOUT_CACHE_LIMIT:
            _LAYOUT_CACHE.popitem(last=False)
    _LAYOUT_CACHE.move_to_end(key)
    cached = _LAYOUT_CACHE[key]
    by_id = {node["id"]: node for node in cached["nodes"]}
    return {"nodes": [{**deepcopy(node), "x": by_id[node["id"]]["x"], "y": by_id[node["id"]]["y"]}
                      for node in nodes], "edges": deepcopy(edges), "layout": deepcopy(cached["layout"])}


def _layout_raw_graph_uncached(nodes, edges):
    """Deterministic display geometry, never a molecular pose or validity test.

    Every raw node/edge is preserved. Paths use a zigzag, simple rings use ring
    geometry, and branched graphs use graph-distance initialization plus bounded
    spring/repulsion relaxation. Disconnected raw components occupy separate
    tiles. NumPy is already required for allocated saved-frame validation.
    """
    import numpy as np

    require(isinstance(nodes, list) and nodes, "Display layout requires actual raw nodes")
    identifiers = [node["id"] for node in nodes]
    require(len(set(identifiers)) == len(identifiers)
            and all(type(node) is int and node >= 0 for node in identifiers), "Raw layout node identities differ")
    adjacency = {node: set() for node in identifiers}
    edge_pairs = set()
    for edge in edges:
        first, second = edge["source"], edge["target"]
        require(first in adjacency and second in adjacency and first != second, "Raw layout edge endpoints differ")
        pair = tuple(sorted((first, second)))
        require(pair not in edge_pairs, "Raw layout contains a duplicate edge")
        edge_pairs.add(pair)
        adjacency[first].add(second)
        adjacency[second].add(first)
    pending = set(identifiers)
    components = []
    while pending:
        queue = [min(pending)]
        found = {queue[0]}
        for current in queue:
            for neighbor in sorted(adjacency[current]):
                if neighbor not in found:
                    found.add(neighbor)
                    queue.append(neighbor)
        pending.difference_update(found)
        components.append(sorted(found))

    tiles = []
    methods = []
    for component in components:
        count = len(component)
        index = {node: number for number, node in enumerate(component)}
        degrees = [len(adjacency[node]) for node in component]
        component_edges = [(index[first], index[second]) for first, second in sorted(edge_pairs)
                           if first in index and second in index]
        if count == 1:
            points = np.zeros((1, 2), dtype=np.float64)
            method = "isolated-node"
        elif len(component_edges) == count - 1 and max(degrees) <= 2:
            endpoints = [node for node in component if len(adjacency[node]) == 1]
            walk = [min(endpoints)]
            while len(walk) < count:
                walk.append(next(node for node in sorted(adjacency[walk[-1]]) if node not in walk))
            points = np.zeros((count, 2), dtype=np.float64)
            for position, node in enumerate(walk):
                points[index[node]] = [position * math.sqrt(3) / 2, (position % 2) * 0.5]
            method = "bond-path-zigzag"
        elif len(component_edges) == count and min(degrees) == max(degrees) == 2:
            walk = [min(component)]
            while len(walk) < count:
                walk.append(next(node for node in sorted(adjacency[walk[-1]]) if node not in walk))
            radius = 1 / (2 * math.sin(math.pi / count))
            points = np.zeros((count, 2), dtype=np.float64)
            for position, node in enumerate(walk):
                angle = 2 * math.pi * position / count
                points[index[node]] = [radius * math.cos(angle), radius * math.sin(angle)]
            method = "bond-ring"
        else:
            distances = np.zeros((count, count), dtype=np.float64)
            for source in component:
                queue, hops = [source], {source: 0}
                for current in queue:
                    for neighbor in sorted(adjacency[current]):
                        if neighbor not in hops:
                            hops[neighbor] = hops[current] + 1
                            queue.append(neighbor)
                for target in component:
                    distances[index[source], index[target]] = hops[target]
            # Graph distances provide a topology-aware start rather than a circle.
            center = np.eye(count) - np.ones((count, count)) / count
            gram = -0.5 * center @ (distances * distances) @ center
            values, vectors = np.linalg.eigh(gram)
            points = vectors[:, -2:] * np.sqrt(np.maximum(values[-2:], 0))
            points = points[:, ::-1]
            for axis in range(2):
                pivot = int(np.argmax(np.abs(points[:, axis])))
                if points[pivot, axis] < 0:
                    points[:, axis] *= -1
            # Fixed node-index perturbations resolve collinear/symmetric starts.
            for number, node in enumerate(component):
                points[number, 0] += 0.025 * math.sin((node + 1) * 2.399963229728653)
                points[number, 1] += 0.025 * math.cos((node + 1) * 2.399963229728653)
            weights = np.zeros_like(distances)
            np.divide(1, distances * distances, out=weights, where=distances > 0)
            linked = np.zeros_like(distances)
            for first, second in component_edges:
                linked[first, second] = linked[second, first] = 1
            for iteration in range(100):
                difference = points[:, None, :] - points[None, :, :]
                length = np.sqrt(np.sum(difference * difference, axis=-1) + 1e-12)
                stress = weights * (distances - length) / length
                springs = 0.45 * linked * (1 - length) / length
                repulsion = 0.10 / (length * length)
                np.fill_diagonal(repulsion, 0)
                forces = np.sum((stress + springs + repulsion)[:, :, None] * difference, axis=1)
                forces /= np.maximum(weights.sum(axis=1), 1)[:, None]
                displacement = 0.14 * forces
                norm = np.sqrt(np.sum(displacement * displacement, axis=1))
                temperature = 0.18 * (1 - iteration / 100) + 0.012
                displacement *= np.minimum(1, temperature / np.maximum(norm, 1e-12))[:, None]
                points += displacement
                points -= points.mean(axis=0)
            # Canonical horizontal principal axis makes narrow layouts readable.
            _, axes = np.linalg.eigh(points.T @ points)
            points = points @ axes[:, ::-1]
            for axis in range(2):
                pivot = int(np.argmax(np.abs(points[:, axis])))
                if points[pivot, axis] < 0:
                    points[:, axis] *= -1
            lengths = [np.linalg.norm(points[first] - points[second]) for first, second in component_edges]
            scale = float(np.median(lengths))
            if scale > 1e-12:
                points /= scale
            method = "graph-distance-spring-repulsion"
        require(np.isfinite(points).all(), "Raw display layout produced nonfinite coordinates")
        low, high = points.min(axis=0), points.max(axis=0)
        width, height = np.maximum(high - low, 0) + 1.4
        tiles.append({"nodes": component, "points": points - low + 0.7,
                      "width": float(width), "height": float(height)})
        methods.append({"nodes": component, "method": method})
    # Shelf packing separates disconnected raw components without joining them.
    order = sorted(range(len(tiles)), key=lambda number: (-len(tiles[number]["nodes"]), tiles[number]["nodes"][0]))
    target_width = max(max(tile["width"] for tile in tiles),
                       math.sqrt(sum(tile["width"] * tile["height"] for tile in tiles) * 1.7))
    x = y = shelf_height = occupied_width = 0.0
    positions = {}
    for number in order:
        tile = tiles[number]
        if x and x + tile["width"] > target_width:
            y += shelf_height
            x, shelf_height = 0.0, 0.0
        for node, point in zip(tile["nodes"], tile["points"]):
            positions[node] = [x + float(point[0]), y + float(point[1])]
        x += tile["width"]
        occupied_width = max(occupied_width, x)
        shelf_height = max(shelf_height, tile["height"])
    occupied_height = y + shelf_height
    output = [{**deepcopy(node), "x": positions[node["id"]][0] / occupied_width,
               "y": positions[node["id"]][1] / occupied_height} for node in nodes]
    require(all(math.isfinite(node[axis]) and 0 <= node[axis] <= 1 for node in output for axis in ("x", "y")),
            "Raw display layout coordinates left the bounded canvas")
    return {"nodes": output, "edges": deepcopy(edges), "layout": {
        "version": LAYOUT_VERSION, "method": "bond-topology-with-separated-components",
        "coordinate_system": "normalized_0_1", "width": occupied_width, "height": occupied_height,
        "deterministic": True, "components": methods, "component_count": len(components),
        "identity_preserved": True, "scope": "Display coordinates only; not a molecular pose, validity judgment, repair or native scoring"}}


def validate_frame(path, state, global_step):
    """Validate stored FP32 arrays and derived argmax, without a forward pass."""
    import numpy as np

    require(sha(path) == state.get("snapshot_sha256"), "Original retained NPZ SHA changed")
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "E_upper", "atom_logits", "bond_logits", "key_data",
                    "gradient_atom_logits", "gradient_bond_logits"}
        require(required.issubset(archive.files), "Retained NPZ lacks original graph/logit/key/gradient fields")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, array in arrays.items():
        require(array.dtype.kind in "biuf" and np.isfinite(array).all(), "Nonfinite/unsafe retained field: " + name)
    for name, shape in (("X", (25, 6)), ("E_upper", (300, 5)),
                        ("atom_logits", (25, 6)), ("bond_logits", (300, 5))):
        array = arrays[name]
        require(array.dtype == np.float32 and array.shape == shape, "Saved FP32 matrix shape/dtype differs: " + name)
    for name in ("X", "E_upper"):
        array = arrays[name]
        require((array >= 0).all() and (array <= 1).all()
                and np.allclose(array.sum(-1), 1.0, rtol=0, atol=1e-6), "Saved matrix left the simplex: " + name)
    require(arrays["key_data"].dtype == np.uint32 and arrays["key_data"].shape == (2,), "Saved native key differs")
    for name, shape in (("gradient_atom_logits", (25, 6)), ("gradient_bond_logits", (300, 5))):
        require(arrays[name].shape == shape, "Saved logit-gradient shape differs")
    comparisons = []
    for field in ("refreshed_surrogate", "autodiff_surrogate", "repair_weight"):
        if field in state:
            require(field in arrays, "Original snapshot lacks scalar: " + field)
            _scalar_equal(arrays[field], state[field], field)
            comparisons.append(field)
    primals = [name for name in arrays if name.endswith("_primal")]
    require(primals, "Retained snapshot lacks native primal evidence")
    for field in primals:
        loss = field[:-len("_primal")] + "_loss"
        expected = state.get(loss, state.get("components", {}).get(loss))
        require(expected is not None, "Native snapshot primal has no exact saved scalar correspondence")
        _scalar_equal(arrays[field], expected, field)
        comparisons.append(loss)
    atom_indices = arrays["X"].argmax(-1).tolist()
    bond_indices = arrays["E_upper"].argmax(-1).tolist()
    pairs = [[i, j] for i in range(25) for j in range(i + 1, 25)]
    readout = state.get("readout", {})
    require(isinstance(readout, dict), "Saved readout is malformed")
    for key, expected in (("element_indices", atom_indices), ("bond_upper_indices", bond_indices),
                          ("pairs", pairs), ("elements", [ATOMS[index] for index in atom_indices])):
        if key in readout:
            require(readout[key] == expected, "Saved readout/NPZ argmax correspondence failed: " + key)
    x, e = arrays["X"].tolist(), arrays["E_upper"].tolist()
    # JSON number conversion must round-trip each original FP32 probability.
    require(np.array_equal(np.asarray(json.loads(json.dumps(x)), dtype=np.float32), arrays["X"])
            and np.array_equal(np.asarray(json.loads(json.dumps(e)), dtype=np.float32), arrays["E_upper"]),
            "FP32 JSON serialization changed molecule probabilities")
    graph = layout_raw_graph(
        [{"id": index, "element": ATOMS[label]} for index, label in enumerate(atom_indices)],
        [{"source": pair[0], "target": pair[1], "label": BONDS[label]}
         for pair, label in zip(pairs, bond_indices) if label])
    frame = {
        "global_step": global_step, "atoms": x, "bonds": e, "matrix_dtype": "float32",
        "atom_labels": [ATOMS[index] for index in atom_indices],
        "bond_labels": [BONDS[index] for index in bond_indices],
        "graph": graph,
        "readout": deepcopy(readout), "losses": _losses(state),
        "source_snapshot": {"path": str(path), "bytes": Path(path).stat().st_size,
                            "sha256": state["snapshot_sha256"]},
        "matrix_sha256": {name: hashlib.sha256(arrays[name].tobytes(order="C")).hexdigest()
                          for name in ("X", "E_upper")},
        "interpretation": "Raw saved argmax graph; original readout only, no repair or new classification",
    }
    return frame, arrays, {"scalar_fields": comparisons, "FP32_roundtrip": True, "argmax_correspondence": True}


def _check_update(update, before, after, arm, kind):
    if kind == "noisy":
        require(update.get("learning_rate") == arm["rate"] and update.get("independent_primal_parity") is True,
                "Original noisy update rate/parity changed")
        require(update.get("before") == before.get("contact_loss") and update.get("after") == after.get("contact_loss"),
                "Noisy scalar transition differs from committed states")
        require(type(update.get("uphill")) is bool and update["uphill"] == (update["after"] > update["before"])
                and type(update.get("equal")) is bool and update["equal"] == (update["after"] == update["before"]),
                "Noisy uphill/equality bookkeeping differs")
        require(type(update.get("no_op")) is bool and type(update.get("sigma")) in (int, float)
                and math.isfinite(update["sigma"]) and update["sigma"] > 0, "Noisy applied-update fields missing")
        return
    attempts = update.get("attempts")
    require(isinstance(attempts, list) and len(attempts) == 1
            and update.get("selection_policy") == "fixed_step"
            and type(update.get("selected_trial")) is int and update["selected_trial"] == 0
            and update.get("fallback_used") is False, "Fixed run changed its one-candidate/no-fallback policy")
    attempt = attempts[0]
    require(type(attempt.get("trial")) is int and attempt["trial"] == 0
            and attempt.get("rate") == update.get("accepted_step_size") == arm["rate"], "Saved fixed step size differs")
    require(update.get("fixed_after") == attempt.get("independent_primal_surrogate")
            and update.get("fixed_before") == before.get("refreshed_surrogate")
            and update.get("fixed_after") == after.get("previous_fixed_surrogate_after"),
            "Fixed independent-primal transition differs from committed states")
    for field, value in before.items():
        if field.endswith("_loss") and type(value) in (int, float):
            name = field[:-len("_loss")]
            require(update.get(name + "_before") == value
                    and update.get(name + "_after") == after.get(field),
                    "Original native scalar transition differs: " + field)
    require(type(update.get("fixed_increased")) is bool
            and update["fixed_increased"] == (update["fixed_after"] > update["fixed_before"])
            and type(update.get("no_op")) is bool and type(attempt.get("strict")) is bool,
            "Fixed uphill/noop bookkeeping differs")


def collect_segment(segment, arm):
    kind = arm["kind"]
    require(kind in ("fixed", "noisy"), "Unknown saved trajectory schema")
    offset, requested = segment["global_offset"], segment["requested_updates"]
    require(type(offset) is int and offset >= 0 and type(requested) is int and requested > 0, "Invalid declared segment range")
    states, state_proof = committed_jsonl(segment["states"])
    updates, update_proof = committed_jsonl(segment["updates"])
    require(len(states) == len(updates) + 1 or (not states and not updates), "Committed state/update prefix is incomplete")
    require(len(updates) <= requested, "Committed prefix exceeds original segment budget")
    for index, state in enumerate(states):
        require(type(state.get("step")) is int and state["step"] == index, "Noncontiguous committed state index")
        require(state.get("global_step", offset + index) == offset + index, "Wrong absolute state index")
        require(type(state.get("snapshot_retained")) is bool, "Missing explicit retained-frame marker")
        expected = index == 0 or index == requested or index % MATRIX_STRIDE == 0
        require(state["snapshot_retained"] == expected, "Original retained-frame cadence changed")
        if not expected:
            require(state.get("snapshot") is None and state.get("snapshot_sha256") is None,
                    "Unsaved state has an invented snapshot")
    for index, update in enumerate(updates):
        field, expected = ("update", index + 1) if kind == "noisy" else ("step", index)
        require(type(update.get(field)) is int and update[field] == expected, "Noncontiguous committed update index")
        if kind == "fixed":
            require(update.get("global_step", offset + index) == offset + index, "Wrong absolute update index")
        _check_update(update, states[index], states[index + 1], arm, kind)
    config_path = bound_file(segment["config"])
    config = load(config_path)
    require(config.get("learning_rate" if kind == "noisy" else "initial_step_size") == arm["rate"], "Original configuration rate differs")
    summary_path = bound_file(segment["summary"])
    summary = load(summary_path)
    require(summary.get("source_manifest_sha256") == segment["source_manifest_sha256"]
            and summary.get("config_sha256") == segment["config"]["sha256"], "Original summary source/config pins differ")
    if "job_id" in segment:
        require(summary.get("job_id") == segment["job_id"], "Original summary job identity differs")
    status = summary.get("status")
    require(status in ("passed", "failed", "running"), "Original scientific status is unknown")
    if "scientific_status" in segment:
        require(status == segment["scientific_status"], "Original scientific status was relabeled")
    cases = summary.get("cases", [])
    candidates = [case for case in cases if isinstance(case, dict) and "states" in case and "updates" in case]
    if "case_report" in segment:
        candidates.append(load(bound_file(segment["case_report"])))
    for case in candidates:
        require(case.get("states", [])[:len(states)] == states
                and case.get("updates", [])[:len(updates)] == updates,
                "Original case and newline-committed records differ")
        require(len(case.get("states", [])) >= len(states) and len(case.get("updates", [])) >= len(updates),
                "Original case is missing committed rows")
    if status == "passed":
        require(len(updates) == requested and len(states) == requested + 1, "Original passed segment is incomplete")
    retained = []
    first_arrays = last_arrays = None
    for index, state in enumerate(states):
        if not state["snapshot_retained"]:
            continue
        path = _saved_path(segment["anchor_directory"], state["snapshot"])
        frame, arrays, proof = validate_frame(path, state, offset + index)
        retained.append((frame, proof))
        if index == 0:
            first_arrays = arrays
        if index == len(states) - 1:
            last_arrays = arrays
    if "retained_index" in segment:
        index_record = load(bound_file(segment["retained_index"]))
        original = index_record.get("retained_frames")
        require(isinstance(original, list) and [row["step"] for row in original]
                == [frame[0]["global_step"] - offset for frame in retained], "Original retained-frame inventory differs")
        for row, (frame, _) in zip(original, retained):
            require(row["snapshot_sha256"] == frame["source_snapshot"]["sha256"], "Original retained index SHA differs")
    return {
        "states": states, "updates": updates, "retained": retained,
        "first_arrays": first_arrays, "last_arrays": last_arrays,
        "proof": {"id": segment["id"], "global_offset": offset, "requested_updates": requested,
                  "committed_updates": len(updates), "states": state_proof, "updates": update_proof,
                  "summary": segment["summary"], "config": segment["config"],
                  "source_manifest_sha256": segment["source_manifest_sha256"],
                  "scientific_status": status, "scientific_error": summary.get("error"),
                  "retained_frames": len(retained)},
    }


def _boundary_equal(previous, current):
    import numpy as np

    require(previous is not None and current is not None, "Segment boundary requires both actual retained checkpoints")
    fields = ("X", "E_upper", "atom_logits", "bond_logits", "key_data")
    for field in fields:
        require(previous[field].dtype == current[field].dtype
                and previous[field].shape == current[field].shape
                and np.array_equal(previous[field], current[field]), "Segment boundary exact array/key mismatch: " + field)


def asset_binding(path, root):
    return {"path": Path(path).relative_to(root).as_posix(), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def collect_arm(arm, output):
    require(re.fullmatch(r"[a-z][a-z0-9_-]*", arm["id"]) is not None, "Unsafe arm identity")
    start, target = arm["global_start"], arm["global_target"]
    require(type(start) is int and type(target) is int and 0 <= start < target
            and arm["target_updates"] == target, "Absolute report range differs")
    segments = arm["segments"]
    scalar_rows, frames, proofs, boundaries = [], [], [], []
    previous = None
    current_end = start
    applied = 0
    counts = {"uphill": 0, "equal": 0, "noop": 0, "forced_fallback": 0}
    for segment in segments:
        require(segment["global_offset"] == current_end, "Gap or overlap in declared segment offsets")
        result = collect_segment(segment, arm)
        require(result["states"], "A declared started segment has no committed initial state")
        if previous is not None:
            _boundary_equal(previous["last_arrays"], result["first_arrays"])
            boundaries.append({"global_step": current_end, "exact_arrays_and_key": True,
                               "earlier_state": previous["states"][-1], "next_segment_state": result["states"][0]})
            scalar_rows[-1].setdefault("boundary_observations", []).append(deepcopy(result["states"][0]))
        for index, state in enumerate(result["states"]):
            if previous is not None and index == 0:
                continue
            update = None if index == 0 else result["updates"][index - 1]
            scalar_rows.append({"global_step": segment["global_offset"] + index,
                                "losses": _losses(state, update), "state": deepcopy(state),
                                "update": deepcopy(update), "segment_id": segment["id"]})
        for index, (frame, proof) in enumerate(result["retained"]):
            if previous is not None and index == 0:
                require(frame["global_step"] == frames[-1]["global_step"], "Boundary frame deduplication would skip a new frame")
                continue
            frame["validation"] = proof
            frames.append(frame)
        for update in result["updates"]:
            before = update["before"] if arm["kind"] == "noisy" else update["fixed_before"]
            after = update["after"] if arm["kind"] == "noisy" else update["fixed_after"]
            counts["uphill"] += int(after > before)
            counts["equal"] += int(after == before)
            counts["noop"] += int(update["no_op"])
            counts["forced_fallback"] += int(update.get("fallback_used", False))
        applied += len(result["updates"])
        current_end += len(result["updates"])
        proofs.append(result["proof"])
        previous = result
    require(current_end == start + applied and current_end <= target, "Absolute committed progress exceeds declared budget")
    require(not scalar_rows or [row["global_step"] for row in scalar_rows] == list(range(start, current_end + 1)),
            "Consolidated scalar states are not contiguous")
    actual_status = ("failed" if any(row["scientific_status"] == "failed" for row in proofs)
                     else "passed" if proofs and current_end == target and all(row["scientific_status"] == "passed" for row in proofs)
                     else "partial" if proofs else "not_started")
    require(arm["scientific_status"] == actual_status, "Report relabeled the original scientific outcomes")
    root = Path(output)
    scalar_path = root / (arm["id"] + "-scalars.json")
    write(scalar_path, {"rows": scalar_rows, "boundary_checks": boundaries,
                        "scope": "All newline-committed scalar records; duplicate boundary observations preserved"})
    chunks = []
    for number, first in enumerate(range(0, len(frames), CHUNK_FRAMES)):
        selected = frames[first:first + CHUNK_FRAMES]
        path = root / (arm["id"] + "-frames-" + str(number).zfill(4) + ".json")
        write(path, {"frames": selected})
        chunks.append({**asset_binding(path, root), "first": selected[0]["global_step"],
                       "last": selected[-1]["global_step"], "count": len(selected),
                       "steps": [row["global_step"] for row in selected]})
    last = None
    if frames:
        last = {"global_step": frames[-1]["global_step"], "chunk": chunks[-1]["path"],
                "index": (len(frames) - 1) % CHUNK_FRAMES}
    return {
        "id": arm["id"], "label": arm["label"], "kind": arm["kind"],
        "target_updates": target, "global_start": start, "global_target": target,
        "committed_updates": applied, "latest_global_step": current_end,
        "scientific_status": actual_status, "presentation_status": "saved_prefix_verified",
        "status_detail": arm.get("status_detail", ""), "metric_labels": arm.get("metric_labels", {}),
        "continuation_status": arm.get("continuation_status", "not_recorded"),
        "rate": arm["rate"], "atom_categories": list(ATOMS), "bond_categories": list(BONDS),
        "scalars": {**asset_binding(scalar_path, root), "count": len(scalar_rows)},
        "frames": {"count": len(frames), "steps": [row["global_step"] for row in frames], "chunks": chunks},
        "latest": last, "update_counts": counts, "segment_proofs": proofs,
        "matrix_stride": MATRIX_STRIDE, "loss_stride": 1,
        "missing_frame_policy": "Show scalar values and an empty matrix/graph view; never interpolate",
        "interpretation": "Saved fixed-pose proxy observations; graph validity or scores do not establish binding/activity",
    }


def render(plan, output):
    require(plan.get("version") == VERSION, "Wrong saved-report plan version")
    require(re.fullmatch(r"edition-[0-9]{4}|edition-final", plan.get("edition", "")) is not None, "Unsafe immutable edition")
    require(isinstance(plan.get("captured_at"), str), "Capture time must be explicit")
    arms = plan.get("arms")
    require(isinstance(arms, list) and len(arms) in (4, 6)
            and len({arm["id"] for arm in arms}) == len(arms), "Duplicate/missing report arms")
    output = Path(output)
    output.mkdir(exist_ok=False)
    result = {"status": "failed", "version": VERSION, "edition": plan["edition"],
              "scientific_success_claim": False, "original_results_changed": False}
    try:
        manifest = {"version": VERSION, "edition": plan["edition"], "captured_at": plan["captured_at"],
                    "arms": [collect_arm(arm, output) for arm in arms],
                    "interpretation": "Presentation success is separate from scientific success; contact, affinity, binder and composite metrics remain distinct"}
        # Inputs must still be byte-identical after rendering every retained frame.
        for arm in arms:
            for segment in arm["segments"]:
                for key in ("states", "updates", "config", "summary", "case_report", "retained_index"):
                    if key in segment:
                        bound_file(segment[key])
        for arm in manifest["arms"]:
            for chunk in arm["frames"]["chunks"]:
                for frame in load(output / chunk["path"])["frames"]:
                    bound_file(frame["source_snapshot"])
        write(output / "manifest.json", manifest)
        result.update(status="passed_saved_only_render", arm_counts={
            arm["id"]: {"updates": arm["committed_updates"], "frames": arm["frames"]["count"],
                        "scientific_status": arm["scientific_status"]} for arm in manifest["arms"]},
            manifest_sha256=sha(output / "manifest.json"),
            scientific_success_claim=False, no_interpolation=True, native_inference_executed=False,
            all_scalar_records_retained=True, saved_FP32_frames_verified=True)
        return manifest
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        write(output / "assessment.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("LSB_JOBID", "").isdigit() and os.environ.get("LSB_QUEUE") == "hpc",
            "Scientific frame validation/rendering requires allocated DTU HPC CPU")
    return render(load(args.plan), args.output)


if __name__ == "__main__":
    main()
