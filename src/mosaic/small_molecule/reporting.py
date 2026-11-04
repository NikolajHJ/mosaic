"""Model-free trajectory analysis and unmodified independent graph sampling.

No JAX, model imports, chemistry repair, or changes to experiment artifacts.
See docs/small_molecule_reporting.md for definitions and interpretation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np

ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
BOND_STATES = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
SCHEMA = "mosaic_analysis_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Artifact escapes source directory: {relative}")
    return path


def validate_arrays(X, E):
    if X.ndim != 2 or X.shape[1] != 6 or len(X) < 2:
        raise ValueError("Expected X[n>=2,6]")
    if E.shape != (len(X) * (len(X) - 1) // 2, 5):
        raise ValueError("Expected E_upper[n*(n-1)/2,5]")
    for a in (X, E):
        if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
            raise ValueError("Nonfinite or out-of-simplex probabilities")
        if not np.allclose(a.sum(-1), 1, rtol=0, atol=1e-6):
            raise ValueError("Probability rows do not sum to one")


@dataclass
class State:
    step: int
    X: np.ndarray
    E: np.ndarray
    loss: float | None = None
    metrics: dict = field(default_factory=dict)
    gX: np.ndarray | None = None
    gE: np.ndarray | None = None
    readout: dict | None = None
    artifact: dict = field(default_factory=dict)


@dataclass
class Run:
    name: str
    root: Path
    states: list[State]
    status: str
    config: dict
    metadata: dict
    target: dict | None = None
    warnings: list[str] = field(default_factory=list)
    presentation: dict = field(default_factory=dict)


def _state(path, row, *, study=False):
    with np.load(path, allow_pickle=False) as arrays:
        X, E = arrays["X"].astype(float), arrays["E_upper"].astype(float)
        gx = arrays["gradient_X"].astype(float) if "gradient_X" in arrays else None
        ge = (
            arrays["gradient_E_upper"].astype(float)
            if "gradient_E_upper" in arrays
            else None
        )
    validate_arrays(X, E)
    return State(
        int(row["step"] if study else row["evaluated_step"]),
        X,
        E,
        row.get("primal_loss") if study else row.get("loss"),
        row.get("aux_metrics", {}),
        gx,
        ge,
        row.get("readout") if study else None,
        {"path": str(path), "sha256": sha256(path)},
    )


def load_runs(source):
    """Accept a trace JSONL, target root, batch root, or upstream study summary.

    Running JSONL is atomic in the current writer. Restrict it to the committed
    checkpoint prefix, so a trace written ahead of a checkpoint is not final.
    """
    source = Path(source).resolve()
    if source.is_file() and source.name in ("run_summary.json", "manifest.json"):
        source = source.parent
    if source.is_file() and source.suffix == ".json":
        document = read_json(source)
        if document.get("schema") == "mosaic_run_collection_v1":
            return _load_collection(source, document)
    if source.is_file() and source.name == "summary.json":
        return _load_study(source)
    if (
        source.is_dir()
        and (source / "summary.json").exists()
        and "cases" in read_json(source / "summary.json", {})
    ):
        return _load_study(source / "summary.json")
    traces = [source] if source.is_file() else sorted(source.rglob("trace.jsonl"))
    if not traces:
        raise ValueError(f"No trace.jsonl or study summary found in {source}")
    runs = []
    for trace in traces:
        directory = trace.parent
        manifest = read_json(directory / "manifest.json", {})
        rows = [
            json.loads(line) for line in trace.read_text().splitlines() if line.strip()
        ]
        pointer = read_json(directory / "latest_checkpoint.json")
        warnings = []
        if pointer:
            cp = safe_path(directory, pointer["path"])
            if sha256(cp) != pointer["sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            with np.load(cp, allow_pickle=False) as arrays:
                count = json.loads(str(arrays["metadata"]))["evaluation_count"]
            if len(rows) < count:
                raise ValueError("Trace is shorter than committed checkpoint")
            if len(rows) > count:
                warnings.append("Ignored trace records beyond the committed checkpoint")
            rows = rows[:count]
        if not rows:
            raise ValueError(f"No committed evaluations yet: {trace}")
        states = []
        for row in rows:
            s = _state(safe_path(directory, row["evaluated_arrays"]), row)
            if row.get("readout"):
                s.readout = read_json(safe_path(directory, row["readout"]))
            states.append(s)
        if any(b.step <= a.step for a, b in pairwise(states)):
            raise ValueError("Evaluated steps must be strictly increasing")
        if any(s.X.shape != states[0].X.shape for s in states):
            raise ValueError("Atom count changed within a trajectory")
        parent = directory.parent if directory.name == "trajectory" else directory
        config = read_json(directory / "config.json", {})
        resolved = config.get("resolved", {})
        rates = resolved.get("rates", {})
        offset = resolved.get(
            "global_step_offset",
            rates.get("canonical_anchor_step", 0) + rates.get("offset", 0),
        )
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("Invalid global step offset")
        for state in states:
            state.artifact["segment_step"] = state.step
            state.step += offset
        target = read_json(parent / "target.json")
        metadata = {
            "manifest": manifest,
            "input_identity": read_json(parent / "input_identity.json"),
            "run_summary": read_json(parent / "run_summary.json"),
            "reference_score": read_json(parent / "reference_score.json"),
            "soft_hard_gap": read_json(parent / "soft_hard_gap.json", []),
            "continuation": read_json(parent / "continuation.json"),
            "pae_gate": read_json(parent / "pae_gate.json"),
            "trace_sha256": sha256(trace),
        }
        metadata["soft_hard_gap"] = [
            dict(r, step=r.get("global_step", r["step"] + offset))
            for r in metadata["soft_hard_gap"]
        ]
        runs.append(
            Run(
                parent.name,
                parent,
                states,
                manifest.get("status", "unknown"),
                config,
                metadata,
                target,
                warnings,
            )
        )
    return runs


def _load_collection(path, document):
    """Load an explicit shortlist and join only continuous, matched segments."""
    root = (path.parent / document.get("root", ".")).resolve()
    entries = document.get("runs", [])
    identifiers = [e["id"] for e in entries]
    if not entries or len(set(identifiers)) != len(identifiers):
        raise ValueError("Collection requires nonempty unique stable run IDs")
    result = []
    for entry in entries:
        parts = []
        for relative in entry["sources"]:
            loaded = load_runs(safe_path(root, relative))
            if len(loaded) != 1:
                raise ValueError("Each collection source must identify one run segment")
            parts.append(loaded[0])
        if not parts:
            raise ValueError("Collection entry has no source segments")
        first = parts[0]
        states = list(first.states)
        boundaries = []
        for part in parts[1:]:
            a, b = states[-1], part.states[0]
            if part.target != first.target:
                raise ValueError("Cannot join segments with different targets")
            for key in ("eta_X", "eta_E", "rng_policy"):
                if part.config.get("trajectory", {}).get(key) != first.config.get(
                    "trajectory", {}
                ).get(key):
                    raise ValueError(f"Cannot join segments with different {key}")
            if (
                a.step != b.step
                or not np.array_equal(a.X, b.X)
                or not np.array_equal(a.E, b.E)
            ):
                raise ValueError("Collection segment boundary is not continuous")
            loss_error = (
                abs(a.loss - b.loss)
                if a.loss is not None and b.loss is not None
                else None
            )
            if loss_error is not None and loss_error > 2e-6:
                raise ValueError("Collection boundary loss mismatch")
            boundaries.append(
                {
                    "step": a.step,
                    "identical_graph": True,
                    "loss_absolute_difference": loss_error,
                    "retained_artifact": a.artifact,
                    "replayed_artifact": b.artifact,
                }
            )
            states.extend(part.states[1:])
        gaps = {
            r["step"]: r for p in parts for r in p.metadata.get("soft_hard_gap", [])
        }
        metadata = {
            "segments": [
                {
                    "root": str(p.root),
                    "config": p.config,
                    "status": p.status,
                    "metadata": p.metadata,
                }
                for p in parts
            ],
            "soft_hard_gap": [gaps[k] for k in sorted(gaps)],
            "segment_boundaries": boundaries,
            "collection": {
                "title": document.get("title"),
                "description": document.get("description"),
                "reviewed_segment_count": document.get("reviewed_segment_count"),
                "star_reset_token": document.get("star_reset_token"),
                "source": str(path),
                "sha256": sha256(path),
            },
        }
        config = {
            "trajectory": {
                **first.config.get("trajectory", {}),
                "steps": states[-1].step - states[0].step,
            },
            "interpretation": "Derived display of explicitly joined segments; native configurations retained per segment",
        }
        result.append(
            Run(
                entry.get("label", entry["id"]),
                first.root,
                states,
                "complete"
                if all(p.status in ("complete", "passed") for p in parts)
                else "incomplete",
                config,
                metadata,
                first.target,
                [w for p in parts for w in p.warnings],
                {
                    "id": entry["id"],
                    "starred": bool(entry.get("starred", False)),
                    "reason": entry.get("reason", ""),
                    "tags": entry.get("tags", []),
                },
            )
        )
    return result


def _load_study(path):
    summary = read_json(path)
    for key in ("model_digest", "context_digest"):
        if summary.get(key + "_before") != summary.get(key + "_after"):
            raise ValueError(f"Study changed frozen {key}")
    runs = []
    for index, case in enumerate(summary["cases"]):
        root = path.parent / f"case-{index}"
        states = [
            _state(safe_path(root, row["snapshot"]), row, study=True)
            for row in case["states"]
        ]
        if not states or [s.step for s in states] != list(range(len(states))):
            raise ValueError("Study states must be nonempty and contiguous")
        expected = summary["config"].get("checks", {}).get("steps")
        if (
            summary.get("status") == "passed"
            and expected is not None
            and len(states) != expected + 1
        ):
            raise ValueError("Passed study has an incomplete trajectory")
        runs.append(
            Run(
                case.get("label", f"case-{index}"),
                root,
                states,
                summary.get("status", "unknown"),
                summary["config"],
                {"study": summary, "summary_sha256": sha256(path)},
            )
        )
    return runs


def project_simplex(rows):
    """Euclidean row projection, used only for detached gradient diagnostics."""
    u = np.sort(rows, axis=-1)[:, ::-1]
    c = np.cumsum(u, axis=-1) - 1
    indices = np.arange(1, rows.shape[1] + 1)
    rho = (u - c / indices > 0).sum(-1) - 1
    theta = c[np.arange(len(rows)), rho] / (rho + 1)
    return np.maximum(rows - theta[:, None], 0)


def tv_rows(a, b):
    return float(np.abs(a - b).sum(-1).mean() / 2)


def distance(a, b):
    return (tv_rows(a.X, b.X) + tv_rows(a.E, b.E)) / 2


def entropy_rows(p):
    return -(p * np.log(np.maximum(p, 1e-300))).sum(-1)


def uncertainty(s):
    existence = 1 - s.E[:, 0]
    present = s.E[:, 1:]
    conditional = np.divide(
        present,
        existence[:, None],
        out=np.zeros_like(present),
        where=existence[:, None] > 0,
    )
    # Weight conditional order entropy by existence; absent pairs contribute zero.
    h_order = entropy_rows(conditional)
    top = np.sort(s.X, axis=-1)
    return {
        "atom_entropy": float(entropy_rows(s.X).mean() / np.log(6)),
        "existence_entropy": float(
            entropy_rows(np.c_[existence, 1 - existence]).mean() / np.log(2)
        ),
        "bond_order_entropy": float(
            np.dot(existence, h_order) / (max(existence.sum(), 1e-300) * np.log(4))
        ),
        "atom_margin": float((top[:, -1] - top[:, -2]).mean()),
        "expected_bonds": float(existence.sum()),
        "atom_entropies": (entropy_rows(s.X) / np.log(6)).tolist(),
        "argmax_labelled_graph_log_probability": float(
            np.log(s.X.max(-1)).sum() + np.log(s.E.max(-1)).sum()
        ),
    }


def readout_graph(atom_indices, bond_indices):
    """Return the unmodified graph, validity, and identity. No fragment removal."""
    from rdkit import Chem, rdBase

    result = {
        "element_indices": list(map(int, atom_indices)),
        "bond_upper_indices": list(map(int, bond_indices)),
        "sanitized": False,
        "connected": False,
        "canonical_smiles": None,
        "fragment_count": None,
        "error": None,
    }
    mol = Chem.RWMol()
    for i in atom_indices:
        mol.AddAtom(Chem.Atom(ELEMENTS[int(i)]))
    bonds = (
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    )
    n = len(atom_indices)
    pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
    for (i, j), category in zip(pairs, bond_indices, strict=True):
        if category:
            mol.AddBond(i, j, bonds[int(category)])
            if category == 4:
                mol.GetAtomWithIdx(i).SetIsAromatic(True)
                mol.GetAtomWithIdx(j).SetIsAromatic(True)
    mol = mol.GetMol()
    result["fragment_count"] = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
    result["connected"] = result["fragment_count"] == 1
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
        # Stage 0 has no stereo variables; do not imply stereo was designed.
        result["canonical_smiles"] = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False
        )
        result["sanitized"] = True
    except Exception as error:  # noqa: BLE001 -- chemistry failures are report data
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def scaffold(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    # An empty Murcko scaffold groups all acyclic molecules, so flag that group.
    return core or "ACYCLIC_EMPTY_SCAFFOLD"


def wilson(k, n, z=1.959963984540054):
    """Pointwise 95% Wilson interval for a fixed Bernoulli event."""
    p, c = k / n, z * z / n
    centre = (p + c / 2) / (1 + c)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + c)
    return [max(0.0, float(centre - radius)), min(1.0, float(centre + radius))]


def _draw_rows(p, rng, size):
    # Only compensate float serialization roundoff after strict validation.
    p = p / p.sum(-1, keepdims=True)
    return np.stack([rng.choice(len(row), size=size, p=row) for row in p], axis=1)


def sample_state(s, *, count=256, seed=17):
    """IID product-categorical samples, with invalid and disconnected draws retained."""
    if count < 1:
        raise ValueError("Sample count must be positive")
    validate_arrays(s.X, s.E)
    rng = np.random.default_rng(np.random.SeedSequence([seed, s.step]))
    atoms, bonds = _draw_rows(s.X, rng, count), _draw_rows(s.E, rng, count)
    draws = [readout_graph(a, b) for a, b in zip(atoms, bonds, strict=True)]
    identities, families = Counter(), Counter()
    valid = connected = disconnected = 0
    for draw in draws:
        valid += int(draw["sanitized"])
        disconnected += int(not draw["connected"])
        if draw["sanitized"] and draw["connected"]:
            connected += 1
            identities[draw["canonical_smiles"]] += 1
            families[scaffold(draw["canonical_smiles"])] += 1

    def distribution(counter):
        return [
            {"identity": name, "count": k, "probability": k / count}
            for name, k in counter.most_common()
        ]

    def entropy(counter, other):
        counts = list(counter.values()) + ([other] if other else [])
        p = np.asarray(counts) / count
        return float(-(p * np.log(p)).sum())

    hm = entropy(identities, count - connected)
    hf = entropy(families, count - connected)
    return {
        "step": s.step,
        "seed": seed,
        "count": count,
        "sampler": "independent categorical X/E; no repair, rejection, or fragment removal",
        "sanitized_fraction": valid / count,
        "valid_connected_fraction": connected / count,
        "disconnected_fraction": disconnected / count,
        "valid_connected_interval_95": wilson(connected, count),
        "molecules": distribution(identities),
        "families": distribution(families),
        "molecule_entropy_nats": hm,
        "family_entropy_nats": hf,
        "within_family_entropy_nats": max(0.0, hm - hf),
        "entropy_policy": "plug-in estimates; all invalid/disconnected draws in one OTHER category; downward biased at finite sample count",
        "draws": draws,
    }


def analyse_run(run, *, windows=(5, 20, 50), reference_eta=0.01, max_recurrence=200):
    if not np.isfinite(reference_eta) or reference_eta <= 0:
        raise ValueError("Reference step size must be positive")
    if any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in windows):
        raise ValueError("Window lengths must be positive integers")
    states, rows = run.states, []
    # Cache only the distance band needed for window metrics. This avoids
    # recomputing O(T*W^2) graph distances for long upstream studies.
    max_window = max(windows, default=0)
    bands = [
        {j: distance(s, states[j]) for j in range(max(0, i - max_window), i)}
        for i, s in enumerate(states)
    ]
    for i, s in enumerate(states):
        row = {
            "step": s.step,
            "loss": s.loss,
            **uncertainty(s),
            "metrics": s.metrics,
            "atom_step_tv": None,
            "bond_step_tv": None,
            "step_distance": None,
            "distance_from_start": distance(s, states[0]),
            "windows": {},
            "projected_gradient_X_rms": None,
            "projected_gradient_E_rms": None,
        }
        for name, p, g in (("X", s.X, s.gX), ("E", s.E, s.gE)):
            if g is not None:
                if g.shape != p.shape or not np.isfinite(g).all():
                    raise ValueError("Gradient shape/nonfinite mismatch")
                mapping = (p - project_simplex(p - reference_eta * g)) / reference_eta
                row[f"projected_gradient_{name}_rms"] = float(
                    np.linalg.norm(mapping) / np.sqrt(len(p))
                )
        if i:
            row.update(
                atom_step_tv=tv_rows(s.X, states[i - 1].X),
                bond_step_tv=tv_rows(s.E, states[i - 1].E),
                step_distance=distance(s, states[i - 1]),
            )
        for w in windows:
            if i < w:
                continue
            recent = states[i - w : i + 1]
            diameter = max(
                bands[r][s] for r in range(i - w + 1, i + 1) for s in range(i - w, r)
            )
            path = sum(bands[r][r - 1] for r in range(i - w + 1, i + 1))
            drift = bands[i][i - w]
            row["windows"][str(w)] = {
                "diameter": diameter,
                "net_displacement": drift,
                "path_length": path,
                "net_path_ratio": drift / path if path else None,
                "span_updates": recent[-1].step - recent[0].step,
            }
        rows.append(row)
    ids = np.unique(
        np.linspace(0, len(states) - 1, min(max_recurrence, len(states))).astype(int)
    )
    recurrence = [[distance(states[i], states[j]) for j in ids] for i in ids]
    # MSD in Euclidean probability coordinates. Interpret scaling only against
    # a null process: optimization increments are correlated and C is bounded.
    vector = np.stack([np.r_[s.X.ravel(), s.E.ravel()] for s in states])
    lags = (
        np.unique(
            np.geomspace(1, max(1, len(states) - 1), min(24, len(states) - 1)).astype(
                int
            )
        )
        if len(states) > 1
        else []
    )
    msd = [
        {
            "lag_saved_states": int(lag),
            "mean_update_span": float(
                np.mean(
                    [
                        states[i + lag].step - states[i].step
                        for i in range(len(states) - lag)
                    ]
                )
            ),
            "mean_squared_displacement": float(
                np.square(vector[lag:] - vector[:-lag]).sum(-1).mean()
            ),
        }
        for lag in lags
    ]
    return {
        "schema": SCHEMA,
        "name": run.name,
        "status": run.status,
        "reference_eta": reference_eta,
        "window_units": "saved evaluations, with actual update spans recorded",
        "rows": rows,
        "recurrence_steps": [states[i].step for i in ids],
        "recurrence": recurrence,
        "mean_squared_displacement": msd,
        "source": {
            "root": str(run.root),
            "metadata": run.metadata,
            "config": run.config,
            "artifacts": [s.artifact for s in states],
        },
        "warnings": run.warnings,
        "assessment": "Descriptive evidence only; no automatic claim of asymptotic convergence, affinity, or binding.",
    }


def molecule_assets(smiles, *, seed=17):
    """2D drawing and one free conformer; never a pose in the target pocket."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(320, 220)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    result = {
        "svg": svg[svg.index("<svg") :],
        "molblock": None,
        "conformer_method": "RDKit ETKDGv3, one unconstrained free conformer; no stereochemistry was optimized",
        "conformer_status": "not attempted",
        "force_field_status": None,
    }
    try:
        mol3 = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed % (2**31 - 1))
        params.numThreads = 1
        with rdBase.BlockLogs():
            code = AllChem.EmbedMolecule(mol3, params)
        if code:
            result["conformer_status"] = "embedding failed"
        else:
            if AllChem.MMFFHasAllMoleculeParams(mol3):
                code = AllChem.MMFFOptimizeMolecule(mol3, maxIters=300)
                result["force_field_status"] = (
                    "MMFF converged" if code == 0 else "MMFF iteration limit"
                )
            result["molblock"] = Chem.MolToMolBlock(Chem.RemoveHs(mol3))
            result["conformer_status"] = "generated; not docked or model-predicted"
    except Exception as error:  # noqa: BLE001 -- preserve conformer failure without substituting chemistry
        result["conformer_status"] = (
            f"generation failed: {type(error).__name__}: {error}"
        )
    return result
