"""Detached, authenticated repair references; no search or RDKit inside AD."""

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import time

import jax
import numpy as np

from ._reference.atom_preserving_chemistry import search_atom_preserving_repair
from ._reference.chemistry import encode_smiles, inspect_graph
from ._reference.chemistry_regularization import ALLOWED_VALENCES, RULE_VERSION
from ._reference.probability_repair import ProbabilityRepairConfig, probability_identity
from ._reference.quality import graph_identity
from ._reference.repair_progress import ProgressSearchConfig, search_progress_repair
from .state import validate_decoded_shape


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def components(atoms, bonds):
    remaining, groups = set(np.flatnonzero(atoms).tolist()), []
    while remaining:
        stack = [min(remaining)]
        remaining.remove(stack[0])
        group = []
        while stack:
            node = stack.pop()
            group.append(node)
            neighbors = sorted(j for j in remaining if bonds[node, j])
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        groups.append(sorted(group))
    return groups


@dataclass(frozen=True)
class PreparedFailureGuidance:
    """Private frozen tensor payload plus a complete JSON provenance receipt.

    Every loss call checks both against the identity; a caller cannot mutate a
    target, category, cap, cut or coverage status without invalidating it.
    Timing is external to the scientific preparation identity.
    """

    arrays: dict
    receipt: dict
    identity: str
    preparation_seconds: float
    search_seconds: tuple = ()

    def verify(self, spec):
        payload = {k: np.asarray(v).tolist() for k, v in self.arrays.items()}
        if (
            self.receipt.get("schema_sha256") != spec.fingerprint()
            or digest({"arrays": payload, "receipt": self.receipt}) != self.identity
        ):
            raise ValueError("Frozen chemistry preparation identity changed")

    def record(self):
        return {
            **self.receipt,
            "identity": self.identity,
            "preparation_seconds": self.preparation_seconds,
            "search_seconds": [
                {"original": a, "additional": b} for a, b in self.search_seconds
            ],
            "arrays": {k: np.asarray(v).tolist() for k, v in self.arrays.items()},
        }


def prepare_failure_guidance(
    decoded, spec, *, candidate_ids=None, update=0, search=None, progress_search=None,
    severity_progress=None, structural_repair=None, repair_proposal=None,
    common_organic_scope=False, atom_exchange=None,
):
    """Prepare one complete scheduled batch from its exact original snapshot.

    An additional pass always starts from that same raw graph. Complete native
    witnesses replace fallback; invalid partial witnesses retain fallback.
    Unknown search errors propagate, rather than becoming censored rows.
    """
    started = time.perf_counter()
    validate_decoded_shape(decoded, spec)
    decoded = decoded.math_precision()
    if search is None:
        search = ProbabilityRepairConfig(
            max_edits=3,
            checks_per_depth=32,
            beam_width=4,
            probability_weight=1.0,
            probability_floor=1e-12,
        )
    if not isinstance(search, ProbabilityRepairConfig) or (
        progress_search is not None
        and not isinstance(progress_search, ProgressSearchConfig)
    ):
        raise TypeError("Require explicit repair search configurations")
    if type(update) is not int or update < 0:
        raise ValueError("Invalid source update")
    if severity_progress is not None:
        from .severity import severity_config

        severity_progress = severity_config(severity_progress)
    if structural_repair is not None:
        from .structural import StructuralRepairConfig, structural_config, search_structural_repair

        structural_repair = structural_config(structural_repair)
    if repair_proposal is not None and (structural_repair is None or not callable(repair_proposal)):
        raise ValueError("An additional proposal requires the original structural repair")
    if type(common_organic_scope) is not bool:
        raise ValueError("Require an explicit boolean common-organic scope")
    if common_organic_scope:
        from .atom_exchange import search_atom_exchange, within_common_organic_scope
        from .environment_policy import BOND_ORDERS, VERSION as SCOPE_VERSION

        if (structural_repair is None or repair_proposal is not None
                or (progress_search is not None and progress_search.allow_atom_edits)):
            raise ValueError("Scoped repair requires structural repair and atom-preserving fallbacks")
    if atom_exchange is not None and (
        not common_organic_scope or structural_repair != StructuralRepairConfig()
        or type(atom_exchange) is not dict
        or set(atom_exchange) != {"atom_edit_penalty", "max_substitutions"}
    ):
        raise ValueError("Atom exchange requires scoped default structural repair and explicit search bounds")
    values = [
        np.asarray(v)
        for v in (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits)
    ]
    b, n = values[0].shape[:2]
    shapes = (
        (b, n, len(spec.atomic_numbers)),
        (b, n, len(spec.formal_charges)),
        (b, n, n, len(spec.bond_names)),
    )
    if (
        n != spec.max_nodes
        or b < 1
        or any(
            v.shape != s or v.dtype.kind != "f" or not np.isfinite(v).all()
            for v, s in zip(values, shapes)
        )
    ):
        raise ValueError("Require finite source logits matching the explicit schema")
    ids = tuple(range(b)) if candidate_ids is None else tuple(candidate_ids)
    if (
        len(ids) != b
        or len(set(ids)) != b
        or any(type(v) is not int or v < 0 for v in ids)
    ):
        raise ValueError("Require unique aligned candidate IDs")
    probabilities = [
        np.asarray(jax.nn.softmax(v, -1))
        for v in (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits)
    ]
    atoms, charges, bonds = (np.argmax(v, -1) for v in probabilities)
    active = atoms != 0
    bonds = np.where(
        active[:, :, None] & active[:, None, :] & ~np.eye(n, dtype=bool), bonds, 0
    )
    charges = np.where(active, charges, spec.formal_charges.index(0))
    arrays = {
        "target_atoms": atoms.copy(),
        "target_charges": charges.copy(),
        "target_bonds": bonds.copy(),
        "active": active.copy(),
        "caps": np.full((b, n), -1, dtype=np.int32),
        "route": np.zeros(b, dtype=np.int32),
        "changed": np.zeros((b, n, n), dtype=bool),
        "cuts": np.zeros((b, n, n * (n - 1) // 2), dtype=bool),
        "cut_active": np.zeros((b, n), dtype=bool),
    }
    records, timings = [], []
    upper = list(zip(*np.triu_indices(n, 1)))
    for row, candidate_id in enumerate(ids):
        raw = (atoms[row], charges[row], bonds[row])
        p = tuple(v[row] for v in probabilities)
        search_started = time.perf_counter()
        structural = None
        exchange_result = None
        if atom_exchange is not None:
            exchange_result = search_atom_exchange(raw, p, spec, search, **atom_exchange)
            structural = exchange_result["selected"]
        elif structural_repair is not None:
            structural = search_structural_repair(
                raw, p, spec, search, structural_repair,
                accept=within_common_organic_scope if common_organic_scope else None,
            )
        if structural is not None and structural["status"] in ("already_valid", "witness_found"):
            base = structural
        else:
            base = search_atom_preserving_repair(raw, p, spec, search)
        beam_search_performed = base is not structural
        projection = None
        scope_rejections = []
        if repair_proposal is not None and base["status"] in ("witness_found", "censored"):
            base, projection = repair_proposal(raw, p, spec, search, structural_repair, base)
        original_seconds = time.perf_counter() - search_started
        additional_seconds = 0.0
        status, target, extra = base["status"], None, None
        if status == "witness_found":
            target = tuple(
                np.asarray(base["weighted_witness"]["labels"][k], dtype=np.int64)
                for k in ("atoms", "charges", "bonds")
            )
            arrays["route"][row] = 1
        elif status not in ("already_valid", "censored", "unsupported"):
            raise RuntimeError("Unknown original repair outcome")
        if common_organic_scope and beam_search_performed and status in ("already_valid", "witness_found"):
            if not within_common_organic_scope(raw if target is None else target, spec):
                scope_rejections.append({"search": "beam", "status": status})
                status, target = "censored", None
                base = {**base, "status": status, "raw_valid": False, "weighted_witness": None,
                        "best_observed_weighted_cost": None, "reason": "Outside common-organic scope"}
        if status in ("censored", "unsupported"):
            arrays["route"][row] = 4
            if (progress_search is not None or severity_progress is not None) and active[row].any():
                search_started = time.perf_counter()
                if severity_progress is None:
                    extra = search_progress_repair(raw, p, spec, progress_search)
                else:
                    from .severity import search_severity_progress

                    extra = search_severity_progress(raw, p, spec, severity_progress)
                additional_seconds = time.perf_counter() - search_started
                if extra["status"] not in (
                    "valid",
                    "partial",
                    "no_progress",
                    "already_valid",
                    "unsupported",
                ):
                    raise RuntimeError("Unknown progress repair outcome")
                witness = extra["witness"]
                if witness is not None:
                    witness.verify(spec)
                    if witness.original != tuple(
                        tuple(int(x) for x in v)
                        if i < 2
                        else tuple(tuple(int(x) for x in line) for line in v)
                        for i, v in enumerate(raw)
                    ):
                        raise ValueError("Progress witness lost original raw identity")
                    target = tuple(
                        np.asarray(v, dtype=np.int64) for v in witness.labels
                    )
                    if common_organic_scope and not within_common_organic_scope(target, spec):
                        scope_rejections.append({"search": "progress", "status": extra["status"]})
                        target = None
                    else:
                        arrays["route"][row] = 2 if witness.kind == "valid" else 3
        if target is not None:
            if not np.array_equal(raw[1], target[1]) or not np.array_equal(
                raw[0] != 0, target[0] != 0
            ):
                raise ValueError("Target changed charge or occupied slots")
            if not np.array_equal(raw[0], target[0]):
                if atom_exchange is not None and arrays["route"][row] == 1 and base is structural:
                    swapped = np.flatnonzero(raw[0] != target[0])
                    selected = base.get("exchange", {})
                    if (len(swapped) != 1 or swapped[0] != selected.get("slot")
                            or raw[0][swapped[0]] != selected.get("old_label")
                            or target[0][swapped[0]] != selected.get("new_label")):
                        raise ValueError("Target differs from the selected single-atom exchange")
                    slot = swapped[0]
                    if spec.atomic_numbers[target[0][slot]] == 16:
                        old_allowed = ALLOWED_VALENCES.get(spec.atomic_numbers[raw[0][slot]], {}).get(
                            spec.formal_charges[raw[1][slot]]
                        )
                        orders = (spec.bond_names[bond] for bond in target[2][slot])
                        if not old_allowed or sum(1.5 if b == "AROMATIC" else BOND_ORDERS[b]
                                                  for b in orders) > max(old_allowed):
                            raise ValueError("New sulfur exceeds the source atom's bond-order cap")
                elif (arrays["route"][row] != 2 or severity_progress is not None
                      or progress_search is None or not progress_search.allow_atom_edits):
                    raise ValueError("Target changed atom identities")
            if common_organic_scope and not within_common_organic_scope(target, spec):
                raise ValueError("Selected target outside common-organic scope")
            if arrays["route"][row] in (1, 2):
                checked = inspect_graph(*target, spec)
                if (
                    checked.smiles is None
                    or encode_smiles(checked.smiles, spec)["smiles"] != checked.smiles
                ):
                    raise ValueError(
                        "Complete witness failed independent native roundtrip"
                    )
            for key, value in zip(
                ("target_atoms", "target_charges", "target_bonds"), target
            ):
                arrays[key][row] = value
            arrays["changed"][row] = np.triu(target[2] != raw[2], 1)
            if not arrays["changed"][row].any() and arrays["route"][row] == 3:
                raise ValueError("Partial witness did not change a bond")
        groups = components(raw[0], raw[2])
        for slot, (a, q) in enumerate(zip(raw[0], raw[1])):
            allowed = (
                ALLOWED_VALENCES.get(spec.atomic_numbers[int(a)], {}).get(
                    spec.formal_charges[int(q)]
                )
                if a
                else None
            )
            arrays["caps"][row, slot] = max(allowed) if allowed else -1
        if len(groups) > 1:
            for k, group in enumerate(groups):
                group = set(group)
                arrays["cut_active"][row, k] = True
                arrays["cuts"][row, k] = [
                    bool(
                        active[row, i]
                        and active[row, j]
                        and ((i in group) != (j in group))
                    )
                    for i, j in upper
                ]
        extra_record = (
            None
            if extra is None
            else {k: v for k, v in extra.items() if k != "witness"}
        )
        if extra_record is not None and extra["witness"] is not None:
            from dataclasses import asdict

            extra_record["witness"] = asdict(extra["witness"])
        records.append(
            {
                "candidate_id": candidate_id,
                "raw_sha256": graph_identity(*raw, spec),
                "probability_sha256": probability_identity(p, spec),
                "base": base,
                "extra": extra_record,
                "route": int(arrays["route"][row]),
                "component_count": len(groups),
                "unsupported_local_atoms": int(
                    np.sum(active[row] & (arrays["caps"][row] < 0))
                ),
            }
        )
        if structural is not None:
            records[-1]["structural"] = structural
            records[-1]["beam_search_performed"] = beam_search_performed
        if common_organic_scope:
            records[-1]["scope_rejections"] = scope_rejections
        if exchange_result is not None:
            records[-1]["atom_exchange"] = {
                "baseline_status": exchange_result["baseline"]["status"],
                "substitutions_tested": exchange_result["substitutions_tested"],
                "selected": structural.get("exchange"),
            }
        if projection is not None:
            # Solver timing is diagnostic and must stay outside target identity.
            projection_time = projection.pop("seconds", 0.)
            records[-1]["repair_proposal"] = projection
        else:
            projection_time = 0.
        timings.append((original_seconds - projection_time, additional_seconds + projection_time))
    receipt = {
        "version": "mosaic-prepared-failure-guidance-v1",
        "schema_sha256": spec.fingerprint(),
        "rule_version": RULE_VERSION,
        "source_update": update,
        "candidate_ids": ids,
        "attempted_rows": b,
        "search": search.policy(),
        "progress_search": None
        if progress_search is None
        else progress_search.record(),
        "rows": records,
        "routes": {
            "0": "already_valid",
            "1": "original_valid",
            "2": "extra_valid",
            "3": "invalid_partial_and_fallback",
            "4": "no_witness_fallback",
        },
        "original_status_counts": dict(
            Counter(row["base"]["status"] for row in records)
        ),
        "failed_rows": sum(
            row["base"]["status"] in ("censored", "unsupported") for row in records
        ),
        "fallback_rows": int(np.count_nonzero(np.isin(arrays["route"], (3, 4)))),
        "original_valid_targets": int(np.count_nonzero(arrays["route"] == 1)),
        "extra_valid_targets": int(np.count_nonzero(arrays["route"] == 2)),
        "invalid_partial_targets": int(np.count_nonzero(arrays["route"] == 3)),
        "original_checks": sum(row["base"]["checks"] for row in records),
        "additional_checks": sum(
            0 if row["extra"] is None else row["extra"]["checks"] for row in records
        ),
        "normalization": {
            "complete": "N+K+K*(K-1)/2, then all scheduled rows",
            "partial": "changed unordered bonds, then all rows",
            "local": "supported risks / all hard occupied atoms, then all rows",
            "cuts": "original component cuts, then all rows",
            "nonempty": "empty fallback rows / all rows",
        },
    }
    if severity_progress is not None:
        receipt["severity_progress"] = severity_progress.record()
    if structural_repair is not None:
        receipt["structural_repair"] = structural_repair.record()
        receipt["structural_checks"] = sum(row["structural"]["checks"] for row in records)
        receipt["beam_checks"] = sum(row["base"]["checks"] for row in records if row["beam_search_performed"])
    if common_organic_scope:
        receipt["common_organic_scope"] = {"version": SCOPE_VERSION, "atom_exchange": atom_exchange}
    payload = {k: v.tolist() for k, v in arrays.items()}
    identity = digest({"arrays": payload, "receipt": receipt})
    materialized = {k: jax.numpy.asarray(v) for k, v in arrays.items()}
    jax.block_until_ready(materialized)
    return PreparedFailureGuidance(
        materialized, receipt, identity, time.perf_counter() - started, tuple(timings)
    )
