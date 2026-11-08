"""Select score-ranked restart points subject to hard diversity constraints."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from rdkit import DataStructs
import torch

from latentmol.design.protenix_restart_selection import _morgan_fingerprints
from latentmol.design.protenix_restarts import Candidate


@dataclass(frozen=True)
class SearchPoint:
    candidate: Candidate
    active: torch.Tensor
    fresh_loss: float | None
    ancestry: tuple[str, ...]
    repair_edits: int | None = None
    repair_smiles: str | None = None
    repair_target_sha256: str | None = None


def _source_wave(point):
    for part in reversed(point.candidate.branch.split("/")):
        if match := re.fullmatch(r"(?:wave|cycle)-(\d+)", part):
            return int(match[1])
    return -1


def _ancestor(point, lookback):
    if not point.ancestry:
        raise ValueError("A search point must have a founder or branch ancestry")
    # A new child adds one branch edge after the source checkpoint.
    return point.ancestry[max(0, len(point.ancestry) - 1 - lookback)]


def _family_capacity(wave, slots, children, step_gap, lookback, fraction):
    if (any(type(value) is not int for value in (wave, slots, children, step_gap, lookback))
            or wave < 0 or slots < 1 or children < 1 or step_gap < 0 or lookback < 0
            or (fraction is not None and (isinstance(fraction, bool) or not 0 < fraction <= 1))):
        raise ValueError("Invalid frontier size, spacing or ancestry constraints")
    return slots * children if fraction is None else math.floor(slots * children * fraction)


def _time_allowed(point, wave, history, min_step_gap):
    candidate = point.candidate
    return (_source_wave(point) < wave and
            all(abs(candidate.step - step) >= min_step_gap
                for step in history.get(candidate.branch, ())))


def _rank(point, loss):
    candidate = point.candidate
    if not math.isfinite(loss):
        raise ValueError("Nonfinite frontier candidate score")
    return (-_source_wave(point), loss, candidate.branch, candidate.step, candidate.smiles or "")


def select_frontier(points, *, wave, slots, children_per_parent, history, min_step_gap,
                    max_similarity, ancestor_lookback, max_ancestor_fraction):
    """Pick up to slots valid parents without relaxing time, graph or ancestry limits."""
    capacity = _family_capacity(wave, slots, children_per_parent, min_step_gap,
                                ancestor_lookback, max_ancestor_fraction)
    if isinstance(max_similarity, bool) or not 0 <= max_similarity <= 1:
        raise ValueError("Invalid frontier Morgan similarity limit")
    ranked = sorted(
        (point for point in points if point.candidate.smiles and point.fresh_loss is not None
         and _time_allowed(point, wave, history, min_step_gap)),
        key=lambda point: _rank(point, point.fresh_loss),
    )
    chosen = []
    branches = set()
    molecules = set()
    families = Counter()
    fingerprints = []
    for point in ranked:
        candidate = point.candidate
        family = _ancestor(point, ancestor_lookback)
        if (candidate.branch in branches or candidate.smiles in molecules
                or families[family] + children_per_parent > capacity):
            continue
        _, fingerprint = next(_morgan_fingerprints([candidate.smiles]))
        if any(similarity > max_similarity
               for similarity in DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)):
            continue
        chosen.append(point)
        branches.add(candidate.branch)
        molecules.add(candidate.smiles)
        families[family] += children_per_parent
        fingerprints.append(fingerprint)
        if len(chosen) == slots:
            break
    return chosen


def select_invalid_frontier(points, *, wave, selected, slots, children_per_parent, history,
                            min_step_gap, ancestor_lookback, max_ancestor_fraction,
                            max_repair_edits, replacing_parent=None):
    """Pick one bounded-repair invalid start under the remaining branch and ancestry budget."""
    capacity = _family_capacity(wave, slots, children_per_parent, min_step_gap,
                                ancestor_lookback, max_ancestor_fraction)
    if type(max_repair_edits) is not int or max_repair_edits < 1:
        raise ValueError("Invalid frontier repair edit limit")
    families = Counter()
    branches = set()
    replacement_found = False
    for point in selected:
        children = children_per_parent
        if replacing_parent is point:
            if replacement_found:
                raise ValueError("Replacement parent appears more than once")
            replacement_found = True
            children -= 1
        if children:
            families[_ancestor(point, ancestor_lookback)] += children
            branches.add(point.candidate.branch)
    if replacing_parent is not None and not replacement_found:
        raise ValueError("Replacement parent must be one of the selected points")
    if sum(families.values()) >= slots * children_per_parent:
        return None
    eligible = (
        point for point in points
        if point.candidate.smiles is None and type(point.repair_edits) is int
        and 1 <= point.repair_edits <= max_repair_edits
        and point.candidate.branch not in branches and _time_allowed(point, wave, history, min_step_gap)
        and families[_ancestor(point, ancestor_lookback)] < capacity
    )
    return min(eligible, key=lambda point: _rank(point, point.candidate.relaxed_loss), default=None)
