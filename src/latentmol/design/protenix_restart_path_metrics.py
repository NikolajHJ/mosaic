"""Measure exact molecular path reuse and structural retention in restart studies."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from statistics import mean, median

import numpy as np
from rdkit import DataStructs

from latentmol.design.protenix_restart_selection import _morgan_fingerprints


def read_path(path):
    """Load one observation per saved update from a branch or flow candidate file."""
    observations = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "kind" in row and row["kind"] != "argmax":
                continue
            step = row["step"]
            if type(step) is not int or step < 0 or step in observations:
                raise ValueError(f"Invalid or duplicate trajectory step {step!r} in {path}")
            valid = row["raw_valid"] if "raw_valid" in row else row["valid"]
            smiles = row["smiles"]
            observations[step] = smiles if valid and smiles and "." not in smiles else None
    return observations


def read_hard_path(directory):
    """Fingerprint exact slot-aligned labels, including saved invalid graphs."""
    keys = {}
    for path in sorted(directory.glob("states-*.npz")):
        with np.load(path, allow_pickle=False) as block:
            steps = block["steps"]
            arrays = [block[name] for name in ("atoms", "charges", "bonds_upper")]
            if any(len(array) != len(steps) for array in arrays):
                raise ValueError(f"Invalid raw trajectory block: {path}")
            for step, atoms, charges, bonds in zip(steps, *arrays):
                step = int(step)
                if step in keys:
                    raise ValueError(f"Duplicate raw trajectory step {step}: {path}")
                keys[step] = hashlib.blake2b(
                    atoms.tobytes() + charges.tobytes() + bonds.tobytes(), digest_size=16).digest()
    return keys


@dataclass(frozen=True)
class PathTrace:
    name: str
    arm: str
    parent_id: str
    parent_smiles: str
    observations: dict[int, str | None]
    source_observations: dict[int, str | None] | None = None
    source_step: int | None = None
    source_valid_smiles: set[str] | None = None
    expected_updates: int | None = None
    hard_observations: dict[int, bytes] | None = None

    @property
    def end_step(self):
        return self.expected_updates if self.expected_updates is not None else max(self.observations, default=0)


def _fraction(count, denominator):
    return count / denominator if denominator else None


def _matches(left, right, *, first=1, last=None):
    common = [step for step in left.keys() & right.keys() if step >= first and (last is None or step <= last)]
    valid = [step for step in common if left[step] is not None and right[step] is not None]
    identical = sum(left[step] == right[step] for step in valid)
    different = min((step for step in valid if left[step] != right[step]), default=None)
    rejoin = (min((step for step in valid if step > different and left[step] == right[step]), default=None)
              if different is not None else None)
    return dict(observed=len(common), jointly_valid=len(valid), identical=identical,
                fraction=_fraction(identical, len(valid)),
                first_different_step=different, first_rejoin_step=rejoin)


def _revisits(observations, source):
    if source is None:
        return None
    hits = sum(smiles in source for smiles in observations)
    return dict(matches=hits, valid=len(observations), fraction=_fraction(hits, len(observations)))


def _overlap(left, right):
    shared = len(left & right)
    return dict(left=len(left), right=len(right), shared=shared,
                jaccard=_fraction(shared, len(left | right)) if left and right else None)


def _valid_set(path, first=1, last=None):
    return {smiles for step, smiles in path.items()
            if smiles is not None and step >= first and (last is None or step <= last)}


def _similarity(fingerprints, left, right):
    if left is None or right is None:
        return None
    return DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right])


def _late_sample(trace, horizon, late_steps, limit=32):
    first = max(1, horizon - late_steps + 1)
    distinct = list(dict.fromkeys(trace.observations[step] for step in sorted(trace.observations)
                                 if first <= step <= horizon
                                 and trace.observations[step] not in (None, trace.parent_smiles)))
    if len(distinct) <= limit:
        return distinct
    return [distinct[round(i * (len(distinct) - 1) / (limit - 1))] for i in range(limit)]


def _nearest_similarity(left, right, fingerprints):
    if not left or not right:
        return None
    a = [fingerprints[smiles] for smiles in left]
    b = [fingerprints[smiles] for smiles in right]
    return (median(max(DataStructs.BulkTanimotoSimilarity(fp, b)) for fp in a)
            + median(max(DataStructs.BulkTanimotoSimilarity(fp, a)) for fp in b)) / 2


def _branch_record(trace, fingerprints, late_steps):
    rows = trace.observations
    end = trace.end_step
    post_start = [smiles for step, smiles in rows.items() if step > 0 and smiles is not None]
    start, endpoint = rows.get(0), rows.get(end)
    late_first = max(1, end - late_steps + 1)
    late = [smiles for step, smiles in rows.items() if step >= late_first and smiles is not None]
    last_valid_step = max((step for step, smiles in rows.items() if smiles is not None), default=None)
    source = trace.source_observations
    source_set = trace.source_valid_smiles
    aligned_source = None
    if source is not None:
        source_set = _valid_set(source, first=0) if source_set is None else source_set
        if trace.source_step is not None:
            source_continuation = {step - trace.source_step: smiles for step, smiles in source.items()
                                   if step > trace.source_step}
            aligned_source = _matches(rows, source_continuation, last=end)
    nonparent = [smiles for smiles in post_start if smiles != trace.parent_smiles]
    nonparent_revisit = _revisits(nonparent, source_set)
    return dict(name=trace.name, arm=trace.arm, parent_id=trace.parent_id, parent_smiles=trace.parent_smiles,
                observed=len(rows), valid=sum(smiles is not None for smiles in rows.values()),
                post_start_valid=len(post_start), distinct_post_start=len(set(post_start)),
                late_valid=len(late), late_distinct=len(set(late)),
                start_smiles=start, start_parent_similarity=_similarity(fingerprints, trace.parent_smiles, start),
                endpoint_step=end, endpoint_observed=end in rows, endpoint_smiles=endpoint,
                endpoint_parent_similarity=_similarity(fingerprints, trace.parent_smiles, endpoint),
                last_valid_step=last_valid_step, last_valid_smiles=rows.get(last_valid_step),
                parent_revisit=_revisits(post_start, {trace.parent_smiles}),
                source_revisit=_revisits(post_start, source_set),
                source_nonparent_revisit=nonparent_revisit,
                source_distinct_nonparent_revisit=_revisits(set(nonparent), source_set),
                source_aligned_continuation=aligned_source)


def _pair_record(left, right, fingerprints, late_steps, late_samples):
    a, b = left.observations, right.observations
    horizon = min(left.end_step, right.end_step)
    early_last = min(25, horizon)
    late_first = max(1, horizon - late_steps + 1)
    a_all, b_all = _valid_set(a, last=horizon), _valid_set(b, last=horizon)
    a_end, b_end = a.get(horizon), b.get(horizon)
    valid_endpoints = a_end is not None and b_end is not None
    hard_match = None
    if left.hard_observations is not None and right.hard_observations is not None:
        hard_match = _matches(left.hard_observations, right.hard_observations, last=horizon)
        del hard_match["jointly_valid"]
    return dict(left=left.name, right=right.name, left_arm=left.arm, right_arm=right.arm,
                same_parent=left.parent_id == right.parent_id, shared_horizon=horizon,
                trajectory_overlap=_overlap(a_all, b_all),
                nonparent_overlap=_overlap(a_all - {left.parent_smiles}, b_all - {right.parent_smiles}),
                early_overlap=_overlap(_valid_set(a, last=early_last), _valid_set(b, last=early_last)),
                late_overlap=_overlap(_valid_set(a, late_first, horizon), _valid_set(b, late_first, horizon)),
                aligned=_matches(a, b, last=horizon), early_aligned=_matches(a, b, last=early_last),
                late_aligned=_matches(a, b, first=late_first, last=horizon),
                aligned_hard_labels=hard_match,
                late_nearest_tanimoto=_nearest_similarity(late_samples[left.name, horizon],
                                                          late_samples[right.name, horizon], fingerprints),
                endpoint=dict(observed=horizon in a and horizon in b, jointly_valid=valid_endpoints,
                              identical=(a_end == b_end) if valid_endpoints else None,
                              tanimoto=_similarity(fingerprints, a_end, b_end)))


def _average(values):
    measured = [value for value in values if value is not None]
    return dict(mean=mean(measured) if measured else None, count=len(measured))


def _aligned_total(records):
    measured = [record for record in records if record is not None]
    total = {name: sum(record[name] for record in measured) for name in ("observed", "jointly_valid", "identical")}
    diverged = sum(row["first_different_step"] is not None for row in measured)
    rejoined = sum(row["first_rejoin_step"] is not None for row in measured)
    return dict(total, measured=len(measured), fraction=_fraction(total["identical"], total["jointly_valid"]),
                diverged=diverged, rejoined=rejoined, rejoin_fraction=_fraction(rejoined, diverged))


def _revisit_total(records):
    measured = [record for record in records if record is not None]
    hits = sum(record["matches"] for record in measured)
    valid = sum(record["valid"] for record in measured)
    return dict(matches=hits, valid=valid, measured=len(measured), fraction=_fraction(hits, valid))


def _pair_summary(records):
    endpoints = [record["endpoint"] for record in records]
    jointly_valid = sum(row["jointly_valid"] for row in endpoints)
    identical = sum(row["identical"] is True for row in endpoints)
    raw = [row["aligned_hard_labels"] for row in records if row["aligned_hard_labels"] is not None]
    raw_observed = sum(row["observed"] for row in raw)
    raw_equal = sum(row["identical"] for row in raw)
    return dict(pairs=len(records),
                trajectory_jaccard=_average(row["trajectory_overlap"]["jaccard"] for row in records),
                nonparent_jaccard=_average(row["nonparent_overlap"]["jaccard"] for row in records),
                early_jaccard=_average(row["early_overlap"]["jaccard"] for row in records),
                late_jaccard=_average(row["late_overlap"]["jaccard"] for row in records),
                late_nearest_tanimoto=_average(row["late_nearest_tanimoto"] for row in records),
                aligned=_aligned_total(row["aligned"] for row in records),
                early_aligned=_aligned_total(row["early_aligned"] for row in records),
                late_aligned=_aligned_total(row["late_aligned"] for row in records),
                aligned_hard_labels=dict(observed=raw_observed, identical=raw_equal,
                                         fraction=_fraction(raw_equal, raw_observed)),
                endpoint=dict(observed=sum(row["observed"] for row in endpoints), jointly_valid=jointly_valid,
                              identical=identical, fraction=_fraction(identical, jointly_valid),
                              tanimoto=_average(row["tanimoto"] for row in endpoints)))


def summarize_paths(traces, *, late_steps=25, reference_arm="untouched"):
    """Compare within-arm paths and compare sibling arms against one reference."""
    if type(late_steps) is not int or late_steps < 1:
        raise ValueError("late_steps must be a positive integer")
    traces = list(traces)
    if len({trace.name for trace in traces}) != len(traces):
        raise ValueError("Trajectory names must be unique")
    pairs = [(left, right) for left, right in combinations(traces, 2)
             if left.arm == right.arm or (left.parent_id == right.parent_id
                                          and reference_arm in (left.arm, right.arm))]
    fingerprint_smiles = {smiles for trace in traces for smiles in (
        trace.parent_smiles, trace.observations.get(0), trace.observations.get(trace.end_step)) if smiles}
    late_samples = {}
    for left, right in pairs:
        horizon = min(left.end_step, right.end_step)
        for trace in (left, right):
            key = (trace.name, horizon)
            if key not in late_samples:
                late_samples[key] = _late_sample(trace, horizon, late_steps)
                fingerprint_smiles.update(late_samples[key])
            if endpoint := trace.observations.get(horizon):
                fingerprint_smiles.add(endpoint)
    fingerprints = dict(_morgan_fingerprints(sorted(fingerprint_smiles)))
    branches = [_branch_record(trace, fingerprints, late_steps) for trace in traces]
    pair_records = [_pair_record(left, right, fingerprints, late_steps, late_samples) for left, right in pairs]
    by_arm = defaultdict(list)
    pair_groups = defaultdict(list)
    for row in branches:
        by_arm[row["arm"]].append(row)
    for row in pair_records:
        left, right = row["left_arm"], row["right_arm"]
        if left == right:
            pair_groups[left, "same_parent" if row["same_parent"] else "different_parent"].append(row)
        else:
            pair_groups[right if left == reference_arm else left, "reference"].append(row)
    arms = {}
    for arm, rows in by_arm.items():
        valid = sum(row["valid"] for row in rows)
        observed = sum(row["observed"] for row in rows)
        observed_ends = sum(row["endpoint_observed"] for row in rows)
        valid_ends = sum(row["endpoint_smiles"] is not None for row in rows)
        arms[arm] = dict(branches=len(rows), parents=len({row["parent_id"] for row in rows}),
            validity=dict(valid=valid, observed=observed, fraction=_fraction(valid, observed)),
            endpoint_validity=dict(valid=valid_ends, observed=observed_ends,
                                   fraction=_fraction(valid_ends, observed_ends)),
            start_parent_similarity=_average(row["start_parent_similarity"] for row in rows),
            endpoint_parent_similarity=_average(row["endpoint_parent_similarity"] for row in rows),
            distinct_post_start=_average(row["distinct_post_start"] for row in rows),
            parent_revisit=_revisit_total(row["parent_revisit"] for row in rows),
            source_revisit=_revisit_total(row["source_revisit"] for row in rows),
            source_nonparent_revisit=_revisit_total(row["source_nonparent_revisit"] for row in rows),
            source_distinct_nonparent_revisit=_revisit_total(row["source_distinct_nonparent_revisit"] for row in rows),
            source_aligned_continuation=_aligned_total(row["source_aligned_continuation"] for row in rows),
            **{group: _pair_summary(pair_groups[arm, group])
               for group in ("same_parent", "different_parent", "reference")})
    return dict(late_steps=late_steps, reference_arm=reference_arm, arms=arms, branches=branches, pairs=pair_records)
