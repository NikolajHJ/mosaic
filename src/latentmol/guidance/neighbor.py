"""Explicit additive neighbor-event guidance for streamed VAE updates.

This adapter does not select targets, replace historical repair guidance or
change a production plan. It freezes the current raw decoder graph in the
existing no-grad prepass and adds the new bond-only event to the recomputed
training logits. CPU/GPU pilots must opt in by passing this object explicitly.
"""
from __future__ import annotations

from copy import deepcopy
import math

from latentmol.chemical_environment import prepare_neighbor_environment, neighbor_environment_loss
from latentmol.chemical_environment_policy import digest, validate_environment_policy


class NeighborEnvironmentGuidance:
    """One policy, separate real/prior center means, and checked chunk ownership.

    The primitive is a mean over assessed occupied centers, not molecules.
    Each chunk therefore receives its fraction of the complete branch's
    assessed centers. Weighting by its molecule fraction would silently change
    the objective when molecule sizes or microbatch sizes differ.
    """

    def __init__(self, spec, policy, branch_weights=None):
        self.spec = spec
        self.policy = validate_environment_policy(policy)
        weights = {"real": .25, "prior": .75} if branch_weights is None else branch_weights
        if (type(weights) is not dict or set(weights) != {"real", "prior"}
                or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                       for v in weights.values()) or sum(weights.values()) <= 0):
            raise ValueError("Require explicit finite nonnegative real/prior guidance weights")
        self.branch_weights = dict(weights)
        self._configuration_identity = digest(dict(schema=spec.fingerprint(), policy=self.policy,
                                                  branch_weights=self.branch_weights))
        self._phase = "new"

    def _verify_configuration(self):
        if self._configuration_identity != digest(dict(schema=self.spec.fingerprint(), policy=self.policy,
                                                       branch_weights=self.branch_weights)):
            raise ValueError("Additive guidance configuration changed during its lifecycle")

    def begin_update(self, *, update):
        self._verify_configuration()
        if type(update) is not int or update < 0:
            raise ValueError("Guidance update must be a nonnegative integer")
        if self._phase not in ("new", "complete"):
            raise ValueError("Previous additive guidance update was not completed")
        self.update = update
        self._chunks, self._used = {}, set()
        self._centers = {"real": 0, "prior": 0}
        self._unassessed = {"real": 0, "prior": 0}
        self._scope_excluded = {"real": 0, "prior": 0}
        self._local_violations = {"real": 0, "prior": 0}
        self._risk_sums = {"real": 0., "prior": 0.}
        self._weighted_loss = 0.
        self._phase = "preparing"

    def _key(self, branch, start, stop):
        self._verify_configuration()
        if (branch not in self.branch_weights or type(start) is not int
                or type(stop) is not int or start < 0 or stop <= start):
            raise ValueError("Require a valid named branch and nonempty chunk interval")
        return branch, start, stop

    def prepare(self, decoded, *, branch, start, stop):
        key = self._key(branch, start, stop)
        if self._phase != "preparing" or key in self._chunks:
            raise ValueError("Guidance context must be prepared once in its prepass")
        if len(decoded.atoms) != stop - start:
            raise ValueError("Guidance prepass chunk length changed")
        prepared = prepare_neighbor_environment(decoded, self.spec, self.policy)
        count = sum(int(active and cap >= 0)
                    for a, c in zip(prepared.context["active"], prepared.context["caps"])
                    for active, cap in zip(a, c))
        self._chunks[key] = prepared, count
        self._centers[branch] += count
        self._unassessed[branch] += sum(len(row["unassessed_slots"])
                                       for row in prepared.context["assessments"])
        self._scope_excluded[branch] += sum(not row["within_common_organic_scope"]
                                           for row in prepared.context["assessments"])
        self._local_violations[branch] += sum(not row["local_valence_ok"]
                                            for row in prepared.context["assessments"])

    def finalize_preparation(self):
        self._verify_configuration()
        if self._phase != "preparing":
            raise ValueError("Guidance prepass must finish before live evaluation")
        for branch in self.branch_weights:
            intervals = sorted((start, stop) for name, start, stop in self._chunks if name == branch)
            cursor = 0
            for start, stop in intervals:
                if start != cursor:
                    raise ValueError("Guidance branch intervals overlap or omit rows")
                cursor = stop
            if not intervals:
                raise ValueError("Both real and prior guidance contexts are required")
        self._phase = "evaluating"

    def loss(self, decoded, *, branch, start, stop):
        key = self._key(branch, start, stop)
        if self._phase != "evaluating" or key not in self._chunks or key in self._used:
            raise ValueError("Live guidance must use its original context exactly once")
        if len(decoded.atoms) != stop - start:
            raise ValueError("Live guidance chunk length changed")
        prepared, count = self._chunks[key]
        result = neighbor_environment_loss(decoded, prepared, self.spec)
        if result["assessed_centers"] != count:
            raise ValueError("Live event lost its prepared assessed-center denominator")
        denominator = self._centers[branch]
        center_fraction = count / denominator if denominator else 0.
        term = result["weighted_loss"] * self.branch_weights[branch] * center_fraction
        self._risk_sums[branch] += float(result["loss"].detach()) * center_fraction
        self._weighted_loss += float(term.detach())
        self._used.add(key)
        return term

    def summary(self):
        self._verify_configuration()
        if self._phase not in ("evaluating", "complete") or self._used != set(self._chunks):
            raise ValueError("Not every prepared guidance context was consumed")
        self._phase = "complete"
        chunks = [dict(branch=key[0], start=key[1], stop=key[2],
                       assessed_centers=count, context_identity=prepared.identity)
                  for key, (prepared, count) in sorted(self._chunks.items())]
        return dict(version="streamed-neighbor-environment-v1", policy=deepcopy(self.policy),
                    configuration_identity=self._configuration_identity,
                    update=self.update, branch_weights=dict(self.branch_weights),
                    assessed_centers=dict(self._centers), branch_mean_risk=dict(self._risk_sums),
                    unassessed_centers=dict(self._unassessed),
                    hard_scope_excluded_rows=dict(self._scope_excluded),
                    hard_local_violation_rows=dict(self._local_violations),
                    weighted_loss=self._weighted_loss, chunks=chunks,
                    contexts_sha256=digest(chunks),
                    normalization="separate real/prior means over all assessed occupied centers",
                    scope="explicit additive local-pilot term; historical repair/reference loss retained")
