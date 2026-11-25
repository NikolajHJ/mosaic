"""Weak batch supervision for complete motifs in raw categorical graph draws.

The two indicators count a *valid molecule* with an aromatic ring or an amide
respectively. Each draw counts once per group. An invalid draw counts in the
denominator and scores zero for both groups. This makes discarding valid
molecules without the desired group an ineffective way to improve the yield.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from latentmol.chemistry import inspect_graph
from latentmol.guidance.motif_completion import AMIDE, completion_loss
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob, sample_hard_graph


def _motif_labels(graphs: HardGraphBatch, spec: GraphSpec, *, failures=None, ring_rows=None):
    """Move each integer tensor across devices once and sanitize with the shared evaluator."""
    arrays = [x.detach().cpu().numpy() for x in (graphs.atoms, graphs.charges, graphs.bonds)]
    labels = np.zeros((len(arrays[0]), 2), dtype=bool)
    valid = 0
    for row, (atoms, charges, bonds) in enumerate(zip(*arrays)):
        result = inspect_graph(atoms, charges, bonds, spec)
        if result.smiles is None:
            continue
        mol = Chem.MolFromSmiles(result.smiles)
        if mol is None:
            if failures is None:
                raise ValueError("Could not read a sanitized graph emitted by the chemistry evaluator")
            failures["smiles_roundtrip"] += 1
            continue
        valid += 1
        labels[row] = (rdMolDescriptors.CalcNumAromaticRings(mol) > 0, mol.HasSubstructMatch(AMIDE))
        if ring_rows is not None:
            rings = mol.GetRingInfo().AtomRings()
            ring_rows[row] = (mol.GetNumHeavyAtoms(), bool(rings), any(len(r) == 3 for r in rings),
                              any(len(r) >= 8 for r in rings))
    return labels, valid


def _batch_shortfall(labels: np.ndarray, floors: tuple[float, float]):
    """Return observed penalty and score-function coefficients for its expectation.

    With B independent draws and two motifs, the loss is the mean over motifs
    of 0.5 * max(floor - hits/B, 0)^2. Conditional on all other labels, each
    motif's penalty is an affine function of row i's binary indicator y_i.
    The slope is penalty_if_present - penalty_if_absent. Multiply the slope
    by y_i minus the leave-one-out frequency (an action-independent baseline)
    and by grad log q(G_i). Summing rows gives an unbiased gradient of the
    *expected finite-batch* penalty; the slope already includes batch scaling.
    This is distinct from applying the hinge to the infinite-population mean.
    """
    batch = len(labels)
    totals = labels.sum(axis=0)
    others = totals - labels
    floor = np.asarray(floors)

    def penalty(hits):
        return 0.5 * np.maximum(floor - hits / batch, 0.0) ** 2

    slope = penalty(others + 1) - penalty(others)
    centered = labels - others / (batch - 1)
    coefficients = slope * centered
    stats = {"draws": batch, "penalty": float(penalty(totals).mean())}
    for column, name in enumerate(("aromatic", "amide")):
        stats[f"{name}_hits"] = int(totals[column])
        stats[f"{name}_active_rows"] = int(np.count_nonzero(slope[:, column]))
        stats[f"{name}_informative"] = bool(np.any(coefficients[:, column]))
    return coefficients.mean(axis=1), stats


def _score_surrogate(decoded, sampled, spec, weights):
    """Zero scalar value with the sampled graph's weighted score gradient."""
    if not np.any(weights):
        return 0.0 * (decoded.atoms[0, 0, 0] + decoded.charges[0, 0, 0] + decoded.bonds[0, 0, 0, 0])
    log_prob = hard_graph_log_prob(decoded, sampled, spec)
    if not torch.isfinite(log_prob).all():
        raise FloatingPointError("A sampled functional-group graph has nonfinite log probability")
    score = torch.sum(log_prob * log_prob.new_tensor(weights))
    return score - score.detach()


@dataclass
class PooledMotifScore:
    """Detached draws and leave-one-out coefficients for one logical frequency batch."""

    samples: dict[int, HardGraphBatch]
    labels: np.ndarray
    weights: np.ndarray
    stats: dict

    def surrogate(self, start, decoded, spec):
        graph = self.samples[start]
        n = len(graph.atoms)
        visible = DecoderOutput(decoded.atoms[:n], decoded.charges[:n], decoded.bonds[:n])
        return _score_surrogate(visible, graph, spec, self.weights[start:start + n])


@torch.no_grad()
def prepare_pooled_motif_score(model, prior, policy, microbatch, generator):
    """Sample before backward, so all chunks use the same finite-batch objective.

    This decoder uses stateless normalization and training-mode dropout. Fork its
    RNG for the no-grad pass so the ordinary gradient pass replays the dropout
    masks at the same physical batch shapes. Graph draws use a private generator.
    Only hard labels and coefficients survive the prepass; no decoder activations
    from an earlier microbatch are needed for a later one.
    """
    count = policy["score_draws"]
    if not 2 <= count <= len(prior) or microbatch < 1:
        raise ValueError("Pooled motif draws must fit the logical prior batch")
    samples, labels = {}, []
    valid_total = 0
    failures = {"smiles_roundtrip": 0}
    ring_policy = policy.get("ring_size")
    ring_rows = None
    if ring_policy is not None:
        from latentmol.guidance.ring_size import ring_size_score, validate_ring_policy

        validate_ring_policy(ring_policy)
        ring_rows = np.zeros((count, 4), dtype=np.int32)
    with torch.random.fork_rng(devices=[prior.device] if prior.is_cuda else []):
        for start in range(0, count, microbatch):
            decoded = model.decode(prior[start:start + microbatch])
            n = min(count - start, len(decoded.atoms))
            visible = DecoderOutput(decoded.atoms[:n], decoded.charges[:n], decoded.bonds[:n])
            sampled = sample_hard_graph(visible, model.spec, generator=generator)
            rows, valid = _motif_labels(sampled, model.spec, failures=failures,
                                       ring_rows=None if ring_rows is None else ring_rows[start:start+n])
            samples[start] = sampled
            labels.append(rows)
            valid_total += valid
    labels = np.concatenate(labels, axis=0)
    weights, stats = _batch_shortfall(labels, (policy["aromatic_floor"], policy["amide_floor"]))
    stats.update(failures, valid=valid_total)
    if ring_rows is not None:
        ring_weights, penalty, ring_stats = ring_size_score(ring_rows)
        weight = ring_policy["weight"]
        weights += weight * ring_weights
        stats.update(ring_stats, motif_shortfall_penalty=stats["penalty"], ring_size_penalty=penalty,
                     ring_size_weight=weight, penalty=stats["penalty"] + weight * penalty)
    return PooledMotifScore(samples, labels, weights, stats)


def functional_group_loss(decoded: DecoderOutput, spec: GraphSpec, generator: torch.Generator,
                          *, aromatic_floor: float, amide_floor: float, completion_weight: float = 0.,
                          ring_encoding: str = "aromatic", amide_relative_weight: float = 1.,
                          completion_mode: str = "absolute", count_roundtrip_failures: bool = False,
                          pooled_score: PooledMotifScore | None = None):
    """Score one graph per selected prior, reusing existing live decoder outputs.

    Input rows should be independent random vectors. Floors refer to yield
    among *all* selected draws, not fractional composition among the valid
    subset. The caller selects the draw budget, independent RNG, cadence and
    scalar weight. The returned value is the observed nonnegative shortfall;
    its derivative is a Monte Carlo score-function estimate at temperature 1.
    Chemistry and all leave-one-out weights are detached. With zero motif
    hits, this frequency estimator gives no information for that motif.
    Optional completion adds local bond-type cross-entropy on validated near
    misses in argmax decodes of the same latent vectors. When the trainer
    supplies a pooled score, this call only returns its observed penalty and
    optional local completion. The trainer adds `pooled_score.surrogate(...)`
    once per chunk to supply the frequency gradient on all sampled priors.
    """
    floors = (aromatic_floor, amide_floor)
    if not all(math.isfinite(floor) and 0 <= floor <= 1 for floor in floors):
        raise ValueError("Functional-group yield floors must be finite fractions in [0, 1]")
    if not math.isfinite(completion_weight) or completion_weight < 0:
        raise ValueError("Local motif completion weight must be finite and nonnegative")
    if completion_mode not in ("absolute", "pattern") or (completion_mode == "pattern" and ring_encoding != "aromatic"):
        raise ValueError("Invalid motif completion objective or aromatic encoding")
    if decoded.atoms.shape[0] < 2:
        raise ValueError("Functional-group batch supervision requires at least two draws")
    if pooled_score is None:
        sampled = sample_hard_graph(decoded, spec, generator=generator)
        failures = {"smiles_roundtrip": 0} if completion_weight or count_roundtrip_failures else None
        labels, valid = _motif_labels(sampled, spec, failures=failures)
        weights, stats = _batch_shortfall(labels, floors)
        stats.update(failures or {}, valid=valid)
        objective = stats["penalty"] + _score_surrogate(decoded, sampled, spec, weights)
        site_stats = stats
    else:
        # The trainer applies the pooled score gradient once per physical chunk.
        # Charge the observed penalty only once. Local completions continue to
        # use their own original batch of candidate sites and sample counts.
        stats = dict(pooled_score.stats, local_draws=len(decoded.atoms))
        objective = stats["penalty"] + decoded.bonds[0, 0, 0, 0] * 0.0
        if completion_weight:
            _, site_stats = _batch_shortfall(pooled_score.labels[:len(decoded.atoms)], floors)
    if completion_weight:
        active = dict(need_ring=site_stats["aromatic_hits"] < aromatic_floor * site_stats["draws"],
                      need_amide=site_stats["amide_hits"] < amide_floor * site_stats["draws"])
        if completion_mode == "pattern":
            from latentmol.guidance.motif_patterns import pattern_completion_loss

            completion, observations = pattern_completion_loss(decoded, spec,
                aromatic_needed=max(0, math.ceil(aromatic_floor * site_stats["draws"]) - site_stats["aromatic_hits"]),
                amide_needed=max(0, math.ceil(amide_floor * site_stats["draws"]) - site_stats["amide_hits"]),
                amide_relative_weight=amide_relative_weight)
            stats["completion_pattern_loss"] = float(completion.detach())
        else:
            completion, observations = completion_loss(decoded, spec, **active, ring_encoding=ring_encoding,
                                                       amide_relative_weight=amide_relative_weight)
            stats["completion_ce"] = float(completion.detach())
        stats.update(observations)
        objective = objective + completion_weight * completion
    return objective, stats
