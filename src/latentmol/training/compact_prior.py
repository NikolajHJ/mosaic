"""Prior guidance for the compact VAE. The structural teacher edits bonds only."""

from collections import deque
import math

import torch
from torch.nn import functional as F

from latentmol.guidance.repair_proposals import ProposalRepairTeacher, RepairProposalConfig
from latentmol.probability_repair import ProbabilityRepairConfig


def atom_count_probabilities(atom_logits):
    """Per-graph P(count = k), k=0..N, under the decoder's categorical slot law."""
    if atom_logits.ndim != 3 or atom_logits.shape[-1] < 2 or atom_logits.shape[1] < 1:
        raise ValueError("Expected atom logits shaped [batch, slots, categories including NO_ATOM]")
    probability = atom_logits.softmax(dim=-1)
    distribution = probability.new_ones((len(probability), 1))
    for slot in probability.unbind(dim=1):
        # Summing real probabilities retains their gradients even for very rare occupancy.
        stay = distribution * slot[:, :1]
        grow = distribution * slot[:, 1:].sum(dim=1, keepdim=True)
        distribution = F.pad(stay, (0, 1)) + F.pad(grow, (1, 0))
    return distribution


def log_atom_count_probabilities(atom_logits):
    """Log P(K=k) for every possible occupied-slot count, even at saturated logits."""
    if atom_logits.ndim != 3 or min(atom_logits.shape[:2]) < 1 or atom_logits.shape[-1] < 2:
        raise ValueError("Expected nonempty [batch, slots, NO_ATOM and real categories]")
    if not atom_logits.is_floating_point():
        raise ValueError("Atom logits must be floating point")
    if atom_logits.dtype in (torch.float16, torch.bfloat16):
        atom_logits = atom_logits.float()
    occupancy = torch.logsumexp(atom_logits[..., 1:], dim=-1) - atom_logits[..., 0]
    log_empty, log_real = F.logsigmoid(-occupancy), F.logsigmoid(occupancy)
    distribution = occupancy.new_zeros((len(occupancy), 1))
    for empty, real in zip(log_empty.unbind(1), log_real.unbind(1)):
        stay = distribution + empty[:, None]
        grow = distribution + real[:, None]
        # At the edges exactly one transition is possible. Only combine reachable states.
        distribution = torch.cat((stay[:, :1], torch.logaddexp(stay[:, 1:], grow[:, :-1]),
                                  grow[:, -1:]), dim=1)
    return distribution


def count_kl_per_slot(log_pmfs, reference_pmf):
    """KL(train size distribution || fresh prior mixture), normalized by slot count.

    Mix the latents first so different latents may specialize in different sizes.
    Zero-frequency reference bins contribute zero. A minibatch estimates the mixture.
    """
    if (log_pmfs.ndim != 2 or not len(log_pmfs) or log_pmfs.shape[1] < 2
            or reference_pmf.shape != log_pmfs.shape[1:]):
        raise ValueError("Expected matching count distributions over 0..max_nodes")
    support = reference_pmf > 0
    reference = reference_pmf[support]
    log_mixture = torch.logsumexp(log_pmfs[:, support], dim=0) - math.log(len(log_pmfs))
    return (reference * (reference.log() - log_mixture)).sum() / (log_pmfs.shape[1] - 1)


def argmax_count_loss(atom_logits, reference_cdf, update):
    """Calibrate the actual atom argmax on fresh priors with train-size quantiles.

    Rank each graph by its soft count using the winning real element against
    NO_ATOM. Assign ordered empirical quantiles, then train the top K slots in
    each graph to beat NO_ATOM and the other slots to fall below it. Only the
    assignment is detached. No target size is stored for any latent vector.
    """
    if (atom_logits.ndim != 3 or min(atom_logits.shape[:2]) < 1 or atom_logits.shape[-1] < 2
            or reference_cdf.shape != atom_logits.shape[1:2] or update < 0):
        raise ValueError("Expected nonempty atom logits and a training CDF on counts 0..slots-1")
    scores = atom_logits if atom_logits.dtype in (torch.float32, torch.float64) else atom_logits.float()
    margins = scores[..., 1:].amax(dim=-1) - scores[..., 0]
    batch, nodes = margins.shape
    with torch.no_grad():
        graph_order = margins.sigmoid().sum(-1).argsort(stable=True)
        # Rotate the within-stratum quantile across updates so even rare training
        # sizes can be assigned in small batches. No training RNG is consumed.
        offset = (0.5 + update * ((math.sqrt(5) - 1) / 2)) % 1
        quantiles = (torch.arange(batch, dtype=reference_cdf.dtype, device=margins.device) + offset) / batch
        sizes = torch.searchsorted(reference_cdf, quantiles, right=True)
        labels = torch.arange(nodes, device=margins.device)[None, :] < sizes[:, None]
    sorted_margins = margins[graph_order].sort(dim=1, descending=True, stable=True).values
    return F.binary_cross_entropy_with_logits(sorted_margins, labels.to(margins.dtype))


def _count_reference(histogram, max_nodes, device):
    """Create PMF and CDF on CPU from the real training-split counts."""
    counts = [0] * (max_nodes + 1)
    for key, frequency in histogram.items():
        size = int(key)
        if not 1 <= size <= max_nodes or type(frequency) is not int or frequency < 0:
            raise ValueError("Invalid training heavy-atom count histogram")
        counts[size] = frequency
    total = sum(counts)
    if not total:
        raise ValueError("Empty training heavy-atom count histogram")
    frequencies = torch.tensor(counts, dtype=torch.float64)
    return ((frequencies / total).to(device=device, dtype=torch.float32),
            (frequencies.cumsum(0)[:-1] / total).to(device=device, dtype=torch.float32))


class CompactOccupancyGuidance:
    """Only the two count losses on a fresh Gaussian minibatch. No repair teacher."""

    def __init__(self, config, histogram, max_nodes, device):
        self.kl_weight = config["count_kl_weight"]
        self.argmax_weight = config["argmax_count_weight"]
        self.reference_pmf, self.reference_cdf = _count_reference(histogram, max_nodes, device)
        # Set from the completed optimizer update count before each forward,
        # including on resume. This keeps the quantile rotation continuous.
        self.update = 0

    def loss(self, model, prior_z):
        atoms = model.decode(prior_z).atoms
        kl = count_kl_per_slot(log_atom_count_probabilities(atoms), self.reference_pmf)
        margin = argmax_count_loss(atoms, self.reference_cdf, self.update)
        hard = atoms.detach().argmax(dim=-1).ne(0).sum(dim=1)
        return self.kl_weight * kl + self.argmax_weight * margin, {
            "count_kl_per_slot": float(kl.detach()), "count_kl_weight": self.kl_weight,
            "argmax_count_loss": float(margin.detach()), "argmax_count_weight": self.argmax_weight,
            "empty_graphs": int((hard == 0).sum()), "at_slot_limit": int((hard == atoms.shape[1]).sum()),
            "hard_count_mean": float(hard.float().mean()), "prior_points": len(prior_z),
        }


def repaired_bond_loss(decoded, targets, spec):
    """Mean conditional bond CE over occupied pairs per graph, then over all priors.

    Accepted targets have the same slot identities as their raw decoder draw.
    Already-valid and censored draws have no bond target. Only bond logits receive
    direct label supervision here. The shared decoder can still transmit gradients
    through its atom and pair states.
    """
    batch = len(decoded.atoms)
    if not batch or len(targets.targets) != batch or len(targets.candidate_ids) != batch:
        raise ValueError("Prior repair targets must match the sampled latent batch")
    rows, graphs = [], []
    for row, target in enumerate(targets.targets):
        if target is not None:
            target.verify(spec)
            if target.candidate_id != targets.candidate_ids[row]:
                raise ValueError("Prior repair target assigned to the wrong latent")
            rows.append(row)
            graphs.append(target.graph)
    if not graphs:
        return decoded.bonds.reshape(-1)[0] * 0
    scores = decoded.symmetric_bond_logits[rows]
    active = torch.cat([graph.atoms for graph in graphs]).to(scores.device) != 0
    labels = torch.cat([graph.bonds for graph in graphs]).to(scores.device)
    pairs = (active[:, :, None] & active[:, None, :]).triu(1)
    if not pairs.any(dim=(1, 2)).all():
        raise ValueError("A verified bond change requires two active atoms")
    costs = -scores.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    return ((costs * pairs).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2))).sum() / batch


class CompactPriorGuidance:
    """The count reference comes from train membership. Each benchmark arm owns its history."""

    def __init__(self, config, spec, device):
        self.spec = spec
        self.bond_weight = config["bond_weight"]
        self.count_weight = config["count_weight"]
        self.count_kl_weight = config.get("count_kl_weight", 0.0)
        self.argmax_count_weight = config.get("argmax_count_weight", 0.0)
        for weight in (self.bond_weight, self.count_weight, self.count_kl_weight, self.argmax_count_weight):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Prior loss weights must be finite and nonnegative")
        window = config["count_window"]
        if type(window) is not int or window < 1:
            raise ValueError("Count window must be a positive number of prior batches")
        self.reference_pmf, self.reference_cdf = _count_reference(config["count_histogram"], spec.max_nodes, device)
        bins = torch.arange(spec.max_nodes + 1, device=device)
        self.cdf_prefix = (bins[:, None] <= bins[None, :-1]).float()
        self.history = deque(maxlen=window - 1)
        self.teacher = ProposalRepairTeacher(
            spec, ProbabilityRepairConfig(max_edits=3, checks_per_depth=32, beam_width=4),
            proposals=RepairProposalConfig(), workers=config["workers"],
        )
        self.sequence = 0
        self.draws = 0

    def loss(self, model, prior_z):
        batch = len(prior_z)
        if batch < 1:
            raise ValueError("Expected at least one random latent")
        decoded = model.decode(prior_z)
        ids = range(self.draws, self.draws + batch)
        targets = self.teacher.build_targets(decoded, candidate_ids=ids, update=self.sequence)
        statuses = targets.receipt["status_counts"]
        if (sum(statuses.values()) != batch or statuses.get("rejected_or_error", 0)
                or targets.candidate_ids != tuple(ids)):
            raise RuntimeError("Prior repair teacher did not account for each sampled graph")
        bond = repaired_bond_loss(decoded, targets, self.spec)
        count_metrics = {}
        if self.count_kl_weight:
            log_pmfs = log_atom_count_probabilities(decoded.atoms)
            count_kl = count_kl_per_slot(log_pmfs, self.reference_pmf)
            current = log_pmfs.exp().sum(dim=0)
            count_metrics = dict(count_kl_per_slot=float(count_kl.detach()),
                                 count_kl_weight=self.count_kl_weight)
        else:
            current = atom_count_probabilities(decoded.atoms).sum(dim=0)
        # Past batches smooth the population estimate; their old graphs/gradients are discarded.
        total = current
        count = batch
        for previous, previous_count in self.history:
            total = total + previous
            count += previous_count
        cdf = total @ self.cdf_prefix.to(dtype=total.dtype) / count
        size = (cdf - self.reference_cdf).square().mean()
        loss = self.bond_weight * bond + self.count_weight * size
        if self.count_kl_weight:
            loss = loss + self.count_kl_weight * count_kl
        if self.argmax_count_weight:
            margin_loss = argmax_count_loss(decoded.atoms, self.reference_cdf, self.sequence)
            loss = loss + self.argmax_count_weight * margin_loss
            count_metrics.update(argmax_count_loss=float(margin_loss.detach()),
                                 argmax_count_weight=self.argmax_count_weight)
        if not loss.requires_grad or not bool(torch.isfinite(loss)):
            raise FloatingPointError("Prior objective must be attached and finite")
        self.history.append((current.detach(), batch))
        self.draws += batch
        self.sequence += 1
        hard_size = decoded.atoms.detach().argmax(dim=-1).ne(0).sum(dim=1)
        if self.argmax_count_weight or self.count_kl_weight:
            count_metrics.update(empty_graphs=int((hard_size == 0).sum()),
                                 hard_count_mean=float(hard_size.float().mean()))
        return loss, dict(prior_points=batch, accepted=targets.receipt["accepted_targets"],
                          prior_already_valid=statuses.get("skipped_already_valid", 0),
                          censored=statuses.get("censored", 0), bond_loss=float(bond.detach()),
                          count_cdf_loss=float(size.detach()), count_window_points=count,
                          at_slot_limit=int((hard_size == self.spec.max_nodes).sum()),
                          **count_metrics)

    def reset_panel(self):
        """Drop warmup targets and count history while retaining the live CPU workers."""
        self.teacher.clear_cache()
        self.history.clear()
        self.sequence = self.draws = 0

    def close(self):
        self.teacher.close()
