"""Repeat raw prior and held-out reconstruction measurements on the same inputs."""

from collections import Counter
import json
import random
from statistics import mean, median

import torch

from latentmol.chemistry import inspect_graph
from latentmol.datasets.data import collate_graphs
from latentmol.representation import HardGraphBatch
from latentmol.sampling import sample_hard_graph


def _inspect(hard, spec):
    hard = hard.to("cpu")
    return [inspect_graph(a, c, b, spec) for a, c, b in zip(hard.atoms, hard.charges, hard.bonds)]


def _generated_roundtrip_metrics(model, parents, device, attempts, *, cycle_bins=False):
    """Evaluate exact chemical identity and count spurious/missing edges in canonical input slots."""
    eligible = valid = exact = extra = missing = 0
    bins = {label: dict(parents=0, valid=0, exact=0, extra=0, missing=0)
            for label in ("0-2", "3-4", "5+")}
    for targets, smiles in parents:
        targets = targets.to(device)
        mu, _ = model.encoder(targets, from_canonical_cache=True)
        decoded = model.decode(mu)
        if not all(t.isfinite().all() for t in (mu, decoded.atoms, decoded.charges, decoded.bonds)):
            raise FloatingPointError("Nonfinite generated round-trip evaluation")
        hard = decoded.harden(model.spec)
        eligible += len(smiles)
        observations = _inspect(hard, model.spec)
        for row, target in zip(observations, smiles, strict=True):
            valid += int(row.smiles is not None)
            exact += int(row.smiles == target)
        extra_edges = ((hard.bonds != 0) & (targets.bonds == 0)).triu(1)
        missing_edges = ((hard.bonds == 0) & (targets.bonds != 0)).triu(1)
        extra += int(extra_edges.sum())
        missing += int(missing_edges.sum())
        if cycle_bins:
            # The targets have already passed the connected, lossless parent filter.
            ranks = (targets.bonds != 0).triu(1).sum((1, 2)) - (targets.atoms != 0).sum(1) + 1
            counts = torch.stack((ranks, extra_edges.sum((1, 2)), missing_edges.sum((1, 2))), dim=1).tolist()
            for (rank, added, lost), row, target in zip(counts, observations, smiles, strict=True):
                group = bins["0-2" if rank <= 2 else "3-4" if rank <= 4 else "5+"]
                group["parents"] += 1
                group["valid"] += int(row.smiles is not None)
                group["exact"] += int(row.smiles == target)
                group["extra"] += added
                group["missing"] += lost
    result = dict(prior_attempts=attempts, eligible_parents=eligible, mean_valid=valid, mean_exact=exact,
                  extra_canonical_slot_edges=extra, missing_canonical_slot_edges=missing)
    if cycle_bins:
        result["by_target_cycle_rank"] = bins
    return result


def _motif_panel(smiles, *, monitor_ring_sizes=False):
    """Count one hit per molecule and group, retaining invalid attempts in the denominator."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors
    from latentmol.guidance.motif_completion import AMIDE

    counts = Counter(attempts=len(smiles), valid=0, aromatic=0, amide=0, smiles_roundtrip=0)
    sizes = Counter()
    ring_rows, scaffolds = [], set()
    if monitor_ring_sizes:
        import numpy as np
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        from latentmol.guidance.ring_size import ring_size_score
    with rdBase.BlockLogs():
        for value in smiles:
            if value is None:
                continue
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                counts["smiles_roundtrip"] += 1
                continue
            counts["valid"] += 1
            counts["aromatic"] += int(rdMolDescriptors.CalcNumAromaticRings(mol) > 0)
            counts["amide"] += int(mol.HasSubstructMatch(AMIDE))
            ring_sizes = {len(ring) for ring in mol.GetRingInfo().AtomRings()}
            sizes.update(ring_sizes)
            large = any(n >= 8 for n in ring_sizes)
            counts["any_ring_8_or_larger"] += int(large)
            if monitor_ring_sizes:
                ring_rows.append((mol.GetNumHeavyAtoms(), bool(ring_sizes), 3 in ring_sizes, large))
                if ring_sizes:
                    scaffolds.add(MurckoScaffoldSmiles(mol=mol))
    result = dict(counts, molecules_by_ring_size=dict(sizes))
    if monitor_ring_sizes:
        _, _, rates = ring_size_score(np.asarray(ring_rows, dtype=int).reshape(-1, 4))
        result.update(ring_size=rates, distinct_nonempty_scaffolds=len(scaffolds - {""}))
    return result


class OccupancyMonitor:
    """Argmax at temperature one, no repair; measure on private RNGs in eval mode."""

    def __init__(self, policy, validation, model, output, initial_update):
        seed = policy["seed"]
        self.batch = policy["batch_size"]
        self.interval = policy["interval_updates"]
        self.early = policy["early_updates"]
        n_prior, n_valid = policy["prior_points"], policy["validation_points"]
        if (any(type(n) is not int or n <= 0 for n in (n_prior, n_valid, self.batch, self.interval))
                or type(seed) is not int or any(type(i) is not int or i <= 0 for i in self.early)):
            raise ValueError("Invalid fixed-panel evaluation budgets")
        provenance = validation.manifest["provenance"]
        start = provenance["source_counts"]["val"]
        if (validation.split != "val" or provenance["added_holdout_source"] != "pubchem"
                or len(validation) - start != provenance["added_holdouts_per_split"]):
            raise ValueError("Expected the new PubChem holdout at the end of the balanced validation split")
        self.rows = random.Random(seed).sample(range(start, len(validation)), n_valid)
        self.inputs = []
        for i in range(0, len(self.rows), self.batch):
            rows = self.rows[i:i + self.batch]
            self.inputs.append((collate_graphs([validation[r] for r in rows]), [validation.smiles[r] for r in rows]))
        self.monitor_ring_sizes = policy.get("monitor_ring_sizes", False)
        self.real_motifs = (_motif_panel([s for _, batch in self.inputs for s in batch], monitor_ring_sizes=self.monitor_ring_sizes)
                            if policy.get("monitor_motifs") else None)
        self.prior = torch.randn(n_prior, model.config.latent, generator=torch.Generator().manual_seed(seed))
        self.initial_update = initial_update
        self.last = None
        self.roundtrip_points = policy.get("generated_roundtrip_points", 0)
        self.cycle_bins = policy.get("generated_cycle_bins", False)
        if type(self.roundtrip_points) is not int or not 0 <= self.roundtrip_points <= n_prior:
            raise ValueError("Invalid number of generated round-trip evaluation points")
        self.starting_parents = None
        self.seed = seed
        self.dataset = validation.manifest["dataset_sha256"]
        self.path = output / "quality-panel.jsonl"
        self.path.touch(exist_ok=False)

    @torch.inference_mode()
    def record(self, model, cursor, *, force=False):
        update = cursor["updates"]
        offset = update - self.initial_update
        if self.last == update or not (force or offset in self.early or update % self.interval == 0):
            return
        device = next(model.parameters()).device
        prior_results, hard_counts, soft_counts = [], [], []
        sampled_smiles = []
        sampler = (torch.Generator(device=device).manual_seed(self.seed + 1)
                   if self.real_motifs is not None else None)
        real_valid = exact = 0
        parents = []
        screened = 0
        was_training = model.training
        model.eval()
        try:
            for z in self.prior.split(self.batch):
                decoded = model.decode(z.to(device))
                hard = decoded.harden(model.spec)
                prior_results.extend(_inspect(hard, model.spec))
                if screened < self.roundtrip_points:
                    from latentmol.training.generated_roundtrip import canonical_generated_parents
                    if not all(t.isfinite().all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
                        raise FloatingPointError("Nonfinite prior logits during generated round-trip evaluation")
                    n = min(len(z), self.roundtrip_points - screened)
                    source = HardGraphBatch(hard.atoms[:n], hard.charges[:n], hard.bonds[:n])
                    parent_batch, info = canonical_generated_parents(source, model.spec, include_smiles=True)
                    if parent_batch is not None:
                        parents.append((parent_batch.to("cpu"), info["smiles"]))
                    screened += n
                if sampler is not None:
                    sampled = sample_hard_graph(decoded, model.spec, generator=sampler)
                    sampled_smiles.extend(x.smiles for x in _inspect(sampled, model.spec))
                hard_counts.extend((hard.atoms != 0).sum(-1).tolist())
                soft_counts.extend((1 - decoded.atoms.softmax(-1)[..., 0]).sum(-1).tolist())
            for hard, smiles in self.inputs:
                mu, _ = model.encoder(hard.to(device), from_canonical_cache=True)
                observed = _inspect(model.decode(mu).harden(model.spec), model.spec)
                real_valid += sum(row.smiles is not None for row in observed)
                exact += sum(row.smiles == target for row, target in zip(observed, smiles, strict=True))
            if self.roundtrip_points:
                regenerated = _generated_roundtrip_metrics(model, parents, device, screened, cycle_bins=self.cycle_bins)
                if self.starting_parents is None:
                    self.starting_parents = parents
                    fixed = dict(regenerated)
                else:
                    fixed = _generated_roundtrip_metrics(model, self.starting_parents, device, screened,
                                                         cycle_bins=self.cycle_bins)
        finally:
            model.train(was_training)
        valid = [row.smiles for row in prior_results if row.smiles is not None]
        event = dict(cursor, seed=self.seed, dataset_sha256=self.dataset, validation_rows=self.rows,
                     prior_points=len(prior_results), prior_valid=len(valid), prior_unique_valid=len(set(valid)),
                     prior_at_capacity=hard_counts.count(model.spec.max_nodes), prior_hard_count_mean=mean(hard_counts),
                     prior_hard_count_median=median(hard_counts), prior_soft_count_mean=mean(soft_counts),
                     prior_size_histogram=dict(Counter(hard_counts)),
                     prior_failures=dict(Counter(row.failure_reason for row in prior_results if row.smiles is None)),
                     validation_points=len(self.rows), reconstruction_valid=real_valid, reconstruction_exact=exact)
        if self.real_motifs is not None:
            event["prior_argmax_motifs"] = _motif_panel([row.smiles for row in prior_results], monitor_ring_sizes=self.monitor_ring_sizes)
            event["prior_categorical_motifs"] = _motif_panel(sampled_smiles, monitor_ring_sizes=self.monitor_ring_sizes)
            event["categorical_seed"] = self.seed + 1
            event["heldout_input_motifs"] = self.real_motifs
        if self.roundtrip_points:
            event["generated_roundtrip"] = dict(current_prior_parents=regenerated,
                                                fixed_at_leg_start=fixed, leg_start_update=self.initial_update)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(event, allow_nan=False) + "\n")
        self.last = update
