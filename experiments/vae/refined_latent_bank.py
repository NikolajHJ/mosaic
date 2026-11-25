"""Generate exact, displacement-bounded encoder targets with a frozen VAE."""

from collections import Counter
import math

import torch

from latentmol.representation import HardGraphBatch
from latentmol.sampling import _validate_graph_fields
from latentmol.training.generated_roundtrip import canonical_generated_parents
from latentmol.training.losses import reconstruction_losses
from latentmol.training.occupancy_monitor import _inspect


@torch.enable_grad()
def _first_hits(model, targets, names, loss_config, *, steps, lr, radius):
    """Return the earliest chemically exact latent for each target, including update zero."""
    with torch.no_grad():
        mu, _ = model.encoder(targets, from_canonical_cache=True)
        if mu.shape != (len(names), model.config.latent) or not torch.isfinite(mu).all():
            raise ValueError("Invalid encoded parent means")
    z = mu.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    teachers = {}
    for step in range(steps + 1):
        decoded = model.decode(z)
        _validate_graph_fields(decoded, model.spec)
        with torch.no_grad():
            observed = _inspect(decoded.harden(model.spec), model.spec)
            for i, result in enumerate(observed):
                if i not in teachers and result.smiles == names[i]:
                    teachers[i] = (step, z[i].detach().cpu().clone())
        if len(teachers) == len(names) or step == steps:
            break
        # Graph mode chooses a detached atom map and differentiates the checkpoint's
        # reconstruction CE through that mapped graph, as in the recovery experiment.
        loss = reconstruction_losses(decoded, targets, loss_config)["reconstruction"]
        if not loss.requires_grad or not torch.isfinite(loss):
            raise FloatingPointError("Detached or nonfinite refinement objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if z.grad is None or not torch.isfinite(z.grad).all():
            raise FloatingPointError("Missing or nonfinite refinement gradient")
        optimizer.step()
        with torch.no_grad():
            offset = z - mu
            scale = (radius / offset.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
            z.copy_(mu + offset * scale)
    return mu.detach().cpu(), teachers


def build_teacher_bank(model, loss_config, *, seed, draws, batch_size=16, steps=20,
                       lr=0.02, radius=4.0, exclude_smiles=()):
    """Screen every seeded draw; return CPU teacher tensors and an eligible-parent audit.

    Call on a frozen evaluation-mode model. The seed and zero-based prior row
    identify the original generating vector. That vector is never a teacher.
    Exclude evaluation identities *before* refinement, and deduplicate even
    unsuccessful parents. Each target passed the same lossless-canonicalization
    filter as the production panel. Exactness compares canonical stereo-free SMILES.
    """
    if (min(draws, batch_size, steps) < 1 or not math.isfinite(lr) or lr <= 0
            or not math.isfinite(radius) or radius <= 0 or loss_config.atom_order != "graph"):
        raise ValueError("Expected positive budgets and graph-matched reconstruction")
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze the VAE and switch to evaluation mode before generating teachers")
    frozen = {name: t._version for name, t in model.state_dict(keep_vars=True).items()}
    device = next(model.parameters()).device
    prior = torch.randn(draws, model.config.latent, generator=torch.Generator().manual_seed(seed))
    excluded, seen = set(exclude_smiles), set()
    screening, rejections = [], Counter()
    rows, hits, means, teachers, names = [], [], [], [], []
    atoms, charges, bonds = [], [], []
    for start in range(0, draws, batch_size):
        with torch.no_grad():
            decoded = model.decode(prior[start:start + batch_size].to(device))
            _validate_graph_fields(decoded, model.spec)
            parents, info = canonical_generated_parents(decoded.harden(model.spec), model.spec, include_smiles=True)
        rejections.update(info["rejections"])
        keep, entries = [], []
        for i, (source, name) in enumerate(zip(info["rows"], info["smiles"], strict=True)):
            entry = dict(prior_row=start + source, smiles=name)
            screening.append(entry)
            if name in excluded:
                entry["kind"] = "excluded"
            elif name in seen:
                entry["kind"] = "duplicate"
            else:
                keep.append(i)
                entries.append(entry)
                seen.add(name)
        if keep:
            targets = HardGraphBatch(parents.atoms[keep], parents.charges[keep], parents.bonds[keep])
            mu, refined = _first_hits(model, targets, [e["smiles"] for e in entries], loss_config,
                                      steps=steps, lr=lr, radius=radius)
            cpu = targets.to("cpu")
            for i, entry in enumerate(entries):
                if i not in refined:
                    entry["kind"] = "failed"
                    continue
                step, teacher = refined[i]
                shift = (teacher - mu[i]).norm().item()
                if not math.isfinite(shift) or shift > radius + 1e-4 or (step == 0 and shift != 0):
                    raise ValueError("Invalid displacement for exact teacher")
                entry.update(kind="rescued" if step else "anchor", hit_step=step, displacement=shift)
                rows.append(entry["prior_row"])
                names.append(entry["smiles"])
                hits.append(step)
                means.append(mu[i])
                teachers.append(teacher)
                atoms.append(cpu.atoms[i])
                charges.append(cpu.charges[i])
                bonds.append(cpu.bonds[i])
        print(f"Teacher draws {min(start + batch_size, draws)}/{draws}: {len(rows)} exact targets", flush=True)
    if len(screening) + sum(rejections.values()) != draws:
        raise RuntimeError("Unaccounted prior draws")
    if any(t._version != frozen[k] for k, t in model.state_dict(keep_vars=True).items()):
        raise RuntimeError("Model parameters or buffers changed while generating teachers")
    counts = Counter(e["kind"] for e in screening)
    nodes, width = model.spec.max_nodes, model.config.latent
    def stack(items, shape, dtype):
        return torch.stack(items) if items else torch.empty((0, *shape), dtype=dtype)
    return dict(atoms=stack(atoms, (nodes,), torch.long), charges=stack(charges, (nodes,), torch.long),
                bonds=stack(bonds, (nodes, nodes), torch.long), mu0=stack(means, (width,), prior.dtype),
                zteacher=stack(teachers, (width,), prior.dtype), prior_rows=torch.tensor(rows, dtype=torch.long),
                hit_steps=torch.tensor(hits, dtype=torch.long), smiles=names,
                kind=["rescued" if step else "anchor" for step in hits], screening=screening,
                summary=dict(seed=seed, draws=draws, eligible=len(screening), unique_refined=len(seen),
                             anchors=counts["anchor"], rescued=counts["rescued"], failed=counts["failed"],
                             excluded=counts["excluded"], duplicates=counts["duplicate"], rejections=dict(rejections),
                             first_hits=[hits.count(i) for i in range(steps + 1)],
                             cumulative_exact=[sum(s <= i for s in hits) for i in range(steps + 1)],
                             batch_size=batch_size, steps=steps, lr=lr, radius=radius))
