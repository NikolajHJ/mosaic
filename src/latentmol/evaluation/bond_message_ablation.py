"""Measure how a trained decoder uses its learned typed-bond messages at inference."""

import torch

from latentmol.chemistry import inspect_graph
from latentmol.chemistry_early_health import preserve_observation_state
from latentmol.representation import HardGraphBatch


def _without_messages(model, latent, modules):
    handles = []
    try:
        for module in modules:
            handles.append(module.register_forward_hook(lambda _module, _args, output: torch.zeros_like(output)))
        return model.decode(latent)
    finally:
        for handle in handles:
            handle.remove()


def _total_variation(left, right):
    return (left.softmax(-1) - right.softmax(-1)).abs().sum(-1) * .5


def bond_message_ablation(model, batches, *, limit=64):
    """Compare ordinary and message-silenced posterior means on verified canonical rows.

    TV is pooled per target slot/pair. Conditional bond rows include NO_BOND,
    omit occupancy gating, and count an undirected pair once. Hard-label change
    includes atom, charge and bond predictions over the entire decoded graph.
    """
    if type(limit) is not int or limit < 1 or model.config.decoder_variant not in ("bond_message", "triangle_bond_message"):
        raise ValueError("Require a bond-message decoder and a positive molecule limit")
    modules = [layer.bond_messages for layer in model.decoder.rounds]
    if not modules or any(module is None for module in modules):
        raise ValueError("Missing a bond-message module in the decoder rounds")
    device = next(model.parameters()).device
    sums, counts = {}, {}
    samples = hard_changes = both_valid = changed_valid_identity = 0
    fields = ("atoms", "charges", "bonds")
    with preserve_observation_state(model, {}):
        for batch in batches:
            if samples >= limit:
                break
            target = HardGraphBatch(*(getattr(batch, f)[:limit - samples].to(device) for f in fields))
            latent, _ = model.encoder(target, from_canonical_cache=True)
            normal = model.decode(latent)
            silenced = _without_messages(model, latent, modules)
            restored = model.decode(latent)
            for name in fields:
                left, right = getattr(normal, name), getattr(silenced, name)
                if not left.isfinite().all() or not right.isfinite().all():
                    raise FloatingPointError("Nonfinite decoder logits in message ablation")
                if not torch.equal(left, getattr(restored, name)):
                    raise RuntimeError("Removing the bond-message ablation did not restore the original logits")
            active = target.atoms.ne(0)
            pairs = (active[:, :, None] & active[:, None, :]).triu(diagonal=1)
            true_bond = target.bonds.ne(0)
            atom_tv = _total_variation(normal.atoms, silenced.atoms)
            bond_tv = _total_variation(normal.symmetric_bond_logits, silenced.symmetric_bond_logits)
            for name, tv, mask in (("target_atoms", atom_tv, active), ("true_bonds", bond_tv, pairs & true_bond),
                                   ("absent_active_pairs", bond_tv, pairs & ~true_bond)):
                sums[name] = sums.get(name, 0.) + float(tv[mask].sum(dtype=torch.float64))
                counts[name] = counts.get(name, 0) + int(mask.sum())
            before, after = (x.harden(model.spec).to("cpu") for x in (normal, silenced))
            changed = torch.stack([getattr(before, f).ne(getattr(after, f)).flatten(1).any(1) for f in fields]).any(0)
            for i in range(len(changed)):
                smi = inspect_graph(before.atoms[i], before.charges[i], before.bonds[i], model.spec).smiles
                other = (inspect_graph(after.atoms[i], after.charges[i], after.bonds[i], model.spec).smiles
                         if changed[i] else smi)
                if smi is not None and other is not None:
                    both_valid += 1
                    changed_valid_identity += int(smi != other)
            hard_changes += int(changed.sum())
            samples += len(changed)
    if not samples:
        raise ValueError("Message ablation requires held-out molecules")
    return dict(samples=samples, silenced_rounds=len(modules), restored_logits_exact=True,
                mean_total_variation={k: sums[k] / n if n else None for k, n in counts.items()},
                rows_per_measure=counts, hard_label_changes=hard_changes,
                hard_label_change_fraction=hard_changes / samples, both_valid=both_valid,
                different_molecules_among_both_valid=changed_valid_identity,
                different_molecule_fraction_among_both_valid=changed_valid_identity / both_valid if both_valid else None)
