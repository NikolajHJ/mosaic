"""Screen Gaussian seeds for three latent representations of the same molecular graph.

Only checkpoint outputs and chemistry are used during screening. The public panel
stores the proposal order, rejection reasons and actual FP32 vectors. A held-out
native score never enters selection. All accepted conclusions are conditional on
the three vectors hard-decoding to the same representable fixed-size molecule.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import torch

from latentmol.chemistry import MoleculeRejected, canonical_slot_order, encode_canonical_smiles, inspect_graph
from latentmol.design.latent_protenix import choose_active, hard_from_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import HardGraphBatch


ARMS = ("original_random", "reencoded_mean", "posterior_sample")
CHECKPOINT_597604 = "abfec8c32ab7d6e00240ee7d68492c7e271f4aa95f14be874674e1fe0cd742f4"


def _normal(model, seed):
    return torch.randn((1, model.config.latent), generator=torch.Generator().manual_seed(seed))


def _encode(model, smiles):
    data = encode_canonical_smiles(smiles, model.spec)
    hard = HardGraphBatch(*(torch.as_tensor(data[k]).long()[None] for k in ("atom_types", "charges", "bond_types")))
    mu, logvar = model.encoder(hard, from_canonical_cache=True)
    if not (torch.isfinite(mu).all() and torch.isfinite(logvar).all()):
        raise FloatingPointError("Nonfinite posterior for representable parent")
    return mu, (logvar * 0.5).exp()


def _identity(model, latent, count):
    """Verify native hard labels in canonical order, including charges and all bonds."""
    if latent.shape != (1, model.config.latent) or not latent.isfinite().all():
        raise ValueError("Malformed latent vector")
    decoded = model.decode(latent)
    if not all(t.isfinite().all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
        raise FloatingPointError("Nonfinite decoder output")
    raw = decoded.harden(model.spec)
    if int(raw.atoms.count_nonzero()) != count:
        raise MoleculeRejected("atom_count")
    labels = [getattr(raw, k)[0] for k in ("atoms", "charges", "bonds")]
    check = inspect_graph(*labels, model.spec)
    if check.smiles is None:
        raise MoleculeRejected(str(check.failure_reason))
    active = choose_active(decoded, model.spec, count)
    fixed = hard_from_graph(graph_from_decoder(decoded, model.spec, temperature=1., fixed_active=active))
    if not all(torch.equal(a, getattr(fixed, k)[0]) for a, k in zip(labels, ("atoms", "charges", "bonds"))):
        raise MoleculeRejected("raw_fixed_label_mismatch")
    # canonical_slot_order verifies a bijection to the exact stored canonical
    # atom, charge and bond tensors (rejects loss of e.g. Kekulé labels).
    order = canonical_slot_order(*labels, model.spec)
    if sorted(order[:count]) != active.nonzero().flatten().tolist():
        raise ValueError("Native order is inconsistent with the fixed active mask")
    return check.smiles, active, order


def _draw_triplet(model, prior, epsilon_seed, count, seen, attempt):
    states = {}
    for arm in ARMS:
        if arm == "original_random":
            z = prior
        elif arm == "reencoded_mean":
            mu, sigma = _encode(model, attempt["smiles"])
            z = mu
        else:
            # Exactly one draw from q(z|m) at learned (unit multiplier) variance.
            z = mu + sigma * _normal(model, epsilon_seed)
        attempt["stage"] = arm
        try:
            smiles, active, order = _identity(model, z, count)
        except MoleculeRejected as exc:
            attempt["reason"] = exc.reason
            return None
        if arm == "original_random":
            attempt["smiles"] = smiles
            if smiles in seen:
                attempt["reason"] = "duplicate_qualified_parent"
                return None
        elif smiles != attempt["smiles"]:
            attempt.update(reason="hard_identity_changed", decoded_smiles=smiles)
            return None
        states[arm] = dict(latent=z[0].tolist(), active=active.nonzero().flatten().tolist(), native_order=order)
    attempt["reason"] = "accepted"
    return states


@torch.no_grad()
def generate_panel(model, checkpoint_sha256, count, target, limit, seed, *, count_device="cpu", prefix=None):
    """Use the first eligible distinct parents in deterministic Gaussian draw order."""
    if not (0 < target <= limit and 0 < count <= model.spec.max_nodes and seed >= 0):
        raise ValueError("Invalid screening budget or atom count")
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("Use the frozen FP32 evaluation model on CPU for screening")
    count_model = model if count_device == "cpu" else deepcopy(model).to(count_device)
    attempts, parents = [], []
    if prefix is not None:
        if (prefix["checkpoint_sha256"], prefix["fixed_count"], prefix["requested"], prefix["seed"],
                prefix["prefilter_device"]) != (checkpoint_sha256, count, target, seed, count_device):
            raise ValueError("The existing screening prefix uses a different recipe")
        attempts, parents = list(prefix["attempts"]), list(prefix["parents"])
        if (prefix["proposals"] != len(attempts) or prefix["accepted"] != len(parents)
                or len(parents) >= target or len(attempts) != prefix["limit"] or len(attempts) >= limit
                or any(r["index"] != i for i, r in enumerate(attempts))):
            raise ValueError("Only a complete exhausted screening prefix can be extended")
    seen = {p["smiles"] for p in parents}
    # Batch only the cheap atom-count filter. Exact graph, positions, labels
    # and latent provenance of every eligible parent are checked serially.
    for offset in range(len(attempts), limit, 16):
        indices = range(offset, min(offset + 16, limit))
        priors = [_normal(model, seed + 2 * i) for i in indices]
        decoded = count_model.decode(torch.cat(priors).to(count_device))
        if not all(t.isfinite().all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
            raise FloatingPointError("Nonfinite batched screening decode")
        counts = decoded.atoms.argmax(-1).count_nonzero(-1).tolist()
        for i, prior, atom_count in zip(indices, priors, counts):
            attempt = dict(index=i, prior_seed=seed + 2 * i, posterior_seed=seed + 2 * i + 1,
                           proposal_atom_count=atom_count, stage="original_random")
            attempts.append(attempt)
            if atom_count != count:
                attempt["reason"] = "atom_count"
                continue
            states = _draw_triplet(model, prior, attempt["posterior_seed"], count, seen, attempt)
            if states is None:
                continue
            seen.add(attempt["smiles"])
            parents.append(dict(name=f"parent-{len(parents)+1:02d}", index=i, smiles=attempt["smiles"], states=states))
            print(f"Matched {len(parents)}/{target} at proposal {i + 1}", flush=True)
            if len(parents) == target:
                break
        if len(parents) == target:
            break
    return dict(version=1, checkpoint_sha256=checkpoint_sha256, fixed_count=count, seed=seed, limit=limit,
                prefilter_device=count_device, verification_device="cpu",
                requested=target, proposals=len(attempts), accepted=len(parents),
                outcomes=dict(Counter(f"{r['stage']}/{r['reason']}" for r in attempts)),
                attempts=attempts, parents=parents)


@torch.no_grad()
def load_matched_starts(model, panel_path, checkpoint_sha256, fixed_count):
    """Restore all three states and verify each raw and native-facing parent on this host."""
    panel = json.loads(Path(panel_path).read_bytes())
    if (panel["version"] != 1 or panel["checkpoint_sha256"] != checkpoint_sha256
            or panel["fixed_count"] != fixed_count or panel["accepted"] != panel["requested"]
            or len(panel["parents"]) != panel["accepted"] or panel["accepted"] < 1):
        raise ValueError("Incomplete matched panel or incompatible checkpoint/context")
    complete = []
    seen = set()
    for parent in panel["parents"]:
        sm = parent["smiles"]
        row = panel["attempts"][parent["index"]]
        if (row["index"] != parent["index"] or row["prior_seed"] != panel["seed"] + 2 * row["index"]
                or row["posterior_seed"] != row["prior_seed"] + 1
                or row["reason"] != "accepted" or row["smiles"] != sm or set(parent["states"]) != set(ARMS)
                or sm in seen):
            raise ValueError("Invalid screening provenance or duplicate parent")
        seen.add(sm)
        mu, sigma = _encode(model, sm)
        expected = (_normal(model, row["prior_seed"]), mu, mu + sigma * _normal(model, row["posterior_seed"]))
        states = {}
        for arm, replay in zip(ARMS, expected):
            data = parent["states"][arm]
            z = torch.tensor(data["latent"], dtype=torch.float32)[None]
            # Heterogeneous CPUs can differ slightly in the encoder forward.
            # The saved FP32 vector is what all search arms actually receive.
            if not torch.allclose(z, replay, rtol=2e-4, atol=2e-5):
                raise ValueError(f"{parent['name']}/{arm}: seeded latent provenance differs on replay")
            result, active, order = _identity(model, z, fixed_count)
            if result != sm or order != data["native_order"] or active.nonzero().flatten().tolist() != data["active"]:
                raise ValueError(f"{parent['name']}/{arm}: frozen starting molecule or mapping changed")
            states[arm] = dict(latent=z, active=active, native_order=torch.tensor(order, dtype=torch.long), step=0,
                               kind=arm, provenance=dict(parent=parent["name"], smiles=sm, proposal=parent["index"],
                               prior_seed=row["prior_seed"], posterior_seed=row["posterior_seed"], native_order=order))
        complete.append((parent["name"], sm, states))
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parents", type=int, default=12)
    parser.add_argument("--limit", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026112297)
    parser.add_argument("--atom-count", type=int, default=25)
    parser.add_argument("--count-device", choices=["cpu", "cuda"], default="cpu",
                        help="Device only for the preliminary batched raw atom count. Accepted states use CPU.")
    parser.add_argument("--extend-panel", type=Path,
                        help="Continue an exhausted seed sequence into a new output. Keep previous attempts.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    digest = file_sha256(args.checkpoint)
    if digest != CHECKPOINT_597604:
        raise ValueError("Study requires the authenticated checkpoint at update 597604")
    model, checkpoint = load_model(args.checkpoint, "cpu")
    if checkpoint["continuation"]["updates"] != 597604:
        raise ValueError("Checkpoint continuation update differs")
    model.eval().requires_grad_(False)
    prefix = json.loads(args.extend_panel.read_bytes()) if args.extend_panel else None
    panel = generate_panel(model, digest, args.atom_count, args.parents, args.limit, args.seed,
                           count_device=args.count_device, prefix=prefix)
    if args.extend_panel:
        panel["prefix_sha256"] = file_sha256(args.extend_panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(panel, out, allow_nan=False, separators=(",", ":"))
    print(json.dumps({k: v for k, v in panel.items() if k not in ("attempts", "parents")}), flush=True)
    load_matched_starts(model, args.output, digest, args.atom_count)
    print("Replayed every raw and canonically aligned fixed-slot starting molecule", flush=True)


if __name__ == "__main__":
    main()
