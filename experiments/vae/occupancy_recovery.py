"""Bounded frozen-VAE occupancy interventions, with separate calibration and evaluation panels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rdkit import rdBase
import torch
from torch.nn import functional as F

from latentmol.chemistry import decode_graph, encode_canonical_smiles
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.evaluate import load_model
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.sampling import sample_hard_graph


CHECKPOINTS = {
    "parent": "b239143056b561d5b08249c5b28e0366722c06d34f3bf94e1faf7acdfead4109",
    "student": "408b6904281fbe5d9f92e8b40c62e301d099b0fdad87109ef0ca5d90ce172922",
}


def occupancy_logits(atoms):
    return atoms[..., 1:].logsumexp(-1) - atoms[..., 0]


def set_occupancy(atoms, logits):
    """Preserve conditional element probabilities and set the exact binary log odds."""
    real = atoms[..., 1:]
    return torch.cat([(real.logsumexp(-1) - logits).unsqueeze(-1), real], dim=-1)


def load_panel(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data["rdkit"] != rdBase.rdkitVersion:
        raise ValueError("RDKit version differs from the canonical graph cache")
    return data


@torch.no_grad()
def encode_panel(model, smiles):
    means, variances = [], []
    for start in range(0, len(smiles), 8):
        items = []
        for text in smiles[start:start + 8]:
            encoded = encode_canonical_smiles(text, model.spec)
            if encoded["smiles"] != text:
                raise ValueError("Non-canonical input in the data panel")
            items.append({k: torch.as_tensor(encoded[v], dtype=torch.long) for k, v in
                          (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
        mu, logvar = model.encoder(collate_graphs(items), from_canonical_cache=True)
        means.append(mu)
        variances.append(logvar.exp())
    return torch.cat(means), torch.cat(variances)


@torch.no_grad()
def collect_features(model, latents):
    rows, odds = [], []
    for batch in latents.split(8):
        decoded, features = model.decoder.decode_with_features(batch)
        rows.append(features.atom_hidden)
        odds.append(occupancy_logits(decoded.atoms))
    return torch.cat(rows), torch.cat(odds)


def fit_occupancy_readout(real_hidden, real_target, prior_hidden, prior_target):
    """A 97-parameter binary readout. Fit on real training graphs and separate Gaussian draws."""
    if len(real_hidden) != len(prior_hidden):
        raise ValueError("Use equally weighted real and prior calibration panels")
    x = torch.cat((real_hidden, prior_hidden)).flatten(0, 1)
    y = torch.cat((real_target, prior_target)).flatten().float()
    center = x.mean(0)
    scale = x.std(0).clamp_min(1e-4)
    x = (x - center) / scale
    weight = torch.zeros(x.shape[-1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y) + 1e-4 * weight.square().sum()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(weight.grad).all():
            raise FloatingPointError("Nonfinite binary readout objective")
        return loss

    opt.step(closure)
    return weight.detach() / scale, (bias.detach() - (center * weight.detach() / scale).sum())


@torch.no_grad()
def parent_occupancy_intervention(parent, student, z):
    """Feed the parent's same-z occupancy into every student refinement pass, single forward each."""
    odds = []
    hook = parent.decoder.atom_head.register_forward_hook(
        lambda module, inputs, output: odds.append(occupancy_logits(output)))
    try:
        parent_decoded = parent.decode(z)
    finally:
        hook.remove()
    stages = iter(odds)
    hook = student.decoder.atom_head.register_forward_hook(
        lambda module, inputs, output: set_occupancy(output, next(stages)))
    try:
        student_decoded = student.decode(z)
    finally:
        hook.remove()
    if next(stages, None) is not None:
        raise ValueError("Incompatible decoder refinement stages")
    return parent_decoded, student_decoded


class Panel:
    """Use the same categorical sampling seed in each arm, including invalid attempts."""

    def __init__(self, seed):
        self.rng = torch.Generator().manual_seed(seed)
        self.rows = []

    def add(self, decoded, spec, expected=None):
        if not all(torch.isfinite(x).all() for x in (decoded.atoms, decoded.charges, decoded.bonds)):
            raise FloatingPointError("Nonfinite decoder output")
        hard, sampled = decoded.harden(spec), sample_hard_graph(decoded, spec, generator=self.rng)
        for i in range(len(hard.atoms)):
            def molecule(graph):
                return decode_graph(graph.atoms[i], graph.charges[i], graph.bonds[i], spec)
            self.rows.append(dict(count=int(hard.atoms[i].ne(0).sum()), smiles=molecule(hard),
                                  sampled=molecule(sampled), expected=None if expected is None else expected[i]))

    def summarize(self, limit, fit_smiles):
        valid = [r["smiles"] for r in self.rows if r["smiles"]]
        counts = torch.tensor([r["count"] for r in self.rows], dtype=torch.float32)
        return dict(n=len(self.rows), valid=len(valid), unique_valid=len(set(valid)),
                    unique_valid_outside_calibration=len(set(valid) - fit_smiles),
                    sample_valid=sum(bool(r["sampled"]) for r in self.rows),
                    exact=sum(r["smiles"] == r["expected"] for r in self.rows) if self.rows[0]["expected"] else None,
                    mean_atoms=float(counts.mean()), median_atoms=float(counts.quantile(.5)),
                    at_capacity=int((counts == limit).sum()), size_hist=dict(Counter(int(v) for v in counts)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "student", "train", "validation", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    train, val = load_panel(args.train), load_panel(args.validation)
    if train.get("split") != "train" or train["dataset_sha256"] != val["dataset_sha256"] or train["spec"] != val["spec"]:
        raise ValueError("Calibration and evaluation must use matching published train and validation panels")
    spec = GraphSpec.from_dict(train["spec"])
    fit_smiles = [r[2] for r in train["rows"]]
    valid_smiles = [r[2] for r in val["rows"][:96]]
    if len(fit_smiles) != 512 or len(valid_smiles) != 96 or set(fit_smiles).intersection(valid_smiles):
        raise ValueError("Require disjoint train and validation inputs")
    models = {}
    for key, sha in CHECKPOINTS.items():
        path = getattr(args, key)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Checkpoint identity changed")
        models[key], _ = load_model(path, "cpu", spec)
        models[key].eval().requires_grad_(False)
    parent, student = models["parent"], models["student"]
    panels = {}
    def record(name, decoded, expected=None):
        if name not in panels:
            panels[name] = Panel(11904)
        panels[name].add(decoded, spec, expected)
    with rdBase.BlockLogs():
        mu, var = encode_panel(student, fit_smiles)
        # The aggregate posterior mean and covariance include within-molecule variance.
        # Shrink the off-diagonal covariance by 10% before a Cholesky draw.
        cov = torch.cov(mu.T) + torch.diag(var.mean(0))
        chol = torch.linalg.cholesky(.9 * cov + .1 * cov.diag().diag())
        fit_z = torch.randn(192, student.config.latent, generator=torch.Generator().manual_seed(11901))
        _, teacher_odds = collect_features(parent, fit_z)
        feature_z, _ = collect_features(student, fit_z)
        feature_x, _ = collect_features(student, mu[:len(fit_z)])
        n_atoms = torch.tensor([sum(v != 0 for v in encode_canonical_smiles(s, spec)["atom_types"])
                                for s in fit_smiles[:len(fit_z)]])
        targets = torch.arange(spec.max_nodes)[None] < n_atoms[:, None]
        w, b = fit_occupancy_readout(feature_x, targets, feature_z, teacher_odds.sigmoid())
        eval_z = torch.randn(128, student.config.latent, generator=torch.Generator().manual_seed(11902))
        with torch.no_grad():
            for z in eval_z.split(8):
                decoded, f = student.decoder.decode_with_features(z)
                teacher, guided = parent_occupancy_intervention(parent, student, z)
                record("parent", teacher)
                record("student", decoded)
                for name, odds in (("parent_occupancy_final", occupancy_logits(teacher.atoms)),
                                   ("fitted_occupancy_final", f.atom_hidden @ w + b)):
                    record(name, DecoderOutput(set_occupancy(decoded.atoms, odds), decoded.charges, decoded.bonds))
                record("parent_occupancy_all_rounds", guided)
                record("gaussian_mean_only", student.decode(mu.mean(0) + z))
                record("gaussian_diagonal", student.decode(mu.mean(0) + z * cov.diag().sqrt()))
                record("gaussian_full", student.decode(mu.mean(0) + z @ chol.T))
            real_z, _ = encode_panel(student, valid_smiles)
            for i in range(0, len(real_z), 8):
                decoded, f = student.decoder.decode_with_features(real_z[i:i + 8])
                expected = valid_smiles[i:i + 8]
                record("real_baseline", decoded, expected)
                corrected = DecoderOutput(set_occupancy(decoded.atoms, f.atom_hidden @ w + b), decoded.charges, decoded.bonds)
                record("real_fitted_occupancy", corrected, expected)
    summary = {key: panel.summarize(spec.max_nodes, set(fit_smiles)) for key, panel in panels.items()}
    report = dict(checkpoints=CHECKPOINTS, calibration_train=len(mu), calibration_prior=len(fit_z),
                  empirical_mean_norm=float(mu.mean(0).norm()), aggregate_sd_mean=float(cov.diag().sqrt().mean()),
                  arms=summary)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "rows.json").write_text(json.dumps({k: p.rows for k, p in panels.items()}))
    for name, stats in summary.items():
        print(name, {k: v for k, v in stats.items() if k != "size_hist"}, flush=True)


if __name__ == "__main__":
    main()
