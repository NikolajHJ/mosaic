"""Small CPU fixtures for deterministic real-molecule anchor preparation."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.design.protenix_synthesis_library import cached_training_rows, prepare_library, select_banks
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def cache_fixture(path, spec):
    rows = [encode_smiles(s, spec) for s in ("CCC", "CCO", "CCN", "COC", "CNC", "CCF", "CCS", "C=CO")]
    path.mkdir()
    np.save(path / "atoms.npy", np.stack([row["atom_types"] for row in rows]))
    np.save(path / "split.npy", np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=np.int8))
    strings = [row["smiles"].encode() for row in rows]
    np.save(path / "smiles_offsets.npy", np.cumsum([0] + [len(s) for s in strings], dtype=np.int64))
    (path / "smiles.utf8").write_bytes(b"".join(strings))
    manifest = dict(version="moses-chembl-combined-v1", spec=spec.to_dict(),
        spec_fingerprint=spec.fingerprint(), dataset_sha256="fixture_dataset",
        sources={"chembl": {"counts": {"train": 6, "val": 1, "test": 1}}},
        files={p.name: {"bytes": p.stat().st_size} for p in path.iterdir()})
    (path / "manifest.json").write_text(json.dumps(manifest))
    return rows


def test_scan_is_deterministic_training_only_and_bounded(tmp_path):
    spec = GraphSpec(max_nodes=8)
    cache_fixture(tmp_path / "cache", spec)
    first, info = cached_training_rows(tmp_path / "cache", spec, occupied_atoms=3, seed=123, scan_limit=8)
    second, _ = cached_training_rows(tmp_path / "cache", spec, occupied_atoms=3, seed=123, scan_limit=8)
    assert [r["index"] for r in first] == [r["index"] for r in second]
    assert all(r["index"] not in (0, 1) for r in first)
    assert info["counts"]["source_rows_examined"] == 8
    assert all(np.count_nonzero(r["atom_types"]) == 3 for r in first)
    with pytest.raises(ValueError, match="2048"):
        cached_training_rows(tmp_path / "cache", spec, scan_limit=2049)


def test_selection_has_equal_disjoint_banks_without_control_sa_threshold():
    rows = [dict(smiles=str(i), sa_score=score) for i, score in enumerate([6., 2., 7., 1., 8.])]
    low, controls = select_banks(rows, bank_size=2, sa_ceiling=3.)
    assert [r["smiles"] for r in low] == ["1", "3"]
    assert [r["smiles"] for r in controls] == ["0", "2"]
    assert {r["smiles"] for r in low}.isdisjoint(r["smiles"] for r in controls)


def test_preparation_records_bad_decodes_and_preserves_checkpoint(tmp_path):
    torch.manual_seed(1913)
    spec, config = GraphSpec(max_nodes=8), ModelConfig(hidden=16, latent=4, layers=1)
    cache_fixture(tmp_path / "cache", spec)
    model = GraphAutoencoder(spec, config)
    # Deliberately make raw decodes empty: this is a diagnostic finding rather
    # than a reason to discard real source molecules or skip the bank.
    for parameter in model.decoder.parameters():
        parameter.data.zero_()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(format_version=1, graph_spec=spec.to_dict(), model_config=asdict(config),
                    model_state=model.state_dict()), checkpoint)
    before = checkpoint.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    output = tmp_path / "bank"
    summary = prepare_library(tmp_path / "cache", checkpoint, digest, output,
                              scan_limit=8, bank_size=2, occupied_atoms=3, sa_ceiling=10.)
    low = json.loads((output / "low-sa.json").read_text())
    control = json.loads((output / "unfiltered.json").read_text())
    assert summary["status"] == "completed" and summary["model_state_unchanged"]
    assert len(low["latents"]) == len(control["latents"]) == 2
    assert all(not row["mu_decode"]["raw"]["valid"] for row in low["molecules"])
    assert checkpoint.read_bytes() == before
    a, b = torch.tensor(low["latents"]), torch.tensor(control["latents"])
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a.norm(dim=-1).mean(), b.norm(dim=-1).mean(), atol=1e-6, rtol=1e-6)
    with pytest.raises(FileExistsError):
        prepare_library(tmp_path / "cache", checkpoint, digest, output)


def test_wrong_checkpoint_identity_refuses_before_creating_output(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a model")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA"):
        prepare_library(tmp_path / "unused", checkpoint, "0" * 64, output)
    assert not output.exists()


def test_source_slot_remapping_preserves_graph_and_explicit_occupancy():
    from latentmol.chemistry import inspect_graph
    from latentmol.design.protenix_synthesis_library import remap_source_slots
    spec = GraphSpec(max_nodes=6)
    source = encode_smiles('CCO', spec)
    original = {key: np.array(source[key], copy=True) for key in ('atom_types', 'charges', 'bond_types')}
    remapped = remap_source_slots(source, [0, 1, 4], spec)
    assert np.flatnonzero(remapped['atom_types']).tolist() == [0, 1, 4]
    assert inspect_graph(remapped['atom_types'], remapped['charges'], remapped['bond_types'], spec).smiles == source['smiles']
    assert remapped['bond_types'][4, 1] == source['bond_types'][2, 1]
    for key in original:
        np.testing.assert_array_equal(source[key], original[key])
    with pytest.raises(ValueError, match='distinct'):
        remap_source_slots(source, [0, 0, 4], spec)
    with pytest.raises(ValueError, match='atom count'):
        remap_source_slots(source, [0, 4], spec)


class InvertibleToy(torch.nn.Module):
    """Direct logit latents expose whether fitting updates z or frozen weights."""
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        n = spec.max_nodes
        self.shapes = [(n, len(spec.atomic_numbers)), (n, len(spec.formal_charges)),
                       (n, n, len(spec.bond_names))]
        self.sizes = [int(np.prod(shape)) for shape in self.shapes]

    def decode(self, z):
        from latentmol.representation import DecoderOutput
        parts = z.split(self.sizes, dim=-1)
        return DecoderOutput(*(part.reshape(len(z), *shape) * self.weight
                               for part, shape in zip(parts, self.shapes)))


def test_aligned_fitting_is_bounded_attached_and_changes_only_latents():
    from latentmol.design.latent_protenix import _model_identity
    from latentmol.design.protenix_synthesis_library import fit_aligned_latents
    spec = GraphSpec(max_nodes=5)
    model = InvertibleToy(spec)
    sources = [encode_smiles(s, spec) for s in ('CCO', 'CCN')]
    means = torch.zeros(2, sum(model.sizes))
    before = _model_identity(model)
    fit = fit_aligned_latents(model, means, sources, [0, 1, 3], updates=12,
                              learning_rate=.2, diagnostic_every=2, decoded_sa_ceiling=10.)
    assert fit['fit']['updates_per_source'] == 12
    assert all(len(records) == 7 for records in fit['diagnostics'])
    assert all(record is not None and record[2]['source_identity_reconstructed'] for record in fit['best'])
    assert all(final < first for first, final in zip(fit['initial_loss'], fit['terminal_loss']))
    assert torch.count_nonzero(means) == 0
    assert torch.count_nonzero(fit['terminal_latents']) > 0
    assert _model_identity(model) == before
    assert model.weight.grad is None and not model.weight.requires_grad
    assert all(torch.isfinite(item[1]).all() for item in fit['best'])


def test_aligned_fitting_records_quality_shortfalls_without_mutating_source():
    from latentmol.design.protenix_synthesis_library import fit_aligned_latents
    spec = GraphSpec(max_nodes=5)
    model = InvertibleToy(spec)
    source = encode_smiles('CCO', spec)
    means = torch.zeros(1, sum(model.sizes))
    fit = fit_aligned_latents(model, means, [source], [0, 1, 3], updates=2,
                              learning_rate=.2, diagnostic_every=1, decoded_sa_ceiling=0.)
    assert fit['best'] == [None]  # A too-strict measured quality threshold is a finding.
    assert fit['diagnostics'][0][-1]['valid']
    assert fit['fit']['updates_per_source'] == 2
    with pytest.raises(ValueError, match='bounded'):
        fit_aligned_latents(model, means, [source], [0, 1, 3], updates=401)
    bad = means.clone(); bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite'):
        fit_aligned_latents(model, bad, [source], [0, 1, 3])


def test_aligned_preparation_serializes_actual_mask_valid_banks(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from latentmol.design import protenix_synthesis_library as module
    from latentmol.design.latent_protenix import _model_identity
    spec = GraphSpec(max_nodes=5)
    model = InvertibleToy(spec)
    model.config = SimpleNamespace(latent=sum(model.sizes))
    checkpoint = tmp_path / 'frozen.pt'; checkpoint.write_bytes(b'fixture')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(module, 'load_model', lambda path, device: (model, {}))
    for name, smiles in [('low', 'CCO'), ('control', 'CCN')]:
        data = dict(checkpoint_sha256=digest, graph_spec=spec.to_dict(),
                    encoder_means=torch.zeros(1, model.config.latent).tolist(),
                    molecules=[dict(smiles=smiles, sa_score=2., source='fixture', index=0)])
        (tmp_path / (name + '.json')).write_text(json.dumps(data))
    panel = dict(checkpoint_sha256=digest, graph_spec=spec.to_dict(), decoder_sha256=_model_identity(model),
                 points=[dict(active_slots=[0, 1, 3])])
    (tmp_path / 'parents.json').write_text(json.dumps(panel))
    result = module.prepare_aligned_library(tmp_path / 'low.json', tmp_path / 'control.json',
        tmp_path / 'parents.json', checkpoint, digest, tmp_path / 'aligned',
        bank_size=2, updates=2, learning_rate=.2, decoded_sa_ceiling=10.)
    assert result['usable'] and result['model_state_unchanged']
    assert result['selection']['actual_bank_size'] == 1
    assert result['selection']['shortfall'] == 1
    low = json.loads((tmp_path / 'aligned/low-sa.json').read_text())
    control = json.loads((tmp_path / 'aligned/unfiltered.json').read_text())
    assert low['active_slots'] == control['active_slots'] == [0, 1, 3]
    assert low['molecules'][0]['used_decode']['fixed_occupancy']['valid']
    assert low['molecules'][0]['decoded_smiles'] != control['molecules'][0]['decoded_smiles']
    assert not low['norm_matching']['enabled']
    with pytest.raises(FileExistsError):
        module.prepare_aligned_library(tmp_path / 'low.json', tmp_path / 'control.json',
            tmp_path / 'parents.json', checkpoint, digest, tmp_path / 'aligned')
