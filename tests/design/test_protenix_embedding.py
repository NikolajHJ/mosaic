from types import SimpleNamespace
from pathlib import Path

import torch
import numpy as np
import pytest

from latentmol.design.protenix_embedding import HiddenChemicalBridge, projection_loss
from latentmol.design.latent_protenix import LatentDesignConfig, evaluate_point
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.design import protenix_embedding


def test_selected_hidden_projection_is_symmetric_finite_and_attached():
    torch.manual_seed(1)
    point = torch.randn(2, 4, requires_grad=True)
    atom = point[:, None, :].expand(-1, 5, -1)
    charge = torch.cat((atom, atom.square()), -1)
    pair = atom[:, :, None, :] + atom[:, None, :, :]
    features = SimpleNamespace(atom_hidden=atom, charge_input=charge, bond_input=pair)
    bank = {key: dict(width=3, prototypes=[[0., 0., 0.], [1., 2., 3.]])
            for key in ('atom_element', 'atom_charge', 'pair_presence')}
    bridge = HiddenChemicalBridge(bank, dict(atom_element=4, atom_charge=8, pair_presence=4), 7)
    indices = torch.tensor([[0, 2, 4], [1, 2, 3]])
    values = bridge(features, indices)
    assert values['atom_element'].shape == (2, 3, 3)
    assert torch.equal(values['pair_presence'], values['pair_presence'].transpose(1, 2))
    assert not values['pair_presence'][:, torch.arange(3), torch.arange(3)].any()
    loss = sum(v.square().mean() for v in values.values())
    gradient, = torch.autograd.grad(loss, point)
    assert gradient.isfinite().all() and gradient.norm() > 0


def test_projection_loss_balances_present_and_none_pairs():
    bank = {key: dict(width=1, prototypes=[[0.], [2.]])
            for key in ('atom_element', 'atom_charge', 'pair_presence')}
    labels = {'atom_element': torch.tensor([[1, 0, 0]]), 'atom_charge': torch.tensor([[0, 1, 0]]),
              'pair_presence': torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]])}
    prediction = {k: torch.zeros((*v.shape, 1), requires_grad=True) for k, v in labels.items()}
    loss, parts = projection_loss(prediction, labels, bank)
    assert parts['pair_presence'].item() == 2.
    loss.backward()
    assert all(v.grad is not None for v in prediction.values())


def test_latent_loop_uses_explicit_hidden_path(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    class Decoder:
        def decode_with_features(self, z):
            h = z[:, None, :].expand(-1, 3, -1)
            logits = DecoderOutput(h[..., :1].expand(-1, -1, len(spec.atomic_numbers)),
                h[..., :1].expand(-1, -1, len(spec.formal_charges)),
                (h[:, :, None, :1]+h[:, None, :, :1]).expand(-1,-1,-1,len(spec.bond_names)))
            return logits, SimpleNamespace(atom_hidden=h)
    decoder = Decoder(); decoder.spec = spec
    class Scorer:
        uses_decoder_hidden = True
        def score_hidden(self, features, graph, active, *, step):
            return features.atom_hidden.sum(), {'input_boundary': 'hidden'}
    monkeypatch.setattr('latentmol.design.latent_protenix.structural_target',
        lambda *args: (None, {'status': 'already_valid'}))
    result = evaluate_point(decoder, Scorer(), torch.ones(1, 4), torch.ones(3, dtype=torch.bool),
                            LatentDesignConfig(occupied_atoms=3), 0)
    assert torch.equal(result['binding_gradient'], torch.full((1, 4), 3.))
    assert not result['chemistry_gradient'].any()


def test_config_string_checkpoint_path_is_normalized_before_hash_and_load(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'epoch45.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    seen = []
    def load(path, device):
        assert isinstance(path, Path) and path == checkpoint and device == 'cpu'
        seen.append(path)
        return torch.nn.Linear(2, 2), {'epoch': 45}
    monkeypatch.setattr(protenix_embedding, 'load_model', load)
    model = protenix_embedding.load_frozen_decoder({'path': str(checkpoint),
        'sha256': protenix_embedding.file_sha256(checkpoint)})
    assert seen == [checkpoint] and not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_saved_latent_start_uses_exact_row_and_declared_slots(tmp_path):
    path=tmp_path/'block.npz'
    np.savez(path,steps=np.array([179,180,181]),latent=np.arange(12,dtype=np.float32).reshape(3,4))
    config={'start':dict(kind='saved_trajectory_latent',raw_block=str(path),
        raw_block_sha256=protenix_embedding.file_sha256(path),step=180,active_slots=[0,2])}
    state=protenix_embedding.load_start_state(config,SimpleNamespace(spec=GraphSpec(max_nodes=3)))
    assert torch.equal(state['latent'],torch.tensor([[4.,5.,6.,7.]]))
    assert state['active'].tolist()==[True,False,True] and state['step']==180
    config['start']['raw_block_sha256']='0'*64
    with pytest.raises(ValueError,match='block changed'):
        protenix_embedding.load_start_state(config,SimpleNamespace(spec=GraphSpec(max_nodes=3)))
