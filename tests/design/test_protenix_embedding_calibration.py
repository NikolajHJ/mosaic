from types import SimpleNamespace

import torch

from latentmol.design.protenix_embedding_calibration import build_neighborhood, contact_errors


def test_neighborhood_pairs_keep_center_context_and_use_descent_directions():
    point = torch.tensor([[2., 3.]])
    temperature = 0.65

    def gradient(z, temp, path, *, details):
        assert torch.equal(z, point) and temp == temperature and not details
        vector = torch.tensor([[1., 0.]]) if path == 'relaxed' else torch.tensor([[0., -2.]])
        return {'loss': float(z.sum())}, vector

    def decode(z, temp):
        assert temp == temperature
        return None, None, z

    native = SimpleNamespace(score_graph=lambda z: (z.sum(), {}))
    probe = SimpleNamespace(gradient=gradient, decode=decode, native=native)
    row = build_neighborhood(probe, 'fixture', 50, point, temperature, .1, torch.Generator().manual_seed(1))
    assert row.directions == ('center', 'direct', 'mapped', 'random')
    assert torch.allclose(row.points[1:3], torch.tensor([[1.9, 3.], [2., 3.1]]))
    assert torch.allclose((row.points[1:] - point).norm(dim=1), torch.full((3,), .1))
    assert torch.equal(row.targets, row.points.sum(1))


def test_contact_calibration_separates_offsets_from_wrong_local_direction():
    targets = torch.tensor([0., -0.125, 0.125])
    offset = targets + 0.5
    value, slope = contact_errors(offset, targets, 0.125)
    assert value.item() == 0.25 and slope.item() == 0

    reversed_scores = (-targets).requires_grad_()
    _, slope = contact_errors(reversed_scores, targets, 0.125)
    assert slope.item() == 4
    slope.backward()
    assert reversed_scores.grad[1] > 0 and reversed_scores.grad[2] < 0
