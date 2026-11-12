import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mosaic_defog import Graph, Prediction


def test_torch_graph_gradient_is_live_and_target_is_constant():
    graph = Graph(
        torch.full((3, 6), 1 / 6, requires_grad=True),
        torch.full((3, 5), 1 / 5, requires_grad=True),
    )
    target = Prediction(
        np.eye(6, dtype=np.float32)[[0, 0, 0]],
        np.eye(5, dtype=np.float32)[[0, 0, 0]],
        0.8,
    )
    loss = target.torch_loss(graph)
    loss.backward()
    np.testing.assert_allclose(
        graph.X.grad.numpy(), 2 * (graph.X.detach().numpy() - target.X) / 3
    )
    np.testing.assert_allclose(
        graph.E_upper.grad.numpy(),
        2 * (graph.E_upper.detach().numpy() - target.E_upper) / 3,
    )
