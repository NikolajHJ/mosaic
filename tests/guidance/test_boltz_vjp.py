"""CPU mock workers exercise real pipes, concurrent dispatch and autograd VJPs."""
import os
from pathlib import Path
import time

import pytest
import torch

from latentmol.boltz.contracts import PreparedComplex
from latentmol.experiments import file_sha256
from latentmol.guidance.boltz_vjp import BoltzVJPError, BoltzVJPScorer
from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key
from latentmol.representation import RelaxedGraph
from test_boltz_search import fixture


class _LinearScorer:
    def __init__(self, config, spec):
        self.spec = spec
        self.config = config
        self.identity = dict(model_digest="mock-linear", context_digest=config["prepared_sha256"], objective="dbmol_total_loss")

    def score_at(self, graph, active):
        if self.config.get("fail_evaluate"):
            raise ValueError("intentional native worker failure")
        time.sleep(self.config.get("sleep", .15))
        value = sum((array*torch.arange(array.shape[-1],dtype=array.dtype)*scale).sum()
                    for array,scale in zip((graph.atoms,graph.charges,graph.bonds),(2.,3.,.5)))
        return value, dict(dbmol_total_loss=float(value.detach()), pid=os.getpid())

    def close(self):
        return dict(frozen_unchanged=True)


def _factory(config, spec, source_sha256):
    return _LinearScorer(config, spec)


def _pool(tmp_path, **config_extra):
    prepared, _, _ = fixture()
    path = tmp_path / "prepared.pt"
    prepared.save(path)
    config = dict(prepared=str(path), prepared_sha256=file_sha256(path), **config_extra)
    scorer = BoltzVJPScorer.load(config, prepared.spec, workers=2, source_sha256="a"*64,
        output=tmp_path/"workers", startup_timeout=90, timeout=15, shutdown_timeout=15,
        _factory_reference="test_boltz_vjp:_factory")
    return prepared, scorer


def _leaf_graph(prepared):
    graph = prepared.seed.to_relaxed(prepared.spec)
    return RelaxedGraph(*(value.detach().clone().requires_grad_() for value in (graph.atoms,graph.charges,graph.bonds)))


def test_concurrent_vjps_have_exact_unaveraged_gradients_and_no_parent_cuda(tmp_path):
    prepared, scorer = _pool(tmp_path)
    try:
        graphs = [_leaf_graph(prepared), _leaf_graph(prepared)]
        results = scorer.score_batch(graphs, [prepared.active]*2)
        assert len({row[1]["pid"] for row in results}) == 2
        timing = scorer.last_worker_metrics
        assert max(row["started_monotonic"] for row in timing) < min(row["returned_monotonic"] for row in timing)
        for graph, (value, _) in zip(graphs, results):
            gradients = torch.autograd.grad(value, (graph.atoms,graph.charges,graph.bonds))
            for tensor,gradient,scale in zip((graph.atoms,graph.charges,graph.bonds),gradients,(2.,3.,.5)):
                expected = torch.arange(tensor.shape[-1],dtype=tensor.dtype)*scale
                torch.testing.assert_close(gradient, expected.expand_as(tensor), atol=0, rtol=0)
        assert not torch.cuda.is_initialized()
        assert scorer.prepared.coordinates.device.type == "cpu"
        assert scorer.device_identity["source"] == "native_worker_pool"
        assert scorer.device_identity["device_count"] == 0  # CPU test factory, no fabricated CUDA claim.
        assert scorer.device_identity["total_memory_bytes"] is None
        assert scorer.close()["frozen_unchanged"]
        # Verification leaves both models/workers alive for subsequent waves.
        with torch.no_grad():
            value, _ = scorer.score_at(graphs[0],prepared.active)
        assert not value.requires_grad and not scorer.last_worker_metrics[0]["need_grad"]
        hard = hard_graph_record(prepared.seed)
        root = tmp_path / "hard-score"
        root.mkdir()
        rows = scorer.score_valid_records(root, [dict(smiles="CN",hard_graph=hard,scoring_key=hard_pose_key(hard))])
        assert rows[0]["status"] == "scored" and not scorer.last_worker_metrics[0]["need_grad"]
        assert (root/"fresh/summary.json").exists()
    finally:
        report = scorer.shutdown()
    assert report["cooperative_cleanup_complete"] and not report["requires_job_stop"]
    assert all(p["exitcode"]==0 and not p["alive"] for p in report["processes"])
    assert scorer.shutdown() == report


def test_worker_failure_is_reported_and_peers_leave_cooperatively(tmp_path):
    prepared, scorer = _pool(tmp_path, fail_evaluate=True)
    graph = _leaf_graph(prepared)
    with pytest.raises(BoltzVJPError, match="intentional native worker failure") as caught:
        scorer.score_batch([graph,graph],[prepared.active]*2)
    assert caught.value.report["cooperative_cleanup_complete"]
    assert not caught.value.report["requires_job_stop"]
    assert not any(row["alive"] for row in caught.value.report["processes"])
    assert "intentional native worker failure" in caught.value.report["failure"]["native"]["error"]
    assert (tmp_path/"workers/worker-00/shutdown.json").exists()


def test_parent_rejects_bad_shapes_without_dispatching_gpu_work(tmp_path):
    prepared, scorer = _pool(tmp_path)
    try:
        graph = _leaf_graph(prepared)
        bad = RelaxedGraph(graph.atoms[:,:-1],graph.charges,graph.bonds)
        with pytest.raises(ValueError, match="full-schema"):
            scorer.score_at(bad,prepared.active)
        assert scorer._counter == 0
    finally:
        assert scorer.shutdown()["cooperative_cleanup_complete"]


def test_timeout_is_bounded_and_forced_client_stop_marks_job_unusable(tmp_path):
    prepared, scorer = _pool(tmp_path, sleep=3)
    scorer.timeout, scorer.shutdown_timeout = .1, .2
    graph = _leaf_graph(prepared)
    with pytest.raises(BoltzVJPError, match="timed out") as caught:
        scorer.score_batch([graph,graph],[prepared.active]*2)
    assert caught.value.report["requires_job_stop"]
    assert any(row["forced_termination"] for row in caught.value.report["processes"])
    assert not any(row["alive"] for row in caught.value.report["processes"])
