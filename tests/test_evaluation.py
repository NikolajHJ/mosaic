"""Evaluation regressions. Execute on HPC only, including CPU test allocations."""

from types import SimpleNamespace
import json
import subprocess

import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.evaluate import neighborhood_metrics, prior_metrics, reconstruction_metrics, summarize_neighborhood
from latentmol.losses import LossConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def _graph(smiles, spec, count=1):
    """Build a repeated, chemically valid target without model-dependent labels."""
    encoded = encode_smiles(smiles, spec)
    return HardGraphBatch(
        torch.tensor(encoded["atom_types"], dtype=torch.long).unsqueeze(0).repeat(count, 1),
        torch.tensor(encoded["charges"], dtype=torch.long).unsqueeze(0).repeat(count, 1),
        torch.tensor(encoded["bond_types"], dtype=torch.long).unsqueeze(0).repeat(count, 1, 1),
    )


class _FixedModel:
    """A constant decoder makes reconstruction errors separable from perturbation."""

    def __init__(self, spec, prediction, mu_rows=None, logvar_value=0.0):
        self.spec = spec
        self.prediction = prediction
        self.mu_rows = mu_rows
        self.logvar_value = logvar_value
        self.offset = 0
        self.decode_sizes = []

    def encode(self, graph):
        count = graph.atoms.shape[0]
        return torch.zeros(count, 2), torch.full((count, 2), self.logvar_value)

    def decode(self, codes):
        count = len(codes)
        self.decode_sizes.append(count)
        predicted = self.prediction
        return DecoderOutput(
            F.one_hot(predicted.atoms[:1].expand(count, -1), len(self.spec.atomic_numbers)).float() * 40.0,
            F.one_hot(predicted.charges[:1].expand(count, -1), len(self.spec.formal_charges)).float() * 40.0,
            F.one_hot(predicted.bonds[:1].expand(count, -1, -1), len(self.spec.bond_names)).float() * 40.0,
        )

    def __call__(self, graph, sample_latent=False):
        assert sample_latent is False  # This evaluator explicitly reports mu reconstruction.
        count = graph.atoms.shape[0]
        if self.mu_rows is None:
            mu = torch.zeros(count, 2)
        else:
            mu = self.mu_rows[self.offset:self.offset + count]
            self.offset += count
        return SimpleNamespace(decoded=self.decode(mu), mu=mu,
                               logvar=torch.full_like(mu, self.logvar_value))


def test_constant_wrong_decode_is_not_a_perturbation_change():
    """An existing valid reconstruction error must not count as a latent change."""
    metrics = summarize_neighborhood(["C", "O"], ["CC", "CC"], ["CC", "CC"])
    assert metrics["different_from_input_valid_fraction"] == 1.0
    assert metrics["valid_changed_from_baseline_fraction"] == 0.0
    assert metrics["baseline_exact_reconstruction_fraction"] == 0.0
    assert metrics["invalid_to_valid_rescue_count"] == 0
    assert metrics["invalid_to_valid_rescue_fraction"] is None


def test_rescue_denominator_and_empty_molecule_validity():
    """Rescues condition on invalid baselines; empty strings never earn validity."""
    metrics = summarize_neighborhood(["C", "O", "N"], [None, "", "N"], ["C", None, "N"],
                                     training_smiles={"N"})
    assert metrics["validity"] == pytest.approx(2 / 3)
    assert metrics["valid_changed_from_baseline_count"] == 1
    assert metrics["baseline_invalid_count"] == 2
    assert metrics["invalid_to_valid_rescue_fraction"] == 0.5
    assert metrics["novel_vs_training_split_fraction"] == 0.5
    empty = summarize_neighborhood(["C"], [None], [""])
    assert empty["validity"] == 0.0
    assert empty["unique_valid_fraction"] is None
    with pytest.raises(ValueError, match="nonempty"):
        summarize_neighborhood([], [], [])


def test_neighborhood_uses_own_baseline_and_bounded_decoder_batches():
    """A large constant-decoder neighborhood stays within 32-graph allocations."""
    spec = GraphSpec(max_nodes=4)
    target = _graph("C", spec, count=65)
    model = _FixedModel(spec, _graph("CC", spec))
    metrics, examples = neighborhood_metrics(model, [target], "cpu", seed=17,
                                             scales=(0.0, 0.2), limit=65)
    assert model.decode_sizes == [32, 32, 1, 32, 32, 1]
    assert len(examples) == 130
    for item in metrics:
        assert item["validity"] == 1.0
        assert item["different_from_input_valid_fraction"] == 1.0
        assert item["valid_changed_from_baseline_fraction"] == 0.0
        assert item["baseline_exact_reconstruction_fraction"] == 0.0
    assert all(item["baseline_decoded"] == "CC" for item in examples)


def test_prior_sampling_needs_no_encoder_and_does_not_reward_duplicates():
    """Prior generation is reproducible and distinct from encoded neighborhoods."""
    spec = GraphSpec(max_nodes=4)

    class PriorModel(_FixedModel):
        config = SimpleNamespace(latent=2)

        def encode(self, graph):
            raise AssertionError("Prior sampling must not encode input molecules")

        def decode(self, codes):
            self.seen_codes.append(codes.clone())
            return super().decode(codes)

    model = PriorModel(spec, _graph("CC", spec))
    model.seen_codes = []
    first, examples = prior_metrics(model, "cpu", 17, 65, training_smiles={"CC"})
    first_codes = torch.cat(model.seen_codes)
    model.seen_codes = []
    torch.manual_seed(919)
    second, _ = prior_metrics(model, "cpu", 17, 65, training_smiles={"CC"})
    torch.testing.assert_close(first_codes, torch.cat(model.seen_codes), rtol=0, atol=0)
    assert first == second
    assert first["valid_count"] == 65 and first["unique_valid_count"] == 1
    assert first["novel_vs_training_split_count"] == 0
    assert len(examples) == 65
    assert model.decode_sizes == [32, 32, 1, 32, 32, 1]


def test_bond_precision_counts_extra_predicted_atoms_and_charge_errors():
    """A false extra bond lowers precision even when every target bond is present."""
    spec = GraphSpec(max_nodes=4)
    target = _graph("CC", spec)
    predicted = _graph("CCC", spec)
    predicted.charges[0, 0] = spec.formal_charges.index(1)
    model = _FixedModel(spec, predicted)
    metrics, _ = reconstruction_metrics(model, [target], LossConfig(), "cpu", chemistry_limit=0,
                                        training_mode="ae")
    assert metrics["target_bonded_pairs"] == 1
    assert metrics["predicted_bonded_pairs"] == 2
    assert metrics["bonded_pair_precision"] == 0.5
    assert metrics["bonded_pair_type_precision"] == 0.5
    assert metrics["bonded_pair_type_recall"] == 1.0
    assert metrics["active_atom_charge_accuracy"] == 0.5
    assert metrics["validity"] is None  # Chemistry was not evaluated.


def test_empty_prediction_is_neither_valid_nor_reconstructed():
    """A decoder emitting only NO_ATOM has zero chemical validity."""
    spec = GraphSpec(max_nodes=4)
    target = _graph("CC", spec)
    empty = HardGraphBatch(torch.zeros_like(target.atoms), target.charges.clone(),
                           torch.zeros_like(target.bonds))
    metrics, _ = reconstruction_metrics(_FixedModel(spec, empty), [target], LossConfig(), "cpu",
                                        training_mode="ae")
    assert metrics["validity"] == metrics["exact_reconstruction"] == 0.0
    assert metrics["active_atom_charge_accuracy"] == 0.0
    assert metrics["bonded_pair_type_precision"] is None
    assert metrics["bonded_pair_type_recall"] == 0.0


def test_graded_diagnostics_retain_every_attempt_and_keep_sampling_modes_separate():
    """Plumbing must preserve invalid labels and the exact chemistry denominator."""
    spec = GraphSpec(max_nodes=4)
    target = _graph("CC", spec, count=10)
    broken = _graph("CO", spec)
    broken.bonds[0, 0, 1] = broken.bonds[0, 1, 0] = 3
    model = _FixedModel(spec, broken)
    metrics, examples = reconstruction_metrics(model, [target], LossConfig(), "cpu",
                                               chemistry_limit=9, training_mode="ae", graded=True, retain_graphs=True)
    quality = metrics["graded_chemistry"]
    assert quality["attempts"] == 9
    assert 0 < quality["mixture"]["score"] < 100
    assert all(x["predicted_graph"] and not x["graded_chemistry"]["valid"] for x in examples)
    assert len(examples) == 9  # No first-eight failure truncation.
    neighborhoods, neighbors = neighborhood_metrics(model, [target], "cpu", 17,
                                                    scales=(0., .2), limit=3)
    assert [x["graded_chemistry"]["attempts"] for x in neighborhoods] == [3, 3]
    assert all("predicted_graph" in x["graded_chemistry"] for x in neighbors)


def test_posterior_diagnostics_use_global_mu_variance_and_label_ae():
    """Global latent variance spans batches; an AE's unused posterior is not scored."""
    spec = GraphSpec(max_nodes=4)
    target = _graph("CC", spec)
    mu_rows = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
    model = _FixedModel(spec, target, mu_rows=mu_rows)
    metrics, _ = reconstruction_metrics(model, [target, target], LossConfig(), "cpu",
                                        training_mode="vae")
    diagnostics = metrics["latent_diagnostics"]
    assert metrics["reconstruction_latent"] == "deterministic_mu"
    assert diagnostics["mu_variance_by_dimension"] == [0.0, 1.0]
    assert diagnostics["active_dimensions"] == 1
    assert diagnostics["active_variance_threshold"] == 0.01
    assert diagnostics["posterior_kl_mean"] == 2.5
    assert diagnostics["posterior_std_mean"] == 1.0
    ae = _FixedModel(spec, target, logvar_value=1000.0)
    ae_metrics, _ = reconstruction_metrics(ae, [target], LossConfig(), "cpu", training_mode="ae")
    assert ae_metrics["latent_diagnostics"]["posterior_statistics_applicable"] is False
    assert ae_metrics["latent_diagnostics"]["posterior_kl_mean"] is None
    assert ae_metrics["latent_diagnostics"]["posterior_std_mean"] is None


@pytest.mark.parametrize("smoke", [False, True])
def test_workflow_reuses_process_and_exact_training_run(monkeypatch, tmp_path, smoke):
    """Exclusive-process GPUs must not acquire a second evaluation CUDA context."""
    from latentmol import train_and_evaluate as workflow

    data_path, run_root = tmp_path / "data", tmp_path / "runs"
    chosen_run = run_root / "exact-returned-training-run"
    config = SimpleNamespace(validation_limit=11, train_limit=7, seed=23, device="cuda")
    train_calls, evaluation_calls = [], []

    def fake_train(received_config, received_data, received_root):
        train_calls.append((received_config, received_data, received_root))
        return chosen_run

    def fake_evaluate(checkpoint, data, root, **options):
        evaluation_calls.append((checkpoint, data, root, options))
        # Evaluation produces a different run path; it must never become the
        # source of the next checkpoint or the workflow's returned training path.
        return root / f"evaluation-{len(evaluation_calls)}"

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("Evaluation must reuse the training process")

    monkeypatch.setattr(workflow, "train", fake_train)
    monkeypatch.setattr(workflow, "evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(subprocess, "Popen", forbidden_subprocess)
    result = workflow.train_and_evaluate(config, data_path, run_root, smoke=smoke)
    assert result is chosen_run
    assert train_calls == [(config, data_path, run_root)]
    expected = [("best.pt", "val", 11), ("last.pt", "val", 11)]
    if smoke:
        expected.append(("last.pt", "train", 7))
    assert evaluation_calls == [
        (chosen_run / name, data_path, run_root,
         {"split": split, "limit": limit, "seed": 23, "device": "cuda"})
        for name, split, limit in expected
    ]


def test_reusable_evaluator_records_and_propagates_failure(monkeypatch, tmp_path):
    """Refactoring the CLI must retain durable exception capture for direct callers."""
    from latentmol import evaluate

    monkeypatch.chdir(tmp_path)

    def fail_dataset(*args, **kwargs):
        raise RuntimeError("sentinel dataset failure")

    monkeypatch.setattr(evaluate, "GraphDataset", fail_dataset)
    with pytest.raises(RuntimeError, match="sentinel dataset failure"):
        evaluate.evaluate_checkpoint(tmp_path / "best.pt", tmp_path / "data",
                                     tmp_path / "runs", device="cpu")
    run_paths = list((tmp_path / "runs").iterdir())
    assert len(run_paths) == 1
    record = json.loads((run_paths[0] / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["config"]["checkpoint"] == str(tmp_path / "best.pt")
    assert "sentinel dataset failure" in (run_paths[0] / "traceback.txt").read_text()
