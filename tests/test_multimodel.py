"""Tiny arithmetic fixtures for orchestration, never scientific scorer evidence.

These may run locally under the user's explicit smoke-test exception. No external
checkpoint, training data, native package, GPU or network connection is used.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.scoring import BranchCapabilities, BranchManifest, InputSupport, evaluate_branches
from latentmol.multimodel.config import Objective, Study, digest
from latentmol.multimodel.engine import Counters, scalar_loss, score_graph
from latentmol.multimodel.runtime import BranchRuntime
from latentmol.multimodel.study import (GRADIENT_POLICY, LEGACY_GRADIENT_POLICY,
                                       Problem, hard_digest, harden, module_digest, plan,
                                       exploratory_validate, require_gate, run_arm, tensor_digest, validate)
from latentmol.multimodel.confirmation import (candidates, diagnose, endpoint_candidates,
                                               generation_provenance, validate_response)


class TinyDecoder(nn.Module):
    """A deterministic two-dimensional toy decoder with ordinary input gradients."""
    def __init__(self):
        super().__init__()
        self.spec = GraphSpec(max_nodes=2, atomic_numbers=(0, 6, 7), formal_charges=(-1, 0, 1),
                              bond_names=("NONE", "SINGLE", "DOUBLE"))
        self.config = SimpleNamespace(latent=2)
        self.register_buffer("constant", torch.tensor(4.0, dtype=torch.float64))
        self.eval()

    def decode(self, z):
        atoms = torch.stack((z[:, 0] * 0, self.constant + z[:, 0], z[:, 1]), -1)[:, None].expand(-1, 2, -1)
        charges = torch.stack((z[:, 1], self.constant + z[:, 0] * 0, -z[:, 1]), -1)[:, None].expand(-1, 2, -1)
        bonds = torch.stack((z[:, 0] * 0, self.constant + z[:, 0], z[:, 1]), -1)[:, None, None].expand(-1, 2, 2, -1)
        return DecoderOutput(atoms, charges, bonds)


class ToyBranch(nn.Module):
    """Mock arithmetic oracle, not a molecular prior or a fitted reward surrogate."""
    def __init__(self, spec, factor=1.0, lossy=False, detached=False):
        super().__init__()
        self.spec, self.lossy, self.detached = spec, lossy, detached
        self.register_buffer("factor", torch.tensor(factor, dtype=torch.float64))
        self.seen = []
        self.manifest = BranchManifest(
            name="toy", graph_spec_sha256=spec.fingerprint(), model_source_version="fixture",
            model_weights_sha256="a" * 64, adapter_version="fixture", adapter_source_sha256="b" * 64,
            capabilities=BranchCapabilities("relaxed_graph", "variable", "pathwise",
                ("atoms", "charges", "bonds"), (6, 7), (-1, 0, 1), spec.bond_names,
                "no native context", "toy arithmetic only"))
        self.eval()

    def assess(self, graph):
        return InputSupport("lossy", ("fixture loss",)) if self.lossy else InputSupport("exact")

    def forward(self, graph):
        self.seen.append(id(graph))
        value = self.factor * (graph.atoms[..., 2].square().sum() + graph.charges[..., 2].sum()
                               + .5 * graph.bonds[..., 2].sum())
        return {"score": value.detach() if self.detached else value}


def configuration():
    return Study.from_dict({"version": 1, "name": "tiny", "device": "cpu",
        "decoder": {"checkpoint": "unused", "sha256": "a" * 64},
        "models": {"a": {"plugin": "boltz2", "objective": {"output": "score", "weight": 2}},
                   "b": {"plugin": "nesso1", "objective": {"output": "score", "direction": "maximize", "weight": .5, "scale": 2}}},
        "arms": {"a": {"models": ["a"]}, "both": {"models": ["a", "b"]},
                 "control": {"models": ["a"], "control": "no_update"}},
        "optimizer": {"steps": 2, "learning_rate": .01, "trust_radius": .1},
        "diagnostics": {"hard_samples": 2, "fd_rtol": .005}})


def problem_fixture():
    decoder = TinyDecoder()
    start = torch.zeros(1, 2, dtype=torch.float64)
    active = torch.ones(2, dtype=torch.bool)
    graph = graph_from_decoder(decoder.decode(start), decoder.spec, fixed_active=active)
    seed = harden(graph, decoder.spec)
    runtimes = {}
    for name, factor in (("a", 1), ("b", 3)):
        branch = ToyBranch(decoder.spec, factor)
        runtimes[name] = BranchRuntime(branch, seed, active,
            lambda branch=branch: branch(seed.to_relaxed(decoder.spec)), {"fixture": True})
    identity = {"decoder_state_sha256": module_digest(decoder), "seed_sha256": hard_digest(seed),
                "branches": {name: {"state_sha256": module_digest(runtime.branch)} for name, runtime in runtimes.items()}}
    return Problem(decoder, runtimes, seed, active, start, identity)


def test_exploratory_smoke_runs_without_native_or_fd_qualification(tmp_path):
    problem, study = problem_fixture(), configuration()
    for runtime in problem.runtimes.values():
        runtime.native_reference = lambda: pytest.fail("Exploratory smoke called native parity oracle")
    # A decoder that does not exactly reconstruct its seed still supplies valid
    # model-shaped relaxed inputs; this is an experiment, not a parity claim.
    problem.start = torch.ones_like(problem.start) * 10
    report = exploratory_validate(problem, study, ["both"], Counters(), tmp_path / "smoke.json")
    assert report["passed"] and report["derivative_qualification"] == "not_run"
    assert set(report["models"]) == {"a", "b"}
    with pytest.raises(ValueError):
        require_gate(problem, study, tmp_path / "smoke.json")


def test_exploratory_smoke_records_flat_attached_path(tmp_path):
    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    branch.forward = lambda graph: {"score": graph.atoms.sum() * 0}
    report = exploratory_validate(problem, study, ["a"], Counters(), tmp_path / "smoke.json")
    assert report["passed"] and report["models"]["a"]["gradient_norm"] == 0
    assert report["observations"][0]["status"] == "flat_at_start"


def test_exploratory_smoke_distinguishes_small_nonzero_signal_from_flat(tmp_path):
    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    branch.forward = lambda graph: {"score": graph.atoms[..., 2].sum() * 1e-13}
    report = exploratory_validate(problem, study, ["a"], Counters(), tmp_path / "smoke.json")
    assert report["passed"] and 0 < report["models"]["a"]["gradient_norm"] <= 1e-12
    assert report["observations"][0]["status"] == "low_signal"
    assert "small nonzero" in report["observations"][0]["message"]


@pytest.mark.parametrize("invalid", ["detached", "nonfinite", "mutated_weights"])
def test_exploratory_smoke_keeps_real_execution_failures(tmp_path, invalid):
    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    if invalid == "detached":
        branch.detached = True
    elif invalid == "nonfinite":
        branch.forward = lambda graph: {"score": graph.atoms.sum() * float("nan")}
    else:
        branch.factor.add_(1)
    with pytest.raises(ValueError):
        exploratory_validate(problem, study, ["a"], Counters(), tmp_path / "smoke.json")
    assert json.loads((tmp_path / "smoke.json").read_text())["status"] == "failed"


def test_cli_run_defaults_to_exploratory_smoke_without_preexisting_gate(tmp_path, monkeypatch):
    from latentmol.multimodel import __main__ as cli
    problem, study = problem_fixture(), configuration()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture")
    monkeypatch.setattr(cli.Study, "load", lambda _: study)
    monkeypatch.setattr(cli, "load_problem", lambda *_: problem)
    monkeypatch.setattr(cli, "validate", lambda *_args, **_kwargs: pytest.fail("Detailed qualification requested"))
    monkeypatch.setattr(cli, "run_arm", lambda *_args, **_kwargs: {"status": "fixture_completed"})
    monkeypatch.setattr("sys.argv", ["multimodel", "run", "--config", str(config_path),
                                    "--run-root", str(tmp_path / "runs"), "--arms", "a"])
    cli.main()
    record_path, = (tmp_path / "runs").glob("*/run.json")
    record = json.loads(record_path.read_text())
    gate = json.loads((record_path.parent / "engineering_gate.json").read_text())
    assert record["status"] == "completed" and record["research_profile"] == "exploratory"
    assert gate["passed"] and gate["derivative_qualification"] == "not_run"


def test_atom_budget_moves_against_an_expansion_reward(tmp_path):
    problem = problem_fixture()
    problem.active = None
    problem.runtimes["a"].branch.forward = lambda graph: {"score": -.01 * graph.expected_size.sum()}
    config = configuration().to_dict()
    assert "atom_budget" not in config["arms"]["a"]
    config["arms"]["budget"] = {"models": ["a"], "atom_budget": {
        "target": 1, "tolerance": .1, "weight": 1., "outside_weight": .02}}
    with pytest.raises(ValueError, match="requires variable occupancy"):
        Study.from_dict(config)
    config["occupancy"] = "variable"
    study = Study.from_dict(config)
    assert Study.from_dict(study.to_dict()).fingerprint() == study.fingerprint()
    plain, guarded = [], []
    run_arm(problem, study, "a", Counters(), tmp_path / "plain", plain.append)
    result = run_arm(problem, study, "budget", Counters(), tmp_path / "guarded", guarded.append)
    assert result["atom_budget"]["original_slots"] == [0]
    initial = guarded[0]["atom_budget"]
    count = initial["expected_atoms"]
    assert count == plain[0]["expected_atoms"]
    assert initial["outside_occupancy"] == pytest.approx(count / 2)
    assert initial["weighted_count_loss"] == pytest.approx((count - 1.1) ** 2)
    assert initial["weighted_outside_loss"] == pytest.approx(.02 * count / 2)
    assert guarded[0]["weighted_loss"] == pytest.approx(
        plain[0]["weighted_loss"] + initial["weighted_count_loss"] + initial["weighted_outside_loss"])
    assert guarded[-1]["expected_atoms"] < count < plain[-1]["expected_atoms"]


def test_sparse_observations_preserve_updates_and_both_endpoints(tmp_path):
    problem, study = problem_fixture(), configuration()
    study = replace(study, optimizer=replace(study.optimizer, steps=12))
    events = []
    sparse = run_arm(problem, study, "a", Counters(), tmp_path / "sparse", events.append,
                     observation_interval=10)
    full = run_arm(problem, study, "a", Counters(), tmp_path / "full", lambda _: None)
    assert len(events) == 13 and sum(row["sampled_observation"] for row in events) == 3
    assert sorted(p.name for p in (tmp_path / "sparse").glob("state-*.pt")) == [
        "state-0000.pt", "state-0010.pt", "state-0012.pt"]
    for field in ("z",):
        a = torch.load(tmp_path / "sparse" / "state-0012.pt", weights_only=True)[field]
        b = torch.load(tmp_path / "full" / "state-0012.pt", weights_only=True)[field]
        assert torch.equal(a, b)
    assert sparse["steps"] == full["steps"] == 12


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_objective_rejects_bad_weight(value):
    with pytest.raises(ValueError):
        Objective("score", weight=value)


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"direction": "up"}, {"offset": "oops"}])
def test_objective_contract(kwargs):
    with pytest.raises(ValueError):
        Objective("score", **kwargs)


def test_plan_is_lazy_and_disables_zero_weight_model(monkeypatch):
    import latentmol.multimodel.registry as registry
    monkeypatch.setattr(registry, "import_module", lambda *_: pytest.fail("Plan imported native model"))
    study = configuration()
    study = replace(study, models={**study.models, "b": replace(study.models["b"], objective=Objective("score", weight=0))})
    assert plan(study, ["both"])["models_to_load"] == ["a"]
    with pytest.raises(ValueError, match="Unknown model"):
        plan(replace(study, models={"a": replace(study.models["a"], plugin="misspelled"), "b": study.models["b"]}), ["a"])


def test_yaml_paths_relative_to_config(tmp_path, monkeypatch):
    import yaml
    raw = configuration().to_dict()
    raw["models"]["a"]["options"] = {"prepared": "assets/input.pt"}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = Study.load(path)
    assert loaded.decoder["checkpoint"] == str(tmp_path / "unused")
    assert loaded.models["a"].options["prepared"] == str(tmp_path / "assets/input.pt")


def test_weighted_gradients_match_exact_joint_scalar_and_share_graph():
    problem, study = problem_fixture(), configuration()
    point = problem.start.clone().requires_grad_(True)
    graph = problem.graph(point, 1)
    counters = Counters()
    report, actual = score_graph(graph, point, runtimes=problem.runtimes, models=study.models,
                                weights={"a": 2, "b": .5}, spec=problem.spec, counters=counters,
                                purpose="test", gradients=True)
    outputs = {name: runtime.branch(graph) for name, runtime in problem.runtimes.items()}
    expected_loss = sum(weight * scalar_loss(outputs[name], study.models[name].objective)[0]
                        for name, weight in (("a", 2), ("b", .5)))
    expected, = torch.autograd.grad(expected_loss, point)
    torch.testing.assert_close(actual, expected)
    assert problem.runtimes["a"].branch.seen[0] == problem.runtimes["b"].branch.seen[0] == id(graph)
    assert report["gradient_cosines"]["a|b"] == pytest.approx(-1)
    assert counters.values["a"]["test"]["backward_completed"] == 1


def test_per_branch_loss_consent_does_not_spread():
    problem = problem_fixture()
    for runtime in problem.runtimes.values():
        runtime.branch.lossy = True
    with pytest.raises(ValueError, match="b: lossy"):
        evaluate_branches(problem.graph(problem.start, 1), {n: r.branch for n, r in problem.runtimes.items()},
                          spec=problem.spec, allow_lossy={"a": True})
    assert not any(r.branch.seen for r in problem.runtimes.values())


def test_detached_objective_and_unreduced_outputs_fail():
    problem, study = problem_fixture(), configuration()
    problem.runtimes["a"].branch.detached = True
    point = problem.start.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="detached"):
        score_graph(problem.graph(point, 1), point, runtimes=problem.runtimes, models=study.models,
                    weights={"a": 1}, spec=problem.spec, counters=Counters(), purpose="test", gradients=True)
    with pytest.raises(ValueError, match="scalar"):
        scalar_loss({"score": torch.ones(2, 2)}, Objective("score"))


def test_assessment_cannot_mutate_the_shared_graph():
    problem, study = problem_fixture(), configuration()
    def mutate(graph):
        graph.charges.add_(0)
        return InputSupport("exact")
    problem.runtimes["a"].branch.assess = mutate
    with pytest.raises(ValueError, match="during eligibility"):
        score_graph(problem.graph(problem.start, 1), problem.start, runtimes=problem.runtimes,
                    models=study.models, weights={"a": 1}, spec=problem.spec,
                    counters=Counters(), purpose="test", gradients=False)


def test_repository_templates_plan_without_assets():
    root = Path(__file__).resolve().parents[1]
    for filename, arm, names in (("multimodel_boltz.yaml", "boltz_only", ["boltz2"]),
                                ("multimodel_ensemble.yaml", "all_three", ["boltz2", "nesso1", "protenix"])):
        value = plan(Study.load(root / "configs" / filename), [arm])
        assert value["models_to_load"] == names


def test_gate_records_exact_identity_and_rejects_config_change(tmp_path):
    problem, study = problem_fixture(), configuration()
    path = tmp_path / "gate.json"
    gate = validate(problem, study, ["both"], Counters(), path)
    assert gate["passed"] and len(gate["models"]["a"]["finite_differences"]) == 6
    assert set(gate["models"]["a"]["channel_gradient_norms"]) == {"atoms", "charges", "bonds"}
    require_gate(problem, study, path)
    with pytest.raises(ValueError, match="exact assets"):
        require_gate(problem, replace(study, temperature=.9), path)


def test_float32_producer_and_reader_share_the_same_diagnostic_contraction(tmp_path):
    problem, study = problem_fixture(), configuration()
    problem.decoder.float()
    problem.start = problem.start.float()
    problem.identity["decoder_state_sha256"] = module_digest(problem.decoder)
    for name, runtime in problem.runtimes.items():
        runtime.branch.float()
        problem.identity["branches"][name]["state_sha256"] = module_digest(runtime.branch)
    path = tmp_path / "float32-gate.json"
    report = validate(problem, study, ["both"], Counters(), path)
    require_gate(problem, study, path)
    for model in report["models"].values():
        projections = {row["direction"]: row["reverse_projection"] for row in model["pointwise_ad"]["rows"]}
        assert all(row["analytic"] == projections[row["direction"]] for row in model["finite_differences"])


@pytest.mark.parametrize("policy", [GRADIENT_POLICY, LEGACY_GRADIENT_POLICY])
def test_stationary_attached_paths_are_valid_under_new_policy_but_preserve_legacy_qualification(tmp_path, policy):
    problem, study = problem_fixture(), configuration()
    problem.runtimes["a"].branch.factor.zero_()
    problem.identity["branches"]["a"]["state_sha256"] = module_digest(problem.runtimes["a"].branch)
    path = tmp_path / "gate.json"
    if policy == LEGACY_GRADIENT_POLICY:
        with pytest.raises(ValueError, match="legacy nonzero|zero latent"):
            validate(problem, study, ["both"], Counters(), path, gradient_policy=policy)
        assert json.loads(path.read_text())["passed"] is False
    else:
        report = validate(problem, study, ["both"], Counters(), path)
        assert report["passed"] and report["models"]["a"]["latent_gradient_norm"] == 0
        assert report["models"]["a"]["optimization_qualification"]["informative_total"] is False
        assert not report["full_network_fd_required"] and not report["useful_gradient_validated"]
        require_gate(problem, study, path)


def test_state_mutation_caught():
    problem = problem_fixture()
    problem.runtimes["a"].branch.factor.add_(1)
    with pytest.raises(ValueError, match="state|weights"):
        problem.verify()


def test_arms_share_initial_sampling_and_controls_do_not_move(tmp_path):
    problem, study = problem_fixture(), configuration()
    for arm in ("a", "control"):
        result = run_arm(problem, study, arm, Counters(), tmp_path / "arms" / arm, lambda x: None)
        assert result["status"] == "completed" and result["attempted"] == 9
    def first(arm):
        return [json.loads(row)["graph_sha256"] for row in (tmp_path / "arms" / arm / "candidates.jsonl").read_text().splitlines()][:3]
    assert first("a") == first("control")
    final = torch.load(tmp_path / "arms/control/state-0002.pt", weights_only=True)
    torch.testing.assert_close(final["z"], problem.start)
    moved = torch.load(tmp_path / "arms/a/state-0002.pt", weights_only=True)["z"]
    assert float(moved.norm()) > 0 and float(moved.norm()) <= study.optimizer.trust_radius
    from latentmol.quality import QualityScoreConfig
    results = diagnose(tmp_path, tmp_path / "quality.json", QualityScoreConfig())
    assert len(results["records"]) == 18
    assert results["records"][0]["quality"]["status"] == "available"


def test_variable_argmax_uses_conditional_bonds():
    spec = GraphSpec(max_nodes=2, atomic_numbers=(0, 6), formal_charges=(0,), bond_names=("NONE", "SINGLE"))
    decoded = DecoderOutput(torch.tensor([[[0., .04], [0., .04]]]), torch.zeros(1, 2, 1),
                            torch.tensor([[[[0., 2.], [0., 2.]], [[0., 2.], [0., 2.]]]]))
    graph = decoded.probabilities()
    assert graph.bonds[0, 0, 1].argmax() == 0  # occupancy gating would wrongly remove it
    assert harden(graph, spec, decoded).bonds[0, 0, 1] == 1
    with pytest.raises(ValueError, match="conditional"):
        harden(graph, spec)


def test_native_identity_changes_are_not_retention():
    request = {"candidate_id": "a", "graph_sha256": "abc", "smiles": "CC"}
    response = {"candidate_id": "a", "graph_sha256": "abc", "input_smiles": "CC", "effective_smiles": "CO",
                "status": "ok", "scores": {"score": 1.}, "fresh_native_preprocessing": True}
    assert validate_response(response, request)["retention_eligible"] is False
    with pytest.raises(ValueError, match="identify"):
        validate_response({**response, "candidate_id": "b"}, request)


def test_native_endpoint_selection_keeps_invalid_attempts_and_ignores_scores():
    rows = [{"arm": "a", "step": step, "draw": draw, "smiles": None if draw == 0 else "CC"}
            for step in range(3) for draw in range(3)]
    result = endpoint_candidates(rows, 2)
    assert len(result) == 4 and {r["step"] for r in result} == {0, 2}
    assert sum(r["smiles"] is None for r in result) == 2


def write_generation(tmp_path):
    """Minimal audited artifact fixture, with actual tiny saved graph records."""
    problem, study = problem_fixture(), configuration()
    run_arm(problem, study, "control", Counters(), tmp_path / "arms/control", lambda _: None)
    identity = {"decoder_sha256": study.decoder["sha256"], "source_sha256": "a" * 64,
                "graph_spec_sha256": problem.spec.fingerprint(), "seed_sha256": hard_digest(problem.seed)}
    record = {"status": "completed", "config": {"stage": "pipeline", "study": study.to_dict(), "arms": ["control"]},
              "source": {"sha256": "a" * 64}}
    (tmp_path / "run.json").write_text(json.dumps(record))
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    return study


@pytest.mark.parametrize("field,value", [("graph_sha256", "b" * 64), ("smiles", "CO")])
def test_saved_candidate_identity_is_recomputed(tmp_path, field, value):
    write_generation(tmp_path)
    path = tmp_path / "arms/control/candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="hash|SMILES"):
        candidates(tmp_path)


def test_generation_gate_rejects_partial_run_and_decoder_mismatch(tmp_path):
    study = write_generation(tmp_path)
    provenance = generation_provenance(tmp_path, study)
    assert provenance["generation_config_sha256"] == study.fingerprint()
    with pytest.raises(ValueError, match="decoder"):
        generation_provenance(tmp_path, replace(study, decoder={**study.decoder, "sha256": "f" * 64}))
    path = tmp_path / "run.json"
    record = json.loads(path.read_text())
    record["status"] = "failed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="completed"):
        generation_provenance(tmp_path, study)


def test_incorrect_derivative_fails_fd_gate(tmp_path):
    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()
        @staticmethod
        def backward(ctx, grad):
            return 2 * grad
    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    original = branch.forward
    branch.forward = lambda graph: {"score": WrongGradient.apply(original(graph)["score"])}
    path = tmp_path / "gate.json"
    with pytest.raises(ValueError, match="finite differences"):
        validate(problem, study, ["both"], Counters(), path, gradient_policy=LEGACY_GRADIENT_POLICY)
    assert json.loads(path.read_text())["models"]["a"]["passed"] is False


@pytest.mark.parametrize("backward_factor", [0., -1., 2.])
def test_pointwise_gate_rejects_wrong_zero_sign_or_scale_below_absolute_fd_tolerance(tmp_path, backward_factor):
    class WrongDerivative(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value * 1e-8

        @staticmethod
        def backward(ctx, cotangent):
            return cotangent * (backward_factor * 1e-8)

        @staticmethod
        def jvp(ctx, tangent):
            return tangent * 1e-8

    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    original = branch.forward
    branch.forward = lambda graph: {"score": WrongDerivative.apply(original(graph)["score"])}
    path = tmp_path / "wrong-pointwise.json"
    with pytest.raises(ValueError, match="pointwise AD failed"):
        validate(problem, study, ["both"], Counters(), path)
    report = json.loads(path.read_text())
    assert not report["passed"] and not report["models"]["a"]["pointwise_ad"]["passed"]


@pytest.mark.parametrize("failure", ["detached", "missing_channel", "nonfinite"])
def test_pointwise_flat_support_never_waives_required_gradient_paths_or_finiteness(tmp_path, failure):
    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    original = branch.forward
    if failure == "detached":
        branch.detached = True
    elif failure == "missing_channel":
        branch.forward = lambda graph: {"score": graph.atoms[..., 2].sum()}
    else:
        branch.forward = lambda graph: {"score": original(graph)["score"] * float("nan")}
    path = tmp_path / "missing-pointwise.json"
    with pytest.raises(ValueError):
        validate(problem, study, ["both"], Counters(), path)
    assert not json.loads(path.read_text())["passed"]


def test_pointwise_reader_recomputes_ad_and_retains_diagnostic_fd_disagreement(tmp_path):
    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    original = branch.forward
    threshold = problem.graph(problem.start, study.temperature).atoms[..., 2].sum().detach() + 1e-6
    # Local derivative is valid below this threshold. Coarse FD steps cross a
    # ReLU boundary, so their secants are not that pointwise derivative.
    branch.forward = lambda graph: {"score": original(graph)["score"]
        + 1000 * torch.relu(graph.atoms[..., 2].sum() - threshold)}
    path = tmp_path / "pointwise.json"
    report = validate(problem, study, ["both"], Counters(), path)
    assert any(not row["passed"] for row in report["models"]["a"]["finite_differences"])
    require_gate(problem, study, path)
    report["models"]["a"]["pointwise_ad"]["rows"][0]["forward_jvp"] *= -1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="AD witness"):
        require_gate(problem, study, path)


@pytest.mark.parametrize("material", [False, True])
def test_forward_ad_scalar_rounding_is_numerical_replay_not_byte_identity(tmp_path, material):
    problem, study = problem_fixture(), configuration()
    branch = problem.runtimes["a"].branch
    original = branch.forward

    def shifted(graph):
        value = original(graph)["score"]
        primal, tangent = torch.autograd.forward_ad.unpack_dual(value)
        if tangent is not None:
            if material:
                primal = primal + .01
            else:
                upper = torch.full_like(primal, float("inf"))
                primal = torch.nextafter(torch.nextafter(primal, upper), upper)
            value = torch.autograd.forward_ad.make_dual(primal, tangent)
        return {"score": value}

    branch.forward = shifted
    path = tmp_path / "replay.json"
    if material:
        with pytest.raises(ValueError, match="pointwise AD failed"):
            validate(problem, study, ["both"], Counters(), path)
    else:
        report = validate(problem, study, ["both"], Counters(), path)
        assert all(not row["primal_exact"] and row["scalar_replay_consistency"]["passed"]
                   for row in report["models"]["a"]["pointwise_ad"]["rows"])
        require_gate(problem, study, path)


@pytest.mark.parametrize("mutation", ["channel", "channel_nan", "seeded", "missing_fd", "epsilon", "endpoint"])
def test_pointwise_reader_checks_channel_evidence_and_complete_finite_diagnostics(tmp_path, mutation):
    problem, study = problem_fixture(), configuration()
    path = tmp_path / "reader.json"
    report = validate(problem, study, ["both"], Counters(), path)
    model = report["models"]["a"]
    if mutation == "channel":
        model["channel_gradient_norms"].pop("charges")
    elif mutation == "channel_nan":
        model["channel_gradient_norms"]["charges"] = float("nan")
    elif mutation == "seeded":
        model["gradient_label_is_seeded_probe"] = True
    elif mutation == "missing_fd":
        model["finite_differences"].pop()
    elif mutation == "epsilon":
        model["finite_differences"][0]["epsilon"] = float("nan")
    else:
        model["finite_differences"][0]["f_plus"] = float("inf")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        require_gate(problem, study, path)


def test_native_command_protocol_records_real_subprocess_and_identity(tmp_path):
    import sys
    from latentmol.multimodel.confirmation import confirm
    study = write_generation(tmp_path)
    script = tmp_path / "native_fixture.py"
    # This is a fixture response, never a production native scorer fallback.
    script.write_text("import json,sys\nr=json.load(open(sys.argv[1])); json.dump(dict(candidate_id=r['candidate_id'],graph_sha256=r['graph_sha256'],input_smiles=r['smiles'],effective_smiles=r['smiles'],status='ok',scores={'fixture':1.0},fresh_native_preprocessing=True),open(sys.argv[2],'w'))\n")
    study = replace(study, native_evaluators={"independent_hard_only": {"command": [sys.executable, str(script), "{input}", "{output}"],
                    "timeout_seconds": 10, "identity": {"fixture_only": True}, "candidates_per_endpoint": 1}})
    report = confirm(study, tmp_path, tmp_path / "native", lambda _: None)
    assert report["models"]["independent_hard_only"]["oracle_attempts"] == 2
    assert report["models"]["independent_hard_only"]["retention_eligible"] == 2
