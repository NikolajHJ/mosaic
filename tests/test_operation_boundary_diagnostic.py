"""Allocated CPU fixtures for identity-clone complete-cut instrumentation.

These small graphs test engineering behavior only. They neither validate native
Boltz/DeFoG nor approve any historical gradient gate. Execute on DTU HPC.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SOURCE = Path(__file__).parents[1] / "src/latentmol/boltz/operation_boundary_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("_operation_boundary_fixture", SOURCE)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


class Atom(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, features):
        c = features["ref_element"] + features["ref_charge"]
        q = c  # Intentional native-style alias with different downstream use.
        p = self.relu(c) + 2 * c  # Ancestor/descendant before clone frontier.
        return q, c, p, lambda value: value


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_encoder = Atom()

    def forward(self, features, affinity=False):
        q, c, p, _ = self.atom_encoder(features)
        return q + 2 * c + p + int(affinity)


class Pair(nn.Module):
    def forward(self, s, z):
        z = z + .1 * s
        s = s + .2 * z  # z is an ancestor of s: captures must be siblings.
        return s, z


class NoSeq(nn.Module):
    def forward(self, z, *_args, **_kwargs):
        return z + .2 * z.sin()


class Stack(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([NoSeq() for _ in range(depth)])

    def forward(self, z, pair_mask, use_kernels=False):
        for block in self.layers:
            z = block(z, pair_mask, None, use_kernels)
        return z


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
        with torch.no_grad():
            self.affinity_out_mlp[0].weight.fill_(1)
            self.affinity_out_mlp[0].bias.fill_(2)

    def forward(self, z):
        return self.affinity_out_mlp(z).sum()


class Member(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.pairformer_stack = Stack(depth)
        self.affinity_heads = Heads()

    def forward(self, s_inputs, z, x_pred, feats, multiplicity=1, use_kernels=False):
        z = z + .3 * s_inputs
        z = self.pairformer_stack(z, pair_mask=None, use_kernels=False)
        return self.affinity_heads(z)


class NativeToy(nn.Module):
    def __init__(self, depths=(2, 2), *, checkpointed=False):
        super().__init__()
        self.affinity_ensemble = True
        self.input_embedder = Embedder()
        self.pairformer_module = nn.Module()
        self.pairformer_module.layers = nn.ModuleList([Pair() for _ in range(4)])
        self.affinity_module1, self.affinity_module2 = Member(depths[0]), Member(depths[1])
        self.eval().requires_grad_(False)
        if checkpointed:
            # Same wrapper used by production: inner bound-forward recomputation
            # must not replay observer hooks on the block's outer __call__.
            from latentmol.boltz.model import checkpoint_eval_block
            stacks = (self.pairformer_module, self.affinity_module1.pairformer_stack,
                      self.affinity_module2.pairformer_stack)
            for stack in stacks:
                for block in stack.layers:
                    checkpoint_eval_block(block)


def forward(model, point, capture=None):
    save = (lambda _name, value: value) if capture is None else capture.save
    graph = {name: save("graph." + name, point * factor)
             for name, factor in zip(observer.FIELDS, (1., 2., 3.))}
    features = {"ref_element": save("adapter.ref_element", graph["atoms"]),
                "ref_charge": save("adapter.ref_charge", graph["charges"]),
                "token_bonds": save("adapter.token_bonds", graph["bonds"])}
    bonds = save("adapter.bond_types", graph["bonds"] * .5)
    s = model.input_embedder(features)
    z = s + features["token_bonds"] + bonds
    for block in model.pairformer_module.layers:
        s, z = block(s, z)
    inputs = model.input_embedder(features, affinity=True)
    result = sum(member(s_inputs=inputs, z=z, x_pred=None, feats=features,
                        multiplicity=1, use_kernels=False)
                 for member in (model.affinity_module1, model.affinity_module2))
    if capture is not None:
        capture.require_complete_membership()
    return result


@pytest.mark.parametrize("depths,checkpointed", [((2, 2), False), ((2, 5), False),
    ((2, 5), True), ((1, 3), True), ((4, 1), True), ((4, 0), True), ((0, 3), True)])
def test_every_complete_frontier_preserves_branching_gradient_and_values(depths, checkpointed):
    model = NativeToy(depths, checkpointed=checkpointed)
    point = torch.tensor([[.7]], requires_grad=True)
    native = forward(model, point)
    direct, = torch.autograd.grad(native, point)
    with observer.BoundaryCapture(model) as capture:
        observed = forward(model, point, capture)
        assert torch.equal(native, observed)
        clone_direct, = torch.autograd.grad(observed, point, retain_graph=True)
        assert observer.chain_completeness(direct, clone_direct)["engineering_chain_complete"]
        assert len(capture.cuts) == 14
        assert capture.topology["affinity_member_depths"] == list(depths)
        assert capture.cuts["affinity_midpoint"] == [f"member.{i}.block.{(depth - 1) // 2}" if depth
                                                     else f"member.{i}.conditioned"
                                                     for i, depth in enumerate(depths)]
        assert capture.cuts["affinity_end"] == [f"member.{i}.block.{depth - 1}" if depth else f"member.{i}.conditioned"
                                                for i, depth in enumerate(depths)]
        for name, members in capture.cuts.items():
            fields = tuple(capture.values[key] for key in members)
            raw = torch.autograd.grad(observed, fields, retain_graph=True, allow_unused=True)
            fixed = tuple(torch.zeros_like(field) if value is None else value.detach()
                          for field, value in zip(fields, raw))
            chained, = torch.autograd.grad(fields, point, grad_outputs=fixed, retain_graph=True)
            assert observer.chain_completeness(direct, chained)["engineering_chain_complete"], name
        assert len(capture.relus) == 4  # Two atom calls and two ensemble heads.
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_checkpoint_recomputation_does_not_replay_outer_capture_hooks():
    model = NativeToy((2, 5), checkpointed=True)
    counts, handles = {}, []
    for member_index, member in enumerate((model.affinity_module1, model.affinity_module2)):
        for block_index, block in enumerate(member.pairformer_stack.layers):
            key = (member_index, block_index)
            counts[key] = 0
            def observe(_module, _args, _output, key=key):
                counts[key] += 1
            handles.append(block.register_forward_hook(observe))
    try:
        point = torch.tensor([[.7]], requires_grad=True)
        with observer.BoundaryCapture(model) as capture:
            loss = forward(model, point, capture)
            direct, = torch.autograd.grad(loss, point, retain_graph=True)
            members = capture.cuts["affinity_midpoint"]
            fields = tuple(capture.values[name] for name in members)
            cotangents = torch.autograd.grad(loss, fields, retain_graph=True)
            chained, = torch.autograd.grad(fields, point, grad_outputs=cotangents)
            assert observer.chain_completeness(direct, chained)["engineering_chain_complete"]
            assert all(count == 1 for count in counts.values())
    finally:
        for handle in handles:
            handle.remove()
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_raw_ancestor_capture_double_counts_but_clone_frontier_does_not():
    point = torch.tensor([1.], requires_grad=True)
    parent = 2 * point
    child = 3 * parent
    loss = parent + child
    cotangents = torch.autograd.grad(loss, (parent, child), retain_graph=True)
    bad, = torch.autograd.grad((parent, child), point, grad_outputs=cotangents, retain_graph=True)
    direct, = torch.autograd.grad(loss, point)
    assert not observer.chain_completeness(direct, bad)["engineering_chain_complete"]
    parent, child = 2 * point, 6 * point
    loss = parent + child
    cotangents = torch.autograd.grad(loss, (parent, child), retain_graph=True)
    good, = torch.autograd.grad((parent, child), point, grad_outputs=cotangents)
    assert observer.chain_completeness(direct, good)["engineering_chain_complete"]


def test_hooks_removed_after_failed_native_forward():
    model = NativeToy()
    with pytest.raises(RuntimeError):
        with observer.BoundaryCapture(model):
            raise RuntimeError("failed")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_repeated_native_calls_rejected_instead_of_guessing_recycle_cut():
    model = NativeToy()
    point = torch.tensor([[.7]], requires_grad=True)
    with observer.BoundaryCapture(model) as capture:
        forward(model, point, capture)
        with pytest.raises(ValueError, match="Repeated boundary"):
            forward(model, point, capture)


def test_relu_telemetry_retains_crossings_and_membership():
    baseline = {"a": torch.tensor([-.1, 0., .2])}
    current = {"a": torch.tensor([.1, .2, .3])}
    result = observer.relu_crossings(current, baseline)["a"]
    assert result["positive_branch_changes"] == 2
    assert result["exact_zero_at_baseline"] == 1
    with pytest.raises(ValueError, match="membership"):
        observer.relu_crossings({}, baseline)


def test_affinity_scalar_matches_original_primitive_widening():
    config = SimpleNamespace(dbmol=SimpleNamespace(objective_arithmetic="contact_and_assembly_float64_v1",
        affinity_regression_weight=1., affinity_logit_weight=1.))
    outputs = {"affinity_pred_value": torch.tensor([[.2]]), "affinity_logits_binary": torch.tensor([[.3]])}
    assert torch.equal(observer.affinity_loss(outputs, config), (outputs["affinity_pred_value"].double()
                                                               - outputs["affinity_logits_binary"].double()).reshape(()))
    config.dbmol.objective_arithmetic = "native_v1"
    with pytest.raises(ValueError, match="precision"):
        observer.affinity_loss(outputs, config)


def test_detached_phase_contract_refuses_native_autograd_graph():
    observer.assert_detached({"x": torch.ones(2)})
    with pytest.raises(ValueError, match="escaped"):
        observer.assert_detached({"x": torch.ones(2, requires_grad=True)})


def test_cut_scope_is_not_changed_after_finite_difference_observations():
    assert observer.selected_indices(64) == [15, 31, 47, 63]
    assert observer.selected_indices(1) == [0]
    with pytest.raises(ValueError):
        observer.selected_indices(0)
    assert observer.affinity_stage_indices([2, 5]) == {"midpoint": (0, 2), "end": (1, 4)}
    assert observer.affinity_stage_indices([1, 1]) == {"midpoint": (0, 0), "end": (0, 0)}
    assert observer.affinity_stage_indices([2, 0]) == {"midpoint": (0, None), "end": (1, None)}
    with pytest.raises(ValueError, match="nonnegative"):
        observer.affinity_stage_indices([2, -1])


def test_chain_qualifier_cannot_hide_low_signal_or_accumulation_error():
    native = {"analytic": 1., "analytic_dot_float64": 1., "actual_secant_linear_prediction": 1.,
              "informative": True, "passed_tolerance": False, "absolute_error": .2}
    chain = deepcopy(native)
    chain["analytic"] += .00001
    assert observer.qualify_residual(native, chain)["failure_localization_eligible"]
    chain["analytic"] += .01
    assert not observer.qualify_residual(native, chain)["failure_localization_eligible"]
    chain = {**native, "informative": False}
    assert not observer.qualify_residual(native, chain)["failure_localization_eligible"]


def test_ordered_intervals_keep_ambiguity_and_do_not_claim_unique_operation():
    passed = {"original_native_reference": {"passed": True}, "measured_chain_reference": {"passed": True},
              "attribution": {"failure_localization_eligible": False}}
    failed = {"original_native_reference": {"passed": False}, "measured_chain_reference": {"passed": False},
              "attribution": {"failure_localization_eligible": True}}
    result = observer.first_interval({"graph": passed, "adapter": failed})
    assert result["from"] == "graph" and result["to"] == "adapter"
    assert result["unique_operation_identified"] is False
    failed["attribution"]["failure_localization_eligible"] = False
    assert observer.first_interval({"graph": failed})["status"].startswith("ambiguous")


def test_exact_two_case_and_reference_metadata_requirements():
    prefix = "/work3/test/runs/source-093-first-suite-attempt001-replay/"
    cases = [{"id": name, "arm": "fixed_initial_mask", "step": 15,
              "parent_artifacts_sha256": {prefix + name + "/attribution.json": "a" * 64}}
             for name in ("boltz_only", "boltz_defog")]
    assert observer.validate_cases({"cases": cases})["cases"] == cases
    cases[1]["step"] = 14
    with pytest.raises(ValueError, match="source091"):
        observer.validate_cases({"cases": cases})


def test_separate_submitter_reuses_source_verifier_fixture_gate_and_gpu_policy():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/hpc"))
    from submit_operation_boundary_diagnostic import workflow
    cpu, native = workflow(Path("/work3/user/releases/observer"), Path("/work3/user"),
        Path("/work3/user/plan.json"), "a" * 64, "b" * 64, "c" * 64)
    assert "tests/test_operation_boundary_diagnostic.py" in cpu["job_command"]
    assert native["dependency"] == "done:cpu"
    assert native["arguments"][native["arguments"].index("-W") + 1] == "2:00"
    assert "select[gpu80gb]" in native["arguments"]
    assert "--archive-sha256" in native["job_command"] and "--manifest-sha256" in native["job_command"]
    assert "latentmol.boltz.operation_boundary_diagnostic" in native["job_command"]
    assert "latentmol.boltz.failed_gate_attribution" not in native["job_command"]


def test_forward_budget_is_enforced_and_controller_has_wall_headroom():
    budget = observer.forward_budget([2, 4])
    assert budget["planned_explicit_affinity_forwards_upper_bound"] == 138
    assert budget["maximum_total_affinity_traversals"] == 144
    assert budget["maximum_auxiliary_full_dbmol_forwards"] == 6
    assert budget["controller_headroom_seconds"] == 300
    assert observer.forward_budget([3, 4])["planned_explicit_affinity_forwards_upper_bound"] == 156
    with pytest.raises(ValueError, match="call budget"):
        observer.forward_budget([4, 4])
    with pytest.raises(ValueError, match="direction counts"):
        observer.forward_budget([1, 4])


def test_controller_completion_requires_persisted_scientific_receipts(tmp_path):
    execution = {"returncode": 0, "timed_out": False, "run_paths": [str(tmp_path)]}
    assert observer.execution_completion(execution)[0] is False
    (tmp_path / "run.json").write_text(json.dumps({"status": "completed"}))
    evidence = {"status": "completed", "diagnostic_completed": True,
                "frozen_state_unchanged": True, "parent_unchanged": True}
    (tmp_path / "attribution.json").write_text(json.dumps(evidence))
    assert observer.execution_completion(execution) == (True, None)
    evidence["diagnostic_completed"] = False
    (tmp_path / "attribution.json").write_text(json.dumps(evidence))
    assert observer.execution_completion(execution)[0] is False
    assert observer.execution_completion({**execution, "returncode": 1}) == (False, "child_failed_or_timed_out")
    assert observer.execution_completion({**execution, "timed_out": True}) == (False, "child_failed_or_timed_out")


def test_controller_failure_raises_after_both_case_receipts_are_preserved(tmp_path, monkeypatch):
    import latentmol.experiments as experiments
    import latentmol.boltz.workflow_process as workflow_process
    cases = [{"id": name, "arm": "fixed_initial_mask", "step": 15, "parent_run": str(tmp_path / name),
              "parent_artifacts_sha256": {f"/work3/runs/source-093-first-suite-attempt001-replay/{name}/attribution.json": "a" * 64}}
             for name in ("boltz_only", "boltz_defog")]
    plan = {"cases": cases, "output_root": str(tmp_path), "name": "fixture-controller"}
    observed = []
    def run_logged(command, *_args, **_kwargs):
        observed.append(command[command.index("--child") + 1])
        return {"returncode": 1, "timed_out": False, "run_paths": []}
    fake_helper = SimpleNamespace(load_plan=lambda *_args: plan,
        authenticated_artifact=lambda _case, path: path, _read=lambda _path: {},
        failed_item=lambda _gate: ("combined", {"direction_results": {"random_1": {"passed": False}}}),
        _sha=lambda _path: "a" * 64)
    monkeypatch.setattr(observer, "helper", lambda: fake_helper)
    monkeypatch.setattr(workflow_process, "run_logged", run_logged)
    run_record = {}
    class FakeRun:
        def __init__(self, *_args):
            self.path = tmp_path / "controller"
            self.path.mkdir()
        def __enter__(self):
            return self
        def annotate(self, **values):
            run_record.update(values)
        def __exit__(self, kind, error, _traceback):
            run_record["status"] = "failed" if error else "completed"
            experiments.write_json(self.path / "run.json", run_record)
    monkeypatch.setattr(experiments, "ExperimentRun", FakeRun)
    with pytest.raises(RuntimeError, match="incomplete: boltz_only, boltz_defog"):
        observer.run_study(tmp_path / "plan.json", "a" * 64)
    retained = json.loads((tmp_path / "controller/cases.json").read_text())
    assert observed == ["boltz_only", "boltz_defog"]
    assert len(retained) == 2 and all(row["diagnostic_completed"] is False for row in retained)
    assert json.loads((tmp_path / "controller/run.json").read_text())["status"] == "failed"


def test_main_propagates_controller_failure_as_nonzero(monkeypatch):
    cases = [{"id": name, "arm": "fixed_initial_mask", "step": 15,
              "parent_artifacts_sha256": {f"/work3/runs/source-093-first-suite-attempt001-replay/{name}/attribution.json": "a" * 64}}
             for name in ("boltz_only", "boltz_defog")]
    monkeypatch.setattr(observer, "helper", lambda: SimpleNamespace(_hash=lambda _value: True,
        load_plan=lambda *_args: {"cases": cases}))
    monkeypatch.setenv("LSB_JOBID", "fixture")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "a" * 64)
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--plan", "/work3/plan.json", "--sha256", "a" * 64])
    def fail(*_args):
        raise RuntimeError("incomplete")
    monkeypatch.setattr(observer, "run_study", fail)
    with pytest.raises(RuntimeError, match="incomplete"):
        observer.main()
