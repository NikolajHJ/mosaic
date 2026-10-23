"""Allocated CPU contracts for a diagnostic that cannot promote training."""
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
from chemistry_guarded_restart import cpu_command
from chemistry_recovery_diagnostic import RECOVERY_FIXTURES, RecoveryDiagnostic


def fixture_controller(tmp_path):
    controller = object.__new__(RecoveryDiagnostic)
    controller.root = tmp_path
    controller.work = Path("/work3/example")
    controller.release = Path("/work3/example/releases/new-diagnostic")
    controller.path = Path("/work3/example/submissions/new-diagnostic.json")
    controller.sha = "a" * 64
    controller.identity = dict(plan_sha256=controller.sha)
    return controller


def test_diagnostic_keeps_all_original_contracts_and_adds_its_own(tmp_path):
    controller = fixture_controller(tmp_path)
    base = cpu_command("python", tmp_path)
    assert controller.cpu_contract_command("python") == [*base, *RECOVERY_FIXTURES]


@pytest.mark.parametrize("mode", ["advance", "train", "queue", "final-evaluation"])
def test_diagnostic_cannot_construct_a_training_callback(tmp_path, mode):
    with pytest.raises(ValueError, match="cannot promote"):
        fixture_controller(tmp_path).controller(mode)


def test_diagnostic_cannot_advance_even_if_called_directly(tmp_path):
    with pytest.raises(ValueError, match="cannot submit"):
        fixture_controller(tmp_path).advance("recovery-diagnostic")


def test_a_failed_cpu_gate_never_queues_gpu_or_training(tmp_path):
    controller = fixture_controller(tmp_path)
    controller.require_cpu = Mock(side_effect=ValueError("failed CPU contracts"))
    controller.queue = Mock()
    with pytest.raises(ValueError, match="failed CPU"):
        controller.after_cpu()
    controller.queue.assert_not_called()
    import json
    saved = json.loads((tmp_path / "after-cpu-decision.json").read_text())
    assert saved["status"] == "stopped"
    assert saved["training"] == "not_submitted"


def test_a_successful_cpu_gate_submits_only_one_bounded_probe(tmp_path):
    controller = fixture_controller(tmp_path)
    reference = tmp_path / "reference.py"
    reference.write_text("# authenticated old loss oracle\n")
    import hashlib
    controller.plan = dict(loss_v1_reference=dict(module=str(reference),
        sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
    controller.require_cpu = Mock()
    controller.require_predecessor_stopped = Mock()
    controller.worker = lambda command, label: command
    controller.queue = Mock(return_value="12345")
    result = controller.after_cpu()
    assert result["training"] == "not_submitted"
    assert result["job_id"] == "12345"
    controller.queue.assert_called_once()
    args, kwargs = controller.queue.call_args
    assert args[0] == "recovery-diagnostic"
    assert "latentmol.chemistry_recovery_diagnostic" in args[1][2]
    assert kwargs == dict(gpu=True, limit="1:00")
    # A lost SSH reply followed by re-entry reads the decision and cannot queue twice.
    assert controller.after_cpu() == result
    controller.queue.assert_called_once()


def test_a_changed_saved_decision_is_rejected(tmp_path):
    import json
    (tmp_path / "after-cpu-decision.json").write_text(json.dumps(dict(identity=dict(plan_sha256="b" * 64))))
    with pytest.raises(ValueError, match="different identity"):
        fixture_controller(tmp_path).after_cpu()


def test_an_orphaned_intent_does_not_guess_whether_submission_occurred(tmp_path):
    (tmp_path / "after-cpu-intent.json").write_text('{"lost_reply":true}')
    controller = fixture_controller(tmp_path)
    controller.queue = Mock()
    with pytest.raises(FileExistsError):
        controller.after_cpu()
    controller.queue.assert_not_called()


def test_a_predecessor_stop_failure_cannot_queue_a_probe(tmp_path):
    controller = fixture_controller(tmp_path)
    controller.require_cpu = Mock()
    controller.require_predecessor_stopped = Mock(side_effect=ValueError("predecessor is live"))
    controller.queue = Mock()
    with pytest.raises(ValueError, match="predecessor is live"):
        controller.after_cpu()
    controller.queue.assert_not_called()
