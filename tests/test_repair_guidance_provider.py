"""HPC engineering checks for fixed-quota oracle orchestration, not native efficacy."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest
import torch

from latentmol.defog.assets import DEFOG_COMMIT, GUACAMOL_SHA256
from latentmol.defog.candidate_repair import repair_request, request_digest as native_request_digest
from latentmol.experiments import write_json
from latentmol.multimodel.engine import Counters
from latentmol.repair_guidance.config import RepairGuidanceConfig
from latentmol.repair_guidance.provider import DeFoGRepairProvider, RepairProviderContractError, request_digest
from latentmol.representation import DecoderOutput, GraphSpec

from test_defog_candidate_repair import _MockNative


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = GraphSpec(max_nodes=5)
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.decode_calls = 0
        self.eval().requires_grad_(False)

    def decode(self, point):
        self.decode_calls += 1
        atoms = torch.full((1, 5, 10), -1e10, device=point.device)
        atoms[..., 1] = 0
        charges = torch.full((1, 5, 5), -1e10, device=point.device)
        charges[..., 2] = 0
        bonds = torch.full((1, 5, 5, 5), -1e10, device=point.device)
        bonds[..., 0] = 0
        bonds[0, 0, 3, 0] = bonds[0, 3, 0, 0] = -1e10
        bonds[0, 0, 3, 1] = bonds[0, 3, 0, 1] = 0
        return DecoderOutput(atoms, charges, bonds)


def _config(tmp_path, **changes):
    python = tmp_path / "worker-python"
    if not python.exists():
        python.write_text("mock interpreter file; never executed by engineering fixture")
    value = {"provider": {"model": "defog_guacamol", "python": str(python),
        "source": {"path": str(tmp_path / "DeFoG"), "commit": DEFOG_COMMIT},
        "checkpoint": {"path": str(tmp_path / "guacamol.ckpt"), "sha256": GUACAMOL_SHA256},
        "steps": 10, "device": "cpu", "timeout_seconds": 5}, "attempts_per_refresh": 3}
    value.update(changes)
    return RepairGuidanceConfig.from_mapping(value, tmp_path)


def _active():
    return torch.tensor([True, False, False, True, False])


def test_config_preserves_venv_interpreter_symlink_across_record_roundtrip(tmp_path):
    """A physical resolve would launch system Python and skip the model venv."""
    path = tmp_path / "native-env" / "bin" / "python"
    path.parent.mkdir(parents=True)
    path.symlink_to(sys.executable)
    base = _config(tmp_path)
    raw = base.record()
    raw["provider"]["python"] = str(path.relative_to(tmp_path))
    config = RepairGuidanceConfig.from_mapping(raw, tmp_path)
    assert config.provider["python"] == str(path.absolute())
    assert config.provider["python"] != str(path.resolve())
    again = RepairGuidanceConfig.from_mapping(config.record(), tmp_path)
    assert again.provider["python"] == config.provider["python"]


class _Runner:
    """Inject mocked native transitions across the actual JSON receipt boundary."""
    def __init__(self, *, statuses=None, mutate=None, timeout=False):
        self.requests = []
        self.statuses = statuses or []
        self.mutate = mutate
        self.timeout = timeout

    def __call__(self, command, **kwargs):
        request_path = Path(command[command.index("--request") + 1])
        output_path = Path(command[command.index("--output") + 1])
        request = json.loads(request_path.read_text())
        index = len(self.requests)
        self.requests.append(request)
        if self.timeout:
            progress = {"version": 1, "request_sha256": request["request_sha256"], "status": "running",
                        "cost": {"native_forward_calls": 1, "native_transitions": 1,
                                 "attempted_native_transitions": 1,
                                 "cost_observation": "partial_lower_bound_until_terminal_receipt"}}
            write_json(output_path.with_name(output_path.name + ".progress.json"), progress)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        status = self.statuses[index] if index < len(self.statuses) else "accepted"
        model = _MockNative(output_labels=(4, 0) if status == "rejected" else (0, 1),
                            fail_on=1 if status == "error" else None)
        response = repair_request(request, model_loader=lambda *_: model, asset_verifier=lambda *_: None)
        if self.mutate:
            self.mutate(response)
        write_json(output_path, response)
        return SimpleNamespace(returncode=1 if response["status"] == "error" else 0)


def test_provider_runs_full_prescribed_quota_and_selects_first_accepted(tmp_path):
    config = _config(tmp_path)
    decoder, runner, counters = _Decoder(), _Runner(statuses=["rejected", "accepted", "accepted"]), Counters()
    provider = DeFoGRepairProvider(config, "a" * 64, runner=runner)
    target, receipt = provider.refresh(decoder, torch.zeros((1, 128)), _active(), 1.0, 0, tmp_path / "arm", counters)
    assert len(runner.requests) == 3
    assert receipt["selected_attempt"] == 1
    assert receipt["accepted_attempts"] == [1, 2]
    assert receipt["planned_attempts"] == receipt["attempted"] == 3
    assert target is not None and not target.graph.atoms.requires_grad
    assert torch.equal(target.active, _active())
    assert receipt["old_target_retained"] is False
    assert decoder.decode_calls == 1
    assert counters.values["defog"]["repair_refresh_0"]["worker_attempted"] == 3
    seeds = [(row["draw_seed"], row["native_seed"]) for row in receipt["attempts"]]
    assert len({seed for pair in seeds for seed in pair}) == 6
    for request in runner.requests:
        assert request_digest(request) == native_request_digest(request) == request["request_sha256"]
        assert request["occupied_slots"] == [0, 3]
        assert request["candidate"]["atoms"][0][1] == 0


def test_failed_refresh_returns_none_instead_of_retaining_previous_target(tmp_path):
    config = _config(tmp_path, attempts_per_refresh=1)
    runner = _Runner(statuses=["accepted", "rejected"])
    provider = DeFoGRepairProvider(config, "a" * 64, runner=runner)
    decoder = _Decoder()
    target0, _ = provider.refresh(decoder, torch.zeros((1, 128)), _active(), 1.0, 0, tmp_path / "arm")
    target1, receipt = provider.refresh(decoder, torch.zeros((1, 128)), _active(), 1.0, 5, tmp_path / "arm")
    assert target0 is not None and target1 is None
    assert receipt["status"] == "no_accepted_target"
    assert receipt["old_target_retained"] is False


def test_native_failure_and_timeouts_are_retained_without_hidden_retry(tmp_path):
    decoder = _Decoder()
    for name, runner in (("failure", _Runner(statuses=["error"] * 3)), ("timeout", _Runner(timeout=True))):
        provider = DeFoGRepairProvider(_config(tmp_path), "a" * 64, runner=runner)
        target, receipt = provider.refresh(decoder, torch.zeros((1, 128)), _active(), 1.0, 0, tmp_path / name)
        assert target is None and len(runner.requests) == 3
        assert receipt["status"] == "no_accepted_target"
        assert all(attempt["status"] == ("error" if name == "failure" else "timeout") for attempt in receipt["attempts"])
        if name == "timeout":
            assert all(attempt["cost_censored"] for attempt in receipt["attempts"])
            assert all(attempt["partial_progress"]["cost"]["native_forward_calls"] == 1 for attempt in receipt["attempts"])


@pytest.mark.parametrize("mutate", [
    lambda response: response.update(version=True),
    lambda response: response["provenance"].update(checkpoint_sha256="b" * 64),
    lambda response: response["provenance"].update(adapter_module_sha256="b" * 64),
    lambda response: response.update(candidate_sha256="b" * 64),
    lambda response: response.update(occupied_slots=[0, 1]),
    lambda response: response.update(native_order_preserved=False),
    lambda response: response["provenance"].update(frozen_state_sha256_after="b" * 64),
    lambda response: response["target"]["charges"][0].__setitem__(0, 3),
    lambda response: response["native_output_labels"]["atoms"][0].__setitem__(0, 1),
    lambda response: response["cost"]["effective_sampling"].update(eta=999.0),
    lambda response: response["native_input_labels"]["atoms"][0].__setitem__(0, 1),
])
def test_contract_mismatch_is_persisted_before_failing_closed(tmp_path, mutate):
    provider = DeFoGRepairProvider(_config(tmp_path), "a" * 64, runner=_Runner(mutate=mutate))
    with pytest.raises(RepairProviderContractError):
        provider.refresh(_Decoder(), torch.zeros((1, 128)), _active(), 1.0, 0, tmp_path / "arm")
    receipt = json.loads((tmp_path / "arm/repair-refresh-00000/refresh.json").read_text())
    assert receipt["status"] == "contract_failed"
    assert receipt["unattempted"] == 2
    assert receipt["attempts"][0]["status"] == "contract_failed"
    assert receipt["attempts"][0]["response"] is not None


def test_zero_weight_disables_decode_sampling_worker_and_directory(tmp_path):
    runner, decoder = _Runner(), _Decoder()
    provider = DeFoGRepairProvider(_config(tmp_path, weight=0), "a" * 64, runner=runner)
    target, receipt = provider.refresh(decoder, torch.zeros((1, 128)), _active(), 1.0, 0, tmp_path / "arm")
    assert target is None and receipt["status"] == "disabled"
    assert not runner.requests and not decoder.decode_calls
    assert not (tmp_path / "arm").exists()


def test_receipts_are_immutable_and_do_not_overwrite_a_refresh(tmp_path):
    provider = DeFoGRepairProvider(_config(tmp_path, attempts_per_refresh=1), "a" * 64, runner=_Runner())
    args = (_Decoder(), torch.zeros((1, 128)), _active(), 1.0, 0, tmp_path / "arm")
    provider.refresh(*args)
    with pytest.raises(FileExistsError):
        provider.refresh(*args)


@pytest.mark.parametrize("setting", [
    {"steps": 1}, {"sampling": {"eta": -1}}, {"sampling": {"typo": 1}},
    {"sampling": {"time_distortion": "beta"}},
    {"source": {"path": "/tmp/defog", "commit": "a" * 40}},
    {"checkpoint": {"path": "/tmp/guacamol.ckpt", "sha256": "a" * 64}},
])
def test_provider_config_rejects_unsupported_pin_or_sampling(tmp_path, setting):
    config = _config(tmp_path).record()
    config["provider"].update(setting)
    with pytest.raises(ValueError):
        RepairGuidanceConfig.from_mapping(config, tmp_path)
