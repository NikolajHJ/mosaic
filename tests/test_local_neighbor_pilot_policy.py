"""Quality regression is a result in exploration, with explicit old-stop coverage."""
import importlib.util
import json
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / 'scripts/local_neighbor_pilot.py'
_spec = importlib.util.spec_from_file_location('local_neighbor_pilot_policy_test', _path)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def panel(valid=800, reconstruction=1.0):
    return dict(panels=[dict(argmax=dict(raw_valid=valid))],
                reconstruction=dict(exact_reconstruction=reconstruction))


def test_negative_quality_signal_recorded_without_stopping(tmp_path):
    warnings = pilot.record_quality_observation(panel(), panel(500, .9), 1024,
        research_profile='exploratory', folder=tmp_path, step=10)
    assert len(warnings) == 2
    record = json.loads((tmp_path / 'quality-warnings.jsonl').read_text())
    assert record['local_update'] == 10 and record['warnings'] == warnings


def test_explicit_qualified_preserves_quality_stop_and_evidence(tmp_path):
    with pytest.raises(RuntimeError, match='safety stop'):
        pilot.record_quality_observation(panel(), panel(500), 1024,
            research_profile='qualified', folder=tmp_path, step=10)
    assert (tmp_path / 'quality-warnings.jsonl').is_file()


def test_small_fluctuation_is_not_a_quality_failure(tmp_path):
    assert pilot.record_quality_observation(panel(), panel(780, .99), 1024,
        research_profile='exploratory', folder=tmp_path, step=10) == []
    assert not (tmp_path / 'quality-warnings.jsonl').exists()
