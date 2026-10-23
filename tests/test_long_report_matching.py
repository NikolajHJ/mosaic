"""Explicit operational-report exceptions retain native compute-cap matching.

Pure saved-configuration fixtures; these do not execute a model, gate, clock,
optimizer or scientific numerical experiment.
"""

from copy import deepcopy

from latentmol.reporting.optimization import _matching_config


def _config(wall_seconds=3600, calls=15000):
    return {"name": "synthetic-long-arm", "output_root": "/saved/synthetic",
            "optimizer": {"steps": 500, "trust_radius": 3.},
            "execution_budget": {"version": "tracked_native_v1",
                                 "wall_seconds": wall_seconds, "max_boltz_calls": calls},
            "repair_guidance": {"weight": .001, "weight_schedule": {
                "version": "linear_updates_v1", "start_weight": .001,
                "end_weight": .1, "start_update": 12, "end_update": 24},
                "provider": {"checkpoint": {"sha256": "synthetic-frozen-identity"}}}}


def test_long_overlay_can_declare_different_wall_ceilings_only():
    control, guided = _config(3600), _config(7200)
    assert _matching_config(control, False, False) == _matching_config(guided, False, False)
    # Operational exemption does not remove the tracked budget's actual scope.
    normalized = _matching_config(control, False, False)
    assert normalized["execution_budget"] == {"version": "tracked_native_v1", "max_boltz_calls": 15000}


def test_long_overlay_still_rejects_a_changed_native_call_cap_or_version():
    reference = _config()
    changed = _config(calls=15001)
    assert _matching_config(reference, False, False) != _matching_config(changed, False, False)
    changed = _config()
    changed["execution_budget"]["version"] = "different-tracked-call-scope"
    assert _matching_config(reference, False, False) != _matching_config(changed, False, False)


def test_matching_projection_does_not_mutate_native_config_receipts():
    control, guided = _config(3600), _config(7200)
    guided["repair_guidance"]["weight"] = .1
    guided["repair_guidance"]["weight_schedule"]["end_weight"] = .2
    before = deepcopy((control, guided))
    assert _matching_config(control, False, True) == _matching_config(guided, False, True)
    assert (control, guided) == before
    assert control["execution_budget"]["wall_seconds"] == 3600
    assert guided["execution_budget"]["wall_seconds"] == 7200
    assert guided["repair_guidance"]["weight_schedule"]["end_weight"] == .2
