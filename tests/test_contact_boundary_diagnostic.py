"""Allocated CPU engineering fixtures; not a saved-state gradient certificate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location("_contact_diagnostic_fixture",
    Path(__file__).parents[1] / "src/latentmol/boltz/contact_boundary_diagnostic.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def test_native_distogram_projection_and_widened_control_preserve_symmetric_structure():
    torch.manual_seed(37)
    native = pytest.importorskip("boltz.model.modules.trunkv2")
    head = native.DistogramModule(token_z=4, num_bins=64).eval().requires_grad_(False)
    model = SimpleNamespace(distogram_module=head)
    pair = torch.randn(1, 3, 3, 4, requires_grad=True)
    assert torch.equal(observer.cached_projection(model, pair), head(pair))
    widened = observer.cached_projection(model, pair, widened=True)
    assert widened.dtype == torch.float64
    assert torch.equal(widened, widened.transpose(1, 2))
    expected = torch.nn.functional.linear(pair.double() + pair.double().transpose(1, 2),
        head.distogram.weight.double(), head.distogram.bias.double()).reshape(1, 3, 3, 1, 64)
    assert torch.equal(widened, expected)
    torch.testing.assert_close(widened.float(), head(pair), rtol=1e-5, atol=1e-6)


def test_full_dbmol_reference_parity_rejects_graph_and_any_subterm_difference():
    values = dict(zip(observer.CONTACT_KEYS, (1., 1.5, .5)))
    reference = {"values": values, "graph_sha256": "a" * 64}
    observer.check_contact_endpoint(values, "a" * 64, reference, "same")
    with pytest.raises(ValueError, match="parity"):
        observer.check_contact_endpoint(values, "b" * 64, reference, "changed graph")
    for key in observer.CONTACT_KEYS:
        with pytest.raises(ValueError, match="parity"):
            observer.check_contact_endpoint({**values, key: values[key] + .01}, "a" * 64, reference, "changed term")


def test_budget_counts_all_declared_cuts_and_native_held_endpoints():
    record = observer.budget((2, 4))
    assert record["contact_only_forwards_per_case"] == [84, 156]
    assert record["maximum_contact_only_forwards"] == 240
    assert record["maximum_physical_trunks_excluding_checkpoint_recomputation"] == 252
    assert record["cut_upstream_reverse_traversals_per_case"] == 30
    assert record["serial_child_seconds"] * 2 + record["controller_headroom_seconds"] == 7200
    assert record["optimizer_updates"] == record["new_defog_calls"] == 0
    with pytest.raises(ValueError, match="original"):
        observer.budget((1, 4))


def test_all_pass_localization_labels_actual_contact_branch():
    boundary = SimpleNamespace(first_interval=lambda _rows: {
        "status": "all_cut_readouts_pass", "remaining_region": "affinity"})
    assert observer.first_interval({}, boundary)["remaining_region"] == "distogram_to_contact_reduction"


def test_contact_witness_retains_all_exact_anchor_and_pocket_ties():
    from latentmol.boltz.dbmol_objective import native_bin_centers
    logits = torch.zeros(1, 4, 4, 1, 64)
    config = SimpleNamespace(objective_arithmetic="contact_and_assembly_float64_v1",
                             distance_cutoff=8., epsilon=1e-8)
    witness = observer.contact_selection(logits, torch.tensor([True, True, False, False]),
        torch.tensor([False, False, True, True]), config, native_bin_centers(logits))
    assert witness["anchor"]["tie_indices"] == [0, 1, 2, 3]
    assert [row["tie_indices"] for row in witness["per_pocket"]] == [[0, 1], [0, 1]]
    assert witness["anchor"]["best_vs_second_margin"] == 0.
