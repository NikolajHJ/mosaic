"""Allocated engineering contracts; no model or numerical execution on laptops.

The tiny core below tests *delegation*, state/frame and RNG bookkeeping. It is
deliberately not a scientific substitute for actual-checkpoint native replay.
"""
from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn
from torch_scatter import scatter_mean

from latentmol.diffsbdd.native import (GaussianStream, NativeState, NativeStepper,
                                     assert_native_replay, authenticate_native_source,
                                     load_native_core)


def _mask(n_samples, counts, device):
    return torch.repeat_interleave(torch.arange(n_samples, device=device), counts)


utils = SimpleNamespace(num_nodes_to_batch_mask=_mask)


class _Dynamics(nn.Module):
    update_pocket_coords = False

    def forward(self, ligand, pocket, t, ligand_mask, pocket_mask):
        return ligand * 0.125, pocket * 0


class EngineeringCore(nn.Module):
    """Toy native API with a recognizable nonzero terminal draw."""

    def __init__(self):
        super().__init__()
        self.dynamics = _Dynamics()
        self.T, self.atom_nf, self.n_dims = 4, 2, 3
        self.norm_values, self.norm_biases = (2.0, 4.0), (None, 0.25)
        self.vnode_idx = None
        self.calls = []
        self.eval()

    def gamma(self, t):
        return -4 + 8 * t

    def alpha(self, gamma, target_tensor):
        return torch.sigmoid(-gamma).sqrt()

    def sigma(self, gamma, target_tensor):
        return torch.sigmoid(gamma).sqrt()

    def normalize(self, ligand=None, pocket=None):
        for value in (ligand, pocket):
            if value is not None:
                value["x"] = value["x"] / self.norm_values[0]
                value["one_hot"] = (value["one_hot"].float() - self.norm_biases[1]) / self.norm_values[1]
        return ligand, pocket

    def unnormalize(self, x, h):
        return x * self.norm_values[0], h * self.norm_values[1] + self.norm_biases[1]

    @staticmethod
    def sample_gaussian(size, device):
        return torch.randn(size, device=device)

    @staticmethod
    def remove_mean_batch(ligand, pocket, ligand_mask, pocket_mask):
        mean = scatter_mean(ligand, ligand_mask, dim=0)
        return ligand - mean[ligand_mask], pocket - mean[pocket_mask]

    @staticmethod
    def assert_mean_zero_with_mask(x, mask):
        assert scatter_mean(x, mask, dim=0).abs().max() < 1e-5

    def sample_normal_zero_com(self, mean, pocket, sigma, lm, pm):
        noise = self.sample_gaussian((len(lm), 5), device=lm.device)
        ligand = mean + sigma[lm] * noise
        p = pocket.clone()
        ligand[:, :3], p[:, :3] = self.remove_mean_batch(ligand[:, :3], p[:, :3], lm, pm)
        return ligand, p

    def xh_given_zt_and_epsilon(self, z, eps, gamma, mask):
        return z / self.alpha(gamma, z)[mask] - eps * self.sigma(gamma, z)[mask] / self.alpha(gamma, z)[mask]

    def sample_p_zs_given_zt(self, s, t, z, p, lm, pm):
        self.calls.append(("reverse", float(s[0]), float(t[0])))
        return self.sample_normal_zero_com(z * 0.9, p, torch.ones_like(s) * 0.05, lm, pm)

    def sample_p_xh_given_z0(self, z, p, lm, pm, batch_size):
        self.calls.append(("terminal",))
        final, p = self.sample_normal_zero_com(z, p, z.new_full((batch_size, 1), 0.01), lm, pm)
        x, _ = self.unnormalize(final[:, :3], final[:, 3:])
        _, h = self.unnormalize(z[:, :3], z[:, 3:])
        px, ph = self.unnormalize(p[:, :3], p[:, 3:])
        return x, torch.nn.functional.one_hot(h.argmax(-1), self.atom_nf), px, ph

    def sample_given_pocket(self, pocket, counts, return_frames=1):
        _, p = self.normalize(pocket=pocket)
        pm = p["mask"]
        lm = _mask(len(counts), counts, p["x"].device)
        rawp = torch.cat((p["x"], p["one_hot"]), -1)
        meanx = scatter_mean(p["x"], pm, dim=0)
        meanh = torch.zeros(len(counts), self.atom_nf)
        mean = torch.cat((meanx, meanh), -1)[lm]
        z, shifted = self.sample_normal_zero_com(mean, rawp, torch.ones_like(counts)[:, None], lm, pm)
        for i in reversed(range(self.T)):
            s = torch.full((len(counts), 1), i) / self.T
            t = torch.full((len(counts), 1), i + 1) / self.T
            z, shifted = self.sample_p_zs_given_zt(s, t, z, shifted, lm, pm)
        x, h, px, ph = self.sample_p_xh_given_z0(z, shifted, lm, pm, len(counts))
        return torch.cat((x, h), -1), torch.cat((px, ph), -1), lm, pm


@pytest.fixture
def core():
    return EngineeringCore()


@pytest.fixture
def pocket():
    return {"x": torch.tensor([[10., 2., -1.], [11., 3., 0.], [12., 2., 1.]]),
            "one_hot": torch.tensor([[1., 0.], [0., 1.], [1., 0.]]),
            "mask": torch.zeros(3, dtype=torch.long), "size": torch.tensor([3])}


def test_replay_delegates_exact_native_steps_and_final_draw(core, pocket):
    receipt = assert_native_replay(NativeStepper(core), pocket, torch.tensor([3]), 17)
    assert receipt["passed"]
    assert [d["key"] for d in receipt["draws"]] == [
        "initial", "reverse:3", "reverse:2", "reverse:1", "reverse:0", "terminal"]
    assert len(core.calls) == 10


def test_keyed_gaussians_do_not_use_or_advance_global_rng():
    before = torch.random.get_rng_state().clone()
    stream = GaussianStream(17)
    a = stream.draw("reverse:2", (4, 5), "cpu")
    stream.draw("unrelated-diagnostic", (7,), "cpu")
    b = stream.draw("reverse:2", (4, 5), "cpu")
    assert torch.equal(a, b)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(a, GaussianStream(23).draw("reverse:2", (4, 5), "cpu"))


def test_joint_frame_recovers_original_pocket(core, pocket):
    stepper = NativeStepper(core)
    stream = GaussianStream(17)
    original = pocket["x"].clone()
    state = stepper.initialize(pocket, torch.tensor([3]), stream)
    for _ in range(core.T):
        restored = (state.pocket[:, :3] + state.frame_offset[state.pocket_mask]) * 2
        assert torch.allclose(restored, original, atol=2e-5, rtol=0)
        state = stepper.reverse_step(state, stream)
    final = stepper.finalize(state, stream)
    assert torch.allclose(final.pocket_positions, original, atol=2e-5, rtol=0)
    assert torch.equal(pocket["x"], original)


def test_correction_exact_noop_consumes_no_noise(core, pocket):
    stepper = NativeStepper(core)
    stream = GaussianStream(17)
    state = stepper.initialize(pocket, torch.tensor([3]), stream)
    baseline = state.ligand.clone()
    before = len(stream.receipts)
    assert stepper.correct(state, baseline, baseline.clone()) is state
    assert len(stream.receipts) == before


def test_normalized_residual_correction_and_joint_translation(core, pocket):
    stepper = NativeStepper(core)
    state = stepper.initialize(pocket, torch.tensor([3]), GaussianStream(17))
    base = torch.zeros_like(state.ligand)
    proposed = base.clone()
    proposed[:, 0] = 0.2
    proposed[:, 3] = 0.1
    new = stepper.correct(state, base, proposed)
    t = torch.tensor([[1.]])
    alpha = core.alpha(core.gamma(t), state.ligand)[0, 0]
    assert torch.allclose(new.ligand[:, 3] - state.ligand[:, 3], alpha * 0.1)
    assert torch.allclose(new.ligand[:, :3], state.ligand[:, :3], atol=1e-6)
    old_global_p = state.pocket[:, :3] + state.frame_offset[state.pocket_mask]
    new_global_p = new.pocket[:, :3] + new.frame_offset[new.pocket_mask]
    assert torch.allclose(old_global_p, new_global_p, atol=1e-6)
    old_global_lig = state.ligand[:, :3] + state.frame_offset[state.ligand_mask]
    new_global_lig = new.ligand[:, :3] + new.frame_offset[new.ligand_mask]
    assert torch.allclose(new_global_lig[:, 0] - old_global_lig[:, 0], alpha * 0.2)


def test_clean_input_uses_loaded_bias_scale_and_frame(core, pocket):
    stepper = NativeStepper(core)
    state = stepper.initialize(pocket, torch.tensor([3]), GaussianStream(17))
    xyz = torch.tensor([[10., 1., 0.], [11., 2., 0.], [12., 1., 0.]])
    h = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    normalized = stepper.clean_input(xyz, h, state)
    assert torch.equal(normalized[:, 3:], (h - 0.25) / 4)
    assert torch.allclose((normalized[:, :3] + state.frame_offset[state.ligand_mask]) * 2, xyz)


def test_terminal_level_and_patch_restoration(core, pocket):
    stepper = NativeStepper(core)
    stream = GaussianStream(17)
    state = stepper.initialize(pocket, torch.tensor([3]), stream)
    with pytest.raises(ValueError, match="level zero"):
        stepper.finalize(state, stream)
    while state.level:
        state = stepper.reverse_step(state, stream)
    with pytest.raises(ValueError, match="Level zero"):
        stepper.reverse_step(state, stream)
    assert "sample_gaussian" not in core.__dict__


def test_nonfinite_state_is_a_hard_error(core, pocket):
    stepper = NativeStepper(core)
    state = stepper.initialize(pocket, torch.tensor([3]), GaussianStream(17))
    invalid = state.ligand.clone()
    invalid[0, 0] = float("nan")
    broken = NativeState(invalid, state.pocket, state.ligand_mask,
                         state.pocket_mask, state.level, state.frame_offset)
    with pytest.raises(ValueError, match="Nonfinite"):
        stepper.clean_estimate(broken)


def test_refuses_trainable_or_training_native_core():
    core = EngineeringCore().train()
    with pytest.raises(ValueError, match="frozen"):
        NativeStepper(core)


def test_unpinned_archive_manifest_is_rejected_before_source_import(tmp_path):
    (tmp_path / ".diffsbdd-source.json").write_text(json.dumps({
        "commit": "0" * 40, "members": {}}))
    with pytest.raises(RuntimeError, match="Unpinned"):
        authenticate_native_source(tmp_path)


def test_checkpoint_hash_is_checked_before_unpickling(tmp_path, monkeypatch):
    import latentmol.diffsbdd.native as native

    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"must not be unpickled")
    monkeypatch.setattr(native, "authenticate_native_source", lambda *args: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("Unauthenticated checkpoint reached torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(RuntimeError, match="checkpoint SHA256"):
        load_native_core(source, checkpoint, "0" * 64)
