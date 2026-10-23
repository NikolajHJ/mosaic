"""Stage 1 paired audit identities and denominator checks; run on HPC."""

import pytest

from latentmol.latent_comparison import _counts, _paired_interval


def _path(anchor, label, valid):
    points = [{"decoded": {"smiles": "C" if flag else None}} for flag in valid]
    return {"left_index": anchor, "pair_class": label, "points": points}


def test_paired_interval_clusters_paths_by_anchor():
    ae = [_path(1, "near", [1, 0, 0, 0, 1]), _path(1, "far", [1, 0, 0, 0, 1]),
          _path(2, "near", [1, 1, 1, 1, 1]), _path(2, "far", [1, 1, 1, 1, 1])]
    vae = [_path(1, "near", [1, 1, 1, 1, 1]), _path(1, "far", [1, 1, 1, 1, 1]),
           _path(2, "near", [1, 1, 1, 1, 1]), _path(2, "far", [1, 1, 1, 1, 1])]
    result = _paired_interval(ae, vae, None, draws=100)
    assert result["anchor_molecules"] == 2
    assert result["intermediate_attempts_per_model"] == 12
    assert result["vae_minus_ae"] == 0.5
    assert _counts(ae)["all_five_valid_and_some_distinct"] == 0
