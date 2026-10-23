"""A C13 native input must retain the pinned protein/MSA and change its ligand."""

import json
from pathlib import Path

import pytest
import yaml

from latentmol.boltz import prepare
from latentmol.experiments import file_sha256


def test_chain_example_keeps_protein_and_declares_c13(tmp_path, monkeypatch):
    official = {
        "sequences": [
            {"protein": {"id": "A", "sequence": "ACDE"}},
            {"ligand": {"id": "B", "smiles": "N[C@@H](C)C(=O)O"}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }

    def fake_download(url, destination):
        destination = Path(destination)
        if url.endswith("affinity.yaml"):
            destination.write_text(yaml.safe_dump(official))
        elif url.endswith("seq1.a3m"):
            destination.write_text(">query\nACDE\n")
        else:
            raise AssertionError(f"Unexpected asset: {url}")

    monkeypatch.setattr(prepare.urllib.request, "urlretrieve", fake_download)
    path = prepare.make_chain_example(tmp_path / "input")
    derived = yaml.safe_load(path.read_text())
    manifest = json.loads((path.parent / "input_manifest.json").read_text())
    assert path.name == "carbon_chain_smoke.yaml"
    assert derived["sequences"][0]["protein"]["sequence"] == "ACDE"
    assert Path(derived["sequences"][0]["protein"]["msa"]).name == "seq1.a3m"
    assert derived["sequences"][1]["ligand"]["smiles"] == "CCCCCCCCCCCCC"
    assert derived["properties"] == official["properties"]
    assert manifest["seed_smiles"] == "CCCCCCCCCCCCC"
    assert manifest["ligand_source"] == "declared_linear_c13_stereo_free"


def _cached_example_fixture(tmp_path, monkeypatch):
    """Use synthetic asset hashes solely to test the pinned-cache mechanism."""
    cache = tmp_path / "cache"
    cache.mkdir()
    official = {"sequences": [
        {"protein": {"id": "A", "sequence": "ACDE"}},
        {"ligand": {"id": "B", "smiles": "CC"}}],
        "properties": [{"affinity": {"binder": "B"}}]}
    (cache / "official_affinity.yaml").write_text(yaml.safe_dump(official))
    (cache / "seq1.a3m").write_text(">query\nACDE\n")
    expected = {name: file_sha256(cache / name)
                for name in ("official_affinity.yaml", "seq1.a3m")}
    monkeypatch.setattr(prepare, "EXAMPLE_ASSET_SHA256", expected)

    def forbidden_download(*args, **kwargs):
        pytest.fail("Cached count preparation must never use the network")

    monkeypatch.setattr(prepare.urllib.request, "urlretrieve", forbidden_download)
    return cache, expected


def test_count_example_uses_verified_cached_assets_without_network(tmp_path, monkeypatch):
    cache, expected = _cached_example_fixture(tmp_path, monkeypatch)
    path = prepare.make_count_example(tmp_path / "input", 4, assets_dir=cache)
    value = yaml.safe_load(path.read_text())
    manifest = json.loads((path.parent / "input_manifest.json").read_text())
    assert path.name == "carbon_chain_04.yaml"
    assert value["sequences"][1]["ligand"]["smiles"] == "CCCC"
    assert value["sequences"][0]["protein"]["sequence"] == "ACDE"
    assert Path(value["sequences"][0]["protein"]["msa"]) == (path.parent / "seq1.a3m").resolve()
    assert manifest["seed_smiles"] == "CCCC"
    assert manifest["derived_yaml_sha256"] == file_sha256(path)
    for name, digest in expected.items():
        assert manifest["assets"][name]["sha256"] == digest
        assert (path.parent / name).read_bytes() == (cache / name).read_bytes()


@pytest.mark.parametrize("asset", ["official_affinity.yaml", "seq1.a3m"])
def test_count_example_rejects_tampered_cache_without_downloading(tmp_path, monkeypatch, asset):
    cache, _ = _cached_example_fixture(tmp_path, monkeypatch)
    path = cache / asset
    path.write_bytes(path.read_bytes() + b"\n# modified\n")
    output = tmp_path / "input"
    with pytest.raises(ValueError, match="Cached official example asset hash mismatch"):
        prepare.make_count_example(output, 4, assets_dir=cache)
    assert not (output / "input_manifest.json").exists()
    assert not (output / "carbon_chain_04.yaml").exists()


def test_count_example_does_not_download_a_missing_cached_asset(tmp_path, monkeypatch):
    cache, _ = _cached_example_fixture(tmp_path, monkeypatch)
    (cache / "seq1.a3m").unlink()
    with pytest.raises(FileNotFoundError):
        prepare.make_count_example(tmp_path / "input", 4, assets_dir=cache)
