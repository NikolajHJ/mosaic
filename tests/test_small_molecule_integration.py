"""Real two-mode acceptance on one allocated GPU and one resident Tiny.

Missing HPC allocation/assets are explicit failures, not silent skips. These
tests are authored only during the access outage; executing them belongs in
the later immutable-source DTU validation allocation.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import jax
import pytest


@pytest.fixture(scope="session")
def native_stage0_suite(tmp_path_factory):
    if not os.environ.get("LSB_JOBID"):
        pytest.fail("Real Stage 0 acceptance requires an allocated DTU LSF job")
    cache = os.environ.get("MOSAIC_STAGE0_CACHE")
    if not cache:
        pytest.fail(
            "Set MOSAIC_STAGE0_CACHE to the authenticated, pre-staged native asset directory"
        )
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.fail("Real Tiny acceptance requires the GPU allocated to this job")
    source_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (source_root / "examples/configs/small_molecule_stage0.json").read_text()
    )
    config.update(cache_dir=str(Path(cache).resolve()), source_root=str(source_root))
    fixture = json.loads((source_root / config["fixture"]).read_text())
    root = Path(
        os.environ.get(
            "MOSAIC_STAGE0_RESULTS", tmp_path_factory.mktemp("stage0-native")
        )
    )
    directory = root / "native_acceptance"
    directory.mkdir(parents=True, exist_ok=False)

    # Lazy imports after cache/allocation checks keep ordinary collection from
    # preparing molecules, downloading assets or loading checkpoint stacks.
    from mosaic.small_molecule.features import prepare_context
    from mosaic.small_molecule.protenix import FrozenProtenixTiny

    with patch.object(
        FrozenProtenixTiny, "load", wraps=FrozenProtenixTiny.load
    ) as loader:
        model = FrozenProtenixTiny.load(cache_dir=cache)
        contexts = {}
        for mode, chains in (
            ("ligand_only", ()),
            ("protein_ligand", (fixture["protein_sequence"],)),
        ):
            contexts[mode] = prepare_context(
                fixture["smiles"],
                protein_chains=chains,
                cache_dir=cache,
                input_name="stage0-" + mode,
            )
        yield {
            "model": model,
            "contexts": contexts,
            "settings": config,
            "directory": directory,
            "loader": loader,
        }
        assert loader.call_count == 1, (
            "Acceptance reloaded Tiny within its resident parent process"
        )


@pytest.mark.slow
@pytest.mark.small_molecule_model
@pytest.mark.parametrize("mode", ["ligand_only", "protein_ligand"])
def test_real_two_mode_stage0_acceptance(native_stage0_suite, mode):
    from mosaic.small_molecule.validation import validate_pipeline

    suite = native_stage0_suite
    context, hard_graph = suite["contexts"][mode]
    key = jax.random.key(suite["settings"]["seed"])
    report = validate_pipeline(
        suite["model"],
        context,
        hard_graph,
        key=key,
        output_directory=suite["directory"] / mode,
        settings=suite["settings"],
    )
    assert report["status"] == "passed", json.dumps(report, indent=2)
    assert report["mode"] == mode
    assert all(report["gates"][letter]["status"] == "passed" for letter in "ABCDEFG")
    assert report["gates"]["F"]["evidence"]["fresh_process"]
    assert report["gates"]["G"]["evidence"]["sampling_calls"] == 0
    assert report["gates"]["G"]["evidence"]["confidence_calls"] == 0
    assert suite["loader"].call_count == 1
