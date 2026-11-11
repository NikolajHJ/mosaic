"""Connect a ligand design run to a prepared Boltz2 affinity model.

Boltz2 predicts the starting pose during a separate preparation step. This
file checks that the saved pose belongs to the requested protein and starts
a GPU worker to score changes to the ligand while holding that pose fixed.
The JAX optimizer remains on CPU and receives gradients from the worker.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def open_affinity(general, general_path, target, output, n_atoms=None):
    import jax

    from .bridge import AffinityWorker, affinity_loss

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Boltz2 needs CPU JAX and a separate CUDA worker.")

    def path(value):
        return (general_path.parent / Path(value).expanduser()).resolve()

    settings = general["boltz2"]
    context = path(settings["prepared_context"])
    receipt = json.loads((context / "preparation.json").read_text(encoding="utf-8"))
    if receipt["protein_sequence"] != target["protein_sequence"]:
        raise ValueError("The prepared Boltz2 context is for a different protein sequence.")
    if n_atoms is not None and receipt["n_atoms"] != n_atoms:
        raise ValueError("The prepared Boltz2 context has a different number of ligand slots.")

    # Only Stage 0 can request a particular pocket. A prepared pose must
    # have passed the audit for those exact residues before using that label.
    audit = receipt.get("pocket_pose_audit", {})
    if target["pocket"]["kind"] != "whole_target":
        expected = sorted(row["sequence_position"] for row in target["residues"])
        if audit.get("status") != "passed" or sorted(audit.get("protein_sequence_positions", [])) != expected:
            raise ValueError("A selected pocket requires a Boltz2 pose audited for those residues.")

    command = [
        str(path(settings["worker_python"])), "-u", "-m", "mosaic.small_molecule.core.boltz2.worker",
        "--context", str(context / "context.pt"),
        "--context-sha256", receipt["context_sha256"],
        "--checkpoint", str(path(settings["checkpoint"])),
    ]
    src = str(Path(__file__).resolve().parents[4])
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, (src, os.environ.get("PYTHONPATH"))))}
    worker = AffinityWorker(command, stderr=output / "boltz_worker.log", env=env)
    try:
        if (
            worker.identity["n_atoms"] != receipt["n_atoms"]
            or worker.identity.get("pocket_definition_sha256") != receipt.get("pocket_definition_sha256")
        ):
            raise ValueError("The prepared context does not match its preparation record.")
        yield affinity_loss(worker), worker.identity, receipt
    finally:
        worker.close()
