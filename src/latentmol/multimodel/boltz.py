"""Load the existing Boltz branch without changing its partial-gradient policy."""

from pathlib import Path

from latentmol.boltz.contracts import AFFINITY_SHA256, PreparedComplex
from latentmol.boltz.model import FrozenBoltzAffinity, load_affinity_model
from latentmol.boltz.scoring import BoltzGraphBranch, boltz_branch_manifest
from latentmol.experiments import file_sha256
from .runtime import BranchRuntime


def load_boltz(*, spec, device, options, source_sha256):
    """Inputs must already be prepared outside the differentiable experiment."""
    required = {"prepared", "prepared_sha256", "checkpoint", "recycling_steps", "allow_lossy_charge"}
    if set(options) - required - {"dbmol"} or required - set(options):
        raise ValueError(f"boltz2 options require exactly {sorted(required)}")
    path = Path(options["prepared"])
    if file_sha256(path) != options["prepared_sha256"]:
        raise ValueError("Boltz prepared-file SHA256 mismatch")
    prepared = PreparedComplex.load(path, device)
    if prepared.spec != spec:
        raise ValueError("Boltz and decoder graph schemas differ")
    model = load_affinity_model(Path(options["checkpoint"])).to(device)
    if options.get("dbmol") is None:
        scorer = FrozenBoltzAffinity(model, options["recycling_steps"])
    else:
        from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig
        from latentmol.boltz.dbmol_model import FrozenBoltzDBMol, load_confidence_model
        profile = DBMolObjectiveConfig.from_mapping(options["dbmol"], Path.cwd())
        profile.verify_reference(prepared, options["prepared_sha256"])
        structure = load_confidence_model(profile.confidence_checkpoint).to(device)
        scorer = FrozenBoltzDBMol(model, structure, profile, options["recycling_steps"])
    manifest = boltz_branch_manifest(prepared, adapter_source_sha256=source_sha256,
                                     recycling_steps=scorer.recycling_steps,
                                     dbmol_profile=getattr(scorer, "manifest_config", None))
    branch = BoltzGraphBranch(scorer, prepared, manifest,
                              allow_lossy_charge=options["allow_lossy_charge"])
    receipt = {"prepared": str(path), "prepared_sha256": file_sha256(path),
               "checkpoint": options["checkpoint"],
               "checkpoint_sha256": AFFINITY_SHA256,
               "validation_status": "existing adapter; new runner unvalidated on HPC"}
    if options.get("dbmol") is not None:
        receipt.update(confidence_checkpoint=profile.confidence_checkpoint,
                       combined_checkpoint_identity=manifest.model_weights_sha256,
                       dbmol_profile=profile.record(),
                       validation_status="DBMol composite: native/component gradient gates required")
    return BranchRuntime(branch, prepared.seed, prepared.active,
                         lambda: scorer.discrete_reference(prepared.features, prepared.coordinates),
                         receipt,
                         branch.verify_context_unchanged)
