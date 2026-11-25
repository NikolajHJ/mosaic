"""Fresh checkpoint validity/diversity observations, separate from training gates.

No repair, score selection, filtering or training occurs. Repeated evaluation
uses the same newly declared prior seeds across epochs40--45. Full invalid and
duplicate raw graphs remain in the saved denominator. Categorical draws sharing
a latent are correlated. These panels are not an independent full-test study.
"""
import gzip
import hashlib
import json
from pathlib import Path

import torch

from .chemistry_early_health import preserve_observation_state
from .chemistry_epoch45_extension import POLICY, validate_extension
from .experiments import file_sha256, write_json
from .large_vae_evaluation import evaluate_codes
from .train import state_digest


def structural_diversity(rows):
    """Stable hash-subsampled unique identities; lower similarity means diversity."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    identities = sorted({r["smiles"] for r in rows if r["valid"]},
                        key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints, scaffolds = [], set()
    for identity in identities:
        mol = Chem.MolFromSmiles(identity)
        if mol is None:
            raise ValueError("Saved valid identity failed reparse")
        scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        if len(fingerprints) < 256:
            fingerprints.append(generator.GetFingerprint(mol))
    total, pairs = 0., 0
    for i, fingerprint in enumerate(fingerprints):
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[:i])
        total += sum(values); pairs += len(values)
    return dict(unique_valid=len(identities), unique_scaffolds=len(scaffolds),
        empty_scaffold_included=("" in scaffolds), fingerprint_cohort_size=len(fingerprints),
        fingerprint_pairs=pairs, mean_tanimoto=(total / pairs if pairs else None),
        internal_diversity=(1. - total / pairs if pairs else None),
        cohort_policy="first256 unique valid canonical identities ordered by SHA256(identity); no score selection",
        limitation="Bounded unique-identity cohort; not full-population diversity or synthesis feasibility")


def classify_saved_graphs(path, spec):
    """Keep native chemistry failures distinct from serialization/schema failures."""
    from .chemistry_classification import classify_repair_graph, classification_summary
    contexts = {"argmax": [], "categorical": []}
    rows = {"argmax": [], "categorical": []}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line); graph = row["graph"]
            bonds = torch.zeros((spec.max_nodes, spec.max_nodes), dtype=torch.long)
            for i, j, label in graph["edges"]:
                bonds[i, j] = bonds[j, i] = label
            assessment = classify_repair_graph(torch.tensor(graph["atoms"], dtype=torch.long),
                torch.tensor(graph["charges"], dtype=torch.long), bonds, spec)
            contexts[row["method"]].append(dict(classification=assessment,
                candidate_id=f'{row["latent_index"]}:{row["categorical_draw"]}'))
            rows[row["method"]].append(row)
    return {method: dict(classification=classification_summary(contexts[method], max_examples_per_failure=4),
                         structural_diversity=structural_diversity(rows[method])) for method in contexts}


def observe_checkpoint(plan, model, generators, output, *, checkpoint, checkpoint_sha256, epoch, updates):
    """Observe a saved checkpoint, accepting either serialized or Path inputs.

    Worker constants and saved epoch rows carry paths as strings, while direct
    callers often supply ``Path`` objects. Normalize once at this public
    boundary; checksum, checkpoint identity and state-preservation guards are
    identical for both representations.
    """
    if validate_extension(plan) is None or type(epoch) is not int or not 40 <= epoch <= 45:
        raise ValueError("Checkpoint observation requires the declared40->45 extension")
    checkpoint = Path(checkpoint)
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Observed checkpoint bytes differ")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve earlier checkpoint-generation observations")
    output.mkdir(parents=True)
    policy = POLICY["generation"]
    before = state_digest(model.state_dict())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("epoch") != epoch or payload.get("continuation", {}).get("updates") != updates
            or state_digest(payload["model_state"]) != before):
        raise ValueError("Observation model/epoch/update differs from the saved checkpoint")
    del payload
    seeds = []
    with preserve_observation_state(model, generators):
        for seed in policy["seeds"]:
            codes = torch.randn(policy["prior_count"], model.config.latent,
                                generator=torch.Generator().manual_seed(seed))
            protocol = dict(evaluation_seed=seed, evaluation_batch_size=policy["evaluation_batch_size"],
                            categorical_samples_per_latent=policy["categorical_samples_per_latent"])
            raw = output / f"prior-seed-{seed}.jsonl"
            metrics = evaluate_codes(model, codes, raw, protocol)
            raw = Path(str(raw) + ".gz")
            seeds.append(dict(seed=seed, sample_seed=seed + 1, prior_count=len(codes),
                latent_sha256=state_digest({"codes": codes}), metrics=metrics,
                detailed=classify_saved_graphs(raw, model.spec),
                raw_graphs=dict(path=str(raw), sha256=file_sha256(raw))))
    if state_digest(model.state_dict()) != before:
        raise RuntimeError("Generation observer changed model state")
    result = dict(version=policy["version"], status="observed_not_a_training_gate", epoch=epoch,
        updates=updates, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256,
        model_sha256=before, plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], policy=policy, seeds=seeds,
        scope="Fresh-seed raw checkpoint observation; no repair/filtering/model rerun for saved raw summaries",
        uncertainty="Four draws per latent are correlated; fixed panels across checkpoints are paired, not independent",
        state_preservation="All native/CUDA/named-generator RNG states and module modes restored; model digest unchanged")
    write_json(output / "metrics.json", result)
    return dict(path=str(output / "metrics.json"), sha256=file_sha256(output / "metrics.json"))
