"""The two frozen structure models Stage 2 can optimize through.

A backend hands `run.py` four things: a loss with the signature
``loss(graph, *, key) -> (value, aux)``, a starting graph, a JSON-serializable
provenance record, and a ``finish()`` callable that tears the backend down and
returns any end-of-run facts worth keeping. Everything downstream — the
operators, the diagnostics, the report, the tests — depends only on that
contract, which is why switching models does not touch them.

Boltz-2 is the default. Both backends minimize the returned value; what that
value *means* differs, and the difference matters:

===================  ==========================  ==============================
                     Boltz-2 (default)           Protenix-Tiny (legacy)
===================  ==========================  ==============================
Objective            ``affinity_pred_value``     mean max contact probability
Direction            lower is a stronger binder  negated score, so lower is
                                                 a higher contact score
Bond order           **differentiable**          **invisible**
Pose                 fixed native predicted      geometry off, ref_pos = 0
Runs on              allocated DTU HPC (LSF)     any CUDA box
===================  ==========================  ==============================

The bond-order row is the one that changes Stage 2's meaning. Protenix-Tiny
receives only bond existence, ``B = 1 - P(NO_BOND)``, so moving probability
between SINGLE/DOUBLE/TRIPLE/AROMATIC produced exactly zero gradient and the
valence constraint acted on a direction the objective could not see. Boltz-2
mixes the relaxed bond-type distribution into its pretrained bond-type
embedding (``bond_types @ model.token_bonds_type.weight``), so bond order is a
live gradient direction and the valence projection genuinely interacts with the
objective. ``boltz2/checks.py`` asserts that typed-bond contrast is nonzero.
"""

import json
from pathlib import Path

import numpy as np
from operators import BOND_STATES, ELEMENTS

ROOT = Path(__file__).resolve().parents[1]
STAGE1_CONFIG = ROOT / "examples/configs/small_molecule_stage1.json"

#: Which aux field each backend's score lives in, and which way is better.
METRICS = {
    "boltz2": {
        "score_metric": "affinity_pred_value",
        "units": "Boltz-2 affinity prediction",
        "interpretation": (
            "predicted binding affinity; lower is a stronger predicted binder, "
            "so the optimizer minimizes it directly. Not an experimental "
            "affinity and not calibrated."
        ),
    },
    "protenix": {
        "score_metric": "contact_score",
        "units": "mean max contact probability in [0, 1]",
        "interpretation": (
            "mean over ligand atoms of the strongest predicted contact with any "
            "protein token under an 8 A cutoff; higher is better, so the loss is "
            "its negation."
        ),
    },
}


def stage1_settings():
    return json.loads(STAGE1_CONFIG.read_text())


def parent_graph(parent):
    """Load a sealed parent run's committed final graph.

    Protocol 5 starts Stage 2 from the frozen Stage 1 optimizer, so only the
    graph is taken here. ``run.py`` then re-evaluates it under the backend and
    requires the parent's recorded final loss back before optimizing, which is
    what makes "same optimizer, plus operators" a checked claim rather than a
    description.
    """
    from mosaic.small_molecule.io import file_sha256, load_graph

    parent = Path(parent)
    pointer = json.loads((parent / "trajectory/latest_checkpoint.json").read_text())
    graph = load_graph(parent / "trajectory" / pointer["path"], pointer["sha256"])
    trace = [
        json.loads(line)
        for line in (parent / "trajectory/trace.jsonl").read_text().splitlines()
    ]
    if not trace[-1].get("final"):
        raise ValueError(f"{parent} did not finish; its last state is not final")
    return graph, {
        "parent": str(parent),
        "parent_checkpoint": pointer,
        "parent_input_sha256": file_sha256(parent / "input_identity.json"),
        "parent_final_loss": trace[-1]["loss"],
    }


def seed_arrays(preparation):
    """Rebuild the prepared carbon-chain seed as plain arrays.

    Read from ``preparation.json`` rather than out of ``context.pt``, because
    the context is a torch pickle and torch lives in the worker's environment,
    not this one. That is safe because the seed is fully determined by the
    declared SMILES: ``seed_smiles`` must be exactly ``N`` carbons, which is a
    linear alkane, so every atom is carbon and the only bonds are SINGLE
    between consecutive slots. Both are asserted, so a change in what
    preparation seeds becomes an error rather than a silent mismatch against
    the pose it was predicted for.

    Kept free of JAX and of ``mosaic`` so it stays testable on a laptop.
    """
    n_atoms = int(preparation["n_atoms"])
    smiles = preparation["seed_smiles"]
    if smiles != "C" * n_atoms:
        raise ValueError(
            f"prepared seed is {smiles!r}, not the {n_atoms}-carbon chain this "
            "backend reconstructs; the pose and the starting graph would differ"
        )
    if len(preparation["atom_indices"]) != n_atoms:
        raise ValueError("prepared atom slot count differs from n_atoms")
    atoms = np.zeros((n_atoms, len(ELEMENTS)))
    atoms[:, ELEMENTS.index("C")] = 1.0
    i, j = np.triu_indices(n_atoms, 1)
    edges = np.zeros((len(i), len(BOND_STATES)))
    chain = j == i + 1
    edges[chain, BOND_STATES.index("SINGLE")] = 1.0
    edges[~chain, BOND_STATES.index("NO_BOND")] = 1.0
    return atoms, edges


def _relaxed_seed(preparation, mixing):
    """Mix the prepared carbon-chain seed toward uniform, protocol 4.2.

    The Boltz-2 pose is predicted once for a specific N-carbon chain and then
    held fixed, so the relaxed start has to be that chain rather than a diffuse
    random graph: a graph unrelated to the pose would be scored against a pose
    that does not belong to it. Mixing with uniform keeps every probability
    strictly positive and the state movable.
    """
    import jax.numpy as jnp

    from mosaic.small_molecule.state import RelaxedGraph, mix_with_uniform

    atoms, edges = seed_arrays(preparation)
    seed = RelaxedGraph(
        jnp.asarray(atoms, dtype=jnp.float32), jnp.asarray(edges, dtype=jnp.float32)
    )
    return mix_with_uniform(seed, mixing), len(atoms)


def boltz2(args):
    """Start the native affinity worker and return its loss.

    Preparation is a separate one-off step that establishes the fixed pose::

        python -m mosaic.small_molecule.boltz2.prepare \\
          --fixture FIXTURE.json --cache BOLTZ_CACHE --output CONTEXT_DIR \\
          --n-atoms 25 [--pocket POCKET.json]

    Both that and the worker refuse to run without an allocated LSF job, so
    this backend needs DTU HPC rather than a local GPU.
    """
    import os
    import sys

    from mosaic.small_molecule.boltz2.bridge import AffinityWorker, affinity_loss

    context_dir = Path(args.boltz_context)
    preparation = json.loads((context_dir / "preparation.json").read_text())
    context_path = context_dir / "context.pt"
    # The worker is a separate process on purpose: it needs torch and the pinned
    # Boltz, while this side needs JAX, and those live in different
    # environments. Default to this interpreter for a single combined
    # environment; point --boltz-python at the Boltz venv when they are split.
    command = [
        str(args.boltz_python or sys.executable),
        "-m",
        "mosaic.small_molecule.boltz2.worker",
        "--context",
        str(context_path),
        "--context-sha256",
        preparation["context_sha256"],
        "--checkpoint",
        str(args.boltz_checkpoint),
        "--device",
        "cpu" if args.allow_cpu else "cuda",
    ]
    if args.allow_local_gpu:
        command.append("--allow-local-gpu")
    environment = None
    if args.boltz_python:
        # A separate interpreter still has to find this checkout's mosaic, so
        # pass our own source root through rather than installing into its venv.
        source = str(ROOT / "src")
        existing = os.environ.get("PYTHONPATH")
        environment = {
            **os.environ,
            "PYTHONPATH": f"{source}{os.pathsep}{existing}" if existing else source,
        }
    worker = AffinityWorker(
        command,
        stderr=Path(args.out) / "boltz_worker.stderr",
        environment=environment,
    )
    try:
        loss = affinity_loss(worker)
        graph, n_atoms = _relaxed_seed(preparation, args.seed_mixing)
        provenance = {
            "backend": "boltz2",
            "worker_identity": worker.identity,
            "preparation": preparation,
            "context_sha256": preparation["context_sha256"],
            "n_atoms": n_atoms,
            "initialization": {
                "kind": "prepared carbon-chain seed mixed with uniform",
                "mixing": args.seed_mixing,
            },
            **METRICS["boltz2"],
        }
        if args.parent:
            # The pose stays the prepared one; only the graph is inherited.
            graph, inherited = parent_graph(args.parent)
            provenance.update(inherited)
            provenance["initialization"] = {"kind": "sealed parent checkpoint"}
    except BaseException:
        worker.close()
        raise

    def finish():
        """Close the worker and report what the device actually cost."""
        peak = worker.peak_vram_bytes
        worker.close()
        return {
            "peak_vram_bytes": peak,
            "peak_vram_gib": round(peak / 1024**3, 2) if peak else None,
            "device_total_vram_gib": round(
                worker.identity["device_total_vram_bytes"] / 1024**3, 2
            )
            if "device_total_vram_bytes" in worker.identity
            else None,
        }

    return loss, graph, provenance, finish


def protenix(args):
    """The Stage 1 contact objective through frozen Protenix-Tiny.

    Kept so the earlier matched runs stay reproducible. Bond order is a null
    gradient direction here, so a Stage 2 result from this backend says only
    that the operators behaved, not that the objective and the chemistry
    interacted.
    """
    import jax

    from mosaic.small_molecule.io import load_context
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.state import initialize_graph

    stage1 = stage1_settings()
    model = FrozenProtenixTiny.load(cache_dir=args.cache_dir)
    provenance = {
        "backend": "protenix",
        "target": args.target,
        "model": model.identities,
        "n_atoms": stage1["n_atoms"],
        "bond_order_gradient": "null: the adapter transmits bond existence only",
        **METRICS["protenix"],
    }

    if args.parent:
        # Reuse the parent's own sealed context, not a freshly prepared one:
        # MSA and featurization choices must be identical to continue it.
        parent = Path(args.parent)
        identity = json.loads((parent / "input_identity.json").read_text())
        context = load_context(parent / "inputs", identity["context_files"])
        graph, inherited = parent_graph(parent)
        provenance.update(inherited)
        provenance["initialization"] = {"kind": "sealed parent checkpoint"}
    else:
        from mosaic.small_molecule.features import prepare_context
        from mosaic.structure_prediction import TargetChain

        panel = json.loads((ROOT / stage1["panel"]).read_text())
        target = next(t for t in panel["targets"] if t["id"] == args.target)
        context, carrier = prepare_context(
            stage1["carrier_smiles"],
            protein_chains=(TargetChain(target["protein_sequence"], use_msa=False),),
            cache_dir=args.cache_dir,
            input_name=f"stage2_{target['id']}",
        )
        if carrier.n_atoms != stage1["n_atoms"]:
            raise ValueError("carrier slot count differs from the sealed Stage 1 N")
        init_key = jax.random.split(jax.random.key(args.seed))[0]
        graph = initialize_graph(
            stage1["n_atoms"], init_key, **stage1["initialization"]
        )
        provenance["initialization"] = stage1["initialization"]

    loss = model.build_loss(
        context=context,
        loss=NativeContactLoss(
            cutoff_angstrom=stage1["loss"]["contact_cutoff_angstrom"]
        ),
        recycling_steps=stage1["recycling_steps"],
    )
    return loss, graph, provenance, dict


BACKENDS = {"boltz2": boltz2, "protenix": protenix}


def build(args):
    """Return ``(loss, graph, provenance, finish)`` for the selected backend."""
    return BACKENDS[args.backend](args)
