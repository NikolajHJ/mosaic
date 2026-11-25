"""Fixed-pose Boltz2 DBMol scoring for the shared restart-search engine.

Continuous input graphs retain their original occupied-slot order. Hard scoring
uses the saved graph labels in that same order; canonical SMILES is only a
chemical identity and cannot replace the ligand-to-coordinate correspondence.
Native Boltz imports and weight loading occur only in ``load``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from time import perf_counter
from types import MethodType

import torch

from latentmol.boltz.contracts import AFFINITY_SHA256, PreparedComplex
from latentmol.boltz.provenance import context_digest
from latentmol.boltz.scoring import BoltzGraphBranch, boltz_branch_manifest
from latentmol.chemistry import inspect_graph
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec, HardGraphBatch, RelaxedGraph


def _record_tensors(record):
    if not isinstance(record, dict) or set(record) != {"atoms", "charges", "bonds"}:
        raise ValueError("Hard graph requires unbatched atoms, charges and bonds")
    atoms, charges, bonds = (torch.as_tensor(record[key]) for key in ("atoms", "charges", "bonds"))
    n = atoms.numel()
    if (atoms.ndim != 1 or charges.shape != (n,) or bonds.shape != (n, n)
            or any(value.dtype != torch.long for value in (atoms, charges, bonds))
            or any(bool((value < 0).any()) for value in (atoms, charges, bonds))
            or not bool((atoms != 0).any())):
        raise ValueError("Hard graph requires nonempty integer labels with compatible shapes")
    return atoms, charges, bonds


def hard_pose_key(hard_graph: dict) -> str:
    """Hash exact occupied labels in fixed native pose slot order, not SMILES.

    This key is scoped to one prepared context and graph schema; its surrounding
    run identity authenticates those assets. Moving padding alone is immaterial.
    Permuting occupied chemical identities generally changes the key and score.
    """
    atoms, charges, bonds = _record_tensors(hard_graph)
    active = atoms != 0
    payload = dict(version="fixed-pose-dbmol-graph-v1", atoms=atoms[active].tolist(),
                   charges=charges[active].tolist(), bonds=bonds[active][:, active].tolist())
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def hard_graph_record(hard: HardGraphBatch) -> dict:
    if hard.atoms.ndim != 2 or hard.atoms.shape[0] != 1:
        raise ValueError("Hard scoring requires one graph")
    return {key: getattr(hard, key)[0].detach().cpu().tolist() for key in ("atoms", "charges", "bonds")}


def _module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_execution(raw=None):
    """Prospective execution choices; absent settings retain the FP32 path.

    Checkpoint cadence changes rematerialization, not block mathematics. TF32
    and BF16 change numerical arithmetic and need measured value/gradient and
    optimization comparisons before adoption. Native fused kernels/compilation
    are deliberately outside this small experiment interface.
    """
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - {"tf32", "autocast", "checkpoint_every", "checkpoint_preserve_rng_state"}:
        raise ValueError("Unsupported Boltz execution option")
    result = dict(tf32=False, autocast="none", checkpoint_every=1, checkpoint_preserve_rng_state=True) | raw
    if (type(result["tf32"]) is not bool or result["autocast"] not in ("none", "bfloat16", "bfloat16_blocks")
            or type(result["checkpoint_preserve_rng_state"]) is not bool
            or type(result["checkpoint_every"]) is not int or not 0 <= result["checkpoint_every"] <= 64):
        raise ValueError("Invalid TF32, autocast or checkpoint cadence setting")
    return result


def _float32_tree(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_float32_tree(item) for item in value)
    if isinstance(value, list):
        return [_float32_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _float32_tree(item) for key, item in value.items()}
    return value


def _checkpoint_cadence(scorer, every, preserve_rng, *, autocast_blocks=False, device_type="cuda"):
    if not preserve_rng:
        for name, module in scorer.named_modules():
            if module.training or getattr(module, "subsample_msa", False):
                raise ValueError(f"Disabling checkpoint RNG preservation requires eval and fixed MSA: {name}")
    configured = []
    for name, block in scorer.named_modules():
        original = getattr(block, "_latentmol_original_forward", None)
        if original is None:
            continue
        if not hasattr(block, "_latentmol_search_checkpoint_forward"):
            block._latentmol_search_checkpoint_forward = block.forward
        match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
        if match is None and every > 1:
            raise ValueError(f"Checkpoint cadence requires an explicit native layer index: {name}")
        keep = every != 0 and (every == 1 or int(match[1]) % every == 0)
        if not preserve_rng and not hasattr(block, "_latentmol_search_no_rng_forward"):
            def no_rng(*args, _native=original, _block=block, **kwargs):
                from torch.utils.checkpoint import checkpoint

                if _block.training:
                    raise ValueError("Checkpoint RNG opt-out requires frozen native eval mode")
                if not torch.is_grad_enabled():
                    return _native(*args, **kwargs)
                return checkpoint(_native, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
            block._latentmol_search_no_rng_forward = no_rng
        # Instance-local callables only: no module-class/global alias or
        # registered weight/buffer changes, and native parent loops stay.
        checkpointed = (block._latentmol_search_checkpoint_forward if preserve_rng
                        else block._latentmol_search_no_rng_forward)
        block.forward = checkpointed if keep else original
        if autocast_blocks:
            # Native install_eval_checkpoints marks only pairformer/MSA neural
            # blocks. Their parent coordinate pooling and discrete binning do
            # not enter this autocast region. Cast at each block boundary so
            # following geometry and heads continue to receive FP32 tensors.
            def neural_amp(*args, _forward=block.forward, **kwargs):
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    result = _forward(*args, **kwargs)
                return _float32_tree(result)
            block.forward = neural_amp
        configured.append((name, block, keep, block.forward))
    if (every != 1 or autocast_blocks) and not configured:
        raise ValueError("Checkpoint cadence found no installed native checkpoint blocks")
    return dict(installed_blocks=len(configured), checkpointed_blocks=sum(row[2] for row in configured),
                direct_blocks=sum(not row[2] for row in configured),
                preserve_rng_state=preserve_rng,
                neural_block_autocast=autocast_blocks,
                rng_scope=("Original checkpoint RNG replay" if preserve_rng else
                           "Eval/fixed-MSA opt-out; recomputation advances process RNG; finite value/gradient comparison required"),
                selection="Per native layers index modulo checkpoint_every; zero disables all"), configured


def _fp32_dbmol_reduction(scorer, device_type):
    original = getattr(scorer, "_reduce", None)
    if not callable(original):
        raise ValueError("BF16 execution requires the explicit DBMol reduction boundary")

    def cast(value):
        return value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value

    def reduce(module, affinity, structure, distogram, features, *, pair=None):
        # Keep FP32 inputs to the existing reduction; its explicitly configured
        # FP64 contact-head/assembly variant, when selected, still widens there.
        with torch.autocast(device_type=device_type, enabled=False):
            return original({key: cast(value) for key, value in affinity.items()},
                            {key: cast(value) for key, value in structure.items()},
                            cast(distogram), features, pair=cast(pair))

    scorer._reduce = MethodType(reduce, scorer)


class BoltzSearchScorer:
    """Two frozen Boltz2 trunks, live relaxed chemistry and detached fixed pose.

    The decoder may remain on CPU while this object resides on the GPU. Device
    copies stay in autograd. A branch's sorted occupied graph slots map to the
    prepared ligand slots in their declared order. Charges use their expected
    formal value; the full categorical charge distribution is explicitly lossy.
    ``close`` verifies identity without unloading, so one model pair can serve
    every optimization wave and the detached hard-scoring calls between waves.
    """

    label = "Boltz2 DBMol fixed-pose restart search"
    gradient_scope = ("Shared conditioned decoder probabilities -> frozen Boltz2 affinity and "
                      "structure/confidence trunks -> DBMol total -> latent; fixed pose and occupancy")
    claim_limit = ("Exploratory fixed-pose DBMol surrogate optimization; changed molecules are not "
                   "fresh-pose predictions, and no diffusion gradient is used")

    def __init__(self, prepared: PreparedComplex, scorer, *, source_sha256: str,
                 allow_lossy_charge=True, asset_identity=None, execution=None):
        if not isinstance(allow_lossy_charge, bool):
            raise ValueError("Declare the lossy expected-charge convention as a boolean")
        profile = getattr(scorer, "manifest_config", None)
        if not isinstance(profile, dict) or profile.get("profile") != "dbmol_fixed_pose_v1":
            raise ValueError("Restart scorer requires the explicit full DBMol profile")
        self.spec, self.prepared, self.scorer = prepared.spec, prepared, scorer
        scorer.eval()
        self.device = prepared.coordinates.device
        self.execution = resolve_execution(execution)
        if self.execution["autocast"] != "none":
            if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 autocast is unsupported by the allocated CUDA device")
        if self.execution["autocast"] == "bfloat16":
            _fp32_dbmol_reduction(scorer, self.device.type)
        checkpoint_record, self._checkpoint_blocks = _checkpoint_cadence(scorer, self.execution["checkpoint_every"],
            self.execution["checkpoint_preserve_rng_state"],
            autocast_blocks=self.execution["autocast"] == "bfloat16_blocks", device_type=self.device.type)
        # These CUDA flags are process-global. Concurrent branches in this
        # process must share one policy, including during checkpoint backward.
        torch.backends.cuda.matmul.allow_tf32 = self.execution["tf32"]
        torch.backends.cudnn.allow_tf32 = self.execution["tf32"]
        self.execution_record = dict(version="boltz-search-execution-v1", **self.execution,
            checkpoint_policy=checkpoint_record,
            numerical_variant=self.execution["tf32"] or self.execution["autocast"] != "none",
            checkpoint_math="Same native block calls; only activation rematerialization cadence changes",
            graph_input_dtype="float32", parameter_dtype="Unchanged frozen checkpoint dtypes",
            autocast_scope=("Whole native scorer before the DBMol reduction boundary, including geometry helpers; "
                            "reduced precision can change rounded native distance-bin assignments"
                            if self.execution["autocast"] == "bfloat16" else
                            "Checkpoint-marked native pairformer/MSA neural blocks only; outputs cast to FP32 at each block boundary; "
                            "parent geometry/binning, heads and reductions stay outside autocast"
                            if self.execution["autocast"] == "bfloat16_blocks" else "Disabled"),
            reduction_scope=("Upcast head primitives, distogram and pair to FP32 with autocast disabled; "
                             "then use the configured native/FP64 DBMol reductions"
                             if self.execution["autocast"] == "bfloat16" else "Unchanged configured DBMol arithmetic"),
            tf32_scope="Process-global CUDA matmul and cuDNN; all concurrent branches must share this policy",
            native_kernels=False, compiled=False)
        self.active = prepared.active.detach().clone()
        self._slots = self.active.nonzero().flatten()
        self.manifest = boltz_branch_manifest(prepared, adapter_source_sha256=source_sha256,
            recycling_steps=scorer.recycling_steps, dbmol_profile=profile)
        self.branch = BoltzGraphBranch(scorer, prepared, self.manifest,
                                      allow_lossy_charge=allow_lossy_charge)
        self._model_digest = _module_digest(scorer)
        self.identity = dict(protocol="boltz2-restart-dbmol-v1", graph_spec=self.spec.to_dict(),
            model_digest=self._model_digest, context_digest=context_digest(prepared),
            dbmol=profile, source_sha256=source_sha256, assets=asset_identity or {},
            allow_lossy_charge=allow_lossy_charge, charge_policy="Expected formal charge, categorical distribution lost",
            pose_policy="One detached native reference pose, fixed across every candidate",
            slot_policy="Ascending occupied graph slots to prepared ligand-slot order",
            hard_score_cache="Exact packed graph labels within this frozen context; never SMILES alone",
            optimized_output="dbmol_total_loss", diffusion_gradient=False,
            physical_trunks_per_forward=2, execution=self.execution_record)
        self.last_aux = {}
        self.last_native_seconds = None
        self.last_batch_size = None
        self.shutdown_status = None

    @classmethod
    def load(cls, config: dict, spec: GraphSpec, device="cuda", *, source_sha256: str):
        from latentmol.boltz.dbmol_model import FrozenBoltzDBMol, load_confidence_model
        from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig
        from latentmol.boltz.model import load_affinity_model

        prepared_path = Path(config["prepared"])
        if file_sha256(prepared_path) != config["prepared_sha256"]:
            raise ValueError("Boltz prepared-file SHA256 differs")
        if config["checkpoint_sha256"] != AFFINITY_SHA256:
            raise ValueError("Boltz affinity checkpoint declaration differs from the pinned model")
        prepared = PreparedComplex.load(prepared_path, device)
        if prepared.spec != spec:
            raise ValueError("Boltz and decoder graph schemas differ")
        profile = DBMolObjectiveConfig.from_mapping(config["dbmol"], Path.cwd())
        profile.verify_reference(prepared, config["prepared_sha256"])
        affinity = load_affinity_model(Path(config["checkpoint"])).to(device)
        confidence = load_confidence_model(profile.confidence_checkpoint).to(device)
        scorer = FrozenBoltzDBMol(affinity, confidence, profile, config.get("recycling_steps", 0))
        assets = {key: config[key] for key in ("prepared", "prepared_sha256", "checkpoint", "checkpoint_sha256")}
        assets["confidence_checkpoint"] = profile.confidence_checkpoint
        return cls(prepared, scorer, source_sha256=source_sha256,
                   allow_lossy_charge=config.get("allow_lossy_charge", True), asset_identity=assets,
                   execution=config.get("execution"))

    def validate_graph(self, graph, active):
        """Check the candidate's shared graph and return its native-input slot mask."""
        n = self.spec.max_nodes
        shapes = ((1, n, len(self.spec.atomic_numbers)), (1, n, len(self.spec.formal_charges)),
                  (1, n, n, len(self.spec.bond_names)))
        values = (graph.atoms, graph.charges, graph.bonds)
        if any(tuple(value.shape) != shape for value, shape in zip(values, shapes)):
            raise ValueError("Boltz search requires one full shared-schema graph")
        if any(value.dtype != torch.float32 or value.device != values[0].device for value in values):
            raise ValueError("Boltz search requires common-device FP32 probabilities")
        active = torch.as_tensor(active)
        if (active.dtype != torch.bool or active.shape != (n,)
                or int(active.sum()) != self._slots.numel()):
            raise ValueError("Boltz search slot mask must preserve the prepared atom count")
        mask = active.to(graph.atoms.device)
        if not torch.allclose(graph.occupancy[0], mask.to(graph.atoms), atol=1e-6, rtol=0):
            raise ValueError("Boltz search requires exact declared occupancy")
        for value in values:
            if (not bool(torch.isfinite(value).all()) or bool((value < 0).any())
                    or not torch.allclose(value.sum(-1), torch.ones_like(value[..., 0]), atol=2e-5, rtol=2e-5)):
                raise ValueError("Boltz search requires finite normalized nonnegative probabilities")
        excluded = ~(mask[:, None] & mask[None, :]) | torch.eye(n, dtype=torch.bool, device=mask.device)
        if (not torch.allclose(graph.bonds, graph.bonds.transpose(1, 2), atol=1e-6, rtol=0)
                or not torch.allclose(graph.bonds[0, ..., 0][excluded], torch.ones_like(graph.bonds[0, ..., 0][excluded]), atol=1e-6, rtol=0)):
            raise ValueError("Boltz search requires symmetric bonds and NONE on padding/diagonal")
        return mask

    def _pack(self, graph, active):
        mask = self.validate_graph(graph, active)
        n = self.spec.max_nodes
        source = mask.nonzero().flatten()
        base = self.prepared.seed.to_relaxed(self.spec)
        atoms = base.atoms.index_copy(1, self._slots, graph.atoms[:, source].to(self.device))
        charges = base.charges.index_copy(1, self._slots, graph.charges[:, source].to(self.device))
        pairs = (self._slots[:, None] * n + self._slots[None, :]).flatten()
        bonds = graph.bonds[:, source[:, None], source[None, :]].to(self.device).reshape(1, -1, len(self.spec.bond_names))
        bonds = base.bonds.reshape(1, n*n, -1).index_copy(1, pairs, bonds).reshape(1, n, n, -1)
        return RelaxedGraph(atoms, charges, bonds)

    def score_at(self, graph, active):
        started = perf_counter()
        packed = self._pack(graph, active)
        if (torch.backends.cuda.matmul.allow_tf32 != self.execution["tf32"]
                or torch.backends.cudnn.allow_tf32 != self.execution["tf32"]):
            raise ValueError("Process-global TF32 policy changed; concurrent Boltz branches must share execution settings")
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.execution["autocast"] == "bfloat16"):
            outputs = self.branch(packed)
        value = outputs.get("dbmol_total_loss")
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(value.isfinite().all()):
            raise ValueError("Require finite scalar full DBMol total")
        aux = {key: float(item.detach()) for key, item in outputs.items()
               if isinstance(item, torch.Tensor) and item.numel() == 1}
        if any(not math.isfinite(item) for item in aux.values()):
            raise ValueError("Nonfinite DBMol component")
        self.last_aux = aux
        self.last_native_seconds = perf_counter() - started
        self.last_batch_size = 1
        return value.reshape(()).to(graph.atoms.device), dict(aux)

    def probe_value_and_grad(self, graph, active, *, measure_memory=True):
        """Serial diagnostic of the production scalar and input derivatives.

        Returns detached CPU gradient tensors for comparisons between explicit
        execution variants. CUDA peak reset/synchronization is process-wide;
        use this probe with other model work idle, not inside concurrent search.
        Flat finite gradients are recorded without a nonzero requirement.
        """
        inputs = tuple(value.detach().clone().requires_grad_(True)
                       for value in (graph.atoms, graph.charges, graph.bonds))
        probe = RelaxedGraph(*inputs)
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            if measure_memory:
                torch.cuda.reset_peak_memory_stats(self.device)
        started = perf_counter()
        value, components = self.score_at(probe, active)
        if not value.requires_grad:
            raise ValueError("Required DBMol probe path is detached")
        if cuda:
            torch.cuda.synchronize(self.device)
        forward_end = perf_counter()
        gradients = torch.autograd.grad(value, inputs, allow_unused=True)
        if any(gradient is None or not bool(gradient.isfinite().all()) for gradient in gradients):
            raise ValueError("Missing/nonfinite DBMol probe input derivative")
        if cuda:
            torch.cuda.synchronize(self.device)
        finished = perf_counter()
        names = ("atoms", "charges", "bonds")
        return dict(loss=float(value.detach()), output_dtype=str(value.dtype), components=components,
            gradients={name: gradient.detach().cpu() for name, gradient in zip(names, gradients)},
            gradient_norms={name: float(gradient.detach().double().norm()) for name, gradient in zip(names, gradients)},
            input_dtypes={name: str(tensor.dtype) for name, tensor in zip(names, inputs)},
            gradient_dtypes={name: str(gradient.dtype) for name, gradient in zip(names, gradients)},
            forward_seconds=forward_end-started, backward_seconds=finished-forward_end,
            total_seconds=finished-started,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)) if cuda and measure_memory else None,
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)) if cuda and measure_memory else None,
            execution=dict(self.execution_record))

    def __call__(self, graph):
        return self.score_at(graph, self.active)

    def score_batch(self, graphs, active_masks):
        rows = list(zip(graphs, active_masks, strict=True))
        if not rows:
            raise ValueError("Require at least one Boltz graph")
        started = perf_counter()
        result = [self.score_at(graph, active) for graph, active in rows]
        self.last_native_seconds = perf_counter() - started
        self.last_batch_size = len(rows)
        return result

    @torch.no_grad()
    def score_valid_records(self, root, records, *, reject_unstable_identity=False):
        """Detached full DBMol scoring, preserving each saved pose-slot graph.

        Repeated SMILES are allowed when their packed graph keys differ. Exact
        duplicate keys within this call share one native evaluation. Results
        retain a scoring_key for callers' context-scoped caches and budgets.
        A search can count an exact graph-derived SMILES that changes when
        reparsed as an unsuccessful attempt, preserving the original graph.
        """
        from rdkit import Chem

        root = Path(root)
        output = root / "fresh"
        output.mkdir(exist_ok=False)
        cache, results = {}, []
        for record in records:
            if "hard_graph" not in record:
                raise ValueError("Fixed-pose DBMol scoring requires saved hard_graph labels, not SMILES alone")
            atoms, charges, bonds = _record_tensors(record["hard_graph"])
            n = self.spec.max_nodes
            if atoms.shape != (n,) or (charges >= len(self.spec.formal_charges)).any() or (atoms >= len(self.spec.atomic_numbers)).any() or (bonds >= len(self.spec.bond_names)).any():
                raise ValueError("Hard graph categories or capacity differ from the declared graph schema")
            checked = inspect_graph(atoms, charges, bonds, self.spec)
            mol = Chem.MolFromSmiles(record["smiles"])
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol is not None else None
            key = hard_pose_key(record["hard_graph"])
            if record.get("scoring_key", key) != key:
                raise ValueError("Hard graph scoring_key differs from the saved pose-slot labels")
            if checked.smiles is None or checked.smiles != canonical:
                if not (reject_unstable_identity and checked.smiles is not None
                        and checked.smiles == record["smiles"]):
                    raise ValueError("Saved hard graph does not encode the declared valid canonical molecule")
                cache[key] = dict(status="rejected_unstable_identity", loss=None,
                                  details=dict(reparsed_smiles=canonical))
                results.append(dict(record, scoring_key=key, **cache[key]))
                continue
            if key not in cache:
                graph = HardGraphBatch(atoms[None], charges[None], bonds[None]).to_relaxed(self.spec)
                loss, aux = self.score_at(graph, atoms != 0)
                cache[key] = dict(status="scored", loss=float(loss), details=aux,
                                  pose_policy="Same fixed reference pose; no conformer regeneration")
            results.append(dict(record, smiles=canonical, scoring_key=key, **cache[key]))
        payload = dict(status="completed", objective="dbmol_total_loss", candidates=results,
            unique_candidates=len(cache), unique_pose_graphs=len(cache),
            unique_molecules=len({row["smiles"] for row in results}), identity=self.identity,
            interpretation="Detached hard-label full DBMol at the same fixed pose; not fresh native pose evaluation")
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return results

    def close(self):
        from latentmol.scoring import require_frozen_eval

        require_frozen_eval(self.branch)
        for name, block, keep, expected in self._checkpoint_blocks:
            if block.forward is not expected:
                raise ValueError(f"Checkpoint execution policy changed after scorer construction: {name}")
        self.branch.verify_context_unchanged()
        if any(parameter.requires_grad or parameter.grad is not None for parameter in self.scorer.parameters()):
            raise ValueError("Frozen Boltz parameters acquired training state/gradients")
        if _module_digest(self.scorer) != self._model_digest:
            raise ValueError("Frozen Boltz model state changed")
        self.shutdown_status = dict(frozen_unchanged=True, context_digest=self.identity["context_digest"])
        return dict(self.shutdown_status)
