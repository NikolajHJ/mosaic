"""Frozen DeFoG inference adapter using pinned author code and checkpoint.

Training/evaluation frameworks are not imported. Checkpoint metadata objects are
restored as passive records; tensor reconstruction uses PyTorch. Inference
modules retain the upstream arithmetic, with imports isolated in a private
namespace. A strict state-dict load is required, without missing tensor keys.
"""

import ast
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

_METADATA = {}


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(
            (
                "omegaconf.",
                "datasets.",
                "analysis.visualization",
                "models.extra_features",
            )
        ):
            return _METADATA.setdefault((module, name), type(name, (), {}))
        return super().find_class(module, name)


def plain_metadata(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: plain_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_metadata(v) for v in value]
    if hasattr(value, "_content"):
        return plain_metadata(value._content)
    if hasattr(value, "_val"):
        return plain_metadata(value._val)
    return value


def upstream_modules(source):
    import torch

    root = "mosaic_defog_inference"
    if root in sys.modules:
        return sys.modules[root]
    for name in (root, root + ".models", root + ".flow_matching"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, package)

    def register(relative, code):
        name = root + "." + relative
        module = types.ModuleType(name)
        module.__package__ = name.rsplit(".", 1)[0]
        module.__dict__["torch"] = torch
        sys.modules[name] = module
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
        # Execute only the pinned, hashed author inference modules inspected here.
        exec(  # noqa: S102
            compile(code, str(source / (relative.replace(".", "/") + ".py")), "exec"),
            module.__dict__,
        )
        return module

    utils_source = (source / "utils.py").read_text()
    tree = ast.parse(utils_source)
    holder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlaceHolder"
    )
    register("utils", ast.get_source_segment(utils_source, holder))
    for name in (
        "flow_matching.flow_matching_utils",
        "flow_matching.utils",
        "flow_matching.noise_distribution",
        "models.layers",
        "models.transformer_model",
        "models.extra_features",
        "models.extra_features_molecular",
        "flow_matching.rate_matrix",
        "flow_matching.time_distorter",
    ):
        code = (source / (name.replace(".", "/") + ".py")).read_text()
        code = code.replace("from src import utils", f"from {root} import utils")
        code = code.replace(
            "from utils import PlaceHolder", f"from {root}.utils import PlaceHolder"
        )
        code = code.replace(
            "from flow_matching import flow_matching_utils",
            f"from {root}.flow_matching import flow_matching_utils",
        )
        code = code.replace("from flow_matching.", f"from {root}.flow_matching.")
        code = code.replace(
            "from models.layers import", f"from {root}.models.layers import"
        )
        register(name, code)
    return sys.modules[root]


class FrozenDeFoG:
    def __init__(self, checkpoint, source, *, threads=3):
        import torch

        self.torch = torch
        torch.set_num_threads(threads)
        self.source = Path(source)
        manifest = json.loads(
            Path(__file__).with_name("prior_manifest.json").read_text()
        )
        for name, expected in manifest["source_file_sha256"].items():
            assert (
                hashlib.sha256((self.source / name).read_bytes()).hexdigest()
                == expected
            ), name
        checkpoint_digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        assert checkpoint_digest == manifest["checkpoint_sha256"], (
            "Prior checkpoint identity mismatch"
        )
        modules = upstream_modules(self.source)
        loader = types.SimpleNamespace(
            __name__="mosaic_passive_metadata",
            Unpickler=MetadataUnpickler,
            load=pickle.load,
        )
        saved = torch.load(
            checkpoint, map_location="cpu", weights_only=False, pickle_module=loader
        )
        hp = saved["hyper_parameters"]
        self.config = plain_metadata(hp["cfg"])
        self.info = hp["dataset_infos"]
        cfg = self.config["model"]
        assert cfg["transition"] == "marginal", cfg["transition"]
        self.model = modules.models.transformer_model.GraphTransformer(
            n_layers=cfg["n_layers"],
            input_dims=self.info.input_dims,
            hidden_mlp_dims=cfg["hidden_mlp_dims"],
            hidden_dims=cfg["hidden_dims"],
            output_dims=self.info.output_dims,
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU(),
        )
        weights = {
            k[len("model.") :]: v
            for k, v in saved["state_dict"].items()
            if k.startswith("model.")
        }
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().requires_grad_(False)
        self.extra = modules.models.extra_features.ExtraFeatures(
            cfg["extra_features"], cfg["rrwp_steps"], self.info
        )
        self.molecular = modules.models.extra_features_molecular.ExtraMolecularFeatures(
            self.info
        )
        self.x_limit = self.info.node_types.float() / self.info.node_types.sum()
        self.e_limit = self.info.edge_types.float() / self.info.edge_types.sum()
        self.allowed = [
            self.info.atom_decoder.index(a) for a in ("C", "N", "O", "S", "F", "Cl")
        ]
        assert self.info.output_dims["E"] == 5
        assert len(self.x_limit) == self.info.output_dims["X"]
        self.identities = {
            "model": "DeFoG MOSES",
            "source_revision": "365bda9affadd5c2307014a0532ddaa244399441",
            "checkpoint_sha256": checkpoint_digest,
            "input_dims": self.info.input_dims,
            "output_dims": self.info.output_dims,
            "atom_decoder": self.info.atom_decoder,
            "allowed_indices": self.allowed,
            "max_n_nodes": int(self.info.max_n_nodes),
            "torch": torch.__version__,
            "device": "cpu",
            "state_dict": "strict: all model tensors matched",
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "source_file_sha256": {
                str(p.relative_to(self.source)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in self.source.rglob("*.py")
            },
        }

    def sampling_context(self):
        """Bind the author's inference-only methods without its Lightning class."""
        if hasattr(self, "_sampling_context"):
            return self._sampling_context
        torch = self.torch
        modules = sys.modules["mosaic_defog_inference"]

        def namespace(value):
            if isinstance(value, dict):
                return types.SimpleNamespace(
                    **{k: namespace(v) for k, v in value.items()}
                )
            return value

        cfg = namespace(self.config)
        noise = modules.flow_matching.noise_distribution.NoiseDistribution(
            cfg.model.transition, self.info
        )
        limit = noise.get_limit_dist()
        rate = modules.flow_matching.rate_matrix.RateMatrixDesigner(
            cfg.sample.rdb, cfg.sample.rdb_crit, cfg.sample.eta, cfg.sample.omega, limit
        )
        distortion = modules.flow_matching.time_distorter.TimeDistorter(
            cfg.train.time_distortion, cfg.sample.time_distortion
        )
        context = types.SimpleNamespace(
            cfg=cfg,
            model=self.model,
            extra_features=self.extra,
            domain_features=self.molecular,
            noise_dist=noise,
            limit_dist=limit,
            rate_matrix_designer=rate,
            time_distorter=distortion,
            conditional=False,
            device=torch.device("cpu"),
        )
        source = (self.source / "graph_discrete_flow_model.py").read_text()
        tree = ast.parse(source)
        model = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GraphDiscreteFlowModel"
        )
        for name in (
            "forward",
            "compute_extra_data",
            "compute_step_probs",
            "sample_p_zs_given_zt",
        ):
            node = next(
                node
                for node in model.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            scope = {
                "torch": torch,
                "F": torch.nn.functional,
                "utils": modules.utils,
                "flow_matching_utils": modules.flow_matching.flow_matching_utils,
            }
            exec(  # noqa: S102
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    str(self.source / "graph_discrete_flow_model.py"),
                    "exec",
                ),
                scope,
            )
            setattr(context, name, types.MethodType(scope[name], context))
        self._sampling_context = context
        return context

    def sample(self, *, count=32, n_atoms=25, steps=50, seed=101):
        """One standalone author-style generation control, not per-AF3 sampling."""
        torch = self.torch
        context = self.sampling_context()
        utilities = sys.modules[
            "mosaic_defog_inference"
        ].flow_matching.flow_matching_utils
        node_mask = torch.ones((count, n_atoms), dtype=torch.bool)
        chain_x, chain_e = [], []
        i, j = np.triu_indices(n_atoms, 1)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(seed)
            graph = utilities.sample_discrete_feature_noise(
                context.limit_dist, node_mask
            )
            chain_x.append(graph.X.argmax(-1).numpy())
            chain_e.append(graph.E.argmax(-1)[:, i, j].numpy())
            for step in range(steps):
                t = torch.full((count, 1), step / steps)
                s = torch.full((count, 1), (step + 1) / steps)
                t = context.time_distorter.sample_ft(
                    t, context.cfg.sample.time_distortion
                )
                s = context.time_distorter.sample_ft(
                    s, context.cfg.sample.time_distortion
                )
                graph, _ = context.sample_p_zs_given_zt(
                    t, s, graph.X, graph.E, graph.y, node_mask
                )
                chain_x.append(graph.X.argmax(-1).numpy())
                chain_e.append(graph.E.argmax(-1)[:, i, j].numpy())
        return np.stack(chain_x), np.stack(chain_e)

    def predict_noisy(self, x, e, t):
        """One author-network prediction on categorical noisy inputs."""
        torch = self.torch
        x, e = (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(e, dtype=torch.float32),
        )
        if x.ndim == 2:
            x, e = x[None], e[None]
        n = x.shape[1]
        node_mask = torch.ones(x.shape[:2], dtype=torch.bool)
        noisy = {
            "X_t": x,
            "E_t": e,
            "y_t": torch.empty((len(x), 0)),
            "t": torch.full((len(x), 1), float(t)),
            "node_mask": node_mask,
        }
        with torch.inference_mode():
            extra, molecular = self.extra(noisy), self.molecular(noisy)
            xx = torch.cat((x, extra.X, molecular.X), dim=-1)
            ee = torch.cat((e, extra.E, molecular.E), dim=-1)
            yy = torch.cat((noisy["y_t"], extra.y, molecular.y, noisy["t"]), dim=-1)
            assert n <= self.info.max_n_nodes
            output = self.model(xx, ee, yy, node_mask)
            px, pe = output.X.softmax(-1), output.E.softmax(-1)
        if not torch.isfinite(px).all() or not torch.isfinite(pe).all():
            raise ArithmeticError("Nonfinite frozen-prior prediction")
        return px.numpy(), pe.numpy()

    def teacher(self, atoms, edges, *, step, seed=71, time=0.5, batch_size=1):
        """Sample a training-compatible noisy categorical graph, then denoise once.

        DeFoG uses t=0 as noise and t=1 as clean. Its categorical probabilities
        are t*current_graph+(1-t)*training_marginal. Upper edges are sampled
        once and mirrored, and diagonal input edges are NO_BOND as in author training.
        Returned predictions are conditioned onto the experiment's six elements;
        excluded Br/H mass is reported, never silently relabeled.
        """
        torch = self.torch
        n = len(atoms)
        rng = torch.Generator().manual_seed(int(seed) + 1000003 * int(step))
        x = torch.zeros((n, len(self.x_limit)))
        x[:, self.allowed] = torch.as_tensor(np.asarray(atoms).copy())
        e = torch.as_tensor(np.asarray(edges).copy())
        px = time * x + (1 - time) * self.x_limit
        pe = time * e + (1 - time) * self.e_limit
        ix = torch.multinomial(px, batch_size, replacement=True, generator=rng).T
        ie = torch.multinomial(pe, batch_size, replacement=True, generator=rng).T
        noisy_x = torch.nn.functional.one_hot(ix, len(self.x_limit)).float()
        noisy_e = torch.zeros((batch_size, n, n, 5))
        i, j = np.triu_indices(n, 1)
        upper = torch.nn.functional.one_hot(ie, 5).float()
        noisy_e[:, i, j] = upper
        noisy_e[:, j, i] = upper
        noisy_e[:, np.arange(n), np.arange(n), 0] = 1.0
        out_x, out_e = self.predict_noisy(noisy_x, noisy_e, time)
        allowed = out_x[..., self.allowed]
        mass = allowed.sum(-1, keepdims=True)
        if mass.min() < 1e-12:
            raise ArithmeticError("Prior predicts effectively no supported atom mass")
        teacher_x = (allowed / mass).mean(0)
        teacher_e = out_e[:, i, j].mean(0)
        excluded_indices = [k for k in range(out_x.shape[-1]) if k not in self.allowed]
        excluded = out_x[..., excluded_indices].sum(-1)
        return (
            teacher_x,
            teacher_e,
            {
                "excluded_atom_mass_mean": float(excluded.mean()),
                "excluded_atom_mass_max": float(excluded.max()),
                "time": time,
                "noise_seed": int(seed) + 1000003 * int(step),
                "batch_size": batch_size,
                "diagonal_policy": "onehot NO_BOND, matching upstream apply_noise training inputs",
                "noisy_X": noisy_x.numpy(),
                "noisy_E_upper": upper.numpy(),
            },
        )
