"""The experiment definition: one YAML file mapped onto the CLI's own options.

`experiment.yaml` is the primary way to define a Stage 2 run. Every knob lives
there, grouped by the protocol section it comes from, so an experiment is one
reviewable file rather than a remembered command line.

The YAML sets the parser's defaults, so an explicit command-line flag still
wins. That makes a sweep easy: keep one file and override the single thing that
varies.

:data:`SCHEMA` is the whole contract — nested YAML path to argparse destination.
Nothing is inferred, and an unknown key is an error rather than a silent no-op,
because a typo'd setting that quietly does nothing is how an experiment ends up
not being the experiment you thought you ran.
"""

from pathlib import Path

#: ``("yaml", "path"): "argparse_dest"``. The only mapping that exists.
SCHEMA = {
    ("backend",): "backend",
    ("boltz2", "context"): "boltz_context",
    ("boltz2", "checkpoint"): "boltz_checkpoint",
    ("boltz2", "python"): "boltz_python",
    ("boltz2", "seed_mixing"): "seed_mixing",
    ("boltz2", "allow_local_gpu"): "allow_local_gpu",
    ("protenix", "target"): "target",
    ("protenix", "cache_dir"): "cache_dir",
    ("trajectory", "steps"): "steps",
    ("trajectory", "eta_X"): "eta_X",
    ("trajectory", "eta_E"): "eta_E",
    ("trajectory", "readout_every"): "readout_every",
    ("trajectory", "seed"): "seed",
    ("trajectory", "parent"): "parent",
    ("operators", "mode"): "mode",
    ("operators", "aromatic_order"): "aromatic_order",
    ("operators", "bridge_probability"): "bridge_probability",
    ("operators", "sparsity_amount"): "sparsity_amount",
    ("operators", "sparsity_threshold"): "sparsity_threshold",
    ("improvements", "readout_caps"): "readout_caps",
    ("improvements", "valence_damping"): "valence_damping",
    ("improvements", "fluidity_threshold"): "fluidity_threshold",
    ("improvements", "fluidity_strength"): "fluidity_strength",
    ("improvements", "bridge_retry"): "bridge_retry",
    ("hardening", "enabled"): "hardening",
    ("hardening", "gradient"): "hardening_gradient",
    ("checks", "gradient_check"): "gradient_check",
}

#: Destinations argparse wants as ``Path`` rather than ``str``.
PATHS = frozenset(
    {
        "boltz_context",
        "boltz_checkpoint",
        "boltz_python",
        "cache_dir",
        "parent",
        "out",
    }
)

DEFAULT = Path(__file__).resolve().parent / "experiment.yaml"


def _walk(node, prefix=()):
    for key, value in node.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _walk(value, path)
        else:
            yield path, value


def load(path):
    """Read one experiment file and return ``{argparse_dest: value}``.

    Raises on an unknown key, listing what was expected at that level, so a
    misspelled setting fails loudly instead of being ignored.
    """
    import yaml

    document = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(document, dict):
        raise TypeError(f"{path} must be a mapping at the top level")
    values = {}
    for route, value in _walk(document):
        if route not in SCHEMA:
            known = sorted(
                ".".join(k) for k in SCHEMA if k[:-1] == route[:-1]
            ) or sorted(".".join(k) for k in SCHEMA)
            raise ValueError(
                f"{path}: unknown setting {'.'.join(route)!r}. "
                f"Expected one of: {', '.join(known)}"
            )
        dest = SCHEMA[route]
        values[dest] = Path(value) if dest in PATHS and value is not None else value
    return values


def apply(parser, argv=None):
    """Parse ``--config`` first, let it set defaults, then parse everything.

    Order of precedence: command-line flag, then the experiment file, then the
    parser's own default. Parsing twice is what buys that, cheaply.
    """
    import argparse

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        parser.set_defaults(**load(known.config))
    args = parser.parse_args(argv)
    args.config = known.config
    return args
