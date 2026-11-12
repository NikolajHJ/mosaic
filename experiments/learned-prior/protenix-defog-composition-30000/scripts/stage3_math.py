"""Small, explicit interventions for the Stage 3 factorial campaign.

These are experiment policies, not new defaults in Mosaic. Graphs always use
the public six-element/five-bond schema and lexicographic unordered pairs.
"""

import math
from dataclasses import fields, replace

import numpy as np


def schedule_value(spec, update, updates=500):
    """Evaluate a zero-based update schedule, including linear endpoints.

    The last gradient update of a 500-update run is index 499. Diagnostic
    evaluations at index 500 retain the final linear value. A cosine is not
    clamped: its declared period is measured in update intervals.
    """
    if isinstance(update, bool) or not isinstance(update, (int, np.integer)) or update < 0:
        raise ValueError("update must be a nonnegative integer")
    if isinstance(spec, (float, int)) and not isinstance(spec, bool):
        value = float(spec)
    elif isinstance(spec, dict):
        kind = spec.get("kind", "constant")
        if kind == "constant":
            value = float(spec["value"])
        elif kind == "linear":
            begin, finish = spec.get("begin", 0), spec.get("finish", updates - 1)
            if (isinstance(begin, bool) or isinstance(finish, bool)
                    or not isinstance(begin, int) or not isinstance(finish, int)
                    or begin < 0 or finish <= begin):
                raise ValueError("linear begin/finish must be increasing nonnegative integers")
            fraction = min(1., max(0., (int(update) - begin) / (finish - begin)))
            value = float(spec["start"]) + fraction * (float(spec["end"]) - float(spec["start"]))
        elif kind == "cosine":
            period = spec["period"]
            if isinstance(period, bool) or not isinstance(period, int) or period < 2:
                raise ValueError("cosine period must be an integer >= 2")
            low, high = float(spec["low"]), float(spec["high"])
            if high < low:
                raise ValueError("cosine high must be >= low")
            value = low + (high - low) * (1 - math.cos(math.tau * int(update) / period)) / 2
        else:
            raise ValueError(f"unknown schedule kind {kind!r}")
    else:
        raise TypeError("schedule must be a number or a dictionary")
    if not math.isfinite(value):
        raise ValueError("schedule value must be finite")
    return value


def _power_rows(rows, exponent):
    """Zero-preserving power normalization with finite derivatives at zero.

    Scaling by the row maximum avoids all-positive entries underflowing for
    low temperatures. A zero entry uses a constant safe base before the power;
    the selected zero branch has zero derivative for temperatures below one.
    """
    import jax.numpy as jnp

    scale = jnp.max(rows, axis=-1, keepdims=True)
    scaled = rows / jnp.maximum(scale, jnp.finfo(rows.dtype).tiny)
    positive = scaled > 0
    powers = jnp.where(positive, jnp.power(jnp.where(positive, scaled, 1.), exponent), 0.)
    return powers / jnp.maximum(jnp.sum(powers, axis=-1, keepdims=True), jnp.finfo(rows.dtype).tiny)


def cool_graph(graph, temperature, gradient="exact"):
    """Return C_T(p)=p**(1/T)/sum(p**(1/T)), on the same graph schema.

    T=1 is the identity in both forward and reverse AD, including at zero.
    ``ste`` keeps the exact same cooled forward values and passes gradients
    through unchanged. No probability floor is introduced by either option.
    """
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.core.state import RelaxedGraph

    # The runner validates scheduled values on the host. Retain direct-call
    # validation while accepting a traced scalar to reuse the native compile.
    traced_temperature = isinstance(temperature, jax.core.Tracer)
    if not traced_temperature:
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
    if gradient not in ("exact", "ste"):
        raise ValueError("cooling gradient must be 'exact' or 'ste'")
    def transform(current):
        x = _power_rows(current.X, 1 / temperature)
        e = _power_rows(current.E_upper, 1 / temperature)
        if gradient == "ste":
            x = current.X + jax.lax.stop_gradient(x - current.X)
            e = current.E_upper + jax.lax.stop_gradient(e - current.E_upper)
        return RelaxedGraph(x, e)

    # Normalizing already-normalized probabilities at T=1 has the right
    # forward values but changes derivatives in ambient probability space.
    # This identity branch preserves both exactly.
    if not traced_temperature:
        return graph if temperature == 1 else transform(graph)
    return jax.lax.cond(jnp.asarray(temperature) == 1, lambda current: current, transform, graph)


def cool_state_graph(graph, previous_temperature, temperature):
    """Apply only the incremental stored-state cooling, then project outside.

    The exponent is T_old/T_new. Applying the full absolute temperature at
    every update would compound cooling even when the schedule stays fixed.
    """
    previous_temperature, temperature = float(previous_temperature), float(temperature)
    if not math.isfinite(previous_temperature) or previous_temperature <= 0:
        raise ValueError("previous_temperature must be finite and positive")
    return cool_graph(graph, temperature / previous_temperature)


def make_initial_graph(seed, initialization, n_atoms=25):
    """Make matched starts: same atom probabilities, different bond density.

    Sparse starts are random single-bond trees within the selected neutral
    capacities. Atom categories are sampled carbon-heavy with nonterminal
    atoms added before halogens; any missing branching capacity is supplied
    by replacing a terminal category with carbon. Fragmented starts delete
    four randomly selected tree edges. Dense starts independently randomize
    all bond-category rows and intentionally need not be chemically valid.
    Atom rows mix 0.02 uniform mass; bond rows mix 0.002 uniform mass.
    """
    import jax.numpy as jnp
    from mosaic.small_molecule.core.state import RelaxedGraph, validate_graph

    aliases = {"sparse-random": "sparse", "dense-random": "dense",
               "sparse-disconnected": "fragmented"}
    initialization = aliases.get(initialization, initialization)
    if initialization not in ("sparse", "dense", "fragmented"):
        raise ValueError("initialization must be sparse, dense or fragmented")
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, int) or n_atoms < 1:
        raise ValueError("n_atoms must be a positive integer")
    rng = np.random.default_rng(seed)
    elements = rng.choice(6, n_atoms, p=[.70, .14, .08, .04, .02, .02])
    caps = np.array([4, 3, 2, 2, 1, 1])[elements]
    if n_atoms > 1:
        if not np.any(caps >= 2):
            elements[0], caps[0] = 0, 4
        while caps.sum() < 2 * (n_atoms - 1):
            index = int(np.argmin(caps))
            elements[index], caps[index] = 0, 4
    nonterminal = rng.permutation(np.flatnonzero(caps >= 2)).tolist()
    terminal = rng.permutation(np.flatnonzero(caps == 1)).tolist()
    order = nonterminal + terminal
    i, j = np.triu_indices(n_atoms, 1)
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(i, j))}
    bond_categories = np.zeros(len(i), np.int32)
    degree = np.zeros(n_atoms, int)
    tree_rows = []
    for count, atom in enumerate(order[1:], 1):
        possible = [prior for prior in order[:count] if degree[prior] < caps[prior]]
        if not possible:
            raise RuntimeError("initial tree exhausted neutral capacity")
        parent = int(rng.choice(possible))
        pair = tuple(sorted((int(atom), parent)))
        row = lookup[pair]
        bond_categories[row] = 1
        tree_rows.append(row)
        degree[[atom, parent]] += 1
    # Derive both secondary starts from the same RNG history. Selecting one
    # family never changes the atom composition or underlying sparse tree.
    deleted = np.asarray(rng.permutation(tree_rows)[:min(4, len(tree_rows))], dtype=np.int32)
    dense_edges = rng.dirichlet(np.ones(5), size=len(i))
    if initialization == "fragmented":
        bond_categories[deleted] = 0
    x = .98 * np.eye(6)[elements] + .02 / 6
    e = (dense_edges if initialization == "dense"
         else .998 * np.eye(5)[bond_categories] + .002 / 5)
    graph = RelaxedGraph(jnp.asarray(x, dtype=jnp.float32), jnp.asarray(e, dtype=jnp.float32))
    validate_graph(graph)
    return graph


def teacher_distance(graph, target, kind="mse", atom_scale=1., bond_scale=1., normalization="row"):
    """Scalar discrepancy to a stopped target; default is mean squared L2/row.

    KL is KL(target||student) with a numerical 1e-8 log floor; zero target
    entries contribute zero. Entry normalization divides each head by its
    category count as well as the number of rows. Empty bond heads yield zero.
    """
    import jax
    import jax.numpy as jnp

    if kind not in ("mse", "kl") or normalization not in ("row", "entry"):
        raise ValueError("unsupported teacher loss or normalization")
    for scale in (atom_scale, bond_scale):
        if not math.isfinite(float(scale)) or scale < 0:
            raise ValueError("teacher head scales must be finite and nonnegative")

    def head(student, reference):
        reference = jax.lax.stop_gradient(jnp.asarray(reference, dtype=student.dtype))
        if reference.shape != student.shape:
            raise ValueError("teacher target shape must match the student graph")
        if not student.shape[0]:
            return jnp.asarray(0., student.dtype)
        if kind == "mse":
            entries = jnp.square(student - reference)
        else:
            entries = reference * (jnp.log(jnp.maximum(reference, 1e-8)) - jnp.log(jnp.maximum(student, 1e-8)))
        return jnp.mean(entries) if normalization == "entry" else jnp.mean(jnp.sum(entries, axis=-1))

    return atom_scale * head(graph.X, target.X) + bond_scale * head(graph.E_upper, target.E_upper)


def simple_soft_valence_loss(graph):
    """New simple comparison: mean squared positive expected-valence excess.

    This is the Stage 2 diagnostic written differentiably, not an assertion
    that it reproduces an archived custom chemistry objective. Under-valence,
    connectivity, valence flexibility and atom identity have no extra terms.
    """
    import jax.numpy as jnp

    n = graph.X.shape[0]
    i, j = jnp.triu_indices(n, 1)
    order = graph.E_upper @ jnp.array([0., 1., 2., 3., 1.5], graph.E_upper.dtype)
    used = jnp.zeros(n, graph.X.dtype).at[i].add(order).at[j].add(order)
    cap = graph.X @ jnp.array([4., 3., 2., 2., 1., 1.], graph.X.dtype)
    return jnp.mean(jnp.square(jnp.maximum(used - cap, 0.)))


def apply_projection(graph, config, update):
    """Apply scheduled Stage 2 primitives and record operator-level movement.

    ``bridge_only`` and ``final_valence=False`` are prospective diagnostic
    ablations. They deliberately do not imply feasibility. Normal modes use
    the published Stage 2 primitives and their order, including bounded retry.
    """
    import jax.numpy as jnp
    from mosaic.small_molecule.core.state import RelaxedGraph
    from mosaic.small_molecule.Stage2 import operators as op

    config = dict(config)
    mode = config.get("mode", "connected")
    allowed_modes = (*op.MODES, "bridge_only")
    if mode not in allowed_modes:
        raise ValueError(f"unsupported projection mode {mode!r}")
    extra_fields = {"every", "start_update", "final_valence"}
    canonical_fields = {field.name for field in fields(op.Stage2Config)}
    unknown = config.keys() - canonical_fields - extra_fields
    if unknown:
        raise ValueError(f"unknown projection settings: {sorted(unknown)}")
    every, start = config.get("every", 1), config.get("start_update", 1)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (update, every, start)) or every < 1 or start < 1 or update < 0:
        raise ValueError("projection needs integer update>=0, every>=1 and start_update>=1")
    final_valence = config.get("final_valence", True)
    if not isinstance(final_valence, bool):
        raise TypeError("final_valence must be boolean")
    canonical = {key: value for key, value in config.items() if key in canonical_fields}
    canonical["mode"] = "connected" if mode == "bridge_only" else mode
    settings = op.Stage2Config(**canonical)
    x, edges = np.asarray(graph.X).copy(), np.asarray(graph.E_upper).copy()
    n = op.check_invariants(x, edges)
    original_x, original_edges = x.copy(), edges.copy()
    count = lambda e: int(len(np.unique(op.components(n, e))))
    receipt = {"update": int(update), "mode": mode, "applied": False,
               "components_before": count(edges), "steps": []}

    def record(name, before_x, before_e, details):
        receipt["steps"].append({"operator": name,
            "atom_displacement_l2": float(np.linalg.norm(x - before_x)),
            "bond_displacement_l2": float(np.linalg.norm(edges - before_e)),
            "components_after": count(edges), **details})

    def valence(name):
        nonlocal edges
        before = edges.copy()
        edges, note = op.project_valence(x, edges, config=settings)
        record(name, x, before, note)

    selected_bridges = []
    if update >= start and (update - start) % every == 0 and mode != "none":
        receipt["applied"] = True
        if settings.fluidity_strength > 0 and settings.fluidity_threshold < 1:
            before_x, before_e = x.copy(), edges.copy()
            x, x_note = op.apply_fluidity_floor(x, threshold=settings.fluidity_threshold, strength=settings.fluidity_strength)
            edges, e_note = op.apply_fluidity_floor(edges, threshold=settings.fluidity_threshold, strength=settings.fluidity_strength)
            record("fluidity", before_x, before_e, {"atom_rows_relaxed": x_note["rows_relaxed"], "bond_rows_relaxed": e_note["rows_relaxed"]})
        if mode != "bridge_only":
            valence("initial_valence")
        receipt["components_after_initial_valence"] = count(edges)
        if mode in ("bridge_only", "connected", "sparse"):
            before = edges.copy()
            edges, note = op.connect_components(x, edges, config=settings)
            selected_bridges.extend(note["bridges"])
            record("bridge", x, before, note)
            receipt["components_after_bridge"] = count(edges)
            if mode != "bridge_only" and final_valence:
                valence("post_bridge_valence")
            receipt["components_after_post_bridge_valence"] = count(edges)
            survived = _bridge_survival(n, edges, note["bridges"])
            receipt["bridges_proposed_initial"] = len(note["bridges"])
            receipt["bridges_surviving_post_valence"] = survived
            if settings.bridge_retry and survived < len(note["bridges"]):
                stronger = replace(settings, bridge_probability=min(1., settings.bridge_probability + .3))
                candidate, retry = op.connect_components(x, edges, config=stronger)
                if mode != "bridge_only" and final_valence:
                    candidate, retry_valence = op.project_valence(x, candidate, config=settings)
                    retry["valence_after"] = retry_valence
                retry_survived = _bridge_survival(n, candidate, retry["bridges"])
                kept = retry_survived > survived
                before = edges.copy()
                if kept:
                    edges = candidate
                    selected_bridges.extend(retry["bridges"])
                record("bridge_retry", x, before, retry | {"kept": kept, "bridges_surviving": retry_survived})
        if mode in ("sparse", "sparse_only"):
            protect = np.zeros(len(edges), dtype=bool)
            for a, b in selected_bridges:
                protect[_pair_row(n, a, b)] = True
            before = edges.copy()
            edges, note = op.shrink_weak_edges(edges, config=settings, protect=protect)
            record("shrink", x, before, note)
            if final_valence:
                valence("post_shrink_valence")
    violations, excess = op.argmax_valence_violations(x, edges, aromatic_order=settings.aromatic_order)
    selected_bridges = sorted({tuple(sorted(pair)) for pair in selected_bridges})
    receipt.update({"components_after": count(edges), "argmax_valence_violations": violations,
                    "argmax_valence_max_excess": excess,
                    "bridges_proposed": len(selected_bridges),
                    "bridges_surviving_final": _bridge_survival(n, edges, selected_bridges),
                    "atom_displacement_l2": float(np.linalg.norm(x - original_x)),
                    "bond_displacement_l2": float(np.linalg.norm(edges - original_edges))})
    op.check_invariants(x, edges)
    return RelaxedGraph(jnp.asarray(x, dtype=graph.X.dtype), jnp.asarray(edges, dtype=graph.E_upper.dtype)), receipt


def _pair_row(n, a, b):
    a, b = sorted((a, b))
    return int(a * (2 * n - a - 1) // 2 + b - a - 1)


def _bridge_survival(n, edges, bridges):
    return int(sum(np.argmax(edges[_pair_row(n, a, b)]) != 0 for a, b in bridges))
