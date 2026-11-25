"""Learn a direct graph displacement and randomize its direction at the same size."""

import hashlib

import numpy as np


N = 25
I, J = np.triu_indices(N, k=1)
SLOTS = np.eye(N)
ELEMENT_I, ELEMENT_J = np.triu_indices(6)


def _validate(graph):
    atom, bond = graph
    for rows, shape in ((atom, (N, 6)), (bond, (len(I), 5))):
        if rows.shape != shape or rows.dtype != np.float32:
            raise ValueError(f"Expected FP32 graph with categorical rows {shape}")
        if not (np.isfinite(rows).all() and (rows >= 0).all()
                and np.allclose(rows.sum(axis=1), 1, atol=2e-6, rtol=0)):
            raise ValueError("Graph requires finite probability-simplex rows")


def _start_key(graph):
    # A canonical molecule identity check across splits also belongs in the driver.
    labels = b"".join(rows.argmax(axis=1).astype(np.uint8).tobytes() for rows in graph)
    return hashlib.sha256(labels).hexdigest()


def _features(graph):
    atom, bond = (rows.astype(np.float64) for rows in graph)
    adjacency = np.zeros((N, N))
    adjacency[I, J] = adjacency[J, I] = 1.0 - bond[:, 0]
    degree = adjacency.sum(axis=1, keepdims=True)
    neighborhood = adjacency @ atom
    atom_features = np.column_stack((np.ones(N), atom, degree, neighborhood,
                                     degree * atom, SLOTS))

    # Both endpoint permutations contribute to the same unordered element pair.
    pair = atom[I, :, None] * atom[J, None, :]
    pair = (pair + pair.swapaxes(1, 2))[:, ELEMENT_I, ELEMENT_J]
    present = 1.0 - bond[:, :1]
    endpoints = np.column_stack((pair, degree[I] + degree[J], np.abs(degree[I] - degree[J]),
                                 (adjacency @ adjacency)[I, J], SLOTS[I] + SLOTS[J]))
    bond_features = np.column_stack((np.ones(len(I)), bond, endpoints, present * endpoints))
    return atom_features, bond_features


def _simplex(rows):
    ordered = np.sort(rows, axis=1)[:, ::-1]
    offset = (ordered.cumsum(axis=1) - 1.0) / np.arange(1, rows.shape[1] + 1)
    count = (ordered > offset).sum(axis=1)
    threshold = offset[np.arange(len(rows)), count - 1]
    return np.maximum(rows - threshold[:, None], 0.0)


def fit(pairs):
    """Fit regularized per-row residuals to the supplied training trajectories."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("At least one independent training trajectory is required")
    training_starts = set()
    features, targets = [[], []], [[], []]
    for start, end in pairs:
        _validate(start)
        _validate(end)
        key = _start_key(start)
        if key in training_starts:
            raise ValueError("Training repeats an ordered hard-graph founder")
        training_starts.add(key)
        for head, design in enumerate(_features(start)):
            features[head].append(design)
            targets[head].append(end[head].astype(np.float64) - start[head])

    heads = []
    for designs, residuals in zip(features, targets):
        design = np.concatenate(designs)
        target = np.concatenate(residuals)
        # Penalize all coefficients. The one-hot columns have redundant directions.
        weights = np.linalg.solve(design.T @ design + np.eye(design.shape[1]), design.T @ target)
        fitted_norm = np.linalg.norm(design @ weights)
        scale = float(np.linalg.norm(target) / fitted_norm) if fitted_norm > 1e-12 else 1.0
        radius = max(float(np.linalg.norm(delta)) for delta in residuals)
        if not (np.isfinite(weights).all() and np.isfinite(scale) and np.isfinite(radius)):
            raise FloatingPointError("Nonfinite student parameters")
        heads.append(dict(weights=weights, scale=scale, radius=radius))
    return dict(heads=heads, training_starts=sorted(training_starts))


def predict(model, start):
    """Return a feasible held-out proposal; all calibration uses training pairs."""
    _validate(start)
    if _start_key(start) in model["training_starts"]:
        raise ValueError("Proposal evaluation overlaps the training founder panel")
    result = []
    for rows, design, head in zip(start, _features(start), model["heads"], strict=True):
        proposal = _simplex(rows + (design @ head["weights"]) * head["scale"])
        delta = proposal - rows
        norm = np.linalg.norm(delta)
        fraction = min(1.0, head["radius"] / norm) if norm > 0 else 1.0
        # Convex interpolation preserves feasibility after capping the head's displacement.
        result.append(((1.0 - fraction) * rows.astype(np.float64) + fraction * proposal).astype(np.float32))
    graph = tuple(result)
    _validate(graph)
    return graph


def matched_random(start, learned_graph, rng):
    """Scramble the learned feasible proposal, preserving both realized L2 norms."""
    _validate(start)
    _validate(learned_graph)
    result = []
    for base, proposal in zip(start, learned_graph):
        sorted_base = np.sort(base, axis=1)
        if not (np.all(sorted_base[:, :-1] == sorted_base[0, 0])
                and np.all(sorted_base[:, -1] == sorted_base[0, -1])):
            raise ValueError("Random control needs equally softened starting rows within each head")
        shuffled = rng.permutation(len(base))
        # Align the original winning categories, then assign losers to random categories.
        source_order = np.argsort(-base[shuffled], axis=1, kind="stable")
        random_keys = rng.random(base.shape)
        random_keys[np.arange(len(base)), base.argmax(axis=1)] = -1.0
        destination_order = np.argsort(random_keys, axis=1)
        randomized = np.empty_like(proposal)
        np.put_along_axis(randomized, destination_order,
                          np.take_along_axis(proposal[shuffled], source_order, axis=1), axis=1)
        expected = np.linalg.norm(proposal.astype(np.float64) - base)
        actual = np.linalg.norm(randomized.astype(np.float64) - base)
        if not np.isclose(expected, actual, rtol=1e-12, atol=1e-14):
            raise ValueError("Random proposal failed to match the realized learned displacement")
        result.append(randomized)
    graph = tuple(result)
    _validate(graph)
    return graph
