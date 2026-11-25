"""Choose a bounded chemistry-valid time panel without using scorer values."""


def select_valid_candidates(rows, *, edit_limit):
    """Retain early and late valid edits, then fill from spaced time targets."""
    if type(edit_limit) is not int or edit_limit < 1:
        raise ValueError("Require a positive integer edit limit")
    if not rows or any(right["step"] <= left["step"] for left, right in zip(rows, rows[1:])):
        raise ValueError("Candidate observations must have strictly increasing steps")
    valid = [row for row in rows if row["valid"]]
    if any(not isinstance(row["smiles"], str) or not row["smiles"] for row in valid):
        raise ValueError("Valid observations require a molecule identity")

    start, endpoint = rows[0], rows[-1]
    baseline = start["smiles"] if start["valid"] else None
    edits = [row for row in valid if row["step"] != start["step"] and row["smiles"] != baseline]
    by_identity = {}
    for row in edits:
        by_identity.setdefault(row["smiles"], row)

    selected = {}
    if baseline is not None:
        selected[baseline] = (start, "initial")
    if edits:
        first, latest = edits[0], edits[-1]
        selected[first["smiles"]] = (first, "first_distinct_valid")
        if edit_limit > 1 and latest["smiles"] != first["smiles"]:
            role = "endpoint" if latest["step"] == endpoint["step"] else "last_distinct_valid"
            selected[latest["smiles"]] = (latest, role)
    remaining = {identity: row for identity, row in by_identity.items() if identity not in selected}
    slots = min(len(remaining), edit_limit - len(selected) + int(baseline is not None))
    for index in range(1, slots + 1):
        target = start["step"] + index * (endpoint["step"] - start["step"]) / (slots + 1)
        identity = min(remaining, key=lambda key: (abs(remaining[key]["step"] - target),
                                                  remaining[key]["step"], key))
        selected[identity] = (remaining.pop(identity), f"valid_intermediate_{index}")
    candidates = [dict(role=role, step=row["step"], smiles=row["smiles"])
                  for row, role in sorted(selected.values(), key=lambda pair: pair[0]["step"])]
    selected_edits = len(candidates) - int(baseline is not None)
    return dict(
        candidates=candidates, observations=len(rows), valid_observations=len(valid),
        distinct_valid_molecules=len(by_identity) + int(baseline is not None),
        distinct_valid_edits=len(by_identity), selected_valid_edits=selected_edits,
        valid_edit_coverage=selected_edits / len(by_identity) if by_identity else None,
        edit_limit=edit_limit,
    )
