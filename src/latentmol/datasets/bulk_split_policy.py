"""Deterministic heavy-atom balancing and exact-identity holdout selection."""

import hashlib


DEFAULT_SEED = 550017


def _validate_capacities(capacities):
    if not capacities or any(type(size) is not int or size < 1 or type(count) is not int or count < 0
                             for size, count in capacities.items()):
        raise ValueError("Heavy-atom capacities need positive integer sizes and nonnegative integer counts")


def capped_waterfill(capacities, target, *, seed=DEFAULT_SEED):
    """Allocate the most even feasible counts, using seeded ties for the last rows.

    At fixed total this minimizes the sum of squared deviations from the flat
    target. If the total is unattainable, every available row is included.
    """
    _validate_capacities(capacities)
    if type(target) is not int or target < 0 or type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Require a nonnegative integer target and a 64-bit unsigned seed")
    available = sum(capacities.values())
    if target >= available:
        return dict(sorted(capacities.items()))

    low, high = 0, max(capacities.values())
    while low < high:
        level = (low + high + 1) // 2
        if sum(min(count, level) for count in capacities.values()) <= target:
            low = level
        else:
            high = level - 1
    quotas = {size: min(capacities[size], low) for size in sorted(capacities)}
    key = seed.to_bytes(8, "little")

    def rank(size):
        digest = hashlib.blake2b(str(size).encode("ascii"), key=key, digest_size=16, person=b"hac-quota").digest()
        return digest, size

    remainder = target - sum(quotas.values())
    unconstrained = sorted((size for size, count in capacities.items() if count > low), key=rank)
    for size in unconstrained[:remainder]:
        quotas[size] += 1
    return quotas


def source_aligned_candidates(canonical_rows, packed_smiles, heavy_atoms):
    """Replay a parent cache's canonical prefix and authenticate identity before source use.

    The parent may contain only a quota-bounded prefix of the finalized corpus.
    The caller can stop after finding enough eligible rows.
    """
    if len(packed_smiles) != len(heavy_atoms):
        raise ValueError("Packed cache identities and heavy-atom counts differ in length")
    index = 0
    for split, smiles, source in canonical_rows:
        if split != "train":
            continue
        if index == len(packed_smiles):
            return
        if smiles != packed_smiles[index]:
            raise ValueError(f"Canonical replay differs from packed training identity at row {index}")
        yield index, int(heavy_atoms[index]), source
        index += 1
    if index != len(packed_smiles):
        raise ValueError("Canonical replay ended before the packed training membership")


def select_pubchem_holdouts(candidates, *, capacities, train_quotas, count_per_split):
    """Add disjoint PubChem identities without consuming a planned training quota.

    Candidates arrive in the packed parent's seeded order. Accepted candidates
    alternate between validation and test. The source is checked per candidate
    and the aggregate slack at that heavy-atom count is decremented once.
    """
    _validate_capacities(capacities)
    if (type(count_per_split) is not int or count_per_split < 1 or set(capacities) != set(train_quotas)
            or any(type(count) is not int or not 0 <= count <= capacities[size]
                   for size, count in train_quotas.items())):
        raise ValueError("Holdout sizes and planned heavy-atom training quotas are incompatible")
    slack = {size: capacities[size] - count for size, count in train_quotas.items()}
    target = 2 * count_per_split
    if sum(slack.values()) < target:
        raise ValueError("Too few available molecules outside the planned training quotas")

    selection = {"val": [], "test": []}
    accepted = 0
    previous = -1
    for index, size, source in candidates:
        if type(index) is not int or index <= previous or size not in slack:
            raise ValueError("Holdout candidates must have increasing indices and known heavy-atom sizes")
        previous = index
        if source == "pubchem" and slack[size] > 0:
            selection["val" if accepted % 2 == 0 else "test"].append(index)
            slack[size] -= 1
            accepted += 1
            if accepted == target:
                return selection
    raise ValueError(f"Only {accepted} PubChem molecules fit the holdout slack; need {target}")
