"""Stable JSON configuration identities and explicit historical verification.

JSON object keys are strings. Hashing a Python dictionary with integer keys
before saving it can sort those keys numerically, whereas hashing its loaded
JSON sorts them lexicographically. New records normalize keys before hashing
and reject collisions such as ``1`` and ``'1'`` rather than losing a value.

The only supported historical reconstruction is the known active-set count
bank: unversioned ``config.contexts`` keys representing counts 1 through 50.
It changes no value and is accepted only if it exactly reproduces the stored
digest. Verification never rewrites the input record or historical artifacts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re


CONFIG_HASH_VERSION = "canonical-json-config-v1"


def normalize_json(value):
    """Return an independent JSON-compatible value without ambiguous keys.

    String and integer object keys are supported. Tuples become arrays, as
    they do in JSON. Boolean/float/None keys are rejected instead of depending
    on Python's implicit key coercion. Nonfinite scalar values are unsupported.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) and type(key) is not int:
                raise ValueError("Configuration object keys must be strings or integers")
            normalized_key = str(key)
            if normalized_key in result:
                raise ValueError(f"Colliding JSON configuration keys: {normalized_key!r}")
            result[normalized_key] = normalize_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Configuration values must be finite JSON scalars, arrays or objects")


def canonical_config_json(config) -> str:
    """Serialize normalized keys before sorting; preserve scalar distinctions.

    Default JSON spacing intentionally matches historical string-key digests.
    This identity is independent of dictionary insertion order and a save/load
    roundtrip, but it does not identify integers with floats or reorder arrays.
    """
    return json.dumps(normalize_json(config), sort_keys=True, allow_nan=False)


def config_sha256(config) -> str:
    return hashlib.sha256(canonical_config_json(config).encode()).hexdigest()


def verify_config_identity(record: dict, *, allow_legacy_active_set_contexts=False) -> dict:
    """Verify a record and return the exact scheme used, without modifying it.

    The compatibility flag belongs only at active-set artifact readers. A
    versioned mismatch, changed value or unknown hash version always fails.
    Reconstruction requires canonical supported count spellings. An unversioned
    direct digest remains accepted for unaffected historical records and fixtures.
    """
    stored = record.get("config_sha256")
    if not isinstance(stored, str) or not re.fullmatch(r"[0-9a-f]{64}", stored):
        raise ValueError("Parent configuration hash is missing or malformed")
    config = record["config"]
    if not isinstance(config, dict):
        raise ValueError("Recorded configuration must be a JSON object")
    normalized = config_sha256(config)  # Validate all keys/values even for legacy records.
    version = record.get("config_hash_version")
    if version not in (None, CONFIG_HASH_VERSION):
        raise ValueError("Unsupported configuration hash version")
    if normalized == stored:
        return {"verified": True, "stored_sha256": stored,
                "scheme": CONFIG_HASH_VERSION if version else "legacy-json-direct",
                "historical_context_keys_reconstructed": False}
    if version is None and allow_legacy_active_set_contexts:
        contexts = config.get("contexts")
        if (isinstance(contexts, dict) and contexts
                and all(isinstance(key, str) and re.fullmatch(r"[1-9][0-9]*", key)
                        and 1 <= int(key) <= 50 for key in contexts)):
            historical = deepcopy(config)
            historical["contexts"] = {int(key): value for key, value in contexts.items()}
            reconstructed = hashlib.sha256(json.dumps(historical, sort_keys=True,
                                                       allow_nan=False).encode()).hexdigest()
            if reconstructed == stored:
                return {"verified": True, "stored_sha256": stored,
                        "scheme": "legacy-active-set-integer-contexts",
                        "historical_context_keys_reconstructed": True,
                        "canonical_sha256": normalized}
    raise ValueError("Parent configuration hash changed")
