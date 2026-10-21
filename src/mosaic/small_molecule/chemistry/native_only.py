"""Explicit no-chemistry protocol for matched scorer-only logit baselines.

No repair search or chemistry preparation is performed. Saved raw readouts are
still captured by the shared trajectory writer. This is not an already-valid
teacher outcome, and is never represented as a passed chemistry assessment.
"""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class NativeOnlyPreparation:
    schema_sha256: str
    update: int

    @property
    def receipt(self):
        return {
            "version": "mosaic-native-only-preparation-v1",
            "schema_sha256": self.schema_sha256,
            "update": self.update,
            "chemistry_applicability": "not part of the objective; no teacher assessment",
            "rows": [],
        }

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.receipt, sort_keys=True).encode()).hexdigest()

    def verify(self, spec):
        if self.schema_sha256 != spec.fingerprint() or type(self.update) is not int or self.update < 0:
            raise ValueError("Native-only preparation schema/update differs")

    def record(self):
        return {**self.receipt, "identity": self.identity}
