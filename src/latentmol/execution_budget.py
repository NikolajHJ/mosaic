"""Explicit execution limits, separate from the molecular objective and gates.

The call ceiling covers Counter-mediated Boltz forward/native-forward calls,
including native AD probes. Internal activation-checkpoint recomputation is not
a separate counted call. An outer process deadline must bound uninstrumented
setup, subprocess denoising and a single long-running native call.
"""

from dataclasses import asdict, dataclass
import time


class ExecutionBudgetExceeded(RuntimeError):
    """Stop before another tracked native call; never turn this into a pass."""


@dataclass(frozen=True)
class ExecutionBudgetConfig:
    version: str
    wall_seconds: int
    max_boltz_calls: int

    def __post_init__(self):
        if self.version != "tracked_native_v1":
            raise ValueError("Unsupported native execution budget")
        for key in ("wall_seconds", "max_boltz_calls"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Execution limits must be positive integers")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Execution budget requires exactly version, wall_seconds, max_boltz_calls")
        return cls(**value)

    def record(self):
        return asdict(self)


class TrackedNativeBudget:
    """Guard before each Counter call and persist any rejected attempt.

    Only declared native forward calls consume the count; all Counter calls
    check elapsed time. A failure remains a failed/partial optimization record.
    The supplied clock makes boundary behavior testable without sleeping.
    """

    def __init__(self, config, persist, *, clock=time.monotonic):
        self.config, self.persist, self.clock = config, persist, clock
        self.started, self.calls = clock(), 0
        self._save("running", 0.)

    def _save(self, status, elapsed, **details):
        receipt = {"version": self.config.version, "limits": self.config.record(),
            "status": status, "elapsed_seconds": elapsed, "tracked_boltz_calls": self.calls,
            "scope": "Counter-mediated Boltz forward/native_forward; internal checkpoint recomputation excluded",
            **details}
        self.persist(receipt)
        return receipt

    def complete(self):
        """Persist terminal limits proof only after all frozen-state checks."""
        elapsed = self.before_call("execution", "completion", "verify")
        return self._save("completed", elapsed, within_limits=True)

    def before_call(self, model, purpose, action):
        elapsed = self.clock() - self.started
        counted = model == "boltz" and action in ("forward", "native_forward")
        reason = ("wall_seconds" if elapsed >= self.config.wall_seconds else
            "max_boltz_calls" if counted and self.calls >= self.config.max_boltz_calls else None)
        if reason:
            self._save("exhausted", elapsed, reason=reason, rejected_call={
                "model": model, "purpose": purpose, "action": action})
            raise ExecutionBudgetExceeded("Native execution budget exhausted: " + reason)
        if counted:
            self.calls += 1
            self._save("running", elapsed)
        return elapsed
