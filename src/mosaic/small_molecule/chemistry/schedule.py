"""Explicit update-indexed coefficients, independent of optimizer trial count."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RepairWeightSchedule:
    """Hold through ``hold_updates``, then interpolate geometrically to the end.

    Updates are one-based. A saved state s uses the coefficient for update s+1;
    the terminal state keeps the last update's coefficient. Trial evaluations
    never advance this schedule. This is a changing surrogate, not one objective.
    """

    initial: float
    final: float
    hold_updates: int
    total_updates: int
    version: str = "delayed-geometric-v1"

    def __post_init__(self):
        if (
            self.version != "delayed-geometric-v1"
            or type(self.hold_updates) is not int
            or type(self.total_updates) is not int
            or not 0 <= self.hold_updates < self.total_updates
            or not math.isfinite(self.initial)
            or not math.isfinite(self.final)
            or not 0 < self.initial <= self.final
        ):
            raise ValueError("Invalid delayed chemistry schedule")

    def for_update(self, update):
        if type(update) is not int or not 1 <= update <= self.total_updates:
            raise ValueError("Update is outside the declared schedule")
        if update <= self.hold_updates:
            return self.initial
        if update == self.total_updates:
            return self.final
        fraction = (update - self.hold_updates) / (
            self.total_updates - self.hold_updates
        )
        return math.exp(
            math.log(self.initial)
            + fraction * (math.log(self.final) - math.log(self.initial))
        )

    def for_state(self, state):
        if type(state) is not int or not 0 <= state <= self.total_updates:
            raise ValueError("State is outside the declared schedule")
        return self.for_update(min(state + 1, self.total_updates))

    def record(self):
        return asdict(self)
