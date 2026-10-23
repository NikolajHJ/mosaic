"""Small runtime boundary shared by built-in and future external model plugins."""

from dataclasses import dataclass
from typing import Callable, Mapping

from torch import Tensor, nn

from latentmol.representation import HardGraphBatch


@dataclass
class BranchRuntime:
    """A frozen branch and an ordinary native reference for its prepared seed.

    ``native_reference`` is a seed-only fixed-context parity reference. It must
    not be presented as freshly preparing/rescoring changed hard molecules.
    ``assets`` records verified checkpoint/context/source identities. Every
    model sees the same slot-ordered chemistry, though its native maps may differ.
    """

    branch: nn.Module
    seed: HardGraphBatch
    active: Tensor | None
    native_reference: Callable[[], Mapping[str, Tensor]]
    assets: dict
    verify: Callable[[], None] | None = None
