"""Optional, fixed-occupancy Boltz-2 feasibility adapter.

Importing this package does not import Boltz or download checkpoints. Geometry,
slot maps and preprocessing are fixed; this is not variable-size Boltz inference.
"""

from .adapter import adapt_graph, fixed_occupancy_graph
from .contracts import PreparedComplex

__all__ = ["PreparedComplex", "adapt_graph", "fixed_occupancy_graph"]
