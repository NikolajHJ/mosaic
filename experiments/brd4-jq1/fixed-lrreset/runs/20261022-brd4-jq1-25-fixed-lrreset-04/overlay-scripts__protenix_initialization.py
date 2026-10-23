"""Declared FP32 placement for exact historical Protenix initialization replay.

The unchanged JAX softmax runs atom rows on CPU and bond rows on the scorer's
device. Its probabilities are copied unchanged to the scorer device. This
restores the authenticated serialized starting point without changing logits,
temperature, softmax formula, loss, update or an identity tolerance. Every later
Protenix state uses the same placement; there is no special-case cached point.
"""
import jax
from mosaic.small_molecule.chemistry.state import GraphLogits
from mosaic.small_molecule.state import RelaxedGraph

VERSION = 'protenix-fp32-cpu-atom-softmax-v1'


class ProtenixReplayLogits(GraphLogits):
    """Keep identical FP32 softmax placement throughout a fresh trajectory."""

    def graph(self):
        cpu = jax.devices('cpu')[0]
        atoms = jax.device_put(self.atoms, cpu)
        probabilities = jax.nn.softmax(atoms, -1)
        return RelaxedGraph(
            jax.device_put(probabilities, self.atoms.sharding),
            jax.nn.softmax(self.bonds, -1),
        )
