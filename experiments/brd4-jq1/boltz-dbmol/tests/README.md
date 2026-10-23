# DBMol composite objective validation

These experiment-owned fixtures and the native validation helper are executed
only on allocated DTU HPC. Local work is limited to standard-library source,
syntax, JSON and byte inspection. They do not load DBMol/VAE code or change an
existing saved study. See the [experiment protocol](../README.md).

- `fixture_objective.py`: exact component signs/reductions, raw ensemble logits,
  contact orientation/minima/epsilon and FP64 head arithmetic.
- `fixture_native_paths.py`: independent relaxed-graph feature mapping, native
  confidence typed-bond path, frame and frozen-source guards.
- `fixture_gradient_contract.py`: synthetic positive and negative controls for the
  mandatory native derivative checker. Synthetic checks are not real-model
  scientific acceptance.
- `fixture_bridge.py`: FP64 scalar transport with exact FP32 graph VJPs, scalar
  cotangents and stopped reporting metrics through the actual JAX adapter.
- `fixture_pilot_contract.py`: declared pilot/segment/rate/frozen/CPU/peer-resource
  guards and exact uninterrupted versus checkpoint-replayed FP32 trajectories.
- `native_validation.py`: independent actual worker primals and raw graph
  gradients for all three binding components and their total, at the original
  raw finite-difference scales. Also checks hard/native parity, scalar/gradient
  recomposition, the actual JAX transport and row-shift null controls, unchanged
  frozen identities, and exact input-byte correspondences.

The original raw X/E checks remain mandatory: epsilon 0.01/0.003/0.001,
15% relative agreement, same sign, two adjacent passing scales, independent
primal calls, finite/nonzero raw derivatives and typed-bond contrast. An
infeasible predeclared probe or failed check is a failure. No smaller scale,
AD-only agreement, clipping, outcome-selected direction, projection or retry
can turn that failed gate into a pass. The original model/runtime/storage
checks remain additional requirements owned by the experiment controller.

Canonical fixtures use explicit allocated collection. Release-only aliases
`tests/test_dbmol_*.py` keep the authenticated scientific source657 paths and
bytes; no compatibility aliases are written into the working checkout.
