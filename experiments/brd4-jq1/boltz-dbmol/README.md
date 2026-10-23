# Boltz-2 fixed-pose DBMol objective pilot

User authorization 23 October 2026: one additional fresh 10,000-update pilot at
LR0.0025, using the composite binding objective independently of chemistry
guidance. The parallel 30,000-update affinity-only comparison has a separate
owner. Saved studies, native models and prepared contexts are unchanged.

The pilot retains 25 occupied neutral slots, model key17, temperature1, the
original chain/uniform initialization and authenticated JQ1 pocket and detached
predicted pose. The loss is native affinity regression minus mean raw binder
logit, plus 2.5 times pocket/anchor contact and 2.5 times the six-term native
confidence reduction. Contact uses `contact_head_and_assembly_float64_v1`;
trunks, checkpoint parameters and raw graph/derivatives remain FP32. The scalar
and JAX bridge remain FP64. There is no VAE, diffusion gradient, new pose,
DeFoG, molecular-weight correction or chemistry repair in this pilot.

Two frozen checkpoint-specific trunks execute physically. Affinity uses the
original official checkpoint; structural distogram and confidence use the
official structural checkpoint with SHA256
`090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1`.
Confidence never receives an affinity-trunk transplant. Pocket token IDs come
from Mosaic's existing passed preparation audit, not a new distance selection.

## Owned files and execution

- [scripts](scripts/README.md): persistent native worker and explicit JAX adapter.
- [tests](tests/README.md): separate allocated fixtures and native gradient audit.
- Reusable scorer: [dbmol.py](../../../src/mosaic/small_molecule/boltz2/dbmol.py).
- Reference inspected read-only: DBMol_exploratory/docs/77 and its native loss
  implementation. This experiment neither imports nor modifies that project.

The parent campaign supplies a dated immutable run/release record, authenticated
source/config/input paths and its exact preparation/submission/collection
commands before launch. Native work is allocation-only; worker invocation is
documented in the script README. No launch command targets an old receipt or
saved run. Fixed steps use exactly one finite LR0.0025 candidate per update,
including finite uphill/equal/no-op updates, without line search, clipping or
gradient normalization.

Full allocated CPU fixtures, independent hard/native parity, actual raw X/E
finite differences at0.01/0.003/0.001 with15% tolerance and two adjacent passes,
component/total attached nonzero gradients, independent primal/null/frozen
checks, real pilot runtime and whole byte/file capacity must pass. The reference
project's pointwise-AD contract is additional background, never a substitute
for Mosaic's full FD gate. Component gradients can oppose; contact minima and
native readout branches remain piecewise differentiable. Gradient validity,
optimization progress, chemical validity and binding activity are separate
claims.

Current status: implementation prepared on23October2026. Local work is
stdlib AST/byte inspection only; no native/numerical validation or scientific
optimization is established by these source files. Exact jobs and results belong
to the parent campaign's dated run record and the global experiment ledger.


Actual scientific source: [pilot01](runs/20261023-boltz-dbmol-pilot-01/README.md),
657 immutable files. Its CPU29798549/callback29798550 failed entry storage before
tests, and the original failures remain preserved. The separate five-file
[storage launcher02](runs/20261023-boltz-dbmol-storage-02/README.md) retains all
scientific bytes and checks, authenticates completed original runs before
removing only their future-growth forecast, and keeps every saved byte charged.
CPU29798699/callback29798700 then failed seven transport fixtures; the exact
failures remain preserved. The separate [fixture-only launcher03](runs/20261023-boltz-dbmol-storage-03/README.md)
corrects those mocks and assertions while retaining all scientific bytes and
all1,135 cases. New CPU29798769/callback29798770 were submitted after fresh
validation capacity passed. No DBMol native gradient pass or optimization start
is established yet.
