# Read-only VAE chemistry compatibility assessment

The current DBMol VAE already uses12persistent CPU workers, ordered asynchronous
CPU/GPU overlap and batched GPU target NLL. These settings were inspected; no
DBMol code, sealed release, job, checkpoint, saved result or health gate changed.

| Mosaic optimization | Current VAE counterpart | Additional opportunity |
| --- | --- | --- |
| Probability-cost/template reuse | Already present in atom-preserving fast path | Check residual duplicate work before adding another cache |
| Persistent ordered search workers | Already12workers, bounded ordered tasks | Avoid nesting4workers inside each of12workers |
| Exact hard native-fact reuse | Target/search LRU exists, but native classification is deliberately fresh | Separate experiment could test exact-fact caching with all native/RNG/health semantics retained |
| GPU local-risk matrix DP | Active VAE objective has no corresponding local-risk term; target NLL is already Torch GPU | It is not a drop-in acceleration of the current objective |

Inspected source locations in DBMol_exploratory:

- `src/latentmol/chemistry_continuation.py`: defaults12workers/4Torchthreads/
  16CPUslots near55; rejects simultaneous valid-reference and failure-guidance
  objectives near731; ordered asynchronous target/anchor pipeline near822 and
  871–900; ordinary repair NLL near937, conditional failure guidance near943.
- `src/latentmol/repair_worker_pool.py`: persistent pool near122; deliberately
  fresh classification near183.
- `src/latentmol/atom_preserving_chemistry.py`: accepted targets are batched for
  GPU categorical NLL near712–725; ordered LRU and fresh-classification logic
  near349–380.
- `src/latentmol/chemistry_valid_reference_guidance.py`: eligible frozen-reference
  targets use batched GPU NLL near274–286.
- `src/latentmol/atom_preserving_fast.py`: cost and search reuse near44/80.
- `src/latentmol/atom_preserving_diagnostics.py`: native template reuse near31.

The actual saved source138 run configuration records workers12, overlaptrue,
microbatch128, realbatch256/priorratio2 and the frozen-reference objective.
Current VAE training is1real:2prior. A diagnostic profile's earlier inherited
1:3 note is not the actual training ratio.

The earlier36-graph CPU diagnostic cohort measured16.623s original serial,
4.736s optimized serial,1.414s with4workers and0.600s with12workers. These are
cohort measurements, not full-training speedups or extra gains available on top
of the current implementation. Any further VAE cache study must remain separate
from active training and prove exact inputs/ordered search/native decisions,
gradient flow, RNG and all original health/readiness/runtime/quota checks.
