# Fixed-pose DBMol pilot entry points

`worker.py` owns two frozen native checkpoints on the SAME allocated CUDA GPU.
It imports only Mosaic's scorer and authenticated native Boltz package, never
DBMol_exploratory. Its line protocol is `mosaic-boltz-dbmol-v1`, deliberately
different from the old affinity-only worker.

From an authenticated Mosaic release root on an allocated DTU GPU, using the
parent campaign's pinned Boltz Python environment:

```text
python scripts/dbmol_worker.py \
  --context <original-prepared-context.pt> --context-sha256 <exact-SHA256> \
  --checkpoint <official-affinity.ckpt> \
  --structure-checkpoint <official-structural.ckpt> --device cuda
```

Paths and resource limits are resolved and sealed by the parent run record.
`scripts/dbmol_worker.py` is the authenticated release-only alias of this
directory's canonical `worker.py`; it is not a flat working-checkout file.
Both checkpoint hashes, pinned Boltz commit, original JQ1 pocket, frame layout
and frozen context are checked before `ready`. Requests are `evaluate`,
`evaluate_components`, `native_reference` and `verify_identity`. Evaluation
accepts `objective` to select total/component/primitive; its default is
`dbmol_total_loss`. Every request carries a unique ID; graph requests additionally
carry the SHA256 of exact FP32 X/E bytes. Independent native hard oracle calls
never run diffusion or alter the prepared seed.

`bridge.py` exposes `DBMolWorker` and `dbmol_loss`. Enable JAX x64 before arrays
are created. The adapter requires FP32 graphs/raw gradients and preserves the
FP64 objective/metrics across JSON and JAX custom VJP. It explicitly names
`dbmol_total_loss`, quantitative affinity, binder probability and raw binder
logit separately; no total is mislabeled as affinity. Higher derivatives are
unsupported. Native component audits remain allocated work and never update
optimizer state.

No script in this directory submits jobs. The parent controller owns immutable
packaging, allocated tests, raw native gradient gates, runtime/quota admission
and actual scientific execution. No local model/numerical test is authorized.
