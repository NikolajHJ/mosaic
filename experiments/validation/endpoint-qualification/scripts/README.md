# Endpoint diagnostic policy

`qualification.py` constructs loss-independent FP32 perturbation plans and
evaluates them with caller-supplied independent-primal, AD and frozen-identity
hooks. Numerical calls require an allocated HPC job. It neither loads a model
nor edits original checkers/results. See the [protocol](../README.md).

`admission.py`, `continuation_driver.py`, `continuation_ops.py` and
`continuation_stage.lsf` implement the separately user-authorized prospective
admission/child-run attempt. See the dated run record and policy. Metadata-only
`inspect_continuation_inputs.py` and `collect_continuation_arguments.py` preserve
read-only remote bindings; `prepare_continuation_plan.py` assembles those exact
identities locally. Sealing/submission helpers refuse existing attempt paths and
unknown outcomes. Original scientific sources/results are never overwritten.

Protenix diagnostic-only controls: `protenix_endpoint_forensics.py` preserves the fixed point inventory; `protenix_forensics_native.py` loads the exact original frozen source/checkpoint; `protenix_forensics_ops.py` gates CPU then a bounded diagnostic GPU allocation. `manage_protenix_forensics.py` provides stdlib prepare/stage/observe commands, and `protenix_forensics_stage.lsf` authenticates source before imports. None dispatches optimizer continuations.

Terminal forensic delivery uses `collect_protenix_forensics.py` (exclusive fresh archive/stream) and `verify_protenix_forensics_delivery.py` (stdlib tar/ZIP/NPY payload and provenance audit). `summarize_protenix_forensics.py` tabulates existing scalar findings on allocated CPU only; its one-shot account-bound `submit_protenix_forensics_summary.py` preserves original science/source/proofs and cannot dispatch optimization. See the dated forensic run for exact evidence and findings.

The `protenix_precision_*` modules own the separate-process prospective precision
reference, endpoint policy and bounded native/CPU orchestration. The associated
`manage_protenix_precision.py` is stdlib-only. Exact scope and immutable paths are
in runs/20261025-protenix-endpoint-fix-01/README.md; no optimizer branch exists.

`verify_protenix_precision_delivery.py` independently checks compact CPU delivery bytes, raw XML case attribution, actual scheduler/LSF and metadata-only startup using the standard library only. It does not evaluate arrays or native gradients.
