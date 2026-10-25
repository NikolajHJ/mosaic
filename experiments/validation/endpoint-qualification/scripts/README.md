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
