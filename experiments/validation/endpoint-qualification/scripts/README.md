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
