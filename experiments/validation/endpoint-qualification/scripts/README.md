# Endpoint diagnostic policy

`qualification.py` constructs loss-independent FP32 perturbation plans and
evaluates them with caller-supplied independent-primal, AD and frozen-identity
hooks. Numerical calls require an allocated HPC job. It neither loads a model
nor edits original checkers/results. See the [protocol](../README.md).
