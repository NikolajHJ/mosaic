# Allocated saved-report fixtures

Tests cover source/data scope, contiguous committed prefixes, segment-boundary
replay, actual retained matrices, exact scalar/hash correspondences and honest
failed/incomplete statuses. Numerical fixtures and rendering run only on HPC.
Viewer source contracts supplement real local-browser presentation QA.

The current six-arm source08 suite has149 cases:19 qualification,73 saved-data,
16 orchestration/storage and41 viewer contracts. It covers both chemistry arms,
their affinity/contact metric selection and separate continuation state. All
pre-existing FP32/simplex/scalar/hash/boundary checks remain mandatory. Source05's
obsolete one-arm literal assertion remains preserved in its failed evidence;
the corrected assertion requires the declared two-arm Boltz affinity branch.
Source07's full148 cases and rendering passed before its stale four-run heading
was caught. Source08 adds an explicit cohort-agnostic header contract and repeats
the full suite without changing scientific/frame validation.
