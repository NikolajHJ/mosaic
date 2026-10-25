# Verified GPU component benchmark

GPU job 29809554 ran on one actual NVIDIA L40S, 25 October 2026
05:57:33–06:03:48 Paris/CET (04:57:33–05:03:48 UTC),375 seconds.
All19 GPU-only cases passed in125.92 seconds, with zero failures/errors/skips.
The live scheduler record had aged out at collection; original same-job LSF
success/entry/assessment/rawXML/source/CPU/input hashes independently verify
SUCCESS_PERSISTED. No scheduler DONE record is fabricated.

The six authenticated saved inputs passed all CPU/GPU integer graph-feature
comparisons and full loss/component/gradient parity tests for both scan and
matrix candidates. Original2e-5 absolute/relative tolerance, nonzero-field FD
epsilons.01/.003/.001,15percent/twoadjacent, and exact original-zero field
applicability remain unchanged. Scan matched the sampled loss/gradient bytes;
matrix reassociation is tolerance-equivalent, not bitwise identical. Maximum
recorded matrix total-value difference4.77e-7 and bond-gradient difference1.49e-8.
These are engineering tests of the existing frozen-reference chemistry loss,
not new model-readiness checks or an independent optimization replication.

Warm synchronized local valence-risk VALUE+GRADIENT GPU timings in milliseconds:

| Batch | Original GPU | Scan GPU | Matrix GPU | Matrix versus original |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.602 | 2.127 | 0.260 | 6.16x faster |
| 32 | 1.435 | 2.266 | 0.380 | 3.77x faster |
| 256 | 1.256 | 2.492 | 1.554 | 24% slower |

Compilation plus first-call seconds (original / scan / matrix):

| Batch | Original | Scan | Matrix |
| ---: | ---: | ---: | ---: |
| 1 | 26.526 | 2.336 | 1.985 |
| 32 | 18.099 | 2.281 | 2.682 |
| 256 | 18.502 | 2.468 | 3.869 |

Matrix is promising for the normal one-molecule case and batch32; it loses
at256. Scan is slower in warm execution for all three batches despite faster
compilation. The integer matrix telemetry takes roughly0.88–0.92ms per batch,
plus measured transfers0.49–0.70ms; there is no matched CPU timing to establish
that this feature conversion is faster than CPU bookkeeping. It does not
establish RDKit validity or replace native oracle checks.

These are GPU-versus-GPU local-kernel measurements; the original local-risk
loss already ran on the GPU. The batch1 saving is about1.34ms, compared with
roughly280ms for cached CPU preparation. Therefore6.16x kernel speed does not
imply6.16x overall optimization speed. Large preparation gains still come
from exact hard-graph caching and CPU workers. Full integrated model/runtime
validation would be required before adopting a new immutable production run.
Existing runs, settings, original losses, saved results and paused monitor stay
unchanged; no automatic production promotion or resubmission occurred.

Raw proofs are gpu-assessment.json, gpu-tests.xml/log, gpu-lsf.log,
gpu-device.txt and gpu-benchmark.json. The independent byte/XML/schema audit
is gpu-verification.json. The original19-file source/archive, all input and
frozen-reference hashes and original643 scientific source files matched.
