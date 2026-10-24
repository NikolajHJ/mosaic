# Boltz LR0.0025: fresh 30,000-update matched pair

User-authorized 23 October 2026: run two fresh fixed-step Boltz-2 trials at
LR0.0025, 30,000 updates each, with and without the existing chemical-validity
guidance. The user explicitly stretched the chemistry ramp: coefficient0.001
through applied update5,000, then geometric growth to100 at update30,000.
This is a new experiment; no old optimization, checkpoint, result or job changes.

Both arms retain occupied neutral N25, chain/uniform mixing0.5, key17, T1,
the original JQ1 pocket, detached prepared Boltz pose/context, zero recycles,
no molecular-weight correction and quantitative-affinity-only native objective.
Exactly one finite fixed-rate candidate is applied per update, including finite
increases/equality/no-op. No clipping, normalization, backtracking or diffusion
gradient is introduced. The separate DBMol-objective pilot has its own owner.

Execution uses three serial segments of10,000 fresh updates per arm. Every
boundary replays exact FP32 atom/bond logits, probabilities and native key;
global chemistry/preparation indices never reset. Each next segment requires
authenticated predecessor scientific completion, fresh native/parity/raw
FD0.01/0.003/0.001 with15%/two-adjacent-scale agreement, nonzero/null/independent
primal/finite/frozen gates, applicable low/high coupled chemistry checks and
BOTH ten-update chemistry pilots, plus runtime and entire-campaign storage
admission. Terminal checks are mandatory at each segment. Failure stops that
arm, preserving its prefix; it never authorizes an automatic successor.

The native worker runs CUDA in its pinned Boltz environment; JAX graph and
chemistry remain CPU FP32. Each allocation is <=23h; no interarm dependency
exists after both initial readiness checks pass. Quota accounting includes
original remaining chemistry/results, current report/diagnostics, both ongoing
50k Protenix chains and declared independent DBMol allowance. Full long dispatch
waits for authenticated Protenix641 pilot projections. Storage checks retain
4GiB/100,000-file headroom and never delete history to fit.

Canonical configuration, driver/protocol/controller/fixtures, release packager,
one-shot submission helper, read-only observer and small receipts are owned by
[attempt01](runs/20261023-boltz-lr0025-01/README.md). Authenticated source625 is
the scientific parent; its only package-library overlay is the exact already
validated optional-global-offset trajectory from source641. Archive-only legacy
paths are declared compatibility aliases, not loose working scripts.

All models, numerical tests and scientific rendering run only on allocated DTU
HPC. Local preparation is standard-library byte/AST/JSON packaging. Current
status: attempt01 CPU failed two unit fixtures after464 passes and remains
failed. [Attempt02](runs/20261023-boltz-lr0025-02/README.md) isolates the inherited
lazy-checker monkeypatch without changing scientific bytes, settings or gates.
Its full allocated CPU suite passed799 actual cases; native validators29798605
and29798607 then failed before model loading on missing manifest provenance.
[Attempt03](runs/20261024-boltz-lr0025-03/run-record.md) seals the authenticated
revision correction and actual generated-command startup checks. DTU SSH
authentication currently blocks its allocated testing and fresh submission;
no new native gradient pass or optimization start is established.

Dated update25October2026 00:36Paris: the independently admitted non-chemistry
chain now uses matrix snapshots/trials every20updates (plus0/end/boundaries and
boundedlatest/best), with EVERY loss/component/update scalar preserved. All78
allocated persistence/routing checks and original CPU/startup/readiness proofs
passed. First GPUjob29809297/callback29809298 is actuallysubmitted/PEND, with no
optimization start inferred. Scientific settings/checks/results unchanged;
delayed-chemistry arms remain deferred. See
[runtime/storage/source receipts](runs/20261025-nonchem-stride20-startup-03/README.md).
