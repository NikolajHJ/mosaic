# Exploratory workflow audit — 30 October 2026

Scope: prospective shared code and policy in DBMol_exploratory and Mosaic.
The user requested quick early signals instead of qualification before every
pilot. No scientific training/model experiment, HPC command or job operation was
performed by this sweep. Existing failed results and live/sealed campaigns retain
their original meaning. Local fixture tests are engineering evidence only.

## Findings and changes

| Finding | Change for new work |
| --- | --- |
| DBMol AGENTS.md contained 1,652 lines mixing current instructions, historical status and special-case gates | Exact file archived; short active instructions now select exploratory defaults and relevant experiment context. Mosaic instructions simplified likewise. |
| Small runs inherited native parity, all-channel AD/FD and reproducibility admission | Shared entry points expose exploratory vs qualified scope. Exploration uses finite backward checks; detailed qualification is explicit. |
| DBMol multimodel required an extra allow-experimental flag and a prior qualification report before any run | Exploratory run/pipeline can execute its bounded finite/attached smoke directly. Qualified reports keep their own reader requirements. |
| New active-set configs inherited precise FD gates from older campaigns | Omitted FD contract selects exploration. Explicit old policies keep their qualified scope unless the user deliberately selects a new profile. |
| Every invocation could hash the entire source/test/config tree and inventory all packages | New exploratory entry points use selected source bytes plus Git summary; complete provenance remains optional/qualified. This is not exact replay evidence. |
| Dense graph snapshots can dominate output for short questions | New DBMol multimodel exploration defaults to stride10 observations, retaining scalar metrics and endpoints. Mosaic graph checks default compact artifacts. |
| Local neighbor pilot aborted at10 of100 updates on a quality drop, obscuring the bounded endpoint | Future exploratory pilot records warnings and continues; explicit qualified mode retains the earlier stop. The failed weight0.1 pilot is not rerun or adopted. |
| Storage admission reserved multiple GiB and could block on unrelated writer provenance | New shared Mosaic admission checks own bounded remaining writes against measured space/known quota, with 16–256 MiB proportional reserve. Missing quota/foreign provenance warns. |
| Storage adapter contained a campaign-specific goal rewrite | New exploratory path never enters that legacy campaign adapter and does not change any goal/config. Explicit historical v2 replay remains available. |
| Every generated directory required a README; new tests had to match a migration inventory exactly | Generated directories are exempt; missing authored notes warn by default. New canonical tests are allowed without fabricated aliases. Missing/duplicate migrated tests and broken/escaping paths still fail. |
| Windows worker pipe polling used socket-only select semantics | Windows uses an owned line-reader queue with tested timeout/EOF/cleanup; Unix selector behavior remains unchanged. |
| Valid one-atom graphs had an empty-edge summary reduction failure | Exploratory gradient summaries handle empty bond arrays; regression added. |
| Small synthetic CPU fixtures demanded an LSF allocation | Appropriate local CPU fixtures are permitted; native/GPU claims still need their actual environment. |

## What remains deliberate

Nonfinite computations, malformed shape/input contracts, detectable missing required
paths, unintended frozen-state writes, own-output collisions/budget violations and
measured capacity shortages remain errors. A JAX zero derivative alone does not
prove the path is live: distinguish a plateau from stop_gradient with an informative
fixture or qualified check when needed. Smoke receipts explicitly do not qualify
derivative accuracy, native parity, repeatability or chemical improvement.

Historical controllers remain campaign-specific replay tools, including fixed-tag
Mosaic managers that explicitly name storage v2. New code must use shared APIs;
this sweep does not rewrite all old launchers or adopt working-tree changes into
running campaigns. New storage bundles must include the shared accounting module
and adapter. Existing DBMol explicit line searches can still spend many native
forwards per update; their optimizer settings were not changed here.

The evidence also shows scientific questions beyond infrastructure: improved
surrogate loss can leave hard molecular identity unchanged, and the neighbor-loss
pilot improved local valence while worsening connectivity. These are negative or
mixed scientific signals to investigate, not reasons to demand more bookkeeping
before another bounded question. This sweep is not a claim that every gradient
failure was spurious or that every repository issue has been found.

## Validation

Focused local positive/negative fixtures cover new exploratory defaults, explicit
qualified/legacy behavior, nonfinite/missing-path errors, quality warnings,
output ownership/capacity, layout discovery and endpoint/observation retention.
Results (suites overlap; do not sum them as distinct cases):

| Suite | Result |
| --- | --- |
| DBMol multimodel/active-set/repair/provenance/pointwise CPU fixtures | 145 passed; two warnings |
| Mosaic exploratory/gradient/chemistry/bridge CPU fixtures | 131 passed; two optional archived-replay tests skipped |
| Mosaic one-atom regression and historical routing/contract fixtures | 179 passed |
| Shared storage metadata/default/legacy/capacity fixtures | 59 passed |
| Root local-pilot quality policy and layout discovery fixtures | 8 passed |

The skipped archived replays require an explicit `MOSAIC_CONTROL_TRAJECTORY`
asset and are not derivative or smoke tests. Exact commands are recorded in
the ledger entry for this sweep.
The full stdlib Mosaic layout check passes in exploratory mode; it reports new
canonical tests outside the historical registry without failing them. No native
model/GPU qualification or scientific experiment is claimed.

Current policy: [exploration](EXPLORATORY_RESEARCH_POLICY.md),
[gradients](GRADIENT_VALIDATION_POLICY.md). Historical v1 gradient CPU evidence
(543 scoped cases) is preserved separately and is not a prerequisite per pilot.
