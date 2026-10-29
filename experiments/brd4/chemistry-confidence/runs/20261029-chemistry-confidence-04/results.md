# Bounded-slope confidence comparison —29October2026

Implementation/gradient validation passed on allocated DTU CPU job29848047.
All35cases passed, zero failures/errors/skips. Native binding benefit remains
untested. This term stays opt-in; no current run or VAE training was changed.

## Implemented policy

Current repair remains unchanged. Optional margin coefficient0.01, target
log-odds margin0.2, Huber slope cap1.0. Guidance is zero for invalid source
argmaxes and for sufficiently confident decisions. Its labels/validity masks
are detached, authenticated and fixed during each optimizer step's trials.
Only the regularizer's slope is capped; native gradients are not clipped.
The loss value grows linearly at large deficits rather than being bounded.
Composition weight is zero; original composition diagnostics remain available.
No atom-dependent valence mixture or atom-editing repair teacher was introduced.

## Checks and comparison scope

- Independent FP64 primal and FP32/FP64 atom/bond FD at0.01/0.003/0.001,
  requiring two adjacent agreements.
- Bounded slopes at small/large deficits, exact zero above confidence, zero
  on invalid-source rows, and mixed-batch normalization.
- Frozen targets when a trial changes argmax, target refresh between steps,
  tamper rejection and composition/native-only mode guards.
- Bitwise prechange value/gradient parity with the option absent, zero-weight
  parity, and correct native+repair+margin gradient addition.
- An actual fixed-step two-update integration fixture changed C->O under a
  strong polynomial native test signal. This proves no hard atom freeze,
  not freedom from exploration bias under real binding gradients.

Twelve paired chemistry-only comparisons used four predefined starts, seeds17/
23/29,128updates/arm,LR0.3,repair coefficient1.0 and identical search policy.
There was no binding objective. Every checkpoint samples256graphs with matched
uniform random draws between arms, without repair or rejection conditioning.
Raw valid/invalid matrices, categorical probabilities and all loss rows are
preserved. JS diagnostics here measure drift from matched initial composition;
they are not training-dataset composition measurements.

## Terminal sampled strict validity

Each cell is a raw count out of256at update128. Compare arms within a row;
checkpoint0and128 use different panels, so their differences also contain
sampling noise. These small differences are not statistical efficacy evidence.

| Start | Seed | Current repair | Repair + margin |
| --- | ---: | ---: | ---: |
| Uncertain valid CCC |17|76|80|
| Uncertain valid CCC |23|61|61|
| Uncertain valid CCC |29|80|81|
| Confident valid CCC |17|255|255|
| Confident valid CCC |23|256|256|
| Confident valid CCC |29|255|255|
| Valid15atoms/3sulfurs |17|8|8|
| Valid15atoms/3sulfurs |23|8|8|
| Valid15atoms/3sulfurs |29|7|7|
| Initially invalid, F with two bonds |17|46|50|
| Initially invalid, F with two bonds |23|48|49|
| Initially invalid, F with two bonds |29|51|52|

All terminal argmaxes were valid in both arms. Already-confident controls had
exactly zero margin throughout. The sulfur-rich molecule stayed valid and
retained3sulfurs in both arms; sampled validity did not improve in that case.
The invalid starts each underwent two bond-label changes and became valid in
both arms; the margin activated only after a valid source was prepared.

No atom-label changes occurred in these chemistry-only paired pilots: there
was no native binding force asking for identity changes. Sample diversity did
not improve uniformly: for uncertain CCC/seed23, unique raw-valid sampled
identities were46with repair and44with margin; for the invalid start/seed23,
they were33and35. Full canonical/fingerprint/scaffold metrics are in raw JSON.
This does not demonstrate long-run diversity preservation or identity locking.

Warm loss+gradient time for128calls, uncertain CCC/seed17: current repair
0.025205906946212053seconds, repair+margin0.028678907081484795seconds.
Preparation and first compilation are separate in raw records. Timing is one
CPU run with fixed arm ordering, not a rigorous GPU/full-model speed benchmark.

## Interpretation and evidence

The implementation behaves as specified, with modest or unchanged sampled
validity in this small isolated study. Do not enable it by default or claim
binding improvement from these measurements. A later matched native-model
comparison should measure margin/native gradient ratios, valid-to-valid atom/
bond transitions, sampled validity/diversity and binding objectives separately,
retaining all existing native/gradient/frozen/runtime/storage gates.

Scientific source manifest:
`48017afdc792ec5e2f6e181c49e3eb4cee2e910671e88ac9b1cb0f1e950f9afd`.
Independent [byte/XML/schema/scheduler audit](independent-verification.json)
passed145source files/81evidence members/72sample archives. [Raw same-job
scheduler/LSF](scheduler.json) confirms DONE on n-62-29-6,12:36:51--12:39:18Paris.
Full byte-verified evidence lives at
`results/hpc/20261029-chemistry-confidence-04/allocated01/verified/`.
Earlier queue/scratch failures remain recorded; none was relabelled as a pass.
