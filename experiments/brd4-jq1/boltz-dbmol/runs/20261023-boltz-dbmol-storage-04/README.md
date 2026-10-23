# DBMol transport-attribution launcher, attempt04

Prepared 23 October 2026 after CPU03 job 29798769 completed all 1,135 cases
without failures, errors or skips. Its callback 29798770 failed strict module
attribution because the nine unchanged bridge tests, located outside pytest's
launcher root, had empty XML class names. Preserve the CPU03 pass and callback
failure as separate actual outcomes. No native GPU was submitted.

This new small launcher preserves scientific source657, manifest
`24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436`, all scientific
settings, native gradient checks and saved studies. It adds the EXACT unchanged
source657 bridge fixture as a launcher member, beside the two already corrected
transport fixtures, so all three execute under one pytest root. Strict module
attribution is retained, and all 1,135 cases repeat under `cpu-04` and
`after-cpu-04` before any native dispatch.

The package contains eight files: controller, two wrappers, metadata fixture,
three transport fixtures and this README. Both executing source manifests and
every launcher byte authenticate before imports. Every earlier failure remains
pinned, and an additional history reader authenticates CPU03's seven passing
groups, same-job LSF success and the separate callback attribution failure.
No CPU03 scientific or fixture failure is invented. Capacity budgets and all
reserves remain unchanged. Preparation and static byte/AST review establish no
new job or pass; the parent owns staging, submission and allocated validation.
