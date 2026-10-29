# Confidence margin attempt002

Prepared29October2026, rejected before submission/no numerical execution. Same explicit user scope
and scientific protocol as [attempt001](../20261029-chemistry-confidence-01/README.md).
First attempt was rejected by batch queue hpc before submission; inspection
confirms hpcint is the site's allocated interactive CPU queue. Only the resource
selection/guard changes, not loss,35fixtures,24paired pilots or tolerance.

Source/test/comparison inputs and hashes are sealed before actual execution.
All numerical work is on real allocated CPU, with source/output/cache in
node-local scratch then byte-verified transfer to laptop. Persistent shared
WORK3/HOME runs are not written; no saved run restarted or changed. This study
does not execute native models or establish binding performance.

Actual11:34:12UTC: site esub255, walltime option unsupported for hpcint, job not
submitted/no job ID. Source/transport receipts preserved. New attempt003 uses
the normal queue interface without -W, with an explicit1700-second process
deadline; same CPU resources and scientific tests. This is not a gate waiver.
