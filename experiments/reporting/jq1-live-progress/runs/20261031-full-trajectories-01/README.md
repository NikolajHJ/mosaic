# Complete saved trajectories from update0

Question: replace separate continuation panels with complete original histories starting at the real global update0, without changing saved experiments or renumbering later segments.

Use the existing canonical report command and renderer/cache. Seven panels consolidate Protenix0-50k, Protenix+chemistry0-50k, Boltz0-30k, Boltz+chemistry0-30k, the DBMol prefix0-7.5k, the independent noisy1k pilot and DeFoG0-30k. Every scalar observation is retained, including duplicate boundary observations; actual FP32 checkpoint arrays and key are checked before joining. Original failed and unqualified outcomes remain in the per-segment proofs. No model/scoring/repair or optimization is run for the report.

The original Protenix0-10k export had been offloaded. Byte-only recovery verified8,748 archived originals plus1,257 surviving originals, restoring all10,001 snapshot files into a NEW report-only input directory under `report-inputs-from-zero-20261031-01`. The old studies and backup archive remain unchanged. Independent restoration proof is in `results/hpc/20261031-report-from-zero-01/restore-verification.json`.

Canonical `report_history.py` handles the dense original format and terminal interrupted prefixes without inventing terminal summaries. Interrupted exports use committed JSONL, original failed LSF, actual latest-state checkpoints and original config/source/submission. The original case export may lag or include an uncommitted trial: only its matching committed overlap is admitted; counts are disclosed. Dense original matrices are all checked on allocated CPU, with every20th plus endpoint selected for display. The numerical `report_data.py` renderer remains byte-identical.

Budget: existing4CPU/4GiB per slot/1h CPU report job; explicit2GiB own report allowance and shared exploratory capacity checks. Restore allowance1GiB/12,000 files independently passed. Numerical fixtures and frame checks only on allocatedHPC; local work is byte/schema/AST handling and browser QA.

Allocated engine241 raw cases passed in CPU29862133. Earlier237-case fixture run29862109 had235passes and two missing-output-directory fixture failures; fixed fixtures passed237 in29862112. Another237-case engine29862121 passed, then metadata capture correctly rejected the legacy signal-termination LSF format. The dedicated partial-history parser now authenticates nonzero signal/exit failures with the exact Subject job identity; strict legacy readers are unchanged. Final241-case engine is15afb7d16d5c4c060e008ba335d6b90904371a169d7d0de57454085db1f3dfe8.

Actual report request: `request-20261031T212940Z-ea5f91ce`, allocated render29862166. Never replay its submission; read the current local request/result and exact receipts before recovery. Edition0011 published21:43:37UTC31October at http://127.0.0.1:8780/report.html. Actual render29862166 DONE/hpc/same-job LSF success, all241 allocated CPU cases, all120 served HTTP files, independent archive/byte/XML audit and all7 actual update-zero selector checks passed. Earlier editions remain immutable. See publication-0011.json.

Command:
```powershell
py -3.13 scripts/reporting/report.py --profile experiments/reporting/jq1-live-progress/configs/dtu-report-home.json --runs experiments/reporting/jq1-live-progress/configs/current-runs.json
```

Future reports should retain these complete chains. The restored protenix-zero/study is a required report input, not a disposable render cache. Interrupted endpoints27175 and28986 are real off-stride frames; browser navigation crossed27175→27195 and28986→29006. The original failures remain visible. The healthy DBMol and fresh50k noise producers were not changed; their existing historical0–7500 and independent1k pilot panels remain distinct.
