# JQ1-pocket 25-slot, 10,000-update comparison

21 October 2026. The user explicitly authorized stopping the old25-slot runs,
implementing the JQ1 pocket for fresh Protenix, Boltz-2 and Boltz-2+chemistry
arms, testing on allocated DTU HPC and submitting each only after its gates
pass. No diffusion gradient may reach the optimized molecule. This supersedes
the previous instruction to preserve those two live arms unchanged, for exactly
jobs29769525/29769540. All historical evidence remains preserved.

## Actual retirement and queue selection

Both exact old jobs were queried before cancellation. `bkill` was issued for
29769525 and29769540, with actual EXIT confirmed21October11:15:33/11:15:34Paris.
Original source04 readiness passes remain passes; these long trajectories are
intentionally interrupted partial runs, not completed10,000-update results or
spontaneous numerical failures. Original artifacts remain in their old run
directories. Stop/queue evidence is kept remotely at
`/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-stop-01/`.

The user then requested the GPU queue with least waiting. Live `bqueues -w`,
`bhosts -gpu -w`, queue limits and `bqueues -u s234818 -w` were inspected. H100's
lighter queue is restricted and absent from this account's eligible list.
Among eligible long GPU queues, L40S had the smallest pending-slot backlog
(110 versus148A10,160A40,1495A100,2735V100 in the captured snapshot). Those are
LSF slot counts, not job counts or a promised waiting time. Interactive queues
do not provide the long allocation needed. Choose **gpul40s for all three arms**
and use matching hardware for readiness/pilots and optimization: one exclusive
48GiB L40S, four CPU slots,8GiB/slot; two-hour native readiness/preparation and
23-hour long allocations. Preserve the raw queue snapshot; no restricted queue
access or exact start-time prediction is implied.

## Pocket, gradient scope and comparison

Use the existing experimentally measured1.60A BRD4-BD1/JQ1 structure
[3MXF](https://www.rcsb.org/structure/3MXF), pinned PDB SHA
`534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f`.
On allocated CPU, derive the protein heavy-atom neighborhood within6A of the
single crystallographic31-heavy-atom JQ1. Solvents/ions are excluded. Alternate
locations are chosen deterministically by occupancy, then blank/A preference.
Bind every selected author residue to its supplied127-residue sequence position
and amino-acid identity. Store the full definition and distances; do not claim a
globally optimal pocket radius or JQ1-like chemistry for generated graphs.

| Arm | New declared pocket treatment | Optimized native term |
| --- | --- | --- |
| Protenix | Full target context; only selected native protein tokens eligible in contact reduction | Negative mean ligand best-pocket contact probability, coefficient1,8A conservative native bins |
| Boltz-2 | Native pocket constraints during fresh pose preparation; passed placement audit; detached fixed coordinates afterward | Native quantitative affinity ensemble, coefficient1 |
| Boltz-2+chemistry | Identical prepared pocket context and initialization to Boltz-only | Same affinity plus existing repair schedule |

Protenix retains its contact algebra with a restricted partner set; it does not
require every pocket residue to contact the ligand and introduces no new mean
coverage/anchor loss. Whole-target historical `NativeContactLoss` calls remain
supported with no pocket argument. Map native residue metadata explicitly rather
than treating author140 as a token index. The earlier unjitted-hard versus jitted-
attached Protenix parity failure is preserved. This new version compiles the
independent original-hard-feature geometry-off reference in the same full
trunk/head scope, checks every hard/injected native feature exactly, and retains
the original2e-5 output tolerance. This is an implementation correction proposal
until the allocated native parity gate actually passes.

Boltz uses native one-based sequence anchors40/99 (authorTrp81/Asn140),6A pocket
constraints with `force:true` and native inference-time potentials only during
fresh carbon-chain pose generation. Native protein/crop/atom maps must retain
all selected pocket identities. At the actual prepared coordinates, both anchor
minimum distances must be <=6A and at least50% of ligand heavy atoms must lie
within6A of the selected pocket's protein atoms. This is a declared placement
gate, not a binding/ligand-RMSD certificate. Failed pose bytes and audits are
saved and must not be promoted to readiness.

Native structure generation happens once outside optimization. Saved coordinates
are detached, affinity heads receive `coordinates.detach()`, and production
scoring uses the input/trunk/affinity path only. No diffusion/coordinate/VAE/latent
optimization or trained surrogate is introduced. The affinity output remains a
whole-complex native scalar at the chosen fixed pose, not a separate per-pocket
affinity predictor. JQ1 supplies a site reference, not a transplanted ligand.

All arms keep25occupied neutral slots/300pairs,50%chain/uniform initialization,
LR0.03,key17,T1,ten-trialbest_primal,10,000total updates,composition0. Boltz
retains zero recycles/noMWcorrection; binder probability is separate. Chemistry
is the existing objective,0.001 through applied5000 then geometric to50at10000;
fixed target/coefficient per trial set. Raw invalid/disconnected graphs stay
visible. No temperature or continuous-chemistry prototype was adopted.

## Validation and conditional submission

All numerical extraction/tests/model/chemistry/evaluation stay allocated DTU HPC.
Local work is static editing, AST/JSON/LF inspection, packaging/transfers and
stdlib byte/XML/schema verification. CPU reruns legacy/bridge, mapping, unchanged
chemistry oracle and controller suites plus new pocket selection/mapping/loss,
pose placement and no-diffusion gradient tests. CPU/controller authentication
requires actual scheduler DONE, actual LSF success and exact raw proof hashes.

Each native arm must pass original hard parity, independent-primal raw atom/bond
FD at.01/.003/.001,15% and two-adjacent criteria, primal/AD parity, finite/frozen
checks and ten-update pilot. Protenix retains positive-bond-order null checks;
Boltz retains typed-bond contrast. Chemistry also requires nonzero invalid-graph
signal, low/high-weight coupled FD and both disposable ten-update pilots.
Pocket definition/context/config/source bindings and no-diffusion scope are new
additional requirements. All original thresholds remain unchanged.

Keep runtime gate `1.5*slowest_pilot_mean*10000+1800<82800` and whole-campaign
fresh conservative quota, summing passed arms' projections plus4GiB reserve.
No partial/native failure can become a completed study. Per-arm independent
callbacks submit only genuinely passed arms; exact IDs/resources must be read
from actual receipts. Failed/missing/aged-out outcomes never become DONE. LSF
whitespace wrapping is normalized with original captures retained; bounded
cache-lag retries require the eventual actual final state. Exclusive attempt
markers prevent blind duplicate submissions after unknown outcomes.

Planned fresh source/run identity `20261021-brd4-jq1-25-01`; this plan alone is
neither a sealed release, submitted GPU nor a passed numerical test. Actual
immutable hashes, jobs and outcomes are added below and in the ledger/receipt.

## Authenticated source01 CPU result and actual native submissions

Source01 is actually sealed at
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-01/source`,488files,
manifest`2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c`;
archive39,462,299bytes SHA
`5e4bf8601f1b09e244a7763dc6c6d8af5ddc76bfd3490830993d7cfd0b2c0906`.
Source01 includes the uncommitted implementation bytes, not merely its Git ref.

CPU29769654 actually DONE11:30:22Paris and after-CPU29769655 actually DONE
11:30:38. Full CPU gate **419 passed**, no failures/errors/skips; three existing
slow/model checks deliberately deselected. Groups93controller/pocket contracts,
233legacy/JAX,34Torch mapping/config/no-diffusion,59unchanged chemistry. Raw XML,
actual scheduler/LSF and all43proof files downloaded and byte/schema-verified at
`results/hpc/20261021-brd4-jq1-25/cpu01/`. Assessment SHA
`e342bda2bd510fff0aba33192f10dfe4d81ccca53e1b83562c83089ab54095e1`;
archive29,621bytes SHA
`12ffe963f2f5e68989bd80eac6f90aee2e08f77746246d5b6a4631fca2c5a18a`.

Allocated extraction produced19pocket residues, author numbers
81,82,83,85,86,87,88,92,93,94,97,135,136,139,140,144,145,146,149.
Definition SHA`9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43`.
This is actual extraction, not a native predicted-pose validation.

Actual native jobs submitted11:30:38Paris: Protenix validation29769668 and
callback29769669; fresh Boltz pocket-pose preparation29769670 and callback
29769671. Latest observed PEND; GPU jobs request the declared gpul40s resources.
No passed native model gate or long submission is established by this CPU pass.
The controller will submit the two separate Boltz validators only after actual
pose preparation/placement passes, then independent long arms after each actual
native/runtime/storage pass. Current receipt:
`scripts/hpc/receipts/20261021-brd4-jq1-25-01/campaign.json`.
