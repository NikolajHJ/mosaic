# Real-target, known-ligand 1000-update study

Requested and executed 2026-10-16. Status: all three 1000-update trajectories,
matched hard controls and enriched report completed; downloaded evidence
verified and final browser QA passed. Actual submissions and outcomes belong in
[the experiment ledger](experiment_ledger.md).

Use the T4 lysozyme L99A construct from [PDB 181L](https://www.rcsb.org/structure/181L)
and its bound benzene, [CCD BNZ](https://www.rcsb.org/ligand/BNZ), as the known
ligand reference. The full 164-residue deposited sequence includes C54T/C97A
as well as L99A. Pinned PDB bytes SHA-256:
`77018feaaa65bb22dea47c784e8c059b0ccc09cd6dc7442b79cce83f3170985f`.

Pivotal target choice: this small neutral reference fits the current fixed
occupancy/zero-charge graph contract. The earlier Stage 1 plan names BRD4/JQ1,
but [the deposited CCD JQ1](https://www.rcsb.org/ligand/JQ1) is charged (+1).
Do not silently neutralize it or claim it is supported. This user-requested
known-ligand comparison is a separate scoped experiment, not completion or
replacement of the pending Stage 1 PAE/BRD4 gates.

Start at update zero from **six neutral carbon atoms in a single-bond chain**,
with 15%, 30% and 50% uniform mixtures, seed 17, highest FP32 matmul and
**1000 total accepted updates per start**. The change from ten to six occupied
slots matches benzene's heavy-atom count. The 100/500-update synthetic-peptide
studies remain separate; their numerical losses are not comparable to this
changed target, size and contact selection.

Allocated preprocessing authenticates the PDB, keeps the complete SEQRES
sequence, checks author residue/sequence mapping, and selects protein residues
with a positive-occupancy heavy atom within 5 Å of BNZ A 400, using blank/A
alternate locations. Those fixed residue IDs define the same DBMol-inspired
pocket/anchor contact NLL used previously (native complete bins below 8 Å).
The crystal coordinates select residue IDs only. **The model receives the
real construct sequence without a structural template or MSA; it does not
receive the experimental protein pose, benzene pose, waters, ions or HED.**
This is a sequence-context contact-score comparison, not docking or a physical
relaxation of a ligand in a fixed crystal structure.

Evaluate hard benzene in the exact same prepared geometry-off context, with
the same occupancy, model, RNG, contact mapping and scalar objective as the
chain. Keep its graph, native logits and scalar loss. The reference is held
fixed and never used as an optimization target. Report its loss alongside
the trajectories and detached molecular identity comparisons. Native features
observe element identity and bond existence, not bond order, chirality or
formal-charge changes; they cannot establish aromaticity recovery or affinity.

Pivotal storage choice: a new **compact artifact policy** keeps every saved
graph, raw atom/bond gradient, loss, RNG, proposal and readout. Initial/final
evaluations and derivative/null probes retain raw native distograms. It omits
persisted trunk tensors and repeated interior-step distograms; all returned
tensors are still checked for finiteness before acceptance. Default full
artifacts and all earlier evidence remain unchanged. A full-versus-compact
unit comparison must establish identical graphs, gradients and loss history.
This is a serialization change, not a model/objective/tolerance change.

Run allocated CPU unit/preparation gates, then a bounded three-update GPU
pilot for all three starts. Authenticate its context and measure output size,
runtime and memory before a fresh 1000-update replay using that exact context.
Preserve the existing derivative, null, AD/primal, measured descent and frozen
identity gates. Stop on failure or insufficient quota. The compact campaign
requires 5 GiB free at launch: 4 GiB reserve plus a provisional 1 GiB new-data
budget; the pilot must establish that the actual projected budget fits before
the long run. No quota threshold is relaxed for the old full-artifact studies.
Render on an allocated CPU after model completion, download and verify all
new evidence, then publish actual outcomes here and in the ledger.

Readiness **29736014 DONE**: 146 selected CPU unit checks passed on allocated
HPC in 53.16 s (three model items deselected, two existing dtype warnings).
Full-versus-compact saved states/gradients matched and omitted-tensor finite
checks remained enforced. Allocated preprocessing produced fixture SHA-256
`6dae87554347d1b14764eeb6a3d10ff3488a145c6db142534a4a3c54a4564d8b`.
Selected author residues are 78, 84, 87, 88, 91, 99, 102, 103, 111, 118, 121,
153. This exact generated fixture is copied into the repository. GPU pilot
**29736017 submitted**, with numerical outcomes pending.

Pilot **29736017 DONE** passed all three mixtures and all native gates. The
first assessment **29736081 EXIT** stopped at its storage gate: 5.00 GiB was
free versus **5.676065794 GiB required**, including 4 GiB reserve, ten-trial
worst-case trajectory output, report and evidence archive. No long run was
submitted from that stopped assessment; numerical tolerances are unchanged.

Pivotal recovery: reverify the already downloaded 500-update archive SHA and
its 4619-file receipt, verify the same remote archive plus preserved raw-run
summary identities on allocated CPU, then remove **only** that redundant
2,020,603,152-byte HPC archive. Its exact verified local archive/extracted
evidence and every raw HPC run/source remain preserved. A new assessment run
must pass the **same** projected resource threshold before submission. The
new narrow cleanup does not authorize changes to any other agent's files.

Recovery and replacement assessment **29736114 DONE**: remote archive hash
matched the reverified local copy, the single redundant archive was removed,
and the same 5.676065794 GiB resource gate passed. The pilot occupies
347,274,779 bytes; the conservative new-run bound is 582,461,779 bytes, report
reserve 134,217,728 bytes and combined archive bound 1,082,982,436 bytes.
Measured 3–4 s/update projects about 10000 s for 3000 updates, excluding the
fresh process's compilation. The known-ligand pilot report rendered on allocated
CPU and browser QA confirmed the reference drawing/score, all three cases,
six slots, real target/contact labels and no JavaScript errors.

Long run **29736126 DONE**, one exclusive A10 GPU/four CPU slots,
32GB total host-memory request, six-hour bound. It requests 1000 total updates
for each mixture from fresh initial graphs, reusing the exact authenticated
pilot context (`45fa41a40b6bf2e3a548b819b29aa1ad4d0b4b8e61067f3aed877a989d0cbdfa`
summary SHA). The completed numerical outcomes and final report are recorded
below; earlier submission receipts remain preserved in the ledger.

Final comparison decision: keep soft trajectory scores explicitly separate
from hard molecular scores. A separate allocated frozen-model control pass
scores the initial hard chain, hard benzene and the raw argmax endpoints using
the authenticated completed study's exact context/model/key/objective.
Identical hard graphs reuse a score by logical array digest. Invalid endpoints
are scored and labelled without repair; this does not confer chemical validity.
The existing `2e-5` primal parity gate applies to replayed hard benzene. Both
frozen digests must remain unchanged. This performs no extra optimization and
does not alter the 1000-update evidence. An allocated report addendum copies
the completed figures/states and adds a matched hard-control table.

## Completed native outputs

The user explicitly confirmed that chemically invalid outputs are acceptable
for this phase and requested plots of exactly what is produced. The report
draws the raw categorical argmax from each saved probability state, with
probability-dependent circle/bond sizes and fixed element/slot labels. It
also exposes the actual probabilities and scalar losses. Sanitization is an
annotation only: it does not filter states, select fragments, repair bonds,
substitute identities or affect optimization/depiction.

All three cases passed 1000 accepted updates, 1001 saved states each, every
native numerical gate and frozen identity checks. Native job 29736126 ran
11:57:27–12:42:46 Europe/Paris on one allocated A10.

| Uniform mixture | Initial → final relaxed primal loss | Sanitized readouts | Final raw graph |
| --- | --- | --- | --- |
| 15% | 1.993563652 → 1.857835531 | 1001/1001 | `CCCCCC`; hard graph unchanged |
| 30% | 1.942351460 → 1.852983117 | 1001/1001 | `CCCCCC`; hard graph unchanged |
| 50% | 1.988362908 → 1.816638827 | 872/1001 | C1–C2–C3–C4–F5–C6; raw F valence 2 |

The 50% raw hard identity switches at **update 872**, with the same chain
bonds. Its invalid fluorine is drawn as produced. Hard-graph Morgan similarity
to benzene is 0 for the valid hexane endpoints and unavailable for the invalid
endpoint. None of these statements asserts biological ligand recovery.

Matched hard-control job **29736511 DONE** passed reference replay/frozen
identity checks. Hard benzene loss is **2.097959280**; hard hexane and the
15%/30% endpoints are **2.013308287**; the invalid 50% endpoint is
**1.870431781**. These are separate one-hot input controls. The trajectory
curves show the relaxed distributions' losses and must not be interpreted as
matched hard-molecule improvements. All scores are the same contact proxy,
not binding affinity or a chemistry constraint.

Base report **29736412 DONE**, hard-control augmentation **29736521 DONE**,
and full evidence retry **29736523 DONE**. The original numerical arrays and
base report remain preserved; augmentation copies them and adds the hard
comparison table. The [enriched offline report](../results/hpc/20261016-known-ligand-1000/download-03/report/report.html)
contains all 3003 saved states, 0..1000 playback, raw molecules/probabilities,
loss curves, heatmaps and the known reference.

## Delivery and verification

Evidence job **29736523 DONE** produced a **795,141,544-byte** archive, SHA-256
`cfe36ec2176432a26b9923cda0526db1cf5a3c657219599e419633995574b96f`.
Local Windows Python 3.13 stdlib I/O verified the archive and **15,775 declared
files** at 2026-10-16 11:16:06 UTC. The completion receipt is
`results/hpc/20261016-known-ligand-1000/download-03/completion.json`.
No numerical model or chemistry work was run locally.

Enriched report **29736521 DONE** has HTML SHA-256
`2603881fb382b9a17f85b848d47525d6828321680937a12d28fefa5b7bee78d5`;
the downloaded copy matches. Browser QA confirmed all three 1000-update
cases, the known reference and matched hard-control table. Selecting 50%
and update 1000 displays C1–C2–C3–C4–F5–C6, P(F5)=0.43394 and primal loss
1.8166388, with the raw graph retained. Element labels remain legible, the
probability size cues render correctly and no JavaScript errors were logged.

Final captured HPC quota was **94.65/100.00 GiB** and **283,729/20,000,000
files**, leaving about **5.35 GiB** and preserving the 4 GiB reserve. Every
raw numerical run, failed gate and immutable source remains retained; the
only recovery deletion was the separately authenticated redundant older
500-update transfer archive described above.
