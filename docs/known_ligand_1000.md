# Real-target, known-ligand 1000-update study

Requested 2026-10-16. Status: implementation/readiness in progress; no completed
1000-update result is claimed here. Actual submissions and outcomes belong in
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

Long run **29736126 submitted and RUN**, one exclusive A10 GPU/four CPU slots,
32GB total host-memory request, six-hour bound. It requests 1000 total updates
for each mixture from fresh initial graphs, reusing the exact authenticated
pilot context (`45fa41a40b6bf2e3a548b819b29aa1ad4d0b4b8e61067f3aed877a989d0cbdfa`
summary SHA). A submitted/running job is not a completed result. Numerical
outcomes and final report remain pending.
