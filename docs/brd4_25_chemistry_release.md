# Approved 25-atom Boltz-2 plus repair submission

21 October 2026. The user accepted the measured repair overhead and explicitly
requested submission. This releases the earlier runtime-review hold. All original
native, derivative, parity, finite-value, frozen-identity, runtime and quota gates
remain required. Temperature sharpening was discussed, not implemented.

The user also explicitly confirmed that gradients must **not** flow through
diffusion back to the molecule. That is the existing implementation: native pose
prediction runs once during preparation; saved coordinates are detached and the
affinity heads receive `coordinates.detach()`. Optimization differentiates the
affinity trunk/input features and existing repair loss with respect to atom/bond
logits, at the fixed pose. Protenix's separate path uses trunk/distogram contact,
without diffusion. These are partial derivatives, not a full denoising gradient.
No model weights, pose, temperature or objective were changed for this approval.

## Scientific configuration retained

Scientific source04 is unchanged: 461 files at
`/work3/s234818/mosaic/releases/20261021-brd4-25-04/source`, manifest
`e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79`.
Its actual scientific CPU347-test pass and all original proofs remain required.
The full127-residue BRD4-BD1 target, 25 occupied neutral slots, 50% chain/uniform
initialization, LR0.03, key17, ten-trial best_primal and 10,000 total updates are
unchanged. Affinity coefficient1; composition0; existing repair coefficient0.001
through applied5000, geometric increase to50 at10000. T1, zero recycles and no
MW correction. Each arm is independent; raw invalid output remains visible.

## Launcher recovery and allocated contracts

Preparation29769178 actually DONE10:30:48Paris. Its old callback29769179 saw a
stale RUN from the scheduler cache immediately after its ended dependency fired,
and stopped before submitting either Boltz validator. The original captured RUN
and failed callback remain preserved. The new controller waits up to seven
queries ten seconds apart for an **actual** final state and LSF completion;
it never promotes a dependency, timeout, missing record or RUN to DONE.

Launcher-only source07 authenticated five files at
`/work3/s234818/mosaic/releases/20261021-brd4-25-release-07/source`, manifest
`80274d8d5f066af3e364f0356182ab0aa124302613800aa8b68fe3ed3cdec964`;
archive10,105bytes SHA
`61d23c4d5f6c7e34b59cdfec60947c1a001c838994fd175d5dc07a5a44c8127b`.
It authenticates source04 scientific CPU and source07 launcher CPU independently,
retains the existing prepared context and executes the original source04 driver.
The new launcher permits the user-approved chemistry arm, accounts for all ready
arms in shared quota plus4GiB reserve, and retains exclusive per-arm submission
markers/unknown-outcome protection. Future long jobs request gpul40s, one exclusive
L40S GPU, four CPU slots,8GiB/slot,23h, without inter-arm dependencies.

Tools CPU29769499 actually DONE10:54:45Paris: **24 passed**, no failures/errors/
skips. Raw XML and all byte proofs were authenticated. Bootstrap29769500 actually
DONE10:55:02 and authenticated actual preparation success before issuing jobs:

| Actual job | Scope | Latest observed status |
| --- | --- | --- |
| 29769502 | Boltz-only native validation | DONE, persisted passed |
| 29769503 | Boltz-only readiness callback | DONE, actual long ID29769525 |
| 29769504 | Boltz plus repair native validation and both pilots | DONE, persisted passed |
| 29769505 | Chemistry readiness callback and conditional submission | DONE, actual long ID29769540 |

Both native validations and callbacks actually finished successfully. Original
hard-input parity, raw atom/typed-bond finite differences, frozen identities and
the required pilots passed. Chemistry additionally passed the nonzero invalid
graph assay and both low/high coupled checks and ten-step pilots. Actual
`long-submissions.json` and each `run-*/submission.json` establish independent
10,000-update submissions: Boltz-only29769525 submitted11:02:11Paris and RUN
since11:04:20; Boltz+repair29769540 submitted11:04:34 and PEND for a physical
GPU at the capture. Each requests one exclusive L40S, four slots,8GiB/slot,
23h, without an inter-arm dependency. These are submissions, not completed runs.

The obsolete Protenix callback29769177 was verified PEND/unstarted and intentionally
cancelled, actual EXIT10:50:49, then replaced by29769501. Protenix validation
29769176 itself ran unchanged10:48:00–10:51:47 and EXIT at the original hard-input
parity threshold2e-5 (maximum reported discrepancy~1.564e-4). This remains a failed
Protenix readiness and prevents its long submission. It does not block the two
independent Boltz validations. No threshold was relaxed or failed arm omitted.

Source05 tools were sealed/uploaded but not staged after static detection of an
old callback-directory collision. Source06 staged five files, but Bash rejected
Windows CRLF before any job/test submission. Both releases are preserved; they
are launcher preparation failures, not model failures. Source07 uses LF wrappers,
passed Bash syntax and the allocated24 contracts. Scientific source04 never changed.

## Evidence and follow-up

All56 CPU/controller/XML/source/preparation/cancellation/Protenix-failure artifacts
were safely downloaded and byte-verified at
`results/hpc/20261021-brd4-25/release07/evidence/`; archive153,209bytes SHA
`055c6df3db215eed088da083f07c5ddfbf4473b1527ffc01892b062bed49aded`, manifest
`fe7ffeb2cb3e1e4d35d3fdc9450e8c0cdc3cb4a22458b694bf121f95e64e689d`.
Receipt: `scripts/hpc/receipts/20261021-brd4-25-release-07/campaign.json`.
No local numerical test/model/chemistry work ran. The existing heartbeat follows
actual native results and full chemistry submission; its submission endpoint is
now satisfied and it is paused. Historical six-slot studies, failed random gradients, reports/servers
and unrelated user PDF/OneNote changes remain unchanged.

The additional224 native/submission proof files were downloaded and safely
byte-verified under `results/hpc/20261021-brd4-25/submission02/`. Archive
1,985,802bytes SHA`9f4061050e70e55160e61558b015c578dcf73e0da890084847808951eea38a8c`;
manifest`a21a5407415f169b251d3f86efb768e254881f851c20496b8893f276e678abdf`.
Both arms share new25-slot context tensor digest
`df93ef76cc26d5888532dc2c36688dbbff4040375f5357a784b8b6a656ec2f34`,
context.pt SHA`099fdac537a08f65a14cb60463de24fa0c4ffd6541a142beaaee6e974977fa08`,
initialization SHA`90553f0e28762fe81fa0c40f840e1c2e4884a0f79a1bbf48553132e3c0e27391`
and the unchanged Boltz model digest. Boltz pilot mean1.74322s/update;
chemistry early/high means2.02183/2.25575s/update. Original conservative runtime
projections27,948.33s/35,636.30s both passed82,800s. These short-pilot estimates
do not guarantee a runtime for all later graphs. Campaign quota was checked
before each actual submission, including both ready arms plus4GiB reserve.
Capture01 stopped on an LSF line wrap within a field name; preserve its partial
directory. Capture02 strictly normalizes whitespace for exact job/status/queue
fields and retains original raw output. No scientific setting or gate changed.

## Proposed next pocket-targeted comparison, not applied to the current jobs

The user asked for a real, established binding pocket on21October. Recommend
the **BRD4-BD1 acetyl-lysine/JQ1 pocket** in the same deposited3MXF construct.
[RCSB3MXF](https://www.rcsb.org/structure/3MXF) is an experimental1.60A JQ1
co-crystal. [SGC's primary probe record](https://www.thesgc.org/chemical-probes/jq1)
reports BRD4(N) JQ1 Kd49.0+/-2.4nM by ITC. An experimental
[mutational affinity map](https://pubmed.ncbi.nlm.nih.gov/24497639/) supports
Trp81/Tyr97/Asn140/Met149 as important binding residues. JQ1 is a BET probe,
so this reference does not establish BRD4 selectivity or activity of generated
molecules. Measured JQ1 affinity is not a calibration of our relaxed-graph score.

Static mapping through `fixture_25.json` (author residues42..168) gives:

| Structural residue | Supplied sequence position, one-based |
| --- | --- |
| Trp81 / Pro82 / Phe83, WPF shelf | 40 / 41 / 42 |
| Tyr97 | 56 |
| Asn140 | 99 |
| Met149 | 108 |

These are sequence positions, **not native token IDs**. Validate their native
protein/crop mapping before use; residue140 cannot be used as an index into
the127-residue input. A full spatial pocket should be extracted on allocated
HPC from protein heavy atoms within a declared6A of crystallographic JQ1 heavy
atoms, with author/sequence/native mappings and source hash pinned. This radius
is a proposed operational definition, not a published optimal threshold. Exclude
solvents/ions from the reference ligand selection and do not demand contact to
every pocket residue from each25-slot design.

Protenix currently uses whole-target `NativeContactLoss`; the existing
`DBMolContactLoss` implements an explicit native-token pocket contact surrogate
without any AE/VAE or latent pipeline. A pocket-targeted objective needs a
declared new configuration and original native checks; it does not resolve the
current independent Protenix hard-input parity failure by itself.

Boltz's [pinned native input documentation](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/docs/prediction.md#constraints)
supports pocket contacts using one-based input sequence positions. Condition
fresh pose preparation on the selected pocket, verify placement there on HPC,
then detach/freeze coordinates for affinity optimization. The affinity head
remains a whole-complex scalar at that pose; there is no per-pocket affinity
switch in the current adapter. Restricting protein context alone is not proof
that the ligand occupies the intended pocket. No diffusion gradient is needed.

Current source04 fixture/input explicitly has no pocket constraint and current
Boltz pose location has not been geometrically audited against JQ1. Preserve
29769525/29769540 as their original controls. No pocket constraint, structural
template, new objective or pocket-specific job was implemented/submitted by this
research request; a subsequent version would require fresh context and full
validation. First audit the existing saved pose to establish whether it already
occupies the intended site before spending another long allocation.
