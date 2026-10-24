# Experiment index

Migrated working-source homes, 22 October 2026. Historical runs and sealed releases are preserved; scientific results are not revalidated by this index.

| Experiment | Question | Owned files | Preserved dated attempts |
| --- | --- | ---: | ---: |
| [native-inference/protenix-tiny](native-inference/protenix-tiny/README.md) | Reproduce a bounded native protein forward with the pinned protenix_tiny_default_v0.5.0 model; this is not a ligand graph-gradient acceptance result. | 3 | 0 |
| [graph-validation/stage0](graph-validation/stage0/README.md) | Validate shared fixed-occupancy graph state, native feature injection, projected optimization and frozen Tiny parity before interpreting graph optimization. | 3 | 0 |
| [graph-validation/relaxed-gradients](graph-validation/relaxed-gradients/README.md) | Check native graph gradients and constrained updates through the frozen Protenix-Tiny path, keeping occupancy and geometry scope explicit. | 2 | 0 |
| [graph-controls/carbon-chain-100](graph-controls/carbon-chain-100/README.md) | Measure the first bounded relaxed-graph trajectory from the declared carbon-chain initialization. | 2 | 0 |
| [graph-controls/carbon-chain-500](graph-controls/carbon-chain-500/README.md) | Compare the longer carbon-chain and declared 50 percent initialization-mixture controls without conflating their starting graphs. | 4 | 0 |
| [graph-controls/known-ligand-1000](graph-controls/known-ligand-1000/README.md) | Prepare and validate a known-ligand complex and compare the native and relaxed controls over the declared 1000-update budget. | 8 | 1 |
| [brd4/contact-5000](brd4/contact-5000/README.md) | Optimize the frozen Protenix contact objective against the authenticated BRD4 fixture over 5000 updates, with bounded pilot/native gates. | 2 | 1 |
| [brd4/uniform-5000](brd4/uniform-5000/README.md) | Test BRD4 contact optimization from the declared uniform initialization while retaining the original native and runtime gates. | 1 | 1 |
| [brd4/noisy-uniform-5000](brd4/noisy-uniform-5000/README.md) | Compare a declared noisy-uniform start against the fixed-input BRD4 controls. | 1 | 1 |
| [brd4/canonical-diffuse-5000](brd4/canonical-diffuse-5000/README.md) | Compare the native canonical/diffused initialization under its separately declared context and validation contract. | 1 | 1 |
| [brd4/best-primal-continuation](brd4/best-primal-continuation/README.md) | Continue selected authenticated states under the declared best-primal acceptance protocol while preserving parent state and fixed objective identity. | 5 | 1 |
| [brd4/chemistry-5000](brd4/chemistry-5000/README.md) | Compare a rule-based chemistry-guided molecular trajectory with the frozen native contact objective, retaining invalid graphs and original repair/search semantics. | 8 | 1 |
| [brd4/chemistry-ramp-10000](brd4/chemistry-ramp-10000/README.md) | Test the declared delayed chemistry coefficient schedule across random and mixed starts for 10000 requested updates; changing chemistry totals are not one fixed comparable objective. | 5 | 1 |
| [brd4/chemistry-compatibility](brd4/chemistry-compatibility/README.md) | Probe compatibility of continuous and repair-based chemistry components independently of successful production optimization. | 4 | 1 |
| [brd4/boltz-affinity-10000](brd4/boltz-affinity-10000/README.md) | Optimize the frozen detached-context Boltz-2 affinity branch with its separate raw native gradients, contact/affinity interpretation and independent arms. | 20 | 6 |
| [brd4/boltz-gradient-diagnostic](brd4/boltz-gradient-diagnostic/README.md) | Investigate failed raw-gradient/finite-difference behavior without promoting a diagnostic or changing the historical native acceptance result. | 11 | 4 |
| [brd4/three-model-25](brd4/three-model-25/README.md) | Compare Protenix, Boltz-2 and Boltz-2 plus chemistry from the same declared 25-slot initialization, with independent native readiness and long-run outcomes. | 16 | 6 |
| [brd4-jq1/backtracking-25](brd4-jq1/backtracking-25/README.md) | Compare the three frozen-model branches in the authenticated JQ1-defined BRD4 pocket using the original declared acceptance policy. | 16 | 8 |
| [brd4-jq1/protenix-lr03](brd4-jq1/protenix-lr03/README.md) | Test the explicitly authorized Protenix-only learning rate 0.3 from the original authenticated initialization. | 7 | 2 |
| [brd4-jq1/fixed-steps](brd4-jq1/fixed-steps/README.md) | Apply the declared fixed-rate single-candidate updates, including finite uphill/equal/no-op updates, with split/recovery engineering attempts recorded separately. | 33 | 11 |
| [brd4-jq1/fixed-lrboost](brd4-jq1/fixed-lrboost/README.md) | Compare Protenix LR 3.0 and both Boltz branches LR 0.05 for 10000 fresh updates, chemistry 0.001 held to 5000 then geometric to 50; preserve failed terminal derivatives. | 24 | 3 |
| [brd4-jq1/fixed-lrreset](brd4-jq1/fixed-lrreset/README.md) | Compare fresh Protenix LR 0.3 and both Boltz branches LR 0.005, with chemistry 0.001 held to 5000 then geometric to 100. Existing attempt01 failure and sealed attempt02 remain independent of this migration. | 11 | 2 |
| [reporting/milestones](reporting/milestones/README.md) | Assemble the saved six-milestone evidence without model evaluation, resampling, molecule repair or reinterpretation of failed scientific checks. | 2 | 0 |
| [reporting/brd4-comparison](reporting/brd4-comparison/README.md) | Combine explicitly declared saved BRD4 studies with honest partial/failed/pending states and fixed-versus-changing loss distinctions. | 5 | 1 |

Shared tools: [releases](../scripts/releases/README.md), [evidence](../scripts/evidence/README.md), [graph](../scripts/graph/README.md), [chemistry](../scripts/chemistry/README.md), [campaign25](../scripts/campaign25/README.md), [boltz](../scripts/boltz/README.md), [reporting](../scripts/reporting/README.md), [hpc](../scripts/hpc/README.md).

[Complete migration and compatibility map](MIGRATION.md) · [Policy](README.md) · [Global logbook](../docs/experiment_ledger.md).


## New Protenix random-initialization chemistry study, 23 October 2026

- [brd4-jq1/protenix-random-chem](brd4-jq1/protenix-random-chem/README.md): one new guarded LR0.3/Dirichlet1 seed17/delayed0.001-to100 run, independent of all saved studies.


- [Protenix extended50000](brd4-jq1/protenix-extended-50000/README.md): random50k with stretched chemistry ramp and exact passed plain10k+40k continuation; CPU832 passed and fresh native checks queued.

## New Boltz LR0.0025 comparisons and composite pilot, 23 October 2026

- [Native Boltz30k pair](brd4-jq1/boltz-lr0025/README.md): fresh LR0.0025, plain and stretched-chemistry30k arms.
- [DBMol composite pilot](brd4-jq1/boltz-dbmol/README.md): independent LR0.0025/10k binding-only pilot, conditional on component and total native gradient gates.


## Chemistry search acceleration

[Chemistry preparation speed study](brd4-jq1/chemistry-search-speed/README.md) compares exact native-fact caching, invariant cost reuse and ordered candidate parallelism on saved N25 inputs, independently of current runs.
