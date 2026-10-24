# brd4-jq1 experiments

Scientific family index. Each study retains its own question, protocol, configuration and attempt history.

- [JQ1-pocket three-arm comparison](backtracking-25/README.md): Compare the three frozen-model branches in the authenticated JQ1-defined BRD4 pocket using the original declared acceptance policy.
- [JQ1 Protenix learning-rate 0.3 study](protenix-lr03/README.md): Test the explicitly authorized Protenix-only learning rate 0.3 from the original authenticated initialization.
- [JQ1 fixed-step comparison](fixed-steps/README.md): Apply the declared fixed-rate single-candidate updates, including finite uphill/equal/no-op updates, with split/recovery engineering attempts recorded separately.
- [JQ1 higher fixed learning rates](fixed-lrboost/README.md): Compare Protenix LR 3.0 and both Boltz branches LR 0.05 for 10000 fresh updates, chemistry 0.001 held to 5000 then geometric to 50; preserve failed terminal derivatives.
- [JQ1 lower-rate restart, chemistry endpoint 100](fixed-lrreset/README.md): Compare fresh Protenix LR 0.3 and both Boltz branches LR 0.005, with chemistry 0.001 held to 5000 then geometric to 100. Existing attempt01 failure and sealed attempt02 remain independent of this migration.

See the [global experiment index](../INDEX.md) and [agent logbook](../../docs/experiment_ledger.md).


Additional23October user-authorized [random Protenix plus delayed chemistry](protenix-random-chem/README.md):
freshLR0.3/Dirichlet1 seed17/chemistry0.001-to100, independent of saved studies.

[Protenix extended50k and exact continuation](protenix-extended-50000/README.md):
user-authorized random50000 stretched chemistry ramp and original passed10k+40000.

[Boltz LR0.0025 comparisons](boltz-lr0025/README.md) and [independent DBMol composite pilot](boltz-dbmol/README.md): fresh30k pair with stretchedchemistry and a separately gated10k binding pilot.


## Chemistry preparation performance

The [separate search speed study](chemistry-search-speed/README.md) tests exact detached-search acceleration on saved pilot graphs. It does not change existing optimizations or automatically promote a loss implementation.
