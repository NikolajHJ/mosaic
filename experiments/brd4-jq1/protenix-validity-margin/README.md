# Bond-local validity margin

Does weak guidance on uncertain bonds improve retention of valid, connected
molecules without losing useful discrete Protenix contact?

The control and treatment start from the exact saved logits at update 6,800
of the completed seed 17 chain run, just before the archived validity loss
at 6,816. Both run 1,000 fresh updates at LR 0.3, T 1, repair weight 10,
no update noise, frozen Protenix Tiny and model key 17. Structural repair
and the original invalid-source loss are shared. Only the treatment adds
the margin. The input source, model and old result remain immutable.

For an already valid source, screen the eight highest-risk near-boundary
occupied bond rows. Check the alternate categories using the original native
sanitization and schema roundtrip with the other labels fixed. Freeze these
allowed sets for the entire step. Penalize a deficit below 0.2 logit units
in the smaller of the valid-vs-invalid pooled log-mass gap and the best-label
gap. The latter catches an unsafe argmax when safe mass is split. The Huber
hinge caps slope at 0.25 per row. The unaveraged row penalties carry weight
0.001 relative to the repair coefficient, for effective weight 0.01 in this
pilot. Valid alternatives can each satisfy the margin. There is no atom,
charge or occupancy guidance from this term. Local validity does not
guarantee that simultaneous edits remain valid.

Both arms retain strict raw validity, validity exits, distinct valid SMILES,
observed probability-weighted repair costs at T 1 and runtime. A censored
search remains unresolved. Rescore raw hard graphs at local updates 0, 100,
..., 1,000 with the same scorer. Compare valid hard contact on those grids
and report soft contact separately. Shared allocation order can affect runtime.
The exploratory comparison uses one checkpoint and does not certify affinity.

Use the existing [paired experiment launcher](../protenix-random-chem/scripts/launch_score_transfer_audit.py).
The [shared screen](../protenix-random-chem/scripts/protenix_chemistry_screen.py)
keeps the option off by default. The
[margin implementation](../protenix-random-chem/scripts/local_validity_margin.py)
and [allocated controls](../protenix-random-chem/scripts/local_validity_margin_controls.py)
are included in the new release only.

For a fresh, authorized tag and a read-only saved-input packet:

```powershell
py -3 experiments/brd4-jq1/protenix-random-chem/scripts/launch_score_transfer_audit.py --inputs INPUTS.json --tag NEW_TAG --kind local-validity-margin
```

Planned attempt 20261105-local-validity-margin-01 uses one A40 with a 60-minute
limit, two serial matched arms, and a 256 MiB whole-run cap. Numerical controls
and model work execute inside the allocation. No existing experiment ran
during preparation.

DTU accepted and released A40 job 29896035. Metadata startup and dispatch
capacity passed. The first verified observation showed PEND. See
[campaign.json](campaign.json) for the exact input and source receipts.

Allocated CPU job 29896051 passed all five focused controls. CPU attempt
29896037 failed before Python startup because LSF reinterpreted an inline shell
argument. The successful retry used a separate script and the same sealed
control bytes. The GPU pair does not depend on either CPU job.
