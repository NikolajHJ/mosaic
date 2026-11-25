# Datasets

Dataset preparation, splits, caches and audits.

Graph schema and categorical sampling remain shared `representation.py` and `sampling.py` primitives.

Core modules include `audit.py`, `data.py`, and `ordering.py`.

`bulk_ingest` normalizes raw source shards with the shared graph schema.
`bulk_finalize` deduplicates their full canonical SMILES across sources and
the existing MOSES–ChEMBL training split. It preserves both held-out sets
and excludes new molecules on reserved ChEMBL holdout scaffolds. It stages
hash buckets on local scratch, then publishes compressed TSV membership
and a checksummed manifest on the output filesystem. The final TSV fields
are canonical SMILES and source. `bulk_cache_build` can encode that membership
to a quota-bounded packed graph cache. `bulk_cache.SparseGraphDataset` provides
lazy tensor batches through the shared graph contract.

`bulk_view.BalancedSparseCorpus` opens indexed heavy-atom-balanced splits
over the packed cache. The indices preserve the existing holdouts and
allow independent training membership without duplicating graph records.

Keep reusable calculations here. Experiment setup and results belong in the
[experiment homes](../../../experiments/README.md). Old flat imports forward
to these same modules; edit the implementation here.

`data.py` here is the grouped API alias: its implementation remains at
`latentmol.data` because existing cache provenance contracts bind that file.

