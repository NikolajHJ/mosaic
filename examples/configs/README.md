# Legacy configuration paths

Canonical configs live with their [owning experiments](../../experiments/INDEX.md).
The [migration map](../../experiments/MIGRATION.md) lists every exact destination.
These old JSON paths remain as byte-identical declarative compatibility copies
for existing commands, direct path readers and historical tooling.

Edit the canonical configuration for new work and synchronize its compatibility
copy. `py -3.13 scripts/layout/verify_layout.py --root .` rejects differing copies.
An authorized scientific protocol change still needs a new declared config/run
identity and all applicable checks; organization does not grant that authority.
