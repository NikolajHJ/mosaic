# Experiments

The [Protenix restart search](latent-design/protenix-vae/README.md) records the chemistry pilot, matched follow-ups, and interactive reports on this branch.
Start new optimization studies with the [direct relaxed-graph baseline](latent-design/protenix-vae/restart-boltz-brd4-direct.md). The [synthesis-aware BRD4 protocol](latent-design/protenix-vae/restart-synthesis-standard.md) describes the earlier VAE latent searches.
The [direct Protenix BRD4 repair comparison](latent-design/direct-jump/repair-search.md) tests four chemistry weights on paired sparse founders and their descendants.
The [Nesso-1 BRD4 pilot](latent-design/direct-jump/nesso-pilot.md) tests affinity gradients on those same founders and measures the cost per update.
The [700-update BRD4 comparison](latent-design/direct-jump/combined-search.md) calibrates Nesso-1 and Protenix gradients on the same relaxed graph and tests them with repair weight 2.
