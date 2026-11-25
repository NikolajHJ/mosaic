# Experiments

Current navigation, updated 3 November 2026. These are questions and links to
saved evidence, not live scheduler status. Related arms/budgets/seeds share a
home; use configs and short result notes rather than another copied runner.

| Question | Home and entry point | Latest saved evidence / next step |
| --- | --- | --- |
| Can AE/VAE training improve generation while retaining reconstruction? | [VAE training](vae/README.md); `latentmol.training.workflow` | [Canonical-input pilot 29883264](vae/invariant-pilot.md) completed. Exact reconstruction matched epoch 45 at 99.80%, random argmax validity was 75.48% vs. 75.55%, and shuffled reconstruction reached 99.61%. |
| Can we assemble 50 million unique VAE-compatible molecular graphs? | [Large corpus](vae/large-corpus.md) | The combined corpus contains 81,409,264 unique training graphs, including 77,738,566 new PubChem graphs. |
| Does atom-mapped interpolation supervision improve held-out paths? | [Interpolation](vae/interpolation.md); `latentmol.training.interpolation_pilot` | In the [slot-aligned follow-up](vae/interpolation.md#slot-aligned-pilot-results), the capped loss preserved reconstruction and nearly matched prior validity. Both arms had 59/64 valid midpoints. |
| Can frozen scorers improve actual molecules through a latent? | [Latent design](latent-design/README.md); `latentmol.multimodel` | Relaxed50k and [hidden-embedding10k](latent-design/protenix-vae/embedding-notes.md) completed. Mapping55.76% raw-valid, final invalid; mapped/decoded score gap persists. [Gradient comparison](latent-design/protenix-vae/embedding-comparison.md) completed. |
| Which chemistry guidance helps, and which causes shortcuts? | [Chemistry guidance](chemistry-guidance/README.md); shared `latentmol.guidance` | Structural proposals selected for new experiments:123/123 repairs and faster than A*. Selected config saved; loss/running training unchanged. Training effects untested. |
| Which AE/VAE architecture is useful? | [Architecture comparisons](architecture/README.md) | Historical matched comparisons and retained configs; reuse the existing study runner for another arm. |

A home needs only the question, relevant config/command, result links and next
step. Generated output folders need no README. Existing source/configs/results
stay at their linked paths; a note is not authority to restart or adopt a change.
[Chronological ledger](../docs/05_experiment_ledger.md) is available for a deeper
investigation, not mandatory reading for every edit.
