# Frozen-model latent design

Question: does optimizing through frozen models change hard molecules usefully?

- [Direct graph jumps on BRD4](direct-jump/experiment.md) compares learned moves, same-size random moves and restarts on shared held-out founders with fresh Protenix and Boltz2 scores.
- [Aromatic-ring formation and frequency](aromatic-rings.md) across VAE draws, BRD4 search trajectories, fresh rankings and experimental reference ligands.
- [Boltz-2 ipSAE gradients](protenix-vae/ipsae.md): matched 500-update
  fixed-pose pilot and isolated confidence-gradient comparison, submitted.
- [Flow-inspired hallucination pilot](protenix-vae/flow-pilot.md): saved
  early-stop replay, learned latent proposals, and a matched DeFoG repair test.
  [Follow-up](protenix-vae/flow-followup.md) compares equal-budget portfolios
  across independent cohorts and tests less disruptive DeFoG schedules.
  [Tuned comparison](protenix-vae/flow-tuned.md) picks a latent SGD rate on
  separate starts before retraining and evaluating the learned jump.
- [VAE + DBMol loss](protenix-vae/dbmol-notes.md): fresh random-latent50k campaign,
  cooling/noise and structural repair; A10029877824 queued with automatic measured
  continuation. This is the existing full native composite objective, not the
  hidden-embedding prototype.

- [Soft molecules in 3D](geometry-inspection.md): saved probabilities overlaid
  on their authenticated fixed Boltz scaffold; no new model evaluations.
- [Soft-input Boltz-2 diffusion](soft-diffusion.md): separate40-step coordinate
  prediction probe DONE29876935; soft features retained, new pose visualized.
  The [five-molecule gallery](soft-diffusion.md#five-molecule-gallery-6-november-2026)
  adds archived experiment inputs and linked probability matrices.
- [VAE + Protenix](protenix-vae/README.md): frozen epoch45 decoder, random latent,
  structural repair and requested 50k schedules. Both pilots and full 50k completed;
  final valid 25-atom graph, 50 distinct valid trajectory identities, zero censored
  repairs. Contact proxy rose from 0.151 to 0.246; hard-graph binding unqualified.
- [One optional atom](protenix-vae/one-atom.md) tests latent occupancy against
  native present and deleted topologies and records the VAE's raw hard decode.
- [Protenix count budget](protenix-vae/atom-budget.md) compares the count penalty
  on and off over matched native topologies spanning 24–28 heavy atoms.
- [Raw VAE occupancy](protenix-vae/raw-occupancy.md) tests whether native
  score gradients change the atom count of the unmodified decoder output.
- [Latent landscape](protenix-vae/landscape.md) measures decoder sharpness,
  gradient coherence and whether equal-length descent steps beat random steps.
- Shared runner: `python -m latentmol.multimodel plan --config configs/multimodel_boltz.yaml`.
- Configs: [Boltz baseline](../../configs/multimodel_boltz.yaml), [ensemble](../../configs/multimodel_ensemble.yaml).
- Entry-point guide: [workflow commands](../../docs/workflows.md).
- Objective reference: [DBMol Boltz loss](../../docs/77_dbmol_boltz_loss_implementation.md).
- Mapping comparison: [study78](../../docs/78_embedding_latent_comparison.md).
- Existing long campaign and V100 handoff: [study81](../../docs/81_guarded_embedding_long_run.md), [study87](../../docs/87_v100_continuation_handoff.md).

The saved campaign observed lower learned loss without establishing a new hard
molecule/binding gain. Existing `scripts/embedding_*_campaign.py` are controllers
for their recorded plans, not the default runner for a new small study. Preserve
their states and source bindings. Evaluate hard readouts alongside surrogate loss.
