# DBMol migration to Mosaic

The active graph-VAE trainer lives in [training/vae](../training/vae/README.md) on `codex/vae-training`. The DBMol commit history and research snapshots are still addressable on the Mosaic fork.

| Material | Location |
| --- | --- |
| Main DBMol archive | [dbmol/main](https://github.com/NikolajHJ/mosaic/tree/dbmol/main) |
| DBMol working-tree source, project notes, timeline, and other experiments | [dbmol/flow-guided-hallucination-worktree](https://github.com/NikolajHJ/mosaic/tree/dbmol/flow-guided-hallucination-worktree) |
| Shareable 24-milestone timeline | [VAE training timeline PDF](vae-training-timeline.pdf) |
| DiffSBDD and Protenix linked worktree histories | [dbmol/diffsbdd-vae-alternation-hpc](https://github.com/NikolajHJ/mosaic/tree/dbmol/diffsbdd-vae-alternation-hpc) and [codex/protenix-restart-search](https://github.com/NikolajHJ/mosaic/tree/codex/protenix-restart-search) |
| Portable Git history, checkout snapshots, checkpoints, PDF, logs, and results | `C:/Users/nikol/mosaic-artifacts/DBMol_exploratory-2026-11-26/` on the research laptop |
| Current VAE training segment, job 30387778 | `~/dbmol-control/vae-e48-attention-20261126a/segment/` on DTU |
| Original pinned checkpoint and provenance for the attention branch | `~/dbmol-control/vae-e48-attention-20261125a/source/` on DTU |
| Training corpus, frozen teacher and environment | Paths in the [epoch 48 recipe](../training/vae/experiments/vae/epoch48-seven-round-generated-roundtrip.json) and [DTU launcher](../training/vae/experiments/vae/run_checkpoint_comparison.sh) |

The local archive contains a Git bundle and separate file snapshots. Use the file snapshots for uncommitted or ignored content, including output directories. The recorded worktree paths and commits are in `worktrees.txt`. Use `git clone <archive>/dbmol-history.bundle <new-path>` to restore Git history and `artifact-manifest.jsonl` to verify file hashes. Restore archival files by copying them into a disposable checkout. Python environments and interpreter/test caches can be recreated from the package dependencies.

The running DTU job uses its existing sealed release. The Mosaic resume entry was checked on CPU with the downloaded, SHA-256-verified production checkpoint at update **632116**. It restored the 5,058,652-parameter architecture, all 421 saved AdamW states, six RNG streams and the cursor at row 13,588,320. Fixed decoder logits matched exactly and the latent gradient was finite. See `dtu-snapshots/job-30387778/mosaic-resume-replay.json` in the archive. The first allocated-GPU resume and checkpoint write from Mosaic remain to be verified. Keep the DTU directories above available as inputs when retiring the local DBMol checkout.
