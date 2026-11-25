"""Fixed validation reconstruction and independent prior draws for compact VAE training."""

from collections import Counter

from latentmol.chemistry_early_health import preserve_observation_state
from latentmol.datasets.data import collate_graphs
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.evaluation.compact_prior_panel import CompactPriorPanel
from latentmol.evaluation.evaluate import reconstruction_metrics
from latentmol.sampling import subset_indices


class CompactTrainingPanel:
    """Reuse the same validation graphs and standard-normal latents after every restart."""

    def __init__(self, val_data, training_config, *, seed, validation_count=256, prior_count=256, batch_size=16):
        if (val_data.split != "val" or training_config.mode != "vae" or
                training_config.loss.atom_order != "graph" or training_config.model.encoder_pooling != "canonical"):
            raise ValueError("Training panel requires a validation split and a canonical graph-matched VAE")
        if type(validation_count) is not int or not 1 <= validation_count <= len(val_data):
            raise ValueError("Validation panel size must fit the available held-out split")
        if type(seed) is not int or not 0 <= seed < 2**63 - 1:
            raise ValueError("Panel requires a nonnegative signed seed with room for its prior seed")
        self.prior = CompactPriorPanel(training_config.model.latent, seed=seed + 1,
                                       count=prior_count, batch_size=batch_size)
        indices = subset_indices(len(val_data), validation_count, seed)
        checked = verify_canonical_training_cache(val_data, indices)
        self.batches = [collate_graphs([val_data[i] for i in indices[start:start + batch_size]])
                        for start in range(0, len(indices), batch_size)]
        counts = Counter(n for batch in self.batches for n in batch.atoms.ne(0).sum(-1).tolist())
        self.spec = val_data.spec
        self.loss = training_config.loss
        self.metadata = {
            "split": "val", "dataset_sha256": val_data.dataset_sha256, "validation_seed": seed,
            "validation_indices": indices, "validation_count": validation_count,
            "canonical_rows_checked": checked,
            "validation_target_size_histogram": {str(k): v for k, v in sorted(counts.items())},
            "prior": self.prior.metadata,
        }

    def observe(self, model):
        """Use posterior means for reconstruction; inspect the raw prior graphs independently."""
        if model.spec != self.spec:
            raise ValueError("Model and validation panel graph specifications disagree")
        with preserve_observation_state(model, {}):
            reconstruction, _ = reconstruction_metrics(
                model, self.batches, self.loss, next(model.parameters()).device,
                chemistry_limit=self.metadata["validation_count"], training_mode="vae",
                from_canonical_cache=True,
            )
        return {"reconstruction": reconstruction, "prior": self.prior.observe(model)}
