"""Bounded independent source138 LSF stages; old VAE/mapping campaigns are read-only assets."""
from embedding_operator import main


if __name__ == "__main__":
    main(namespace="source138-epoch36-embedding-comparison-attempt",
         config_relative="configs/epoch36_embedding_comparison.json",
         experiment_module="latentmol.boltz.embedding_comparison_experiment",
         operator_relative="scripts/hpc/embedding_comparison_operator.py", gpu_wall="24:00",
         extra_artifacts=("starts.json", "start-gates.json", "case-summary.json", "fresh-native/summary.json",
            "native-followup-quota.json", "fresh-native/progress.json", "fresh-native/stopped.json",
            *[f"case-{index:03d}/{name}" for index in range(1, 4) for name in
              ("native-oracle.json", "relaxed-start-gate.json", "learned-start-gate.json",
               "production-binding.json")]))
