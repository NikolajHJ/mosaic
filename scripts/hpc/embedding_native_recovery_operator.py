"""Independent allocated recovery of saved native requests; no optimization."""
from embedding_operator import main


if __name__ == "__main__":
    main(namespace="source141-embedding-native-recovery-attempt",
         config_relative="configs/embedding_native_recovery.json",
         experiment_module="latentmol.boltz.embedding_native_recovery_experiment",
         operator_relative="scripts/hpc/embedding_native_recovery_operator.py", gpu_wall="6:00",
         extra_artifacts=("validation/result.json", "validation/parent-authentication.json",
                          "parent-authentication.json", "cuda-ownership-gate.json",
                          "fresh-native/summary.json", "fresh-native/progress.json",
                          "fresh-native/stopped.json", "native_smoke_gate.json"))
