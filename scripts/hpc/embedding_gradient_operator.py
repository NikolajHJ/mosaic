"""Independent one-hour GPU diagnosis; existing comparison and VAE are read-only."""
from embedding_operator import main


if __name__ == "__main__":
    main(namespace="source139-epoch36-embedding-gradient-diagnostic-attempt",
         config_relative="configs/epoch36_embedding_gradient_diagnostic.json",
         experiment_module="latentmol.boltz.embedding_gradient_diagnostic",
         operator_relative="scripts/hpc/embedding_gradient_operator.py", gpu_wall="1:00",
         extra_artifacts=("starts.json", "cpu-decoder-probe.json", "diagnostic.json", "native-oracle.json"))
