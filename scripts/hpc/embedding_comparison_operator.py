"""Bounded independent source138 LSF stages; old VAE/mapping campaigns are read-only assets."""

import os
import sys


CPU_KERNEL_PROFILE = "mkl_compatible_four_threads_v1"


def configure_cpu_environment(argv, environ):
    """Return a copy with the allocated CPU execution profile, without side effects.

    The shared CLI parser remains authoritative. We inspect only its action and
    final --stage argument; both standard argparse spellings are supported.
    boltz_run.sh already supplies four MKL/OMP threads for this allocation. Only
    the demonstrated MKL branch selector changes, before any PyTorch import.
    submit/collect and every GPU invocation retain their incoming environment.
    """
    result = dict(environ)
    if not argv or argv[0] != "verify-run":
        return result
    stage, index = None, 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            break
        if argument == "--stage":
            index += 1
            stage = argv[index] if index < len(argv) else None
        elif argument.startswith("--stage="):
            stage = argument.split("=", 1)[1]
        index += 1
    if stage == "cpu":
        result["MKL_CBWR"] = "COMPATIBLE"
    return result


if __name__ == "__main__":
    # os.execv in the shared operator inherits this environment, applying the
    # same kernel recipe to full pytest, checkpoint parity and start preparation.
    os.environ.update(configure_cpu_environment(sys.argv[1:], os.environ))
    from embedding_operator import main

    main(namespace="source138-epoch36-embedding-comparison-attempt",
         config_relative="configs/epoch36_embedding_comparison.json",
         experiment_module="latentmol.boltz.embedding_comparison_experiment",
         operator_relative="scripts/hpc/embedding_comparison_operator.py", gpu_wall="24:00",
         extra_artifacts=("starts.json", "precision-correction.json", "start-gates.json", "case-summary.json", "fresh-native/summary.json",
            "native-followup-quota.json", "fresh-native/progress.json", "fresh-native/stopped.json",
            *[f"case-{index:03d}/{name}" for index in range(1, 4) for name in
              ("native-oracle.json", "relaxed-start-gate.json", "learned-start-gate.json",
               "production-binding.json")]))
