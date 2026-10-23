"""Allocated CUDA matched-search optimizer fixture before the real VAE FD gate.

Failure/absence stops the full allocation. The tiny fixture proves engineering
behavior; the full controller separately checks the actual pretrained model.
"""
import os
import pytest
import torch


def main():
    if not os.environ.get("LSB_JOBID") or not torch.cuda.is_available():
        raise RuntimeError("Full comparison requires allocated working CUDA")
    # Execute in this same process. Do not retain one CUDA process while
    # spawning another on an exclusive-process GPU. This process exits before
    # the shell launches the full model's separate, sequential Python command.
    result = pytest.main(["-q", "tests/test_search_decoder_study.py", "-k", "matched_search_hook and cuda"])
    if result != 0:
        raise RuntimeError(f"Allocated CUDA matched-search fixture failed: pytest exit{result}")


if __name__ == "__main__":
    main()
