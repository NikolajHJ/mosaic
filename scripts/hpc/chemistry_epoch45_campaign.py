"""Allocated complete contracts -> exact epoch40 handoff -> guarded serial45.

This controller preserves the prior CPU/profile, actual-argv, CUDA and all
scientific/health/runtime/resume/quota gates. Generation observations are additive
and cannot change the objective or select a more convenient checkpoint.
"""
import argparse
from pathlib import Path

from chemistry_campaign import Campaign, read
from chemistry_valid_reference_campaign import ValidReferenceCampaign
from latentmol.chemistry_epoch45_extension import validate_extension
from latentmol.experiments import file_sha256


EXTRA_FIXTURES = ["tests/test_chemistry_epoch45_extension.py", "tests/test_chemistry_epoch45_generation.py",
                  "tests/test_cuda_process.py"]


class Epoch45Campaign(ValidReferenceCampaign):
    def __init__(self, path, sha):
        super().__init__(path, sha)
        if validate_extension(self.plan) is None:
            raise ValueError("Require explicit epoch40->45 authorization")

    def controller(self, mode, **values):
        command = super().controller(mode, **values)
        return [s.replace("scripts/hpc/chemistry_valid_reference_campaign.py",
                          "scripts/hpc/chemistry_epoch45_campaign.py") for s in command]

    def cpu_contract_command(self, executable):
        return [*super().cpu_contract_command(executable), *EXTRA_FIXTURES]

    def worker(self, command, label):
        if command[:4] == ["python", "-m", "latentmol.chemistry_continuation", "train"]:
            self.check_launch(command)
            segment = command[command.index("--segment") + 1]
            wrapper = ["python", "scripts/hpc/chemistry_epoch45_worker.py", "--plan", str(self.path),
                "--plan-sha256", self.sha, "--segment", segment, "--", *command]
            return Campaign.worker(self, wrapper, label)
        return super().worker(command, label)

    def cuda_check(self):
        # Inherit every mandatory fixture and the exact zero-skips policy;
        # append extension contracts rather than replacing historical coverage.
        import chemistry_valid_reference_campaign as original
        before = original.CUDA_FIXTURES
        original.CUDA_FIXTURES = list(dict.fromkeys([*before, *EXTRA_FIXTURES]))
        try:
            return super().cuda_check()
        finally:
            original.CUDA_FIXTURES = before

    def advance(self, after_stage):
        if after_stage != "benchmark":
            number = int(after_stage.removeprefix("segment-"))
            completion = read(self.root / f"segment-{number:03d}-completion.json")
            monitor = read(self.root / f"generation-segment-{number:03d}.json")
            if (monitor.get("status") != "passed_observation_integrity"
                    or monitor.get("plan_sha256") != self.sha
                    or monitor.get("completion_sha256") != file_sha256(self.root / f"segment-{number:03d}-completion.json")):
                raise ValueError("Checkpoint-generation observation integrity failed")
            for entry in monitor["observations"]:
                # Observation receipts are serialized JSON, so paths arrive as
                # strings even when the observer originally used pathlib.
                if file_sha256(Path(entry["path"])) != entry["sha256"]:
                    raise ValueError("Saved per-epoch observation changed")
            if monitor["observed_epochs"][-1] != completion["completed_epoch"]:
                raise ValueError("Final checkpoint observation is missing")
        return super().advance(after_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu", "advance", "cuda-check"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    campaign = Epoch45Campaign(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    elif args.mode == "after-cpu":
        print(campaign.after_cpu())
    elif args.mode == "cuda-check":
        campaign.cuda_check()
    elif args.after_stage:
        print(campaign.advance(args.after_stage))
    else:
        parser.error("Require exact predecessor stage")


if __name__ == "__main__":
    main()
