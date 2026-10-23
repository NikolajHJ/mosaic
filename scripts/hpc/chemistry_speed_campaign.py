"""One sequential native speed suite, then one authenticated continuation.

Controllers use metadata only. Scientific imports and all fixtures/profiles/
benchmarks run inside allocated workers. Each candidate is a separate child
process reset from the immutable epoch-35 parent; its disposable updates never
become a training checkpoint. Failed candidates and raw native gates remain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from chemistry_campaign import Campaign, FIXTURES, OPTIMIZED_FIXTURES, authenticated, read, quota_receipt, validate_plan
from latentmol.chemistry_speed_selection import choose_candidate, summarize_candidate
from latentmol.experiments import file_sha256, write_json


EXTRA_FIXTURES = ["tests/test_async_repair_teacher.py", "tests/test_chemistry_speed_selection.py",
                  "tests/test_chemistry_speed_campaign.py"]

SMALL_ALLOCATION_POLICY = "max24h_16cpu_v1"


def suite_gpu_slots(plan):
    """Keep historical 32-slot suites readable; explicitly version smaller ones.

    The reduced family uses twelve detached repair workers and four model
    threads. A 24-hour allocation reserves the final hour for safe shutdown;
    the original epoch-fit/retention/native gates remain unchanged.
    """
    policy = plan.get("allocation_policy")
    if policy is None:
        expected = 32
    elif policy == SMALL_ALLOCATION_POLICY:
        expected = 16
        if (type(plan.get("protocol")) is not dict
                or type(plan["protocol"].get("segment_wall_seconds")) is not int
                or not 0 < plan["protocol"]["segment_wall_seconds"] <= 23 * 3600
                or type(plan.get("execution", {}).get("workers")) is not int
                or plan["execution"]["workers"] != 12):
            raise ValueError("Small chemistry suite must retain twelve workers and <=23h internal time")
    else:
        raise ValueError("Unsupported chemistry suite allocation policy")
    if type(plan.get("execution", {}).get("gpu_cpus")) is not int or plan["execution"]["gpu_cpus"] != expected:
        raise ValueError("Chemistry suite CPU allocation differs from its versioned policy")
    return expected


def require_allocation_limit(plan, limit):
    """Enforce the user cap at submission, without rewriting scientific gates."""
    if plan.get("allocation_policy") == SMALL_ALLOCATION_POLICY:
        try:
            hours, minutes = limit.split(":")
            valid = (hours.isdecimal() and minutes.isdecimal() and 0 <= int(minutes) < 60
                     and 0 < 60 * int(hours) + int(minutes) <= 24 * 60)
        except (AttributeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Chemistry allocation exceeds the user 24-hour cap")


def require_observation_coverage(variants, rows, threshold):
    """Never promote an incomplete search over the predeclared candidate family."""
    if type(rows) is not list:
        raise ValueError("Missing ordered candidate observations")
    first = [value for value in variants if value["prior_ratio"] == 3]
    head = rows[:len(first)]
    if len(head) != len(first) or [(row.get("id"), row.get("prior_ratio")) for row in head] != [
            (value["id"], 3) for value in first]:
        raise ValueError("Incomplete or reordered 1:3 candidate family")
    preference = choose_candidate(head, preferred_epoch_seconds=threshold)
    expected = first if preference["status"] != "fallback_required" else [
        *first, *(value for value in variants if value["prior_ratio"] == 2)]
    if [(row.get("id"), row.get("prior_ratio")) for row in rows] != [
            (value["id"], value["prior_ratio"]) for value in expected]:
        raise ValueError("Incomplete, unexpected or reordered fallback candidate family")


class SpeedSuite(Campaign):
    """The real shared CPU/GPU submission receipts, never fictional variant jobs."""

    def __init__(self, path, sha):
        super().__init__(path, sha)
        variants = self.plan.get("variants")
        if (type(variants) is not list or not variants or self.plan.get("selection_threshold_seconds") != 36000
                or self.plan.get("suite_gpu_wall_seconds") != 19800):
            raise ValueError("Unsupported sealed speed suite")
        ids = [value["id"] for value in variants]
        if len(ids) != len(set(ids)) or not any(value["prior_ratio"] == 2 for value in variants):
            raise ValueError("Missing/duplicate speed candidate family")
        self.variants = variants
        slots = suite_gpu_slots(self.plan)
        for variant in variants:
            plan = validate_plan(variant["plan"], variant["plan_sha256"])
            if (plan["release"] != self.plan["release"] or plan["archive_sha256"] != self.plan["archive_sha256"]
                    or plan["manifest_sha256"] != self.plan["manifest_sha256"]
                    or plan["parent_checkpoint_sha256"] != self.plan["parent_checkpoint_sha256"]
                    or plan["config_sha256"] != self.plan["config_sha256"]
                    or plan["dataset_sha256"] != self.plan["dataset_sha256"]
                    or plan["split_sha256"] != self.plan["split_sha256"]
                    or plan["parent_epoch"] != 35 or plan["parent_updates"] != 230860
                    or plan["speed_variant_id"] != variant["id"]
                    or plan["speed_suite_root"] != str(self.root)
                    or plan["protocol"]["prior_ratio"] != variant["prior_ratio"]
                    or plan["protocol"]["benchmark_updates"] != variant["benchmark_updates"]
                    or plan["protocol"]["microbatch_size"] != variant["microbatch_size"]
                    or plan["execution"]["workers"] != variant["workers"]
                    or plan["execution"]["overlap"] != variant["overlap"]
                    or plan["execution"]["gpu_cpus"] != slots
                    or plan.get("allocation_policy") != self.plan.get("allocation_policy")):
                raise ValueError("Candidate escaped the common source/parent/data/execution bindings")
            if self.plan.get("allocation_policy") == SMALL_ALLOCATION_POLICY:
                suite_gpu_slots(plan)

    def controller(self, mode, **values):
        command = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/chemistry_speed_campaign.py",
                   "--suite", str(self.path), "--suite-sha256", self.sha, "--mode", mode]
        for key, value in values.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *command]

    def arguments(self, stage, gpu=False, limit="0:15", dependency=None):
        require_allocation_limit(self.plan, limit)
        command = super().arguments(stage, gpu=gpu, limit=limit, dependency=dependency)
        if stage == "cpu" and not gpu:
            # The CPU preflight runs the reference/12-worker profile only.
            # Sixteen slots preserve four Torch + twelve search workers.
            # Historical large-pool suites reserve32 GPU slots; the explicitly
            # versioned smaller family uses16 for both CPU and GPU allocations.
            command[command.index("-n") + 1] = "16"
        return command

    def queue_initial(self):
        write_json(self.root / "initial_quota.json", quota_receipt(2))
        # A terminated pending predecessor is not a failed scientific check.
        retired = self.plan.get("retired_jobs", [])
        if type(retired) is not list or any(type(job) is not str or not job.isdecimal() for job in retired):
            raise ValueError("Malformed predecessor retirement list")
        for job in ("29745445", "29745446", "29745514", "29745515", "29745516", *retired):
            status = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                                    capture_output=True, text=True, timeout=20)
            if status.stdout.strip() != "EXIT" and "not found" not in status.stderr.lower():
                raise ValueError("Superseded source115 job is not confirmed terminal")
        cpu = self.queue("cpu", self.worker(self.controller("cpu"), "source116-speed-cpu"), limit="2:00")
        gpu = self.queue("benchmark", self.worker(self.controller("benchmark"), "source116-speed-benchmark"),
                         gpu=True, limit="6:00", dependency="done(" + cpu + ")")
        callback = self.queue("after-benchmark", self.controller("promote"), dependency="ended(" + gpu + ")")
        return {"cpu": cpu, "benchmark": gpu, "promotion_callback": callback,
                "status": "submitted_not_passed_no_training_yet"}

    def cpu(self):
        if not os.environ.get("LSB_JOBID") or os.environ.get("LSB_DJOB_NUMPROC") != "16":
            raise RuntimeError("Speed CPU fixtures require their 16-slot allocation")
        # All source-bound candidate protocols are parsed in the allocation.
        from latentmol.chemistry_continuation import read_plan
        for variant in self.variants:
            read_plan(variant["plan"], variant["plan_sha256"])
        command = [sys.executable, "-m", "pytest", "-q", *FIXTURES, *OPTIMIZED_FIXTURES, *EXTRA_FIXTURES]
        reply = subprocess.run(command, cwd=self.release)
        gate = {"status": "failed", "returncode": reply.returncode, "identity": self.identity,
                "job_id": os.environ["LSB_JOBID"], "command": command,
                "candidate_plan_sha256": {row["id"]: row["plan_sha256"] for row in self.variants}}
        if reply.returncode == 0:
            baseline = self.variants[0]
            profile = self.root / "repair_profile.json"
            command = [sys.executable, "scripts/benchmark_atom_preserving_search.py", "--plan", baseline["plan"],
                       "--plan-sha256", baseline["plan_sha256"], "--output", str(profile)]
            result = subprocess.run(command, cwd=self.release)
            gate.update(profile_command=command, profile_returncode=result.returncode,
                        profile_plan_sha256=baseline["plan_sha256"])
            if result.returncode == 0:
                gate.update(status="passed", profile_sha256=file_sha256(profile))
        write_json(self.root / "cpu_gate.json", gate)
        if gate["status"] != "passed":
            raise RuntimeError("Speed-suite CPU fixtures/profile failed")

    def require_cpu(self):
        self.complete("cpu")
        gate = read(self.root / "cpu_gate.json")
        if (gate.get("status") != "passed" or gate.get("returncode") != 0 or gate.get("profile_returncode") != 0
                or gate.get("identity") != self.identity or gate.get("job_id") != self.receipt_job("cpu")
                or gate.get("candidate_plan_sha256") != {row["id"]: row["plan_sha256"] for row in self.variants}):
            raise ValueError("Shared allocated CPU gate/plan coverage differs")
        authenticated(self.root / "repair_profile.json", gate["profile_sha256"])
        profile = read(self.root / "repair_profile.json")
        expected = {**self.identity, "plan_sha256": self.variants[0]["plan_sha256"]}
        if (profile.get("status") != "passed" or profile.get("identity") != expected
                or profile.get("job_id") != gate["job_id"]):
            raise ValueError("Shared profile does not authenticate this parent/source/allocation")

    def observe_gate(self, variant, returncode):
        """Authenticate a child's completed native result before timing selection."""
        plan = read(variant["plan"])
        root = Path(plan["campaign"])
        gate_path = root / "benchmark_gate.json"
        gate = read(gate_path)
        run = Path(gate["run_dir"]).resolve(strict=True)
        record = read(run / "run.json")
        if (not run.is_relative_to(root / "benchmark")
                or record.get("status") != "completed" or gate.get("status") != "passed"
                or record.get("environment", {}).get("job_id") != self.receipt_job("benchmark")
                or record.get("source_archive_sha256") != self.plan["archive_sha256"]
                or record.get("source", {}).get("sha256") != self.plan["manifest_sha256"]
                or record.get("config", {}).get("_plan_sha256") != variant["plan_sha256"]
                or gate.get("plan_sha256") != variant["plan_sha256"]
                or gate.get("parent_sha256") != self.plan["parent_checkpoint_sha256"]
                or gate.get("execution") != plan["execution"]
                or file_sha256(run / "gate.json") != file_sha256(gate_path)):
            raise ValueError("Native benchmark source/parent/plan/job/completion differs")
        row = summarize_candidate(variant, gate, returncode, training_molecules=self.plan["training_molecules"])
        row.update(gate_sha256=file_sha256(gate_path), gate_path=str(gate_path), run_dir=str(run))
        return row

    def benchmark(self):
        self.require_cpu()
        slots = suite_gpu_slots(self.plan)
        if not os.environ.get("LSB_JOBID") or os.environ.get("LSB_DJOB_NUMPROC") != str(slots):
            raise RuntimeError(f"Sequential native benchmark requires its declared {slots}-slot GPU allocation")
        started = time.monotonic()
        cuda_command = ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1", sys.executable,
                        "-m", "pytest", "-q", "tests/test_chemistry_resume_cuda.py"]
        with (self.root / "cuda-fixtures.out").open("x") as out, (self.root / "cuda-fixtures.err").open("x") as err:
            fixture = subprocess.run(cuda_command, cwd=self.release, stdout=out, stderr=err, timeout=900)
        cuda_gate = {"identity": self.identity, "job_id": os.environ["LSB_JOBID"], "command": cuda_command,
                     "returncode": fixture.returncode, "status": "passed" if fixture.returncode == 0 else "failed"}
        write_json(self.root / "cuda-fixtures.json", cuda_gate)
        if fixture.returncode:
            raise RuntimeError("Mandatory unskipped CUDA fixtures failed; no candidate promoted")
        rows = []
        for ratio in (3, 2):
            if ratio == 2 and choose_candidate(rows)["status"] != "fallback_required":
                break
            for variant in self.variants:
                if variant["prior_ratio"] != ratio:
                    continue
                remaining = self.plan["suite_gpu_wall_seconds"] - (time.monotonic() - started)
                if remaining < 900:
                    raise RuntimeError("Speed suite incomplete: insufficient time to compare every declared candidate")
                command = [sys.executable, "-m", "latentmol.chemistry_continuation", "benchmark",
                           "--plan", variant["plan"], "--plan-sha256", variant["plan_sha256"]]
                path = self.root / "candidate-logs"
                path.mkdir(exist_ok=True)
                status = {"id": variant["id"], "prior_ratio": ratio, "eligible": False,
                          "selection_epoch_seconds": None}
                try:
                    with (path / (variant["id"] + ".out")).open("x") as out, (path / (variant["id"] + ".err")).open("x") as err:
                        child = subprocess.Popen(command, cwd=self.release, stdout=out, stderr=err, start_new_session=True)
                        try:
                            code = child.wait(timeout=min(6300, remaining - 300))
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait(timeout=20)
                            raise RuntimeError("Candidate timeout; child process group terminated, no training state promoted")
                    status = self.observe_gate(variant, code)
                except Exception as exc:
                    status["rejection"] = f"{type(exc).__name__}: {exc}"
                rows.append(status)
                write_json(self.root / "observations.json", {"identity": self.identity, "rows": rows,
                           "scope": "Disposable candidate measurements; no training started"})
        fallback = any(row["prior_ratio"] == 2 for row in rows)
        require_observation_coverage(self.variants, rows, self.plan["selection_threshold_seconds"])
        selected = choose_candidate(rows, preferred_epoch_seconds=self.plan["selection_threshold_seconds"], fallback_evaluated=fallback)
        result = {"identity": self.identity, "observations": rows, "selection": selected,
                  "cuda_fixture_gate_sha256": file_sha256(self.root / "cuda-fixtures.json"),
                  "benchmark_job_id": self.receipt_job("benchmark"), "utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started,
                  "ranking": "maximum timed complete-update wall scaled to real epoch plus unchanged 600s allowance",
                  "all_candidates_reset_parent": True, "training_started": False}
        write_json(self.root / "selection.json", result)
        write_json(self.root / "selection.sha256.json", {"sha256": file_sha256(self.root / "selection.json")})
        if selected["status"] != "selected_not_training_started":
            raise RuntimeError("No passing chemistry continuation candidate")

    def selected(self):
        self.require_cpu()
        self.complete("benchmark")
        pointer = read(self.root / "selection.sha256.json")
        authenticated(self.root / "selection.json", pointer["sha256"])
        result = read(self.root / "selection.json")
        authenticated(self.root / "cuda-fixtures.json", result["cuda_fixture_gate_sha256"])
        cuda = read(self.root / "cuda-fixtures.json")
        if (cuda.get("status") != "passed" or cuda.get("returncode") != 0
                or cuda.get("identity") != self.identity or cuda.get("job_id") != self.receipt_job("benchmark")):
            raise ValueError("Mandatory native CUDA fixture receipt differs")
        selection = result["selection"]
        require_observation_coverage(self.variants, result["observations"], self.plan["selection_threshold_seconds"])
        if (result.get("identity") != self.identity or result.get("benchmark_job_id") != self.receipt_job("benchmark")
                or selection.get("status") != "selected_not_training_started"):
            raise ValueError("Winner selection is not a completed authenticated suite result")
        recomputed = choose_candidate(result["observations"], preferred_epoch_seconds=self.plan["selection_threshold_seconds"],
                                      fallback_evaluated=any(row["prior_ratio"] == 2 for row in result["observations"]))
        if recomputed != selection:
            raise ValueError("Winner selection policy/result differs")
        variant = next(row for row in self.variants if row["id"] == selection["winner"])
        for observed in result["observations"]:
            if observed["eligible"]:
                declared = next(row for row in self.variants if row["id"] == observed["id"])
                if self.observe_gate(declared, 0) != observed:
                    raise ValueError("Passing native comparison gate changed after ranking")
        return variant, pointer["sha256"]


class SpeedCandidate(Campaign):
    """All later callbacks retain suite selection and common CPU provenance."""

    def __init__(self, suite, variant):
        super().__init__(variant["plan"], variant["plan_sha256"])
        self.suite, self.variant = suite, variant

    def require_cpu(self):
        variant, digest = self.suite.selected()
        if variant != self.variant:
            raise ValueError("Unselected speed candidate cannot submit training")
        write_json(self.root / "authenticated_selection.json", {"suite_sha256": self.suite.sha,
                   "selection_sha256": digest, "variant_id": variant["id"], "plan_sha256": self.sha})

    def receipt_job(self, stage):
        return self.suite.receipt_job(stage) if stage in ("cpu", "benchmark") else super().receipt_job(stage)

    def complete(self, stage):
        return self.suite.complete(stage) if stage in ("cpu", "benchmark") else super().complete(stage)

    def controller(self, mode, **values):
        return self.suite.controller(mode, variant=self.variant["id"], **values)

    def arguments(self, stage, gpu=False, limit="0:15", dependency=None):
        require_allocation_limit(self.suite.plan, limit)
        return super().arguments(stage, gpu=gpu, limit=limit, dependency=dependency)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "benchmark", "promote", "advance"), required=True)
    parser.add_argument("--variant")
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    suite = SpeedSuite(args.suite, args.suite_sha256)
    if args.mode == "queue":
        print(json.dumps(suite.queue_initial(), indent=2))
    elif args.mode == "cpu":
        suite.cpu()
    elif args.mode == "benchmark":
        suite.benchmark()
    elif args.mode == "promote":
        variant, _ = suite.selected()
        candidate = SpeedCandidate(suite, variant)
        candidate.advance("benchmark")
    else:
        variant, _ = suite.selected()
        if args.variant != variant["id"] or not args.after_stage:
            parser.error("Only the selected variant with an explicit predecessor can advance")
        SpeedCandidate(suite, variant).advance(args.after_stage)


if __name__ == "__main__":
    main()
