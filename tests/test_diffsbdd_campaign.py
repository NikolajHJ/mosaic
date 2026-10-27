"""Allocated standard-library campaign guard tests; no numerical imports/jobs."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "diffsbdd_campaign_under_test", Path(__file__).resolve().parents[1] / "scripts/diffsbdd_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads(campaign.CONFIG.read_text())
        self.config_path = self.root / "config.json"
        self._config()

    def _config(self):
        self.config_path.write_text(json.dumps(self.config))
        return self.config_path

    def _seal(self, *, extra_member=None):
        directory = self.root / "sealed"
        directory.mkdir()
        config_bytes = self.config_path.read_bytes()
        files = {"configs/protocol.json": hashlib.sha256(config_bytes).hexdigest(),
                 "src/stub.py": hashlib.sha256(b"pass\n").hexdigest()}
        raw_manifest = json.dumps({"version": "diffsbdd-source-manifest-v1", "files": files}).encode()
        payloads = {"source/configs/protocol.json": config_bytes,
                    "source/src/stub.py": b"pass\n",
                    "source/source-manifest.json": raw_manifest}
        archive = directory / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            info = tarfile.TarInfo("source")
            info.type = tarfile.DIRTYPE
            output.addfile(info)
            for name, raw in payloads.items():
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                output.addfile(info, io.BytesIO(raw))
            if extra_member is not None:
                output.addfile(extra_member,
                               io.BytesIO(b"x") if extra_member.isreg() else None)
        seal = {"campaign": self.config["campaign"],
                "config_sha256": campaign.sha(self.config_path),
                "source_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                "archive_sha256": campaign.sha(archive),
                "archive_bytes": archive.stat().st_size, "source_members": len(files),
                "source_bytes": sum(len(raw) for raw in payloads.values())}
        (directory / "seal.json").write_text(json.dumps(seal))
        return directory, seal

    def test_failed_storage_admission_prevents_every_remote_stage_write(self):
        directory, _ = self._seal()
        output = self.root / "stage.json"
        with patch.object(campaign, "admission", return_value={"passed": False, "shortfall_bytes": 123}), \
             patch.object(campaign, "remote_script", side_effect=AssertionError("remote write")) as remote, \
             patch.object(campaign.subprocess, "run", side_effect=AssertionError("scp/bsub")) as process:
            result = campaign.stage(self.config_path, directory, output)
        self.assertFalse(result["passed"])
        self.assertFalse(result["remote_writes"])
        self.assertEqual(result["status"], "blocked_storage_before_any_remote_write")
        self.assertFalse(remote.called)
        self.assertFalse(process.called)
        self.assertFalse(json.loads(output.read_text())["jobs_submitted"])

    def test_sealed_release_keeps_named_config_and_protocol_as_identical_members(self):
        repository = self.root / "repository"
        source = repository / "src/latentmol"
        source.mkdir(parents=True)
        (source / "stub.py").write_text("pass\n")
        inputs = repository / "experiments/diffsbdd-vae-alternation/inputs"
        inputs.mkdir(parents=True)
        (inputs / "3MXF.pdb").write_bytes(b"opaque source bytes; no model or chemistry execution\n")
        assets = self.root / "assets"
        assets.mkdir()
        (assets / "assets.json").write_text(json.dumps({
            "checkpoint_sha256": self.config["upstream"]["checkpoint_sha256"],
            "source_manifest_sha256": self.config["upstream"]["source_manifest_sha256"]}))
        output = self.root / "prepared"
        with patch.object(campaign, "ROOT", repository), \
             patch.object(campaign.subprocess, "check_output", side_effect=["codex/test\n", "a" * 40 + "\n"]), \
             patch.object(campaign, "remote_script", side_effect=AssertionError("remote mutation")):
            result = campaign.prepare(self.config_path, assets, output)
        released = output / "source"
        named = released / "configs/diffsbdd_vae_alternation_v1.json"
        protocol = released / "configs/protocol.json"
        self.assertEqual(named.read_bytes(), self.config_path.read_bytes())
        self.assertEqual(protocol.read_bytes(), named.read_bytes())
        manifest = json.loads((released / "source-manifest.json").read_text())["files"]
        self.assertEqual(manifest["configs/protocol.json"], manifest["configs/diffsbdd_vae_alternation_v1.json"])
        self.assertEqual(manifest["configs/protocol.json"], result["config_sha256"])
        self.assertFalse(result["models_loaded"])
        self.assertFalse(result["numerical_tests_run"])

    def test_failed_storage_admission_prevents_every_remote_submission(self):
        directory, _ = self._seal()
        with patch.object(campaign, "admission", return_value={"passed": False}), \
             patch.object(campaign, "remote_script", side_effect=AssertionError("bsub")) as remote, \
             patch.object(campaign.subprocess, "run", side_effect=AssertionError("transport")) as process:
            result = campaign.submit(self.config_path, directory, "cpu", self.root / "submit.json")
        self.assertEqual(result["status"], "blocked_storage_before_submission")
        self.assertFalse(remote.called)
        self.assertFalse(process.called)

    def test_storage_caps_and_forecast_factor_are_nonwaivable(self):
        original = deepcopy(self.config)
        mutations = (
            ("own_initial_cap_bytes", 1024**3),
            ("own_initial_cap_bytes", 3 * 1024**3),
            ("additional_unreserved_headroom_bytes", 9 * 1024**3),
            ("additional_unreserved_headroom_bytes", 11 * 1024**3),
            ("never_reduce_existing_forecasts_or_reserves", False),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                self.config = deepcopy(original)
                self.config["storage"][key] = value
                with self.assertRaisesRegex(ValueError, "Storage"):
                    campaign.configuration(self._config())
        self.config = deepcopy(original)
        self.config["runtime_forecast_factor"] = 1.0
        with self.assertRaisesRegex(ValueError, "forecast"):
            campaign.configuration(self._config())

    def test_stage_resource_caps_cannot_be_enlarged_or_changed_before_transport(self):
        original = deepcopy(self.config)
        for stage in campaign.STAGES:
            for key, value in (("slots", 8), ("wall_minutes", 1441),
                               ("internal_seconds", 90000), ("queue", "other")):
                with self.subTest(stage=stage, key=key):
                    self.config = deepcopy(original)
                    self.config["stages"][stage][key] = value
                    with self.assertRaisesRegex(ValueError, "Allocation caps"):
                        campaign.configuration(self._config())

    def test_archive_is_hashed_and_bounded_before_remote_admission(self):
        extra = tarfile.TarInfo("source-evil/escape.py")
        extra.size = 1
        directory, _ = self._seal(extra_member=extra)
        with patch.object(campaign, "admission", side_effect=AssertionError("remote admission ran")), \
             patch.object(campaign, "remote_script", side_effect=AssertionError("remote write")):
            with self.assertRaisesRegex(ValueError, "exact source"):
                campaign.stage(self.config_path, directory, self.root / "bad-stage.json")

    def test_all_hostile_tar_paths_are_rejected(self):
        paths = ("source-evil/file", "/source/file", "source/../file", "source/./file",
                 "source//file", "source\\file", "source/C:/file", "other/source/file", "source/\x00file")
        for name in paths:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    campaign.safe_archive_member(name)
        self.assertEqual(campaign.safe_archive_member("source/src/ok.py"), "src/ok.py")

    def test_archive_rejects_links_devices_and_FIFO_before_extraction(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                # Each immutable local seal uses a fresh independent directory.
                other_root = self.root / kind.hex()
                other_root.mkdir()
                previous = self.root
                self.root = other_root
                extra = tarfile.TarInfo("source/special")
                extra.type = kind
                extra.linkname = "../../escape"
                directory, seal = self._seal(extra_member=extra)
                with self.assertRaisesRegex(ValueError, "special member"):
                    campaign.inspect_source_archive(directory / "source.tar.gz", seal, 2 * 1024**3)
                self.root = previous

    def test_archive_detects_unsealed_inventory_and_too_small_byte_cap(self):
        extra = tarfile.TarInfo("source/extra.py")
        extra.size = 1
        directory, seal = self._seal(extra_member=extra)
        with self.assertRaisesRegex(ValueError, "member cap|inventory"):
            campaign.inspect_source_archive(directory / "source.tar.gz", seal, 2 * 1024**3)
        with self.assertRaisesRegex(ValueError, "cap"):
            campaign.inspect_source_archive(directory / "source.tar.gz", seal, 1)

    def test_negative_receipt_is_never_replaced_by_later_positive_observation(self):
        path = self.root / "admission.json"
        negative = {"passed": False, "shortfall_bytes": 123}
        campaign.save_new(path, negative)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            campaign.save_new(path, {"passed": True})
        self.assertEqual(path.read_bytes(), before)

    def test_fresh_quota_cannot_raise_old_capacity_or_reduce_old_forecast(self):
        prior_path = self.root / "prior-capacity.json"
        prior = {"status": "passed", "available_bytes": 50 * 1024**3,
                 "required_bytes": 49_684_462_417}
        prior_path.write_text(json.dumps(prior))
        self.config["storage"]["other_campaign_capacity_path"] = str(prior_path)
        self._config()

        def read_only_remote(script, output, **kwargs):
            captured = io.StringIO()
            with redirect_stdout(captured), patch.object(
                    subprocess, "run", return_value=SimpleNamespace(
                        returncode=0, stdout="s234818|306121||   20.00 GiB|  100.00 GiB||\n", stderr="")):
                exec(compile(script, "<allocated-read-only-quota-fixture>", "exec"), {})
            return json.loads(captured.getvalue())

        with patch.object(campaign, "remote_script", side_effect=read_only_remote):
            result = campaign.admission(self.root / "fresh.json", self.config_path)
        self.assertEqual(result["available_bytes_conservative"], prior["available_bytes"])
        self.assertEqual(result["other_campaign_forecasts_and_reserves_bytes"], prior["required_bytes"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["required_bytes"], prior["required_bytes"] + 12 * 1024**3)

    def _remote_submit_fixture(self, bsub):
        source = self.root / "remote-source"
        operator = source / "scripts/hpc/diffsbdd_operator.py"
        operator.parent.mkdir(parents=True)
        operator.write_text("def authenticate_source(*args): return {}\n"
                            "def predecessors(*args): return []\n"
                            "STAGES=('cpu','native','scorer','fit','audit','smoke','pilot')\n")
        remote_root = self.root / "remote-runs"

        def remote(script, output, **kwargs):
            script = script.replace(self.config["release_root"], source.as_posix())
            script = script.replace(self.config["run_root"], remote_root.as_posix())
            captured = io.StringIO()
            with redirect_stdout(captured), patch.object(subprocess, "run", side_effect=bsub):
                exec(compile(script, "<allocated-submission-fixture>", "exec"), {})
            return json.loads(captured.getvalue())

        return remote, remote_root

    def test_unknown_bsub_result_is_preserved_and_cannot_be_retried(self):
        directory, _ = self._seal()
        calls = []

        def bsub(argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=0, stdout="no job id available", stderr="")

        remote, root = self._remote_submit_fixture(bsub)
        with patch.object(campaign, "admission", return_value={"passed": True}), \
             patch.object(campaign, "remote_script", side_effect=remote):
            result = campaign.submit(self.config_path, directory, "cpu", self.root / "first.json")
            with self.assertRaisesRegex(RuntimeError, "already attempted"):
                campaign.submit(self.config_path, directory, "cpu", self.root / "second.json")
        self.assertEqual(result["status"], "submission_failed_or_unknown_do_not_retry")
        self.assertEqual(len(calls), 1)
        self.assertTrue((root / "submission_cpu_intent.json").exists())
        self.assertTrue((root / "submission_cpu.json").exists())

    def test_bsub_timeout_leaves_intent_and_blocks_ambiguous_retry(self):
        directory, _ = self._seal()
        calls = []

        def bsub(argv, **kwargs):
            calls.append(argv)
            raise subprocess.TimeoutExpired(argv, 45)

        remote, root = self._remote_submit_fixture(bsub)
        with patch.object(campaign, "admission", return_value={"passed": True}), \
             patch.object(campaign, "remote_script", side_effect=remote):
            with self.assertRaises(subprocess.TimeoutExpired):
                campaign.submit(self.config_path, directory, "cpu", self.root / "first.json")
            with self.assertRaisesRegex(RuntimeError, "already attempted"):
                campaign.submit(self.config_path, directory, "cpu", self.root / "second.json")
        self.assertEqual(len(calls), 1)
        self.assertTrue((root / "submission_cpu_intent.json").exists())
        self.assertFalse((root / "submission_cpu.json").exists())
