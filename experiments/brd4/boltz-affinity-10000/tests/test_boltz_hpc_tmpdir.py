"""Allocated Linux CPU regression checks for native Boltz AF_UNIX startup."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest


HELPER = _layout_path(_repository_root(__file__) / "scripts/hpc/boltz_node_tmpdir.sh")


class BoltzNodeTemporaryDirectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != "posix" or not os.environ.get("LSB_JOBID"):
            raise RuntimeError(
                "These operating-system checks require allocated DTU HPC"
            )

    def run_launcher(self, body, job_id="987654321", extra_env=None):
        environment = os.environ.copy()
        if job_id is None:
            environment.pop("LSB_JOBID", None)
        else:
            environment["LSB_JOBID"] = job_id
        environment.update(extra_env or {})
        return subprocess.run(
            [
                "bash",
                "-c",
                'set -euo pipefail\nsource "$1"\n' + body,
                "mosaic-boltz-tmpdir-test",
                str(HELPER),
                sys.executable,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_short_directory_supports_real_multiprocessing_style_unix_socket(self):
        result = self.run_launcher(
            """printf '%s\\n' "$TMPDIR"
"$2" - <<'PY'
import os
from pathlib import Path
import socket
import tempfile
directory = tempfile.mkdtemp(prefix="pymp-")
address = str(Path(directory) / "listener-abcdefgh")
assert len(os.fsencode(address)) <= 107
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
    listener.bind(address)
    assert Path(address).is_socket()
print("bound")
PY
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        path, marker = result.stdout.splitlines()
        self.assertEqual(marker, "bound")
        self.assertTrue(path.startswith("/tmp/mosaic-boltz.987654321."))
        self.assertFalse(Path(path).exists())

    def test_original_long_candidate_reproduces_unix_socket_error(self):
        with tempfile.TemporaryDirectory(prefix="mosaic-long-", dir="/tmp") as root:
            directory = Path(root) / ("x" * 110)
            directory.mkdir()
            address = str(directory / "listener-abcdefgh")
            self.assertGreater(len(os.fsencode(address)), 107)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                with self.assertRaisesRegex(OSError, "AF_UNIX path too long"):
                    listener.bind(address)

    def test_normal_exit_removes_owned_directory_and_contents(self):
        result = self.run_launcher(
            'printf \'%s\\n\' "$TMPDIR"\ntouch "$TMPDIR/owned-file"\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(Path(result.stdout.strip()).exists())

    def test_error_exit_cleans_directory_and_preserves_original_failure(self):
        result = self.run_launcher(
            'printf \'%s\\n\' "$TMPDIR"\ntouch "$TMPDIR/owned-file"\nexit 37\n'
        )
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertFalse(Path(result.stdout.strip()).exists())

    def test_missing_allocation_fails_before_exposing_temporary_directory(self):
        result = self.run_launcher("printf '%s\\n' \"$TMPDIR\"\n", job_id=None)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("requires a numeric LSB_JOBID", result.stderr)

    def test_malformed_job_identifier_is_rejected(self):
        result = self.run_launcher(
            "printf '%s\\n' \"$TMPDIR\"\n", job_id="../../escape"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("requires a numeric LSB_JOBID", result.stderr)

    def test_same_job_launches_get_distinct_directories(self):
        first = self.run_launcher("printf '%s\\n' \"$TMPDIR\"\n")
        second = self.run_launcher("printf '%s\\n' \"$TMPDIR\"\n")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotEqual(first.stdout.strip(), second.stdout.strip())
        self.assertFalse(Path(first.stdout.strip()).exists())
        self.assertFalse(Path(second.stdout.strip()).exists())

    def test_inherited_long_tmpdir_does_not_determine_node_socket_directory(self):
        inherited = "/zhome/" + "long-home-component/" * 12
        result = self.run_launcher(
            "printf '%s\\n' \"$TMPDIR\"\n", extra_env={"TMPDIR": inherited}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        path = result.stdout.strip()
        self.assertNotEqual(path, inherited)
        self.assertLess(len(os.fsencode(path)), 70)
        self.assertTrue(path.startswith("/tmp/mosaic-boltz.987654321."))
        self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
