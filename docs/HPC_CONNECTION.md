# Current DTU HPC connection for local agents

Adopted 29 October 2026. **Use the installed unattended connector as the default
local transport for HPC commands and file transfers on Nikolajs_laptop.**
This guide applies to Windows account `nikol`, DTU account `s234818`, and the
verified `login1.hpc.dtu.dk` connection. The same installed helper serves the
DBMol_exploratory and Mosaic checkouts. Other machines/accounts require their
own authorized setup; cloning either repository does not install the helper.

## Commands and status checks

Run these in local PowerShell:

```powershell
$dtu = 'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
& $dtu status
& $dtu exec 'hostname'
& $dtu exec "bash -lc 'bjobs -u s234818'"
& $dtu exec --timeout 60 'YOUR_AUTHORIZED_REMOTE_COMMAND'
```

`status` reports local credential metadata and host-pin readiness without printing
the password. `exec 'hostname'` makes a fresh authenticated connection; it does
not submit a job. The helper uses the existing Windows SSH agent and the generic
Windows credential `DTU-HPC:s234818`, so routine use needs no VPN connection or
password entry. Keep credential values out of commands, files, logs and chat.

`exec` accepts **one remote-command string**, forwards stdin/stdout/stderr, and
returns the remote exit code. The default command timeout is 120 seconds; set
`--timeout` explicitly between 1 and 3600 seconds where needed. Preserve existing
login-shell setup (`bash -lc`, for example), interpreter/module setup, payloads,
scientific guards and receipt handling when changing a local transport call.

For a local Python controller that already constructs a reviewed remote command
and sends a payload through stdin, the transport shape is:

```python
import subprocess

connector = r"C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe"
result = subprocess.run(
    [connector, "exec", "--timeout", "60", remote_command],
    input=payload,
    text=True,
    capture_output=True,
    timeout=70,
)
```

`remote_command` and `payload` are the original authorized controller inputs.
Inspect `result.returncode` and preserve its output/error receipts. A completed
SSH process does not by itself establish scientific acceptance.

## File transfers

```powershell
$dtu = 'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
& $dtu upload --timeout 120 'C:\path\local-file' '/absolute/remote/file'
& $dtu download --timeout 120 '/absolute/remote/file' 'C:\path\new-local-file'
```

Remote paths must be absolute. Downloads refuse an existing local destination;
uploads have normal SCP overwrite behavior. Choose destinations within the
authorized task and retain required size/hash checks. Transfers currently use
the same pinned login1 host; a dedicated large-data transfer-server workflow is
not configured by this helper.

## Existing scripts and connection failures

Plain `ssh`, `scp`, `ssh dtu-hpc` and local scripts hardcoded to those commands
are unchanged. They do not automatically read the saved Windows credential.
For new operations, invoke the helper explicitly. When reusing an existing
controller, adapt only its local transport as needed; preserve the remote
command, state, receipts and guards. Do not rewrite sealed HPC releases to
change laptop authentication, and do not replay historical receipt scripts.

Do not make a routine VPN/password request before trying the configured helper.
If it fails, inspect the exit code, non-secret stderr, local `status`, and its
`receipts/` entries. Stop repeated authentication attempts; distinguish local
credential/key problems, network/host-key failures and remote command failures.
Host-key checking remains enforced. Unexpected authentication prompts are
refused rather than supplied with the saved password.

After a lost submission response, reconcile the actual scheduler state and
existing receipts before retrying: a connection failure does not prove that
`bsub` failed or that an existing job stopped. Connection setup does not grant
new experiment, cleanup, restart, monitoring or gate-waiver authority.

## Installation, evidence and rollback

The current executable is `C:/Users/nikol/DTU-HPC/helper-v1/dtu-hpc.exe`.
**Do not use the old `AppData/Local/DTU-HPC` path:** Windows redirected it into
the app's private MSIX storage, which ordinary user PowerShell could not see.
The helper and configuration backup were copied outside AppData with hash
verification; the original copies and historical receipts remain intact.

The installed helper passed 12 local checks and nine original fresh authenticated
connections covering command execution, read-only scheduler access, stdin/stdout,
remote exit status and a 61-byte hash-verified SCP roundtrip with remote cleanup.
After relocation, both the agent and the user's ordinary PowerShell returned
`hpclogin1` without input. These are connection tests, not model/numerical tests
or a statement about current job progress. Reboot persistence is still untested.

At 21:28--21:42 CET on 29 October, the agent independently verified a fresh
login, stdin forwarding, scheduler access and a hash-matched result download,
then used the connector to upload and submit the already-authorized source145
CPU validation job29851083 and collect its successful terminal results. The
[command receipt](../outputs/hpc-results/helper-connection-verification-1793305717772247900.json),
[download receipt](../outputs/hpc-results/helper-download-verification-1793305865558822200-receipt.json),
[submission receipt](../outputs/hpc-submissions/source145-neighbor-environment-validation-attempt004/stage-submit-1793306306292703900.json)
and [terminal result](../outputs/hpc-results/source145-neighbor-environment-validation-attempt004/snapshot-1793306498640264200/result.json)
bind that complete workflow. No VPN or SSH configuration was changed during
these checks. A successful current connection does not test reboot persistence
or availability during a future network outage.

Local references:

- `C:/Users/nikol/DTU-HPC/helper-v1/README.txt`: implementation, usage and rollback.
- `C:/Users/nikol/DTU-HPC/helper-v1/checks/summary001.json`: original functional checks.
- `C:/Users/nikol/DTU-HPC/helper-v1/checks/summary002-path-repair.json`: current path and user-terminal confirmation.
- `C:/Users/nikol/DTU-HPC/relocation-manifest001.json`: verified copy provenance.
- `C:/Users/nikol/DTU-HPC/backups/20261029T121748Z-f90adb27/relocated-manifest002.json`: configuration backup paths and hashes.

The laptop must be awake, online and running the agent under the Windows account
that owns the key and credential. Update the named credential locally when the
DTU password changes. To stop using the helper, stop invoking it; the original
SSH configuration remains available. No background monitor or scheduled task
is installed by this transport.
