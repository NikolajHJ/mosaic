"""Allocated CPU transport/ownership controls for the generic Protenix RPC.

No pretrained model is loaded by these fixtures. Native co-residency, full
precision, arbitrary-boundary gradients and runtime still need allocated GPU.
"""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/protenix_reference_worker.py"
SPEC = importlib.util.spec_from_file_location("protenix_reference_rpc_fixture", PATH)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)
PIN = "a" * 64


@pytest.fixture(autouse=True)
def allocated_only():
    assert str(os.environ.get("LSB_JOBID", "")).isdigit(), "RPC fixtures require allocated DTU HPC"


def request(number=0, gradient=True, key=None):
    import numpy as np
    fields = {"X": np.full((25, 6), 1 / 6, dtype=np.float64),
              "E_upper": np.full((300, 5), .2, dtype=np.float64)}
    return {"id": number, "gradient": gradient,
        "X": fields["X"].tolist(), "E_upper": fields["E_upper"].tolist(),
        "graph_sha256": hashlib.sha256(b"".join(fields[n].tobytes() for n in ("X", "E_upper"))).hexdigest(),
        "key_data": [0, 17] if key is None else key}


def test_spec_declares_qualified_copy_not_saved_endpoint_or_optimizer():
    spec = r.reference_spec()
    assert spec["qualified_source"]["files"] == 20 and spec["qualified_source"]["manifest_sha256"] == r.QUALIFIED_PIN
    assert spec["science_source"]["files"] == 643 and spec["science_source"]["manifest_sha256"] == r.SCIENCE_PIN
    assert spec["max_bytes"] == 64 * 1024**2 and spec["max_requests"] == 128
    assert spec["optimizer_updates"] == 0 and spec["production_execution_changed"] is False
    assert spec["key_transport"]["selection"] == "actual_live_caller_per_request"
    assert "endpoint" not in spec and "snapshot" not in spec and "key_data" not in spec
    assert "step-10000.npz" not in json.dumps(spec)


@pytest.mark.parametrize("key", [[0, 17], [91, 3355], [4294967295, 0]])
def test_arbitrary_actual_caller_keys_are_preserved_without_inventing_seed17(key):
    value = request(key=key); before = deepcopy(value)
    assert r.Protocol().accept(value) == "evaluate" and value == before
    assert value["key_data"] == key


@pytest.mark.parametrize("change", ["extra", "missing", "id_bool", "id_gap", "gradient_int", "digest_bool",
    "key_bool", "key_shape", "key_negative", "key_overflow", "X_shape", "E_rowshape", "value_string",
    "value_bool", "nan", "infinity"])
def test_malformed_request_cannot_reach_any_native_evaluation(change):
    value = request()
    if change == "extra": value["optimizer"] = True
    elif change == "missing": del value["key_data"]
    elif change == "id_bool": value["id"] = False
    elif change == "id_gap": value["id"] = 2
    elif change == "gradient_int": value["gradient"] = 1
    elif change == "digest_bool": value["graph_sha256"] = True
    elif change == "key_bool": value["key_data"][0] = True
    elif change == "key_shape": value["key_data"].append(3)
    elif change == "key_negative": value["key_data"][0] = -1
    elif change == "key_overflow": value["key_data"][0] = 4294967296
    elif change == "X_shape": value["X"].pop()
    elif change == "E_rowshape": value["E_upper"][0].pop()
    elif change == "value_string": value["X"][0][0] = ".1"
    elif change == "value_bool": value["X"][0][0] = False
    elif change == "nan": value["X"][0][0] = float("nan")
    else: value["X"][0][0] = float("inf")
    with pytest.raises(ValueError): r.Protocol().accept(value)


def test_request_sequence_has_no_replay_gap_or_postclose_path():
    protocol = r.Protocol()
    assert protocol.accept(request()) == "evaluate"
    with pytest.raises(ValueError, match="replay/gap"): protocol.accept(request())
    assert protocol.accept(request(1, gradient=False, key=[23, 99])) == "evaluate"
    assert protocol.accept({"action": "verify_identity"}) == "verify_identity"
    assert protocol.accept({"action": "close"}) == "close"
    with pytest.raises(ValueError, match="after close"): protocol.accept(request(2))


@pytest.mark.parametrize("mode", ["empty", "primal_only"])
def test_close_without_actual_primal_and_gradient_cannot_claim_audit_pass(mode):
    protocol = r.Protocol()
    if mode == "primal_only": protocol.accept(request(gradient=False))
    with pytest.raises(ValueError, match="Unevaluated"): protocol.accept({"action": "close"})


@pytest.mark.parametrize("message", [{"action": "optimize"}, {"action": "close", "id": 1}, {"action": "train"}])
def test_control_protocol_has_no_optimizer_or_undeclared_fields(message):
    with pytest.raises(ValueError, match="control message"): r.Protocol().accept(message)


def test_evaluation_and_control_message_budgets_fail_closed(monkeypatch):
    monkeypatch.setattr(r, "MAX_REQUESTS", 1)
    protocol = r.Protocol(); protocol.accept(request())
    with pytest.raises(ValueError, match="budget"): protocol.accept(request(1))
    protocol = r.Protocol()
    for _ in range(10): protocol.accept({"action": "verify_identity"})
    with pytest.raises(ValueError, match="message budget"): protocol.accept({"action": "verify_identity"})


def test_FP64_transport_digest_and_values_survive_exact_readback():
    import numpy as np
    value = request(); r.Protocol().accept(value)
    fields = r.decode_graph(value)
    for name in ("X", "E_upper"):
        assert fields[name].dtype == np.float64 and fields[name].tolist() == value[name]
    fields["X"][0, 0] += .001
    assert value["X"][0][0] == 1 / 6
    value["graph_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="changed FP64 graph bytes"): r.decode_graph(value)


@pytest.mark.parametrize("change", ["negative", "overone", "row_sum"])
def test_reference_probability_validation_never_repairs_invalid_point(change):
    import numpy as np
    value = request()
    value["X"][0][0] = -.1 if change == "negative" else 1.1 if change == "overone" else .5
    before = deepcopy(value)
    value["graph_sha256"] = hashlib.sha256(b"".join(np.asarray(value[n], dtype=np.float64).tobytes()
                                                     for n in ("X", "E_upper"))).hexdigest()
    with pytest.raises(ValueError, match="no repair"): r.decode_graph(value)
    assert value["X"] == before["X"]


@pytest.mark.parametrize("raw", ['{"id":0,"id":1}', '{"key":{"x":1,"x":2}}', '{"value":NaN}', '{"value":Infinity}'])
def test_JSON_duplicates_and_nonfinite_constants_are_rejected(raw):
    with pytest.raises(ValueError): r.decode_json(raw)


def capsule(source, files):
    source.mkdir(parents=True)
    mapping = {}
    for name, data in files.items():
        p = source / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        mapping[name] = hashlib.sha256(data).hexdigest()
    raw = r._raw({"files": mapping}); (source / "source_manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("change", ["member", "extra", "pin", "count", "unsafe"])
def test_source_scope_or_byte_changes_are_rejected_before_imports(tmp_path, change):
    source = tmp_path / "source"
    pin = capsule(source, {"plan.json": b"{}\n", "scripts/worker.py": b"# safe\n", "empty.txt": b""})
    assert len(r.verify_source(source, pin, 3)["files"]) == 3
    assert r.bound(r.binding(source / "empty.txt")) == source / "empty.txt"
    if change == "member": (source / "scripts/worker.py").write_bytes(b"# changed\n")
    elif change == "extra": (source / "extra.py").write_bytes(b"# extra\n")
    elif change == "unsafe":
        m = r._read(source / "source_manifest.json"); m["files"]["../escape"] = PIN
        raw = r._raw(m); (source / "source_manifest.json").write_bytes(raw); pin = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError): r.verify_source(source, "b" * 64 if change == "pin" else pin,
                                                   4 if change == "count" else None)


@pytest.mark.parametrize("change", ["outside", "relative", "existing", "traversal"])
def test_native_output_is_fresh_and_cannot_overwrite_saved_data(tmp_path, monkeypatch, change):
    root = tmp_path / "run"; root.mkdir(); monkeypatch.setattr(r, "ROOT", root)
    path = root / "reference"
    assert r.validate_output(path) == path
    if change == "outside": path = tmp_path / "original"
    elif change == "relative": path = Path("reference")
    elif change == "existing": path.mkdir()
    else: path = root / ".." / "original"
    with pytest.raises(ValueError): r.validate_output(path)


@pytest.mark.parametrize("queue,slots", [("gpul40s", "4"), ("hpc", "2"), ("hpc", "8")])
def test_metadata_startup_requires_its_actual_CPU_queue_and_four_slots(monkeypatch, queue, slots):
    monkeypatch.setenv("LSB_QUEUE", queue); monkeypatch.setenv("LSB_DJOB_NUMPROC", slots)
    with pytest.raises(ValueError): r.allocated_environment(startup=True)


def test_startup_command_authenticates_metadata_then_stops_before_model_loading(tmp_path, monkeypatch):
    root = tmp_path / "owned"; output = root / "cpu/payload/protenix-reference-startup.json"
    output.parent.mkdir(parents=True); monkeypatch.setattr(r, "ROOT", root)
    monkeypatch.setenv("LSB_QUEUE", "hpc"); monkeypatch.setenv("LSB_DJOB_NUMPROC", "4")
    calls = []
    def validate(path, pin):
        calls.append((path, pin)); return {"own_source_manifest_sha256": "c" * 64}
    monkeypatch.setattr(r, "validate_metadata", validate)
    monkeypatch.setattr(r, "worker", lambda *_: pytest.fail("Startup entered numerical worker"))
    path = tmp_path / "plan.json"
    r.main(["--reference-plan", str(path), "--reference-plan-sha256", PIN,
            "--startup-only", "--startup-output", str(output)])
    assert calls == [(path, PIN)]
    result = r._read(output)
    assert result["status"] == "passed_metadata_only" and not result["numerical_setup_entered"]
    assert result["optimizer_updates"] == 0 and result["plan_sha256"] == PIN
    with pytest.raises(FileExistsError): r._write(output, {"overwrite": True})


@pytest.mark.parametrize("mode", ["wrong_output", "native_output", "no_output"])
def test_bad_startup_scope_cannot_even_reach_input_authentication(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setenv("LSB_QUEUE", "hpc"); monkeypatch.setenv("LSB_DJOB_NUMPROC", "4")
    monkeypatch.setattr(r, "validate_metadata", lambda *_: pytest.fail("Invalid startup reached inputs"))
    args = ["--reference-plan", "/not-opened", "--reference-plan-sha256", PIN, "--startup-only"]
    if mode != "no_output": args += ["--startup-output", str(tmp_path / "wrong")]
    if mode == "native_output": args += ["--output", str(tmp_path / "reference")]
    with pytest.raises(ValueError, match="startup output scope"): r.main(args)


def test_module_and_startup_prefix_have_no_numerical_imports_or_fixed_endpoint_runner():
    tree = ast.parse(PATH.read_bytes()); imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import): imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node, ast.ImportFrom): imports.add((node.module or "").split(".")[0])
    assert not imports & {"numpy", "jax", "equinox", "mosaic", "torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute): assert node.attr not in {"_reference", "ENDPOINT", "native_evaluator"}


class ToyRuntime:
    """Transport stand-in only; it cannot represent a native scientific pass."""
    def __init__(self): self.ids = []; self.keys = []; self.guard_calls = 0
    def guard(self): self.guard_calls += 1
    def ready(self): return {"status": "toy_metadata_ready", "native_model_loaded": False}
    def verify_identity(self): return {"status": "toy_unchanged"}
    def evaluate(self, value, fields):
        self.ids.append(value["id"]); self.keys.append(value["key_data"])
        return {"id": value["id"], "graph_sha256": value["graph_sha256"], "key_data": value["key_data"], "value": 0.0}
    def finish(self): return {"status": "toy_transport_only", "native_model_loaded": False}


def test_stream_keeps_actual_keys_requests_and_control_messages_separate():
    values = [request(), {"action": "verify_identity"}, request(1, key=[998, 123]), {"action": "close"}]
    source = io.StringIO("".join(json.dumps(x) + "\n" for x in values)); output = io.StringIO(); runtime = ToyRuntime()
    protocol, audit = r.serve(runtime, source, output)
    assert runtime.ids == [0, 1] and runtime.keys == [[0, 17], [998, 123]]
    assert protocol.closed and audit["status"] == "toy_transport_only"
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(replies) == 4 and replies[-1]["key_data"] == [998, 123]


@pytest.mark.parametrize("stream", ["", "{}", '{"action":"verify_identity","action":"close"}\n'])
def test_stream_end_unterminated_or_duplicate_message_never_closes_successfully(stream):
    with pytest.raises(ValueError): r.serve(ToyRuntime(), io.StringIO(stream), io.StringIO())
