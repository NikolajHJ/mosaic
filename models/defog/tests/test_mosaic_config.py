from contextlib import ExitStack

import jax.numpy as jnp
import numpy as np
import pytest
from mosaic_defog import Graph, Prediction


class Teacher:
    def __init__(self, **options):
        self.options, self.calls, self.closed = options, [], False
        self.identity = {"scope": "test teacher"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def predict(self, graph, *, time):
        self.calls.append(time)
        return Prediction(np.asarray(graph.X), np.asarray(graph.E_upper), time)


def test_configuration_composes_existing_preparation_and_schedules(
    monkeypatch, tmp_path
):
    from mosaic_defog import integration

    teachers = []

    def load(**options):
        teacher = Teacher(**options)
        teachers.append(teacher)
        return teacher

    monkeypatch.setattr(integration, "DeFoGTeacher", load)
    graph = Graph(jnp.full((3, 6), 1 / 6), jnp.full((3, 5), 0.2))

    def previous(native, steps):
        return lambda graph, step: (
            lambda graph, key=None: (
                native(graph, key=key)[0],
                {"forward_hardness": step / steps},
            )
        )

    with ExitStack() as resources:
        prepare, identity = integration.mosaic_prepare_loss(
            {
                "source": "source",
                "checkpoint": "model.ckpt",
                "python": "env/python",
                "weight": {"kind": "linear", "start": 0, "end": 0.1},
                "time": {"kind": "linear", "start": 0.2, "end": 0.95},
            },
            resources,
            directory=tmp_path,
            prepare_loss=previous,
        )
        objective = prepare(lambda g, key=None: (g.X.sum(), {}), 4)(graph, 3)
        value, aux = objective(graph)
        assert float(value) == pytest.approx(3)
        assert aux["forward_hardness"] == 0.75
        assert aux["defog_weight"] == pytest.approx(0.1)
        assert teachers[0].calls == pytest.approx([0.95])
        assert teachers[0].options["source"] == tmp_path / "source"
        assert identity["target"].startswith("detached")
    assert teachers[0].closed


@pytest.mark.parametrize("mode", ["none", "valence", "connected", "sparse_only", "sparse"])
@pytest.mark.parametrize("hardening", [False, True])
def test_actual_stage2_yaml_hook_preserves_projection(
    monkeypatch, tmp_path, mode, hardening
):
    pipeline = pytest.importorskip("mosaic.small_molecule.Stage2.pipeline")
    design = pytest.importorskip("mosaic.small_molecule.Stage1.pipeline")
    from mosaic_defog import integration

    monkeypatch.setenv("LSB_JOBID", "mock-test-no-native-model")
    monkeypatch.setattr(integration, "DeFoGTeacher", Teacher)
    calls = []

    def optimize(*args, after_update, prepare_loss, stage_details):
        graph = Graph(jnp.full((3, 6), 1 / 6), jnp.full((3, 5), 0.2))
        objective = prepare_loss(lambda g, key=None: (g.X.sum(), {}), 3)(graph, 1)
        aux = objective(graph, key=None)[1]
        assert "defog_loss" in aux
        if hardening:
            assert float(aux["forward_hardness"]) == 0.5
        else:
            assert "forward_hardness" not in aux
        assert stage_details["operators"]["mode"] == mode
        assert stage_details["defog"]["teacher"]["scope"] == "test teacher"
        assert (after_update is None) == (mode == "none")
        calls.append(True)

    monkeypatch.setattr(design, "optimize", optimize)
    stage = {
        "operators": {"mode": mode},
        "defog": {"source": "source", "checkpoint": "model.ckpt"},
        "hardening": {"enabled": hardening, "gradient": "passthrough"},
    }
    pipeline.run(
        {"initialization": {"checkpoint": "shared.npz"}},
        stage,
        tmp_path / "general.yaml",
        tmp_path / "stage.yaml",
        tmp_path / "output",
    )
    assert calls == [True]


@pytest.mark.parametrize("model_name", ["boltz2", "protenix-tiny"])
@pytest.mark.parametrize("use_guidance", [False, True])
@pytest.mark.parametrize("use_hardening", [False, True])
def test_actual_design_loop_keeps_native_scores_and_forward_budget(
    monkeypatch, tmp_path, model_name, use_guidance, use_hardening
):
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace

    import jax

    pipeline = pytest.importorskip("mosaic.small_molecule.Stage1.pipeline")
    experiment = pytest.importorskip("mosaic.small_molecule.core.experiment")
    boltz = pytest.importorskip("mosaic.small_molecule.core.boltz2.setup")
    diagnostics = pytest.importorskip("mosaic.small_molecule.Stage1.diagnostics")
    features = pytest.importorskip("mosaic.small_molecule.core.features")
    protenix = pytest.importorskip("mosaic.small_molecule.core.protenix")
    state = pytest.importorskip("mosaic.small_molecule.core.state")
    hardening = pytest.importorskip("mosaic.small_molecule.Stage2.hardening")
    from mosaic_defog import Guidance

    native_calls = []
    sign = 1 if model_name == "boltz2" else -1
    score_name = "affinity_pred_value" if sign == 1 else "contact_score"

    def native(g, *, key=None):
        # Count executed forwards, including an existing hardening JIT, rather
        # than counting Python traces of that JIT.
        jax.debug.callback(lambda _: native_calls.append(True), 0, ordered=True)
        score = jnp.mean(g.X**2) + 0.1 * jnp.mean(g.E_upper**2)
        return sign * score, {score_name: score}

    class Targets(Teacher):
        def predict(self, graph, *, time):
            self.calls.append(time)
            return Prediction(
                np.eye(6, dtype=np.float32)[np.zeros(len(graph.X), int)],
                np.eye(5, dtype=np.float32)[np.zeros(len(graph.E_upper), int)],
                time,
            )

    @contextmanager
    def open_model(*args):
        yield native, {"model": "CPU toy"}, {"context": "toy"}

    class Model:
        def __init__(self):
            self.identities = {"model": "CPU toy"}

        @classmethod
        def load(cls, **kwargs):
            return cls()

        def build_loss(self, **kwargs):
            return native

    def readout(graph):
        return state.graph_from_hard(
            np.asarray(graph.X).argmax(-1), np.asarray(graph.E_upper).argmax(-1)
        ), {"scope": "argmax only; no chemistry dependency"}

    # Exercise the actual loop with a differentiable toy native model. Native
    # assets, GPU setup and chemistry conversion are outside this regression.
    monkeypatch.setattr(boltz, "open_affinity", open_model)
    monkeypatch.setattr(protenix, "FrozenProtenixTiny", Model)
    monkeypatch.setattr(protenix, "ligand_pae_gate_prerequisites", lambda *args: {})
    monkeypatch.setattr(features, "prepare_context", lambda *args, **kwargs: (
        SimpleNamespace(identities={"context": "toy"}), None
    ))
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(diagnostics, "raw_readout", readout)
    monkeypatch.setattr(experiment, "resolve_target", lambda *args: {
        "structure_sha256": "toy", "chain": "A", "protein_sequence": "A"
    })
    general_path, stage_path = tmp_path / "general.yaml", tmp_path / "stage.yaml"
    general_path.write_text("{}")
    stage_path.write_text("{}")
    general = {
        "protein": {"structure": "toy.pdb", "chain": "A"}, "n_atoms": 4,
        "epochs": 3, "learning_rate": 0.01, "model": model_name,
        "initialization": {"random": {}},
    }
    teacher = Targets()

    def prepare(native, steps):
        previous = (
            hardening.hardening_factory(native, steps, "passthrough")
            if use_hardening else lambda graph, step: native
        )

        def for_state(graph, step):
            active = previous(graph, step)
            return Guidance(teacher, weight=2).prepare(active, graph, step) if use_guidance else active

        return for_state

    with ExitStack() as resources:
        pipeline.optimize(
            general, {"stage": "stage2", "readout_every": 2}, general_path,
            stage_path, tmp_path / "out", resources,
            prepare_loss=prepare if use_guidance or use_hardening else None,
        )
    trace = [json.loads(line) for line in (tmp_path / "out/trace.jsonl").read_text().splitlines()]
    readouts = [json.loads(line) for line in (tmp_path / "out/hard_readouts.jsonl").read_text().splitlines()]
    assert len(trace) == 4
    assert len(teacher.calls) == (len(trace) if use_guidance else 0)
    # One model evaluation per state and the original hard forward per readout.
    # Only hardening needs an additional raw-soft readout evaluation.
    assert len(native_calls) == len(trace) + len(readouts) * (2 if use_hardening else 1)
    with np.load(tmp_path / "out/trajectory.npz") as trajectory:
        scores = np.mean(trajectory["X"]**2, axis=(1, 2)) + 0.1 * np.mean(
            trajectory["E_upper"]**2, axis=(1, 2)
        )
        forward_X, forward_E = trajectory["X"], trajectory["E_upper"]
        if use_hardening:
            hardness = np.minimum(np.arange(len(trace)) / (general["epochs"] - 1), 1)[:, None, None]
            forward_X = forward_X + hardness * (np.eye(6)[forward_X.argmax(-1)] - forward_X)
            forward_E = forward_E + hardness * (np.eye(5)[forward_E.argmax(-1)] - forward_E)
        forward_scores = np.mean(forward_X**2, axis=(1, 2)) + 0.1 * np.mean(
            forward_E**2, axis=(1, 2)
        )
    for row, score in zip(trace, forward_scores, strict=True):
        assert row[score_name] == pytest.approx(float(score))
        if use_guidance:
            assert row["loss"] > row["defog_native_loss"]
            assert row["defog_native_loss"] == pytest.approx(sign * float(score))
        else:
            assert row["loss"] == pytest.approx(sign * float(score))
            assert not any(name.startswith("defog_") for name in row)
        if use_hardening:
            assert row["forward_hardness"] == min(row["state"] / 2, 1)
        else:
            assert "forward_hardness" not in row
    for row in readouts:
        assert row[f"soft_{score_name}"] == pytest.approx(float(scores[row["state"]]))
    assert json.loads((tmp_path / "out/manifest.json").read_text())["status"] == "complete"
