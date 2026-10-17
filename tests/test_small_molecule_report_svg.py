"""Detached playback keeps raw labels/topology and absolute probability sizes."""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


def test_svg_preserves_argmax_and_probability_sizes_without_repair():
    path = Path(__file__).resolve().parents[1] / "scripts/protenix_trajectory_svg.py"
    spec = importlib.util.spec_from_file_location("trajectory_svg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X = np.array([[1, 0, 0, 0, 0, 0], [0.2, 0.8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    E = np.array([[0.1, 0, 0.9, 0, 0], [0.6, 0.4, 0, 0, 0], [0.1, 0, 0, 0, 0.9]])
    before = (X.copy(), E.copy())
    root = ET.fromstring(module.graph_svg(X, E, 17, False))
    ns = {"s": "http://www.w3.org/2000/svg"}
    groups = {g.get("id"): g for g in root.findall("s:g", ns)}
    assert {k for k in groups if k.startswith("bond-")} == {
        "bond-1-2-0",
        "bond-1-2-1",
        "bond-2-3-0",
        "bond-2-3-1",
    }
    assert [groups[f"atom-{i}"].find("s:text", ns).text for i in (1, 2, 3)] == [
        "C1",
        "N2",
        "Cl3",
    ]
    assert float(groups["atom-1"].find("s:circle", ns).get("r")) == pytest.approx(
        0.34 * 70
    )
    assert float(groups["atom-2"].find("s:circle", ns).get("r")) == pytest.approx(
        (0.18 + 0.16 * 0.8) * 70
    )
    assert groups["bond-2-3-1"].find("s:line", ns).get("stroke-dasharray") == "6 4"
    assert "INVALID" in "".join(root.itertext())
    assert np.array_equal(X, before[0]) and np.array_equal(E, before[1])
