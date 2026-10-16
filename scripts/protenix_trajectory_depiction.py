"""Probability-weighted raw graph depictions for the detached HPC report.

The hard labels/topology select what is drawn. A shared absolute probability
scale controls visual size only; it never changes occupancy or molecular data.
"""

import io
import os
import xml.etree.ElementTree as ET


DEPICTION_POLICY = {
    "atom_probability": "X[i, argmax(X[i])] (displayed element; occupancy stays fixed)",
    "atom_radius": "0.18 + 0.16 * probability in fixed layout units",
    "atom_label_size": "fixed 10 points, bold element plus slot; independent of probability",
    "bond_probability": "1 - E_upper[pair, NO_BOND] for drawn hard bonds",
    "bond_line_width": "0.8 + 3.2 * probability in typographic points",
    "scale": "absolute 0..1, identical across cases and updates; no frame rescaling",
    "topology": "raw argmax; omit hard NO_BOND pairs; preserve hard bond categories",
    "layout": "fixed slot coordinates; visual sizes are not physical atomic radii",
}


def depict_graph(X, E, step, valid, *, svg=False):
    """Draw the same hard graph with subtle probability-dependent node/edge sizes."""
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Trajectory depiction requires an allocated DTU job")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np

    elements = ("C", "N", "O", "S", "F", "Cl")
    bond_names = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    n = X.shape[0]
    positions = np.array([(1.4 * i, 0.7 * (i % 2)) for i in range(n)])
    labels = X.argmax(-1)
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.96)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, max(1.0, positions[-1, 0]) + 0.5)
    ax.set_ylim(-1.5, 1.4)
    ax.axis("off")
    tooltips = {}
    pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
    for pair, ((i, j), category) in enumerate(zip(pairs, E.argmax(-1), strict=True)):
        category = int(category)
        if category == 0:
            continue  # Do not turn soft mass on absent edges into hard bonds.
        probability = float(1 - E[pair, 0])
        start, end = positions[i], positions[j]
        vector = end - start
        normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
        offsets = {
            1: (0,),
            2: (-0.045, 0.045),
            3: (-0.085, 0, 0.085),
            4: (-0.045, 0.045),
        }
        for line, offset in enumerate(offsets[category]):
            points = np.stack((start, end)) + offset * normal
            gid = f"bond-{i + 1}-{j + 1}-{line}"
            ax.plot(
                points[:, 0],
                points[:, 1],
                color="#38555f",
                linewidth=0.8 + 3.2 * probability,
                linestyle="--" if category == 4 and line == 1 else "-",
                solid_capstyle="round",
                zorder=1,
                gid=gid,
            )
            tooltips[gid] = (
                f"Bond {i + 1}–{j + 1}: P(exists)={probability:.5f}; "
                f"hard category {bond_names[category]}"
            )
    for i, (position, label) in enumerate(zip(positions, labels, strict=True)):
        probability = float(X[i, label])
        gid = f"atom-{i + 1}"
        circle = Circle(
            position,
            radius=0.18 + 0.16 * probability,
            facecolor="#e7f3f1",
            edgecolor="#087f8c",
            linewidth=0.9,
            zorder=2,
            gid=gid,
        )
        ax.add_patch(circle)
        ax.text(
            *position,
            f"{elements[label]}{i + 1}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#102e38",
            zorder=3,
            gid=gid + "-label",
        )
        tooltip = f"Atom {i + 1}: P({elements[label]})={probability:.5f}; occupied slot"
        tooltips[gid] = tooltips[gid + "-label"] = tooltip
    fig.text(
        0.5,
        0.16,
        f"Update {step} · raw argmax · {'sanitizes' if valid else 'INVALID'}",
        ha="center",
        fontsize=9,
        color="#38555f",
    )
    fig.text(
        0.5,
        0.055,
        "Atom size: P(element)  ·  Bond width: P(exists)  ·  Fixed scale",
        ha="center",
        fontsize=7,
        color="#5b7078",
    )
    buffer = io.BytesIO()
    # Real SVG text keeps element labels readable in inline HTML. Glyph-path
    # xlink references can lose their namespace when parsed through innerHTML.
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(buffer, format="svg" if svg else "png", dpi=100)
    plt.close(fig)
    if not svg:
        return buffer.getvalue()
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    root = ET.fromstring(buffer.getvalue())
    for group in root.iter(f"{{{namespace}}}g"):
        if group.get("id") in tooltips:
            title = ET.Element(f"{{{namespace}}}title")
            title.text = tooltips[group.get("id")]
            group.insert(0, title)
    return ET.tostring(root, encoding="unicode")
