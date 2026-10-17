"""Fast SVG for saved graphs, using the report's fixed probability size rules.

This avoids constructing a plotting figure for every playback frame. It uses
the same raw argmax categories, slot coordinates, radii and line widths as the
PNG snapshot renderer; no chemistry conversion or probability changes occur.
"""

from html import escape
import math


def graph_svg(X, E, step, valid):
    elements = ("C", "N", "O", "S", "F", "Cl")
    names = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    n = len(X)
    scale = min(70.0, 500.0 / (1.4 * max(n - 1, 1)))
    left = (600 - 1.4 * (n - 1) * scale) / 2
    positions = [(1.4 * i, 0.7 * (i % 2)) for i in range(n)]

    def point(x, y):
        return left + x * scale, 116 - y * scale

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" '
        'width="600" height="240" role="img" '
        'aria-label="Saved raw molecular graph">',
        '<rect width="600" height="240" fill="white"/>',
    ]
    pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
    for index, ((i, j), category) in enumerate(zip(pairs, E.argmax(-1), strict=True)):
        category = int(category)
        if category == 0:
            continue
        probability = float(1 - E[index, 0])
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        offsets = {
            1: (0,),
            2: (-0.045, 0.045),
            3: (-0.085, 0, 0.085),
            4: (-0.045, 0.045),
        }
        title = escape(
            f"Bond {i + 1}–{j + 1}: P(exists)={probability:.5f}; hard category {names[category]}"
        )
        for line, offset in enumerate(offsets[category]):
            ax, ay = point(x0 + offset * nx, y0 + offset * ny)
            bx, by = point(x1 + offset * nx, y1 + offset * ny)
            dash = ' stroke-dasharray="6 4"' if category == 4 and line == 1 else ""
            parts.append(
                f'<g id="bond-{i + 1}-{j + 1}-{line}"><title>{title}</title>'
                f'<line x1="{ax:.5f}" y1="{ay:.5f}" x2="{bx:.5f}" y2="{by:.5f}" '
                f'stroke="#38555f" stroke-width="{(0.8 + 3.2 * probability) * 100 / 72:.5f}" '
                f'stroke-linecap="round"{dash}/></g>'
            )
    for i, category in enumerate(X.argmax(-1)):
        label = elements[int(category)]
        probability = float(X[i, category])
        x, y = point(*positions[i])
        radius = (0.18 + 0.16 * probability) * scale
        title = escape(f"Atom {i + 1}: P({label})={probability:.5f}; occupied slot")
        parts.append(
            f'<g id="atom-{i + 1}"><title>{title}</title>'
            f'<circle cx="{x:.5f}" cy="{y:.5f}" r="{radius:.5f}" '
            'fill="#e7f3f1" stroke="#087f8c" stroke-width="1.25"/>'
            f'<text x="{x:.5f}" y="{y:.5f}" text-anchor="middle" '
            'dominant-baseline="central" font-family="DejaVu Sans,Arial,sans-serif" '
            f'font-size="13.33333" font-weight="bold" fill="#102e38">{label}{i + 1}</text></g>'
        )
    parts.extend(
        [
            '<text x="300" y="196" text-anchor="middle" font-family="Arial,sans-serif" '
            f'font-size="12" fill="#38555f">Update {int(step)} · raw argmax · '
            f"{'sanitizes' if valid else 'INVALID'}</text>",
            '<text x="300" y="223" text-anchor="middle" font-family="Arial,sans-serif" '
            'font-size="10" fill="#5b7078">Atom size: P(element) · Bond width: P(exists) · Fixed scale</text>',
            "</svg>",
        ]
    )
    return "".join(parts)
