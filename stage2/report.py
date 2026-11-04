"""Open the results of a Stage 2 run or study.

Two halves, because Stage 2 needs both.

The existing model-free renderer already does the hard part: a self-contained
``report.html`` with the argmax molecule over time (slider, playback, jumps
between identity changes, fixed atom slots with changed atoms highlighted), the
full element-probability matrix per step, the six per-atom probability curves,
the all-atom heatmap and the IID sampling panels. That is reused as-is, not
reimplemented.

What it does not have is the Stage 2 result. Since bond order is a null
gradient direction for this adapter, contact score cannot judge these arms;
feasibility does. So this also writes a feasibility comparison across arms:
components, discrete valence violations, relaxed-state fluidity, bond mass and
the relaxed-to-discrete gap, plus the pitfall verdicts per arm.

Runs on a laptop. The renderer half needs the lightweight report environment
(NumPy, matplotlib, RDKit, gemmi); the feasibility half needs NumPy and
matplotlib only.
"""

import argparse
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts/render_small_molecule_report.py"

PANELS = (
    ("components", "Connected components of the argmax graph", "components"),
    ("argmax_valence_violations", "Atoms over their valence ceiling (argmax)", "atoms"),
    ("atom_max_probability", "Mean max atom probability (fluidity)", "probability"),
    ("expected_degree", "Expected degree (bond mass)", "degree"),
    ("loss", "Objective (minimized; lower is better on either backend)", "loss"),
    ("score", "Backend score", "score"),
    # Hardening attenuates the gradient reaching p, so these two are how you
    # tell a locked topology from a shrinking step size.
    ("gradient_norm_X", "Raw atom gradient norm", "norm"),
    ("gradient_norm_E", "Raw bond gradient norm", "norm"),
)


def arm_directories(target):
    """A single run, or every arm of a study directory."""
    target = Path(target)
    if (target / "trajectory").is_dir():
        return [target]
    return sorted(p for p in target.iterdir() if (p / "trajectory").is_dir())


def feasibility_figures(summaries, output):
    """One panel per Stage 2 criterion, every arm overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for key, title, ylabel in PANELS:
        series = [
            (
                s["run"],
                [(r["step"], r[key]) for r in s["per_step"] if r.get(key) is not None],
            )
            for s in summaries
        ]
        series = [(name, points) for name, points in series if points]
        if not series:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 3.4), dpi=150)
        for name, points in series:
            axis.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                linewidth=1.2,
                label=name,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("update")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.5)
        if len(series) > 1:
            axis.legend(fontsize=7, frameon=False)
        figure.tight_layout()
        for suffix in (".png", ".svg"):
            figure.savefig(output / f"{key}{suffix}")
        plt.close(figure)
        figures.append(f"{key}.png")
    return figures


def feasibility_markdown(summaries, figures):
    backends = {(s.get("config") or {}).get("backend") for s in summaries}
    lines = [
        "# Stage 2 feasibility comparison",
        "",
        "Judge these arms on feasibility -- components, discrete valence",
        "violations, sanitized fraction and the relaxed-to-discrete gap -- not on",
        "the objective alone. The objective column is the minimized loss, so",
        "lower is better on either backend.",
        "",
    ]
    if "protenix" in backends:
        lines += [
            "One arm here ran on Protenix-Tiny, whose adapter passes only bond",
            "existence. Bond order is a null gradient direction on that backend, so",
            "the valence projection acts on something its objective cannot see. On",
            "Boltz-2 bond order is differentiable and the two genuinely interact.",
            "",
        ]
    lines += [
        (
            "| arm | mode | improvements | final score | components | violations"
            " | valid+connected | unique SMILES | reversals |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        last = summary["per_step"][-1]
        config = (summary.get("config") or {}).get("stage2") or {}
        valid = sum(
            1
            for r in summary["per_step"]
            if r.get("sanitized") and r.get("components") == 1
        )
        smiles = {
            r["canonical_smiles"]
            for r in summary["per_step"]
            if r.get("sanitized") and r.get("canonical_smiles")
        }
        score = last.get("score")
        lines.append(
            f"| {summary['run']} | {config.get('mode', '-')} "
            f"| {', '.join(config.get('active_improvements') or []) or 'none'} "
            f"| {'-' if score is None else f'{score:.5f}'} "
            f"| {last.get('components', '-')} "
            f"| {last.get('argmax_valence_violations', '-')} "
            f"| {valid} | {len(smiles)} | {summary['reversals']['reversals']} |"
        )
    lines += ["", "## Pitfall verdicts", ""]
    for summary in summaries:
        lines += [f"### {summary['run']}", ""]
        for finding in summary["pitfalls"]:
            mark = {"ok": "ok", "flag": "**FLAG**", "n/a": "n/a"}[finding["verdict"]]
            lines.append(
                f"- {mark} `{finding['id']}` ({finding['source']}): {finding['detail']}"
            )
        lines.append("")
    if figures:
        lines += ["## Figures", ""] + [f"- `{name}`" for name in figures] + [""]
    return "\n".join(lines)


def render_trajectories(target, output, args):
    """Hand the saved runs to the existing model-free renderer."""
    command = [
        sys.executable,
        str(RENDERER),
        str(target),
        "--output",
        str(output),
        "--samples",
        str(args.samples),
        "--checkpoints",
        str(args.checkpoints),
        "--max-browser-states",
        str(args.max_browser_states),
    ]
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        print(
            f"\nrenderer exited {completed.returncode}. The feasibility comparison\n"
            "below is still written. The renderer needs the report environment:\n"
            "  .venv-report/bin/python -m pip install -r scripts/requirements-report.txt\n"
            "  .venv-report/bin/python stage2/report.py ...",
            flush=True,
        )
    return completed.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="a Stage 2 run or study directory")
    parser.add_argument("--output", type=Path, help="defaults to TARGET/report")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--checkpoints", type=int, default=9)
    parser.add_argument("--max-browser-states", type=int, default=201)
    parser.add_argument(
        "--skip-trajectories",
        action="store_true",
        help="feasibility comparison only, no RDKit needed",
    )
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()

    from diagnostics import summarize

    directories = arm_directories(args.target)
    if not directories:
        raise SystemExit(f"no run directories under {args.target}")
    output = args.output or Path(args.target) / "report"
    output.mkdir(parents=True, exist_ok=True)

    summaries = [summarize(directory) for directory in directories]
    (output / "diagnostics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    figures = feasibility_figures(summaries, output)
    page = output / "feasibility.md"
    page.write_text(feasibility_markdown(summaries, figures) + "\n")

    rendered = False
    if not args.skip_trajectories:
        rendered = render_trajectories(args.target, output / "trajectories", args)

    print(f"\nwrote {page}")
    for name in figures:
        print(f"wrote {output / name}")
    if rendered:
        print(f"wrote {output / 'trajectories' / 'report.html'}")
    if args.open_browser:
        html = output / "trajectories" / "report.html"
        webbrowser.open((html if html.exists() else page).resolve().as_uri())


if __name__ == "__main__":
    main()
