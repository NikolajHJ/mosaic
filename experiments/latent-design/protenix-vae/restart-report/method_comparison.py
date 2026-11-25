"""Make a short PDF from saved corrected adaptive/continuous/independent panels.

Run ``figures`` with matplotlib, then ``pdf`` with reportlab. Reads JSON only;
does not import the design models, run scoring or change experiment outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


ARMS = ("adaptive", "continuous", "independent")
LABELS = {"adaptive": "Adaptive", "continuous": "Continuous", "independent": "Independent"}
COLORS = {"adaptive": "#087F68", "continuous": "#C87922", "independent": "#4F68B1"}


def summarize(snapshot):
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    panels = []
    for name, panel in sorted(data["panels"].items()):
        result, run, waves = panel["result"], panel["run"], panel["waves"]
        if (result["status"] != "completed" or panel["launch"]["returncode"] != 0
                or run["fresh_scoring_policy"] != "equal_allowance"
                or [row["cycle"] for row in waves] != list(range(1, 9))):
            raise ValueError("Report requires completed corrected eight-wave panels")
        rows = {}
        for arm in ARMS:
            if (result["completed_updates_by_arm"][arm] != 48_000
                    or result["relaxed_calls_by_arm"][arm] != 48_096
                    or sum(w["fresh_attempt_allowance_by_arm"][arm] for w in waves) != 576):
                raise ValueError("Unexpected method budget")
            leaders = panel["final_archive"][arm]
            best, mean = min(p["fresh_loss"] for p in leaders), statistics.mean(p["fresh_loss"] for p in leaders[:3])
            if abs(best - waves[-1]["best"][arm]) > 1e-10 or abs(mean - waves[-1]["top_three_mean"][arm]) > 1e-10:
                raise ValueError("Final archive and wave summary disagree")
            rows[arm] = dict(best=best, top_three_mean=mean, diverse=waves[-1]["diverse_high_scorers"][arm],
                attempts=result["fresh_attempts_by_arm"][arm], successes=result["fresh_scored_by_arm"][arm],
                optimization_hours=sum(w["optimization_seconds_by_arm"][arm] for w in waves)/3600,
                fresh_hours=sum(w["fresh_seconds_by_arm"][arm] for w in waves)/3600)
        panels.append(dict(panel=name.removeprefix("seed-").upper(), seed=run["seed"], job_id=panel["job_id"],
            gpu={"gpul40s": "L40S", "gpuv100": "V100"}.get(panel["queue"], panel["queue"]),
            source=panel["launch"]["source_archive_sha256"], rows=rows, waves=waves,
            elapsed_hours=result["elapsed_seconds"]/3600))
    if len(panels) != 3 or len({p["source"] for p in panels}) != 1:
        raise ValueError("Expected three panels from one corrected source")
    wins = {metric: sum(p["rows"]["adaptive"][metric] < min(p["rows"][arm][metric] for arm in ARMS[1:])
                       for p in panels) for metric in ("best", "top_three_mean")}
    wins["diverse"] = sum(p["rows"]["adaptive"]["diverse"] > max(p["rows"][arm]["diverse"] for arm in ARMS[1:]) for p in panels)
    return dict(collected_at=data["collected_at"], snapshot=str(snapshot.resolve()),
        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), panels=panels, wins=wins,
        mean_best={arm: statistics.mean(p["rows"][arm]["best"] for p in panels) for arm in ARMS},
        mean_diverse={arm: statistics.mean(p["rows"][arm]["diverse"] for p in panels) for arm in ARMS})


def figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#3A4A5C", "xtick.color": "#506071", "ytick.color": "#506071",
        "axes.edgecolor": "#CBD5DF", "grid.color": "#E5EBF0", "savefig.facecolor": "white"})
    styles = {"adaptive": ("-", "o"), "continuous": (":", "^"), "independent": ("--", "s")}
    all_y = [wave["best"][arm] for p in summary["panels"] for wave in p["waves"] for arm in ARMS]
    low, high = min(all_y)-.025, max(all_y)+.025
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.75), sharex=True, sharey=True)
    for ax, p in zip(axes, summary["panels"], strict=True):
        ax.axvspan(0, 6, color="#EFF3F6", zorder=0)
        for arm in ARMS:
            x = [w["cumulative_updates_by_arm"][arm]/1000 for w in p["waves"]]
            y = [w["best"][arm] for w in p["waves"]]
            line, marker = styles[arm]
            ax.plot(x, y, color=COLORS[arm], linestyle=line, marker=marker, markersize=4.5,
                    linewidth=2, label=LABELS[arm], drawstyle="steps-post")
            ax.annotate(f"{y[-1]:.4f}", (48, y[-1]), xytext=(7, 0), textcoords="offset points",
                        va="center", color=COLORS[arm], fontsize=9, fontweight="bold")
        ax.set_title(f"Seed {p['panel']}  |  {p['seed']}  |  {p['gpu']}", loc="left", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Best fresh loss")
        ax.set_xlim(0, 54)
        ax.set_ylim(low, high)
        ax.grid(axis="y", linewidth=.7)
        ax.set_xticks([0, 6, 12, 24, 36, 48])
    axes[-1].set_xlabel("Cumulative optimizer updates per method (thousands)", labelpad=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 1.005), ncol=3, frameon=False)
    fig.subplots_adjust(left=.115, right=.89, top=.92, bottom=.085, hspace=.42)
    fig.savefig(output/"learning-curves.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.15))
    x = np.arange(3)
    for i, arm in enumerate(ARMS):
        values = [p["rows"][arm]["diverse"] for p in summary["panels"]]
        bars = ax.bar(x+(i-1)*.23, values, .21, label=LABELS[arm], color=COLORS[arm], zorder=3)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x, [f"Seed {p['panel']}" for p in summary["panels"]])
    ax.set_ylabel("Diverse low-loss molecules")
    ax.set_ylim(0, max(p["rows"][arm]["diverse"] for p in summary["panels"] for arm in ARMS)*1.24)
    ax.grid(axis="y", linewidth=.7, zorder=0)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.28), fontsize=8.8)
    fig.subplots_adjust(left=.10, right=.99, bottom=.16, top=.80)
    fig.savefig(output/"diverse-yield.png", dpi=240)
    plt.close(fig)


def pdf(summary, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

    for name, file in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf")):
        pdfmetrics.registerFont(TTFont(name, "C:/Windows/Fonts/"+file))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial", boldItalic="Arial-Bold")
    ink, muted = colors.HexColor("#142A3D"), colors.HexColor("#536477")
    teal, light = colors.HexColor(COLORS["adaptive"]), colors.HexColor("#ECF6F2")
    width = A4[0]-80
    styles = {
        "eyebrow": ParagraphStyle("eyebrow", fontName="Arial-Bold", fontSize=8.4, leading=12, textColor=teal, spaceAfter=9),
        "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=25, leading=29, textColor=ink, spaceAfter=9),
        "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=13, leading=17, textColor=ink, spaceBefore=13, spaceAfter=7),
        "body": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14, textColor=ink, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=8.8, leading=12.5, textColor=muted, spaceAfter=7),
        "tiny": ParagraphStyle("tiny", fontName="Arial", fontSize=7.6, leading=10.3, textColor=muted, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=9, leading=12, textColor=ink),
    }
    def para(text, style="body"):
        return Paragraph(text, styles[style])
    def table(data, col_widths, header=True, adaptive_rows=()):
        converted = [[para(str(v), "cell") for v in row] for row in data]
        t = Table(converted, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
        commands = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#BDC9D3"))]
        if header: commands += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF1F5"))]
        for row in adaptive_rows: commands.append(("BACKGROUND", (0, row), (-1, row), light))
        t.setStyle(TableStyle(commands))
        return t

    target = output/"search-method-comparison.pdf"
    doc = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=40, rightMargin=40,
                           topMargin=40, bottomMargin=39, title="Adaptive molecular search versus simple baselines",
                           author="DBMol exploratory research", pageCompression=1)
    nav = (("Results", "page-1"), ("Trajectories", "page-2"), ("Budgets", "page-3"))
    def frame(c, d):
        page = d.page
        c.saveState()
        c.bookmarkPage(f"page-{page}")
        c.addOutlineEntry(nav[page-1][0], f"page-{page}", level=0)
        c.setStrokeColor(colors.HexColor("#D9E3EA")); c.line(40, 29, A4[0]-40, 29)
        c.setFont("Arial", 7.5); c.setFillColor(muted)
        c.drawString(40, 17, "DBMol exploratory research  |  16 November 2026")
        x = 327
        for label, destination in nav:
            c.setFillColor(teal if destination == f"page-{page}" else muted)
            c.drawString(x, 17, label)
            length = pdfmetrics.stringWidth(label, "Arial", 7.5)
            c.linkRect("", destination, (x, 13, x+length, 26), relative=0, thickness=0)
            x += length+13
        c.setFillColor(muted); c.drawRightString(A4[0]-40, 17, f"{page} / 3")
        c.restoreState()

    panels = summary["panels"]
    improvement = summary["mean_best"]["independent"]-summary["mean_best"]["adaptive"]
    story = [para("BRD4 / PROTENIX  -  CORRECTED MATCHED-BUDGET STUDY", "eyebrow"),
             para("Adaptive search vs<br/>simple baselines", "title"),
             para("Three completed paired seeds. 48,000 optimizer updates per method per seed.", "small")]
    cards = table([[f"<b>{summary['wins']['best']} / 3 paired wins</b><br/>best fresh loss",
                    f"<b>{improvement:.4f} lower loss</b><br/>mean gap vs independent",
                    f"<b>{summary['mean_diverse']['adaptive']:.0f} vs {summary['mean_diverse']['independent']:.0f} molecules</b><br/>mean diverse low-loss yield"]],
                  [width/3]*3, header=False, adaptive_rows=(0,))
    story += [cards, Spacer(1, 8), para("Adaptive also wins top-three mean loss and diverse low-loss yield in all three seeds. This is encouraging evidence for archive feedback on this one target.")]
    story += [para("What was compared", "h2"), table([
        ["<b>Adaptive restarts</b>", "12 common initial starts; then 4 archive-selected parents x 3 children per wave. Two slots continue parents exactly; other slots use latent perturbations."],
        ["<b>One continuous path</b>", "One initialization optimized for 48,000 uninterrupted updates. The 96 windows of 500 updates are reporting boundaries, with exact latent continuation."],
        ["<b>Independent starts</b>", "96 separate random Gaussian VAE latents, each optimized for 500 updates. No feedback from previous winners; starts are not validity-filtered."],
    ], [125, width-125], header=False)]
    story += [para("Final quality at 48,000 updates", "h2")]
    rows = [["Seed", "Method", "Best loss", "Top-3 mean", "Diverse hits"]]
    for p in panels:
        for i, arm in enumerate(ARMS):
            r = p["rows"][arm]
            rows.append([p["panel"] if i == 0 else "", f'<font color="{COLORS[arm]}">{LABELS[arm]}</font>',
                         f"{r['best']:.5f}", f"{r['top_three_mean']:.5f}", str(r["diverse"])])
    story += [table(rows, [45, 135, 100, 115, width-395], adaptive_rows=(1, 4, 7)), Spacer(1, 7),
              para("Loss: fresh Protenix contact score; lower is better. Diverse hits: valid scored molecules with loss &lt;= 2.30, greedily filtered to pairwise Morgan similarity &lt;= 0.60. Counts are per seed, not pooled unique molecules.", "small"),
              para("<b>Budget caveat:</b> optimizer updates and maximum fresh-scoring allowances match. Actual fresh-score usage and total consumed compute differ; see page 3.", "small"), PageBreak()]

    story += [para("PROGRESS AT SHARED UPDATE BUDGETS", "eyebrow"), para("Where the advantage appears", "title"),
              para("Best freshly scored valid molecule retained by each method. All panels use the same vertical scale; lower is better.")]
    story += [Image(str(output/"learning-curves.png"), width=width, height=width*6.75/7.2), Spacer(1, 7),
              para("Points are completed-wave readouts, every 6,000 updates. Stair steps show the most recently saved archive best, rather than measurements at every update. Adaptive and independent share all 12 starts in wave 1 (grey region); parent-feedback differences begin in wave 2. Continuous uses the first common start.", "small"),
              para("Improvement is uneven: some seed/method combinations plateau for several waves. The comparison supports better final outcomes at this budget, not guaranteed improvement in every generation.", "small"), PageBreak()]

    story += [para("COMPUTE, YIELD AND INTERPRETATION", "eyebrow"), para("What the matched budget means", "title"),
              para("Each method receives <b>48,000 updates, 48,096 relaxed scorer observations and at most 576 fresh-score attempts</b> per seed. Unused fresh allowances are not transferred or filled with repeated scoring.")]
    costs = [["Seed", "Method", "Fresh attempts", "Successful", "Opt. hours", "Fresh hours"]]
    for p in panels:
        for i, arm in enumerate(ARMS):
            r = p["rows"][arm]
            costs.append([p["panel"] if i == 0 else "", f'<font color="{COLORS[arm]}">{LABELS[arm]}</font>',
                          str(r["attempts"]), str(r["successes"]), f"{r['optimization_hours']:.2f}", f"{r['fresh_hours']:.2f}"])
    story += [table(costs, [44, 103, 100, 85, 90, width-422], adaptive_rows=(1, 4, 7)), Spacer(1, 6),
              para("Seed A ran on L40S; B/C on V100. Methods share the same GPU within a seed. Hours are logged arm optimization and fresh-scoring times; shared setup, selection overhead and other unattributed work are excluded. Adaptive executes first and may absorb native warmup. These are descriptive timings, not a strict GPU-hour match.", "small"),
              para("Low-loss diversity at the final budget", "h2"),
              Image(str(output/"diverse-yield.png"), width=width, height=width*2.15/7.2), Spacer(1, 6),
              para("What we can conclude", "h2"),
              para("<b>Archive feedback earns its branch slots here.</b> Adaptive beats fresh independent starts in all three seeds with similar or fewer consumed fresh attempts: 532 vs 576, 566 vs 564, and 474 vs 563. One long path produces far fewer new scoring candidates and the weakest final archives.", "small"),
              para("<b>Limits:</b> one BRD4/Protenix setting and three seeds. Fresh hard-molecule scoring uses new conformers and a different contact reduction from the relaxed optimization objective. These scores establish neither experimental affinity nor synthesis feasibility. Adaptive also adds proposal-screening and frontier-selection work.", "small"),
              para("Setup: frozen epoch-45 VAE; LR 60, step cap 0.25, 25 occupied slots, chemistry/valence weights 1, temperature 1, no optimization noise. Report excludes the superseded minimum-pool-scoring campaign. Seeds A/B/C: 541101 / 651101 / 761101.", "tiny"),
              para("Evidence: corrected jobs 29998584 / 29998585 / 29998586; eight completed wave-result.json records and final archives per panel, collected "+summary["collected_at"][:10]+". Source snapshot "+panels[0]["source"][:16]+"... . Full records: data/remote.json; derived values: data/summary.json.", "tiny")]
    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("figures", "pdf"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.snapshot)
    if args.phase == "figures":
        (args.output/"data").mkdir(exist_ok=True)
        (args.output/"data"/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        figures(summary, args.output)
        print("Saved matched-method figures and validated summary")
    else:
        print(pdf(summary, args.output))


if __name__ == "__main__":
    main()
