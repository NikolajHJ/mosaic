"""Compose a portable review from previously rendered, authenticated reports.

This is standard-library presentation work. It copies saved scalar/probability
records and existing HPC plot bytes; it never imports a model, evaluates
chemistry, generates scientific plots, or changes a scientific outcome.
"""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re


EARLY = Path(
    "results/hpc/20261016-carbon-chain-100/"
    "20261016-gradient-evidence-carbon-chain-report-04"
)
BRD4 = Path(
    "results/hpc/20261017-brd4-parallel/complete-report-01/"
    "20261017-brd4-best-primal-report-02"
)
SOURCES = {
    "original": {
        "report": EARLY / "runs/20261016-carbon-chain-report-04/payload/report",
        "manifest": EARLY / "evidence_manifest.json",
        "manifest_name": "runs/20261016-carbon-chain-report-04/payload/report",
        "plot": "loss.png",
    },
    "brd4": {
        "report": BRD4 / "20261017-brd4-best-primal-report-02/payload/report",
        "manifest": BRD4 / "evidence_manifest.json",
        "manifest_name": "20261017-brd4-best-primal-report-02/payload/report",
        "plot": "loss.png",
    },
    "chemistry": {
        "report": Path("results/hpc/20261019-brd4-chemistry/report-view-01/report"),
        "plot": "loss_components.png",
    },
    "ramp": {
        "report": Path("results/hpc/20261019-brd4-delayed-chemistry-report/viewer"),
        "plot": "loss_components.png",
    },
    "boltz": {
        "report": Path("results/hpc/20261020-brd4-boltz-affinity-report/viewer"),
        "plot": "loss_components.png",
    },
}

# These are six milestones, not the six lowest losses. Targets, parameterization
# and objectives differ. Editorial observations quote authenticated saved analyses.
MILESTONES = [
    {
        "id": "original-protenix", "source": "original", "case": 1,
        "title": "Original Protenix", "subtitle": "The first carbon-chain optimization",
        "date": "16 Oct 2026", "job": "29735673", "updates": 100, "slots": 10,
        "model": "Protenix-Tiny", "metric": "contact", "metric_label": "Pocket contact NLL",
        "target": "Synthetic 20-residue protein", "initialization": "30% chain/uniform",
        "lr": "0.01", "chemistry": "None", "color": "#b96723", "valid": True,
        "policy": "Strict projected-probability updates; native bond features observe existence only.",
        "finding": "The contact loss falls while the hard molecule stays decane.",
        "explanation": "The relaxed probabilities move before any argmax atom or bond changes. This established the optimization plumbing on an engineering protein fixture.",
        "caveat": "Ten slots and a synthetic pocket: its NLL is not comparable to the later BRD4 contact scores or Boltz affinity.",
        "figure_caption": "Original HPC figure, unchanged: both 15% and 30% starts. This milestone selects the 30% trajectory; the original legend identifies its curve.",
        "validity": "101 / 101 saved readouts sanitized; hard decane throughout.",
        "jumps": [[0, "Start"], [25, "25"], [50, "50"], [100, "Final"]],
        "document": "docs/carbon_chain_100.md",
    },
    {
        "id": "brd4-contact", "source": "brd4", "case": 2,
        "title": "BRD4, contact only", "subtitle": "More contact, invalid chemistry",
        "date": "18 Oct 2026", "job": "29745503", "updates": 5000, "slots": 6,
        "model": "Protenix-Tiny", "metric": "contact", "metric_label": "Whole-target contact loss",
        "target": "BRD4-BD1 / 3MXF", "initialization": "50% chain/uniform",
        "lr": "0.03", "chemistry": "None", "color": "#7257b4", "valid": False,
        "policy": "Fresh best_primal arm, up to ten halving trials. Other curves in the original figure include declared strict prefixes/continuations.",
        "finding": "A stronger contact score does not ensure a chemically valid graph.",
        "explanation": "The selected 50% start ends with overbonded fluorine. All six initialization controls in this family ended invalid, despite improved contact scores.",
        "caveat": "Whole-target contact is a proximity proxy. This run did not optimize a measured affinity or a specific crystallographic binding pocket.",
        "figure_caption": "Original six-initialization HPC figure, unchanged. The selected 50% trajectory is purple. All six original curves and continuation markers remain visible.",
        "validity": "Final saved graph invalid; 615 / 5,001 readouts sanitized in this selected arm.",
        "jumps": [[0, "Start"], [615, "First invalid"], [2500, "2,500"], [5000, "Final"]],
        "document": "docs/brd4_best_primal_continuation.md",
    },
    {
        "id": "first-chemistry", "source": "chemistry", "case": 0,
        "title": "First chemical-repair run", "subtitle": "Valid, but still hexane",
        "date": "19 Oct 2026", "job": "29753152", "updates": 5000, "slots": 6,
        "model": "Protenix-Tiny", "metric": "contact", "metric_label": "Whole-target contact loss",
        "target": "BRD4-BD1 / 3MXF", "initialization": "50% chain/uniform",
        "lr": "0.03", "chemistry": "Repair 0.5 + element JS 0.1", "color": "#16838b", "valid": True,
        "policy": "Logit/softmax optimization; all 5,000 applied steps improved their own fixed-reference surrogate.",
        "finding": "The first chemistry-guided run preserves a valid seed molecule.",
        "explanation": "Every saved hard molecule remains hexane and every repair loss is zero. The run validates integration, but does not demonstrate active repair of this trajectory.",
        "caveat": "The parameterization changed from projected probabilities to logits. This is not an isolated chemistry-loss ablation. The loss is rule-based repair, not DeFoG.",
        "figure_caption": "Original HPC figure, unchanged: the new chemistry run and its historical contact-only control. Changing repair targets do not define one fixed total objective.",
        "validity": "5,001 / 5,001 saved states valid; repair zero throughout.",
        "jumps": [[0, "Start"], [1250, "1,250"], [2500, "2,500"], [5000, "Final"]],
        "document": "docs/brd4_chemistry_5000.md",
    },
    {
        "id": "delayed-mix", "source": "ramp", "case": 0,
        "title": "Protenix, delayed chemistry", "subtitle": "10,000 updates from the 50% mix",
        "date": "19 Oct 2026", "job": "29753667", "updates": 10000, "slots": 6,
        "model": "Protenix-Tiny", "metric": "contact", "metric_label": "Whole-target contact loss",
        "target": "BRD4-BD1 / 3MXF", "initialization": "50% chain/uniform",
        "lr": "0.03", "chemistry": "Repair 0.001 to 50; element JS 0.1", "color": "#3265ae", "valid": True,
        "policy": "Repair weight 0.001 through applied update 5,000, then geometric ramp to 50 at 10,000. All 10,000 steps improved their fixed-reference surrogate.",
        "finding": "A longer run improves contact without changing the hard hexane seed.",
        "explanation": "The chemistry coefficient grows, but its raw repair term stays zero in all 10,001 states. Increasing the coefficient cannot activate a zero term.",
        "caveat": "The final graph is valid hexane. The stronger contact score does not establish BRD4 affinity or discovery of a different ligand.",
        "figure_caption": "Original paired HPC figure, unchanged: 50% mix and random initialization, including contact, repair components and coefficient schedule.",
        "validity": "10,001 / 10,001 saved states valid; repair zero throughout.",
        "jumps": [[0, "Start"], [5000, "5,000"], [7500, "7,500"], [10000, "Final"]],
        "document": "docs/brd4_delayed_chemistry_report.md",
    },
    {
        "id": "delayed-random", "source": "ramp", "case": 1,
        "title": "Protenix, random start", "subtitle": "A new valid endpoint",
        "date": "19 Oct 2026", "job": "29753668", "updates": 10000, "slots": 6,
        "model": "Protenix-Tiny", "metric": "contact", "metric_label": "Whole-target contact loss",
        "target": "BRD4-BD1 / 3MXF", "initialization": "Dirichlet(1), seed 20261019",
        "lr": "0.03", "chemistry": "Repair 0.001 to 50; element JS 0.1", "color": "#478043", "valid": True,
        "policy": "Same delayed chemistry schedule as the mix arm. All 10,000 steps improved their own fixed-reference surrogate.",
        "finding": "Random initialization reaches a connected, valid FCCC(F)F endpoint.",
        "explanation": "Repair falls from 1.4736 to zero. The first saved native/schema-valid state appears at 8,959; 1,012 states meet that criterion. Validity is not monotone afterward.",
        "caveat": "This is a valid output of a single exploratory run, not evidence that it is a known or useful BRD4 ligand. The optimized score is contact, not affinity.",
        "figure_caption": "Original paired HPC figure, unchanged. This milestone selects the random arm; the 50% arm remains visible in the figure as a control.",
        "validity": "1,012 / 10,001 saved states native/schema valid; final FCCC(F)F.",
        "jumps": [[0, "Start"], [5000, "5,000"], [8959, "First valid"], [10000, "Final"]],
        "document": "docs/brd4_delayed_chemistry_report.md",
    },
    {
        "id": "boltz-affinity", "source": "boltz", "case": 0,
        "title": "Boltz-2 affinity", "subtitle": "Affinity gains meet a validity boundary",
        "date": "20 Oct 2026", "job": "29759212", "updates": 10000, "slots": 6,
        "model": "Boltz-2", "metric": "affinity", "metric_label": "Fixed-pose affinity score",
        "target": "BRD4-BD1 / fixed predicted pose", "initialization": "50% chain/uniform",
        "lr": "0.03", "chemistry": "Repair 0.001 to 50; element JS 0.1", "color": "#b74b5b", "valid": False,
        "policy": "All 10,000 steps improved their fixed-reference surrogate. Native parity/gradient gates passed for this mix arm. The random arm is not certified or submitted.",
        "finding": "Affinity improves, then gives ground as chemistry guidance increases.",
        "explanation": "The best relaxed score is -0.1036 at state 8,294, whose hard graph is invalid. State 9,639 is valid SCCCCCl at an associated relaxed score of 0.5465. The final graph is invalid again.",
        "caveat": "Boltz affinity_pred_value uses log10(IC50 in micromolar), lower better. Here it scores a relaxed graph in a fixed pose, with zero recycles and no molecular-weight correction. It is not an experimental affinity or a fresh prediction of the hard molecule.",
        "figure_caption": "Original Boltz HPC figure, unchanged: affinity, separately logged binder probability, chemistry and recorded update behavior. No Protenix score is overlaid on the affinity axis.",
        "validity": "1,686 / 10,001 saved states valid; 280 validity switches over the last 500 updates.",
        "jumps": [[0, "Start"], [8294, "Best affinity / invalid"], [9639, "Valid SCCCCCl"], [9999, "Last valid"], [10000, "Final / invalid"]],
        "document": "docs/brd4_boltz_balance_review.md",
    },
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def selected_legacy(text, index):
    """Decode cases individually so a large six-case report stays bounded."""
    offset = text.index("const DATA=") + len("const DATA=")
    if not text.startswith('{"cases":[', offset):
        raise ValueError("Unexpected legacy report structure")
    offset += len('{"cases":[')
    decoder = json.JSONDecoder()
    for number in range(index + 1):
        while text[offset] in " ,\r\n":
            offset += 1
        case, offset = decoder.raw_decode(text, offset)
        if number == index:
            return case
        del case
    raise ValueError("Selected legacy case absent")


def authenticate_source(root, source):
    directory = root / source["report"]
    if "manifest" in source:
        proof_path = root / source["manifest"]
        record = json.loads(proof_path.read_text(encoding="utf-8"))
        files = record["files"]
        prefix = source["manifest_name"] + "/"
    else:
        proof_path = directory / "report_receipt.json"
        record = json.loads(proof_path.read_text(encoding="utf-8"))
        if record.get("complete") is not True:
            raise ValueError("Source report is incomplete")
        files, prefix = record["files"], ""
    bound = {}
    for name in ["report.html", source["plot"]]:
        raw = (directory / name).read_bytes()
        expected = files[prefix + name]
        if len(raw) != expected["bytes"] or digest(raw) != expected["sha256"]:
            raise ValueError("Previously authenticated report bytes differ: " + name)
        bound[name] = {"bytes": len(raw), "sha256": digest(raw)}
    return {
        "source_report": str(source["report"] / "report.html"),
        "source_byte_proof": str(proof_path.relative_to(root)),
        "source_byte_proof_sha256": digest(proof_path.read_bytes()),
        "files": bound,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, payloads, figures, provenance = [], [], {}, {}
    for source_key, source in SOURCES.items():
        provenance[source_key] = authenticate_source(root, source)
        directory = root / source["report"]
        raw = (directory / "report.html").read_bytes()
        text = raw.decode("utf-8")
        figures[source_key] = "data:image/png;base64," + base64.b64encode(
            (directory / source["plot"]).read_bytes()
        ).decode()
        items = [m for m in MILESTONES if m["source"] == source_key]
        data = None
        if source_key not in {"original", "brd4"}:
            match = re.search(
                r'<script id="saved-data" type="application/json">(.*?)</script>', text, re.S
            )
            if not match:
                raise ValueError("Saved source payload absent")
            data = json.loads(match[1])
            if data.get("complete") is not True:
                raise ValueError("Source payload is not complete")
        for item in items:
            if data is None:
                case = selected_legacy(text, item["case"])
                if case.get("source_job_id", item["job"]) != item["job"]:
                    raise ValueError("Legacy job identity differs")
                original = case["states"]
                frames = [
                    {**{k: v for k, v in f.items() if k not in {"svg", "E_upper", "loss"}},
                     "E": f["E_upper"], "contact": f["loss"]}
                    for f in original
                ]
                identity = {k: v for k, v in case.items() if k != "states"}
            else:
                case = data["studies"][item["case"]]
                if case["job_id"] != item["job"]:
                    raise ValueError("Native job identity differs")
                frames = case["frames"]
                identity = {k: v for k, v in case.items() if k != "frames"}
                identity.update({k: data.get(k) for k in ["model_digest", "context_digest"]})
            if len(frames) != item["updates"] + 1:
                raise ValueError("Wrong saved budget")
            for step, frame in enumerate(frames):
                if frame["step"] != step or len(frame["X"]) != item["slots"]:
                    raise ValueError("Saved state continuity/slot schema differs")
                if len(frame["E"]) != item["slots"] * (item["slots"] - 1) // 2:
                    raise ValueError("Saved edge schema differs")
                if any(len(row) != 6 for row in frame["X"]) or any(len(row) != 5 for row in frame["E"]):
                    raise ValueError("Saved categorical schema differs")
            if frames[-1]["readout"]["sanitized"] is not item["valid"]:
                raise ValueError("Editorial endpoint disagrees with saved readout")
            # The roundtrip below is byte/schema integrity work. No scientific
            # statistic, chemical validity or model prediction is recomputed.
            payload = packed({"frames": frames, "identity": identity})
            if json.loads(payload)["frames"] != frames:
                raise ValueError("Presentation serialization changed saved values")
            compressed = gzip.compress(payload, compresslevel=6, mtime=0)
            payloads.append(
                '<script type="application/octet-stream" id="payload-'
                + item["id"] + '">' + base64.b64encode(compressed).decode() + "</script>"
            )
            record = dict(item)
            record.update({
                "states": len(frames),
                "initial": frames[0][item["metric"]],
                "final": frames[-1][item["metric"]],
                "endpoint": frames[-1]["readout"]["canonical_smiles"],
                "payload_sha256": digest(payload),
                "compressed_sha256": digest(compressed),
                "compressed_bytes": len(compressed),
                "source_identity": identity,
            })
            records.append(record)
            print(json.dumps({"milestone": item["id"], "states": len(frames), "compressed_bytes": len(compressed)}), flush=True)
            del payload, frames, case, identity
        del text, raw, data
    # Keep the six milestones in editorial chronology, independent of source I/O.
    records.sort(key=lambda record: [m["id"] for m in MILESTONES].index(record["id"]))
    meta = {
        "title": "From contact to affinity", "date": "21 October 2026",
        "period": "16-20 October 2026", "milestones": records, "figures": figures,
        "provenance": provenance,
        "scope": "Saved-report composition only; no new model, chemistry, numerical tests or scientific plots.",
        "selection": "Six chronological Mosaic milestones, not a loss ranking. No completed Mosaic DeFoG experiment found; none included.",
    }
    template = (root / "scripts/templates/mosaic_milestones.html").read_text(encoding="utf-8")
    html = template.replace("__MILESTONE_META__", packed(meta).decode().replace("<", "\\u003c"))
    html = html.replace("__MILESTONE_PAYLOADS__", "\n".join(payloads))
    (output / "report.html").write_text(html, encoding="utf-8", newline="\n")
    proof = {
        "version": "mosaic-six-milestones-presentation-v1",
        "scope": meta["scope"], "selection": meta["selection"],
        "sources": provenance, "milestones": [{k: v for k, v in r.items() if k != "source_identity"} for r in records],
        "assembler_sha256": digest(Path(__file__).read_bytes()),
        "template_sha256": digest((root / "scripts/templates/mosaic_milestones.html").read_bytes()),
        "files": {"report.html": {"bytes": (output / "report.html").stat().st_size,
                                  "sha256": digest((output / "report.html").read_bytes())}},
        "new_numerical_validation": False, "new_scientific_plot_generation": False,
        "browser_qa": "pending",
    }
    (output / "presentation_receipt.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "states": [r["states"] for r in records], "report": proof["files"]["report.html"]}))


if __name__ == "__main__":
    main()
