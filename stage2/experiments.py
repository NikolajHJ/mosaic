"""Declare the Stage 2 studies as data and run them as one command.

Three studies. ``matched`` is the protocol's experiment order 5.6: the frozen
Stage 1 control, valence only, connectivity, sparsity, and the hardening pair.
``improvements`` is a one-factor-at-a-time sweep: a baseline arm plus one arm
per improvement, each switched on alone, so each effect is attributable.
``hardening`` compares the three ways the gradient can cross the hardening
step against a matched relaxed control, which is what separates "topology
locked" from "the effective step size shrank".

Shared settings come from one experiment YAML; an arm overrides only the keys
it names, and any extra flag on this command line is forwarded to every arm.

Arms run sequentially because they share one GPU. ``--print`` emits the command
lines instead of running anything, which is what to use on a laptop or to hand
the batch to a scheduler.
"""

import argparse
import json
import shutil
from pathlib import Path

from settings import DEFAULT

#: Arms as overrides on :func:`run.add_arguments` defaults.
STUDIES = {
    "matched": [
        ("control", {"mode": "none"}),
        ("A-valence", {"mode": "valence"}),
        ("B-connected", {"mode": "connected"}),
        ("C-sparse-only", {"mode": "sparse_only"}),
        ("C-sparse", {"mode": "sparse"}),
        ("D-relaxed", {"mode": "connected"}),
        ("D-hardening", {"mode": "connected", "hardening": True}),
    ],
    "hardening": [
        ("relaxed-control", {"mode": "connected"}),
        ("inherited", {"mode": "connected", "hardening": True}),
        (
            "passthrough",
            {
                "mode": "connected",
                "hardening": True,
                "hardening_gradient": "passthrough",
            },
        ),
        (
            "rescaled",
            {
                "mode": "connected",
                "hardening": True,
                "hardening_gradient": "rescaled",
            },
        ),
    ],
    "improvements": [
        ("baseline", {"mode": "connected"}),
        ("readout-caps", {"mode": "connected", "readout_caps": True}),
        ("valence-damping", {"mode": "connected", "valence_damping": 0.5}),
        (
            "fluidity-floor",
            {
                "mode": "connected",
                "fluidity_threshold": 0.95,
                "fluidity_strength": 0.05,
            },
        ),
        ("bridge-retry", {"mode": "connected", "bridge_retry": True}),
        ("aromatic-1.0", {"mode": "connected", "aromatic_order": 1.0}),
    ],
}

#: What each improvement is for, printed by ``--list`` and kept in the study file.
RATIONALE = {
    "inherited": "Mosaic's existing transition; its Jacobian attenuates the "
    "gradient by about (1-h) + h/K along the ramp",
    "passthrough": "textbook identity-Jacobian STE; removes that attenuation "
    "at its source",
    "rescaled": "keeps the inherited transform and compensates the magnitude, "
    "separating representation hardening from the step-size effect",
    "readout-caps": "soft valence can hold while the argmax molecule is over-valent",
    "valence-damping": "an exact projection every step may freeze topology changes",
    "fluidity-floor": "protocol 4.3 C.10, premature collapse to near one-hot rows",
    "bridge-retry": "valence reprojection can remove a bridge straight after adding it",
    "aromatic-1.0": "aromatic weight 1.5 is a surrogate; 1.0 is a looser bound",
}


def arm_arguments(name, overrides, args, forwarded):
    """Build the namespace ``run.run`` expects for one arm.

    Settings come from three places, each beating the one before it: the
    experiment file, any flags forwarded on this command line, then the arm's
    own overrides. So an arm differs from the study baseline in exactly the
    keys it names, and the study shares one reviewable file.
    """
    import run as run_module
    from settings import load

    parser = run_module.add_arguments(argparse.ArgumentParser())
    if args.config:
        parser.set_defaults(**load(args.config))
    namespace = parser.parse_args([*forwarded, "--out", str(Path(args.out) / name)])
    namespace.config = args.config
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def command_line(name, namespace, config):
    """The equivalent ``run.py`` invocation, for --print and for the record."""
    flags = ["python stage2/run.py"]
    if config:
        flags.append(f"--config {config}")
    flags += [
        f"--backend {namespace.backend}",
        f"--mode {namespace.mode}",
        f"--out {namespace.out}",
        f"--steps {namespace.steps}",
    ]
    if namespace.backend == "boltz2":
        # Unset means the experiment file has not been pointed at a context yet.
        if namespace.boltz_context:
            flags.append(f"--boltz-context {namespace.boltz_context}")
        if namespace.boltz_checkpoint:
            flags.append(f"--boltz-checkpoint {namespace.boltz_checkpoint}")
        if namespace.allow_local_gpu:
            flags.append("--allow-local-gpu")
    else:
        flags.append(f"--target {namespace.target}")
    if not namespace.gradient_check:
        flags.append("--no-gradient-check")
    if namespace.parent:
        flags.append(f"--parent {namespace.parent}")
    if namespace.hardening:
        flags.append("--hardening")
        flags.append(f"--hardening-gradient {namespace.hardening_gradient}")
    if namespace.readout_caps:
        flags.append("--readout-caps")
    if namespace.bridge_retry:
        flags.append("--bridge-retry")
    if namespace.valence_damping != 1.0:
        flags.append(f"--valence-damping {namespace.valence_damping}")
    if namespace.fluidity_strength:
        flags.append(
            f"--fluidity-threshold {namespace.fluidity_threshold}"
            f" --fluidity-strength {namespace.fluidity_strength}"
        )
    if namespace.aromatic_order != 1.5:
        flags.append(f"--aromatic-order {namespace.aromatic_order}")
    return " ".join(flags)


def compare(out):
    """Collect the arms' diagnostics into one comparison table."""
    from diagnostics import summarize

    out = Path(out)
    rows = []
    for directory in sorted(p for p in out.iterdir() if (p / "trajectory").is_dir()):
        summary = summarize(directory)
        last = summary["per_step"][-1]
        flagged = [f["id"] for f in summary["pitfalls"] if f["verdict"] == "flag"]
        rows.append(
            {
                "arm": directory.name,
                "states": summary["states"],
                "final_score": last.get("score"),
                "final_components": last.get("components"),
                "final_violations": last.get("argmax_valence_violations"),
                "final_sanitized": last.get("sanitized"),
                "valid_connected_states": sum(
                    1
                    for r in summary["per_step"]
                    if r.get("sanitized") and r.get("components") == 1
                ),
                "unique_valid_smiles": len(
                    {
                        r["canonical_smiles"]
                        for r in summary["per_step"]
                        if r.get("sanitized") and r.get("canonical_smiles")
                    }
                ),
                "topology_reversals": summary["reversals"]["reversals"],
                "flagged": flagged,
            }
        )
    return rows


def print_comparison(rows):
    if not rows:
        print("no completed arms yet")
        return
    header = (
        f"{'arm':<18}{'score':>10}{'comp':>6}{'viol':>6}"
        f"{'valid':>7}{'smiles':>8}{'revers':>8}  flagged"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        score = "-" if row["final_score"] is None else f"{row['final_score']:.5f}"
        print(
            f"{row['arm']:<18}{score:>10}{row['final_components'] or 0:>6}"
            f"{row['final_violations'] or 0:>6}{row['valid_connected_states']:>7}"
            f"{row['unique_valid_smiles']:>8}{row['topology_reversals']:>8}"
            f"  {', '.join(row['flagged']) or 'none'}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(STUDIES), default="matched")
    parser.add_argument(
        "--out", type=Path, help="study directory; not needed for --list"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT,
        help="experiment YAML shared by every arm",
    )
    parser.add_argument("--only", nargs="+", help="run only these arm names")
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="print the commands instead of running them",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="skip running and just rebuild the comparison",
    )
    parser.add_argument("--list", action="store_true", dest="list_only")
    # Anything else is forwarded to every arm's own parser, so a quick sweep is
    # the study plus one flag and nothing needs declaring twice.
    args, forwarded = parser.parse_known_args()

    if args.list_only:
        for study, arms in STUDIES.items():
            print(f"\n{study}")
            for name, overrides in arms:
                note = RATIONALE.get(name)
                print(
                    f"  {name:<18}{json.dumps(overrides)}"
                    + (f"\n  {'':<18}why: {note}" if note else "")
                )
        return

    if args.out is None:
        parser.error("--out is required unless you pass --list")
    arms = [a for a in STUDIES[args.study] if not args.only or a[0] in args.only]
    out = Path(args.out)

    if not args.compare_only:
        out.mkdir(parents=True, exist_ok=True)
        planned = [
            (name, arm_arguments(name, overrides, args, forwarded))
            for name, overrides in arms
        ]
        shared = planned[0][1]
        (out / "study.json").write_text(
            json.dumps(
                {
                    "study": args.study,
                    "experiment_file": None
                    if args.config is None
                    else str(args.config),
                    "backend": shared.backend,
                    "target": shared.target if shared.backend == "protenix" else None,
                    "parent": None if shared.parent is None else str(shared.parent),
                    "steps": shared.steps,
                    "forwarded_flags": forwarded,
                    "arms": [
                        {
                            "name": name,
                            "command": command_line(name, namespace, args.config),
                            "rationale": RATIONALE.get(name),
                        }
                        for name, namespace in planned
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        if args.print_only:
            for name, namespace in planned:
                print(command_line(name, namespace, args.config))
            return
        import run as run_module

        for name, namespace in planned:
            directory = Path(namespace.out)
            if (directory / "run_summary.json").exists():
                print(f"SKIP {name}: already complete", flush=True)
                continue
            if directory.exists():
                # An arm with no summary never finished: interrupted, or it
                # failed. Its partial artifacts cannot be analyzed and would
                # otherwise make `run.py` refuse the directory, so a resumed
                # study would die on the first casualty instead of redoing it.
                print(f"REDO {name}: discarding an unfinished attempt", flush=True)
                shutil.rmtree(directory)
            try:
                run_module.run(namespace)
            except Exception as error:  # noqa: BLE001 - see below
                # Deliberately broad. This is the top of an unattended queue:
                # whatever one arm raises, the remaining arms must still run,
                # and narrowing the type would only reintroduce the failure
                # mode this exists to prevent. The arm's own traceback and
                # attempt_failure.json are already on disk.
                print(f"FAILED {name}: {type(error).__name__}: {error}", flush=True)

    rows = compare(out)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print_comparison(rows)
    print(f"\nwrote {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
