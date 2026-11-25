"""Direct atom/bond probability trajectories and paired held-out decision records."""

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase


RATE = {"boltz2": 1.0, "protenix": 60.0}
PAIR_I, PAIR_J = np.triu_indices(25, 1)
ELEMENTS = (6, 7, 8, 16, 9, 17)
BOND_TYPES = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
EVAL_STEPS = 8
TRAIN_STEPS = 16
FRESH_EDIT_SLOTS = 4


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hard_smiles(graph):
    atoms, bonds = (x.argmax(axis=1) for x in graph)
    mol = Chem.RWMol()
    for label in atoms:
        mol.AddAtom(Chem.Atom(ELEMENTS[label]))
    for k in np.flatnonzero(bonds):
        a, b, category = int(PAIR_I[k]), int(PAIR_J[k]), bonds[k]
        mol.AddBond(a, b, BOND_TYPES[category])
        if category == 4:
            mol.GetAtomWithIdx(a).SetIsAromatic(True)
            mol.GetAtomWithIdx(b).SetIsAromatic(True)
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    with rdBase.BlockLogs():
        if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def soften(row, stream):
    # Independent soft initializations share one hard founder and the same law.
    rng = np.random.default_rng([row["seed"], stream])
    mixing = (rng.uniform(.01, .20), rng.uniform(.001, .04))
    labels = (row["atoms"], row["bonds"])
    graph = tuple(((1 - m) * np.eye(k)[idx] + m / k).astype(np.float32)
                  for idx, k, m in zip(labels, (6, 5), mixing))
    if hard_smiles(graph) != row["smiles"]:
        raise ValueError(f"Hard founder mismatch at {row['seed']}")
    return graph


def refine(backend, graph, steps, rate):
    # The frozen Stage 3 operators are NumPy/SciPy and work with both model runtimes.
    from operators import Stage2Config, apply_operators, project_simplex_rows

    config = Stage2Config(mode="connected")
    states = [graph]
    losses = []
    for _ in range(steps):
        loss, grad_atom, grad_bond = backend.gradient(*graph)
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite relaxed objective")
        proposed = []
        for x, g in zip(graph, (grad_atom, grad_bond)):
            if x.shape != g.shape or not np.isfinite(g).all():
                raise ValueError("Nonfinite or incompatible native derivative")
            proposed.append(project_simplex_rows(x - rate * g))
        atom, bond, _ = apply_operators(*proposed, config)
        graph = atom.astype(np.float32), bond.astype(np.float32)
        states.append(graph)
        losses.append(float(loss))
    return states, losses


def candidates(states, founder):
    # All observations are read. Selection uses time of first occurrence only.
    unique = {}
    for step, graph in enumerate(states):
        smiles = hard_smiles(graph)
        if smiles and smiles != founder:
            unique.setdefault(smiles, step)
    items = list(unique.items())
    if len(items) > FRESH_EDIT_SLOTS:
        which = np.linspace(0, len(items) - 1, FRESH_EDIT_SLOTS).round().astype(int)
        items = [items[i] for i in which]
    return [dict(smiles=s, step=i) for s, i in items], len(unique)


def train_trajectories(backend, rows, folder, rate):
    records = []
    for row in rows:
        seed, founder = row["seed"], row["smiles"]
        start = soften(row, 0)
        states, losses = refine(backend, start, TRAIN_STEPS, rate)
        edits, distinct = candidates(states, founder)
        # Store the ordered graph behind each candidate as well as the fallback iterate.
        with (folder / f"{seed}.npz").open("xb") as fp:
            np.savez_compressed(fp, atom=np.stack([g[0] for g in states]),
                                bond=np.stack([g[1] for g in states]))
        records.append(dict(seed=seed, founder=founder, edits=edits, distinct=distinct, losses=losses))
        print(f"teacher {seed} {len(edits)} fresh edit slots used", flush=True)
    return records


def train_student(train, train_rows):
    from student import fit

    scores = read(train / "scores.json")
    records = read(train / "panel.json")
    if [(r["seed"], r["founder"]) for r in records] != [(r["seed"], r["smiles"]) for r in train_rows]:
        raise ValueError("Teacher trajectories do not match the expected training split")
    pairs, labels = [], []
    for row in records:
        base = scores[row["founder"]]
        better = [c for c in row["edits"] if base is not None and scores[c["smiles"]] is not None
                  and base - scores[c["smiles"]] > .005]
        step = min(better, key=lambda c: scores[c["smiles"]])["step"] if better else EVAL_STEPS
        with np.load(train / f"{row['seed']}.npz", allow_pickle=False) as stored:
            graph = stored["atom"], stored["bond"]
            pairs.append((tuple(x[0] for x in graph), tuple(x[step] for x in graph)))
        labels.append(dict(seed=row["seed"], step=step, fresh_improvement=bool(better)))
    return fit(pairs), labels


def eval_trajectories(backend, rows, model, rate):
    from student import matched_random, predict

    records = []
    for row in rows:
        seed, founder = row["seed"], row["smiles"]
        start = soften(row, 0)
        learned = predict(model, start)
        random = matched_random(start, learned, np.random.default_rng([seed, 2]))
        initial = {"learned": learned, "random": random, "restart": soften(row, 1)}
        arms = {}
        for arm, graph in initial.items():
            states, losses = refine(backend, graph, EVAL_STEPS, rate)
            edits, distinct = candidates(states, founder)
            norms = [float(np.linalg.norm(x - y)) for x, y in zip(graph, start)]
            arms[arm] = dict(calls=len(losses), jump_l2=norms, edits=edits, distinct=distinct, losses=losses)
        if not np.allclose(arms["learned"]["jump_l2"], arms["random"]["jump_l2"], atol=2e-5, rtol=2e-5):
            raise ValueError("Applied probability displacements have different norms")
        records.append(dict(seed=seed, founder=founder, arms=arms))
        print(f"held-out {seed} completed three arms", flush=True)
    return records


def summarize(records, scores):
    summary = []
    for r in records:
        base = scores[r["founder"]]
        arms = {}
        for name, panel in r["arms"].items():
            values = [scores[c["smiles"]] for c in panel["edits"]]
            usable = [v for v in values if v is not None]
            best = min(usable + ([] if base is None else [base]), default=None)
            arms[name] = dict(calls=panel["calls"], best_loss=best, candidates=len(values),
                              scored=len(usable), jump_l2=panel["jump_l2"])
        summary.append(dict(seed=r["seed"], founder=r["founder"], baseline_loss=base, arms=arms))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "eval", "collect"))
    parser.add_argument("--model", required=True, choices=list(RATE))
    parser.add_argument("--cohort", required=True, type=int, choices=(0, 1))
    parser.add_argument("--founders", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    all_rows = read(args.founders)
    if (len(all_rows) != 112 or len({r["smiles"] for r in all_rows}) != 112
            or len({r["seed"] for r in all_rows}) != 112):
        raise ValueError("Expected globally disjoint training and test founders")
    rows = [r for r in all_rows if r["cohort"] == args.cohort]
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if len(train_rows) != 24 or len(test_rows) != 32:
        raise ValueError("Incomplete training or held-out panel")
    if args.phase == "collect":
        folder = args.run / "eval"
        records = read(folder / "panel.json")
        if [(r["seed"], r["founder"]) for r in records] != [(r["seed"], r["smiles"]) for r in test_rows]:
            raise ValueError("Scored panel differs from held-out founders")
        rows = summarize(records, read(folder / "scores.json"))
        write(folder / "summary.json", dict(model=args.model, cohort=args.cohort, rows=rows))
        return
    if args.model == "protenix":
        from protenix_native import make_backend
    else:
        from boltz_native import make_backend
    folder = args.run / args.phase
    folder.mkdir(exist_ok=False)
    backend = make_backend(1701 + args.cohort, folder)
    try:
        rate = RATE[args.model]
        if args.phase == "train":
            records = train_trajectories(backend, train_rows, folder, rate)
            panels = records
        else:
            model, labels = train_student(args.run / "train", train_rows)
            write(folder / "training-targets.json", labels)
            records = eval_trajectories(backend, test_rows, model, rate)
            panels = [p for r in records for p in r["arms"].values()]
    finally:
        frozen = backend.close()
        write(folder / "native.json", {"identity": backend.identity, "frozen_after": frozen})
    write(folder / "panel.json", records)
    requests = {r["founder"] for r in records}
    requests.update(c["smiles"] for p in panels for c in p["edits"])
    write(folder / "requests.json", sorted(requests))


if __name__ == "__main__":
    main()
