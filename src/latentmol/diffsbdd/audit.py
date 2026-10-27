"""Held-out native denoiser audit of a development-selected chemical mapping.

The first distinct audit identities are fixed before any conformer generation
or network output is observed. Two independently seeded ETKDG conformers are
validation geometries, not poses admitted to the alternating optimization loop.
No molecule is replaced after a conformer or denoiser failure. All numerical
execution belongs to an authenticated, allocated HPC stage.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.data import sha256_file
from latentmol.representation import HardGraphBatch
from latentmol.train import state_digest
from .bridge import ATOMIC_NUMBER, PILOT_ELEMENTS, exact_native_probabilities
from .native import _replace_gaussian
from .paired_data import (check_deadline, deterministic_seed, load_dataset,
                          vae_identity, write_json)


VERSION = "diffsbdd-heldout-native-denoiser-audit-v1"
TIMES = (0.0, 0.04, 0.10, 0.20)
MODES = ("learned", "exact_six_conditioned", "exact_native_full")
COMPONENTS = ("coordinates", "chemistry", "mean_coordinates", "mean_chemistry")


def select_audit_rows(data, count=200):
    """Select distinct decoded identities in saved order without score filtering."""
    if type(count) is not int or count < 1:
        raise ValueError("Audit identity count must be a positive integer")
    selected, seen = [], set()
    for index, (split, identity) in enumerate(zip(data["split"], data["identity"])):
        if split != "audit" or identity in seen:
            continue
        seen.add(identity)
        selected.append(index)
        if len(selected) == count:
            break
    return selected


def _frozen(modules, *, strict=True):
    result = {}
    for name, module in modules.items():
        if strict and (module.training or any(p.requires_grad or p.grad is not None for p in module.parameters())):
            raise ValueError("Denoiser audit requires frozen eval modules: " + name)
        result[name] = dict(state_sha256=state_digest(module.state_dict()), training=module.training,
            trainable_parameters=sum(p.requires_grad for p in module.parameters()),
            parameters_with_gradients=sum(p.grad is not None for p in module.parameters()))
    return result


def _canonical_molecule_and_map(data, row, spec):
    """Verify the saved mapping, including charges/bonds on symmetric graphs."""
    identity = data["identity"][row]
    encoded = encode_smiles(identity, spec)
    if encoded["smiles"] != identity:
        raise ValueError("Audit canonical molecular identity differs from paired data")
    molecule = Chem.MolFromSmiles(identity)
    if molecule is None:
        raise ValueError("Audit identity is not a sanitized molecule")
    active = torch.nonzero(data["active"][row], as_tuple=False).flatten()
    mapping = data["slot_to_canonical"][row, active].long()
    count = molecule.GetNumAtoms()
    if (len(active) != count or sorted(mapping.tolist()) != list(range(count))
            or bool((data["slot_to_canonical"][row, ~data["active"][row]] != -1).any())):
        raise ValueError("Audit slot map does not cover exactly the canonical atoms")
    for local, slot in enumerate(active.tolist()):
        canonical = int(mapping[local])
        if (int(data["atoms"][row, slot]) != int(encoded["atom_types"][canonical])
                or int(data["charges"][row, slot]) != int(encoded["charges"][canonical])):
            raise ValueError("Audit slot map changes element or formal charge")
        for other_local, other_slot in enumerate(active.tolist()):
            if int(data["bonds"][row, slot, other_slot]) != int(
                    encoded["bond_types"][canonical, int(mapping[other_local])]):
                raise ValueError("Audit slot map changes a typed bond")
    return molecule, active, mapping


def conformer_positions(molecule, seed):
    """One deterministic conformer attempt, with explicit Hs and no relaxation."""
    mol = Chem.AddHs(Chem.Mol(molecule))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed) % (2**31 - 1)
    parameters.clearConfs = True
    parameters.useRandomCoords = False
    code = AllChem.EmbedMolecule(mol, parameters)
    if code != 0 or mol.GetNumConformers() != 1:
        raise ValueError("ETKDGv3 conformer generation failed: " + str(code))
    heavy = Chem.RemoveHs(mol)
    if (heavy.GetNumAtoms() != molecule.GetNumAtoms()
            or Chem.MolToSmiles(heavy, isomericSmiles=False)
            != Chem.MolToSmiles(molecule, isomericSmiles=False)):
        raise ValueError("Conformer generation changed heavy-atom identity")
    positions = torch.tensor(heavy.GetConformer().GetPositions(), dtype=torch.float32)
    if not bool(positions.isfinite().all()):
        raise ValueError("Conformer positions are nonfinite")
    return positions


def _native_noise(core, ligand, pocket, time, noise):
    """Use the unchanged native noising/centering operation with explicit noise."""
    xh_lig = torch.cat((ligand["x"], ligand["one_hot"]), -1)
    xh_pocket = torch.cat((pocket["x"], pocket["one_hot"]), -1)
    calls = []

    def provider(size, device):
        if calls or tuple(size) != tuple(noise.shape) or torch.device(device) != noise.device:
            raise RuntimeError("Native audit requested unexpected Gaussian draws")
        calls.append(tuple(size))
        return noise

    with _replace_gaussian(core, provider):
        noisy, shifted_pocket, _ = core.noised_representation(
            xh_lig, xh_pocket, ligand["mask"], pocket["mask"], core.gamma(time))
    if len(calls) != 1:
        raise RuntimeError("Native audit did not consume its exact common-noise draw")
    return noisy, shifted_pocket


def _error(predicted, reference):
    if predicted.shape != reference.shape or not bool(predicted.isfinite().all() & reference.isfinite().all()):
        raise ValueError("Native denoiser returned malformed or nonfinite epsilon")
    return dict(mse=float((predicted.double() - reference.double()).square().mean()),
                target_power=float(reference.double().square().mean()),
                maximum_absolute_error=float((predicted - reference).abs().max()))


def _native_transition_mean(core, noisy, pocket, time, ligand_mask, pocket_mask, level):
    """Run the actual native transition with zero Gaussian innovation.

    Native joint centering is linear, so zero innovation gives its conditional
    mean at t>0. At t=0 the native terminal coordinate mean and deterministic
    hard categorical readout are reported in their native output units.
    """
    calls = []

    def zeros(size, device):
        if calls or tuple(size) != tuple(noisy.shape):
            raise RuntimeError("Native transition-mean audit requested unexpected noise")
        calls.append(tuple(size))
        return torch.zeros(tuple(size), device=device, dtype=noisy.dtype)

    with _replace_gaussian(core, zeros):
        if level > 0:
            previous = torch.clamp(time - 1 / core.T, min=0)
            result, _ = core.sample_p_zs_given_zt(previous, time, noisy, pocket, ligand_mask, pocket_mask)
        else:
            x, h, _, _ = core.sample_p_xh_given_z0(noisy, pocket, ligand_mask, pocket_mask, 1)
            result = torch.cat((x, h.to(x)), -1)
    if len(calls) != 1:
        raise RuntimeError("Native transition-mean audit did not consume exactly one zero innovation")
    return result


def _percentile(values, fraction=0.95):
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_errors(rows, expected_identities, *, max_nrmse=0.05, max_p95=0.10):
    """Equal identity pooling after averaging its two conformers, at each time."""
    grouped = {}
    for row in rows:
        for mode, outputs in row["errors"].items():
            for component, metric in outputs.items():
                key = (mode, row["time"], component, row["identity"])
                grouped.setdefault(key, []).append({**metric, "conformer": row["conformer"]})
    by_mode, checks = {}, {}
    for mode in MODES:
        modes = by_mode.setdefault(mode, {})
        for level in TIMES:
            components = modes.setdefault(f"{level:g}", {})
            for component in COMPONENTS:
                identity_metrics = []
                for (found_mode, found_level, found_component, identity), observations in grouped.items():
                    if (found_mode, found_level, found_component) != (mode, level, component):
                        continue
                    mse = sum(value["mse"] for value in observations) / len(observations)
                    power = sum(value["target_power"] for value in observations) / len(observations)
                    identity_metrics.append(dict(identity=identity, conformers=len(observations), mse=mse,
                        two_independent_conformers=len(observations) == 2 and
                            {value["conformer"] for value in observations} == {0, 1},
                        target_power=power, nrmse=math.sqrt(mse / max(power, 1e-12)),
                        maximum_absolute_error=max(value["maximum_absolute_error"] for value in observations)))
                mean_mse = (sum(value["mse"] for value in identity_metrics) / len(identity_metrics)
                            if identity_metrics else None)
                mean_power = (sum(value["target_power"] for value in identity_metrics) / len(identity_metrics)
                              if identity_metrics else None)
                nrmse = math.sqrt(mean_mse / max(mean_power, 1e-12)) if identity_metrics else None
                p95 = _percentile([value["nrmse"] for value in identity_metrics])
                example_values = [math.sqrt(value["mse"] / max(value["target_power"], 1e-12))
                    for (found_mode, found_level, found_component, _), observations in grouped.items()
                    if (found_mode, found_level, found_component) == (mode, level, component)
                    for value in observations]
                example_p95 = _percentile(example_values)
                components[component] = dict(identities=len(identity_metrics),
                    complete_two_conformer_identities=sum(value["two_independent_conformers"] for value in identity_metrics),
                    mse=mean_mse, target_power=mean_power, nrmse=nrmse, p95_per_identity_nrmse=p95,
                    p95_per_conformer_example_nrmse=example_p95,
                    tiny_target_identities=sum(value["target_power"] < 1e-12 for value in identity_metrics),
                    maximum_absolute_error=max((value["maximum_absolute_error"] for value in identity_metrics), default=None))
                if mode == "learned":
                    prefix = f"{level:g}:{component}"
                    checks[prefix + ":coverage"] = (len(identity_metrics) == expected_identities
                        and all(value["two_independent_conformers"] for value in identity_metrics))
                    checks[prefix + ":nrmse"] = nrmse is not None and nrmse <= max_nrmse
                    checks[prefix + ":p95"] = p95 is not None and p95 <= max_p95
                    checks[prefix + ":p95_conformer_example"] = example_p95 is not None and example_p95 <= max_p95
    return dict(by_mode=by_mode, checks=checks)


@torch.no_grad()
def run_denoiser_audit(vae, boundary, bridge, dataset_path, core, pocket, output, *,
                       identities=200, conformer_seeds=(17, 23), seed=17,
                       deadline=None, max_nrmse=0.05, max_p95=0.10):
    """Audit the sealed head once; failure remains failure with all denominators."""
    if (identities != 200 or tuple(conformer_seeds) != (17, 23)
            or max_nrmse != 0.05 or max_p95 != 0.10):
        raise ValueError("Changing the sealed 200-identity/two-conformer audit requires a new protocol")
    if boundary.atom_encoder is not core.dynamics.atom_encoder:
        raise ValueError("Audit boundary must be the actual loaded native core encoder")
    identity = vae_identity(vae)
    modules = dict(vae=vae, boundary=boundary, bridge=bridge, diffusion=core)
    before = _frozen(modules)
    data, manifest, manifest_sha = load_dataset(dataset_path, vae)
    selected = select_audit_rows(data, identities)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = next(core.parameters()).device
    required = {"x", "one_hot", "size", "mask"}
    if not required <= set(pocket):
        raise ValueError("Native crystal pocket tensor contract is incomplete")
    raw_pocket = {key: pocket[key].detach().clone().to(device) for key in required}
    if (raw_pocket["size"].numel() != 1 or int(raw_pocket["size"][0]) != len(raw_pocket["x"])
            or raw_pocket["mask"].dtype != torch.long or bool(raw_pocket["mask"].any())
            or raw_pocket["one_hot"].shape != (len(raw_pocket["x"]), core.atom_nf)
            or not bool(raw_pocket["x"].isfinite().all()) or len(raw_pocket["x"]) == 0):
        raise ValueError("Denoiser audit requires one finite native full-atom crystal pocket")
    center = raw_pocket["x"].mean(0)
    records, errors, attempted, successful = [], [], 0, 0
    result = dict(version=VERSION, passed=False, status="running_not_passed", vae_checkpoint=identity,
        paired_manifest_sha256=manifest_sha, source_data_identity=manifest.get("source_data_identity"),
        selected_rows=selected, selected_identities=[data["identity"][index] for index in selected],
        expected_identities=identities, expected_conformers=identities * 2, conformer_seeds=list(conformer_seeds),
        times=list(TIMES), modes=list(MODES), thresholds=dict(nrmse=max_nrmse, p95=max_p95),
        geometry_policy="two seeded ETKDGv3 conformers; no UFF; center at crystal pocket mean; validation only",
        failures_replaced=False, desired_score_labels=False,
        pocket_center=center.cpu().tolist(), pocket_tensor_sha256=state_digest(raw_pocket),
        rdkit_version=rdBase.rdkitVersion)
    write_json(output / "selection.json", {key: result[key] for key in
        ("version", "paired_manifest_sha256", "selected_rows", "selected_identities",
         "expected_identities", "expected_conformers", "conformer_seeds", "times")})
    try:
        for row in selected:
            check_deadline(deadline)
            molecule, slots, canonical = _canonical_molecule_and_map(data, row, vae.spec)
            decoded, hidden = vae.decode_with_features(data["z"][row:row + 1].to(device))
            hard = decoded.harden(vae.spec)
            stored = HardGraphBatch(*(data[key][row:row + 1].to(device=device, dtype=torch.long)
                                      for key in ("atoms", "charges", "bonds")))
            if any(not torch.equal(getattr(hard, key), getattr(stored, key))
                   for key in ("atoms", "charges", "bonds")):
                raise ValueError("Audit VAE no longer reproduces stored natural hard teacher")
            active = data["active"][row:row + 1].to(device)
            learned = bridge.native_probabilities(hidden.atom_hidden, boundary, active)[0, slots.to(device)]
            exact = {mode: exact_native_probabilities(decoded, vae.spec, active, boundary.native_elements,
                allowed_elements=PILOT_ELEMENTS if mode == "exact_six_conditioned" else None)
                for mode in MODES if mode != "learned"}
            predicted = {"learned": learned, **{mode: value.probabilities[0, slots.to(device)]
                                                  for mode, value in exact.items()}}
            number_to_native = {ATOMIC_NUMBER[symbol]: index for index, symbol in enumerate(boundary.native_elements)}
            native_types = torch.tensor([number_to_native[vae.spec.atomic_numbers[int(hard.atoms[0, slot])]]
                                         for slot in slots], device=device)
            teacher = F.one_hot(native_types, core.atom_nf).float()
            identity_record = dict(row=row, identity=data["identity"][row], group=data["group"][row],
                atom_count=len(slots), slot_order=slots.tolist(), slot_to_canonical=canonical.tolist(), conformers=[],
                exact_excluded_mass={mode: value.excluded_mass[active].mean().item() for mode, value in exact.items()})
            tensor_record = {"row": row, "identity": data["identity"][row], "conformers": []}
            for conformer_index, conformer_seed in enumerate(conformer_seeds):
                check_deadline(deadline)
                attempted += 1
                actual_seed = deterministic_seed(seed, data["identity"][row], "audit_conformer", conformer_seed)
                conformer_record = dict(index=conformer_index, declared_seed=conformer_seed,
                                        seed=actual_seed, rdkit_seed=actual_seed % (2**31 - 1), passed=False)
                identity_record["conformers"].append(conformer_record)
                try:
                    canonical_xyz = conformer_positions(molecule, actual_seed)
                except (ValueError, RuntimeError) as error:
                    conformer_record.update(error_type=type(error).__name__, error=str(error))
                    continue
                xyz = canonical_xyz[canonical].to(device)
                xyz = xyz - xyz.mean(0) + center
                ligand = dict(x=xyz, one_hot=teacher, size=torch.tensor([len(xyz)], device=device),
                              mask=torch.zeros(len(xyz), dtype=torch.long, device=device))
                normalized_ligand, normalized_pocket = core.normalize(
                    {key: value.clone() for key, value in ligand.items()},
                    {key: value.clone() for key, value in raw_pocket.items()})
                normalized_ligand["x"], normalized_pocket["x"] = core.remove_mean_batch(
                    normalized_ligand["x"], normalized_pocket["x"],
                    normalized_ligand["mask"], normalized_pocket["mask"])
                values = dict(seed=actual_seed, positions=xyz.cpu(), levels=[])
                for level in TIMES:
                    check_deadline(deadline)
                    noise_seed = deterministic_seed(seed, data["identity"][row], "denoiser_audit", conformer_seed, level)
                    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                    noise = torch.randn((len(xyz), 3 + core.atom_nf), generator=generator).to(device)
                    t = xyz.new_tensor([[level]])
                    noisy, shifted = _native_noise(core, normalized_ligand, normalized_pocket, t, noise)
                    reference, _ = core.dynamics(noisy, shifted, t, ligand["mask"], raw_pocket["mask"])
                    reference_mean = _native_transition_mean(core, noisy, shifted, t,
                        ligand["mask"], raw_pocket["mask"], level)
                    level_errors, outputs = {}, {}
                    for mode, probabilities in predicted.items():
                        check_deadline(deadline)
                        substitute = dict(normalized_ligand)
                        substitute["one_hot"] = boundary.normalize(probabilities)
                        mapped_noisy, mapped_pocket = _native_noise(core, substitute, normalized_pocket, t, noise)
                        if (not torch.equal(mapped_noisy[:, :3], noisy[:, :3])
                                or not torch.equal(mapped_pocket, shifted)):
                            raise RuntimeError("Chemical substitution changed shared coordinate noise/pocket")
                        output_eps, _ = core.dynamics(mapped_noisy, mapped_pocket, t,
                                                       ligand["mask"], raw_pocket["mask"])
                        output_mean = _native_transition_mean(core, mapped_noisy, mapped_pocket, t,
                            ligand["mask"], raw_pocket["mask"], level)
                        level_errors[mode] = dict(coordinates=_error(output_eps[:, :3], reference[:, :3]),
                            chemistry=_error(output_eps[:, 3:], reference[:, 3:]),
                            mean_coordinates=_error(output_mean[:, :3], reference_mean[:, :3]),
                            mean_chemistry=_error(output_mean[:, 3:], reference_mean[:, 3:]))
                        outputs[mode] = dict(predicted_epsilon=output_eps.cpu(), native_transition_mean=output_mean.cpu())
                    observation = dict(row=row, identity=data["identity"][row],
                        conformer=conformer_index, time=level, errors=level_errors, noise_seed=noise_seed)
                    errors.append(observation)
                    with (output / "observations.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(observation, sort_keys=True, allow_nan=False) + "\n")
                    values["levels"].append(dict(time=level, noise_seed=noise_seed, noise=noise.cpu(),
                        reference=reference.cpu(), reference_transition_mean=reference_mean.cpu(), predictions=outputs))
                successful += 1
                conformer_record.update(passed=True, positions=xyz.cpu().tolist())
                tensor_record["conformers"].append(values)
            tensors_path = output / (f"identity-{len(records):03d}.pt")
            torch.save(tensor_record, tensors_path)
            identity_record.update(tensors_path=tensors_path.name, tensors_sha256=sha256_file(tensors_path))
            records.append(identity_record)
            write_json(output / "identities.json", {"identities": records})
        summary = summarize_errors(errors, identities, max_nrmse=max_nrmse, max_p95=max_p95)
        identities25 = {data["identity"][row] for row in selected if int(data["active"][row].sum()) == 25}
        subset25 = summarize_errors([row for row in errors if row["identity"] in identities25],
            len(identities25), max_nrmse=max_nrmse, max_p95=max_p95)
        checks = dict(distinct_identity_coverage=len(selected) == identities,
                      conformer_attempt_coverage=attempted == 2 * identities,
                      conformer_success_coverage=successful == 2 * identities,
                      pilot_25_atom_subset_present=bool(identities25), **summary["checks"],
                      **{"pilot25:" + key: value for key, value in subset25["checks"].items()})
        result.update(summary=summary["by_mode"], checks=checks,
                      pilot_25_atom_subset=dict(identities=len(identities25), summary=subset25["by_mode"]),
                      transition_mean_policy="native zero-innovation transition; t0 native terminal coordinate mean/hard categorical output",
                      passed=all(checks.values()), status="passed" if all(checks.values()) else "failed_not_passed")
    except Exception as error:
        result.update(status="failed_not_passed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        after = _frozen(modules, strict=False)
        unchanged = before == after and getattr(vae, "diffsbdd_checkpoint_identity", None) == identity
        if not unchanged:
            result.update(passed=False, status="failed_frozen_state_changed")
        result.update(attempted_conformers=attempted, successful_conformers=successful,
            completed_identity_records=len(records), completed_observations=len(errors),
            selected_atom_count_histogram=dict(Counter(str(int(data["active"][row].sum())) for row in selected)),
            frozen_state_sha256_before=before, frozen_state_sha256_after=after,
            frozen_modules_unchanged=unchanged)
        result.setdefault("checks", {})["frozen_modules_unchanged"] = unchanged
        write_json(output / "audit-result.json", result)
    if not unchanged:
        raise RuntimeError("Frozen module changed during held-out denoiser audit")
    return result
