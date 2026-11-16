import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build")
service = load("serve")


class ExportTests(unittest.TestCase):
    def test_legacy_hardening_auxiliary_is_not_reported_as_relaxed_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.fixture(Path(tmp))
            config = builder.read_json(run / 'stage2_config.json')
            config['hardening'] = True
            builder.write_json(run / 'stage2_config.json', config)
            path = run / 'soft_hard_gap.json'
            builder.write_json(path, [{'step': 0, 'soft_loss': 3, 'hard_loss': 2, 'soft_score': 99}])
            before = path.read_bytes()
            point = builder.export_run(run, 'Legacy hardening')['soft_hard_gap'][0]
            self.assertIsNone(point['soft_score'])
            self.assertEqual(point['optimization_forward_score'], 99)
            self.assertEqual((point['soft_loss'], point['hard_loss']), (3, 2))
            self.assertEqual(path.read_bytes(), before)

    def test_argmax_is_retained_before_display_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=self.fixture(Path(tmp))
            x=np.array([[.4999999,.5000001,0,0,0,0]]*2,dtype=np.float32)
            e=np.array([[.4999999,.5000001,0,0,0]],dtype=np.float32)
            for i in (0,1):
                np.savez(run/f'trajectory/evaluations/{i}.npz',X=x,E_upper=e,gradient_X=np.zeros_like(x),gradient_E_upper=np.zeros_like(e))
            frame=builder.export_run(run,'Close probabilities')['frames'][0]
            self.assertEqual(np.argmax(frame['X'][0]),0)
            self.assertEqual(np.argmax(frame['E'][0]),0)
            self.assertEqual(frame['argmax_atoms'],[1,1])
            self.assertEqual(frame['argmax_bonds'],[1])

    def test_covalent_stretch_depends_on_atom_identity_not_only_absolute_distance(self):
        distance=np.array([[0,2.4],[2.4,0]])
        radii=np.array([.76,.71,.66,1.05,.57,1.02])
        edges=np.eye(5)[[1]]
        carbon=builder.covalent_stretch(np.eye(6)[[0,0]],edges,distance,radii)
        sulfur=builder.covalent_stretch(np.eye(6)[[3,3]],edges,distance,radii)
        self.assertEqual(carbon['covalent_stretched_bonds'],1)
        self.assertAlmostEqual(carbon['maximum_covalent_stretch_ratio'],2.4/1.52)
        self.assertEqual(sulfur['covalent_stretched_bonds'],0)

    def test_native_results_require_related_job_and_exact_identity_for_step_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer=Path(tmp);run=outer/'proposal';run.mkdir()
            native=outer/'native_validation';native.mkdir()
            frames=[{'step':7,'readout':{'canonical_smiles':'CN'}}]
            builder.write_json(outer/'trial_spec.json',{'task':'native_proposal'})
            builder.write_json(outer/'run_summary.json',{'complete':True})
            results={'chosen':{'identity':{'canonical_smiles':'CN','step':7},
                               'affinity':{'affinity_probability_binary':.7,'affinity_probability_binary1':.6,'affinity_probability_binary2':.8,'affinity_pred_value':-.4}}}
            builder.write_json(native/'native_results.json',results)
            builder.write_json(native/'native_pose_audit.json',{'chosen':{'exact_affinity_scored_pose':True,'cavity_placement_pass':True}})
            payload=builder.native_evaluation(run,frames)
            self.assertTrue(payload['complete'])
            self.assertEqual(payload['candidates'][0]['proposal_frame'],0)
            self.assertTrue(payload['candidates'][0]['exact_scored_pose_checked'])
            self.assertEqual(payload['candidates'][0]['affinity']['affinity_pred_value'],-.4)
            results['chosen']['identity']['canonical_smiles']='CC'
            builder.write_json(native/'native_results.json',results)
            self.assertIsNone(builder.native_evaluation(run,frames)['candidates'][0]['proposal_frame'])
            results['chosen']['affinity']['affinity_probability_binary']=1.1
            builder.write_json(native/'native_results.json',results)
            with self.assertRaisesRegex(ValueError,'outside'):builder.native_evaluation(run,frames)
            builder.write_json(outer/'trial_spec.json',{'task':'optimization'})
            self.assertIsNone(builder.native_evaluation(run,frames))

    def fixture(self, root):
        run = root / "study/over-valent-start"
        trajectory = run / "trajectory"
        (trajectory / "evaluations").mkdir(parents=True)
        (trajectory / "readouts").mkdir()
        x = np.eye(6)[[4, 4]].astype(np.float32)  # fluorine: vmax = 1
        e = np.eye(5)[[3]].astype(np.float32)  # triple bond: valence = 3
        config = builder.Stage2Config(mode="valence")
        after, receipt = builder.project_valence(x, e, config=config)
        rows = []
        for step, edges in enumerate((e, after)):
            np.savez(trajectory / f"evaluations/{step}.npz", X=x, E_upper=edges,
                     gradient_X=np.zeros_like(x), gradient_E_upper=np.zeros_like(edges))
            row = {"evaluated_step": step, "final": step == 1, "loss": 5-step,
                   "evaluated_arrays": f"evaluations/{step}.npz", "aux_metrics": {}}
            if step == 0:
                row["post_step_operators"] = {"mode": "valence", "valence": receipt}
                row["readout"] = "readouts/0.json"
                builder.write_json(trajectory / row["readout"], {"sanitized": False})
            rows.append(row)
        (trajectory / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        builder.write_json(run / "stage2_config.json", {"stage2": {"mode": "valence"}, "eta_X": .1, "eta_E": .1})
        return run

    def test_reconstruction_and_update_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.fixture(Path(tmp))
            payload = builder.export_run(run, "Test")
            first, final = payload["frames"]
            np.testing.assert_allclose(first["projection"]["before"], [3, 3])
            np.testing.assert_allclose(first["projection"]["after"], [1, 1], atol=2e-6)
            self.assertTrue(first["projection"]["replay_matches_receipt"])
            self.assertEqual(first["metrics"]["projection_over_atoms"], 2)
            self.assertGreater(first["metrics"]["observed_step_E"], 0)
            self.assertNotIn("observed_step_E", final["metrics"])
            self.assertIsNone(final["projection"])
            self.assertEqual(first["metrics"]["sanitized"], 0)
            self.assertIsNone(final["metrics"]["sanitized"])
            self.assertIsNone(first["metrics"]["charge_entropy"])

    def test_source_description_and_rebuild_preserve_curation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root)
            builder.write_json(run / "visualization.json", {"description": "Test the valence cap."})
            output = root / "library"
            before = (run / "trajectory/trace.jsonl").read_bytes()
            builder.build([root / "study"], output)
            library = service.Library(output)
            rid = library.runs()["runs"][0]["id"]
            data = json.loads(gzip.decompress((output / f"runs/{rid}/data.json.gz").read_bytes()))
            self.assertEqual(data["description"], "Test the valence cap.")
            library.mutate("description", {"id": rid, "description": "A simpler description."})
            library.mutate("star", {"id": rid, "starred": True})
            builder.build([root / "study"], output)
            saved = service.Library(output).runs()["runs"][0]
            self.assertEqual(saved["description"], "A simpler description.")
            self.assertTrue(saved["starred"])
            self.assertEqual(before, (run / "trajectory/trace.jsonl").read_bytes())

    def test_traversal_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                builder.safe_artifact(Path(tmp) / "trajectory", "../outside.npz")

    def test_geometry_must_match_scored_context_and_stays_separate_from_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.fixture(Path(tmp))
            config = builder.read_json(run / "stage2_config.json")
            config["context_sha256"] = "a"*64
            builder.write_json(run / "stage2_config.json", config)
            geometry = {"context_sha256": "a"*64, "ligand_pair_distances_angstrom": [[0, 12], [12, 0]]}
            payload = builder.export_run(run, "Test", geometry)
            first = payload["frames"][0]["metrics"]
            self.assertEqual(first["frozen_long_bonds"], 1)
            self.assertEqual(first["maximum_frozen_bond_length_A"], 12)
            self.assertEqual(first["sanitized"], 0)
            self.assertEqual(payload["frozen_distances_angstrom"][0][1], 12)
            geometry['ligand_pair_distances_angstrom']=[[0,2.4],[2.4,0]]
            stretching=builder.export_run(run,'Broad gate alone misses stretching',geometry)['frames'][0]['metrics']
            self.assertEqual(stretching['frozen_bond_length_pass'],1)
            self.assertEqual(stretching['covalent_stretched_bonds'],1)
            self.assertEqual(stretching['sanitized'],0)
            geometry['ligand_pair_distances_angstrom']=[[0,2.5000001],[2.5000001,0]]
            boundary=builder.export_run(run,'Retain warning-boundary precision',geometry)
            self.assertEqual(boundary['frames'][0]['metrics']['frozen_long_bonds'],1)
            self.assertGreater(boundary['frozen_distances_angstrom'][0][1],2.5)
            geometry["context_sha256"] = "b"*64
            with self.assertRaises(ValueError):
                builder.export_run(run, "Wrong geometry", geometry)

    def test_rebuild_does_not_restore_deleted_or_purged_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            output = root / "library"
            builder.build([root / "study"], output)
            library = service.Library(output)
            rid = library.runs()["runs"][0]["id"]
            library.mutate("delete", {"ids": [rid]})
            builder.build([root / "study"], output)
            self.assertFalse((output / "runs" / rid).exists())
            self.assertTrue((output / ".trash" / rid).exists())
            library.mutate("purge", {"ids": [rid], "confirm_permanent": True})
            builder.build([root / "study"], output)
            self.assertFalse((output / "runs" / rid).exists())
            self.assertTrue(library.runs()["runs"][0]["purged"])


class CurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ids = ["a"*20, "b"*20, "c"*20]
        for rid in self.ids:
            p = self.root / "runs" / rid
            p.mkdir(parents=True)
            (p / "data.json.gz").write_bytes(b"fixture")
        builder.write_json(self.root / "index.json", {"runs": [{"id": rid, "name": rid} for rid in self.ids]})
        self.library = service.Library(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bulk_star_protection_rolls_back_and_restore_persists(self):
        a, b, c = self.ids
        self.library.mutate("star", {"id": b, "starred": True})
        with self.assertRaises(ValueError):
            self.library.mutate("delete", {"ids": [a, b]})
        self.assertTrue((self.root / "runs" / a).is_dir())
        self.library.mutate("delete", {"ids": [a, c]})
        self.assertTrue((self.root / ".trash" / a).is_dir())
        self.assertEqual(sum(r["deleted"] for r in service.Library(self.root).runs()["runs"]), 2)
        self.library.mutate("restore", {"ids": [a, c]})
        self.assertFalse(any(r["deleted"] for r in self.library.runs()["runs"]))
        self.assertTrue(self.library.runs()["runs"][1]["starred"])

    def test_unknown_ids_and_duplicates_cannot_mutate_files(self):
        for ids in (["../outside"], [self.ids[0], self.ids[0]], ["d"*20]):
            with self.assertRaises(ValueError):
                self.library.mutate("delete", {"ids": ids})
        self.assertTrue(all((self.root / "runs" / rid).is_dir() for rid in self.ids))

    def test_permanent_deletion_requires_trash_and_explicit_confirmation(self):
        rid = self.ids[0]
        with self.assertRaises(ValueError):
            self.library.mutate("purge", {"ids": [rid], "confirm_permanent": True})
        self.library.mutate("delete", {"ids": [rid]})
        with self.assertRaises(ValueError):
            self.library.mutate("purge", {"ids": [rid]})
        self.library.mutate("purge", {"ids": [rid], "confirm_permanent": True})
        self.assertFalse((self.root / ".trash" / rid).exists())
        self.assertTrue(self.library.runs()["runs"][0]["purged"])


if __name__ == "__main__":
    unittest.main()


class MoleculeShortcutTests(unittest.TestCase):
    def test_republish_preserves_molecule_stars_notes_and_trash(self):
        publisher = load('publish_evaluation')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'library';library.mkdir()
            (library/'index.json').write_text(json.dumps({'schema_version':1,'runs':[]}))
            report=root/'comparison.json'
            report.write_text(json.dumps({'schema_version':1,'complete':True,'guided_complete':True,'records':[{'id':'C25','gnina_summary':{'minimizedAffinity':{'mean':-5},'CNNaffinity':{'mean':5}}}]}))
            publisher.publish(report,library)
            manager=service.Library(library);entry=manager.runs()['runs'][0]
            self.assertEqual(entry['kind'],'molecule')
            self.assertFalse((library/'runs'/entry['id']/'data.json.gz').exists())
            manager.mutate('star',{'id':entry['id'],'starred':True})
            manager.mutate('description',{'id':entry['id'],'description':'Keep my molecule note'})
            publisher.publish(report,library)
            saved=manager.runs()['runs'][0]
            self.assertTrue(saved['starred']);self.assertEqual(saved['description'],'Keep my molecule note')
            manager.mutate('star',{'id':entry['id'],'starred':False})
            manager.mutate('delete',{'ids':[entry['id']]})
            publisher.publish(report,library)
            self.assertFalse((library/'runs'/entry['id']).exists())
            self.assertTrue((library/'.trash'/entry['id']).exists())
            manager.mutate('restore',{'ids':[entry['id']]})
            self.assertTrue((library/'runs'/entry['id']/'meta.json').exists())
