"""Portable presentation metadata controls; no scientific fixtures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('collection', Path(__file__).resolve().parents[1] / 'scripts/collection.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.viewer = self.root / 'viewer'
        self.edition = self.viewer / 'editions/edition-0001'
        self.edition.mkdir(parents=True)
        manifest = {'arms': [{'id': 'a', 'global_start': 0, 'latest_global_step': 20, 'scientific_status': 'failed'}]}
        self.write(self.edition / 'manifest.json', manifest)
        self.write(self.edition / 'assessment.json', {'status': 'passed'})
        self.write(self.edition / 'cpu-certificate.json', {'status': 'passed'})
        (self.edition / 'report.html').write_text('<html>saved report</html>')
        self.write(self.viewer / 'latest.json', {'edition': 'edition-0001', 'manifest_sha256': module.sha(self.edition / 'manifest.json')})
        self.write(self.edition / 'report-receipt.json', {'job_id': '1', 'files': {p.name: {'sha256': module.sha(p), 'bytes': p.stat().st_size} for p in self.edition.iterdir()}})
        self.config = {'viewer': str(self.viewer), 'collection': 'test', 'base_url': 'http://localhost:8780', 'title': '</script><test>', 'subtitle': '', 'date': '', 'highlights': [], 'comparison_note': '', 'runs': [{'id': 'a', 'target': 20, 'label': 'A'}]}
        self.path = self.root / 'config.json'

    def write(self, path, data):
        path.write_text(json.dumps(data), encoding='utf8')

    def assemble(self):
        self.write(self.path, self.config)
        return module.assemble(self.path)

    def test_preserves_failure_and_escapes_script_data(self):
        result = self.assemble()
        output = Path(result['output'])
        self.assertEqual(module.load(output / 'selection.json')['runs'][0]['scientific_status'], 'failed')
        self.assertIn('\\u003c/script>', (output / 'report.html').read_text(encoding='utf8'))
        with self.assertRaises(FileExistsError):
            self.assemble()

    def test_missing_or_incomplete_selection_writes_nothing(self):
        self.config['runs'][0]['target'] = 21
        with self.assertRaises(ValueError):
            self.assemble()
        self.assertFalse((self.viewer / 'collections').exists())

    def test_changed_canonical_input_rejected(self):
        (self.edition / 'report.html').write_text('changed')
        with self.assertRaises(ValueError):
            self.assemble()

    def test_external_report_is_exact_copy_and_bound(self):
        artifact = self.root / 'foreign.html'
        artifact.write_bytes(b'<html>foreign saved vocabulary</html>')
        proof = self.root / 'foreign.json'
        self.write(proof, {'outputs': {'index.html': module.sha(artifact)}})
        evidence = self.root / 'evidence.json'
        self.write(evidence, {'actual_job': '2'})
        self.config['runs'].append({'id': 'vae', 'target': 50, 'attachment': {'path': str(artifact), 'sha256': module.sha(artifact), 'manifest': str(proof), 'manifest_sha256': module.sha(proof), 'evidence': str(evidence), 'evidence_sha256': module.sha(evidence)}})
        result = self.assemble()
        self.assertEqual((Path(result['output']) / 'attachments/vae.html').read_bytes(), artifact.read_bytes())
        self.config['collection'] = 'second'
        artifact.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.assemble()


if __name__ == '__main__':
    unittest.main()
