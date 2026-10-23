"""Download official UniProt records and freeze a small sequence-only panel.

This is file preparation, not model evaluation. A one-query A3M explicitly
declares the absence of an evolutionary MSA; no server is contacted by jobs.
The canonical, full-length human sequences are used without domain cropping.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

PANEL = [('CA2', 'P00918'), ('DHFR', 'P00374'), ('FKBP1A', 'P62942'),
         ('MAPK1', 'P28482'), ('PIM1', 'P11309'), ('AKR1B1', 'P15121'),
         ('PTPN1', 'P18031'), ('GSTP1', 'P09211'), ('NQO1', 'P15559'),
         ('PPIA', 'P62937')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = []
    digest = lambda b: hashlib.sha256(b).hexdigest()
    for gene, accession in PANEL:
        url = f'https://rest.uniprot.org/uniprotkb/{accession}.json'
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        record = json.loads(payload)
        sequence = record['sequence']['value']
        if (record['primaryAccession'] != accession or record['organism']['taxonId'] != 9606
                or gene not in [g.get('geneName', {}).get('value') for g in record['genes']]
                or not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY')):
            raise ValueError(f'Unexpected canonical sequence identity: {accession}')
        raw = args.output / f'{accession}.json'
        raw.write_bytes(payload)
        msa = args.output / f'{accession}.a3m'
        msa.write_bytes(f'>{accession}\n{sequence}\n'.encode())
        targets.append({'id': gene.lower(), 'gene': gene, 'accession': accession,
                        'name': record['proteinDescription']['recommendedName']['fullName']['value'],
                        'sequence': sequence, 'sequence_sha256': digest(sequence.encode()),
                        'residues': len(sequence), 'sequence_scope': 'full_canonical_uniprot',
                        'raw_record': {'path': raw.name, 'sha256': digest(payload)},
                        'msa': {'path': msa.name, 'sha256': digest(msa.read_bytes())},
                        'source_url': url, 'entry_audit': record['entryAudit']})
    (args.output / 'targets.json').write_text(json.dumps({
        'version': 1, 'downloaded_at': datetime.now(timezone.utc).isoformat(),
        'msa_policy': 'single_query_no_evolutionary_msa',
        'license': 'UniProt CC BY 4.0; https://www.uniprot.org/help/license',
        'context_policy': 'one full-length chain; no cofactors/metals/templates/pocket constraints',
        'targets': targets}, indent=2) + '\n')
    print(json.dumps([{'gene': t['gene'], 'residues': t['residues']} for t in targets]))


if __name__ == '__main__':
    main()
