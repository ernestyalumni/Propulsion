"""Registered notes and reproducible evidence. No arbitrary path or execution API."""
import hashlib
import json
from pathlib import Path
import re


def outside_repo(repo, path):
    path = path.resolve()
    if path == repo.resolve() or repo.resolve() in path.parents:
        raise ValueError('Builds and evidence must be outside the source repository')
    return path


def input_hashes(repo):
    paths = [repo / p for p in ('ReadingRoom/resources.json', 'ReadingRoom/notes.py',
             'ReadingRoom/build_notes.py', 'Studies/mechanics.py', 'Studies/symbolic.py', 'Studies/verify.py')]
    paths += sorted((repo / 'documents/notes').rglob('*.tex'))
    paths += sorted((repo / 'documents/notes').rglob('*.bib'))
    return {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class NotesCatalog:
    def __init__(self, repo, artifacts, references=None):
        self.repo = repo.resolve()
        self.artifacts = outside_repo(self.repo, artifacts)
        self.roots = {'propulsion': self.repo, 'artifacts': self.artifacts}
        for name in ('mathphysics','Monoclaw','CompPhys'):
            candidates = [self.repo.parent / name, self.repo.parent.parent / 'workspace2/repos' / name]
            self.roots[name] = next((path for path in candidates if path.is_dir()), candidates[0])
        for name,path in (references or {}).items():
            if name not in ('mathphysics','Monoclaw','CompPhys'):
                raise ValueError('Unknown reference repository: '+name)
            self.roots[name] = Path(path).resolve()
        self.manifest = json.loads((self.repo / 'ReadingRoom/resources.json').read_text())
        self.assets = {}
        for asset in self.manifest['assets']:
            if not re.fullmatch(r'[a-z0-9-]+',asset['id']) or asset['id'] in self.assets:
                raise ValueError('Invalid or duplicate resource ID')
            path = Path(asset['path'])
            if path.is_absolute() or '..' in path.parts or asset['root'] not in self.roots:
                raise ValueError('Invalid registered path')
            self.assets[asset['id']] = asset
        ids = set()
        for topic in self.manifest['topics']:
            if not re.fullmatch(r'[a-z0-9-]+', topic['id']) or topic['id'] in ids:
                raise ValueError('Invalid or duplicate topic ID')
            ids.add(topic['id'])
            if any(key not in self.assets for key in topic['assets']):
                raise ValueError('Topic references unknown resource')

    def resolve(self, ident):
        asset = self.assets[ident]
        root = self.roots[asset['root']].resolve()
        path = (root / asset['path']).resolve()
        path.relative_to(root)  # reject symlink escape, even for registered files
        if not path.is_file():
            raise FileNotFoundError(ident)
        return path

    def evidence(self):
        try:
            report = json.loads((self.artifacts / 'verification.json').read_text())
            if not isinstance(report,dict) or report.get('schema') != 1 or not isinstance(report.get('topics'), dict):
                raise ValueError('Invalid report')
            for record in report['topics'].values():
                if not isinstance(record,dict) or any(record.get(kind) is not None and not isinstance(record[kind],dict) for kind in ('symbolic','numerical')):
                    raise ValueError('Invalid topic evidence')
            report['current'] = report.get('inputs') == input_hashes(self.repo)
            return report
        except (OSError, ValueError):
            return {'current':False, 'topics':{}, 'created':None}

    def public(self, books):
        evidence = self.evidence()
        try:
            current_inputs = input_hashes(self.repo)
        except OSError:
            current_inputs = None
        try:
            build = json.loads((self.artifacts / 'build.json').read_text())
            if not isinstance(build,dict) or not isinstance(build.get('outputs',{}),dict):
                build = {}
        except (OSError, ValueError):
            build = {}
        assets = {}
        for ident, asset in self.assets.items():
            row = {k:v for k,v in asset.items() if k != 'root'}
            try:
                path = self.resolve(ident)
                row['available'] = True
            except (OSError,ValueError):
                row['available'] = False
            row['url'] = '/note-asset/'+ident
            if asset.get('anchor'):
                row['url'] += '#nameddest='+asset['anchor']
            if asset['root'] == 'artifacts' and asset['path'].endswith('.pdf'):
                expected = build.get('outputs',{}).get(asset['path'])
                row['build_current'] = bool(row['available'] and current_inputs is not None and build.get('inputs') == current_inputs and
                    expected == hashlib.sha256(path.read_bytes()).hexdigest())
            assets[ident] = row
        topics = []
        for original in self.manifest['topics']:
            row = dict(original)
            row['assets'] = [assets[key] for key in row['assets']]
            row['locators'] = []
            for locator in original['locators']:
                book = books.books.get(locator['book'])
                section = books.section(locator['book'],locator['section']) if book else None
                row['locators'].append(dict(locator, available=section is not None,
                    printed_page=section['printed_page'] if section else None,
                    pdf_page=section['pdf_page'] if section else None,
                    exact=section['exact'] if section else False))
            record = evidence['topics'].get(row['id'],{})
            row['verification'] = {
                'current':bool(record) and evidence['current'], 'created':evidence.get('created') if record else None,
                'numerical':bool(evidence['current'] and (record.get('numerical') or {}).get('passed') is True),
                'symbolic':bool(evidence['current'] and (record.get('symbolic') or {}).get('passed') is True),
                'recorded':bool(record), 'sage_version':evidence.get('sage_version') if record else None}
            topics.append(row)
        return {'topics':topics, 'documents':[assets[k] for k in self.manifest['documents']],
                'evidence_url':'/note-asset/verification' if (self.artifacts/'verification.json').is_file() else None}
