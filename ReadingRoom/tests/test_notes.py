"""Traceability and evidence protections using the real registry and temporary assets."""
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import Catalog, REPO, WORKSPACE, ReadingServer
from notes import NotesCatalog, input_hashes, outside_repo


class NotesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root/'artifacts'
        self.artifacts.mkdir()
        self.catalog = Catalog(WORKSPACE/'Data/Exports/ForPropulsion')
        self.notes = NotesCatalog(REPO,self.artifacts,{'mathphysics':self.root/'absent'})

    def tearDown(self):
        self.temp.cleanup()

    def test_topics_have_resolvable_source_and_book_locators(self):
        public = self.notes.public(self.catalog)
        self.assertEqual(len(public['topics']),5)
        for topic in public['topics']:
            self.assertTrue(all(l['available'] for l in topic['locators']))
            source = self.notes.resolve(topic['id']+'-source')
            self.assertIn('\\topic{'+topic['id']+'}',source.read_text())
            self.assertFalse(topic['verification']['numerical'])
        nozzle = next(t for t in public['topics'] if t['id']=='nozzle')
        self.assertFalse(next(l for l in nozzle['locators'] if l['book']=='sutton')['exact'])

    def test_missing_optional_repo_and_pdf_are_visible(self):
        topic = next(t for t in self.notes.public(self.catalog)['topics'] if t['id']=='rigid-body')
        assets = {a['id']:a for a in topic['assets']}
        self.assertFalse(assets['dgdt']['available'])
        self.assertFalse(assets['rigid-body-pdf']['available'])
        self.assertTrue(assets['rotations']['available'])

    def test_evidence_loses_verified_status_after_source_change(self):
        clone = self.root/'repo'
        for folder in ('ReadingRoom','Studies','documents/notes'):
            shutil.copytree(REPO/folder,clone/folder,ignore=shutil.ignore_patterns('node_modules','.pdd','__pycache__','docs'))
        notes = NotesCatalog(clone,self.artifacts)
        report = {'schema':1,'inputs':input_hashes(clone),'topics':{
            'oscillator':{'numerical':{'passed':True},'symbolic':{'passed':True}}}}
        (self.artifacts/'verification.json').write_text(json.dumps(report))
        row = notes.public(self.catalog)['topics'][0]
        self.assertTrue(row['verification']['symbolic'])
        (clone/'Studies/mechanics.py').write_text('# changed implementation')
        row = notes.public(self.catalog)['topics'][0]
        self.assertTrue(row['verification']['recorded'])
        self.assertFalse(row['verification']['current'])
        self.assertFalse(row['verification']['symbolic'])
        self.assertFalse(row['verification']['numerical'])

    def test_pdf_hash_and_provenance_detect_replacement(self):
        pdf = self.artifacts/'master-portrait.pdf'
        pdf.write_bytes(b'%PDF-initial')
        (self.artifacts/'build.json').write_text(json.dumps({'inputs':input_hashes(REPO),
            'outputs':{'master-portrait.pdf':hashlib.sha256(pdf.read_bytes()).hexdigest()}}))
        self.assertTrue(self.notes.public(self.catalog)['documents'][0]['build_current'])
        pdf.write_bytes(b'%PDF-changed')
        self.assertFalse(self.notes.public(self.catalog)['documents'][0]['build_current'])

    def test_registered_symlink_cannot_escape_and_outputs_cannot_enter_repo(self):
        (self.artifacts/'master-portrait.pdf').symlink_to(REPO/'README.md')
        with self.assertRaises(ValueError):
            self.notes.resolve('master-portrait')
        with self.assertRaises(ValueError):
            outside_repo(REPO,REPO/'documents/output.pdf')
        public = self.notes.public(self.catalog)
        self.assertFalse(public['documents'][0]['available'])

    def test_corrupt_evidence_never_claims_success(self):
        for invalid in ('{bad','[]','{"schema":1,"topics":{"oscillator":[]}}',
                        '{"schema":1,"topics":{"oscillator":{"numerical":true}}}'):
            (self.artifacts/'verification.json').write_text(invalid)
            self.assertFalse(self.notes.evidence()['current'])
        (self.artifacts/'build.json').write_text('[]')
        self.assertEqual(len(self.notes.public(self.catalog)['topics']),5)

    def test_http_serves_only_registered_assets_and_never_executes(self):
        server = ReadingServer(0,WORKSPACE/'Data/Exports/ForPropulsion',self.root/'state',self.artifacts)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            def get(path,method='GET'):
                connection = http.client.HTTPConnection('127.0.0.1',server.server_address[1])
                connection.request(method,path)
                response = connection.getresponse()
                result = response.status,response.getheader('Content-Type'),response.read()
                connection.close()
                return result
            before = server.store.read()
            status,content,body = get('/note-asset/symbolic')
            self.assertEqual(status,200)
            self.assertTrue(content.startswith('text/plain'))
            self.assertIn(b'from sage.all',body)
            self.assertEqual(get('/api/notes')[0],200)
            for path in ('/note-asset/../server.py','/note-asset/%2e%2e%2fMEMORY.md','/note-asset/not-registered','/note-asset/symbolic/extra'):
                self.assertIn(get(path)[0],(400,404))
            self.assertEqual(get('/note-asset/symbolic','POST')[0],403)
            self.assertEqual(server.store.read(),before)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
