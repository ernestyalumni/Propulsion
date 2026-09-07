"""Behavioral tests use temporary progress; never alter source bundles."""
import copy
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Catalog, REPO, WORKSPACE, ReadingServer, StateStore, contained


class ReadingRoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="propulsion-tests-")
        cls.exports = WORKSPACE / "Data/Exports/ForPropulsion"
        cls.server = ReadingServer(0, cls.exports, Path(cls.temp.name))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, path, payload=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        opts = {"Content-Type": "application/json", "X-Reading-Token": self.server.token}
        opts.update(headers or {})
        conn.request("POST" if payload is not None else "GET", path,
                     json.dumps(payload) if payload is not None else None, opts)
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return result

    def payload(self, **kwargs):
        result = {"revision": self.server.store.read()["revision"], "book": "wie", "patch": {
            "bookmark": {"section": "5.4", "page": 353, "zoom": 1.25, "scroll": 0.42},
            "section": {"id": "5.4", "notes": "Rotation ≠ representation", "questions": "Which frame?",
                        "next": "Compare active/passive rotations", "checks": {
                            "read": True, "discussed": False, "derived": False, "implemented": False}}}}
        result.update(kwargs)
        return result

    def test_catalog_resolves_all_books_and_mapping_certainty(self):
        status, _, body = self.request("/api/bootstrap")
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual({b["id"] for b in result["books"]}, {"nr", "wie", "sutton", "hp"})
        self.assertEqual(result["warnings"], [])
        wie = self.server.catalog.section("wie", "5.4")
        sutton = self.server.catalog.section("sutton", "3.3")
        self.assertEqual((wie["pdf_page"], wie["printed_page"], wie["exact"]), (352, 334, True))
        self.assertEqual((sutton["pdf_page"], sutton["exact"]), (75, False))
        for book in result["books"]:
            self.assertNotIn("/media/", book["pdf_path"])
            self.assertTrue(Path(book["pdf_path"]).is_file())
            self.assertIn("historical_status", book["chapters"][0])

    def test_missing_bundles_reported_without_fake_books(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Catalog(Path(directory))
            self.assertEqual(catalog.books, {})
            self.assertEqual(len(catalog.warnings), 4)

    def test_disk_resume_and_handoff_preserve_section_notes(self):
        status, _, body = self.request("/api/progress", self.payload())
        self.assertEqual(status, 200, body)
        restarted = StateStore(Path(self.temp.name), self.server.catalog)
        entry = restarted.read()["books"]["wie"]
        self.assertEqual(entry["bookmark"]["page"], 353)
        self.assertEqual(entry["bookmark"]["scroll"], 0.42)
        self.assertEqual(entry["sections"]["5.4"]["notes"], "Rotation ≠ representation")
        status, _, body = self.request("/api/handoff")
        self.assertEqual(status, 200)
        self.assertIn("Which frame?", body.decode())
        self.assertIn("PDF page: 353", body.decode())
        self.assertTrue((Path(self.temp.name) / "HANDOFF.md").is_file())

    def test_navigation_never_asserts_completion(self):
        payload = self.payload(book="nr", patch={"bookmark": {"page": 931, "section": "17.1", "zoom": 1, "scroll": 0}})
        status, _, _ = self.request("/api/progress", payload)
        self.assertEqual(status, 200)
        self.assertEqual(self.server.store.read()["books"]["nr"]["sections"], {})

    def test_stale_tab_cannot_overwrite_new_notes(self):
        first = self.payload()
        second = copy.deepcopy(first)
        second["patch"]["section"]["notes"] = "stale replacement"
        self.assertEqual(self.request("/api/progress", first)[0], 200)
        self.assertEqual(self.request("/api/progress", second)[0], 409)
        self.assertNotEqual(self.server.store.read()["books"]["wie"]["sections"]["5.4"]["notes"], "stale replacement")

    def test_bad_records_never_change_state(self):
        for field, value in [("page", 0), ("page", 971), ("page", True), ("zoom", 20), ("scroll", -1), ("section", "fake")]:
            with self.subTest(field=field, value=value):
                payload = self.payload()
                payload["patch"]["bookmark"][field] = value
                before = self.server.store.read()
                self.assertEqual(self.request("/api/progress", payload)[0], 400)
                self.assertEqual(before, self.server.store.read())
        payload = self.payload()
        payload["patch"]["section"]["checks"]["implemented"] = "yes"
        self.assertEqual(self.request("/api/progress", payload)[0], 400)

    def test_cross_origin_and_missing_token_requests_rejected(self):
        before = self.server.store.read()
        self.assertEqual(self.request("/api/progress", self.payload(), {"Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.request("/api/progress", self.payload(), {"X-Reading-Token": "wrong"})[0], 403)
        self.assertEqual(self.request("/api/bootstrap", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("/api/bootstrap", headers={"Origin": "null"})[0], 403)
        self.assertEqual(before, self.server.store.read())

    def test_arbitrary_paths_not_served(self):
        for path in ["/../../MEMORY.md", "/book/wie/../../MEMORY.md", "/lab/../MEMORY.md",
                     "/lab/Cosmos/../../.git/config", "/vendor/pdfjs-dist/build/%2e%2e/%2e%2e/package.json",
                     "/server.py", "/.git/config", "/book/nr/NR_C301"]:
            with self.subTest(path=path):
                self.assertIn(self.request(path)[0], (400, 404))

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "escape").symlink_to(REPO.parent)
            with self.assertRaises(ValueError):
                contained(root, root / "escape/Propulsion/README.md")

    def test_pdf_ranges_and_documents_are_available(self):
        for ident in ("nr", "wie", "sutton", "hp"):
            status, headers, body = self.request("/book/" + ident + "/pdf", headers={"Range": "bytes=0-4"})
            self.assertEqual(status, 206)
            self.assertEqual(body, b"%PDF-")
            self.assertTrue(headers["Content-Range"].startswith("bytes 0-4/"))
            self.assertEqual(self.request("/book/" + ident + "/roadmap")[0], 200)
        self.assertEqual(self.request("/api/text/nr?page=931")[0], 200)
        self.assertEqual(self.request("/book/wie/chapter-005")[0], 200)
        self.assertEqual(self.request("/book/sutton/chapter-003")[0], 200)
        self.assertEqual(self.request("/book/nr/pdf", headers={"Range": "bytes=99999999999-"})[0], 416)

    def test_saves_do_not_modify_exported_snapshots_or_pdfs(self):
        paths = list(self.exports.glob("*/context/reading-program/*/progress.json"))
        paths += [files["pdf"] for files in self.server.catalog.files.values()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.assertEqual(self.request("/api/progress", self.payload())[0], 200)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def test_corrupt_state_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                StateStore(Path(directory), self.server.catalog)
            self.assertEqual(path.read_text(), "{broken")

    def test_write_failure_reports_error_and_preserves_previous_state(self):
        before = self.server.store.read()
        with patch.object(self.server.store, "atomic_write", side_effect=PermissionError("test failure")):
            status, _, body = self.request("/api/progress", self.payload())
        self.assertEqual(status, 500)
        self.assertIn("Save failed", body.decode())
        self.assertEqual(before, self.server.store.read())

    def test_handoff_failure_keeps_canonical_save_and_reports_warning(self):
        original = self.server.store.atomic_write
        def fail_handoff(path, text):
            if path.name == "HANDOFF.md":
                raise PermissionError("test handoff failure")
            original(path, text)
        payload = self.payload()
        with patch.object(self.server.store, "atomic_write", side_effect=fail_handoff):
            status, _, body = self.request("/api/progress", payload)
        self.assertEqual(status, 200)
        self.assertIn("HANDOFF.md", json.loads(body)["warning"])
        self.assertEqual(self.server.store.read()["revision"], payload["revision"] + 1)


if __name__ == "__main__":
    unittest.main()
