"""Local-Qwen-generated tests; two fixture/case corrections reviewed by Codex.

See docs/LOCAL_PDD.md for generation provenance and remaining PDD limitations.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notes import outside_repo


class TestArtifactBoundary(unittest.TestCase):
    """Regression tests for outside_repo(repo, path)."""

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_repo(self) -> Path:
        repo = self.temp_path / "repo"
        repo.mkdir()
        return repo

    def _make_parent(self) -> Path:
        return self.temp_path

    def test_1_reject_repo_itself(self) -> None:
        """Reject the repo itself with assertRaises(ValueError)."""
        repo = self._make_repo()
        path = repo
        with self.assertRaises(ValueError):
            outside_repo(repo, path)

    def test_2_reject_repo_descendant_nonexistent(self) -> None:
        """Reject repo/'documents'/'unbuilt.pdf' even though it does not exist."""
        repo = self._make_repo()
        path = repo / "documents" / "unbuilt.pdf"
        with self.assertRaises(ValueError):
            outside_repo(repo, path)

    def test_3_accept_parent_artifacts(self) -> None:
        """Accept parent/'artifacts'/'unbuilt.pdf', equal to its resolve();
        assert that neither artifacts directory nor output file exists afterwards."""
        parent = self._make_parent()
        repo = self._make_repo()
        artifacts = parent / "artifacts"
        path = artifacts / "unbuilt.pdf"

        result = outside_repo(repo, path)
        self.assertEqual(result, path.resolve())

        self.assertFalse(artifacts.exists())
        self.assertFalse(path.exists())

    def test_4_accept_parent_repo_other(self) -> None:
        """Accept parent/'repo-other'/'unbuilt.pdf' (prefix matching is not containment)."""
        parent = self._make_parent()
        repo = self._make_repo()
        repo_other = parent / "repo-other"
        repo_other.mkdir()
        path = repo_other / "unbuilt.pdf"

        result = outside_repo(repo, path)
        self.assertEqual(result, path.resolve())

    def test_5_reject_alias_symlink_to_repo(self) -> None:
        """A parent/'alias' symlink to repo: reject alias/'unbuilt.pdf'."""
        parent = self._make_parent()
        repo = self._make_repo()
        alias = parent / "alias"
        alias.symlink_to(repo)
        path = alias / "unbuilt.pdf"

        with self.assertRaises(ValueError):
            outside_repo(repo, path)

    def test_6_accept_symlink_outside_repo(self) -> None:
        """Create parent/'outside'; create repo/'output-link' symlink to outside;
        accept repo/'output-link'/'unbuilt.pdf' and return outside/'unbuilt.pdf' resolved.
        The policy concerns physical destination, not lexical spelling."""
        parent = self._make_parent()
        repo = self._make_repo()
        outside = parent / "outside"
        outside.mkdir()
        output_link = repo / "output-link"
        output_link.symlink_to(outside)
        path = output_link / "unbuilt.pdf"

        result = outside_repo(repo, path)
        expected = outside / "unbuilt.pdf"
        self.assertEqual(result, expected.resolve())


if __name__ == "__main__":
    unittest.main()
