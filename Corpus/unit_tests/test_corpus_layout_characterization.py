"""
@file test_corpus_layout_characterization.py

@details Characterization tests for the corpus as it exists today.

These lock down the on-disk facts that the parse-once record (story 1) and the
configured corpus root (story 2) must respect. They assert current behavior, not
desired behavior: a failure here means the corpus changed underneath the
stories, and the stories need re-reading before any code is generated from them.

Read-only. Nothing here writes to the corpus or invokes an OCR model.
"""

from pathlib import Path

import pytest

from conftest import read_json


# The nine sources that carry OCR output today.
PARSED_BOOK_SLUGS = (
    "EngineeringPhysics/Chirikjian-StochasticModelsInformationTheoryLieGroups-v1",
    "EngineeringPhysics/Chirikjian-StochasticModelsInformationTheoryLieGroups-v2",
    "EngineeringPhysics/HorowitzHill-ArtOfElectronics3e",
    "EngineeringPhysics/Lieuwen-UnsteadyCombustorPhysics",
    "EngineeringPhysics/Natanzon-CombustionInstability",
    "EngineeringPhysics/Sidi-SpacecraftDynamicsControl",
    "Physics/Arnold-MathematicalMethodsClassicalMechanics-2e",
    "Physics/Goldstein-ClassicalMechanics-3e",
    "Physics/Srednicki-QuantumFieldTheory",
)

# The subset whose conflict-resolution stage has run to completion.
RESOLVED_BOOK_SLUGS = (
    "EngineeringPhysics/Lieuwen-UnsteadyCombustorPhysics",
    "EngineeringPhysics/Natanzon-CombustionInstability",
    "EngineeringPhysics/Sidi-SpacecraftDynamicsControl",
    "Physics/Goldstein-ClassicalMechanics-3e",
    "Physics/Srednicki-QuantumFieldTheory",
)

LIEUWEN_SLUG = "EngineeringPhysics/Lieuwen-UnsteadyCombustorPhysics"
ARNOLD_SLUG = "Physics/Arnold-MathematicalMethodsClassicalMechanics-2e"


@pytest.mark.parametrize("book_slug", PARSED_BOOK_SLUGS)
def test_every_parsed_book_holds_an_ocr_compare_directory(books_root, book_slug):
    assert (books_root / book_slug / "ocr-compare").is_dir()


@pytest.mark.parametrize("book_slug", PARSED_BOOK_SLUGS)
def test_every_parsed_book_holds_nougat_output_reconciled_and_marker_markdown(
        books_root,
        book_slug):
    """The three artifacts a text-extraction stage leaves behind."""
    ocr_compare_path = books_root / book_slug / "ocr-compare"

    assert (ocr_compare_path / "nougat_out").is_dir()
    assert (ocr_compare_path / "reconciled").is_dir()
    assert list(ocr_compare_path.glob("*.marker.md"))


@pytest.mark.parametrize("book_slug", PARSED_BOOK_SLUGS)
def test_every_parsed_book_holds_the_equations_contract(books_root, book_slug):
    """equations.json is the machine-readable contract of the reconcile stage."""
    equations = read_json(
        books_root / book_slug / "ocr-compare" / "reconciled" / "equations.json")

    assert set(equations) >= {"summary", "equations"}
    assert set(equations["summary"]) >= {
        "agree",
        "conflict",
        "marker_only",
        "nougat_only"}


@pytest.mark.parametrize("book_slug", PARSED_BOOK_SLUGS)
def test_conflict_resolution_has_run_for_five_of_the_nine_books(
        books_root,
        book_slug):
    """Parsing has stages, and the corpus is mid-way through them.

    HorowitzHill, Arnold and both Chirikjian volumes stop after reconciliation,
    so a completion record cannot be a single boolean per source.
    """
    resolved_path = (
        books_root / book_slug / "ocr-compare" / "reconciled"
        / "equations_resolved.json")

    assert resolved_path.is_file() == (book_slug in RESOLVED_BOOK_SLUGS)


def test_a_failed_parse_looks_identical_to_a_good_one_from_the_directory_tree(
        books_root):
    """The reason completeness cannot be inferred from directory presence.

    Arnold carries the same directories and the same marker markdown as
    Lieuwen, but its reconcile found nothing: zero agreements, zero conflicts,
    one marker-only equation, and a nougat page emitted twenty-four times. That
    parse failed and must stay re-runnable.
    """
    lieuwen_path = books_root / LIEUWEN_SLUG / "ocr-compare"
    arnold_path = books_root / ARNOLD_SLUG / "ocr-compare"

    for artifact_path in (lieuwen_path, arnold_path):
        assert (artifact_path / "nougat_out").is_dir()
        assert (artifact_path / "reconciled").is_dir()
        assert list(artifact_path.glob("*.marker.md"))

    lieuwen_summary = read_json(
        lieuwen_path / "reconciled" / "equations.json")["summary"]
    arnold_summary = read_json(
        arnold_path / "reconciled" / "equations.json")["summary"]

    assert lieuwen_summary["agree"] == 373
    assert lieuwen_summary["conflict"] == 306

    assert arnold_summary["agree"] == 0
    assert arnold_summary["conflict"] == 0
    assert arnold_summary["marker_only"] == 1
    assert max(arnold_summary["nougat_repeated"].values()) == 24


def test_no_source_carries_a_parse_completion_record_today(books_root):
    """The gap story 1 closes.

    Nothing on disk records that a parse finished, so nothing prevents a
    re-parse. Delete this test when story 1 lands — it is the baseline, not a
    requirement.
    """
    for book_slug in PARSED_BOOK_SLUGS:
        ocr_compare_path = books_root / book_slug / "ocr-compare"

        assert not (ocr_compare_path / "parse_record.json").exists()
        assert not (ocr_compare_path / ".parse_complete").exists()


def test_the_books_index_does_not_cover_every_parsed_source(books_root):
    """BOOKS.tsv is a partial slug-to-pdf map, not an index of the corpus."""
    engineering_physics_path = books_root / "EngineeringPhysics"
    indexed_slugs = {
        line.split("\t")[0]
        for line in (engineering_physics_path / "BOOKS.tsv").read_text(
            encoding="utf-8").splitlines()
        if line.strip()}
    slug_directories = {
        path.name for path in engineering_physics_path.iterdir() if path.is_dir()}

    assert indexed_slugs < slug_directories


def test_the_repository_holds_no_parsed_data_products(repository_root):
    """Story 2's boundary, asserted against the working tree as it stands."""
    ignored_directory_names = {".git", ".venv", ".pdd"}
    offending_paths = []

    for path in repository_root.rglob("*"):
        if ignored_directory_names & set(path.relative_to(repository_root).parts):
            continue
        if path.name in ("nougat_out", "ocr-compare", "equations.json"):
            offending_paths.append(path)
        elif path.name.endswith(".marker.md"):
            offending_paths.append(path)

    assert offending_paths == []
