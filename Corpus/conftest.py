"""
@file conftest.py

@details Fixtures for the corpus characterization suite.

Following the Stunticons convention, pytest adds this directory's parent to
sys.path, so no PYTHONPATH mangling is needed.

The corpus lives outside this repository, under the root named by
PROPULSION_CORPUS_ROOT. Nothing here hardcodes an absolute path: when the
variable is unset, or names a directory that is not mounted, every test in this
suite skips.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def corpus_root():
    configured_root = os.environ.get("PROPULSION_CORPUS_ROOT")
    if not configured_root:
        pytest.skip("PROPULSION_CORPUS_ROOT is not set")

    root_path = Path(configured_root)
    if not root_path.is_dir():
        pytest.skip(f"corpus root is not mounted: {root_path}")

    return root_path


@pytest.fixture(scope="session")
def books_root(corpus_root):
    books_path = corpus_root / "Public" / "books"
    if not books_path.is_dir():
        pytest.skip(f"corpus holds no books tree: {books_path}")

    return books_path


@pytest.fixture(scope="session")
def repository_root():
    return Path(__file__).resolve().parent.parent


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
