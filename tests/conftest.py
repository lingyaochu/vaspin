"""Configure for pytest"""

from pathlib import Path

import pytest


@pytest.fixture
def data_path():
    """The paths of files prepared for tests."""
    return Path(__file__).parent / "data"
