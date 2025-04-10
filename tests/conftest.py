"""Configure for pytest"""

import pytest
from pathlib import Path


@pytest.fixture
def data_path():
    """The paths of files prepared for tests."""
    return Path(__file__).parent / "data"
