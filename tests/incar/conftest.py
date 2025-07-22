"""Fixtures for INCAR tests."""

import pytest


@pytest.fixture(scope="session")
def data_path_incar(data_path):
    """Fixture to provide the path to the INCAR data directory."""
    return data_path / "incar"
