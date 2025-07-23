"""Test module for Incar class core functionality."""

import copy
import json

import pytest

from vaspin import Incar
from vaspin.incar.tags import Tag


class TestIncarCore:
    """Tests for the Incar class functionality."""

    @pytest.fixture(scope="class")
    def incar_instance(self):
        """Fixture to create an instance of Incar."""
        return Incar({"ENCUT": 520, "ISMEAR": 0})

    def test_init_with_data(self, incar_instance):
        """Test initialization with data provided"""
        assert isinstance(incar_instance, Incar)
        assert "ENCUT" in incar_instance
        assert "ISMEAR" in incar_instance
        assert incar_instance["ENCUT"] == 520
        assert incar_instance["ISMEAR"] == 0
        assert incar_instance.get("ENCUT") == Tag(name="ENCUT", value=520)
        assert incar_instance.get("ISMEAR") == Tag(name="ISMEAR", value=0)
        assert incar_instance.get("NON_EXISTENT") is None
        assert len(incar_instance) == 2

    def test_init_without_data(self):
        """Test initialization without data"""
        incar_empty = Incar()
        assert isinstance(incar_empty, Incar)
        assert len(incar_empty) == 0
        incar_empty["ENCUT"] = 520
        incar_empty["ISMEAR"] = 0
        assert incar_empty["ENCUT"] == 520
        assert incar_empty["ISMEAR"] == 0
        assert len(incar_empty) == 2

    def test_init_from_file(self, data_path_incar):
        """Test initialization from a file"""
        incar_file = data_path_incar / "INCAR"
        incar_from_file = Incar.from_file(incar_file.as_posix())
        assert isinstance(incar_from_file, Incar)
        assert len(incar_from_file) == 19

        with open(data_path_incar / "incar.json", "r") as f:
            incar_data = json.load(f)

        for key, value in incar_data.items():
            assert key in incar_from_file
            assert incar_from_file[key] == value

    def test_init_from_file_exception(self):
        """Test init Incar from file with exception"""
        with pytest.raises(TypeError):
            Incar.from_file(12345)  # type: ignore
        with pytest.raises(TypeError):
            Incar.from_file(None)  # type: ignore
        with pytest.raises(TypeError):
            Incar.from_file(123.45)  # type: ignore
        with pytest.raises(TypeError):
            Incar.from_file([1, 2, 3])  # type: ignore
        with pytest.raises(TypeError):
            Incar.from_file({"key": "value"})  # type: ignore

        with pytest.raises(FileNotFoundError):
            Incar.from_file("NON_EXISTENT_FILE")

    def test_delte_item(self, incar_instance):
        """Test deletion of an item"""
        incar_to_delete = copy.copy(incar_instance)
        del incar_to_delete["ENCUT"]
        assert "ENCUT" not in incar_to_delete
        assert len(incar_to_delete) == 1

    def test_write_incar(self, tmp_path, incar_instance):
        """Test writing INCAR to a file"""
        incar_instance.write_incar(tmp_path.as_posix(), name="INCAR_TEST")
        written_file = tmp_path / "INCAR_TEST"
        assert written_file.exists()
        with open(written_file, "r") as f:
            content = f.read()
            assert "ENCUT = 520" in content
            assert "ISMEAR = 0" in content
