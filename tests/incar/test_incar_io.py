"""Test cases for reading and writing INCAR files."""

import json

import pytest

from vaspin.incar.tags import Tag
from vaspin.io import incar_to_json, read_incar, write_incar


class TestReadIncar:
    """Test cases for reading INCAR files."""

    @pytest.fixture(scope="class")
    def incar_file(self, data_path_incar):
        """The INCAR file to be read"""
        return (data_path_incar / "INCAR").as_posix()

    @pytest.fixture(scope="class")
    def incar_json_file(self, data_path_incar):
        """The INCAR JSON file to be read"""
        return (data_path_incar / "incar.json").as_posix()

    def test_read_incar(self, incar_file, incar_json_file):
        """Test reading INCAR file"""
        incar = read_incar(incar_file)
        with open(incar_json_file, "r") as f:
            expected_data = json.load(f)
        assert incar == expected_data, "INCAR data does not match expected data."

    def test_incar_to_json_unreasonable_file(self, tmp_path):
        """Test reading an invalid INCAR file"""
        invalid_path = tmp_path / "invalid"
        invalid_path.mkdir()
        with pytest.raises(ValueError, match="Cannot parse INCAR file"):
            incar_to_json(invalid_path.as_posix())

    def test_read_incar_invalid_format(self, data_path_incar):
        """Test reading an INCAR file with invalid format"""
        invalid_incar_path = data_path_incar / "INCAR_invalid"
        with pytest.raises(ValueError, match="Invalid INCAR line:LHYPERFINE"):
            read_incar(invalid_incar_path.as_posix())


class TestWriteIncar:
    """Test cases for writing INCAR files."""

    @pytest.fixture(scope="class")
    def tags(self, data_path_incar):
        """The tags to be written to INCAR"""
        with open(data_path_incar / "incar.json", "r") as f:
            data = json.load(f)
        tags = [Tag(name=tag, value=value) for tag, value in data.items()]
        return tags

    @pytest.fixture(scope="class")
    def incar_content(self, data_path_incar):
        """The expected content of the INCAR file"""
        with open(data_path_incar / "INCAR_written", "r") as f:
            incar_string = f.read()
        return incar_string

    def test_write_incar(self, tmp_path, tags, incar_content):
        """Test writing INCAR file"""
        write_incar(tags=tags, directory=tmp_path, name="INCAR_test")
        with open(tmp_path / "INCAR_test", "r") as f:
            content = f.read() + "\n"
        assert content == incar_content, (
            "Written INCAR content does not match expected content."
        )
