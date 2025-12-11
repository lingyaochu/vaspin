"""Test suite for io module"""

import numpy as np
import pytest

from vaspin.io import poscar_to_json, write_poscar


def test_write_poscar(tmp_path):
    """Test writing POSCAR file"""
    lattice = np.eye(3)
    atoms = np.array(["Si", "Si", "C"])
    coor_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        ]
    )
    comment = "written by test"

    write_poscar(
        lattice=lattice,
        atoms=atoms,
        coor_frac=coor_frac,
        directory=str(tmp_path),
        comment=comment,
    )

    poscar_path = tmp_path / "POSCAR"
    assert poscar_path.exists()

    expected_content = """written by test
 1.0
   1.0000000000000000   0.0000000000000000   0.0000000000000000
   0.0000000000000000   1.0000000000000000   0.0000000000000000
   0.0000000000000000   0.0000000000000000   1.0000000000000000
 C Si
 1 2
Direct
   0.0000000000000000   0.0000000000000000   0.0000000000000000
   0.5000000000000000   0.5000000000000000   0.5000000000000000
   0.2500000000000000   0.2500000000000000   0.2500000000000000
"""
    # use readlines to avoid trailing newline issues
    with open(poscar_path, "r") as f:
        assert f.read().strip() == expected_content.strip(), (
            f"Content of {poscar_path} does not match expected content."
        )


def test_poscar_to_json_invalid_format(tmp_path):
    """Test poscar_to_json with an invalid POSCAR file format"""
    invalid_poscar_content = """Invalid POSCAR
not_a_number
1 0 0
0 1 0
0 0 1
Si
1
Direct
0 0 0
"""
    invalid_poscar_path = tmp_path / "POSCAR_invalid"
    with open(invalid_poscar_path, "w") as f:
        f.write(invalid_poscar_content)

    with pytest.raises(ValueError):
        poscar_to_json(str(invalid_poscar_path))


def test_poscar_to_json_unreadable_file(tmp_path):
    """Test poscar_to_json with an unreadable file (e.g., a directory)"""
    unreadable_path = tmp_path / "unreadable_dir"
    unreadable_path.mkdir()

    with pytest.raises(ValueError, match="Cannot parse POSCAR file"):
        poscar_to_json(str(unreadable_path))
