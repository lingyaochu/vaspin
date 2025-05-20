"""Test POSCAR class."""

import numpy as np
import pytest

from vaspin import Poscar
from vaspin.utils import PosData

FLOAT_TOL = 1e-5


@pytest.fixture
def sample_posdata():
    """Fixture for sample PosData."""
    return PosData(
        lattice=np.array([[10.8, 0, 0], [0, 9.8, 0], [0, 0, 11.1]]),
        species=np.array(["Si", "C"]),
        number=np.array([2, 2]),
        frac=np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        ),
    )


@pytest.fixture
def cate_sample_posdata():
    """Catesian coordinates fixture for sample PosData."""
    return np.array(
        [[0.0, 0.0, 0.0], [0.0, 4.9, 5.55], [5.4, 0.0, 5.55], [5.4, 4.9, 0.0]]
    )


@pytest.fixture
def sample_poscar(data_path):
    """Fixture for sample POSCAR."""
    return (data_path / "POSCAR").as_posix()


def test_init_from_posdata(sample_posdata, cate_sample_posdata):
    """Test initializing Poscar from PosData."""
    poscar = Poscar(sample_posdata)
    assert poscar.lattice == pytest.approx(sample_posdata.lattice, abs=FLOAT_TOL), (
        "lattice is wrong"
    )
    assert np.all(poscar.atoms == np.array(["Si", "Si", "C", "C"])), "atoms are wrong"
    assert poscar.coor_frac == pytest.approx(sample_posdata.frac, abs=FLOAT_TOL), (
        "fractional coordinates are wrong"
    )
    assert poscar.coor_cate == pytest.approx(cate_sample_posdata, abs=FLOAT_TOL), (
        "cartesian coordinates are wrong"
    )
    assert poscar.abc == pytest.approx(
        {"a": 10.8, "b": 9.8, "c": 11.1}, abs=FLOAT_TOL
    ), "lattice lengths are wrong"


def test_init_from_POSCAR(sample_poscar):
    """Test initializing Poscar from POSCAR file."""
    pos = Poscar.from_file(sample_poscar)
    assert pos.comment == "The test POSCAR file"
    assert pos.lattice == pytest.approx(
        np.array(
            [
                [4.3591197305808720, 0.0, 0.0],
                [0.0, 4.3591197305808720, 0.0],
                [0.0, 0.0, 4.3591197305808720],
            ]
        ),
        abs=FLOAT_TOL,
    )
    assert np.all(pos.atoms == np.array(["Si", "Si", "Si", "Si", "C", "C", "C", "C"]))
    assert pos.coor_frac == pytest.approx(
        np.array(
            [
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.00, 0.00, 0.00],
                [0.00, 0.50, 0.50],
                [0.50, 0.00, 0.50],
                [0.50, 0.50, 0.00],
            ]
        ),
        FLOAT_TOL,
    )
    assert pos.coor_cate == pytest.approx(
        np.array(
            [
                [3.269339797935654, 1.089779932645218, 3.269339797935654],
                [3.269339797935654, 3.269339797935654, 1.089779932645218],
                [1.089779932645218, 1.089779932645218, 1.089779932645218],
                [1.089779932645218, 3.269339797935654, 3.269339797935654],
                [0.0, 0.0, 0.0],
                [0.0, 2.179559865290436, 2.179559865290436],
                [2.179559865290436, 0.0, 2.179559865290436],
                [2.179559865290436, 2.179559865290436, 0.0],
            ]
        ),
        FLOAT_TOL,
    )
    assert pos.abc == pytest.approx(
        {"a": 4.3591197305808720, "b": 4.3591197305808720, "c": 4.3591197305808720},
        FLOAT_TOL,
    )


def test_init_from_none():
    """Test initializing Poscar from non-exist file."""
    nonfile = "FILE_NOT_EXSIST"
    with pytest.raises(FileNotFoundError, match=f"File does not exist: {nonfile}"):
        Poscar.from_file(nonfile)
