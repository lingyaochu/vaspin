"""Test about strain tensor."""

import numpy as np
import pytest

from vaspin.utils import StrainTensor

FLOAT_TOL = 1e-5

LATTICE = {"lattice1": np.array([[10, 0, 0], [0, 20, 0], [0, 0, 40]])}


@pytest.fixture
def sample_strain_input():
    """Fixture for sample strain input."""
    return {"xx": 0.01, "yy": 0.015, "zz": -0.01, "xy": -0.015, "xz": 0.02, "yz": -0.02}


@pytest.fixture
def sample_strain_tensor_sym():
    """Fixture for sample symmetric strain tensor."""
    return np.array(
        [[0.01, -0.015, 0.02], [-0.015, 0.015, -0.02], [0.02, -0.02, -0.01]]
    )


@pytest.fixture
def sample_strain_tensor_unsym():
    """Fixture for sample unsymmetric strain tensor."""
    return np.array([[0.01, -0.03, 0.04], [0.0, 0.015, -0.04], [0.0, 0.0, -0.01]])


@pytest.fixture
def sample_strain_input_list(sample_strain_input):
    """Fixture for sample strain input as a list."""
    return list(sample_strain_input.values())


@pytest.fixture
def sample_strain_input_tuple(sample_strain_input_list):
    """Fixture for sample strain input as a tuple."""
    return tuple(sample_strain_input_list)


@pytest.fixture
def sample_strain_input_array(sample_strain_input_list):
    """Fixture for sample strain input as a Numpy array."""
    return np.array(sample_strain_input_list)


def test_build_strain(
    sample_strain_input, sample_strain_tensor_sym, sample_strain_tensor_unsym
):
    """Test build strain tensor from normal input"""
    test_strain = StrainTensor(**sample_strain_input)
    assert test_strain.get_matrix_sym() == pytest.approx(
        sample_strain_tensor_sym, abs=FLOAT_TOL
    )
    assert test_strain.get_matrix_unsym() == pytest.approx(
        sample_strain_tensor_unsym, abs=FLOAT_TOL
    )


@pytest.mark.parametrize(
    "input_seq_name",
    [
        "sample_strain_input_list",
        "sample_strain_input_tuple",
        "sample_strain_input_array",
    ],
)
def test_build_strain_from_squence(
    input_seq_name, request, sample_strain_tensor_sym, sample_strain_tensor_unsym
):
    """Test build strain tensor from sequence input

    Args:
        input_seq_name: A sequence(list, tuple, or Numpy array) of values.
        request: pytest fixture request object to access the fixture
        sample_strain_tensor_sym: Fixture for sample symmetric strain tensor.
        sample_strain_tensor_unsym: Fixture for sample unsymmetric strain tensor.
    """
    input_seq = request.getfixturevalue(input_seq_name)
    test_strain = StrainTensor.from_sequence(input_seq)
    assert test_strain.get_matrix_sym() == pytest.approx(
        sample_strain_tensor_sym, abs=FLOAT_TOL
    )
    assert test_strain.get_matrix_unsym() == pytest.approx(
        sample_strain_tensor_unsym, abs=FLOAT_TOL
    )
