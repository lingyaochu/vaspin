"""Test suite for the phonon module."""

import numpy as np
import pytest

from vaspin.post.phonon import Phonon
from vaspin.utils.datatype import PhoData


class TestPhonon:
    """Test class for the Phonon class."""

    @pytest.fixture(scope="class")
    def sample_phodata(self):
        """Fixture to provide sample phonon data.

        The data is the phonon calculated from primitive cell of 3C-SiC
        """
        data = {
            "freq": np.array(
                [23.578038, 23.578038, 23.578038, -0.016739, -0.016739, -0.016739]
            ),
            "mode": np.array(
                [
                    [[-0.534727, -0.113898, 0.000000], [0.818942, 0.174436, 0.000000]],
                    [[0.000000, -0.000000, -0.546723], [-0.000000, 0.000000, 0.837314]],
                    [[0.113898, -0.534727, 0.000000], [-0.174436, 0.818942, 0.000000]],
                    [[0.720167, 0.427145, -0.000000], [0.470232, 0.278904, 0.000000]],
                    [
                        [0.000000, -0.000000, -0.837314],
                        [-0.000000, 0.000000, -0.546723],
                    ],
                    [[-0.427145, 0.720167, -0.000000], [-0.278904, 0.470232, 0.000000]],
                ]
            ),
            "mass": np.array([28.086, 12.011]),
        }
        return PhoData(**data)

    @pytest.fixture(scope="class")
    def sample_disp(self):
        """Fixture to provide sample displacement data.

        The data is mode / sqrt(mass) for the sample phonon data
        for the test purpose of Phonon.mode2disp method.
        """
        return np.array(
            [
                [
                    [-0.10089907, -0.02149172, 0.00000000],
                    [0.23629991, 0.05033227, 0.00000000],
                ],
                [
                    [0.00000000, -0.00000000, -0.10316262],
                    [-0.00000000, 0.00000000, 0.24160102],
                ],
                [
                    [0.02149172, -0.10089907, 0.00000000],
                    [-0.05033227, 0.23629991, 0.00000000],
                ],
                [
                    [0.13589024, 0.08059914, -0.00000000],
                    [0.13568211, 0.08047577, 0.00000000],
                ],
                [
                    [0.00000000, -0.00000000, -0.15799502],
                    [-0.00000000, 0.00000000, -0.15775305],
                ],
                [
                    [-0.08059914, 0.13589024, -0.00000000],
                    [-0.08047577, 0.13568211, 0.00000000],
                ],
            ]
        )

    @pytest.fixture(scope="class")
    def phonon_obj(self, sample_phodata):
        """Fixture to create a Phonon object from sample phonon data."""
        return Phonon(sample_phodata)

    def test_init(self, sample_phodata, phonon_obj):
        """Test the initialization of the Phonon class."""
        assert phonon_obj.freq == pytest.approx(sample_phodata.freq, abs=1e-7), (
            "The frequency data does not match"
        )

        for mode, expected_mode in zip(
            phonon_obj.mode, sample_phodata.mode, strict=False
        ):
            for atom_disp, expected_disp in zip(mode, expected_mode, strict=False):
                assert atom_disp == pytest.approx(expected_disp, abs=1e-7), (
                    "The mode data does not match"
                )

        assert phonon_obj.mass == pytest.approx(sample_phodata.mass, abs=1e-7), (
            "The mass data does not match"
        )

    def test_mode2disp(self, phonon_obj, sample_disp):
        """Test the mode2disp method"""
        for idx, disp in enumerate(sample_disp):
            computed_disp = phonon_obj.mode2disp(idx)
            for atom_disp, expected_disp in zip(computed_disp, disp, strict=False):
                assert atom_disp == pytest.approx(expected_disp, abs=1e-7), (
                    f"Displacement for mode {idx} does not match"
                )
