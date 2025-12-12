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

    @pytest.mark.parametrize(
        "inputfreq, expect_output",
        [
            (23.578038, 97.510885),
            (0.016739, 0.069225),
            (39.207278, 162.148200),
            (
                np.array([0.016739, 23.578038, 39.207278]),
                np.array([0.069225, 97.510885, 162.148200]),
            ),
            (
                None,
                np.array(
                    [97.510885, 97.510885, 97.510885, -0.069225, -0.069225, -0.069225]
                ),
            ),
        ],
    )
    def test_to_meV(self, phonon_obj, inputfreq, expect_output):
        """Test the to_meV method with single frequency input"""
        output = phonon_obj.to_meV(inputfreq)
        assert output == pytest.approx(expect_output, abs=1e-3), (
            "The output frequency in meV does not match the expected value"
        )

    @pytest.mark.parametrize("inputindex", [-1, 6])
    def test_check_index(self, phonon_obj, inputindex):
        """Test the check_index method with invalid index"""
        with pytest.raises(IndexError):
            phonon_obj.check_index(inputindex)

    def test_mode2disp(self, phonon_obj, sample_disp):
        """Test the mode2disp method"""
        for idx, disp in enumerate(sample_disp):
            computed_disp = phonon_obj.mode2disp(idx)
            for atom_disp, expected_disp in zip(computed_disp, disp, strict=False):
                assert atom_disp == pytest.approx(expected_disp, abs=1e-7), (
                    f"Displacement for mode {idx} does not match"
                )

    def test_disp(self, phonon_obj, sample_disp):
        """Test the disp property"""
        computed_disp = phonon_obj.disp
        for idx, disp in enumerate(sample_disp):
            for atom_disp, expected_disp in zip(computed_disp[idx], disp, strict=False):
                assert atom_disp == pytest.approx(expected_disp, abs=1e-7), (
                    f"Displacement for mode {idx} does not match"
                )

    def test_translation_judge(self, sample_disp):
        """Test the translation_judge method"""
        for idx, disp in enumerate(sample_disp):
            if idx < 3:
                assert Phonon.translation_judge(disp) is False, (
                    f"Mode {idx} should not be a translation mode"
                )
            else:
                assert Phonon.translation_judge(disp) is True, (
                    f"Mode {idx} should be a translation mode"
                )

    def test_translation(self, phonon_obj):
        """Test the translation property"""
        translation_idx = phonon_obj.translation
        assert len(translation_idx) == 3, "There should be 3 translation modes"
        assert all(idx in [3, 4, 5] for idx in translation_idx), (
            "Translation modes indices do not match expected values"
        )

    def test_from_file(self, data_path, phonon_obj):
        """Test the from_file method of the Phonon class"""
        outcar_path = data_path / "OUTCAR-phonon-SiC"
        poscar_path = data_path / "POSCAR-SiC"
        phonon_obj_fromfile = Phonon.from_file(outcar_path, poscar_path)

        keys = ["freq", "mode", "mass"]
        for key in keys:
            assert hasattr(phonon_obj_fromfile, key), (
                f"Phonon object should have attribute {key}"
            )

        assert phonon_obj_fromfile.freq == pytest.approx(phonon_obj.freq, abs=1e-7), (
            "Frequency data from file does not match expected value"
        )

        for mode, expected_mode in zip(
            phonon_obj_fromfile.mode, phonon_obj.mode, strict=False
        ):
            for atom_disp, expected_disp in zip(mode, expected_mode, strict=False):
                assert atom_disp == pytest.approx(expected_disp, abs=1e-6), (
                    "Mode data from file does not match expected value"
                )

    @pytest.mark.parametrize(
        "T, free_energy_expect",
        [(0, 0.14626639), (300, 0.14446110), (1000, 0.04560230)],
    )
    def test_free_energy(self, phonon_obj, T, free_energy_expect):
        """Test the free_energy method of the Phonon class"""
        free_energy = phonon_obj.free_energy(T)
        assert free_energy == pytest.approx(free_energy_expect, abs=1e-6), (
            f"Free energy at {T} K does not match expected value"
        )

    @pytest.mark.parametrize("T", [-300, -20, -1])
    def test_free_energy_error(self, phonon_obj, T):
        """Test the free_energy method with invalid temperature"""
        with pytest.raises(
            AssertionError, match="The temperature must be greater than or equal to 0 K"
        ):
            phonon_obj.free_energy(T)

    @pytest.mark.parametrize(
        "T, phonon_freq, phonon_number_expect",
        [
            (300, 23.578038, 0.02355037),
            (1000, 23.578038, 0.47607434),
            (300, -0.016739, -373.93864462),
            (
                300,
                None,
                np.array(
                    [
                        0.02355037,
                        0.02355037,
                        0.02355037,
                        -373.93864462,
                        -373.93864462,
                        -373.93864462,
                    ]
                ),
            ),
        ],
    )
    def test_phonon_number(self, phonon_obj, T, phonon_freq, phonon_number_expect):
        """Test the phonon_number method of the Phonon class"""
        phonon_number = phonon_obj.phonon_number(T, phonon_freq)
        assert phonon_number == pytest.approx(phonon_number_expect, abs=1e-6), (
            f"Phonon number at {T} K for frequency {phonon_freq}"
            f" does not match expected value"
        )

    @pytest.mark.parametrize("T", [-300, -20, 0])
    def test_phonon_number_error(self, phonon_obj, T):
        """Test the phonon_number method with invalid temperature"""
        with pytest.raises(
            AssertionError, match="The temperature must be greater than 0 K"
        ):
            phonon_obj.phonon_number(T)

    @pytest.mark.parametrize(
        "disp, coe_expected",
        [
            (
                np.array(
                    [
                        [-0.10196246, -0.02344259, 0.10300463],
                        [0.23452024, 0.05175483, -0.24175878],
                    ]
                ),
                np.array([1.0, -1.0, 0.01, -0.01, 0.001, -0.001]),
            ),
            (
                np.array(
                    [
                        [17.26218793, 1.97225633, -3.15990046],
                        [17.23575297, 1.96924405, -3.15506097],
                    ]
                ),
                np.array([0.0, 0.0, 0.0, 100.34, 20, -45]),
            ),
            (
                np.array(
                    [
                        [-1.16689884, 0.27883231, 2.09420136],
                        [2.73281044, -0.65301071, -4.90450077],
                    ]
                ),
                np.array([10.5, -20.3, -5, 0.0, 0.0, 0.0]),
            ),
        ],
    )
    def test_decompose(self, phonon_obj, disp, coe_expected):
        """Test the decompose method of the Phonon class"""
        decomposed_coe = phonon_obj.decompose(disp)
        assert len(decomposed_coe) == len(coe_expected), (
            "The length of decomposed coefficients does not match the number of modes"
        )
        assert decomposed_coe == pytest.approx(coe_expected, abs=1e-3), (
            "Decomposed coefficients do not match the expected values"
        )

    @pytest.mark.parametrize(
        "disp",
        [
            np.array([1, 2, 3]),
            np.array([[1, 2]]),
        ],
    )
    def test_decompose_error_shape(self, phonon_obj, disp):
        """Test the decompose method with invalid displacement shape"""
        with pytest.raises(
            AssertionError,
            match=r"The input displacement must be a 2D array with shape \(n, 3\)",
        ):
            phonon_obj.decompose(disp)

    @pytest.mark.parametrize(
        "disp", [np.array([[1, 2, 3]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
    )
    def test_decompose_error_length(self, phonon_obj, disp):
        """Test the decompose method with invalid displacement length"""
        with pytest.raises(
            AssertionError, match="The length of displacement must match the atoms"
        ):
            phonon_obj.decompose(disp)
