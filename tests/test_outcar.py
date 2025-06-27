"""Test suite for OUTCAR parser."""

from typing import Any, List

import pytest

from vaspin import VaspOutcarParser
from vaspin.utils.datatype import SymTensor

FLOAT_TOL = 1e-7


def set_parser(outcar_path: str, handlers: List[str]) -> dict[str, Any]:
    """Set up the parser with the given path and handlers."""
    parser = VaspOutcarParser(outcar_path)
    parser.set_handlers(handlers)
    parser.parse()
    return parser.data


class TestOptOUTCAR:
    """Test class for OUTCAR from an optimization job."""

    @pytest.fixture(scope="class")
    def parsed_data(self, data_path):
        """Fixture for parsed data from OUTCAR-opt."""
        outcar_path = (data_path / "OUTCAR-opt").as_posix()
        handlers = ["N ions", "Energy", "Forces", "Site potential"]
        return set_parser(outcar_path, handlers)

    def test_energy(self, parsed_data):
        """Test energy extraction from OUTCAR."""
        sample_data_energy = [
            -19.09630729,
            -19.19799451,
            -19.24878374,
            -19.24974113,
            -19.24977097,
        ]
        assert "Ionic energy" in parsed_data, "Energy are not parsed"
        assert parsed_data["Ionic energy"] == pytest.approx(
            sample_data_energy, abs=FLOAT_TOL
        ), "Energy data mismatch"

    def test_forces(self, parsed_data):
        """Test forces extraction from OUTCAR."""
        sample_data_forces = [
            [[-2.803711, -1.652483, -1.652483], [2.803711, 1.652483, 1.652483]],
            [[1.286982, 0.927788, 0.927788], [-1.286982, -0.927788, -0.927788]],
            [[0.145375, 0.160714, 0.160714], [-0.145375, -0.160714, -0.160714]],
            [[0.005776, -0.033844, -0.033844], [-0.005776, 0.033844, 0.033844]],
            [[-0.004093, -0.005553, -0.005553], [0.004093, 0.005553, 0.005553]],
        ]
        assert "forces" in parsed_data, "Forces are not parsed"
        forces = parsed_data["forces"]
        assert len(forces) == len(sample_data_forces), "Number of ionic steps mismatch"
        for i, (actual_step, expected_step) in enumerate(
            zip(forces, sample_data_forces, strict=False)
        ):
            for j, (force_on_atom_actual, force_on_atom_expected) in enumerate(
                zip(actual_step, expected_step, strict=False)
            ):
                assert force_on_atom_actual == pytest.approx(
                    force_on_atom_expected, abs=FLOAT_TOL
                ), f"Forces data mismatch at ionic step {i}, atom {j}"

    def test_site_potential(self, parsed_data):
        """Test site potential extraction from OUTCAR."""
        sample_data_site_potential = [-45.4085, -45.4085]
        assert "site_potential" in parsed_data, "Site potential are not parsed"
        assert parsed_data["site_potential"] == pytest.approx(
            sample_data_site_potential, abs=FLOAT_TOL
        ), "Site potential data mismatch"


class TestSpinOUTCAR:
    """Test class for OUTCAR from a spin calculation."""

    @pytest.fixture(scope="class")
    def parsed_data(self, data_path):
        """Fixture for parsed data from OUTCAR-spin."""
        outcar_path = (data_path / "OUTCAR-spin").as_posix()
        handlers = ["N ions", "D tensor", "Hyperfine fermi", "Hyperfine dipolar"]
        return set_parser(outcar_path, handlers)

    def test_zfs(self, parsed_data):
        """Test ZFS tensor extraction from OUTCAR."""
        sample_data_zfs = (
            SymTensor()
            .from_sequence([-0.000, 0.000, 0.000, 720.811, 720.811, 720.811])
            .get_matrix_sym()
            .tolist()
        )
        assert "D tensor" in parsed_data, "ZFS tensor are not parsed"
        for actual_row, expected_row in zip(
            parsed_data["D tensor"], sample_data_zfs, strict=False
        ):
            assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
                "ZFS tensor data mismatch"
            )

    def test_hyperfine_fermi(self, parsed_data):
        """Test the fermi contact term of hyperfine extraction from OUTCAR."""
        sample_data_hyperfine_fermi = {
            "A_pw": [-0.302, -0.302, -0.302, 21.232, 21.232, 21.232, 5.690],
            "A_ps": [-0.310, -0.310, -0.310, 21.364, 21.364, 21.364, 5.691],
            "A_ae": [-9.042, -9.042, -9.042, 165.189, 165.189, 165.189, 13.576],
            "A_c": [0.120, 0.120, 0.120, -34.766, -34.766, -34.766, -7.416],
        }
        assert "hyperfine_fermi" in parsed_data, "Hyperfine fermi are not parsed"
        assert parsed_data["hyperfine_fermi"] == pytest.approx(
            sample_data_hyperfine_fermi, abs=FLOAT_TOL
        ), "Hyperfine fermi data mismatch"

    def test_hyperfine_dipolar(self, parsed_data):
        """Test the dipolar term of hyperfine extraction from OUTCAR."""
        origindata = [
            [0.455, -0.910, 0.455, 0.217, -1.001, 0.217],
            [0.455, 0.455, -0.910, -1.001, 0.217, 0.217],
            [-0.910, 0.455, 0.455, 0.217, 0.217, -1.001],
            [-0.453, 0.907, -0.454, -24.135, 23.460, -24.135],
            [-0.453, -0.453, 0.907, 23.460, -24.135, -24.135],
            [0.907, -0.453, -0.454, -24.135, -24.135, 23.460],
            [0.000, 0.000, -0.000, 3.359, 3.359, 3.359],
        ]
        sample_data_hyperfine_dipolar = [
            SymTensor().from_sequence(row).get_matrix_sym().tolist()
            for row in origindata
        ]
        assert "hyperfine_dipolar" in parsed_data, "Hyperfine dipolar are not parsed"
        assert len(parsed_data["hyperfine_dipolar"]) == len(
            sample_data_hyperfine_dipolar
        ), "Number of hyperfine dipolar tensors mismatch"
        for actual_tensor, expected_tensor in zip(
            parsed_data["hyperfine_dipolar"],
            sample_data_hyperfine_dipolar,
            strict=False,
        ):
            for actual_row, expected_row in zip(
                actual_tensor, expected_tensor, strict=False
            ):
                assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
                    "Hyperfine dipolar tensor data mismatch"
                )


class TestPhononOUTCAR:
    """Test class for OUTCAR from a phonon calculation."""

    @pytest.fixture(scope="class")
    def parsed_data(self, data_path):
        """Fixture for parsed data from OUTCAR-phonon."""
        outcar_path = (data_path / "OUTCAR-phonon").as_posix()
        handlers = ["N ions", "Phonon"]
        return set_parser(outcar_path, handlers)

    def test_phonon(self, parsed_data):
        """Test phonon extraction from OUTCAR."""
        sample_data_phonon = {
            "frequencies": [
                39.207278,
                39.207278,
                39.207278,
                0.000001,
                -0.000000,
                -0.000001,
            ],
            "eigenmodes": [
                [
                    [0.224710, 0.596676, -0.305751],
                    [-0.224710, -0.596676, 0.305751],
                ],
                [
                    [0.277594, 0.210727, 0.615253],
                    [-0.277594, -0.210727, -0.615253],
                ],
                [
                    [0.610284, -0.315550, -0.167275],
                    [-0.610284, 0.315550, 0.167275],
                ],
                [[0.084778, 0.701990, -0.004830], [0.084778, 0.701990, -0.004830]],
                [
                    [0.701268, -0.084464, 0.032997],
                    [0.701268, -0.084464, 0.032997],
                ],
                [
                    [-0.032181, 0.008746, 0.706320],
                    [-0.032181, 0.008746, 0.706320],
                ],
            ],
        }
        assert "phonon" in parsed_data, "Phonon data are not parsed"
        assert "frequencies" in parsed_data["phonon"], (
            "Phonon frequencies are not parsed"
        )
        assert "eigenmodes" in parsed_data["phonon"], "Phonon eigenmodes are not parsed"
        assert parsed_data["phonon"]["frequencies"] == pytest.approx(
            sample_data_phonon["frequencies"], abs=FLOAT_TOL
        ), "Phonon frequencies data mismatch"
        for actual_mode, expected_mode in zip(
            parsed_data["phonon"]["eigenmodes"],
            sample_data_phonon["eigenmodes"],
            strict=False,
        ):
            for actual_disp, expected_disp in zip(
                actual_mode, expected_mode, strict=False
            ):
                assert actual_disp == pytest.approx(expected_disp, abs=FLOAT_TOL), (
                    "Phonon eigenmodes data mismatch"
                )


class TestNormalOUTCAR:
    """Test class for OUTCAR from a normal calculation."""

    @pytest.fixture(scope="class")
    def parsed_data(self, data_path):
        """Fixture for parsed data from OUTCAR-normal."""
        outcar_path = (data_path / "OUTCAR-normal").as_posix()
        handlers = ["N ions", "N electrons", "Dielectric ele", "Dielectric ion"]
        return set_parser(outcar_path, handlers)

    def test_ions(self, parsed_data):
        """Test number of ions extraction from OUTCAR."""
        sample_data_ions = 8
        assert "N ions" in parsed_data, "Number of ions are not parsed"
        assert parsed_data["N ions"] == sample_data_ions, "Number of ions data mismatch"

    def test_electrons(self, parsed_data):
        """Test number of electrons extraction from OUTCAR."""
        sample_data_electrons = 32.00
        assert "N electrons" in parsed_data, "Number of electrons are not parsed"
        assert parsed_data["N electrons"] == pytest.approx(
            sample_data_electrons, abs=FLOAT_TOL
        ), "Number of electrons data mismatch"

    def test_dielectric_ele(self, parsed_data):
        """Test electronic part dielectric tensor extraction from OUTCAR."""
        sample_data_dielectric_ele = [
            [7.097446, 0.000000, 0.000000],
            [0.000000, 7.097446, -0.000000],
            [0.000000, 0.000000, 7.446904],
        ]
        assert "dielectric_ele" in parsed_data, (
            "Electron part of dielectric tensor are not parsed"
        )
        for actual_row, expected_row in zip(
            parsed_data["dielectric_ele"], sample_data_dielectric_ele, strict=False
        ):
            assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
                "Electron part of dielectric tensor data mismatch"
            )

    def test_dielectric_ion(self, parsed_data):
        """Test ionic part dielectric tensor extraction from OUTCAR."""
        sample_data_dielectric_ion = [
            [3.256224, -0.000000, 0.000000],
            [-0.000000, 3.256224, 0.000000],
            [0.000000, 0.000000, 3.654108],
        ]
        assert "dielectric_ion" in parsed_data, (
            "Ion part of dielectric tensor are not parsed"
        )
        for actual_row, expected_row in zip(
            parsed_data["dielectric_ion"], sample_data_dielectric_ion, strict=False
        ):
            assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
                "Ion part of dielectric tensor data mismatch"
            )
