"""Test suite for OUTCAR parser."""

from typing import Any, List

import pytest

from vaspin import VaspOutcarParser
from vaspin.utils.datatype import SymTensor

FLOAT_TOL = 1e-7


@pytest.fixture
def outcar_opt(data_path):
    """Fixture for sample OUTCAR for energy, forces, the last site potential"""
    return (data_path / "OUTCAR-opt").as_posix()


@pytest.fixture
def outcar_spin(data_path):
    """Fixture for sample OUTCAR for ZFS tensor and hyperfine"""
    return (data_path / "OUTCAR-spin").as_posix()


@pytest.fixture
def outcar_phonon(data_path):
    """Fixture for sample OUTCAR for phonon"""
    return (data_path / "OUTCAR-phonon").as_posix()


@pytest.fixture
def outcar_normal(data_path):
    """Fixture for sample OUTCAR for normal properties

    The normal properties include:
    - the total number of ions
    - the total number of electrons
    - the dielectric tensor
    """
    return (data_path / "OUTCAR-normal").as_posix()


@pytest.fixture
def handlers_opt():
    """Fixture for handlers for optimization"""
    return ["N ions", "Energy", "Forces", "Site potential"]


@pytest.fixture
def handlers_spin():
    """Fixture for handlers for spin properties"""
    return ["N ions", "D tensor", "Hyperfine fermi", "Hyperfine dipolar"]


@pytest.fixture
def handlers_phonon():
    """Fixture for handlers for phonon properties"""
    return ["N ions", "Phonon"]


@pytest.fixture
def handlers_normal():
    """Fixture for handlers for normal properties"""
    return ["N ions", "N electrons", "Dielectric ele", "Dielectric ion"]


@pytest.fixture
def sample_data_energy():
    """The expected data for number of ions"""
    return [-19.09630729, -19.19799451, -19.24878374, -19.24974113, -19.24977097]


@pytest.fixture
def sample_data_forces():
    """The expected data for forces"""
    return [
        [[-2.803711, -1.652483, -1.652483], [2.803711, 1.652483, 1.652483]],
        [[1.286982, 0.927788, 0.927788], [-1.286982, -0.927788, -0.927788]],
        [[0.145375, 0.160714, 0.160714], [-0.145375, -0.160714, -0.160714]],
        [[0.005776, -0.033844, -0.033844], [-0.005776, 0.033844, 0.033844]],
        [[-0.004093, -0.005553, -0.005553], [0.004093, 0.005553, 0.005553]],
    ]


@pytest.fixture
def sample_data_site_potential():
    """The expected data for the last site potential"""
    return [-45.4085, -45.4085]


@pytest.fixture
def sample_data_zfs():
    """The expected data for ZFS tensor"""
    return (
        SymTensor()
        .from_sequence([-0.000, 0.000, 0.000, 720.811, 720.811, 720.811])
        .get_matrix_sym()
        .tolist()
    )


@pytest.fixture
def sample_data_hyperfine_fermi():
    """The expected data for hyperfine fermi"""
    return {
        "A_pw": [-0.302, -0.302, -0.302, 21.232, 21.232, 21.232, 5.690],
        "A_ps": [-0.310, -0.310, -0.310, 21.364, 21.364, 21.364, 5.691],
        "A_ae": [-9.042, -9.042, -9.042, 165.189, 165.189, 165.189, 13.576],
        "A_c": [0.120, 0.120, 0.120, -34.766, -34.766, -34.766, -7.416],
    }


@pytest.fixture
def sample_data_hyperfine_dipolar():
    """The expected data for hyperfine dipolar"""
    origindata = [
        [0.455, -0.910, 0.455, 0.217, -1.001, 0.217],
        [0.455, 0.455, -0.910, -1.001, 0.217, 0.217],
        [-0.910, 0.455, 0.455, 0.217, 0.217, -1.001],
        [-0.453, 0.907, -0.454, -24.135, 23.460, -24.135],
        [-0.453, -0.453, 0.907, 23.460, -24.135, -24.135],
        [0.907, -0.453, -0.454, -24.135, -24.135, 23.460],
        [0.000, 0.000, -0.000, 3.359, 3.359, 3.359],
    ]
    return [
        SymTensor().from_sequence(row).get_matrix_sym().tolist() for row in origindata
    ]


@pytest.fixture
def sample_data_phonon():
    """The expected data for phonon"""
    return {
        "frequencies": [
            39.207278,
            39.207278,
            39.207278,
            0.000001,
            -0.000000,
            -0.000001,
        ],
        "eigenmodes": [
            [[0.224710, 0.596676, -0.305751], [-0.224710, -0.596676, 0.305751]],
            [[0.277594, 0.210727, 0.615253], [-0.277594, -0.210727, -0.615253]],
            [[0.610284, -0.315550, -0.167275], [-0.610284, 0.315550, 0.167275]],
            [[0.084778, 0.701990, -0.004830], [0.084778, 0.701990, -0.004830]],
            [[0.701268, -0.084464, 0.032997], [0.701268, -0.084464, 0.032997]],
            [[-0.032181, 0.008746, 0.706320], [-0.032181, 0.008746, 0.706320]],
        ],
    }


@pytest.fixture
def sample_data_ions():
    """The expected data for number of ions"""
    return 8


@pytest.fixture
def sample_data_electrons():
    """The expected data for number of electrons"""
    return 32.00


@pytest.fixture
def sample_data_dielectric_ele():
    """The expected data for the electron contribution to dielctric tensor"""
    return [
        [7.097446, 0.000000, 0.000000],
        [0.000000, 7.097446, -0.000000],
        [0.000000, 0.000000, 7.446904],
    ]


@pytest.fixture
def sample_data_dielectric_ion():
    """The expected data for the ion contribution to dielctric tensor"""
    return [
        [3.256224, -0.000000, 0.000000],
        [-0.000000, 3.256224, 0.000000],
        [0.000000, 0.000000, 3.654108],
    ]


def set_parser(outcar_path: str, handlers: List[str]) -> dict[str, Any]:
    """Set up the parser with the given path and handlers."""
    parser = VaspOutcarParser(outcar_path)
    parser.set_handlers(handlers)
    parser.parse()
    return parser.data


@pytest.fixture
def parsed_data_opt(outcar_opt, handlers_opt):
    """Fixture to get parsed data for optimization OUTCAR"""
    return set_parser(outcar_opt, handlers_opt)


@pytest.fixture
def parsed_data_spin(outcar_spin, handlers_spin):
    """Fixture to get parsed data for spin OUTCAR"""
    return set_parser(outcar_spin, handlers_spin)


@pytest.fixture
def parsed_data_phonon(outcar_phonon, handlers_phonon):
    """Fixture to get parsed data for phonon OUTCAR"""
    return set_parser(outcar_phonon, handlers_phonon)


@pytest.fixture
def parsed_data_normal(outcar_normal, handlers_normal):
    """Fixture to get parsed data for normal OUTCAR"""
    return set_parser(outcar_normal, handlers_normal)


def test_energy(parsed_data_opt, sample_data_energy):
    """Test energy extraction from OUTCAR"""
    assert "Ionic energy" in parsed_data_opt, "Energy are not parsed"
    assert parsed_data_opt["Ionic energy"] == pytest.approx(
        sample_data_energy, abs=FLOAT_TOL
    ), "Energy data mismatch"


def test_forces(parsed_data_opt, sample_data_forces):
    """Test forces extraction from OUTCAR"""
    assert "forces" in parsed_data_opt, "Forces are not parsed"
    forces = parsed_data_opt["forces"]
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


def test_site_potential(parsed_data_opt, sample_data_site_potential):
    """Test site potential extraction from OUTCAR"""
    assert "site_potential" in parsed_data_opt, "Site potential are not parsed"
    assert parsed_data_opt["site_potential"] == pytest.approx(
        sample_data_site_potential, abs=FLOAT_TOL
    ), "Site potential data mismatch"


def test_zfs(parsed_data_spin, sample_data_zfs):
    """Test ZFS tensor extraction from OUTCAR"""
    assert "D tensor" in parsed_data_spin, "ZFS tensor are not parsed"
    for actual_row, expected_row in zip(
        parsed_data_spin["D tensor"], sample_data_zfs, strict=False
    ):
        assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
            "ZFS tensor data mismatch"
        )


def test_hyperfine_fermi(parsed_data_spin, sample_data_hyperfine_fermi):
    """Test the fermi contact term of hyperfine extraction from OUTCAR"""
    assert "hyperfine_fermi" in parsed_data_spin, "Hyperfine fermi are not parsed"
    assert parsed_data_spin["hyperfine_fermi"] == pytest.approx(
        sample_data_hyperfine_fermi, abs=FLOAT_TOL
    ), "Hyperfine fermi data mismatch"


def test_hyperfine_dipolar(parsed_data_spin, sample_data_hyperfine_dipolar):
    """Test the dipolar term of hyperfine extraction from OUTCAR"""
    assert "hyperfine_dipolar" in parsed_data_spin, "Hyperfine dipolar are not parsed"
    assert len(parsed_data_spin["hyperfine_dipolar"]) == len(
        sample_data_hyperfine_dipolar
    ), "Number of hyperfine dipolar tensors mismatch"
    for actual_tensor, expected_tensor in zip(
        parsed_data_spin["hyperfine_dipolar"],
        sample_data_hyperfine_dipolar,
        strict=False,
    ):
        for actual_row, expected_row in zip(
            actual_tensor, expected_tensor, strict=False
        ):
            assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
                "Hyperfine dipolar tensor data mismatch"
            )


def test_phonon(parsed_data_phonon, sample_data_phonon):
    """Test phonon extraction from OUTCAR"""
    assert "phonon" in parsed_data_phonon, "Phonon data are not parsed"
    assert "frequencies" in parsed_data_phonon["phonon"], (
        "Phonon frequencies are not parsed"
    )
    assert "eigenmodes" in parsed_data_phonon["phonon"], (
        "Phonon eigenmodes are not parsed"
    )
    assert parsed_data_phonon["phonon"]["frequencies"] == pytest.approx(
        sample_data_phonon["frequencies"], abs=FLOAT_TOL
    ), "Phonon frequencies data mismatch"
    for actual_mode, expected_mode in zip(
        parsed_data_phonon["phonon"]["eigenmodes"],
        sample_data_phonon["eigenmodes"],
        strict=False,
    ):
        for actual_disp, expected_disp in zip(actual_mode, expected_mode, strict=False):
            assert actual_disp == pytest.approx(expected_disp, abs=FLOAT_TOL), (
                "Phonon eigenmodes data mismatch"
            )


def test_ions(parsed_data_normal, sample_data_ions):
    """Test number of ions extraction from OUTCAR"""
    assert "N ions" in parsed_data_normal, "Number of ions are not parsed"
    assert parsed_data_normal["N ions"] == sample_data_ions, (
        "Number of ions data mismatch"
    )


def test_electrons(parsed_data_normal, sample_data_electrons):
    """Test number of electrons extraction from OUTCAR"""
    assert "N electrons" in parsed_data_normal, "Number of electrons are not parsed"
    assert parsed_data_normal["N electrons"] == pytest.approx(
        sample_data_electrons, abs=FLOAT_TOL
    ), "Number of electrons data mismatch"


def test_dielectric_ele(parsed_data_normal, sample_data_dielectric_ele):
    """Test electronic part dielectric tensor extraction from OUTCAR"""
    assert "dielectric_ele" in parsed_data_normal, (
        "Electron part of dielectric tensor are not parsed"
    )
    for actual_row, expected_row in zip(
        parsed_data_normal["dielectric_ele"], sample_data_dielectric_ele, strict=False
    ):
        assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
            "Electron part of dielectric tensor data mismatch"
        )


def test_dielectric_ion(parsed_data_normal, sample_data_dielectric_ion):
    """Test ionic part dielectric tensor extraction from OUTCAR"""
    assert "dielectric_ion" in parsed_data_normal, (
        "Ion part of dielectric tensor are not parsed"
    )
    for actual_row, expected_row in zip(
        parsed_data_normal["dielectric_ion"], sample_data_dielectric_ion, strict=False
    ):
        assert actual_row == pytest.approx(expected_row, abs=FLOAT_TOL), (
            "Ion part of dielectric tensor data mismatch"
        )
