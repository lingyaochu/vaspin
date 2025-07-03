"""Test POSCAR class."""

import copy

import numpy as np
import pytest

from vaspin import Poscar
from vaspin.utils import PosData

FLOAT_TOL = 1e-5


class TestPoscar:
    """Test class for Poscar"""

    @pytest.fixture(scope="class")
    def sample_posdata(self):
        """Fixture for sample PosData."""
        return PosData(
            lattice=np.array([[10.8, 0, 0], [0, 9.8, 0], [0, 0, 11.1]]),
            species=np.array(["Si", "C"]),
            number=np.array([2, 2]),
            frac=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
            ),
        )

    @pytest.fixture(scope="class")
    def cate_sample_posdata(self):
        """Catesian coordinates fixture for sample PosData."""
        return np.array(
            [[0.0, 0.0, 0.0], [0.0, 4.9, 5.55], [5.4, 0.0, 5.55], [5.4, 4.9, 0.0]]
        )

    @pytest.fixture(scope="class")
    def sample_poscar_file_frac(self, data_path):
        """Fixture for sample POSCAR file with fractional coordinates"""
        return (data_path / "POSCAR_frac").as_posix()

    @pytest.fixture(scope="class")
    def sample_poscar_file_cate(self, data_path):
        """Fixture for sample POSCAR file with cartesian coordinates"""
        return (data_path / "POSCAR_cate").as_posix()

    @pytest.fixture(scope="class")
    def sample_poscar_frac(self, sample_poscar_file_frac):
        """Fixture for sample Poscar object with fractional coordinates"""
        return Poscar.from_file(sample_poscar_file_frac)

    @pytest.fixture(scope="class")
    def sample_poscar_cate(self, sample_poscar_file_cate):
        """Fixture for sample Poscar object with cartesian coordinates"""
        return Poscar.from_file(sample_poscar_file_cate)

    def test_init_from_posdata(self, sample_posdata, cate_sample_posdata):
        """Test initializing Poscar from PosData."""
        poscar = Poscar(sample_posdata)
        assert poscar.lattice == pytest.approx(sample_posdata.lattice, abs=FLOAT_TOL), (
            "lattice is wrong"
        )
        assert np.all(poscar.atoms == np.array(["Si", "Si", "C", "C"])), (
            "atoms are wrong"
        )
        assert poscar.coor_frac == pytest.approx(sample_posdata.frac, abs=FLOAT_TOL), (
            "fractional coordinates are wrong"
        )
        assert poscar.coor_cate == pytest.approx(cate_sample_posdata, abs=FLOAT_TOL), (
            "cartesian coordinates are wrong"
        )
        assert poscar.abc == pytest.approx(
            {"a": 10.8, "b": 9.8, "c": 11.1}, abs=FLOAT_TOL
        ), "lattice lengths are wrong"

    def test_init_from_noposdata(self):
        """Test initializing Poscar from non-PosData."""
        with pytest.raises(TypeError, match="posdata must be a PosData object"):
            Poscar("not a PosData instance")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "sample_poscar", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_init_from_file(self, sample_poscar, request):
        """Test initializing Poscar from file."""
        sample_poscar = request.getfixturevalue(sample_poscar)
        assert isinstance(sample_poscar, Poscar), (
            "from_file did not return Poscar instance"
        )
        assert hasattr(sample_poscar, "data"), "from_file did not set `data` attribute"
        assert isinstance(sample_poscar.data, PosData), (
            "from_file did not set `data` to PosData instance"
        )

    def test_init_from_file_nostr(self):
        """Test initializing Poscar from non-string file path."""
        with pytest.raises(TypeError, match="filepath must be a string"):
            Poscar.from_file(12345)  # type: ignore[arg-type]

    def test_init_from_none(self):
        """Test initializing Poscar from non-exist file."""
        nonfile = "FILE_NOT_EXSIST"
        with pytest.raises(FileNotFoundError, match=f"File does not exist: {nonfile}"):
            Poscar.from_file(nonfile)

    @pytest.mark.parametrize(
        "sample_poscar, comment_expected",
        [
            ("sample_poscar_frac", "The test POSCAR file fractional"),
            ("sample_poscar_cate", "The test POSCAR file cartesian"),
        ],
    )
    def test_comment_property(self, sample_poscar, comment_expected, request):
        """Test the comment property of Poscar."""
        sample_poscar = request.getfixturevalue(sample_poscar)
        assert sample_poscar.comment == comment_expected, (
            "The comment property did not match expected value"
        )

    @pytest.mark.parametrize(
        "sample_poscar", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_lattice_property(self, sample_poscar, request):
        """Test the lattice property of Poscar"""
        expected_lattice = np.array(
            [
                [4.3591197305808720, 0.0, 0.0],
                [0.0, 4.3591197305808720, 0.0],
                [0.0, 0.0, 4.3591197305808720],
            ]
        )
        sample_poscar = request.getfixturevalue(sample_poscar)
        assert sample_poscar.lattice == pytest.approx(
            expected_lattice, abs=FLOAT_TOL
        ), "The lattice property did not match expected value"

    @pytest.mark.parametrize(
        "sample_poscar", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_atoms_property(self, sample_poscar, request):
        """Test the atoms property of Poscar"""
        sample_poscar = request.getfixturevalue(sample_poscar)
        expected_atoms = np.array(["Si", "Si", "Si", "Si", "C", "C", "C", "C"])
        assert np.all(sample_poscar.atoms == expected_atoms), (
            "The atoms property did not match expected value"
        )

    @pytest.mark.parametrize(
        "sample_poscar", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_coor_frac_property(self, sample_poscar, request):
        """Test the coor_frac property of Poscar"""
        sample_poscar = request.getfixturevalue(sample_poscar)
        expected_coor_frac = np.array(
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
        )
        assert sample_poscar.coor_frac == pytest.approx(
            expected_coor_frac, abs=FLOAT_TOL
        ), "The coor_frac property did not match expected value"

    @pytest.mark.parametrize(
        "sample_poscar", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_coor_cate_property(self, sample_poscar, request):
        """Test the coor_cate property of Poscar"""
        sample_poscar = request.getfixturevalue(sample_poscar)
        expected_coor_cate = np.array(
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
        )
        assert sample_poscar.coor_cate == pytest.approx(
            expected_coor_cate, abs=FLOAT_TOL
        ), "The coor_cate property did not match expected value"

    @pytest.mark.parametrize(
        "sample_poscar", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_abc_property(self, sample_poscar, request):
        """Test the abc property of Poscar"""
        sample_poscar = request.getfixturevalue(sample_poscar)
        expected_abc = {
            "a": 4.3591197305808720,
            "b": 4.3591197305808720,
            "c": 4.3591197305808720,
        }
        assert sample_poscar.abc == pytest.approx(expected_abc, abs=FLOAT_TOL), (
            "The abc property did not match expected value"
        )

    @pytest.mark.parametrize(
        "sample_poscar", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_volume_property(self, sample_poscar, request):
        """Test the volume property of Poscar"""
        sample_poscar = request.getfixturevalue(sample_poscar)
        expected_volume = 82.83167463354684
        assert sample_poscar.volume == pytest.approx(expected_volume, abs=FLOAT_TOL), (
            "The volume property did not match expected value"
        )

    def test_eq(
        self,
        sample_poscar_file_frac,
        sample_poscar_frac,
        sample_poscar_cate,
        sample_posdata,
    ):
        """Test the __eq__ method of Poscar."""
        poscar_different = Poscar(sample_posdata)
        poscar_copy = Poscar.from_file(sample_poscar_file_frac)
        assert poscar_copy == sample_poscar_frac, (
            "The copied Poscar did not match the original"
        )
        assert poscar_different != sample_poscar_frac, (
            "The different Poscar should not match"
        )
        assert sample_poscar_frac == sample_poscar_cate, (
            "The two Poscar objects should match"
        )

    def test_eq_wrong_type(self, sample_poscar_frac):
        """Test the __eq__ method with a wrong type."""
        wrong_type = "not a Poscar instance"
        assert sample_poscar_frac != wrong_type

    def test_eq_different_lattice(self, sample_poscar_frac):
        """Test the __eq__ method with different lattice."""
        posdata = copy.deepcopy(sample_poscar_frac.data)
        posdata.lattice = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        poscar_with_different_lattice = Poscar(posdata)
        assert sample_poscar_frac != poscar_with_different_lattice, (
            "The Poscar with different lattice should not match"
        )

    def test_eq_different_atoms(self, sample_poscar_frac):
        """Test the __eq__ method with different atoms."""
        posdata = copy.deepcopy(sample_poscar_frac.data)
        posdata.atoms = np.array(["Si"] * 5 + ["C"] * 3)
        poscar_with_different_atoms = Poscar(posdata)
        assert sample_poscar_frac != poscar_with_different_atoms, (
            "The Poscar with different atoms should not match"
        )

    def test_eq_with_displacement(self, sample_poscar_frac):
        """Test the __eq__ method with displacement"""
        posdata = copy.deepcopy(sample_poscar_frac.data)
        posdata.frac[0] += np.array([0.01, 0.01, 0.01])
        poscar_with_displacement = Poscar(posdata)
        assert sample_poscar_frac != poscar_with_displacement, (
            "The Poscar with displacement should not match"
        )

    def test_calculate_lattice(self):
        """Test the calculate_lattice static method."""
        coe = 2.0
        lat = np.eye(3)
        expected_lattice = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])
        calculated_lattice = Poscar.calculate_lattice(coe, lat)
        assert calculated_lattice == pytest.approx(expected_lattice, abs=FLOAT_TOL), (
            "The calculated lattice did not match expected value"
        )

    def test_calculate_volume(self):
        """Test the calculate_volume static method."""
        lattice = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])
        expected_volume = 8.0
        calculated_volume = Poscar.calculate_volume(lattice)
        assert np.isclose(calculated_volume, expected_volume, atol=FLOAT_TOL)

    def test_wrap_frac(self, sample_poscar_frac):
        """Test wrap_frac method."""
        coords = np.array(
            [
                [0.22, 0.88, 0.33],
                [-0.22, -0.33, -0.44],
                [-2.08, 0.33, 0.44],
                [1.44, 1.55, 1.66],
                [10.35, 20.44, -30.56],
                [-1.0, 0.0, 2.0],
            ]
        )
        wrapped = sample_poscar_frac.wrap_frac(coords)
        expected = np.array(
            [
                [0.22, 0.88, 0.33],
                [0.78, 0.67, 0.56],
                [0.92, 0.33, 0.44],
                [0.44, 0.55, 0.66],
                [0.35, 0.44, 0.44],
                [0.0, 0.0, 0.0],
            ]
        )
        assert np.allclose(wrapped, expected, atol=FLOAT_TOL), (
            "The wrapped coordinates did not match expected values"
        )

    @pytest.mark.parametrize(
        "coords, distances_expected",
        [
            (
                np.array([0, 0, 0]),
                np.array(
                    [
                        1.8875542,
                        1.8875542,
                        1.8875542,
                        1.8875542,
                        0.0,
                        3.0823631,
                        3.0823631,
                        3.0823631,
                    ]
                ),
            ),
            (
                4,
                np.array(
                    [
                        1.8875542,
                        1.8875542,
                        1.8875542,
                        1.8875542,
                        0.0,
                        3.0823631,
                        3.0823631,
                        3.0823631,
                    ]
                ),
            ),
        ],
    )
    def test_distance(self, sample_poscar_frac, coords, distances_expected):
        """Test distance method."""
        distances = sample_poscar_frac.distance(target=coords)
        assert np.allclose(distances, distances_expected, atol=FLOAT_TOL), (
            """The distance calculation get wrong results"""
        )

    @pytest.mark.parametrize(
        "lattice, corners_expected",
        [
            (
                None,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 4.3591197305808720],
                        [0.0, 4.3591197305808720, 0.0],
                        [0.0, 4.3591197305808720, 4.3591197305808720],
                        [4.3591197305808720, 0.0, 0.0],
                        [4.3591197305808720, 0.0, 4.3591197305808720],
                        [4.3591197305808720, 4.3591197305808720, 0.0],
                        [4.3591197305808720, 4.3591197305808720, 4.3591197305808720],
                    ]
                ),
            ),
            (
                np.array([[1, 1, 0], [0.5, 1, 0.5], [0.3, 0.4, 2]]),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.3, 0.4, 2],
                        [0.5, 1, 0.5],
                        [0.8, 1.4, 2.5],
                        [1, 1, 0],
                        [1.3, 1.4, 2],
                        [1.5, 2, 0.5],
                        [1.8, 2.4, 2.5],
                    ]
                ),
            ),
        ],
    )
    def test_get_corner(self, sample_poscar_frac, lattice, corners_expected):
        """Test get_corner method"""
        corners = sample_poscar_frac.get_corner(lattice)
        assert np.allclose(corners, corners_expected, atol=FLOAT_TOL), (
            "The calculated corners is wrong"
        )

    def test_frac2cate(self, sample_poscar_frac):
        """Test frac2cate method."""
        cate_coords_expected = np.array(
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
        )
        cate_coords = sample_poscar_frac.frac2cate(sample_poscar_frac.coor_frac)
        assert np.allclose(cate_coords, cate_coords_expected, atol=FLOAT_TOL), (
            "The cartesian coordinates did not match expected values"
        )

    def test_cate2frac(self, sample_poscar_cate):
        """Test cate2frac method"""
        frac_coords_expected = np.array(
            [
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        frac_coords = sample_poscar_cate.cate2frac(sample_poscar_cate.coor_cate)
        assert np.allclose(frac_coords, frac_coords_expected, atol=FLOAT_TOL), (
            "The fractional coordinates did not match expected values"
        )

    def test_random_disp_cate(self, sample_poscar_frac):
        """Test random_disp the cate branch"""
        magnitude = 1.3
        cate_disp = sample_poscar_frac.random_disp(magnitude=magnitude, method="cate")
        assert cate_disp.shape == sample_poscar_frac.coor_cate.shape, (
            "The shape of displacement does not match the coordinates"
        )
        assert np.all(np.abs(cate_disp) <= magnitude), (
            "The random displacement exceeds the specified magnitude"
        )

    def test_random_disp_sphere(self, sample_poscar_frac):
        """Test random_disp the sphere branch"""
        magnitude = 1.3
        sphere_disp = sample_poscar_frac.random_disp(
            magnitude=magnitude, method="sphere"
        )
        assert sphere_disp.shape == sample_poscar_frac.coor_cate.shape, (
            "The shape of the displacement does not match the coordinates"
        )
        assert np.all(np.abs(sphere_disp) <= magnitude), (
            "The random displacement exceeds the specified magnitude"
        )

    def test_random_disp_wrong_method(self, sample_poscar_frac):
        """Test random_disp with wrong method supplied."""
        method = "fly"
        with pytest.raises(
            ValueError, match=f"Unsupported method '{method}'. Use 'cate' or 'sphere'."
        ):
            sample_poscar_frac.random_disp(method=method)

    @pytest.mark.parametrize(
        "coords, expected_output",
        [
            (np.array([0.0, 0.0, 0.0]), (True, 4)),
            (np.array([1.0, 1.0, 1.0]), (False, 0)),
            (np.array([0.3, 0.3, 0.3]), (False, 0)),
            (np.array([0.0, -0.5, 2.5]), (False, 0)),
            (np.array([0.5, 0.0, 0.5]), (True, 6)),
        ],
    )
    def test_check_coor(self, sample_poscar_frac, coords, expected_output):
        """Test check_coor method."""
        is_in_cell, index = sample_poscar_frac.check_coor(coords)
        assert (is_in_cell, index) == expected_output, (
            "The check_coor method did not return expected output"
        )

    @pytest.mark.parametrize(
        "poscar_tobe_written", ["sample_poscar_frac", "sample_poscar_cate"]
    )
    def test_write_poscar(self, poscar_tobe_written, tmp_path, request):
        """Test write_poscar method"""
        poscar_tobe_written = request.getfixturevalue(poscar_tobe_written)
        file_name = "POSCAR_test"
        poscar_tobe_written.write_poscar(directory=tmp_path.as_posix(), name=file_name)
        assert (tmp_path / file_name).exists(), "The output file was not created"

        poscar_reload = Poscar.from_file((tmp_path / file_name).as_posix())
        assert poscar_reload == poscar_tobe_written, (
            """The written POSCAR did not match the original"""
        )

    @pytest.mark.parametrize("atom_index", [-1, 8])
    def test_move2center_error(self, sample_poscar_frac, atom_index):
        """Test move2center method with the error case"""
        with pytest.raises(
            IndexError, match=rf"Atom index {atom_index} out of range \(0-7\)"
        ):
            sample_poscar_frac.move2center(target=atom_index)

    def test_move2center_index(self, sample_poscar_frac):
        """Test move2center method with index provided"""
        target_index = 2
        coords_moved_expected = np.array(
            [
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ]
        )
        coords_moved = sample_poscar_frac.move2center(target=target_index)
        assert np.allclose(coords_moved, coords_moved_expected, atol=FLOAT_TOL), (
            "The coordinates after moving to center did not match expected values"
        )

    def test_move2center_coords(self, sample_poscar_frac):
        """Test move2center method with coordinates provided"""
        target_coords = np.array([0.75, 0.75, 0.75])
        coords_moved_expected = np.array(
            [
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.75, 0.75, 0.75],
                [0.75, 0.25, 0.25],
                [0.25, 0.75, 0.25],
                [0.25, 0.25, 0.75],
            ]
        )
        coords_moved = sample_poscar_frac.move2center(target=target_coords)
        assert np.allclose(coords_moved, coords_moved_expected, atol=FLOAT_TOL), (
            "The coordinates after moving to center did not match expected values"
        )
