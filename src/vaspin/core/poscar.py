# -*- coding: utf-8 -*-
"""Poscar Module

Contains functionality for handling VASP POSCAR files and structure manipulation.
"""

import numpy as np
from numpy import float64

from ..types.array import FloatArray, StrArray
from ..utils import MASS_DICT, PosData, StrainTensor, wrap_frac
from .io import read_poscar, write_poscar


class Poscar:
    """A class for handling POSCAR files in VASP"""

    def __init__(self, posfile: str):
        """Initialize a Poscar object

        Args:
            posfile: Path to POSCAR file or JSON file

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Invalid file format
        """
        __data = read_poscar(posfile)

        # Calculate lattice vectors using static method to avoid naming conflicts
        __lattice = Poscar.calculate_lattice(__data["coe"], __data["lattice"])

        __abc = {
            "a": np.linalg.norm(__lattice[0]),
            "b": np.linalg.norm(__lattice[1]),
            "c": np.linalg.norm(__lattice[2]),
        }
        # Calculate volume using static method to avoid naming conflicts
        __volume = Poscar.calculate_volume(__lattice)

        __atom = np.array(__data["species"], dtype=str)
        __number = np.array(__data["number"], dtype=int)

        __species = np.repeat(__atom, __number)

        # Use list comprehension instead of generator expression to ensure correct array creation
        __mass = np.array([MASS_DICT[i] for i in __species], dtype=float)

        if __data["coortype"] == "Direct":
            __coor_frac = np.array(__data["coordinate"])
            __coor_cate = self.frac2cate(__coor_frac, __lattice)

        elif __data["coortype"] == "Cartesian":
            __coor_cate = np.array(__data["coordinate"])
            __coor_frac = self.cate2frac(__coor_cate, __lattice)

        else:
            raise ValueError(
                f"Unsupported coordinate type: {__data['coortype']}, only 'Direct' or 'Cartesian' are supported"
            )

        self.data = PosData(
            coe=__data["coe"],
            lattice=__lattice,
            species=__species,
            atom=__atom,
            number=__number,
            frac=__coor_frac,
            cate=__coor_cate,
            volume=__volume,
            mass=__mass,
            abc=__abc,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two Poscar objects for equality"""
        if not isinstance(other, Poscar):
            return NotImplemented
        return self.data == other.data

    @property
    def lattice(self) -> FloatArray:
        """Get the lattice"""
        return self.data.lattice

    @property
    def coor_frac(self) -> FloatArray:
        """Get the fractional coordinates"""
        return self.data.frac

    @property
    def coor_cate(self) -> FloatArray:
        """Get the cartesian coordinates"""
        return self.data.cate

    @property
    def species(self) -> StrArray:
        """Get the atomic species"""
        return self.data.species

    @property
    def abc(self) -> dict[str, float64]:
        """Get the lattice length"""
        return self.data.abc

    @staticmethod
    def calculate_lattice(coe: float, lat: list[list[float]]) -> FloatArray:
        """Calculate lattice vectors based on input coefficient and lattice

        Args:
            coe: Lattice constant
            lat: Lattice vectors

        Returns:
            Lattice vectors
        """
        return np.array(lat) * coe

    @staticmethod
    def calculate_volume(lattice: FloatArray) -> np.float64:
        """Calculate unit cell volume

        Args:
            lattice: Lattice vectors

        Returns:
            Unit cell volume
        """
        return np.linalg.det(lattice)

    def frac2cate(
        self, coor_frac: FloatArray, lattice: None | FloatArray = None
    ) -> FloatArray:
        """Convert fractional coordinates to Cartesian coordinates

        Args:
            coor_frac: Fractional coordinates
            lattice: Lattice vectors

        Returns:
            Cartesian coordinates
        """
        if lattice is None:
            lattice = self.lattice
        return np.dot(coor_frac, lattice)

    def cate2frac(
        self, coor_cate: FloatArray, lattice: None | FloatArray = None
    ) -> FloatArray:
        """Convert Cartesian coordinates to fractional coordinates

        Args:
            coor_cate: Cartesian coordinates
            lattice: Lattice Vectors

        Returns:
            Fractional coordinates
        """
        if lattice is None:
            lattice = self.lattice
        return np.dot(coor_cate, np.linalg.inv(lattice))

    def random_disp(self, magnitude: float = 0.1, method: str = "cate") -> FloatArray:
        """Generate random displacement

        Args:
            magnitude: Displacement magnitude
            method: Displacement method
                - 'cate': Uniform random displacement in Cartesian coordinates
                - 'sphere': Uniform random displacement on a sphere surface, ensuring displacement magnitude is within [0, magnitude]

        Returns:
            Random displacement array with same shape as atomic coordinates

        Raises:
            ValueError: When an unsupported method is provided
        """
        assert magnitude >= 0, "Displacement magnitude must be non-negative"
        # Get number of atoms
        n_atoms = self.coor_cate.shape[0]

        if method == "cate":
            # Generate uniform random displacement in Cartesian coordinates
            return np.random.uniform(-magnitude, magnitude, self.coor_cate.shape)

        elif method == "sphere":
            phi = np.random.uniform(0, 2 * np.pi, size=n_atoms)
            cos_theta = np.random.uniform(-1, 1, size=n_atoms)
            theta = np.arccos(cos_theta)

            r = np.power(np.random.uniform(0, magnitude**3, size=n_atoms), 1 / 3)

            sin_theta = np.sin(theta)
            disp_random = np.zeros((n_atoms, 3))
            disp_random[:, 0] = r * sin_theta * np.cos(phi)
            disp_random[:, 1] = r * sin_theta * np.sin(phi)
            disp_random[:, 2] = r * cos_theta

            return disp_random
        else:
            raise ValueError(
                f"Unsupported method: {method}. Use either 'cate' or 'sphere'."
            )

    def write_poscar(
        self,
        lattice: FloatArray | None = None,
        species: StrArray | None = None,
        coor_frac: FloatArray | None = None,
        directory: str = ".",
        comment: str = "generated by mother python",
        name: str = "POSCAR",
    ) -> None:
        """Write data to POSCAR file

        Args:
            lattice: Lattice data, defaults to object's own data
            species: Atomic species data, defaults to object's own data
            coor_frac: Fractional coordinate data, defaults to object's own data
            directory: Directory to write file to, defaults to current directory
            comment: Description in first line of POSCAR file
            name: Name of POSCAR file, defaults to POSCAR

        Returns:
            None

        Examples:
            >>> self.write_poscar()

        """
        if lattice is None:
            lattice = self.lattice

        if coor_frac is None:
            coor_frac = self.coor_frac

        if species is None:
            species = self.species

        sort = np.argsort(species, kind="stable")
        coor_frac = coor_frac[sort]
        species = species[sort]

        write_poscar(lattice, species, coor_frac, directory, comment, name)

        if not np.array_equal(self.species, species):
            print(
                "Warning: The order of the elements in the POSCAR file may be changed."
            )
        return None

    def wrap_frac(self, coor_frac: FloatArray) -> FloatArray:
        """Convert fractional coordinates to be within 0-1 range

        Args:
            coor_frac: Fractional coordinates

        Returns:
            Fractional coordinates within 0-1 range
        """
        return wrap_frac(coor_frac)

    def move2center(self, target: int | FloatArray) -> FloatArray:
        """Move target to the center of the cell and change all the atoms in the cell accordingly.

        Args:
            target: Either atom index or fractional coordinates to be centered

        Returns:
            New fractional coordinates with target at center

        Raises:
            ValueError: If the atom index is out of range
        """
        if isinstance(target, int):
            if target < 0 or target >= len(self.coor_frac):
                raise ValueError(
                    f"Atom index {target} out of range (0-{len(self.coor_frac) - 1})"
                )
            center_coor = self.coor_frac[target]
        else:
            center_coor = target

        return self.wrap_frac(self.coor_frac - center_coor + np.array([0.5, 0.5, 0.5]))

    def distance(
        self,
        target: int | FloatArray,
    ) -> FloatArray:
        """Calculate distances between a target atom or coordinate and other atoms in the structure

        Args:
            target: Either index of target atom in structure or specific fractional coordinate

        Returns:
            distance_list: Array of distances between target and all atoms, in the same order as atoms
        """
        coor_frac_moved = self.move2center(target)
        cell_center = np.array([0.5, 0.5, 0.5])

        # Calculate distances
        distance_arrow = self.frac2cate(coor_frac_moved - cell_center)
        distance_list = np.linalg.norm(distance_arrow, axis=1)

        return distance_list

    def lattice_rotation(self, matrix_rot: FloatArray) -> FloatArray:
        """Rotate lattice vectors

        Args:
            matrix_rot: Standard rotation matrix in 3D space

        Returns:
            New rotated lattice vector matrix
        """
        lattice = self.lattice
        if matrix_rot.size != 9:
            raise ValueError("Rotation matrix must be 3x3")
        matrix_rot = matrix_rot.reshape(3, 3)

        det = np.linalg.det(matrix_rot)
        if not np.isclose(abs(det), 1.0):
            raise ValueError(
                f"Invalid rotation matrix, determinant should be ±1, but got {det}"
            )

        lattice_rotated = lattice @ matrix_rot.T

        return lattice_rotated

    def apply_strain(self, strain: StrainTensor) -> FloatArray:
        """Apply strain to lattice vectors

        Args:
            strain: Strain tensor object

        Returns:
            New strained lattice vector matrix
        """
        strain_matrix = strain.get_matrix_unsym()

        identity = np.eye(3)
        deform_matrix = identity + strain_matrix

        lattice_strained = self.lattice @ deform_matrix.T

        return lattice_strained
