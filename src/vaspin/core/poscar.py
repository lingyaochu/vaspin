# -*- coding: utf-8 -*-
"""Poscar Module

Contains functionality for handling VASP POSCAR files and structure manipulation.
"""

import numpy as np
from numpy import float64

from itertools import product

from typing import Literal

from vaspin.types.array import FloatArray, StrArray, IntArray
from vaspin.utils import MASS_DICT, PosData, StrainTensor, wrap_frac
from vaspin.core.io import read_poscar, write_poscar


class Poscar:
    """A class for handling POSCAR files in VASP"""

    def __init__(self, posdata: PosData):
        """Initialize a Poscar object

        Args:
            posdata: the PosData object containing the information

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Invalid file format
        """
        self.data = posdata

    @classmethod
    def from_file(cls, filepath: str) -> "Poscar":
        """Create a Poscar object from a json or POSCAR file

        Args:
            filepath: Path to json or POSCAR file

        Returns:
            Poscar object
        """
        __data = read_poscar(filepath)
        if __data["coortype"] == "Direct":
            __data["frac"] = np.array(__data["coordinate"])

        else:
            __data["frac"] = np.dot(
                np.array(__data["coordinate"]), np.linalg.inv(__data["lattice"])
            )

        # delete the "coordinate" and "coortype" key-value pairs
        __data.pop("coordinate")
        __data.pop("coortype")
        return cls(PosData(**__data))

    def __eq__(self, other: object) -> bool:
        """Compare two Poscar objects for equality"""
        if not isinstance(other, Poscar):
            return NotImplemented
        return self.data == other.data

    @property
    def comment(self) -> str:
        """Get the comment"""
        return self.data.comment

    @property
    def lattice(self) -> FloatArray:
        """Get the lattice"""
        return self.data.lattice.copy()

    @property
    def coor_frac(self) -> FloatArray:
        """Get the fractional coordinates"""
        return self.data.frac.copy()

    @property
    def coor_cate(self) -> FloatArray:
        """Get the cartesian coordinates"""
        return self.data.cate.copy()

    @property
    def atoms(self) -> StrArray:
        """Get the atom lists"""
        return self.data.atoms.copy()

    @property
    def abc(self) -> dict[str, float]:
        """Get the lattice length"""
        return self.data.abc.copy()

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

    def random_disp(
        self, magnitude: float = 0.1, method: Literal["cate", "sphere"] = "cate"
    ) -> FloatArray:
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

        if method not in ["cate", "sphere"]:
            raise ValueError(
                f"Unsupported method: {method}. Use either 'cate' or 'sphere'."
            )

        if method == "cate":
            # Generate uniform random displacement in Cartesian coordinates
            return self._random_disp_cate(magnitude)

        else:
            return self._random_disp_sphere(magnitude)

    def _random_disp_cate(self, magnitude: float) -> FloatArray:
        """Generate random displacement in Cartesian coordinates

        Args:
            magnitude: Displacement magnitude

        Returns:
            Random displacement array with same shape as atomic coordinates
        """
        return np.random.uniform(-magnitude, magnitude, self.coor_cate.shape)

    def _random_disp_sphere(self, magnitude: float) -> FloatArray:
        """Generate random displacement on a sphere surface, ensuring displacement magnitude is within [0, magnitude]

        Args:
            magnitude: Displacement magnitude

        Returns:
            Random displacement array with same shape as atomic coordinates
        """
        n_atoms = self.coor_cate.shape[0]

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

    def write_poscar(
        self,
        lattice: FloatArray | None = None,
        atoms: StrArray | None = None,
        coor_frac: FloatArray | None = None,
        directory: str = ".",
        comment: str | None = None,
        name: str = "POSCAR",
    ) -> None:
        """Write data to POSCAR file

        Args:
            lattice: Lattice data, defaults to object's own data
            atoms: the atom list data, defaults to object's own data
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

        if atoms is None:
            atoms = self.atoms

        if comment is None:
            comment = self.comment

        sort = np.argsort(atoms, kind="stable")
        coor_frac = coor_frac[sort]
        atoms = atoms[sort]

        write_poscar(lattice, atoms, coor_frac, directory, comment, name)

        if not np.array_equal(self.atoms, atoms):
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

    def check_coor(
        self, coor_frac: FloatArray, tol: float = 1e-3
    ) -> tuple[np.bool, int]:
        """Check one coordinate whether in current Poscar class, return its index if exsist

        Args:
            coor_frac: the fractional coordinate to be checked
            tol: the absolute error tolerance

        Return:
            whether exsist
            the index of the target coordinate, return 0 if not exsist
        """
        exsist = np.all(np.isclose(self.coor_frac, coor_frac, atol=tol), axis=1)
        return np.any(exsist), int(np.argmax(exsist))

    def _species_numbers_coor(
        self, atoms: StrArray | None = None, coor_frac: FloatArray | None = None
    ) -> tuple[StrArray, IntArray, FloatArray]:
        """An auxiliary function for get species and numbers from atoms list and the corresponding fractional coordinates

        Under processing, the order of the atoms may change, so we would change the fractional coordinates also, to maintain the consistancy.

        Args:
            atoms: the atom list, default to be the atoms in current Poscar
            coor_frac: the fractional coordinates, default to be the coordinates in current Poscar

        Returns:
            species: the unique elements list
            numbers: the list of numbers of the unqiue elements
            new_coor: the new fractional coordinates that obeys the new order
        """
        if atoms is None:
            atoms = self.atoms
        if coor_frac is None:
            coor_frac = self.coor_frac

        sort = np.argsort(atoms)
        new_atoms = atoms[sort]
        new_coor = coor_frac[sort]

        species, numbers = np.unique(new_atoms, return_counts=True)
        return species, numbers, new_coor

    def defect_create_sub(self, coor_frac: FloatArray, newatom: str) -> PosData:
        """Create a substitution defect in the POSCAR

        Args:
            coor_frac: the coordinate to create defect
            newatom: the substitution atom
        """
        exsist, index = self.check_coor(coor_frac)
        if not exsist:
            raise ValueError(
                "You can't create a substitution at this site, the site is empty!"
            )

        if newatom == self.atoms[index]:
            raise ValueError("The sub atom is the same as the original, please check!")

        atoms_new = self.atoms
        atoms_new[index] = newatom

        species_new, numbers_new, coor_frac_new = self._species_numbers_coor(atoms_new)

        coor_sub = self.coor_frac[index]
        return PosData(
            lattice=self.lattice,
            species=species_new,
            number=numbers_new,
            frac=coor_frac_new,
            comment=f"{self.comment}  {newatom}_{self.atoms[index]} at ({coor_sub[0]:.3f}, {coor_sub[1]:.3f}, {coor_sub[2]:.3f})",
        )

    def defect_create_vac(self, coor_frac: FloatArray) -> PosData:
        """Create a vacancy defect in the POSCAR

        Args:
            coor_frac: the coordinate to create defect
        """
        exsist, index = self.check_coor(coor_frac)
        if not exsist:
            raise ValueError(
                "You can't create a vacancy at this site, the site is empty!"
            )
        atoms_new = np.delete(self.atoms, index, axis=0)
        coor_frac_new = np.delete(self.coor_frac, index, axis=0)

        species_final, number_final, coor_frac_final = self._species_numbers_coor(
            atoms_new, coor_frac_new
        )

        coor_vac = self.coor_frac[index]
        return PosData(
            lattice=self.lattice,
            species=species_final,
            number=number_final,
            frac=coor_frac_final,
            comment=f"{self.comment}  Va_{self.atoms[index]} at ({coor_vac[0]:.3f}, {coor_vac[1]:.3f}, {coor_vac[2]:.3f})",
        )
