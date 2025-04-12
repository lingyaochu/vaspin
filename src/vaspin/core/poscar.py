# -*- coding: utf-8 -*-
"""Poscar Module

Contains functionality for handling VASP POSCAR files and structure manipulation.
"""

import numpy as np
from numpy import float64

import os

from itertools import product

from vaspin.types.array import FloatArray, StrArray, IntArray
from vaspin.utils import MASS_DICT, PosData, StrainTensor, wrap_frac
from vaspin.core.io import read_poscar, write_poscar


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

    def get_corner(self, lattice: FloatArray | None = None) -> FloatArray:
        """Get the cartesian coordinates of eight corners of the cell

        Args:
            lattice: Lattice vectors

        Returns:
            The eight corners of the cell
        """
        if lattice is None:
            lattice = self.lattice

        ranges = [range(2) for _ in range(3)]
        corners = np.array(list(product(*ranges)))
        return corners @ lattice

    def _prepare_transformation_matrix(self, transmat: IntArray) -> IntArray:
        """Prepare the transformation matrix for supercell generation

        Args:
            transmat: the Transformation matrix

        Returns:
            transmat: the identified transformation matrix
        """
        assert transmat.size == 9 or transmat.size == 3, (
            "transmat should have either 3 or 9 values"
        )
        if transmat.size == 9 and transmat.shape != (3, 3):
            print("the transmat is not a 3*3 matrix, reshape it.")
            transmat = transmat.reshape(3, 3)
        if transmat.size == 3:
            transmat = np.diag(transmat)
        return transmat

    def _generate_candidate_atoms_and_coor(
        self, lattice_sc: FloatArray
    ) -> tuple[StrArray, FloatArray]:
        """Generate candidate atoms and coordinates for the supercell

        firstly, we find the corners of the supercell,
        and then we find the range that primitive cell need to expand to add atoms,
        the returned atoms and coordinates is larger than the supercell size for sure.

        Args:
            lattice_sc: the supercell lattice vectors

        Returns:
            candidates_species: the candidate species
            candidates_coor_frac_sc: the candidate fractional coordinates in supercell
        """
        sc_corners = self.get_corner(lattice_sc)
        sc_corners_frac = self.cate2frac(sc_corners)

        min_multiplier = np.floor(np.min(sc_corners_frac, axis=0)).astype(int)
        max_multiplier = np.ceil(np.max(sc_corners_frac, axis=0)).astype(int)
        multiply_range = [range(min_multiplier[i], max_multiplier[i]) for i in range(3)]

        prim_coor_frac = self.coor_frac
        prim_species = self.species

        candidates_coor_frac_prim = []
        candidates_species = []

        for i, j, k in product(*multiply_range):
            translation_vec = np.array([i, j, k])
            translated_coor = prim_coor_frac + translation_vec
            candidates_coor_frac_prim.extend(translated_coor)
            candidates_species.extend(prim_species)

        candidates_coor_frac_prim = np.array(candidates_coor_frac_prim)
        candidates_species = np.array(candidates_species)

        candidates_coor_cate = self.frac2cate(candidates_coor_frac_prim)

        candidates_coor_frac_sc = self.cate2frac(candidates_coor_cate, lattice_sc)
        return candidates_species, candidates_coor_frac_sc

    def _filter_outbound_atoms(
        self,
        candidates_species: StrArray,
        candidates_coor_frac_sc: FloatArray,
        position_tolerence: float = 1e-5,
    ) -> tuple[StrArray, FloatArray]:
        """Filter out the atoms that are out of the supercell range

        Args:
            candidates_species: the candidate species to be filtered
            candidates_coor_frac_sc: the candidate fractional coordinates to be filtered
            position_tolerence: the tolerance for the position, default is 1e-5

        Returns:
            species_final: the filtered species
            coor_frac_final: the filtered fractional coordinates
        """
        # dealing boundary atoms
        mapped_coor = candidates_coor_frac_sc % 1.0
        mapped_coor[np.abs(mapped_coor - 1.0) < position_tolerence] = 0.0
        mapped_coor[np.abs(mapped_coor) < position_tolerence] = 0.0

        # round the float decimals
        decimals = int(-np.log10(position_tolerence)) + 1
        rounded_coor = np.round(mapped_coor, decimals=decimals)
        rounded_coor[np.abs(rounded_coor - 1.0) < 10 ** (-decimals)] = 0.0

        sort_indices = np.lexsort(
            (
                rounded_coor[:, 2],
                rounded_coor[:, 1],
                rounded_coor[:, 0],
                candidates_species,
            )
        )

        unique_indices_list = []

        last_unique_idx = sort_indices[0]
        unique_indices_list.append(last_unique_idx)

        last_species = candidates_species[last_unique_idx]
        last_rounded_coor = rounded_coor[last_unique_idx]

        for i in range(1, len(sort_indices)):
            current_idx = sort_indices[i]
            current_species = candidates_species[current_idx]
            current_rounded_coor = rounded_coor[current_idx]

            is_different = False
            if current_species != last_species:
                is_different = True
            elif not np.all(
                np.abs(current_rounded_coor - last_rounded_coor) < 10 ** (-decimals)
            ):
                is_different = True

            if is_different:
                unique_indices_list.append(current_idx)
                last_unique_idx = current_idx
                last_species = current_species
                last_rounded_coor = current_rounded_coor

        unique_indices = np.array(unique_indices_list, dtype=int)
        species_final = candidates_species[unique_indices]
        coor_frac_final = mapped_coor[unique_indices]

        return species_final, coor_frac_final

    def build_sc(self, transmat: IntArray) -> "Poscar":
        """Build supercell according to the transformation matrix

        mat:
            [[t11, t12, t13],
            [t21, t22, t23],
            [t31, t32, t33]]

        transform:
        a' = t11 * a + t12 * b + t13 * c
        b' = t21 * a + t22 * b + t23 * c
        c' = t31 * a + t32 * b + t33 * c

        Args:
            transmat: the transformation matrix, which is the transpose of the VESTA convention
        """
        transmat = self._prepare_transformation_matrix(transmat)

        ncells = int(np.linalg.det(transmat))
        if ncells < 0:
            print("It is recommended to keep the chirality unchanged.")

        lattice_sc = transmat @ self.lattice

        candidates_species, candidates_coor_frac_sc = (
            self._generate_candidate_atoms_and_coor(lattice_sc)
        )
        species_final, coor_frac_final = self._filter_outbound_atoms(
            candidates_species, candidates_coor_frac_sc
        )

        # check if the number of atoms is correct
        expected_natoms = len(self.coor_frac) * abs(ncells)
        if len(coor_frac_final) != expected_natoms:
            raise ValueError(
                f"the number of atoms is wrong, expect {expected_natoms}, but get {len(coor_frac_final)}"
            )

        # build new Poscar
        self.write_poscar(lattice_sc, species_final, coor_frac_final, name="POSCAR_tmp")
        poscar_sc = Poscar("POSCAR_tmp")
        os.remove("POSCAR_tmp")
        return poscar_sc
