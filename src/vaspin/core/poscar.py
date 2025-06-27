# -*- coding: utf-8 -*-
"""Poscar Module

Contains functionality for handling VASP POSCAR files and structure manipulation.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import List, Literal, Self, Tuple

import numpy as np

from vaspin.core.io import read_poscar, write_poscar
from vaspin.types.array import FloatArray, IntArray, StrArray
from vaspin.utils import PosData, StrainTensor, wrap_frac


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
        if not isinstance(posdata, PosData):
            raise ValueError("posdata must be a PosData object")

        self.data = posdata

    @classmethod
    def from_file(cls, filepath: str) -> Self:
        """Create a Poscar object from a json or POSCAR file

        Args:
            filepath: Path to json or POSCAR file

        Returns:
            Poscar object
        """
        if not isinstance(filepath, str):
            raise ValueError("filepath must be a string")

        __data = read_poscar(filepath)
        __data["lattice"] = np.array(__data["lattice"])
        __data["species"] = np.array(__data["species"])
        __data["number"] = np.array(__data["number"])

        if __data["coortype"][0] in ["c", "C", "K", "k"]:
            __data["frac"] = np.dot(
                np.array(__data["coordinate"]), np.linalg.inv(__data["lattice"])
            )

        else:
            __data["frac"] = np.array(__data["coordinate"])

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

    @property
    def volume(self) -> np.float64:
        """Get the unit cell volume"""
        return self.calculate_volume(self.lattice)

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
                - 'sphere': Uniform random displacement on a sphere surface
                    ensuring displacement magnitude is within [0, magnitude]

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
        """Generate random displacement on a sphere surface

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
        """Move target to the center of the cell

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
        """Calculate distances between the target and other atoms in the structure

        Args:
            target: Either index of an atom in structure or a fractional coordinate

        Returns:
            distance_list: Array of distances between target and all atoms
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
            candidates_atoms: the candidate atoms
            candidates_coor_frac_sc: the candidate fractional coordinates in supercell
        """
        sc_corners = self.get_corner(lattice_sc)
        sc_corners_frac = self.cate2frac(sc_corners)

        min_multiplier = np.floor(np.min(sc_corners_frac, axis=0)).astype(int)
        max_multiplier = np.ceil(np.max(sc_corners_frac, axis=0)).astype(int)
        multiply_range = [range(min_multiplier[i], max_multiplier[i]) for i in range(3)]

        prim_coor_frac = self.coor_frac
        prim_atoms = self.atoms

        candidates_coor_frac_prim = []
        candidates_atoms = []

        for i, j, k in product(*multiply_range):
            translation_vec = np.array([i, j, k])
            translated_coor = prim_coor_frac + translation_vec
            candidates_coor_frac_prim.extend(translated_coor)
            candidates_atoms.extend(prim_atoms)

        candidates_coor_frac_prim = np.array(candidates_coor_frac_prim)
        candidates_atoms = np.array(candidates_atoms)

        candidates_coor_cate = self.frac2cate(candidates_coor_frac_prim)

        candidates_coor_frac_sc = self.cate2frac(candidates_coor_cate, lattice_sc)
        return candidates_atoms, candidates_coor_frac_sc

    def _filter_outbound_atoms(
        self,
        candidates_atoms: StrArray,
        candidates_coor_frac_sc: FloatArray,
        ncells: int,
        position_tolerence: float = 1e-5,
    ) -> tuple[StrArray, FloatArray]:
        """Filter out the atoms that are out of the supercell range

        Args:
            candidates_atoms: the candidate species to be filtered
            candidates_coor_frac_sc: the candidate fractional coordinates to be filtered
            ncells: the number of primitive cells in the supercell
            position_tolerence: the tolerance for the position, default is 1e-5

        Returns:
            species_final: the filtered species
            coor_frac_final: the filtered fractional coordinates
        """
        # dealing with boundary condition
        decimal = round(-np.log10(position_tolerence)) + 1
        rounded_coor = np.round(candidates_coor_frac_sc, decimals=decimal)
        wraped_coor = wrap_frac(rounded_coor)
        wraped_coor[np.abs(wraped_coor - 1.0) < position_tolerence] = 0.0
        wraped_coor[np.abs(wraped_coor) < position_tolerence] = 0.0

        # we should sort the int values to make sure the order is correct
        # no matter how we round the floats, they are still floats in the memory.
        # the lexsort will *exactly* sort the floats, and the order is not guaranteed.
        coor_tosort = np.trunc(wraped_coor * 1 / position_tolerence).astype(int)
        sort_indices = np.lexsort(
            (coor_tosort[:, 2], coor_tosort[:, 1], coor_tosort[:, 0])
        )

        sorted_coor = wraped_coor[sort_indices]
        coor_diff = np.diff(sorted_coor, axis=0)
        is_different = np.any(np.abs(coor_diff) >= position_tolerence, axis=1)
        unique_mask_sorted = np.concatenate(([True], is_different))
        unique_indices = sort_indices[unique_mask_sorted]

        expected_atoms = len(self.atoms) * abs(ncells)
        if len(unique_indices) != expected_atoms:
            raise ValueError(
                f"the number of atoms is wrong, expect {expected_atoms},"
                f" but get {len(unique_indices)}"
            )

        return candidates_atoms[unique_indices], wraped_coor[unique_indices]

    def build_sc(self, transmat: IntArray) -> PosData:
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
            transmat: the transformation matrix,
                which is the transpose of the VESTA convention
        """
        transmat = self._prepare_transformation_matrix(transmat)
        ncells = round(np.linalg.det(transmat))
        if ncells < 0:
            print("It is recommended to keep the chirality unchanged.")

        lattice_sc = transmat @ self.lattice

        candidates_atoms, candidates_coor_frac_sc = (
            self._generate_candidate_atoms_and_coor(lattice_sc)
        )
        atoms_final, coor_frac_final = self._filter_outbound_atoms(
            candidates_atoms, candidates_coor_frac_sc, ncells
        )

        # build new PosData
        species, numbers, coor_frac_sc = self._species_numbers_coor(
            atoms_final, coor_frac_final
        )
        posdata_sc = PosData(
            lattice=lattice_sc, species=species, number=numbers, frac=coor_frac_sc
        )
        return posdata_sc

    def check_coor(
        self, coor_frac: FloatArray, tol: float = 1e-3
    ) -> tuple[np.bool_, int]:
        """Check one coordinate whether in current Poscar class, return index if exist

        Args:
            coor_frac: the fractional coordinate to be checked
            tol: the absolute error tolerance

        Return:
            whether exist
            the index of the target coordinate, return 0 if not exist
        """
        exist = np.all(np.isclose(self.coor_frac, coor_frac, atol=tol), axis=1)
        return np.any(exist), int(np.argmax(exist))

    def _species_numbers_coor(
        self, atoms: StrArray | None = None, coor_frac: FloatArray | None = None
    ) -> tuple[StrArray, IntArray, FloatArray]:
        """Get species and numbers from atoms and coordinates list

        Under processing, the order of the atoms may change,
        so we would change the fractional coordinates also

        Args:
            atoms: the atom list, default to be the atoms in current Poscar
            coor_frac: the fractional coordinates, default as current Poscar.coor_frac

        Returns:
            species: the unique elements list
            numbers: the list of numbers of the unique elements
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
        exist, index = self.check_coor(coor_frac)
        if not exist:
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
            comment=f"{self.comment}  {newatom}_{self.atoms[index]} at"
            f" ({coor_sub[0]:.3f}, {coor_sub[1]:.3f}, {coor_sub[2]:.3f})",
        )

    def defect_create_vac(self, coor_frac: FloatArray) -> PosData:
        """Create a vacancy defect in the POSCAR

        Args:
            coor_frac: the coordinate to create defect
        """
        exist, index = self.check_coor(coor_frac)
        if not exist:
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
            comment=f"{self.comment}  Va_{self.atoms[index]} at "
            f"({coor_vac[0]:.3f}, {coor_vac[1]:.3f}, {coor_vac[2]:.3f})",
        )

    def max_sphere_radius(self) -> np.floating:
        """The maximum radius of a sphere that can be inscribed in the unit cell.

        Returns:
            radius: The maximum radius of the sphere.
        """
        plane_area = np.linalg.norm(np.cross(self.lattice[1], self.lattice[2]))
        for i in range(1, 3):
            area = np.linalg.norm(np.cross(self.lattice[i - 2], self.lattice[i - 1]))
            plane_area = area if area > plane_area else plane_area

        return self.volume / plane_area / 2


@dataclass
class StruMapping:
    """The mapping relation from one structure to another structure with same lattice"""

    stru_from: Poscar
    stru_to: Poscar
    dtol: float = 0.5

    # Atom mapping from stru_from to stru_to
    mapping: List[Tuple[int, int, Tuple[str, str], float]] = field(init=False)

    def __post_init__(self):
        """Get the mapping relation from one structure to another"""
        self.mapping = self.atom_relation()

    def atom_relation(self) -> List[Tuple[int, int, Tuple[str, str], float]]:
        """Project atoms from one structure to another

        Args:
            stru_from: the Poscar object of the structure to be projected
            stru_to: the Poscar object of the target structure

        Returns:
            relation: the list of indices of the target structure corresponding to
                each atom in the original structure, None if no corresponding atom found
                within the distance tolerance

            distance_site: the distance between the projected atom and the target atom
        """
        mapping = []
        for idfrom, coor in enumerate(self.stru_from.coor_frac):
            distance = self.stru_to.distance(coor)
            idto = int(np.argmin(distance))
            mapping.append(
                (
                    idfrom,
                    idto,
                    (self.stru_from.atoms[idfrom], self.stru_to.atoms[idto]),
                    distance[idto],
                )
            )

        return mapping

    def multi_mapped(self):
        """Get the atoms that are mapped to the same target atom"""
        target_mapping = defaultdict(list)

        for idfrom, idto, specie_relation, dist in self.mapping:
            target_mapping[idto].append((idfrom, idto, specie_relation, dist))

        multi_mapped = {
            idto: idfrom_list
            for idto, idfrom_list in target_mapping.items()
            if len(idfrom_list) > 1
        }
        return multi_mapped

    def un_mapped(self):
        """Get the atoms that are not mapped in the target structure"""
        mapped = {idto for _, idto, _, _ in self.mapping}
        all_idto = set(range(len(self.stru_to.atoms)))
        un_mapped = all_idto - mapped

        return un_mapped

    def different_species(self):
        """Get the map relation whose species are different"""
        different = [
            (difrom, idto, specie_relation, dist)
            for difrom, idto, specie_relation, dist in self.mapping
            if specie_relation[0] != specie_relation[1]
        ]
        return different

    @staticmethod
    def multi_cast(
        multimap: dict[int, List[Tuple[int, int, Tuple[str, str], float]]],
    ) -> List[Tuple[int, int, Tuple[str, str], float]]:
        """Cast the multi_mapped result to get the redundant atoms based on distance"""
        redundant_atoms = []
        for _idto, idfrom_list in multimap.items():
            # Sort by distance
            idfrom_list.sort(key=lambda x: x[3])
            redundant_atoms.extend(idfrom_list[1:])
        return redundant_atoms


class Defect:
    """A class for handling point defects in POSCAR files"""

    def __init__(self, poscar_sc: Poscar, poscar_de: Poscar):
        """Initialize a Defect object

        Args:
            poscar_sc: the Poscar object of the perfect cell
            poscar_de: the Poscar object of the defect cell
        """
        if not isinstance(poscar_sc, Poscar) or not isinstance(poscar_de, Poscar):
            raise ValueError("poscar_sc and poscar_de must be Poscar objects")

        self.poscar_sc = poscar_sc
        self.poscar_de = poscar_de

        self.map = StruMapping(stru_from=poscar_de, stru_to=poscar_sc)

    @property
    def defect_info(self):
        """Identify the defect type and the location in the supercell"""
        interstitial_candi = self.map.multi_mapped()
        vacancy_candi = self.map.un_mapped()
        sub_candi = self.map.different_species()

        defect_sites = []
        defect_type = []
        defect_name = []

        if len(interstitial_candi) != 0:
            redundant_atoms = StruMapping.multi_cast(interstitial_candi)
            for idfrom, idto, _specie_relation, _dist in redundant_atoms:
                defect_sites.append(self.poscar_sc.coor_frac[idto])
                defect_type.append("Interstitial")
                defect_name.append(f"{self.poscar_de.atoms[idfrom]}_i")

        if len(vacancy_candi) != 0:
            for idto in vacancy_candi:
                defect_sites.append(self.poscar_sc.coor_frac[idto])
                defect_type.append("Vacancy")
                defect_name.append("Va_" + self.poscar_sc.atoms[idto])

        if len(sub_candi) != 0:
            for _idfrom, idto, specie_relation, _dist in sub_candi:
                defect_sites.append(self.poscar_sc.coor_frac[idto])
                defect_type.append("Substitution")
                defect_name.append(f"{specie_relation[0]}_{specie_relation[1]}")

        if len(defect_sites) == 0:
            raise ValueError("No defects found in the structure")

        return {
            "defect_sites": np.array(defect_sites),
            "defect_type": np.array(defect_type),
            "defect_name": np.array(defect_name),
        }

    @property
    def defect_center(self) -> FloatArray:
        """Get the center of the defect."""
        defect_sites = self.defect_info["defect_sites"]

        if len(defect_sites) == 0:
            raise ValueError("No defect sites found.")

        return np.mean(defect_sites, axis=0)
