"""The script to analyze local structure around a defect"""

import argparse

import numpy as np

from vaspin import Poscar, PosData
from vaspin.types import FloatArray


def get_defect_coor(pos, defect_site: str | int | list) -> FloatArray:
    """Get the defect coordinates from the original structure.

    Args:
        pos: the original structure contains the defect
        defect_site: the defect site, which should be a symbol or a list of coordinates

    Returns:
        defect_coor: the defect coordinates in fractional coordinates
    """
    if (
        not isinstance(defect_site, str)
        and not isinstance(defect_site, list)
        and not isinstance(defect_site, int)
    ):
        raise TypeError("defect_site should be a symbol or a list of coordinates")

    if isinstance(defect_site, str):
        defect_coor = _get_defect_coor_from_symbol(pos, defect_site)

    elif isinstance(defect_site, int):
        defect_coor = _get_defect_coor_from_index(pos, defect_site)

    elif isinstance(defect_site, list):
        if len(defect_site) != 3:
            raise ValueError("The coordinate should be a list of 3 coordinates")
        if np.any(np.array(defect_site) < 0) or np.any(np.array(defect_site) > 1):
            raise ValueError("The coordinate should be in the range of 0 to 1")

        defect_coor = np.array(defect_site)

    return defect_coor


def _get_defect_coor_from_symbol(pos: Poscar, symbol: str) -> FloatArray:
    """Get the defect coordinates from the original structure by symbol.

    Args:
        pos: the original structure contains the defect
        symbol: the defect symbol

    Returns:
        defect_coor: the defect coordinates in fractional coordinates
    """
    if symbol not in pos.atoms:
        raise ValueError(f"{symbol} is not in the structure")
    if np.sum(pos.atoms == symbol) > 1:
        raise ValueError(
            f"{symbol} is not unique in the structure, please use coordinates"
        )

    defect_coor = pos.coor_frac[pos.atoms == symbol]

    return defect_coor


def _get_defect_coor_from_index(pos: Poscar, index: int) -> FloatArray:
    """Get the defect coordinates from the original structure by index.

    Args:
        pos: the original structure contains the defect
        index: the defect index, but starts from 1

    Returns:
        defect_coor: the defect coordinates in fractional coordinates
    """
    if index < 1 or index > len(pos.atoms):
        raise ValueError(f"The index should be in the range of 1 to {len(pos.atoms)}")

    defect_coor = pos.coor_frac[index - 1]
    return defect_coor


def get_local_stru(pos: Poscar, defect_coor: FloatArray, dmax: float) -> PosData:
    """Get the local structure around a defect site.

    Args:
        pos: the original structure contains the defect
        defect_coor: the fractional coordinates of the defect site
        dmax: Additional distance beyond nearest neighbor for local cutoff

    Returns:
        local_stru: the local structure around the defect site in the same lattice
    """
    distance_list = pos.distance(defect_coor)
    max_range = np.sort(distance_list)[1] + dmax

    local_indexes = np.where(distance_list < max_range)[0]
    local_atoms = pos.atoms[local_indexes]
    local_coors = pos.coor_frac[local_indexes]

    local_species, local_numbers, local_frac = pos._species_numbers_coor(
        local_atoms, local_coors
    )

    return PosData(
        lattice=pos.lattice,
        species=local_species,
        number=local_numbers,
        frac=local_frac,
    )


def main():
    """The main logic"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, default="POSCAR", help="The structure file"
    )
    parser.add_argument(
        "-a", "--atom", help="input the dopant symbol, or index counting from 1"
    )
    parser.add_argument(
        "-p",
        "--position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="input the fractional coordinates of the defect",
    )
    parser.add_argument(
        "-d",
        "--dmax",
        type=float,
        default=0.5,
        help="Extra distance beyond nearest neighbor for local cutoff. (Default: 0.5)",
    )
    parser.add_argument(
        "-c",
        "--center",
        action="store_true",
        help="put the local structure in the center of supercell, default is False",
    )
    args = parser.parse_args()

    if args.atom and args.position:
        raise ValueError("Please input either atom or position, not both")
    if not args.atom and not args.position:
        raise ValueError("Please input either atom or position")

    defect_site = args.atom if args.atom else args.position

    pos = Poscar.from_file(args.file)
    defect_coor = get_defect_coor(pos, defect_site)
    local_posdata = get_local_stru(pos, defect_coor, args.dmax)
    local_pos = Poscar(local_posdata)

    local_coor = local_pos.coor_frac
    if args.center:
        local_coor = local_pos.move2center(defect_coor)

    local_pos.write_poscar(coor_frac=local_coor, name="POSCAR_local")
