"""cli for calculating the eFNV correction for charged defects in VASP calculations."""

import argparse

import numpy as np

from vaspin import Poscar, VaspOutcarParser
from vaspin.types.array import FloatArray


def epsilon(outcar: str) -> FloatArray:
    """Get the dielectric tensor from OUTCAR file

    Args:
        outcar: Path to the OUTCAR file contains the dielectric tensor.
    """
    epsilon_handlers = ["Dielectric ele", "Dielectric ion"]
    outcarparser = VaspOutcarParser(outcar)
    outcarparser.set_handlers(epsilon_handlers)
    outcarparser.parse()
    epsilon_ele = outcarparser.data["dielectric_ele"]
    epsilon_ion = outcarparser.data["dielectric_ion"]

    return np.array(epsilon_ele) + np.array(epsilon_ion)


# for now, the defect site should be manually specified, but it won't be long.
def efnv_correction(
    defect_cell: Poscar, defect_site: FloatArray, epsilon: FloatArray, charge: int
):
    """Do a eFNV correction for charged defects in VASP calculations."""


def main():
    """Main function to calculate the eFNV correction."""
    parser = argparse.ArgumentParser(
        description="Calculate the eFNV correction for charged defects."
    )

    parser.add_argument(
        "-dp",
        "--defectposcar",
        type=str,
        help="Path to the POSCAR file containing the defect cell",
    )
    parser.add_argument(
        "-sp",
        "--supercellposcar",
        type=str,
        help="Path to the POSCAR file containing the perfect supercell",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=str,
        help="Path to the OUTCAR file containing the dielectric tensor",
    )
