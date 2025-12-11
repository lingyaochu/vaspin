"""cli for calculating the eFNV correction for charged defects in VASP calculations."""

import argparse
import json
from pathlib import Path
from pprint import pprint

import numpy as np

from vaspin import Poscar, VaspOutcarParser
from vaspin.post.efnv import Efnv
from vaspin.types.array import FloatArray

HANDLER_DIELE = ["Dielectric ele", "Dielectric ion"]
HANDLER_POT = ["Site potential"]


def get_dielectric(outcar_file: str) -> FloatArray:
    """Extract dielectric constant from OUTCAR file."""
    parser = VaspOutcarParser(outcar_file)
    parser.set_handlers(HANDLER_DIELE)
    parser.parse()

    dielectric_ele = parser.data["dielectric_ele"]
    dielectric_ion = parser.data["dielectric_ion"]

    return np.array(dielectric_ele) + np.array(dielectric_ion)


def get_site_potential(outcar_file: str) -> FloatArray:
    """Extract site potential from OUTCAR file."""
    parser = VaspOutcarParser(outcar_file)
    parser.set_handlers(HANDLER_POT)
    parser.parse()

    return np.array(parser.data["site_potential"])


def main():
    """Main function to run the eFNV correction calculation."""
    parser = argparse.ArgumentParser(
        description="Efnv correction for charged defects in VASP calculations."
    )
    parser.add_argument(
        "-sc", "--supercell", type=str, help="The directory of the perfect cell"
    )
    parser.add_argument(
        "-de", "--defect", type=str, help="The directory of the defect cell"
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=str,
        help="The directory of the OUTCAR file containing the dielectric constant",
    )
    parser.add_argument("-c", "--charge", type=int, help="The charge of the defect")

    args = parser.parse_args()

    outcar_sc_path = Path(args.supercell) / "OUTCAR"
    poscar_sc_path = Path(args.supercell) / "CONTCAR"
    outcar_de_path = Path(args.defect) / "OUTCAR"
    poscar_de_path = Path(args.defect) / "CONTCAR"
    outcar_diele_path = Path(args.epsilon) / "OUTCAR"

    poscar_sc = Poscar.from_file(poscar_sc_path.as_posix())
    poscar_de = Poscar.from_file(poscar_de_path.as_posix())
    dielectric = get_dielectric(outcar_diele_path.as_posix())
    pot_sc = get_site_potential(outcar_sc_path.as_posix())
    pot_de = get_site_potential(outcar_de_path.as_posix())

    efnv_corr = Efnv(poscar_sc, poscar_de, pot_sc, pot_de, dielectric, args.charge)

    correction = {
        "pc term": efnv_corr.correction_point_charge,
        "align term": efnv_corr.correction_potential_align,
        "total": efnv_corr.correction_total,
    }

    pprint(correction)
    with open(Path(args.defect) / "efnv.json", "w") as f:
        json.dump(correction, f, indent=4)
