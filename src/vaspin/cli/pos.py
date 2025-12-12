#!/usr/bin/python3
"""this script is to get information from poscar"""

import argparse
import json
from pathlib import Path
from typing import Literal

from vaspin import Poscar
from vaspin.io import read_poscar

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--directory", default=".", help="the directory of the POSCAR file"
)
parser.add_argument(
    "-j",
    "--job",
    default="json",
    choices=["json", "lattice", "disp"],
    help="the job need to be done",
)
parser.add_argument(
    "-l",
    "--length",
    default=0.0,
    type=float,
    help="the length of random displacement, only for job=disp",
)
parser.add_argument(
    "-m",
    "--method",
    default="cate",
    choices=["cate", "sphere"],
    help="the method to distort the structure, only for job=disp",
)
args = parser.parse_args()


def write_json() -> None:
    """Write the POSCAR file to JSON file"""
    pos_json_str = read_poscar(args.directory + "/POSCAR")
    (Path(args.directory) / "pos.json").write_text(json.dumps(pos_json_str, indent=4))


def get_lattice() -> None:
    """Get the lattice parameters from POSCAR file"""
    pos = Poscar.from_file(args.directory + "/POSCAR")
    a = "{:.7f}".format(pos.abc["a"])
    b = "{:.7f}".format(pos.abc["b"])
    c = "{:.7f}".format(pos.abc["c"])
    print(a, b, c)


def pos_disp(
    length: float = args.length, method: Literal["cate", "sphere"] = args.method
) -> None:
    """Randomly distort the structure by length"""
    pos = Poscar.from_file(args.directory + "/POSCAR")
    cate = pos.coor_cate + pos.random_disp(magnitude=length, method=method)
    pos.write_poscar(
        coor_frac=pos.cate2frac(cate), directory=args.directory, name="POSCAR_disp"
    )


def main():
    """The entry point of the script"""
    operation = {
        "json": write_json,
        "lattice": get_lattice,
        "disp": lambda: pos_disp(args.length, args.method),
    }
    if args.job in operation:
        operation[args.job]()
    else:
        print("The job is not supported")
        exit(1)
