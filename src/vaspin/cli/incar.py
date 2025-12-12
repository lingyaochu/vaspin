"""The cli for INCAR file handling and manipulation."""

import argparse
from typing import Callable

from vaspin import Incar
from vaspin.incar.validator import IncarValidator
from vaspin.types import PathType


def validate_incar(incar_file: PathType) -> None:
    """Check the INCAR file for validity.

    Check the INCAR file for common errors and inconsistencies using IncarValidator.
    Print the results to the console for use view.

    Args:
        incar_file: Path to the INCAR file.
    """
    incar = Incar.from_file(incar_file)
    validator = IncarValidator()
    validate_result = validator.validate(incar)
    validate_result.msg_print()
    return None


def main():
    """Main function to run the INCAR validation CLI."""
    job_dict: dict[str, Callable] = {
        "vali": validate_incar,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job", type=str, choices=list(job_dict.keys()))
    parser.add_argument("-d", "--directory", type=str, default="./")

    args = parser.parse_args()
    job_dict[args.job](args.directory + "INCAR")
