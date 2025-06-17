"""This script is to get force and energy from OUTCAR file"""

import argparse

import numpy as np
from uniplot import plot

from vaspin import VaspOutcarParser
from vaspin.types.array import FloatArray

FORCE_HANDLER = ["N ions", "N electrons", "Energy", "Forces"]


def get_max_force(forces: FloatArray) -> FloatArray:
    """Get the maximum force of each step from the list of forces."""
    max_forces_steps = []
    for forcestep in forces:
        max_force = np.max(np.abs(np.linalg.norm(forcestep, axis=1)))
        max_forces_steps.append(max_force)

    return np.array(max_forces_steps)


def plot_data(energy_data, force_data):
    """Plot the energy and forces in shell."""
    lines = True
    character = "ascii"

    plot(energy_data, lines=lines, character_set=character, title="Energy")
    print("\n")
    plot(force_data, lines=lines, character_set=character, title="Forces")
    print("\n")
    if (ionsteps := len(force_data)) > 5:
        plot(
            energy_data[-5:],
            np.arange(ionsteps + 1)[-5:],
            lines=lines,
            character_set=character,
            title="Energy In The Last 5 Steps",
        )
        print("\n")
        plot(
            force_data[-5:],
            np.arange(ionsteps + 1)[-5:],
            lines=lines,
            character_set=character,
            title="Forces In The Last 5 Steps",
        )


def main():
    """Main function to parser force and energy from OUTCAR file."""
    parser = argparse.ArgumentParser(
        description="Parse forces and energy from OUTCAR file."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="Directory to parse data from.",
    )

    args = parser.parse_args()
    directory = args.directory
    outcar_path = f"{directory}/OUTCAR"

    outcar_parser = VaspOutcarParser(outcar_path)
    outcar_parser.set_handlers(FORCE_HANDLER)

    outcar_parser.parse(verbose=False)

    forces = np.array(outcar_parser.force)
    energy = np.array(outcar_parser.energy)

    if forces is not None:
        max_forces = get_max_force(forces)
    else:
        raise ValueError("No forces found in OUTCAR file.")

    plot_data(energy, max_forces)


if __name__ == "__main__":
    main()
