"""Test about strain tensor."""

import numpy as np

from vaspin.utils import StrainTensor

LATTICE = {"lattice1": np.array([[10, 0, 0], [0, 20, 0], [0, 0, 40]])}


strain = StrainTensor(xy=0.1)


def apply_strain(lattice: np.ndarray, strain: StrainTensor) -> np.ndarray:
    """Apply strain to a lattice."""
    strain_matrix = strain.get_matrix_unsym()
    return lattice @ strain_matrix.T + lattice


if __name__ == "__main__":
    print(apply_strain(LATTICE["lattice1"], strain))
