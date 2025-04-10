"""some dataclass for vaspin"""

from dataclasses import dataclass, field
from typing import ClassVar, Sequence, Tuple

import numpy as np

from vaspin.types.array import FloatArray, IntArray, StrArray


@dataclass
class StrainTensor:
    """dataclass for 3*3 strain tensor"""

    xx: float = 0.0
    yy: float = 0.0
    zz: float = 0.0
    xy: float = 0.0
    yz: float = 0.0
    xz: float = 0.0

    _COMPONENTS: ClassVar[Tuple[str, ...]] = ("xx", "yy", "zz", "xy", "yz", "xz")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "StrainTensor":
        """Create a StrainTensor from a sequence (list, tuple, or NumPy array) of values."""
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError("`values` must be a list, tuple, or numpy array.")

        if len(values) > len(cls._COMPONENTS):
            raise ValueError("Too many values provided for StrainTensor.")

        if not all(isinstance(v, (int, float)) for v in values):
            raise TypeError("All values must be numeric (int or float).")

        kwargs = {comp: val for comp, val in zip(cls._COMPONENTS, values)}
        return cls(**kwargs)

    def get_matrix_sym(self) -> FloatArray:
        """Return the full symmetric strain tensor"""
        return np.array(
            [
                [self.xx, self.xy, self.xz],
                [self.xy, self.yy, self.yz],
                [self.xz, self.yz, self.zz],
            ]
        )

    def get_matrix_unsym(self) -> FloatArray:
        """Return the asymmetric deformation matrix for unidirectional strain

        only for appling to lattice purpose.
        """
        return np.array(
            [
                [self.xx, 2 * self.xy, 2 * self.xz],
                [0, self.yy, 2 * self.yz],
                [0, 0, self.zz],
            ]
        )


@dataclass
class PosData:
    """the data structure of POSCAR class"""

    # we would only support direct coordinates

    lattice: FloatArray
    species: StrArray
    number: IntArray
    frac: FloatArray

    coe: float = 1.0
    comment: str = "Generate by vaspin"

    atoms: StrArray = field(init=False)
    volume: float = field(init=False)
    cate: FloatArray = field(init=False)
    abc: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization hook"""
        self.atoms = np.repeat(self.species, self.number)
        self.volume = np.linalg.det(self.lattice)
        self.cate = np.dot(self.frac, self.lattice)
        self.abc = {
            "a": float(np.linalg.norm(self.lattice[0])),
            "b": float(np.linalg.norm(self.lattice[1])),
            "c": float(np.linalg.norm(self.lattice[2])),
        }
