"""The phonon module"""

from functools import cached_property
from typing import Self, overload

import numpy as np

from vaspin.core.io import read_phonon, read_poscar
from vaspin.types.array import FloatArray, IntArray
from vaspin.utils.constants import ENERGY_UNITS
from vaspin.utils.datatype import PhoData, PosData


class Phonon:
    """The Phonon class represents phonon data from VASP calculations."""

    def __init__(self, phodata: PhoData):
        """Initialize the Phonon object with PhoData"""
        self.data = phodata
        self.freq = phodata.freq
        self.mode = phodata.mode
        self.mass = phodata.mass

    @classmethod
    def from_file(cls, phononfile: str, posfile) -> Self:
        """Create a Phonon object from an OUTCAR file

        Args:
            phononfile: Path to file storing phonon data, OUTCAR or phonon.json
            posfile: Path to POSCAR file or pos.json file

        Returns:
            Phonon object
        """
        data = read_phonon(phononfile)

        pos_data = read_poscar(posfile)
        mass = PosData(**pos_data).mass

        freq = np.array(data["frequencies"])
        mode = np.array(data["eigenmodes"])
        return cls(PhoData(freq, mode, mass))

    @cached_property
    def disp(self) -> FloatArray:
        """The displacement in Cartesian coordinates of all modes"""
        return np.array([self.mode2disp(idx) for idx in range(len(self.freq))])

    @cached_property
    def translation(self) -> IntArray:
        """Get the index of translation modes"""
        translation_idx = []
        for idx, disp in enumerate(self.disp):
            if self.translation_judge(disp):
                translation_idx.append(idx)

        return np.array(translation_idx, dtype=int)

    def check_index(self, idx: int) -> None:
        """Check if the index is valid for the phonon data

        Args:
            idx: Index to check
        """
        if idx < 0 or idx >= len(self.freq):
            raise IndexError(
                f"Index {idx} is out of bounds for data with length {len(self.freq)}."
            )

    @overload
    def to_meV(self, freq: float) -> float: ...

    @overload
    def to_meV(self, freq: FloatArray) -> FloatArray: ...

    def to_meV(self, freq: float | FloatArray | None = None) -> float | FloatArray:
        """Convert frequency to meV

        Args:
            freq: Frequency in THz, defaults to self.freq

        Returns:
            Frequency in meV
        """
        if freq is None:
            freq = self.freq
        return freq * ENERGY_UNITS["THz"] * 1e3

    @overload
    def phonon_number(self, T: float, freq: float) -> float: ...

    @overload
    def phonon_number(self, T: float, freq: FloatArray) -> FloatArray: ...

    def phonon_number(
        self, T: float, freq: float | FloatArray | None = None
    ) -> float | FloatArray:
        """Calculate phonon number at temperature T

        The phonon number is calculated using the Bose-Einstein distribution.

        Args:
            T: Temperature in Kelvin
            freq: Frequency in THz, defaults to self.freq

        Returns:
            Phonon number
        """
        if freq is None:
            freq = self.freq

        assert T > 0, "The temperature must be greater than 0 K"

        return 1 / (
            np.exp(freq * ENERGY_UNITS["THz"] / (T * ENERGY_UNITS["kelvin"])) - 1
        )

    def mode2disp(self, idx: int) -> FloatArray:
        """Convert the i-th eigenmode output by VASP to disp in Cartesian coord"""
        self.check_index(idx)
        return self.mode[idx] / np.sqrt(self.mass)[:, np.newaxis]

    def decompose(self, disp: FloatArray) -> FloatArray:
        """Decompose displacement into eigenmodes

        Args:
            disp: Displacement in Cartesian coordinates

        Returns:
            Decomposed coefficients in eigenmodes
        """
        assert len(disp.shape) == 2 and disp.shape[1] == 3, (
            "The input displacement must be a 2D array with shape (n, 3)"
        )
        assert len(disp) == len(self.freq), (
            "The length of displacement must match the normal modes"
        )

        disp_normal = disp * np.sqrt(self.mass)[:, np.newaxis]

        projection = self.mode * disp_normal
        return np.sum(projection, axis=(1, 2))

    def free_energy(self, T: np.floating) -> np.floating:
        r"""Calculate free energy at temperature T

        The equation is
        \sum_i 1/2 \hbar \omega_i + k_B T \ln(1 - e^{-\hbar \omega_i / k_B T})

        Args:
            T: Temperature in Kelvin
        """
        assert T >= 0, "The temperature must be greater than or equal to 0 K"
        mode_notrans = np.setdiff1d(
            np.arange(len(self.freq), dtype=int), self.translation
        )

        energy_part = 0.5 * ENERGY_UNITS["THz"] * self.freq[mode_notrans]

        entropy_part = (
            ENERGY_UNITS["kelvin"]
            * T
            * np.log(
                1
                - np.exp(
                    -ENERGY_UNITS["THz"]
                    * self.freq[mode_notrans]
                    / (T * ENERGY_UNITS["kelvin"])
                )
            )
        )

        if T == 0:
            return np.sum(energy_part)

        return np.sum(energy_part + entropy_part)

    @staticmethod
    def translation_judge(disp: FloatArray) -> bool:
        """Judge if the displacement is a translation

        Args:
            disp: Displacement in Cartesian coordinates
        """
        assert len(disp.shape) == 2 and disp.shape[1] == 3, (
            "The input displacement must be a 2D array with shape (n, 3)"
        )

        # calculate the standard deviation of every column
        disp_std = np.std(disp, axis=0)

        return bool(np.all(disp_std < 1e-3))
