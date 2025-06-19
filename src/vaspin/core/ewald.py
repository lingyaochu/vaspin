"""The Ewald summation module

This module is heavily inspired by the/adapted from the `pydefect` package.

Original project: https://github.com/kumagai-group/pydefect
Original copyright: Copyright (c) 2020 kumagai-group

The LICENSE file can be found at: /LICENSES/pydefect/LICENSE
"""

import numpy as np
from scipy.special import erfc

from vaspin.types.array import FloatArray, IntArray

CUTOFF = 15


class Ewald:
    """Ewald summation class for electrostatic potential in periodic system."""

    def __init__(self, lattice: FloatArray, dielectric: FloatArray, cutoff=CUTOFF):
        """Initialize the Ewald summation parameters."""
        self.lattice = lattice
        self.rec_lattice = np.linalg.inv(self.lattice).T * 2 * np.pi
        self.dielectric = dielectric
        self.die_effective = np.sqrt(np.linalg.det(self.dielectric))
        self.volume = np.abs(np.linalg.det(self.lattice))
        self.cube_length = np.cbrt(self.volume)
        self.cutoff = cutoff

    @property
    def ewald_param(self):
        """Calculate the Ewald parameter."""
        return (
            self.ewald_gamma
            * self.cube_length
            / np.sqrt(np.linalg.det(self.dielectric))
        )

    @property
    def ewald_gamma(self):
        """Calculate the gamma parameter for Ewald summation by default value of CUTOFF

        This is gamma value defined in Phys. Rev. B 89, 195205 (2014).
        """
        lr = np.cbrt([np.linalg.norm(i) for i in self.lattice])
        lg = np.cbrt([np.linalg.norm(i) for i in self.rec_lattice])

        return np.sqrt(lg / lr / 2)

    @property
    def lattice_energy(self):
        """The charge interaction energy in the periodic defect system"""
        r_part = self.ewald_sum_r(np.array([0, 0, 0]))
        g_part = self.ewald_sum_g(np.array([0, 0, 0]))
        return (r_part + g_part + self.pot_diff + self.pot_cancel) / 2

    def site_potential(self, coor_frac: FloatArray):
        """Calculate the electrostatic potential at a given fractional coordinate.

        The quantity is defined in Phys. Rev. B 89, 195205 (2014). Eq(14)
        """
        r_part = self.ewald_sum_r(coor_frac)
        g_part = self.ewald_sum_g(coor_frac)
        return r_part + g_part + self.pot_diff

    def ewald_sum_r(self, coor_frac: FloatArray):
        """Calculate the real space Ewald summation.

        The first term in Phys. Rev. B 89, 195205 (2014). Eq(14)

        Args:
            coor_frac: Fractional coordinates of a lattice site.
        """
        sum = 0
        coor_cate = np.dot(coor_frac, self.lattice)

        rvectors = self.rvectors()
        if coor_frac == np.array([0, 0, 0]):
            rvectors = np.delete(rvectors, len(rvectors) // 2, axis=0)

        for r in rvectors:
            root_r_inv_dielectri = np.sqrt(
                np.einsum(
                    "i,ij,j->",
                    r - coor_cate,
                    np.linalg.inv(self.dielectric),
                    r - coor_cate,
                )
            )
            sum += erfc(self.ewald_gamma * root_r_inv_dielectri) / root_r_inv_dielectri

        return sum / (4 * np.pi * self.die_effective)

    @property
    def pot_diff(self):
        """The energy term caused by finite gaussian charge.

        The second term in Phys. Rev. B 89, 195205 (2014). Eq(14)
        """
        return -1 / 4 / self.volume / self.ewald_gamma**2

    @property
    def pot_cancel(self):
        """The  the cancellation term to `ewald_sum_g`

        The fourth term in Phys. Rev. B 89, 195205 (2014). Eq(8)
        """
        return -self.ewald_gamma / (
            2 * np.pi * np.sqrt(np.pi * np.linalg.det(self.dielectric))
        )

    def ewald_sum_g(self, coor_frac: FloatArray):
        """Calculate the reciprocal space Ewald summation.

        The third term in Phys. Rev. B 89, 195205 (2014). Eq(14)

        Args:
            coor_frac: Fractional coordinates of a lattice site.
        """
        coor_cate = np.dot(coor_frac, self.lattice)
        sum = 0
        for g in self.gvectors():
            g_dielectric_g = np.einsum("i,ij,j->", g, self.dielectric, g)
            sum += (
                np.exp(-g_dielectric_g / 4 / self.ewald_gamma**2)
                / g_dielectric_g
                # the distribution of G is symmetric, the i*sin() term vanish
                * np.cos(np.dot(g, coor_cate))
            )

        return sum / self.volume

    @staticmethod
    def grid_points(gridsize: IntArray):
        """Generate grid points for Ewald summation."""
        fx, fy, fz = [np.arange(-gridsize[i], gridsize[i] + 1, 1) for i in range(3)]

        # this may accelerate the calculation
        gridz, gridy, gridx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape(
            (3, -1)
        )
        return np.array([gridx, gridy, gridz], dtype=float).T

    def rvectors(self):
        """Calculate the real space vectors for Ewald summation."""
        rpoints = self.grid_points(self.r_gridsize)

        return np.dot(rpoints, self.lattice)

    def gvectors(self):
        """Calculate the reciprocal vectors for Ewald summation."""
        gpoints = self.grid_points(self.g_gridsize)

        gpoints = np.delete(gpoints, len(gpoints) // 2, axis=0)
        return np.dot(gpoints, self.rec_lattice)

    @property
    def r_gridsize(self) -> IntArray:
        """Calculate the grid size for real space Ewald summation."""
        max_length = self.cutoff / self.ewald_gamma
        return np.array(
            [np.ceil(max_length / np.linalg.norm(self.lattice[i])) for i in range(3)],
            dtype=int,
        )

    @property
    def g_gridsize(self) -> IntArray:
        """Calculate the grid size for reciprocal space Ewald summation."""
        max_length = 2 * self.ewald_gamma * self.cutoff
        return np.array(
            [
                np.ceil(max_length / np.linalg.norm(self.rec_lattice[i]))
                for i in range(3)
            ],
            dtype=int,
        )
