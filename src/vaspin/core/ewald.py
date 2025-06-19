"""The Ewald summation module"""

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
            self.ewald_param_modified
            * self.cube_length
            / np.sqrt(np.linalg.det(self.dielectric))
        )

    @property
    def ewald_param_modified(self):
        """Calculate the modified Ewald parameter."""
        lr = np.cbrt([np.linalg.norm(i) for i in self.lattice])
        lg = np.cbrt([np.linalg.norm(i) for i in self.rec_lattice])

        return np.sqrt(lg / lr / 2)

    @property
    def pot_diff(self):
        """Calculate the potential difference for Ewald summation."""
        return -1 / 4 / self.volume / self.ewald_param_modified**2

    def site_potential(self, coor_frac: FloatArray):
        """Calculate the electrostatic potential at a given fractional coordinate."""
        r_part = self.ewald_sum_r(include_origin=True, shift=coor_frac)
        g_part = self.ewald_sum_g(coor_frac)
        return self.pot_diff + r_part + g_part

    def ewald_sum_r(self, include_origin: bool, shift: FloatArray):
        """Calculate the real space Ewald summation."""
        sum = 0
        for r in self.rvectors(include_origin=include_origin, shift=shift):
            root_r_inv_dielectri = np.sqrt(
                np.einsum("i,ij,j->", r, np.linalg.inv(self.dielectric), r)
            )
            sum += (
                erfc(self.ewald_param_modified * root_r_inv_dielectri)
                / root_r_inv_dielectri
            )

        return sum / (4 * np.pi * self.die_effective)

    def ewald_sum_g(self, coor_frac: FloatArray):
        """Calculate the reciprocal space Ewald summation."""
        coor_cate = np.dot(coor_frac, self.lattice)
        sum = 0
        for g in self.gvectors():
            g_dielectric_g = np.einsum("i,ij,j->", g, self.dielectric, g)
            sum += (
                np.exp(-g_dielectric_g / 4 / self.ewald_param_modified**2)
                / g_dielectric_g
                * np.cos(np.dot(g, coor_cate))
            )

        return sum / self.volume

    @staticmethod
    def grid_points(gridsize: IntArray, shift: FloatArray | None = None):
        """Generate grid points for Ewald summation."""
        if shift is None:
            shift = np.array([0, 0, 0])

        fx, fy, fz = [
            np.arange(-gridsize[i], gridsize[i] + 1, 1) - shift[i] for i in range(3)
        ]

        # this may accelerate the calculation
        gridz, gridy, gridx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape(
            (3, -1)
        )
        return np.array([gridx, gridy, gridz], dtype=float).T

    def rvectors(self, include_origin: bool = True, shift: FloatArray | None = None):
        """Calculate the real space vectors for Ewald summation."""
        rpoints = self.grid_points(self.r_gridsize, shift)

        if not include_origin:
            rpoints = np.delete(rpoints, len(rpoints) // 2, axis=0)

        return np.dot(rpoints, self.lattice)

    def gvectors(self):
        """Calculate the reciprocal vectors for Ewald summation."""
        gpoints = self.grid_points(self.g_gridsize)

        gpoints = np.delete(gpoints, len(gpoints) // 2, axis=0)
        return np.dot(gpoints, self.rec_lattice)

    @property
    def r_gridsize(self) -> IntArray:
        """Calculate the grid size for real space Ewald summation."""
        max_length = self.cutoff / self.ewald_param_modified
        return np.array(
            [np.ceil(max_length / np.linalg.norm(self.lattice[i])) for i in range(3)],
            dtype=int,
        )

    @property
    def g_gridsize(self) -> IntArray:
        """Calculate the grid size for reciprocal space Ewald summation."""
        max_length = 2 * self.ewald_param_modified * self.cutoff
        return np.array(
            [
                np.ceil(max_length / np.linalg.norm(self.rec_lattice[i]))
                for i in range(3)
            ],
            dtype=int,
        )
