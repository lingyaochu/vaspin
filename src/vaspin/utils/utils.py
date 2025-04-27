# -*- coding: utf-8 -*-
"""Utility functions for VASP file handling and structure manipulation."""

import os

import numpy as np

from vaspin.types.array import FloatArray


def clean(target_dir: str) -> None:
    """Create the target directory if it does not exist.

    Args:
        target_dir: Path to the target directory

    Returns:
        None
    """
    if os.path.exists(target_dir):
        pass
    else:
        os.makedirs(target_dir)
    return None


def wrap_frac(coordinates: FloatArray) -> FloatArray:
    """Normalize fractional coordinates to the [0,1) interval.

    Args:
        coordinates: Fractional coordinates

    Returns:
        Normalized fractional coordinates
    """
    return coordinates % 1.0


def find_rotation(arrow1: FloatArray, arrow2: FloatArray) -> FloatArray:
    """Find the rotation matrix that rotates arrow1 to arrow2.

    If arrow1 and arrow2 have different lengths,
    the rotation matrix for their corresponding unit vectors will be found

    Args:
        arrow1: Original vector
        arrow2: Target vector

    Returns:
        Rotation matrix

    """
    assert arrow1.size == arrow2.size == 3, "the dimension of input arrow should be 3"
    assert np.linalg.norm(arrow1) > 1e-10 and np.linalg.norm(arrow2) > 1e-10, (
        "the length of input arrow should not be 0"
    )

    arrow1 = np.array(arrow1, dtype=np.float64)
    arrow2 = np.array(arrow2, dtype=np.float64)
    arrow1 = arrow1 / np.linalg.norm(arrow1)
    arrow2 = arrow2 / np.linalg.norm(arrow2)

    dot_product = np.clip(np.dot(arrow1, arrow2), -1.0, 1.0)  # 避免数值误差导致的问题
    cross_product = np.cross(arrow1, arrow2)
    cross_norm = np.linalg.norm(cross_product)

    if cross_norm < 1e-10:
        if dot_product > 0:
            return np.eye(3)
        else:
            if abs(arrow1[0]) < abs(arrow1[1]):
                rotation_axis = np.cross(arrow1, np.array([1.0, 0.0, 0.0]))
            else:
                rotation_axis = np.cross(arrow1, np.array([0.0, 1.0, 0.0]))
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            theta = np.pi
    else:
        rotation_axis = cross_product / cross_norm
        theta = np.arccos(dot_product)

    x, y, z = rotation_axis
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c

    _matrix = np.zeros((3, 3))
    _matrix[0, 0] = c + x * x * t
    _matrix[0, 1] = x * y * t - z * s
    _matrix[0, 2] = x * z * t + y * s
    _matrix[1, 0] = y * x * t + z * s
    _matrix[1, 1] = c + y * y * t
    _matrix[1, 2] = y * z * t - x * s
    _matrix[2, 0] = z * x * t - y * s
    _matrix[2, 1] = z * y * t + x * s
    _matrix[2, 2] = c + z * z * t

    return _matrix
