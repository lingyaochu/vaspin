"""Test suite for utility functions in vaspin.utils.utils module"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vaspin.utils.utils import clean, find_rotation, wrap_frac


class TestClean:
    """Test class for the clean function"""

    def test_create_new_directory(self):
        """Test clean creates a new directory if it does not exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = f"{tmpdir}/new_dir"
            assert not Path(new_dir).exists(), (
                "Directory should not exist before clean is called"
            )
            clean(new_dir)

            assert Path(new_dir).exists(), "clean do not create the directory"
            assert Path(new_dir).is_dir(), "clean creates a file instead of a directory"

    def test_existing_directory_empty(self):
        """Test clean does nothing if the directory already exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert Path(tmpdir).exists(), "There should be a directory existing"

            clean(tmpdir)

            assert Path(tmpdir).exists(), (
                "clean should not remove the existing directory"
            )
            assert Path(tmpdir).is_dir(), (
                "clean should not change the existing directory"
            )

    def test_create_nested_directories(self):
        """Test clean creates nested directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = f"{tmpdir}/a/b/c/d"
            assert next(Path(tmpdir).iterdir(), None) is None, (
                "No subfiles or directory should exist"
            )

            clean(nested_dir)
            assert Path(nested_dir).exists(), (
                "clean do not create the nested directories"
            )
            assert Path(nested_dir).is_dir(), (
                "clean creates a file instead of a directory"
            )


class TestWrapFrac:
    """Test class for the wrap_frac function"""

    @pytest.mark.parametrize(
        "input_coords,expected_coords",
        [
            # normal case
            ([0.22, 0.88, 0.33], [0.22, 0.88, 0.33]),
            # negative values in [-1,0)
            ([-0.13, 0.15, 0.06], [0.87, 0.15, 0.06]),
            ([-0.22, -0.33, -0.44], [0.78, 0.67, 0.56]),
            # negative values < -1
            ([-2.08, 0.33, 0.44], [0.92, 0.33, 0.44]),
            ([-3.44, -10.5, -200.33], [0.56, 0.5, 0.67]),
            # values within [1,2)
            ([1.03, 0.33, 0.44], [0.03, 0.33, 0.44]),
            ([1.44, 1.55, 1.66], [0.44, 0.55, 0.66]),
            # values > 2
            ([20.34, 0.04, 0.35], [0.34, 0.04, 0.35]),
            ([20.33, 50.43, 300.56], [0.33, 0.43, 0.56]),
            # mixed values
            ([-0.33, 1.45, -3.44], [0.67, 0.45, 0.56]),
            ([10.35, 20.44, -30.56], [0.35, 0.44, 0.44]),
            # boundary case
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
            ([-1.0, 0.0, 2.0], [0.0, 0.0, 0.0]),
        ],
    )
    def test_wrap_frac_single(self, input_coords, expected_coords):
        """Test wrap_frac with various input coordinates"""
        input = np.array(input_coords)
        result = wrap_frac(input)
        expected = np.array(expected_coords)
        assert result == pytest.approx(expected, abs=1e-7), (
            f"wrap_frac({input}) = {result}, expected {expected}"
        )

    def test_wrap_frac_multi(self):
        """Test wrap_frac with multiple coordinates"""
        input = np.array(
            [
                [0.22, 0.88, 0.33],
                [-0.22, -0.33, -0.44],
                [-2.08, 0.33, 0.44],
                [1.44, 1.55, 1.66],
                [10.35, 20.44, -30.56],
                [-1.0, 0.0, 2.0],
            ]
        )

        expected = np.array(
            [
                [0.22, 0.88, 0.33],
                [0.78, 0.67, 0.56],
                [0.92, 0.33, 0.44],
                [0.44, 0.55, 0.66],
                [0.35, 0.44, 0.44],
                [0.0, 0.0, 0.0],
            ]
        )
        result = wrap_frac(input)
        for i, (result_coord, expected_coord) in enumerate(
            zip(result, expected, strict=False)
        ):
            assert result_coord == pytest.approx(expected_coord, abs=1e-7), (
                f"wrap_frac({input[i]}) = {result_coord}, expected {expected_coord}"
            )


class TestFindRotation:
    """Test class for the find_rotation function"""

    def test_length_mismatch(self):
        """Test find_rotation raises AssertionError for length mismatch"""
        with pytest.raises(
            AssertionError, match="the dimension of input arrow should be 3"
        ):
            find_rotation(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

        with pytest.raises(
            AssertionError, match="the dimension of input arrow should be 3"
        ):
            find_rotation(
                np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])
            )

        with pytest.raises(
            AssertionError, match="the dimension of input arrow should be 3"
        ):
            find_rotation(np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0]))

    def test_zero_length(self):
        """Test find_rotation raises AssertionError for zero-length vectors"""
        with pytest.raises(
            AssertionError, match="the length of input arrow should not be 0"
        ):
            find_rotation(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

        with pytest.raises(
            AssertionError, match="the length of input arrow should not be 0"
        ):
            find_rotation(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

    def test_parallel(self):
        """Test find_rotation when no rotation is needed"""
        identity = np.eye(3)
        result = find_rotation(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        for i, (result_row, identity_row) in enumerate(
            zip(result, identity, strict=False)
        ):
            assert result_row == pytest.approx(identity_row, abs=1e-7), (
                f"find_rotation no rotation row {i} = {result_row},"
                f" expected {identity_row}"
            )

    @pytest.mark.parametrize(
        "arrow1,arrow2,expected",
        [
            ([1, 0, 0], [-1, 0, 0], [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            ([0, 1, 0], [0, -1, 0], [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        ],
    )
    def test_anti_parallel(self, arrow1, arrow2, expected):
        """Test find_rotation when vectors are anti-parallel"""
        result = find_rotation(np.array(arrow1), np.array(arrow2))
        expected = np.array(expected)
        for i, (result_row, expected_row) in enumerate(
            zip(result, expected, strict=False)
        ):
            assert result_row == pytest.approx(expected_row, abs=1e-7), (
                f"find_rotation({arrow1}, {arrow2}) row {i} = {result_row},"
                f" expected {expected_row}"
            )

    @pytest.mark.parametrize(
        "arrow1,arrow2,expected",
        [
            ([1, 0, 0], [0, 1, 0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            ([2, 0, 0], [0, 3, 0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        ],
    )
    def test_find_rotation(self, arrow1, arrow2, expected):
        """Test find_rotation with various input vectors"""
        result = find_rotation(np.array(arrow1), np.array(arrow2))
        expected = np.array(expected)
        for i, (result_row, expected_row) in enumerate(
            zip(result, expected, strict=False)
        ):
            assert result_row == pytest.approx(expected_row, abs=1e-7), (
                f"find_rotation({arrow1}, {arrow2}) row {i} = {result_row},"
                f" expected {expected_row}"
            )
