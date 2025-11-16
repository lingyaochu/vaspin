"""Tests for the Tag class in the INCAR module."""

import re

import pytest

from vaspin.incar.tags import Tag


class TestTags:
    """Tests for the Tag class and its validation logic."""

    def test_float_validation(self):
        """Test float validation for tags."""
        _tag = Tag(name="ENCUT", value=520)

    def test_integer_validation(self):
        """Test integer validation for tags."""
        _tag = Tag(name="NCORE", value=4)
        with pytest.raises(TypeError):
            _tag = Tag(name="NCORE", value=1.5)

    def test_bool_validation(self):
        """Test boolean validation for tags."""
        _tag = Tag(name="LDMATRIX", value=True)

    def test_enum_validation_normal(self):
        """Test enum validation for tags."""
        _tag = Tag(name="ISMEAR", value=0)
        _tag = Tag(name="ISPIN", value=2)
        _tag = Tag(name="LDAUTYPE", value=2)
        _tag = Tag(name="PREC", value="Accurate")

    def test_enum_validation_wrong_type(self):
        """Test enum validation with wrong type."""
        with pytest.raises(
            TypeError, match="Tag 'ISMEAR' expects type int for ENUM, but got str."
        ):
            _tag = Tag(name="ISMEAR", value="0")

    def test_enum_validation_not_in_enum(self):
        """Test enum validation with a value not in the valid choices."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Value '5' for tag 'LDAUTYPE' is not valid. "
                "Allowed values are: [1, 2, 4]"
            ),
        ):
            _tag = Tag(name="LDAUTYPE", value=5)

    def test_enum_validation_not_in_range(self):
        """Test enum validation with a value not in the valid range."""
        error_msg = (
            "Value '-6' for tag 'ISMEAR' is not valid. "
            "Must be in range [>= 1]. "
            "Other allowed values are: [-15, -14, -5, -4, -3, -2, -1, 0]."
        )
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            _tag = Tag(name="ISMEAR", value=-6)

    def test_list_validation(self):
        """Test list validation for tags."""
        _tag = Tag(name="LDAUL", value=[2, 2, 2])

        with pytest.raises(TypeError):
            _tag = Tag(name="LDAUL", value=[2, "a", 2])

    def test_str_repr(self):
        """Test string representation of the Tag class."""
        tag = Tag(name="ENCUT", value=520)
        assert str(tag) == (
            "ENCUT = 520  "
            "# specifies the energy cutoff for the plane-wave basis set in eV."
        )

    def test_wrong_tag(self):
        """Test handling of a wrong or very rare tag."""
        wrong_tag = {"name": "WRONGTAG", "value": "wrong"}
        with pytest.raises(
            ValueError, match="Tag 'WRONGTAG' seems to be invalid or very rare."
        ):
            Tag(**wrong_tag)
