"""The incar tags module"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# TAG_DATABASE is defined but not used in the file, so it's commented out.
# TAG_DATABASE = Path(__file__).parent / "data" / "tags_info"
TAG_FILE_PREDEFINED = Path(__file__).parent / "data" / "tag_data.json"

ALL_TAGS_PREDEFINED = json.loads(TAG_FILE_PREDEFINED.read_text())

TAGS_PREDEFINED = list(ALL_TAGS_PREDEFINED.keys())


class IncarType(Enum):
    """Enumeration for the base Python type of an INCAR tag."""

    INTEGER = 1
    FLOAT = 2
    BOOL = 3
    STRING = 4
    LIST = 5
    ENUM = 6
    NONE = 7


class TagDefinition:
    """Holds the schema and validation logic for a specific INCAR tag."""

    def __init__(self, name: str, definition: dict):
        """Initialize a TagDefinition instance."""
        self.name = name
        self.description = definition.get("description", "")
        self.type_str = definition.get("type", "string")
        self.valid_choices: list[Any] | None | dict[str, Any] = definition.get(
            "valid_values"
        )
        self.base_type, self.sub_type = self._parse_type(self.type_str)

    @staticmethod
    def _parse_type(type_str: str) -> tuple[IncarType, type]:
        """Parses the type string to determine the base and sub-type.

        For simple types, returns the corresponding IncarType and None for sub_type.
        For list type or Enum type, returns the appropriate IncarType and the subtype.

        Args:
            type_str: The type string read from file.

        Returns:
            base_type: The base IncarType
            sub_type: the subtype for list or Enum IncarType,
                type(None) for simple IncarType.
        """
        type_str_lower = type_str.lower()

        simple_types = {
            "int": (IncarType.INTEGER, type(None)),
            "float": (IncarType.FLOAT, type(None)),
            "bool": (IncarType.BOOL, type(None)),
            "str": (IncarType.STRING, type(None)),
        }
        if type_str_lower in simple_types:
            return simple_types[type_str_lower]

        subtype_map: dict[str, type] = {
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
        }
        if type_str_lower.startswith("list"):
            match = re.match(r"list\[(\w+)\]", type_str_lower)
            if match:
                return IncarType.LIST, subtype_map.get(match.group(1), type(None))
        elif type_str_lower.startswith("enum:"):
            subtype_name = type_str_lower.split(":")[1]
            return IncarType.ENUM, subtype_map.get(subtype_name, type(None))

        return IncarType.NONE, type(None)  # pragma: no cover

    def validate(self, value: int | float | bool | str | list):
        """Validates a given value against the tag's definition."""
        py_type_map: dict[IncarType, type | tuple[type, ...]] = {
            IncarType.INTEGER: int,
            IncarType.FLOAT: (float, int),
            IncarType.BOOL: bool,
            IncarType.STRING: str,
            IncarType.LIST: list,
        }

        if self.base_type == IncarType.ENUM:
            self._validate_enum(value)
            return

        if self.base_type == IncarType.NONE:
            raise ValueError(
                "Tag '{self.name}' has type NONE, cannot validate value. "
                "This is a bug, please report it."
            )  # pragma: no cover

        expected_py_type = py_type_map[self.base_type]
        if not isinstance(value, expected_py_type):
            raise TypeError(
                f"Tag '{self.name}' expects type {self.base_type.name}, "
                f"but got {type(value).__name__}."
            )

        if self.base_type == IncarType.LIST and isinstance(value, list):
            self._validate_list(value)

    def _validate_enum(self, value: Any) -> None:
        """Validate enum type tags."""
        if self.valid_choices is None:  # pragma: no cover
            raise ValueError(
                f"Tag '{self.name}' is of type ENUM but has no valid choices defined."
                f"This is a bug, please report it."
            )

        if not isinstance(value, self.sub_type):
            raise TypeError(
                f"Tag '{self.name}' expects type {self.sub_type.__name__} for ENUM, "
                f"but got {type(value).__name__}."
            )

        if isinstance(self.valid_choices, list):
            if value not in self.valid_choices:
                raise ValueError(
                    f"Value '{value}' for tag '{self.name}' is not valid. "
                    f"Allowed values are: {self.valid_choices}"
                )
            return

        if isinstance(self.valid_choices, dict):
            self._validate_enum_dict(value, self.valid_choices)
        else:  # pragma: no cover
            raise ValueError(
                f"Wrong `valid_choices` for Tag '{self.name}'"
                "Expected a list or a dictionary. "
                "This should not happen, please report it."
            )
        return None

    def _validate_enum_dict(self, value: Any, valid_choices: dict[str, Any]):
        """Validate enum type tags with a dictionary of valid values."""
        min_val = valid_choices.get("min")
        max_val = valid_choices.get("max")
        additional = valid_choices.get("additional", [])

        if value in additional:
            return

        in_range = (min_val is None or value >= min_val) and (
            max_val is None or value <= max_val
        )

        if not in_range:
            range_parts = []
            if min_val is not None:
                range_parts.append(f">= {min_val}")
            if max_val is not None:  # pragma: no cover
                range_parts.append(f"<= {max_val}")

            msg = f"Value '{value}' for tag '{self.name}' is not valid."
            if range_parts:
                msg += f" Must be in range [{', '.join(range_parts)}]."
            else:  # pragma: no cover
                raise ValueError(
                    f"Tag {self.name} has no valid range defined, "
                    "This should be a bug, please report it."
                )
            if additional:
                msg += f" Other allowed values are: {additional}."
            else:  # pragma: no cover
                raise ValueError(
                    f"Tag {self.name} has no additional allowed values defined, "
                    "This should be a bug, please report it."
                )
            raise ValueError(msg)
        else:  # pragma: no cover
            raise ValueError(
                f"Some unexpected error occurred while validating tag '{self.name}'."
            )

    def _validate_list(self, value: list):
        """Validate the values in the list is of the correct subtype if specified."""
        if self.sub_type is type(None):  # pragma: no cover
            raise ValueError(
                f"Tag '{self.name}' is of type LIST but has no subtype defined. "
                f"This is a bug, please report it."
            )

        if all(isinstance(item, self.sub_type) for item in value):
            return

        wrong_item = []
        wrong_type = []
        for item in value:
            if not isinstance(item, self.sub_type):
                wrong_item.append(item)
                wrong_type.append(type(item).__name__)
        raise TypeError(
            f"Tag '{self.name}' expects a list of {self.sub_type.__name__}, "
            f"but got {len(wrong_item)} items with types: {', '.join(wrong_type)}."
            f"Wrong items: {', '.join(map(str, wrong_item))}."
        )


TAG_DEFINITIONS = {
    name: TagDefinition(name, data) for name, data in ALL_TAGS_PREDEFINED.items()
}


@dataclass
class Tag:
    """A class representing an instance of a tag in the INCAR file."""

    name: str
    value: Any

    def __post_init__(self):
        """Validate the tag upon creation."""
        if self.name not in TAG_DEFINITIONS:
            # TODO: add a similarity judge program to get the most similar tag
            raise ValueError(f"Tag '{self.name}' seems to be invalid or very rare.")

        self.definition.validate(self.value)

    @property
    def definition(self) -> TagDefinition:
        """Access the tag's static definition."""
        return TAG_DEFINITIONS[self.name]

    def __str__(self) -> str:
        """String representation of the tag."""
        desc = self.definition.description
        return (
            f"{self.name} = {self.value}  # {desc}"
            if desc
            else f"{self.name} = {self.value}"
        )
