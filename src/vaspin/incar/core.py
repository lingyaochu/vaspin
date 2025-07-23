"""The Incar module provides functionality to handle INCAR file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional, Self

from vaspin.core.io import read_incar, write_incar

from .tags import Tag

# from .validator import IncarCheckResult, IncarValidator


class Incar:
    """A class to represent the INCAR file."""

    def __init__(self, data: Optional[dict[str, Any]] = None) -> None:
        """Initialize the INCAR object with data.

        Args:
            data: A dictionary containing INCAR tags and their values.
        """
        self._tags: dict[str, Tag] = {}
        if data is None:
            data = {}
        for key, value in data.items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an INCAR tag, with validation."""
        upper_key = key.upper()
        self._tags[upper_key] = Tag(name=upper_key, value=value)

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        return self._tags[key.upper()].value

    def __delitem__(self, key: str) -> None:
        """Delete an INCAR tag."""
        del self._tags[key.upper()]

    def __iter__(self) -> Iterator[str]:
        """Iterate over tag names."""
        return iter(self._tags)

    def __len__(self) -> int:
        """Get the number of items."""
        return len(self._tags)

    def get(self, key: str, default: Any = None) -> Any:
        """Get item by key, with a default value if not found."""
        if key.upper() in self._tags:
            return self._tags[key.upper()].value
        else:
            return default

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        """Read INCAR from a file.

        Args:
            filepath: Path to the INCAR file.

        Returns:
            An instance of Incar
        """
        if not isinstance(filepath, (str, Path)):
            raise TypeError(f"Expected str or Path, got {type(filepath)}")
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist.")
        incar_data = read_incar(str(path))
        return cls(incar_data)

    def write_incar(self, directory: str = ".", name: str = "INCAR") -> None:
        """Write INCAR to a file.

        Args:
            directory: Directory where the INCAR file will be written.
            name: Name of the INCAR file.
        """
        write_incar(list(self._tags.values()), directory, name)

    # def check(self, poscar=None, potcar=None, kpoints=None) -> IncarCheckResult:
    #     """Check the compatibility of INCAR tags using the validator."""
    #     validator = IncarValidator()
    #     return validator.validate(self, poscar, potcar, kpoints)

    # def judge(self) -> str:
    #     """Judge the calculation type."""
    #     raise NotImplementedError

    # @classmethod
    # def from_preset(cls, name: str) -> Self:
    #     """Create INCAR from a preset."""
    #     raise NotImplementedError
