"""A module for validating INCAR files."""

from __future__ import annotations

from typing import NamedTuple, Optional

from .core import Incar
from .rules import ValidationLevel, ValidationResult, ValidationRuleRegistry


class IncarCheckResult(NamedTuple):
    """A data class to store the results of an INCAR check."""

    warnings: list[ValidationResult]
    errors: list[ValidationResult]
    suggestions: list[ValidationResult]


class IncarValidator:
    """A class to validate the INCAR file using a set of rules."""

    def __init__(
        self,
        include_rules: Optional[list[str]] = None,
        exclude_rules: Optional[list[str]] = None,
    ):
        """Initializes the validator with a specified set of rules.

        Args:
            include_rules: A list of rule names to use for validation.
                If provided, only these rules will be used.
            exclude_rules: A list of rule names to exclude from validation.
                All other registered rules will be used.

        Raises:
            ValueError: If both `include_rules` and `exclude_rules` are provided.
        """
        if include_rules and exclude_rules:
            raise ValueError("Cannot specify both include_rules and exclude_rules.")

        if include_rules is not None:
            self._rules = ValidationRuleRegistry.get_rules(include_rules)
        elif exclude_rules is not None:
            self._rules = ValidationRuleRegistry.get_rules_except(exclude_rules)
        else:
            self._rules = ValidationRuleRegistry.get_all_rules()

    def validate(self, incar: Incar) -> IncarCheckResult:
        """Validate the INCAR file by executing all registered rules.

        The poscar, potcar, and kpoints arguments are not used in the base
        implementation but are kept for future rules that might need them.

        Args:
            incar: The INCAR object to validate.

        Returns:
            The check result containing errors, warnings, and suggestions.
        """
        errors: list[ValidationResult] = []
        warnings: list[ValidationResult] = []
        suggestions: list[ValidationResult] = []

        for rule in self._rules:
            result = rule(incar)
            if result is None:
                continue
            match result.level:
                case ValidationLevel.ERROR:
                    errors.append(result)
                case ValidationLevel.WARNING:
                    warnings.append(result)
                case ValidationLevel.SUGGESTION:
                    suggestions.append(result)

        return IncarCheckResult(
            warnings=warnings, errors=errors, suggestions=suggestions
        )
