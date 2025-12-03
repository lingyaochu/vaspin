"""A module for validation rules and their registry."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, ClassVar, NamedTuple, Optional

from .core import Incar


class ValidationLevel(Enum):
    """Severity level of a validation result."""

    ERROR = "error"
    WARNING = "warning"
    SUGGESTION = "suggestion"


class ValidationResult(NamedTuple):
    """Represents the outcome of a single validation check."""

    level: ValidationLevel
    message: str
    name: str


ValidationRule = Callable[[Incar], Optional[ValidationResult]]


class ValidationRuleRegistry:
    """A registry for managing and accessing validation rules."""

    _rules: ClassVar[dict[str, ValidationRule]] = {}

    @classmethod
    def register(
        cls, name: str, level: ValidationLevel, message: str
    ) -> Callable[[Callable[[Incar], bool | dict[str, Any]]], ValidationRule]:
        """A decorator to create and register a validation rule.

        This decorator simplifies the creation of validation rules by allowing you
        to write a simple check function. The rule is automatically registered
        in the registry, allowing it to be referenced by name.

        Args:
            name: The name of the rule (e.g., 'encut-not-set').
            level: The validation level for the rule.
            message: The message to be displayed if validation fails.
                This can be a format string.

        Returns:
            A decorator that transforms a check function into a validation rule.
        """

        def decorator(
            check_func: Callable[[Incar], bool | dict[str, Any]],
        ) -> ValidationRule:
            def wrapper(incar: Incar) -> Optional[ValidationResult]:
                validation_failed = check_func(incar)
                if not validation_failed:
                    return None

                msg = (
                    message.format(**validation_failed)
                    if isinstance(validation_failed, dict)
                    else message
                )
                return ValidationResult(level=level, message=msg, name=name)

            wrapper.__doc__ = check_func.__doc__
            wrapper.__name__ = check_func.__name__
            wrapper.__module__ = check_func.__module__
            cls._rules[name] = wrapper
            return wrapper

        return decorator

    @classmethod
    def check_names(cls, names: list[str]) -> list[str]:
        """Checks if the give names are registered in the registry.

        Args:
            names: A list of rule names to check.

        Returns:
            A list of names that are registered in the registry.

        Raises:
            ValueError: If any of the names are not registered.
        """
        if not names:
            raise ValueError("No rule names provided.")

        unknown_names = [name for name in names if name not in cls._rules]

        if unknown_names:
            raise ValueError(f"Rules {', '.join(unknown_names)} are not registered.")
        return names

    @classmethod
    def get_rules(cls, names: list[str]) -> list[ValidationRule]:
        """Returns a list of validation rules for the given names."""
        valid_names = cls.check_names(names)
        return [cls._rules[name] for name in valid_names]

    @classmethod
    def get_rules_except(cls, names_to_exclude: list[str]) -> list[ValidationRule]:
        """Returns all validation rules except for the ones specified."""
        valid_names = cls.check_names(names_to_exclude)
        return [rule for name, rule in cls._rules.items() if name not in valid_names]

    @classmethod
    def get_all_rules(cls) -> list[ValidationRule]:
        """Returns a list of all registered validation rules."""
        return list(cls._rules.values())


@ValidationRuleRegistry.register(
    "encut-not-set",
    ValidationLevel.ERROR,
    "ENCUT is not set. You'd better set it to get consistent results",
)
def encut_not_set(incar: Incar) -> bool:
    """Checks if ENCUT is set in the INCAR file."""
    return "ENCUT" not in incar


@ValidationRuleRegistry.register(
    "parallel-settings-missing",
    ValidationLevel.SUGGESTION,
    "NCORE or NSIM is not set. Setting them can save your time and money.",
)
def parallel_not_set(incar: Incar) -> bool:
    """Checks if NCORE or NSIM is set or not."""
    return "NCORE" not in incar and "NSIM" not in incar


@ValidationRuleRegistry.register(
    "hse-with-optimization",
    ValidationLevel.WARNING,
    "Do not run ionic optimization with HSE functionals. "
    "You have no such time and money to waste.",
)
def no_hse_opt(incar: Incar) -> bool:
    """Checks if HSE06 is used with IBRION != -1, which is not recommended."""
    hybrid = incar.get("LHFCALC", False)
    ibrion = incar.get("IBRION", -1)
    return hybrid and ibrion != -1


@ValidationRuleRegistry.register(
    "hse-symmetry-incompatible",
    ValidationLevel.ERROR,
    "With hybrid functionals (LHFCALC=True), ISYM should be -1, 0, or 3",
)
def hse_sym(incar: Incar) -> bool:
    """Checks if symmetry is set right when using hybrid functionals"""
    hybrid = incar.get("LHFCALC", False)
    isym = incar.get("ISYM", None)
    if isym is None:
        return False
    return hybrid and isym not in [-1, 0, 3]


@ValidationRuleRegistry.register(
    "potim-too-large",
    ValidationLevel.WARNING,
    "POTIM is set to {potim} for IBRION = {ibrion}. "
    "It is recommended to use POTIM <= 0.5 for stable ionic relaxation.",
)
def opt_potim(incar: Incar) -> dict[str, Any] | bool:
    """Checks if POTIM is set right for ionic relaxation."""
    ibrion = incar.get("IBRION", -1)
    potim = incar.get("POTIM", 0.5)
    if ibrion in [1, 2, 3] and (potim > 0.5):
        return {"potim": potim, "ibrion": ibrion}
    return False


@ValidationRuleRegistry.register(
    "phonon-lreal",
    ValidationLevel.WARNING,
    "When doing Phonon calculation, LREAL should be False",
)
def pho_lreal(incar: Incar) -> bool:
    """When calculating phonon, LREAL should be set to False"""
    ibrion_pho = [5, 6, 7, 8]
    lreal_valid = ["false", None]
    ibrion = incar.get("IBRION", None)
    lreal = incar.get("LREAL", None)
    return ibrion in ibrion_pho and lreal not in lreal_valid


@ValidationRuleRegistry.register(
    "phonon-nfree",
    ValidationLevel.SUGGESTION,
    "When doing phonon calculation with finite difference method, "
    "you'd better set NFREE to 4 to make your results robust.",
)
def pho_nfree(incar: Incar) -> bool:
    """Using finite diff method, it's recommended to set NFREE"""
    ibrion_pho_diff = [5, 6]
    ibrion = incar.get("IBRION", -1)
    nfree = incar.get("NFREE", None)
    return ibrion in ibrion_pho_diff and nfree != 4
