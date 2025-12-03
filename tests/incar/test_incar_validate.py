"""Test module for INCAR validator."""

import re

import pytest

from vaspin import Incar
from vaspin.incar import (
    IncarValidator,
    ValidationLevel,
    ValidationResult,
    ValidationRuleRegistry,
)
from vaspin.incar.rules import (
    encut_not_set,
    hse_sym,
    no_hse_opt,
    opt_potim,
    parallel_not_set,
    pho_lreal,
    pho_nfree,
)


@pytest.fixture(scope="function")
def all_rules():
    """Fixture to provide all validation rules for testing."""
    return {
        "encut-not-set": encut_not_set,
        "parallel-settings-missing": parallel_not_set,
        "hse-with-optimization": no_hse_opt,
        "hse-symmetry-incompatible": hse_sym,
        "potim-too-large": opt_potim,
        "phonon-lreal": pho_lreal,
        "phonon-nfree": pho_nfree,
    }


class TestIncarRulesSingle:
    """Test class for single INCAR validation rules."""

    def test_encut_not_set(self):
        """Test the 'encut-not-set' validation rule."""
        incar = Incar({"ISMEAR": 0})
        validator = IncarValidator(include_rules=["encut-not-set"])
        result = validator.validate(incar)
        expected_result = ValidationResult(
            level=ValidationLevel.ERROR,
            message="ENCUT is not set. You'd better set it to get consistent results",
            name="encut-not-set",
        )
        assert result.errors == [expected_result]

        incar["ENCUT"] = 520
        result = validator.validate(incar)
        assert result.errors == []

    def test_parallel_setting_missing(self):
        """Test the 'parallel-settings-missing' validation rule."""
        incar = Incar({"ENCUT": 520})
        validator = IncarValidator(include_rules=["parallel-settings-missing"])
        result = validator.validate(incar)
        expected_result = ValidationResult(
            level=ValidationLevel.SUGGESTION,
            message=(
                "NCORE or NSIM is not set. Setting them can save your time and money."
            ),
            name="parallel-settings-missing",
        )
        assert result.suggestions == [expected_result]

        incar["NCORE"] = 4
        incar["NSIM"] = 16
        result = validator.validate(incar)
        assert result.suggestions == []

    def test_hse_with_opt(self):
        """Test the 'hse-with-optimization' validation rule."""
        incar = Incar({"IBRION": 2, "ENCUT": 520, "LHFCALC": True})
        validator = IncarValidator(include_rules=["hse-with-optimization"])
        result = validator.validate(incar)
        expected_result = ValidationResult(
            level=ValidationLevel.WARNING,
            message=(
                "Do not run ionic optimization with HSE functionals. "
                "You have no such time and money to waste."
            ),
            name="hse-with-optimization",
        )
        assert result.warnings == [expected_result]
        incar["LHFCALC"] = False
        result = validator.validate(incar)
        assert result.warnings == []

    @pytest.mark.parametrize("isym", [1, 2])
    def test_hse_symmetry_incompatible(self, isym):
        """Test the 'hse-symmetry-incompatible' validation rule."""
        incar = Incar({"LHFCALC": True, "ISYM": isym})
        validator = IncarValidator(include_rules=["hse-symmetry-incompatible"])
        result = validator.validate(incar)
        expected_result = ValidationResult(
            level=ValidationLevel.ERROR,
            message=(
                "With hybrid functionals (LHFCALC=True), ISYM should be -1, 0, or 3"
            ),
            name="hse-symmetry-incompatible",
        )
        assert result.errors == [expected_result]

    @pytest.mark.parametrize("isym", [-1, 0, 3, None])
    def test_hse_symmetry_compatible(self, isym):
        """Test the 'hse-symmetry-incompatible' validation rule for compatible ISYM."""
        tags = {"LHFCALC": True}
        if isym is not None:
            tags["ISYM"] = isym
        incar = Incar(tags)
        validator = IncarValidator(include_rules=["hse-symmetry-incompatible"])
        result = validator.validate(incar)

        assert result.errors == []

    def test_potim_too_large(self):
        """Test the 'potim-too-large' validation rule."""
        potim, ibrion = (1.0, 2)
        incar = Incar({"IBRION": 2, "POTIM": 1.0})
        validator = IncarValidator(include_rules=["potim-too-large"])
        result = validator.validate(incar)
        expected_result = ValidationResult(
            level=ValidationLevel.WARNING,
            message=(
                f"POTIM is set to {potim} for IBRION = {ibrion}. "
                "It is recommended to use POTIM <= 0.5 for stable ionic relaxation."
            ),
            name="potim-too-large",
        )
        assert result.warnings == [expected_result]

        incar["POTIM"] = 0.4
        result = validator.validate(incar)
        assert result.warnings == []

    def test_phonon_lreal(self):
        """Test the 'phonon-lreal' validation rule."""
        incar = Incar({"IBRION": 5, "LREAL": "true"})
        validator = IncarValidator(include_rules=["phonon-lreal"])
        result = validator.validate(incar)
        expected_result = ValidationResult(
            level=ValidationLevel.WARNING,
            message="When doing Phonon calculation, LREAL should be False",
            name="phonon-lreal",
        )
        assert result.warnings == [expected_result]

        incar["LREAL"] = "false"
        result = validator.validate(incar)
        assert result.warnings == []

    @pytest.mark.parametrize("nfree", [1, 2])
    def test_phonon_nfree(self, nfree):
        """Test the 'phonon-nfree' validation rule."""
        incar = Incar({"IBRION": 5, "NFREE": nfree})
        validator = IncarValidator(include_rules=["phonon-nfree"])
        result = validator.validate(incar)
        expected_result = ValidationResult(
            level=ValidationLevel.SUGGESTION,
            message=(
                "When doing phonon calculation with finite difference method, "
                "you'd better set NFREE to 4 to make your results robust."
            ),
            name="phonon-nfree",
        )
        assert result.suggestions == [expected_result]

        incar["NFREE"] = 4
        result = validator.validate(incar)
        assert result.suggestions == []


class TestRegistry:
    """Test for the registry class"""

    def test_check_names_empty(self):
        """Test behavior when no rule names are provided."""
        with pytest.raises(ValueError, match=re.escape("No rule names provided.")):
            ValidationRuleRegistry.check_names([])

    def test_check_names_invalid(self):
        """Test behavior when invalid rule names are provided."""
        with pytest.raises(
            ValueError, match=re.escape("Rules noregistered are not registered.")
        ):
            ValidationRuleRegistry.check_names(["noregistered"])

    @pytest.mark.parametrize(
        "rule_names, expected_results",
        [
            (["encut-not-set"], [encut_not_set]),
            (["parallel-settings-missing"], [parallel_not_set]),
            (["hse-with-optimization"], [no_hse_opt]),
            (["hse-symmetry-incompatible"], [hse_sym]),
            (["potim-too-large"], [opt_potim]),
            (["encut-not-set", "hse-symmetry-incompatible"], [encut_not_set, hse_sym]),
            (
                ["encut-not-set", "parallel-settings-missing", "potim-too-large"],
                [encut_not_set, parallel_not_set, opt_potim],
            ),
        ],
    )
    def test_get_rules(self, rule_names, expected_results):
        """Test the get_rules method of ValidationRuleRegistry."""
        rules = ValidationRuleRegistry.get_rules(rule_names)
        assert rules == expected_results

    def test_get_all_rules(self, all_rules):
        """Test the get_all_rules method of ValidationRuleRegistry."""
        rules = ValidationRuleRegistry.get_all_rules()
        assert set(rules) == set(all_rules.values())

    @pytest.mark.parametrize(
        "except_rules",
        [
            ["encut-not-set", "hse-symmetry-incompatible"],
            ["phonon-lreal", "potim-too-large"],
            ["encut-not-set", "hse-symmetry-incompatible", "potim-too-large"],
        ],
    )
    def test_get_rules_except(self, all_rules, except_rules):
        """Test the get_rules_except method of ValidationRuleRegistry."""
        expected_rules = [
            rule for name, rule in all_rules.items() if name not in except_rules
        ]
        rules = ValidationRuleRegistry.get_rules_except(except_rules)
        assert set(rules) == set(expected_rules)


class TestIncarValidator:
    """Test class for IncarValidator."""

    def test_init_validator_error(self):
        """Initialize IncarValidator with both include and exclude rules"""
        with pytest.raises(
            ValueError,
            match=re.escape("Cannot specify both include_rules and exclude_rules."),
        ):
            IncarValidator(
                include_rules=["encut-not-set"],
                exclude_rules=["parallel-settings-missing"],
            )

    @pytest.mark.parametrize(
        "include_rules",
        [
            ["encut-not-set"],
            ["parallel-settings-missing"],
            ["hse-with-optimization"],
            ["hse-symmetry-incompatible"],
            ["potim-too-large"],
            ["phonon-lreal"],
            ["phonon-nfree"],
        ],
    )
    def test_init_with_include(self, include_rules, all_rules):
        """Initialize IncarValidator with include_rules."""
        validator = IncarValidator(include_rules=include_rules)
        assert set(validator._rules) == {all_rules[name] for name in include_rules}

    @pytest.mark.parametrize(
        "exclude_rules",
        [
            ["encut-not-set"],
            ["parallel-settings-missing"],
            ["hse-with-optimization"],
            ["hse-symmetry-incompatible"],
            ["potim-too-large"],
            ["phonon-lreal"],
            ["phonon-nfree"],
            ["encut-not-set", "hse-symmetry-incompatible"],
            ["phonon-lreal", "potim-too-large"],
            ["encut-not-set", "hse-symmetry-incompatible", "potim-too-large"],
        ],
    )
    def test_init_with_exclude(self, all_rules, exclude_rules):
        """Initialize IncarValidator with exclude_rules."""
        expected_rules = [
            rule for name, rule in all_rules.items() if name not in exclude_rules
        ]
        validator = IncarValidator(exclude_rules=exclude_rules)
        assert set(validator._rules) == set(expected_rules)

    def test_init_without_para(self, all_rules):
        """Initialize IncarValidator without any parameters."""
        validator = IncarValidator()
        assert set(validator._rules) == set(all_rules.values())
