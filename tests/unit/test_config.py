"""Tests for the central configuration object."""

from __future__ import annotations

import json

import pytest

from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import ConfigurationError


class TestDefaults:
    def test_defaults_are_usable(self):
        config = KitConfig()
        assert config.random_state == 42
        assert 0.0 < config.validation_size < 1.0

    def test_config_is_immutable(self):
        config = KitConfig()
        with pytest.raises(AttributeError):
            config.random_state = 7

    def test_to_dict_is_json_serialisable(self):
        payload = json.dumps(KitConfig().to_dict())
        assert "random_state" in payload


class TestValidation:
    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
    def test_validation_size_must_be_a_proper_fraction(self, value):
        with pytest.raises(ConfigurationError, match="validation_size"):
            KitConfig(validation_size=value)


    @pytest.mark.parametrize(
        "field",
        [
            "missing_warning_threshold",
            "near_constant_threshold",
            "id_uniqueness_threshold",
            "leakage_correlation_threshold",
            "task_detection_unique_ratio",
        ],
    )
    def test_ratio_fields_are_bounded(self, field):
        with pytest.raises(ConfigurationError, match=field):
            KitConfig(**{field: 1.5})

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_outlier_multiplier_must_be_positive(self, value):
        with pytest.raises(ConfigurationError, match="outlier_iqr_multiplier"):
            KitConfig(outlier_iqr_multiplier=value)

    @pytest.mark.parametrize("value", [1.0, 0.5, -3.0])
    def test_multicollinearity_threshold_must_exceed_one(self, value):
        with pytest.raises(ConfigurationError, match="multicollinearity_vif_threshold"):
            KitConfig(multicollinearity_vif_threshold=value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_multicollinearity_threshold_must_be_finite(self, value):
        """NaN would flag every column; infinity would silently flag none."""
        with pytest.raises(ConfigurationError, match="must be finite"):
            KitConfig(multicollinearity_vif_threshold=value)

    @pytest.mark.parametrize("value", ["10", True, None])
    def test_multicollinearity_threshold_must_be_a_number(self, value):
        with pytest.raises(ConfigurationError, match="must be a number"):
            KitConfig(multicollinearity_vif_threshold=value)

    def test_multicollinearity_threshold_just_above_one_is_accepted(self):
        import math

        value = math.nextafter(1.0, 2.0)
        assert KitConfig(multicollinearity_vif_threshold=value).multicollinearity_vif_threshold == value

    def test_an_integer_threshold_is_accepted(self):
        assert KitConfig(multicollinearity_vif_threshold=5).multicollinearity_vif_threshold == 5

    def test_task_detection_needs_at_least_two_classes(self):
        with pytest.raises(ConfigurationError, match="task_detection_max_classes"):
            KitConfig(task_detection_max_classes=1)

    @pytest.mark.parametrize("value", ["", "   ", None, 3])
    def test_output_column_names_must_be_non_empty_strings(self, value):
        with pytest.raises(ConfigurationError, match="probability_column"):
            KitConfig(probability_column=value)


class TestReplace:
    def test_replace_returns_a_new_validated_config(self):
        original = KitConfig()
        updated = original.replace(high_cardinality_threshold=10)
        assert updated.high_cardinality_threshold == 10
        assert original.high_cardinality_threshold == 50

    def test_replace_revalidates(self):
        with pytest.raises(ConfigurationError):
            KitConfig().replace(validation_size=1.5)

    def test_replace_rejects_unknown_options(self):
        with pytest.raises(ConfigurationError, match="Unknown configuration options"):
            KitConfig().replace(learning_rate=0.1)


class TestCvFoldsIsGone:
    """G0-06: a setting documented as steering cross-validation, which nothing ran.

    ``cv_folds`` was declared, validated and recorded in every artifact's config
    fingerprint, and no code path read it: there is no cross-validation in the
    package. A configured value that changes nothing is the silent failure this
    library refuses elsewhere, so it is removed rather than left to be believed.
    The fingerprint migration is recorded in test_capability_fingerprint_migration.
    """

    def test_it_is_not_a_field(self):
        assert "cv_folds" not in KitConfig().to_dict()

    def test_constructing_with_it_is_refused(self):
        with pytest.raises(TypeError):
            KitConfig(cv_folds=5)

    def test_replacing_it_is_refused_by_name(self):
        with pytest.raises(ConfigurationError, match="cv_folds"):
            KitConfig().replace(cv_folds=5)

    def test_nothing_in_the_package_mentions_it(self):
        from pathlib import Path

        package = Path(__file__).resolve().parents[2] / "aidatasetkit"
        offenders = [
            str(path.relative_to(package))
            for path in package.rglob("*.py")
            if "cv_folds" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []
