"""Tests for :class:`ModelFactory` and :class:`ModelStrategy`."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
import pytest

from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import (
    AmbiguousModelAliasError,
    IncompatibleModelError,
    InvalidModelParameterError,
    MissingDependencyError,
    UnknownModelError,
    ValidationError,
)
from aidatasetkit.core.types import (
    Backend,
    Estimator,
    Interpretability,
    TargetProfile,
    TaskType,
)
from tests.conftest import (
    BUILT_IN_CLASSIFIERS,
    BUILT_IN_CLUSTERERS,
    BUILT_IN_REGRESSORS,
)

from aidatasetkit.models import (
    ModelCapabilities,
    ModelFactory,
    ModelRegistry,
    ModelStrategy,
)

BINARY_TARGET = TargetProfile(
    task_type=TaskType.CLASSIFICATION, n_classes=2, is_binary=True
)
MULTICLASS_TARGET = TargetProfile(task_type=TaskType.CLASSIFICATION, n_classes=4)
REGRESSION_TARGET = TargetProfile(task_type=TaskType.REGRESSION)


class BinaryOnlyStrategy(ModelStrategy):
    """A classifier that cannot do multiclass, which neither built-in model models."""

    name: ClassVar[str] = "binary_only_classifier"
    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        supports_predict_proba=True,
        requires_scaling=False,
        supports_sparse_input=True,
        supports_multiclass=False,
        handles_missing_values=False,
        interpretability_level=Interpretability.HIGH,
    )

    def build(self, **params: Any):
        raise NotImplementedError("compatibility tests never construct estimators")


class TestConstruction:
    def test_a_canonical_name_produces_an_estimator(self):
        estimator = ModelFactory.create("logistic_regression")
        assert isinstance(estimator, Estimator)
        assert type(estimator).__name__ == "LogisticRegression"

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("logistic", "LogisticRegression"),
            ("logreg", "LogisticRegression"),
            ("ridge", "Ridge"),
            ("gnb", "GaussianNB"),
        ],
    )
    def test_an_unshared_alias_produces_the_right_estimator(self, alias, expected):
        assert type(ModelFactory.create(alias)).__name__ == expected

    @pytest.mark.parametrize(
        ("alias", "task", "expected"),
        [
            ("dummy", "classification", "DummyClassifier"),
            ("dummy", "regression", "DummyRegressor"),
            ("baseline", "classification", "DummyClassifier"),
            ("baseline", "regression", "DummyRegressor"),
            ("random_forest", "classification", "RandomForestClassifier"),
            ("random_forest", "regression", "RandomForestRegressor"),
            ("knn", "classification", "KNeighborsClassifier"),
            ("knn", "regression", "KNeighborsRegressor"),
        ],
    )
    def test_a_shared_alias_needs_the_task_to_settle_it(self, alias, task, expected):
        """Since S6 both families answer to the same short names, narrowed by task."""
        assert type(ModelFactory.create(alias, task=task)).__name__ == expected

    @pytest.mark.parametrize("alias", ["dummy", "baseline", "random_forest", "knn"])
    def test_and_the_bare_form_refuses_to_guess(self, alias):
        """The registry names both candidates rather than choosing one."""
        with pytest.raises(AmbiguousModelAliasError, match="Pass task="):
            ModelFactory.create(alias)

    def test_every_call_returns_a_new_instance(self):
        first = ModelFactory.create("dummy_classifier")
        second = ModelFactory.create("dummy_classifier")
        assert first is not second

    def test_the_estimator_comes_back_unfitted(self):
        assert not hasattr(ModelFactory.create("logistic_regression"), "classes_")

    def test_a_strategy_can_be_obtained_without_building(self):
        strategy = ModelFactory.strategy("logistic_regression")
        assert isinstance(strategy, ModelStrategy)
        assert strategy.name == "logistic_regression"
        assert strategy.capabilities.requires_scaling

    def test_a_registration_can_be_obtained_without_constructing(self):
        entry = ModelFactory.registration("logreg")
        assert entry.canonical_name == "logistic_regression"

    def test_the_strategy_repr_names_the_model(self):
        assert "logistic_regression" in repr(ModelFactory.strategy("logistic_regression"))


class TestParameters:
    def test_explicit_parameters_reach_the_estimator(self):
        estimator = ModelFactory.create("logistic_regression", C=0.5, max_iter=500)
        params = estimator.get_params()
        assert params["C"] == 0.5
        assert params["max_iter"] == 500

    def test_defaults_apply_when_nothing_is_supplied(self):
        assert ModelFactory.create("logistic_regression").get_params()["max_iter"] == 1000

    def test_an_override_replaces_only_the_named_default(self):
        params = ModelFactory.create("logistic_regression", C=0.25).get_params()
        assert params["C"] == 0.25
        assert params["max_iter"] == 1000

    def test_an_unknown_parameter_is_refused_and_lists_what_is_accepted(self):
        with pytest.raises(InvalidModelParameterError) as error:
            ModelFactory.create("logistic_regression", nonsense=1)
        message = str(error.value)
        assert "nonsense" in message
        assert "max_iter" in message

    def test_no_parameter_is_silently_discarded(self):
        with pytest.raises(InvalidModelParameterError):
            ModelFactory.create("dummy_classifier", learning_rate=0.1)

    def test_an_unrelated_type_error_is_not_relabelled_as_a_bad_parameter(self):
        """A missing required argument is not an unknown-parameter error.

        Catching every TypeError produced the self-contradictory message
        "does not accept the parameter(s) []" and discarded the real cause.
        """

        class NeedsAnArgument:
            def __init__(self, required):
                self.required = required

        with pytest.raises(TypeError, match="required"):
            ModelStrategy._construct(NeedsAnArgument, {}, "needy")

    def test_an_unknown_parameter_is_still_translated(self):
        class TakesOne:
            def __init__(self, alpha=1.0):
                self.alpha = alpha

        with pytest.raises(InvalidModelParameterError, match="beta"):
            ModelStrategy._construct(TakesOne, {"beta": 2.0}, "takes_one")

    def test_an_estimator_accepting_any_keyword_is_left_alone(self):
        """Nothing can be called unknown when the signature accepts everything."""

        class TakesAnything:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        built = ModelStrategy._construct(TakesAnything, {"whatever": 1}, "anything")
        assert built.kwargs == {"whatever": 1}

    def test_an_invalid_value_surfaces_from_the_estimator_at_fit_time(self):
        """scikit-learn validates values in fit, not in __init__; that is left alone."""
        from sklearn.utils._param_validation import InvalidParameterError

        estimator = ModelFactory.create("logistic_regression", C=-1)
        with pytest.raises(InvalidParameterError):
            estimator.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))

    def test_the_configured_seed_reaches_the_estimator(self):
        estimator = ModelFactory.create(
            "logistic_regression", config=KitConfig(random_state=7)
        )
        assert estimator.get_params()["random_state"] == 7

    def test_an_explicit_seed_beats_the_configured_one(self):
        estimator = ModelFactory.create(
            "logistic_regression", config=KitConfig(random_state=7), random_state=99
        )
        assert estimator.get_params()["random_state"] == 99


class TestTaskValidation:
    def test_a_wrong_task_target_fails_before_any_fitting(self):
        with pytest.raises(IncompatibleModelError, match="classification tasks"):
            ModelFactory.create("logistic_regression", target=REGRESSION_TARGET)

    def test_the_check_happens_at_construction_not_at_fit(self):
        """The point is to fail here rather than inside sklearn several steps later."""
        with pytest.raises(IncompatibleModelError):
            ModelFactory.strategy("dummy_classifier", target=REGRESSION_TARGET)

    def test_a_matching_target_is_accepted(self):
        assert ModelFactory.create("logistic_regression", target=BINARY_TARGET) is not None

    def test_a_multiclass_target_is_accepted_by_every_classifier(self):
        for name in ModelFactory.available(task=TaskType.CLASSIFICATION):
            assert ModelFactory.create(name, target=MULTICLASS_TARGET) is not None

    def test_compatible_models_can_be_listed_for_a_target(self):
        assert set(ModelFactory.compatible_with(BINARY_TARGET)) == set(
            BUILT_IN_CLASSIFIERS
        )

    def test_a_regression_target_selects_the_regressors_and_only_those(self):
        assert set(ModelFactory.compatible_with(REGRESSION_TARGET)) == set(
            BUILT_IN_REGRESSORS
        )

    def test_the_families_do_not_overlap(self):
        """Task separation, asked of the public selection surface.

        Every classifier is unreachable through a regression target and every
        regressor through a classification one. If a model were registered in the
        wrong family this is where it would show.

        The clusterers are unreachable through *either*, which is the stronger
        statement S9 added: a target of any kind excludes all six, because
        `compatible_with` filters on the target's task family and no target ever
        resolves to clustering.
        """
        classifiers = set(ModelFactory.compatible_with(BINARY_TARGET))
        regressors = set(ModelFactory.compatible_with(REGRESSION_TARGET))
        assert not (classifiers & regressors)
        assert not (classifiers | regressors) & set(BUILT_IN_CLUSTERERS)
        assert classifiers | regressors | set(BUILT_IN_CLUSTERERS) == set(
            ModelFactory.available()
        )

    def test_listing_compatible_models_ranks_nothing(self):
        """A filter, not a recommendation."""
        assert list(ModelFactory.compatible_with(BINARY_TARGET)) == sorted(
            ModelFactory.compatible_with(BINARY_TARGET)
        )

    def test_a_binary_only_model_is_excluded_from_a_multiclass_target(self):
        """Exercises the capability filter, not just the task filter.

        With only all-multiclass models registered, deleting the capability check
        from ``compatible_with`` changes no result. A binary-only classifier is
        the case that makes it load-bearing.
        """
        registry = ModelRegistry()
        registry.register(BinaryOnlyStrategy)
        assert ModelFactory.compatible_with(BINARY_TARGET, registry=registry) == (
            "binary_only_classifier",
        )
        assert ModelFactory.compatible_with(MULTICLASS_TARGET, registry=registry) == ()

    def test_the_capability_filter_is_not_merely_the_task_filter(self):
        registry = ModelRegistry()
        registry.register(BinaryOnlyStrategy)
        assert registry.available(task=TaskType.CLASSIFICATION) == (
            "binary_only_classifier",
        )
        assert ModelFactory.compatible_with(MULTICLASS_TARGET, registry=registry) == ()

    def test_a_binary_only_model_refuses_a_multiclass_target_at_construction(self):
        registry = ModelRegistry()
        registry.register(BinaryOnlyStrategy)
        with pytest.raises(IncompatibleModelError, match="two classes only"):
            ModelFactory.strategy(
                "binary_only_classifier", target=MULTICLASS_TARGET, registry=registry
            )


class TestErrorCases:
    def test_an_unknown_model_is_refused(self):
        with pytest.raises(UnknownModelError, match="No model is registered"):
            ModelFactory.create("magic_classifier")

    def test_a_near_miss_is_suggested(self):
        with pytest.raises(UnknownModelError, match="Did you mean"):
            ModelFactory.create("logistic_regresion")

    def test_an_ambiguous_alias_is_refused_rather_than_guessed(self, scratch_factory):
        registry, _ = scratch_factory
        with pytest.raises(AmbiguousModelAliasError, match="Nothing is chosen"):
            ModelFactory.create("shared", registry=registry)

    def test_task_context_resolves_the_ambiguous_alias(self, scratch_factory):
        registry, _ = scratch_factory
        estimator = ModelFactory.create(
            "shared", task=TaskType.CLASSIFICATION, registry=registry
        )
        assert type(estimator).__name__ == "DummyClassifier"

    def test_a_target_alone_resolves_the_ambiguous_alias(self, scratch_factory):
        """A caller who supplied a target has already said which family they mean."""
        registry, _ = scratch_factory
        estimator = ModelFactory.create("shared", target=BINARY_TARGET, registry=registry)
        assert type(estimator).__name__ == "DummyClassifier"

    def test_a_canonical_name_contradicted_by_task_is_refused(self):
        """Gate point 11: the wrong-task failure must not wait for fit()."""
        with pytest.raises(IncompatibleModelError, match="serves classification"):
            ModelFactory.create("logistic_regression", task="regression")

    def test_an_invalid_task_string_is_rejected(self):
        with pytest.raises(ValidationError, match="not a valid TaskType"):
            ModelFactory.create("logistic_regression", task="not_a_task")

    def test_a_missing_optional_dependency_explains_how_to_install_it(self, scratch_factory):
        registry, _ = scratch_factory
        with pytest.raises(MissingDependencyError, match="pip install"):
            ModelFactory.create("fictional_classifier", registry=registry)

    def test_the_missing_dependency_names_the_package(self, scratch_factory):
        registry, _ = scratch_factory
        with pytest.raises(MissingDependencyError, match="not_a_real_package"):
            ModelFactory.strategy("fictional_classifier", registry=registry)

    def test_an_unavailable_model_is_absent_from_the_default_listing(self, scratch_factory):
        registry, _ = scratch_factory
        assert "fictional_classifier" not in ModelFactory.available(registry=registry)
        assert "fictional_classifier" in ModelFactory.available(
            include_unavailable=True, registry=registry
        )


class TestDiscovery:
    def test_available_lists_the_built_in_models(self):
        assert ModelFactory.available() == tuple(
            sorted(BUILT_IN_CLASSIFIERS + BUILT_IN_REGRESSORS + BUILT_IN_CLUSTERERS)
        )

    def test_available_filters_by_task(self):
        assert ModelFactory.available(task="regression") == BUILT_IN_REGRESSORS
        assert ModelFactory.available(task=TaskType.CLASSIFICATION) == (
            BUILT_IN_CLASSIFIERS
        )
        assert ModelFactory.available(task="clustering") == BUILT_IN_CLUSTERERS

    def test_the_three_task_filters_partition_the_catalog(self):
        classification = ModelFactory.available(task=TaskType.CLASSIFICATION)
        regression = ModelFactory.available(task=TaskType.REGRESSION)
        clustering = ModelFactory.available(task=TaskType.CLUSTERING)
        assert not (set(classification) & set(regression))
        assert not (set(classification) & set(clustering))
        assert not (set(regression) & set(clustering))
        assert set(classification) | set(regression) | set(clustering) == set(
            ModelFactory.available()
        )

    def test_the_catalog_carries_structured_metadata(self):
        entries = {entry["canonical_name"]: entry for entry in ModelFactory.catalog()}
        logistic = entries["logistic_regression"]
        assert logistic["aliases"] == ["logistic", "logreg"]
        assert logistic["task_type"] == "classification"
        assert logistic["backend"] == "sklearn"
        assert logistic["available"] is True
        assert logistic["requires_package"] is None
        assert logistic["is_baseline"] is False
        assert logistic["capabilities"]["requires_scaling"] is True

    def test_the_catalog_is_json_serialisable(self):
        payload = json.loads(json.dumps(ModelFactory.catalog()))
        assert len(payload) == (
            len(BUILT_IN_CLASSIFIERS)
            + len(BUILT_IN_REGRESSORS)
            + len(BUILT_IN_CLUSTERERS)
        )

    def test_the_catalog_holds_no_estimator_objects(self):
        for entry in ModelFactory.catalog():
            assert all(
                not hasattr(value, "fit") for value in entry.values()
            ), "an estimator leaked into the catalog"

    def test_the_registry_is_the_only_source_of_the_model_list(self):
        from aidatasetkit.models import default_registry

        assert ModelFactory.available() == default_registry().available()

    def test_the_catalog_does_not_rank_or_recommend(self):
        names = [entry["canonical_name"] for entry in ModelFactory.catalog()]
        assert names == sorted(names)


class TestRegistryInjection:
    def test_the_factory_works_against_an_isolated_registry(self, scratch_factory):
        """The scratch registry holds its own models, not the built-in set."""
        registry, _ = scratch_factory
        assert "logistic_regression" not in ModelFactory.available(registry=registry)
        assert "fake_regressor" in ModelFactory.available(registry=registry)
        assert "fake_regressor" not in ModelFactory.available()

    def test_the_default_registry_is_used_when_none_is_given(self):
        assert "logistic_regression" in ModelFactory.available()


@pytest.fixture
def scratch_factory():
    """A registry holding an ambiguous alias and an uninstallable model."""
    from aidatasetkit.models.classification.dummy import DummyClassifierStrategy

    class FakeRegressor(ModelStrategy):
        name: ClassVar[str] = "fake_regressor"
        capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
            task_type=TaskType.REGRESSION,
            backend=Backend.SKLEARN,
            supports_predict_proba=False,
            requires_scaling=False,
            supports_sparse_input=True,
            supports_multiclass=False,
            handles_missing_values=False,
            interpretability_level=Interpretability.HIGH,
        )

        def build(self, **params: Any):
            raise NotImplementedError

    class FictionalClassifier(ModelStrategy):
        name: ClassVar[str] = "fictional_classifier"
        capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
            task_type=TaskType.CLASSIFICATION,
            backend=Backend.XGBOOST,
            supports_predict_proba=True,
            requires_scaling=False,
            supports_sparse_input=True,
            supports_multiclass=True,
            handles_missing_values=True,
            interpretability_level=Interpretability.LOW,
        )

        def build(self, **params: Any):
            raise NotImplementedError

    registry = ModelRegistry()
    registry.register(DummyClassifierStrategy, aliases=("shared",))
    registry.register(FakeRegressor, aliases=("shared",))
    registry.register(FictionalClassifier, requires_package="not_a_real_package")
    return registry, FictionalClassifier
