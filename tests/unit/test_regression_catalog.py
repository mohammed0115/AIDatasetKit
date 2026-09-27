"""The regression catalog: what exists, what it is called, and how it is found.

Nothing here fits anything. These are the questions a user asks before choosing a
model -- what can this library do, what may I type, and will the answer be the
same tomorrow -- and the answers come from the registry alone.

The shape mirrors ``test_classification_catalog.py`` deliberately. Regression is
not a second modelling subsystem, so it is asked the same questions and answers
through the same registry, factory, and catalog.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from aidatasetkit.core.exceptions import (
    AmbiguousModelAliasError,
    IncompatibleModelError,
    InvalidModelParameterError,
)
from aidatasetkit.core.types import (
    Backend,
    Estimator,
    PreprocessingProfile,
    TaskType,
)
from aidatasetkit.models import ModelFactory, default_registry

from tests.conftest import (
    BUILT_IN_CLASSIFIERS,
    BUILT_IN_CLUSTERERS,
    BUILT_IN_REGRESSORS,
    isolated_env,
)

#: The approved catalog, named once in conftest and asserted here.
EXPECTED = BUILT_IN_REGRESSORS

#: Canonical name -> the concise alias it answers to.
#:
#: Seven of the nine reuse the alias their classification twin already carries,
#: which is the arrangement the registry was designed for and which
#: ``TestAliasesSurviveTheArrivalOfRegression`` anticipated before any of these
#: models existed. ``linear_regression`` gets none: it is already the short,
#: ordinary name, and an alias restating it would occupy the namespace for
#: nothing.
EXPECTED_ALIASES = {
    "decision_tree_regressor": ("decision_tree",),
    "dummy_regressor": ("dummy", "baseline"),
    "extra_trees_regressor": ("extra_trees",),
    "gradient_boosting_regressor": ("gradient_boosting",),
    "hist_gradient_boosting_regressor": ("hist_gradient_boosting",),
    "knn_regressor": ("knn",),
    "linear_regression": (),
    "random_forest_regressor": ("random_forest",),
    "ridge_regression": ("ridge",),
}


class TestTheCatalogHoldsWhatWasApproved:
    def test_it_contains_exactly_nine_regressors(self):
        assert len(ModelFactory.available(task="regression")) == 9

    def test_it_contains_exactly_the_approved_models(self):
        assert ModelFactory.available(task="regression") == EXPECTED

    def test_every_regressor_is_constructible_here(self):
        for name in EXPECTED:
            assert isinstance(ModelFactory.create(name), Estimator)

    @pytest.mark.parametrize("name", EXPECTED)
    def test_each_one_declares_the_regression_task(self, name):
        assert ModelFactory.registration(name).task_type is TaskType.REGRESSION

    @pytest.mark.parametrize("name", EXPECTED)
    def test_each_one_is_backed_by_scikit_learn(self, name):
        assert ModelFactory.registration(name).backend is Backend.SKLEARN

    def test_exactly_one_model_is_the_baseline(self):
        baselines = [
            entry.canonical_name
            for entry in default_registry().catalog(task="regression")
            if entry.capabilities.is_baseline
        ]
        assert baselines == ["dummy_regressor"]

    def test_no_regressor_claims_class_probabilities(self):
        for entry in default_registry().catalog(task="regression"):
            assert entry.capabilities.supports_predict_proba is False
            assert entry.capabilities.supports_multiclass is False

    def test_no_unsupervised_model_beyond_clustering_arrived(self):
        """S6 was regression, and this gate held until S9 deliberately opened it.

        Clustering was in the list until six clusterers were registered under
        their own executed capability contracts. Anomaly detection and
        dimensionality reduction still belong to a later stage, and a model
        appearing in either without one is what this catches.
        """
        for task in (
            TaskType.ANOMALY_DETECTION,
            TaskType.DIMENSIONALITY_REDUCTION,
        ):
            assert ModelFactory.available(task=task, include_unavailable=True) == ()

    def test_clustering_arrived_as_its_own_family_rather_than_inside_this_one(self):
        """The S6 gate opening must not mean regressors leaked a task type."""
        assert len(ModelFactory.available(task=TaskType.CLUSTERING)) == 6
        assert not set(ModelFactory.available(task=TaskType.CLUSTERING)) & set(EXPECTED)


class TestTaskSeparationIsReal:
    """A regressor must not be reachable as a classifier, or the reverse."""

    def test_the_two_families_are_disjoint(self):
        assert not (set(BUILT_IN_CLASSIFIERS) & set(EXPECTED))

    def test_together_with_clustering_they_are_the_whole_catalog(self):
        assert set(ModelFactory.available()) == (
            set(BUILT_IN_CLASSIFIERS) | set(EXPECTED) | set(BUILT_IN_CLUSTERERS)
        )

    @pytest.mark.parametrize("name", EXPECTED)
    def test_a_regressor_cannot_be_fetched_as_a_classifier(self, name):
        with pytest.raises(IncompatibleModelError, match="serves regression"):
            ModelFactory.registration(name, task="classification")

    @pytest.mark.parametrize("name", BUILT_IN_CLASSIFIERS)
    def test_a_classifier_cannot_be_fetched_as_a_regressor(self, name):
        with pytest.raises(IncompatibleModelError, match="serves classification"):
            ModelFactory.registration(name, task="regression")

    def test_the_filter_is_what_separates_them_rather_than_the_spelling(self):
        """``linear_regression`` ends in "regression" and so does the classifier
        named ``logistic_regression``. Nothing here reads either name."""
        assert (
            ModelFactory.registration("logistic_regression").task_type
            is TaskType.CLASSIFICATION
        )
        assert (
            ModelFactory.registration("linear_regression").task_type
            is TaskType.REGRESSION
        )
        assert "logistic_regression" not in ModelFactory.available(task="regression")

    def test_compatible_with_never_crosses_the_line(self):
        from aidatasetkit.core.types import TargetProfile

        regression = TargetProfile(task_type=TaskType.REGRESSION)
        classification = TargetProfile(
            task_type=TaskType.CLASSIFICATION, n_classes=2, is_binary=True
        )
        assert set(ModelFactory.compatible_with(regression)) == set(EXPECTED)
        assert set(ModelFactory.compatible_with(classification)) == set(
            BUILT_IN_CLASSIFIERS
        )


#: One fresh interpreter, several questions. Each subprocess pays the whole
#: import cost, so asking six questions in six of them wastes most of a minute.
_FRESH_SCRIPT = (
    "import sys;"
    "from aidatasetkit.models import ModelFactory;"
    "print(','.join(ModelFactory.available(task='regression')));"
    "print(len(ModelFactory.available(task='regression')));"
    "print(ModelFactory.create('random_forest', task='regression')"
    ".__class__.__name__);"
    "print(any('regression.forest' in m for m in sys.modules));"
    "print(len(ModelFactory.available()))"
)


def _fresh(seed: str) -> list[str]:
    """Run the script in a new interpreter under ``seed`` and return its answers."""
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        env=isolated_env(seed),
    )
    return result.stdout.strip().splitlines()


@pytest.fixture(scope="module")
def fresh_runs() -> dict[str, list[str]]:
    """The same questions answered under two different hash seeds.

    Two rather than three: each spawn re-imports pandas and scikit-learn and is
    the most expensive item in the suite, and the in-process assertions above
    supply a third independent hash ordering.
    """
    return {seed: _fresh(seed) for seed in ("0", "12345")}


class TestRegistrationNeedsNoManualImport:
    """A user types ModelFactory.create('ridge'). Nothing else."""

    def test_a_fresh_interpreter_sees_every_regressor(self, fresh_runs):
        assert {run[1] for run in fresh_runs.values()} == {"9"}

    def test_importing_only_the_top_level_package_is_enough(self, fresh_runs):
        assert {run[2] for run in fresh_runs.values()} == {"RandomForestRegressor"}

    def test_no_strategy_module_has_to_be_imported_by_hand(self, fresh_runs):
        """Naming a strategy module directly must not be what makes it exist."""
        assert {run[3] for run in fresh_runs.values()} == {"True"}

    def test_both_families_register_together_without_colliding(self, fresh_runs):
        assert {run[4] for run in fresh_runs.values()} == {"24"}

    def test_ordering_does_not_depend_on_the_hash_seed(self, fresh_runs):
        """Import order and dict iteration must not reach the published order."""
        assert {run[0] for run in fresh_runs.values()} == {",".join(EXPECTED)}


class TestOrderingIsDeterministic:
    def test_available_is_sorted_by_canonical_name(self):
        names = ModelFactory.available(task="regression")
        assert list(names) == sorted(names)

    def test_repeated_calls_agree(self):
        assert ModelFactory.available(task="regression") == ModelFactory.available(
            task="regression"
        )

    def test_the_catalog_follows_the_same_order(self):
        catalog = ModelFactory.catalog(task="regression")
        assert [entry["canonical_name"] for entry in catalog] == list(EXPECTED)


class TestAliases:
    @pytest.mark.parametrize("canonical,aliases", sorted(EXPECTED_ALIASES.items()))
    def test_the_registry_reports_exactly_those_aliases(self, canonical, aliases):
        assert default_registry().aliases_of(canonical) == aliases

    @pytest.mark.parametrize(
        "canonical,aliases",
        [(c, a) for c, a in sorted(EXPECTED_ALIASES.items()) if a],
    )
    def test_each_alias_resolves_to_its_model_when_the_task_is_given(
        self, canonical, aliases
    ):
        for alias in aliases:
            assert (
                ModelFactory.registration(alias, task="regression").canonical_name
                == canonical
            )

    def test_linear_regression_deliberately_has_none(self):
        assert default_registry().aliases_of("linear_regression") == ()

    def test_no_alias_shadows_a_canonical_name(self):
        canonical = set(ModelFactory.available())
        for aliases in EXPECTED_ALIASES.values():
            assert not (set(aliases) & canonical)

    def test_the_ridge_alias_is_unshared_and_needs_no_task(self):
        """There is no ridge classifier in the catalog, so nothing narrows it."""
        assert ModelFactory.registration("ridge").canonical_name == "ridge_regression"

    @pytest.mark.parametrize(
        "alias",
        ["decision_tree", "dummy", "baseline", "extra_trees", "gradient_boosting",
         "hist_gradient_boosting", "knn", "random_forest"],
    )
    def test_a_shared_alias_refuses_to_guess_and_names_both_candidates(self, alias):
        with pytest.raises(AmbiguousModelAliasError) as raised:
            ModelFactory.registration(alias)
        message = str(raised.value)
        assert "Pass task=" in message
        classifier = ModelFactory.registration(alias, task="classification")
        regressor = ModelFactory.registration(alias, task="regression")
        assert classifier.canonical_name in message
        assert regressor.canonical_name in message

    def test_an_alias_matching_only_one_family_says_so_rather_than_guessing(self):
        """``gnb`` has no regression twin; asking for one is a contradiction."""
        with pytest.raises(IncompatibleModelError, match="none of which serves regression"):
            ModelFactory.registration("gnb", task="regression")


class TestCatalogSerialisation:
    @pytest.fixture
    def catalog(self):
        return ModelFactory.catalog(task="regression")

    def test_the_whole_catalog_survives_json(self, catalog):
        assert json.loads(json.dumps(catalog))[0]["canonical_name"] == EXPECTED[0]

    def test_no_estimator_object_leaks_into_it(self, catalog):
        """Every value is a JSON primitive, not merely free of three substrings.

        ``jsonable`` falls back to ``repr()`` for anything it does not recognise,
        so a leaked object arrives as a plausible-looking string. Checking the
        types is the only guard that cannot be walked past.
        """

        def primitive(value) -> bool:
            if isinstance(value, dict):
                return all(isinstance(k, str) and primitive(v) for k, v in value.items())
            if isinstance(value, list):
                return all(primitive(v) for v in value)
            return value is None or isinstance(value, (str, int, float, bool))

        for entry in catalog:
            assert primitive(entry), entry["canonical_name"]

    def test_nothing_in_it_can_be_fitted(self, catalog):
        for entry in catalog:
            assert all(not hasattr(value, "fit") for value in entry.values())

    def test_every_entry_reports_its_capabilities(self, catalog):
        for entry in catalog:
            assert set(entry["capabilities"]) >= {
                "supports_predict_proba",
                "requires_scaling",
                "supports_sparse_input",
                "supports_multiclass",
                "handles_missing_values",
                "preprocessing_profile",
            }

    def test_every_entry_reports_its_defaults(self, catalog):
        for entry in catalog:
            assert entry["default_params"], entry["canonical_name"]

    def test_every_entry_declares_the_regression_task(self, catalog):
        for entry in catalog:
            assert entry["task_type"] == "regression"
            assert entry["capabilities"]["supports_predict_proba"] is False

    @pytest.mark.parametrize(
        "name,key,value",
        [
            ("dummy_regressor", "strategy", "mean"),
            ("linear_regression", "fit_intercept", True),
            ("ridge_regression", "alpha", 1.0),
            ("random_forest_regressor", "n_estimators", 100),
            ("extra_trees_regressor", "n_estimators", 100),
            ("gradient_boosting_regressor", "n_estimators", 100),
            ("hist_gradient_boosting_regressor", "max_iter", 100),
            ("hist_gradient_boosting_regressor", "early_stopping", "auto"),
            ("knn_regressor", "n_neighbors", 5),
        ],
    )
    def test_the_catalog_publishes_the_budget_a_user_is_getting(
        self, catalog, name, key, value
    ):
        by_name = {entry["canonical_name"]: entry for entry in catalog}
        assert by_name[name]["default_params"][key] == value

    def test_the_regularisation_strength_is_not_hidden(self, catalog):
        """Alpha is the whole difference between the two linear models."""
        by_name = {entry["canonical_name"]: entry for entry in catalog}
        assert "alpha" in by_name["ridge_regression"]["default_params"]
        assert "alpha" not in by_name["linear_regression"]["default_params"]

    def test_the_validation_split_is_not_hidden(self, catalog):
        """Above ~10k rows this model withholds a tenth of the training data."""
        by_name = {entry["canonical_name"]: entry for entry in catalog}
        published = by_name["hist_gradient_boosting_regressor"]["default_params"]
        assert "early_stopping" in published

    def test_the_seedless_models_publish_no_seed(self, catalog):
        """Three regressors take no random_state; inventing one would be theatre.

        ``DummyRegressor`` is the one that differs from its classification twin:
        ``DummyClassifier`` accepts a seed and this does not.
        """
        by_name = {entry["canonical_name"]: entry for entry in catalog}
        for name in ("dummy_regressor", "linear_regression", "knn_regressor"):
            assert "random_state" not in by_name[name]["default_params"], name

    def test_and_those_three_are_exactly_the_ones_whose_constructor_takes_none(self):
        seedless = {
            name
            for name in EXPECTED
            if "random_state" not in ModelFactory.create(name).get_params()
        }
        assert seedless == {"dummy_regressor", "linear_regression", "knn_regressor"}

    def test_every_seeded_model_publishes_the_same_seed(self, catalog):
        seeds = {
            entry["default_params"].get("random_state")
            for entry in catalog
            if "random_state" in entry["default_params"]
        }
        assert seeds == {42}

    def test_the_seed_agrees_with_the_classification_family(self):
        """One configuration change makes every model reproducible together."""
        seeds = {
            entry["default_params"].get("random_state")
            for entry in ModelFactory.catalog()
            if entry["default_params"] and "random_state" in entry["default_params"]
        }
        assert len(seeds) == 1


class TestErrors:
    def test_an_unknown_parameter_is_refused_by_name(self):
        with pytest.raises(InvalidModelParameterError, match="n_estimatorss"):
            ModelFactory.create("random_forest_regressor", n_estimatorss=10)

    def test_the_refusal_lists_what_is_accepted(self):
        with pytest.raises(InvalidModelParameterError, match="n_estimators"):
            ModelFactory.create("extra_trees_regressor", nonsense=1)

    def test_the_suggestion_finds_the_regression_models(self):
        from aidatasetkit.core.exceptions import UnknownModelError

        with pytest.raises(UnknownModelError, match="ridge_regression"):
            ModelFactory.create("ridge_regresion")

    @pytest.mark.parametrize(
        "name,param,value",
        [
            ("dummy_regressor", "strategy", "median"),
            ("linear_regression", "fit_intercept", False),
            ("ridge_regression", "alpha", 0.25),
            ("decision_tree_regressor", "max_depth", 3),
            ("random_forest_regressor", "n_estimators", 7),
            ("extra_trees_regressor", "n_estimators", 7),
            ("gradient_boosting_regressor", "learning_rate", 0.05),
            ("hist_gradient_boosting_regressor", "max_iter", 7),
            ("knn_regressor", "n_neighbors", 3),
        ],
    )
    def test_overrides_reach_the_estimator(self, name, param, value):
        assert ModelFactory.create(name, **{param: value}).get_params()[param] == value

    def test_the_regularisation_strength_is_never_chosen_by_the_library(self):
        """Alpha may be passed. It is never searched for, and never tuned here."""
        assert ModelFactory.create("ridge_regression").get_params()["alpha"] == 1.0
        assert (
            ModelFactory.create("ridge_regression", alpha=10.0).get_params()["alpha"]
            == 10.0
        )


class TestPreprocessingProfilesAreShared:
    """Nine regressors, four profiles -- all four already required by a classifier."""

    @pytest.fixture
    def by_profile(self) -> dict[PreprocessingProfile, list[str]]:
        groups: dict[PreprocessingProfile, list[str]] = {}
        for entry in default_registry().catalog(task="regression"):
            groups.setdefault(entry.capabilities.preprocessing_profile(), []).append(
                entry.canonical_name
            )
        return groups

    def test_there_are_four_distinct_profiles(self, by_profile):
        assert len(by_profile) == 4

    def test_every_natively_missing_aware_model_shares_one(self, by_profile):
        """Four models, one preprocessor: no scaler, dense, gaps left alone."""
        shared = PreprocessingProfile(
            requires_scaling=False,
            supports_sparse_input=False,
            handles_missing_values=True,
        )
        assert sorted(by_profile[shared]) == [
            "decision_tree_regressor",
            "extra_trees_regressor",
            "hist_gradient_boosting_regressor",
            "random_forest_regressor",
        ]

    def test_the_baseline_stands_alone_because_it_reads_nothing(self, by_profile):
        """It is the only regressor that takes sparse and NaN in the same matrix."""
        shared = PreprocessingProfile(
            requires_scaling=False,
            supports_sparse_input=True,
            handles_missing_values=True,
        )
        assert by_profile[shared] == ["dummy_regressor"]

    def test_the_booster_stands_alone(self, by_profile):
        shared = PreprocessingProfile(
            requires_scaling=False,
            supports_sparse_input=True,
            handles_missing_values=False,
        )
        assert by_profile[shared] == ["gradient_boosting_regressor"]

    def test_a_penalised_solve_and_a_distance_vote_share_one(self, by_profile):
        """No taxonomy of algorithms would put these together.

        A penalised linear solve, an unpenalised one, and a nearest-neighbour
        average share a preprocessor because they give the same three answers,
        and nothing else about them is consulted. This is the clearest
        demonstration in the catalog that the cache key is a capability triple.
        """
        shared = PreprocessingProfile(
            requires_scaling=True,
            supports_sparse_input=True,
            handles_missing_values=False,
        )
        assert sorted(by_profile[shared]) == [
            "knn_regressor",
            "linear_regression",
            "ridge_regression",
        ]

    def test_both_linear_models_ask_for_scaling_for_different_reasons(self):
        """The declarations agree; the reasons do not, and both are measured.

        Ridge because its penalty reads the unit a column was recorded in.
        Ordinary least squares because ``scipy.linalg.lstsq`` truncates a small
        singular value and silently discards a real feature. An earlier version
        of this catalog declared OLS scale-free on the strength of a frame that
        happened to sit just under the truncation threshold; adversarial review
        found an ordinary one that sat above it.
        """
        ordinary = ModelFactory.registration(
            "linear_regression"
        ).capabilities.preprocessing_profile()
        penalised = ModelFactory.registration(
            "ridge_regression"
        ).capabilities.preprocessing_profile()
        assert ordinary == penalised
        assert ordinary.requires_scaling and penalised.requires_scaling

    def test_the_two_boosters_do_not_share_a_profile(self):
        gradient = ModelFactory.registration(
            "gradient_boosting_regressor"
        ).capabilities.preprocessing_profile()
        hist = ModelFactory.registration(
            "hist_gradient_boosting_regressor"
        ).capabilities.preprocessing_profile()
        assert gradient != hist

    def test_a_regressor_shares_its_profile_with_a_classifier(self, by_profile):
        """Reuse crosses the task boundary, because the key knows nothing about it."""
        tree = ModelFactory.registration(
            "decision_tree_regressor"
        ).capabilities.preprocessing_profile()
        classifier = ModelFactory.registration(
            "decision_tree_classifier"
        ).capabilities.preprocessing_profile()
        assert tree == classifier

    def test_no_profile_key_mentions_a_model_or_a_task(self, by_profile):
        for profile in by_profile:
            assert profile.key == (
                f"scaling={int(profile.requires_scaling)},"
                f"sparse={int(profile.supports_sparse_input)},"
                f"native_nan={int(profile.handles_missing_values)}"
            )
            assert "regress" not in profile.key
