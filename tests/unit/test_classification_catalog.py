"""The classification catalog: what exists, what it is called, and how it is found.

Nothing here fits anything. These are the questions a user asks before choosing a
model -- what can this library do, what may I type, and will the answer be the
same tomorrow -- and the answers come from the registry alone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import ClassVar

import pytest

from aidatasetkit.core.exceptions import (
    AmbiguousModelAliasError,
    IncompatibleModelError,
    InvalidModelParameterError,
    UnknownModelError,
)
from aidatasetkit.core.types import (
    Backend,
    Estimator,
    Interpretability,
    PreprocessingProfile,
    TaskType,
)
from aidatasetkit.models import (
    ModelCapabilities,
    ModelFactory,
    ModelRegistry,
    ModelStrategy,
    default_registry,
)

from tests.conftest import BUILT_IN_CLASSIFIERS, BUILT_IN_PROFILE_COUNT

#: The approved catalog, named once in conftest and asserted here.
EXPECTED = BUILT_IN_CLASSIFIERS

#: Canonical name -> the concise alias it answers to.
EXPECTED_ALIASES = {
    "decision_tree_classifier": ("decision_tree",),
    "extra_trees_classifier": ("extra_trees",),
    "gaussian_nb": ("gnb",),
    "gradient_boosting_classifier": ("gradient_boosting",),
    "hist_gradient_boosting_classifier": ("hist_gradient_boosting",),
    "knn_classifier": ("knn",),
    "random_forest_classifier": ("random_forest",),
}


class TestTheCatalogHoldsWhatWasApproved:
    def test_it_contains_exactly_nine_classifiers(self):
        assert len(ModelFactory.available(task="classification")) == 9

    def test_it_contains_exactly_the_approved_models(self):
        assert ModelFactory.available(task="classification") == EXPECTED

    def test_no_regression_model_has_appeared(self):
        assert ModelFactory.available(task="regression") == ()

    def test_every_classifier_is_constructible_here(self):
        for name in EXPECTED:
            assert isinstance(ModelFactory.create(name), Estimator)

    @pytest.mark.parametrize("name", EXPECTED)
    def test_each_one_declares_the_classification_task(self, name):
        assert ModelFactory.registration(name).task_type is TaskType.CLASSIFICATION

    @pytest.mark.parametrize("name", EXPECTED)
    def test_each_one_is_backed_by_scikit_learn(self, name):
        assert ModelFactory.registration(name).backend is Backend.SKLEARN

    def test_exactly_one_model_is_the_baseline(self):
        baselines = [
            entry.canonical_name
            for entry in default_registry().catalog(task="classification")
            if entry.capabilities.is_baseline
        ]
        assert baselines == ["dummy_classifier"]


#: One fresh interpreter, several questions. Each subprocess pays the whole
#: import cost, so asking six questions in six of them wastes most of a minute.
_FRESH_SCRIPT = (
    "import sys;"
    "from aidatasetkit.models import ModelFactory;"
    "print(','.join(ModelFactory.available(task='classification')));"
    "print(len(ModelFactory.available(task='classification')));"
    "print(ModelFactory.create('random_forest').__class__.__name__);"
    "print(any('classification.forest' in m for m in sys.modules))"
)


def _fresh(seed: str) -> list[str]:
    """Run the script in a new interpreter under ``seed`` and return its answers."""
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": ""},
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


class TestOrderingIsDeterministic:
    def test_available_is_sorted_by_canonical_name(self):
        names = ModelFactory.available(task="classification")
        assert list(names) == sorted(names)

    def test_repeated_calls_agree(self):
        assert ModelFactory.available(task="classification") == ModelFactory.available(
            task="classification"
        )

    def test_the_catalog_follows_the_same_order(self):
        catalog = ModelFactory.catalog(task="classification")
        assert [entry["canonical_name"] for entry in catalog] == list(EXPECTED)

    def test_ordering_does_not_depend_on_the_hash_seed(self, fresh_runs):
        """Import order and dict iteration must not reach the published order."""
        assert {run[0] for run in fresh_runs.values()} == {",".join(EXPECTED)}


class TestRegistrationNeedsNoManualImport:
    """A user types ModelFactory.create('random_forest'). Nothing else."""

    def test_a_fresh_interpreter_sees_every_classifier(self, fresh_runs):
        assert {run[1] for run in fresh_runs.values()} == {"9"}

    def test_importing_only_the_top_level_package_is_enough(self, fresh_runs):
        assert {run[2] for run in fresh_runs.values()} == {"RandomForestClassifier"}

    def test_no_strategy_module_has_to_be_imported_by_hand(self, fresh_runs):
        """Naming a strategy module directly must not be what makes it exist."""
        assert {run[3] for run in fresh_runs.values()} == {"True"}


class TestAliases:
    @pytest.mark.parametrize("canonical,aliases", sorted(EXPECTED_ALIASES.items()))
    def test_the_concise_alias_resolves_to_its_model(self, canonical, aliases):
        for alias in aliases:
            assert ModelFactory.registration(alias).canonical_name == canonical

    @pytest.mark.parametrize("canonical,aliases", sorted(EXPECTED_ALIASES.items()))
    def test_the_registry_reports_those_aliases(self, canonical, aliases):
        assert default_registry().aliases_of(canonical) == aliases

    def test_the_gaussian_model_does_not_squat_on_the_general_name(self):
        """MultinomialNB and BernoulliNB are naive Bayes too, and come later."""
        assert "naive_bayes" not in default_registry()

    def test_no_alias_shadows_a_canonical_name(self):
        canonical = set(ModelFactory.available(task="classification"))
        for aliases in EXPECTED_ALIASES.values():
            assert not (set(aliases) & canonical)

    def test_an_alias_can_be_narrowed_by_task(self):
        assert (
            ModelFactory.registration("knn", task="classification").canonical_name
            == "knn_classifier"
        )

    def test_asking_for_a_classifier_as_a_regressor_is_a_contradiction(self):
        with pytest.raises(IncompatibleModelError):
            ModelFactory.registration("random_forest", task="regression")


class TestAliasesSurviveTheArrivalOfRegression:
    """S6 adds regression twins. These aliases must still resolve then.

    Proved on a scratch registry holding both families rather than on the real
    one, because the point is what happens *after* a model that does not exist
    yet is registered.
    """

    @pytest.fixture
    def both_families(self) -> ModelRegistry:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        scratch = ModelRegistry()

        class Classifier(ModelStrategy):
            name: ClassVar[str] = "random_forest_classifier"
            capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
                task_type=TaskType.CLASSIFICATION,
                backend=Backend.SKLEARN,
                supports_predict_proba=True,
                requires_scaling=False,
                supports_sparse_input=True,
                supports_multiclass=True,
                handles_missing_values=True,
                interpretability_level=Interpretability.MEDIUM,
            )

            def build(self, **params):
                return RandomForestClassifier(**params)

        class Regressor(ModelStrategy):
            name: ClassVar[str] = "random_forest_regressor"
            capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
                task_type=TaskType.REGRESSION,
                backend=Backend.SKLEARN,
                supports_predict_proba=False,
                requires_scaling=False,
                supports_sparse_input=True,
                supports_multiclass=False,
                handles_missing_values=True,
                interpretability_level=Interpretability.MEDIUM,
            )

            def build(self, **params):
                return RandomForestRegressor(**params)

        scratch.register(Classifier, aliases=("random_forest",))
        scratch.register(Regressor, aliases=("random_forest",))
        return scratch

    def test_the_shared_alias_is_accepted_by_both_families(self, both_families):
        assert len(both_families) == 2

    def test_task_narrows_it_to_the_classifier(self, both_families):
        entry = both_families.resolve("random_forest", task="classification")
        assert entry.canonical_name == "random_forest_classifier"

    def test_task_narrows_it_to_the_regressor(self, both_families):
        entry = both_families.resolve("random_forest", task="regression")
        assert entry.canonical_name == "random_forest_regressor"

    def test_without_a_task_it_refuses_rather_than_guessing(self, both_families):
        with pytest.raises(AmbiguousModelAliasError, match="Pass task="):
            both_families.resolve("random_forest")

    def test_the_canonical_names_never_became_ambiguous(self, both_families):
        assert (
            both_families.resolve("random_forest_classifier").task_type
            is TaskType.CLASSIFICATION
        )


class TestCatalogSerialisation:
    @pytest.fixture
    def catalog(self):
        return ModelFactory.catalog(task="classification")

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

    @pytest.mark.parametrize(
        "name,key,value",
        [
            ("random_forest_classifier", "n_estimators", 100),
            ("extra_trees_classifier", "n_estimators", 100),
            ("gradient_boosting_classifier", "n_estimators", 100),
            ("hist_gradient_boosting_classifier", "max_iter", 100),
            ("hist_gradient_boosting_classifier", "early_stopping", "auto"),
            ("knn_classifier", "n_neighbors", 5),
            ("gaussian_nb", "var_smoothing", 1e-9),
        ],
    )
    def test_the_catalog_publishes_the_budget_a_user_is_getting(
        self, catalog, name, key, value
    ):
        by_name = {entry["canonical_name"]: entry for entry in catalog}
        assert by_name[name]["default_params"][key] == value

    def test_the_validation_split_is_not_hidden(self, catalog):
        """Above ~10k rows this model withholds a tenth of the training data."""
        by_name = {entry["canonical_name"]: entry for entry in catalog}
        published = by_name["hist_gradient_boosting_classifier"]["default_params"]
        assert "early_stopping" in published

    def test_the_seedless_models_publish_no_seed(self, catalog):
        """KNN and GaussianNB take no random_state; inventing one would be theatre."""
        by_name = {entry["canonical_name"]: entry for entry in catalog}
        assert "random_state" not in by_name["knn_classifier"]["default_params"]
        assert "random_state" not in by_name["gaussian_nb"]["default_params"]

    def test_every_seeded_model_publishes_the_same_seed(self, catalog):
        seeds = {
            entry["default_params"].get("random_state")
            for entry in catalog
            if "random_state" in entry["default_params"]
        }
        assert len(seeds) == 1


class TestErrorsDidNotRegressWhileTheCatalogGrew:
    def test_an_unknown_name_still_raises(self):
        with pytest.raises(UnknownModelError):
            ModelFactory.create("random_forrest")

    def test_the_suggestion_now_finds_the_new_models(self):
        with pytest.raises(UnknownModelError, match="random_forest"):
            ModelFactory.create("random_forrest")

    def test_an_unknown_parameter_is_still_refused_by_name(self):
        with pytest.raises(InvalidModelParameterError, match="n_estimatorss"):
            ModelFactory.create("random_forest", n_estimatorss=10)

    def test_the_refusal_lists_what_is_accepted(self):
        with pytest.raises(InvalidModelParameterError, match="n_estimators"):
            ModelFactory.create("extra_trees", nonsense=1)

    @pytest.mark.parametrize(
        "name,param,value",
        [
            ("decision_tree", "max_depth", 3),
            ("random_forest", "n_estimators", 7),
            ("extra_trees", "n_estimators", 7),
            ("gradient_boosting", "learning_rate", 0.05),
            ("hist_gradient_boosting", "max_iter", 7),
            ("knn", "n_neighbors", 3),
            ("gaussian_nb", "var_smoothing", 1e-7),
        ],
    )
    def test_overrides_reach_the_estimator(self, name, param, value):
        assert ModelFactory.create(name, **{param: value}).get_params()[param] == value

    def test_class_weight_may_be_passed_but_is_never_chosen(self):
        """The library must not select a rebalancing strategy on the user's behalf."""
        assert ModelFactory.create("random_forest").get_params()["class_weight"] is None
        assert (
            ModelFactory.create("random_forest", class_weight="balanced").get_params()[
                "class_weight"
            ]
            == "balanced"
        )


class TestPreprocessingProfilesAreShared:
    """Nine classifiers, five profiles. That is the whole point of caching on them."""

    @pytest.fixture
    def by_profile(self) -> dict[PreprocessingProfile, list[str]]:
        groups: dict[PreprocessingProfile, list[str]] = {}
        for entry in default_registry().catalog(task="classification"):
            groups.setdefault(
                entry.capabilities.preprocessing_profile(), []
            ).append(entry.canonical_name)
        return groups

    def test_there_are_five_distinct_profiles(self, by_profile):
        assert len(by_profile) == BUILT_IN_PROFILE_COUNT

    def test_every_natively_missing_aware_model_shares_one(self, by_profile):
        """Four models, one preprocessor: no scaler, dense, gaps left alone."""
        shared = PreprocessingProfile(
            requires_scaling=False,
            supports_sparse_input=False,
            handles_missing_values=True,
        )
        assert sorted(by_profile[shared]) == [
            "decision_tree_classifier",
            "extra_trees_classifier",
            "hist_gradient_boosting_classifier",
            "random_forest_classifier",
        ]

    def test_the_baseline_stands_alone_because_it_reads_nothing(self, by_profile):
        """It is the only model that takes sparse and NaN in the same matrix."""
        shared = PreprocessingProfile(
            requires_scaling=False, supports_sparse_input=True, handles_missing_values=True
        )
        assert by_profile[shared] == ["dummy_classifier"]

    def test_the_scaled_sparse_models_share_one(self, by_profile):
        shared = PreprocessingProfile(
            requires_scaling=True, supports_sparse_input=True, handles_missing_values=False
        )
        assert sorted(by_profile[shared]) == ["knn_classifier", "logistic_regression"]

    def test_the_two_boosters_do_not_share_a_profile(self, by_profile):
        gradient = ModelFactory.registration(
            "gradient_boosting"
        ).capabilities.preprocessing_profile()
        hist = ModelFactory.registration(
            "hist_gradient_boosting"
        ).capabilities.preprocessing_profile()
        assert gradient != hist

    def test_every_profile_key_is_a_stable_string(self, by_profile):
        for profile in by_profile:
            assert profile.key == profile.key
            assert "scaling=" in profile.key
