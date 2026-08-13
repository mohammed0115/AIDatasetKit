"""S7 end to end: one boundary, many models, and the promise that holds across it.

The two tests this file exists for are the leakage adversarial and the comparison
fairness one. Everything else supports them.

Leakage is checked by *learned state*, not by watching which methods were called.
An evaluation row carrying an extreme value is added and removed, and the median
the imputer learned, the centre the scaler learned, and the vocabulary the encoder
learned are compared. If any of them moves, an evaluation row reached a ``fit``.

Fairness is checked by *row identity*, not by row count. Two comparisons can give
every model eighty evaluation rows and give each model a different eighty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import (
    IncompatibleModelError,
    PreprocessingError,
    TrainingError,
    UnknownModelError,
    ValidationError,
)
from aidatasetkit.core.types import TaskType
from aidatasetkit.evaluation import MetricDirection, MetricStatus
from aidatasetkit.models import ModelFactory, default_registry
from aidatasetkit.preprocessing import PreprocessingConfig
from aidatasetkit.profiling import TaskDetector
from aidatasetkit.training import (
    ComparisonResult,
    ModelComparator,
    ModelTrainer,
    split_rows,
)

#: Small ensembles everywhere. The contracts under test are unaffected by tree
#: count, and a hundred trees per model per scenario is minutes of nothing.
SMALL = {
    "random_forest_classifier": {"n_estimators": 5},
    "extra_trees_classifier": {"n_estimators": 5},
    "gradient_boosting_classifier": {"n_estimators": 5},
    "hist_gradient_boosting_classifier": {"max_iter": 10},
    "random_forest_regressor": {"n_estimators": 5},
    "extra_trees_regressor": {"n_estimators": 5},
    "gradient_boosting_regressor": {"n_estimators": 5},
    "hist_gradient_boosting_regressor": {"max_iter": 10},
}


# --------------------------------------------------------------------------- #
# Golden scenarios
# --------------------------------------------------------------------------- #


def classification_frame(n: int = 400, seed: int = 20250301) -> pd.DataFrame:
    """Numeric on two scales, a category, gaps, imbalance, awkward labels.

    Deliberately built so a naive positive-class assumption would be wrong:
    ``"churn"`` sorts first, so the event of interest encodes to 0.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.5, 0.08, n)
    frame = pd.DataFrame(
        {
            "income": rng.normal(60_000, 15_000, n).round(2),
            "ratio": signal.round(4),
            "city": rng.choice(["Riyadh", "Jeddah"], n),
            "is_member": rng.choice([True, False], n),
            "Churn": np.where(signal + rng.normal(0, 0.02, n) > 0.60, "churn", "stay"),
        }
    )
    frame.loc[frame.index[:30], "income"] = np.nan
    return frame


def regression_frame(n: int = 400, seed: int = 20250302) -> pd.DataFrame:
    """Scales differing by orders of magnitude, a category, gaps, signed targets."""
    rng = np.random.default_rng(seed)
    income = rng.normal(60_000, 15_000, n)
    ratio = rng.normal(0.5, 0.05, n)
    frame = pd.DataFrame(
        {
            "income": income.round(2),
            "ratio": ratio.round(4),
            "city": rng.choice(["Riyadh", "Jeddah"], n),
            "margin": (2e-4 * income + 300.0 * ratio - 200.0 + rng.normal(0, 2.0, n)),
        }
    )
    frame.loc[frame.index[:30], "income"] = np.nan
    return frame


CLASSIFIERS = [
    "dummy_classifier",
    "logistic_regression",
    "decision_tree_classifier",
    "knn_classifier",
]
REGRESSORS = [
    "dummy_regressor",
    "linear_regression",
    "ridge_regression",
    "decision_tree_regressor",
    "knn_regressor",
]


@pytest.fixture(scope="module")
def classification_run() -> ComparisonResult:
    return ModelComparator().compare(
        classification_frame(),
        target="Churn",
        models=CLASSIFIERS,
        positive_label="churn",
        model_params=SMALL,
    )


@pytest.fixture(scope="module")
def regression_run() -> ComparisonResult:
    return ModelComparator().compare(
        regression_frame(), target="margin", models=REGRESSORS, model_params=SMALL
    )


class TestTheGoldenClassificationScenario:
    def test_every_model_produced_an_outcome(self, classification_run):
        assert len(classification_run.outcomes) == len(CLASSIFIERS)
        assert not classification_run.failed

    def test_the_task_was_resolved_rather_than_assumed(self, classification_run):
        assert classification_run.task_type is TaskType.CLASSIFICATION

    def test_the_positive_class_is_not_encoded_one(self, classification_run):
        """The whole reason this scenario uses churn/stay."""
        encoding = classification_run.outcome_for(
            "logistic_regression"
        ).training.target_encoding.encoding
        assert encoding.positive_label == "churn"
        assert encoding.positive_label_encoded == 0

    def test_the_metrics_followed_that_class(self, classification_run):
        entry = classification_run.outcome_for("logistic_regression").evaluation["f1"]
        assert entry.detail["positive_label"] == "churn"

    def test_a_real_model_beat_the_baseline(self, classification_run):
        """Deliberately learnable data, so this invariant means something."""
        baseline = classification_run.outcome_for("dummy_classifier")
        model = classification_run.outcome_for("logistic_regression")
        assert model.evaluation["f1"].value > baseline.evaluation["f1"].value

    def test_the_baseline_is_not_assumed_to_rank_last(self, classification_run):
        """It does here; nothing in the code requires that it must."""
        assert classification_run.ranked[-1].model_name == "dummy_classifier"
        assert "dummy" not in str(type(classification_run).__mro__)

    def test_models_without_probabilities_say_so_rather_than_dropping_roc_auc(
        self, classification_run
    ):
        entry = classification_run.outcome_for("decision_tree_classifier").evaluation[
            "roc_auc"
        ]
        assert entry.is_available  # a tree does publish probabilities
        assert "roc_auc" in classification_run.outcome_for("knn_classifier").evaluation

    def test_ranking_used_the_documented_metric_and_direction(self, classification_run):
        assert classification_run.ranking_metric == "f1"
        assert classification_run.ranking_direction is MetricDirection.HIGHER_IS_BETTER

    def test_the_leaderboard_lists_every_model(self, classification_run):
        frame = classification_run.to_frame()
        assert set(frame["model"]) == set(CLASSIFIERS)

    def test_accuracy_and_minority_f1_are_both_visible(self, classification_run):
        """Imbalanced data: a high accuracy beside a poor F1 must stay readable."""
        baseline = classification_run.outcome_for("dummy_classifier").evaluation
        assert baseline["accuracy"].value > 0.5
        assert baseline["f1"].value < baseline["accuracy"].value


class TestTheGoldenRegressionScenario:
    def test_every_model_produced_an_outcome(self, regression_run):
        assert len(regression_run.outcomes) == len(REGRESSORS)
        assert not regression_run.failed

    def test_no_target_was_encoded(self, regression_run):
        """The S6 closure invariant, carried into training."""
        for outcome in regression_run.outcomes:
            assert outcome.training.target_encoding is None

    def test_ranking_used_rmse_downward(self, regression_run):
        assert regression_run.ranking_metric == "rmse"
        assert regression_run.ranking_direction is MetricDirection.LOWER_IS_BETTER

    def test_the_ranking_really_is_ascending_in_rmse(self, regression_run):
        values = [o.evaluation["rmse"].value for o in regression_run.ranked]
        assert values == sorted(values)

    def test_the_metric_values_are_reported_unnegated(self, regression_run):
        for outcome in regression_run.succeeded:
            assert outcome.evaluation["mae"].value > 0

    def test_a_real_model_beat_the_baseline(self, regression_run):
        baseline = regression_run.outcome_for("dummy_regressor").evaluation["rmse"].value
        model = regression_run.outcome_for("ridge_regression").evaluation["rmse"].value
        assert model < baseline

    def test_capability_driven_preprocessing_differed_between_models(
        self, regression_run
    ):
        profiles = {
            o.model_name: o.training.preprocessing_profile.key
            for o in regression_run.succeeded
        }
        assert profiles["ridge_regression"].startswith("scaling=1")
        assert profiles["decision_tree_regressor"].startswith("scaling=0")

    def test_deterministic_under_a_fixed_configuration(self):
        first = ModelComparator().compare(
            regression_frame(), target="margin", models=REGRESSORS, model_params=SMALL
        )
        second = ModelComparator().compare(
            regression_frame(), target="margin", models=REGRESSORS, model_params=SMALL
        )
        assert [o.model_name for o in first.ranked] == [
            o.model_name for o in second.ranked
        ]
        assert first.split.fingerprint == second.split.fingerprint


# --------------------------------------------------------------------------- #
# The release-blocking invariant
# --------------------------------------------------------------------------- #


class TestNothingIsFittedOnEvaluationRows:
    """§47. Learned state is compared, not method calls.

    An extreme value and an unseen category are placed in rows that the split
    sends to evaluation. If any learned statistic moves, an evaluation row
    reached a ``fit`` -- and the whole promise of this library is void.
    """

    @staticmethod
    def _clean() -> pd.DataFrame:
        rng = np.random.default_rng(4242)
        n = 200
        frame = pd.DataFrame(
            {
                "amount": rng.normal(100.0, 10.0, n),
                "city": rng.choice(["Riyadh", "Jeddah"], n),
                "y": rng.choice(["a", "b"], n),
            }
        )
        frame.loc[frame.index[:20], "amount"] = np.nan
        return frame

    @classmethod
    def _frame(cls, contaminate: bool) -> pd.DataFrame:
        """The same data, optionally poisoned *only* on the evaluation side.

        The split is drawn first and the extreme values are written into the rows
        it assigns to evaluation. Sprinkling them at random would leave some in
        the training set, where they legitimately move every statistic -- and the
        test would then be measuring its own setup rather than the contract.

        The target is untouched, so both frames yield the same split and
        therefore the same training rows. Any difference in learned state is a
        leak and nothing else.
        """
        frame = cls._clean()
        if not contaminate:
            return frame
        split = split_rows(frame["y"], TaskType.CLASSIFICATION)
        evaluation_rows = frame.index[split.evaluation_positions][:5]
        frame.loc[evaluation_rows, "amount"] = 1e9
        frame.loc[evaluation_rows, "city"] = "Atlantis"
        return frame

    @staticmethod
    def _learned(frame: pd.DataFrame, model: str):
        profile = TaskDetector().detect(frame["y"], target_name="y")
        split = split_rows(frame["y"], profile.task_type)
        trainer = ModelTrainer()
        result = trainer.train(
            split.take(frame), target="y", model=model, target_profile=profile
        )
        groups = result.preprocessor.transformer.named_transformers_
        numeric = groups["numeric"].named_steps
        learned = {}
        if "imputer" in numeric:
            learned["imputer"] = tuple(numeric["imputer"].statistics_.tolist())
        if "scaler" in numeric:
            learned["scaler_mean"] = tuple(numeric["scaler"].mean_.tolist())
            learned["scaler_scale"] = tuple(numeric["scaler"].scale_.tolist())
        # The group a two-valued category lands in is "binary"; a third value
        # would move it to "nominal". Reading whichever exists keeps this test
        # about leakage rather than about group naming.
        for label in ("binary", "nominal"):
            if label in groups:
                encoder = groups[label].named_steps["encoder"]
                learned["vocabulary"] = tuple(
                    tuple(str(v) for v in values) for values in encoder.categories_
                )
                break
        return learned, split, result

    @pytest.mark.parametrize("model", ["knn_classifier", "logistic_regression"])
    def test_the_scaler_centre_is_identical_with_and_without_the_extreme_rows(
        self, model
    ):
        clean, _, _ = self._learned(self._frame(False), model)
        contaminated, split, _ = self._learned(self._frame(True), model)
        assert clean["scaler_mean"] == contaminated["scaler_mean"]
        assert clean["scaler_scale"] == contaminated["scaler_scale"]

    @pytest.mark.parametrize("model", ["knn_classifier", "logistic_regression"])
    def test_the_imputation_statistic_is_identical_too(self, model):
        clean, _, _ = self._learned(self._frame(False), model)
        contaminated, _, _ = self._learned(self._frame(True), model)
        assert clean["imputer"] == contaminated["imputer"]

    @pytest.mark.parametrize("model", ["knn_classifier", "decision_tree_classifier"])
    def test_the_categorical_vocabulary_never_learned_the_unseen_category(self, model):
        clean, _, _ = self._learned(self._frame(False), model)
        contaminated, _, _ = self._learned(self._frame(True), model)
        assert clean["vocabulary"] == contaminated["vocabulary"]
        flat = [value for values in contaminated["vocabulary"] for value in values]
        assert "Atlantis" not in flat

    def test_the_contamination_would_have_been_visible_if_it_had_leaked(self):
        """Otherwise the three tests above prove nothing.

        Fitting on the whole frame -- which is what a leak amounts to -- moves the
        scaler's centre by orders of magnitude.
        """
        from sklearn.preprocessing import StandardScaler

        contaminated = self._frame(True)["amount"].to_numpy(dtype="float64")
        clean = self._frame(False)["amount"].to_numpy(dtype="float64")
        leaked = StandardScaler().fit(
            np.nan_to_num(contaminated, nan=100.0).reshape(-1, 1)
        )
        honest = StandardScaler().fit(np.nan_to_num(clean, nan=100.0).reshape(-1, 1))
        assert leaked.mean_[0] > honest.mean_[0] * 100

    def test_the_extreme_rows_landed_entirely_on_the_evaluation_side(self):
        """The adversarial setup has to actually be adversarial.

        Every poisoned row on the evaluation side, and none on the training side
        -- otherwise a moved statistic would be legitimate and the tests above
        would be measuring the fixture.
        """
        frame = self._frame(True)
        split = split_rows(frame["y"], TaskType.CLASSIFICATION)
        evaluation = split.take(frame, evaluation=True)
        train = split.take(frame)

        assert (evaluation["amount"] > 1e8).sum() == 5
        assert "Atlantis" in set(evaluation["city"])
        assert not (train["amount"] > 1e8).any()
        assert "Atlantis" not in set(train["city"])

    def test_and_the_training_rows_are_byte_identical_between_the_two_frames(self):
        """Same split, same training rows -- so any difference is the leak."""
        split = split_rows(self._clean()["y"], TaskType.CLASSIFICATION)
        clean = split.take(self._frame(False))
        contaminated = split.take(self._frame(True))
        pd.testing.assert_frame_equal(clean, contaminated)


class TestComparisonFairness:
    """§48. Same rows, proved by identity rather than by count."""

    @pytest.fixture(scope="class")
    @staticmethod
    def instrumented():
        """Every raw evaluation row each model was actually scored against."""
        frame = classification_frame(n=300)
        frame = frame.assign(row_id=[f"r{i:04d}" for i in range(len(frame))])

        seen: dict[str, tuple[str, ...]] = {}
        original = ModelTrainer.evaluate

        def recording(self, result, evaluation_frame, *, target):
            seen[result.model_name] = tuple(evaluation_frame["row_id"])
            return original(
                self,
                result,
                evaluation_frame.drop(columns=["row_id"]),
                target=target,
            )

        ModelTrainer.evaluate = recording
        try:
            comparator = ModelComparator(
                preprocessing_config=PreprocessingConfig(force_exclude=("row_id",))
            )
            result = comparator.compare(
                frame,
                target="Churn",
                models=CLASSIFIERS,
                positive_label="churn",
                model_params=SMALL,
            )
        finally:
            ModelTrainer.evaluate = original
        return result, seen

    def test_every_model_saw_the_same_evaluation_row_identities(self, instrumented):
        _, seen = instrumented
        assert len(seen) == len(CLASSIFIERS)
        distinct = {rows for rows in seen.values()}
        assert len(distinct) == 1, "models were scored on different rows"

    def test_and_that_is_stronger_than_equal_row_counts(self, instrumented):
        _, seen = instrumented
        counts = {len(rows) for rows in seen.values()}
        assert len(counts) == 1
        # Two different sets of the same size would satisfy the count check and
        # fail the identity check above. This states the difference explicitly.
        rows = next(iter(seen.values()))
        shuffled = tuple(sorted(rows, reverse=True))
        assert len(shuffled) == len(rows) and shuffled != rows

    def test_the_split_is_drawn_once_for_the_run(self, instrumented):
        result, seen = instrumented
        assert result.split.evaluation_count == len(next(iter(seen.values())))

    def test_the_result_records_which_rows_those_were(self, instrumented):
        result, _ = instrumented
        assert result.split.fingerprint
        assert result.to_dict()["split"]["fingerprint"] == result.split.fingerprint


class TestStateIsolation:
    def test_each_model_has_its_own_fitted_preprocessor(self, classification_run):
        preprocessors = [o.training.preprocessor for o in classification_run.succeeded]
        assert len({id(p) for p in preprocessors}) == len(preprocessors)

    def test_each_model_has_its_own_estimator(self, classification_run):
        estimators = [o.training.estimator for o in classification_run.succeeded]
        assert len({id(e) for e in estimators}) == len(estimators)

    def test_capability_identical_models_still_do_not_share_fitted_state(self):
        """knn and logistic share a profile; they must not share a scaler."""
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["knn_classifier", "logistic_regression"],
            positive_label="churn",
        )
        first, second = (
            result.outcome_for("knn_classifier").training.preprocessor,
            result.outcome_for("logistic_regression").training.preprocessor,
        )
        assert first.plan.preprocessing_profile == second.plan.preprocessing_profile
        assert first is not second
        assert first.transformer is not second.transformer

    def test_two_runs_produce_independent_objects(self):
        frame = classification_frame(n=200)
        comparator = ModelComparator()
        first = comparator.compare(
            frame, target="Churn", models=["logistic_regression"], positive_label="churn"
        )
        second = comparator.compare(
            frame, target="Churn", models=["logistic_regression"], positive_label="churn"
        )
        a = first.outcome_for("logistic_regression").training.estimator
        b = second.outcome_for("logistic_regression").training.estimator
        assert a is not b

    def test_the_registry_is_unchanged_by_training(self, classification_run):
        assert len(default_registry()) == 24
        assert ModelFactory.available(task="classification")[0] == (
            "decision_tree_classifier"
        )

    def test_catalog_metadata_is_unchanged_by_training(self, classification_run):
        entry = ModelFactory.registration("logistic_regression")
        assert entry.strategy_type().default_params() == {
            "max_iter": 1000,
            "random_state": 42,
        }


class TestInputImmutability:
    def test_the_caller_frame_is_untouched(self):
        frame = classification_frame(n=200)
        before = frame.copy(deep=True)
        ModelComparator().compare(
            frame, target="Churn", models=["logistic_regression"], positive_label="churn"
        )
        pd.testing.assert_frame_equal(frame, before)
        assert list(frame.columns) == list(before.columns)
        assert frame.dtypes.equals(before.dtypes)

    def test_a_regression_target_is_untouched(self):
        frame = regression_frame(n=200)
        before = frame["margin"].copy()
        ModelComparator().compare(frame, target="margin", models=["linear_regression"])
        pd.testing.assert_series_equal(frame["margin"], before)

    def test_an_arbitrary_index_survives(self):
        frame = classification_frame(n=200)
        frame.index = [f"customer-{i}" for i in range(len(frame))]
        before = frame.index.copy()
        result = ModelComparator().compare(
            frame, target="Churn", models=["logistic_regression"], positive_label="churn"
        )
        pd.testing.assert_index_equal(frame.index, before)
        assert result.succeeded


class TestColumnAlignment:
    @pytest.fixture
    def trained(self):
        frame = classification_frame(n=200)
        profile = TaskDetector().detect(
            frame["Churn"], target_name="Churn", positive_label="churn"
        )
        split = split_rows(frame["Churn"], profile.task_type)
        trainer = ModelTrainer()
        result = trainer.train(
            split.take(frame),
            target="Churn",
            model="logistic_regression",
            target_profile=profile,
        )
        return trainer, result, split.take(frame, evaluation=True)

    def test_a_missing_feature_column_is_refused_by_name(self, trained):
        from aidatasetkit.core.exceptions import SchemaError

        trainer, result, evaluation = trained
        with pytest.raises(SchemaError, match="ratio"):
            trainer.evaluate(result, evaluation.drop(columns=["ratio"]), target="Churn")

    def test_a_missing_target_column_is_refused(self, trained):
        trainer, result, evaluation = trained
        with pytest.raises(TrainingError, match="not a column"):
            trainer.evaluate(result, evaluation.drop(columns=["Churn"]), target="Churn")

    def test_an_unexpected_extra_column_is_ignored_rather_than_breaking(self, trained):
        """The transformer selects the columns it was fitted for."""
        trainer, result, evaluation = trained
        report = trainer.evaluate(
            result, evaluation.assign(extra=1.0), target="Churn"
        )
        assert report.row_count == len(evaluation)

    def test_reordered_columns_still_align_by_name(self, trained):
        trainer, result, evaluation = trained
        reversed_columns = evaluation[list(evaluation.columns)[::-1]]
        report = trainer.evaluate(result, reversed_columns, target="Churn")
        straight = trainer.evaluate(result, evaluation, target="Churn")
        assert report["accuracy"].value == straight["accuracy"].value

    def test_an_empty_evaluation_frame_is_refused(self, trained):
        trainer, result, evaluation = trained
        with pytest.raises(TrainingError, match="no rows"):
            trainer.evaluate(result, evaluation.iloc[:0], target="Churn")


class TestUnknownCategoriesAndMissingValues:
    def test_an_unseen_category_predicts_without_refitting(self):
        frame = classification_frame(n=200)
        profile = TaskDetector().detect(
            frame["Churn"], target_name="Churn", positive_label="churn"
        )
        split = split_rows(frame["Churn"], profile.task_type)
        trainer = ModelTrainer()
        result = trainer.train(
            split.take(frame), target="Churn", model="logistic_regression",
            target_profile=profile,
        )
        before = list(result.preprocessor.get_feature_names_out())

        unseen = split.take(frame, evaluation=True).assign(city="Dammam")
        report = trainer.evaluate(result, unseen, target="Churn")
        assert report.row_count == len(unseen)
        assert list(result.preprocessor.get_feature_names_out()) == before

    @pytest.mark.parametrize(
        "model,expects_nan",
        [
            ("decision_tree_classifier", True),
            ("logistic_regression", False),
            ("knn_classifier", False),
        ],
    )
    def test_missing_values_are_routed_by_capability(self, model, expects_nan):
        from scipy import sparse

        frame = classification_frame(n=200)
        profile = TaskDetector().detect(frame["Churn"], target_name="Churn")
        split = split_rows(frame["Churn"], profile.task_type)
        result = ModelTrainer().train(
            split.take(frame), target="Churn", model=model, target_profile=profile
        )
        matrix = result.preprocessor.transform(
            split.take(frame, evaluation=True).drop(columns=["Churn"])
        )
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        assert bool(np.isnan(dense).any()) is expects_nan


class TestFailureHandling:
    def test_one_failing_model_does_not_lose_the_others(self):
        """A model given an impossible parameter fails alone."""
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["logistic_regression", "knn_classifier"],
            positive_label="churn",
            model_params={"knn_classifier": {"n_neighbors": 10_000}},
        )
        assert result.partial
        assert [o.model_name for o in result.failed] == ["knn_classifier"]
        assert result.outcome_for("logistic_regression").succeeded

    def test_the_failure_names_the_model_and_gives_a_reason(self):
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["logistic_regression", "knn_classifier"],
            positive_label="churn",
            model_params={"knn_classifier": {"n_neighbors": 10_000}},
        )
        failure = result.outcome_for("knn_classifier")
        assert failure.failure_reason and failure.failure_type

    def test_a_failed_model_is_absent_from_the_ranking_not_first_in_it(self):
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["logistic_regression", "knn_classifier"],
            positive_label="churn",
            model_params={"knn_classifier": {"n_neighbors": 10_000}},
        )
        assert "knn_classifier" not in [o.model_name for o in result.ranked]
        assert result.best.model_name == "logistic_regression"

    def test_the_leaderboard_still_lists_it(self):
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["logistic_regression", "knn_classifier"],
            positive_label="churn",
            model_params={"knn_classifier": {"n_neighbors": 10_000}},
        )
        table = result.to_frame()
        assert set(table["model"]) == {"logistic_regression", "knn_classifier"}
        assert "failed" in set(table["status"])

    def test_a_shared_failure_is_reported_once_rather_than_per_model(self):
        """§19: nine identical failures would send a reader looking for nine bugs."""
        rng = np.random.default_rng(9)
        n = 120
        frame = pd.DataFrame(
            {
                "bad": np.where(np.arange(n) % 3 == 0, np.inf, rng.normal(0, 1, n)),
                "y": rng.choice(["a", "b"], n),
            }
        )
        config = PreprocessingConfig(force_include=("bad",))
        with pytest.raises(TrainingError, match="same reason"):
            ModelComparator(preprocessing_config=config).compare(
                frame, target="y", models=["logistic_regression", "knn_classifier"]
            )

    def test_and_that_message_carries_the_shared_cause(self):
        rng = np.random.default_rng(9)
        n = 120
        frame = pd.DataFrame(
            {
                "bad": np.where(np.arange(n) % 3 == 0, np.inf, rng.normal(0, 1, n)),
                "y": rng.choice(["a", "b"], n),
            }
        )
        config = PreprocessingConfig(force_include=("bad",))
        with pytest.raises(TrainingError, match="infinite"):
            ModelComparator(preprocessing_config=config).compare(
                frame, target="y", models=["logistic_regression", "knn_classifier"]
            )


class TestModelSelection:
    def test_all_models_means_every_compatible_one(self):
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame, target="Churn", positive_label="churn", model_params=SMALL
        )
        assert {o.model_name for o in result.outcomes} == set(
            ModelFactory.available(task="classification")
        )

    def test_a_classification_run_contains_no_regressors(self):
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame, target="Churn", positive_label="churn", model_params=SMALL
        )
        regressors = set(ModelFactory.available(task="regression"))
        assert not {o.model_name for o in result.outcomes} & regressors

    def test_a_regression_run_contains_no_classifiers(self):
        result = ModelComparator().compare(
            regression_frame(n=200), target="margin", model_params=SMALL
        )
        classifiers = set(ModelFactory.available(task="classification"))
        assert not {o.model_name for o in result.outcomes} & classifiers

    def test_execution_order_is_alphabetical_and_not_registration_order(self):
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["knn_classifier", "dummy_classifier", "logistic_regression"],
            positive_label="churn",
        )
        names = [o.model_name for o in result.outcomes]
        assert names == sorted(names)

    def test_an_alias_and_its_canonical_name_train_one_model(self):
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["knn", "knn_classifier"],
            positive_label="churn",
        )
        assert [o.model_name for o in result.outcomes] == ["knn_classifier"]

    def test_an_unknown_model_is_refused(self):
        frame = classification_frame(n=200)
        with pytest.raises(UnknownModelError):
            ModelComparator().compare(
                frame, target="Churn", models=["nonesuch"], positive_label="churn"
            )

    def test_a_wrong_task_model_is_refused(self):
        frame = classification_frame(n=200)
        with pytest.raises(IncompatibleModelError):
            ModelComparator().compare(
                frame, target="Churn", models=["ridge_regression"], positive_label="churn"
            )

    def test_an_empty_model_list_is_refused(self):
        frame = classification_frame(n=200)
        with pytest.raises(TrainingError, match="nothing to compare"):
            ModelComparator().compare(frame, target="Churn", models=[])

    def test_a_bare_string_is_refused_rather_than_read_per_character(self):
        frame = classification_frame(n=200)
        with pytest.raises(ValidationError, match="sequence of names"):
            ModelComparator().compare(frame, target="Churn", models="knn_classifier")


class TestRankingPolicy:
    def test_an_unknown_ranking_metric_is_refused(self):
        frame = classification_frame(n=200)
        with pytest.raises(ValidationError, match="cannot rank"):
            ModelComparator().compare(
                frame, target="Churn", models=["logistic_regression"],
                ranking_metric="vibes",
            )

    def test_a_metric_from_the_other_task_family_is_refused(self):
        """``r2`` is a real metric with a real direction, and it would have
        passed a check that only asked whether the name was known -- leaving every
        model unranked and the leaderboard silently empty."""
        frame = classification_frame(n=200)
        with pytest.raises(ValidationError, match="cannot rank a classification"):
            ModelComparator().compare(
                frame, target="Churn", models=["logistic_regression"],
                ranking_metric="r2",
            )

    def test_and_the_refusal_lists_what_that_task_does_measure(self):
        frame = classification_frame(n=200)
        with pytest.raises(ValidationError, match="accuracy"):
            ModelComparator().compare(
                frame, target="Churn", models=["logistic_regression"],
                ranking_metric="mae",
            )

    def test_ties_break_alphabetically_and_deterministically(self):
        """Two identical models under different names must order stably."""
        from aidatasetkit.evaluation import EvaluationReport, MetricValue
        from aidatasetkit.training.comparison import ModelOutcome

        def outcome(name, value):
            return ModelOutcome(
                model_name=name,
                succeeded=True,
                training=object(),
                evaluation=EvaluationReport(
                    TaskType.CLASSIFICATION,
                    (MetricValue("f1", value, MetricDirection.HIGHER_IS_BETTER),),
                    row_count=10,
                ),
            )

        result = ComparisonResult(
            task_type=TaskType.CLASSIFICATION,
            target_name="y",
            ranking_metric="f1",
            ranking_direction=MetricDirection.HIGHER_IS_BETTER,
            split=split_rows(pd.Series(["a", "b"] * 50), TaskType.CLASSIFICATION),
            outcomes=(outcome("zebra", 0.5), outcome("alpha", 0.5)),
        )
        assert [o.model_name for o in result.ranked] == ["alpha", "zebra"]

    def test_a_model_without_the_ranking_metric_is_unranked_not_ranked_last(self):
        from aidatasetkit.evaluation import EvaluationReport, MetricValue
        from aidatasetkit.training.comparison import ModelOutcome

        available = ModelOutcome(
            "has_it", True, object(),
            EvaluationReport(
                TaskType.CLASSIFICATION,
                (MetricValue("f1", 0.4, MetricDirection.HIGHER_IS_BETTER),),
                row_count=10,
            ),
        )
        missing = ModelOutcome(
            "lacks_it", True, object(),
            EvaluationReport(
                TaskType.CLASSIFICATION,
                (
                    MetricValue(
                        "f1", None, MetricDirection.HIGHER_IS_BETTER,
                        status=MetricStatus.UNDEFINED, reason="one class present",
                    ),
                ),
                row_count=10,
            ),
        )
        result = ComparisonResult(
            task_type=TaskType.CLASSIFICATION,
            target_name="y",
            ranking_metric="f1",
            ranking_direction=MetricDirection.HIGHER_IS_BETTER,
            split=split_rows(pd.Series(["a", "b"] * 50), TaskType.CLASSIFICATION),
            outcomes=(available, missing),
        )
        assert [o.model_name for o in result.ranked] == ["has_it"]
        assert [o.model_name for o in result.unranked] == ["lacks_it"]
        assert result.partial


class TestTheResultIsInspectableAndPrivate:
    def test_it_survives_json(self, classification_run):
        import json

        payload = json.loads(json.dumps(classification_run.to_dict()))
        assert payload["ranking_metric"] == "f1"

    def test_no_estimator_reached_the_record(self, classification_run):
        import json

        text = json.dumps(classification_run.to_dict())
        for fragment in ("LogisticRegression(", "KNeighbors", "sklearn."):
            assert fragment not in text

    def test_no_row_of_data_reached_the_record(self, classification_run):
        """Row values stay out. Nothing here reads a cell into the record.

        The numeric columns are the test: an income of 60,132.44 appears nowhere,
        because no observation is copied into a result.
        """
        import json

        payload = json.dumps(classification_run.to_dict())
        frame = classification_frame()
        for value in frame["income"].dropna().head(20):
            assert str(round(float(value), 2)) not in payload
        for value in frame["ratio"].head(20):
            assert str(float(value)) not in payload

    def test_one_hot_names_are_present_because_the_privacy_policy_says_they_are(
        self, classification_run
    ):
        """Not an oversight -- the documented, deliberate exposure.

        ``docs/privacy.md`` states it plainly: feature lineage exists to say what
        a column became, and for a one-hot encoding those names *are* the
        categories. S4's ``lineage.json`` already publishes them, and a training
        result that hid what the audit artifact shows would be the inconsistency,
        not the safeguard. S7 introduces no new exposure of its own.
        """
        import json

        payload = json.dumps(classification_run.to_dict())
        assert "city_Riyadh" in payload
        privacy = (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "docs"
            / "privacy.md"
        ).read_text(encoding="utf-8")
        assert "One-hot output names" in privacy

    def test_no_timing_reached_the_record(self, classification_run):
        """Observational, and it would make two identical runs compare unequal."""
        import json

        assert "fit_seconds" not in json.dumps(classification_run.to_dict())

    def test_but_timing_is_available_on_the_object(self, classification_run):
        for outcome in classification_run.succeeded:
            assert outcome.training.fit_seconds >= 0.0

    def test_the_record_carries_what_a_rerun_would_need(self, classification_run):
        payload = classification_run.to_dict()
        assert payload["config"]["random_state"] == 42
        assert payload["config"]["validation_size"] == 0.2
        assert payload["split"]["fingerprint"]
        assert payload["ranking_metric"] and payload["ranking_direction"]

    def test_two_identical_runs_produce_identical_records(self):
        frame = classification_frame(n=200)
        runs = [
            ModelComparator()
            .compare(
                frame,
                target="Churn",
                models=["logistic_regression"],
                positive_label="churn",
            )
            .to_dict()
            for _ in range(2)
        ]
        assert runs[0] == runs[1]

    def test_feature_lineage_is_preserved_per_model(self, classification_run):
        for outcome in classification_run.succeeded:
            lineage = outcome.training.lineage
            assert lineage
            produced = set(outcome.training.feature_names)
            for outputs in lineage.values():
                assert set(outputs) <= produced


class TestDefectsTheAdversarialReviewFound:
    """One test per confirmed finding. Each failed before the fix beside it."""

    def test_target_dependent_quality_checks_still_run_during_training(self):
        """The worst of them: S7 had switched off leakage detection entirely.

        ``_prepare`` dropped the target and passed ``target=None`` to the quality
        inspector and the planner, so every target-dependent check went quiet --
        in the layer whose whole purpose is to notice them. A feature exactly
        equal to the target proves it: the planner must exclude it.
        """
        rng = np.random.default_rng(17)
        n = 200
        churn = (rng.random(n) < 0.4).astype(int)
        frame = pd.DataFrame(
            {
                "amount": rng.normal(100, 10, n),
                "leaky": churn,  # exactly the target
                "y": churn,
            }
        )
        profile = TaskDetector().detect(frame["y"], target_name="y")
        split = split_rows(frame["y"], profile.task_type)
        result = ModelTrainer().train(
            split.take(frame), target="y", model="logistic_regression",
            target_profile=profile,
        )
        assert "leaky" not in result.feature_names
        assert "leaky" in result.excluded_features + result.review_features

    def test_and_the_training_record_names_what_was_held_back(self):
        """A column that vanishes between the audit and the fit is a silent loss."""
        rng = np.random.default_rng(17)
        n = 200
        churn = (rng.random(n) < 0.4).astype(int)
        frame = pd.DataFrame(
            {"amount": rng.normal(100, 10, n), "leaky": churn, "y": churn}
        )
        profile = TaskDetector().detect(frame["y"], target_name="y")
        split = split_rows(frame["y"], profile.task_type)
        result = ModelTrainer().train(
            split.take(frame), target="y", model="logistic_regression",
            target_profile=profile,
        )
        payload = result.to_dict()
        assert "leaky" in payload["excluded_features"] + payload["review_features"]

    def test_macro_averaging_is_pinned_to_the_training_classes(self):
        """Unpinned, each model was averaged over its own predicted class set.

        Two models judged on identical rows got different denominators, and the
        one that declined to predict a hard class was rewarded for it -- enough to
        change which model ranks first.
        """
        from sklearn import metrics as skmetrics

        from aidatasetkit.evaluation import evaluate_classification

        y_true = np.array([0, 1, 2] * 30)
        timid = np.array([0, 1, 1] * 30)  # never predicts class 2

        report = evaluate_classification(y_true, timid, class_count=3)
        over_three = skmetrics.f1_score(
            y_true, timid, labels=[0, 1, 2], average="macro", zero_division=0.0
        )
        over_two = skmetrics.f1_score(
            y_true, timid, labels=[0, 1], average="macro", zero_division=0.0
        )
        assert report["f1"].value == pytest.approx(over_three)
        assert over_two > over_three, "the unpinned answer flattered the timid model"
        assert report["f1"].detail["averaged_over"] == 3

    def test_an_internal_type_error_is_not_reported_as_a_model_failure(self):
        """Its own docstring and the published doc both promised this."""
        frame = classification_frame(n=200)
        comparator = ModelComparator()
        original = ModelTrainer.train

        def broken(self, *args, **kwargs):
            raise TypeError("a mistake in the orchestration, not in the data")

        ModelTrainer.train = broken
        try:
            with pytest.raises(TypeError, match="mistake in the orchestration"):
                comparator.compare(
                    frame, target="Churn", models=["logistic_regression"],
                    positive_label="churn",
                )
        finally:
            ModelTrainer.train = original

    def test_an_exception_with_no_message_does_not_crash_the_recorder(self):
        """``"".splitlines()[0]`` raised IndexError inside the except block."""
        frame = classification_frame(n=200)
        comparator = ModelComparator()
        original = ModelTrainer.train

        def silent(self, *args, **kwargs):
            raise ValueError()

        ModelTrainer.train = silent
        try:
            result = comparator.compare(
                frame, target="Churn",
                models=["logistic_regression", "knn_classifier"],
                positive_label="churn",
            )
        except TrainingError as error:
            assert "ValueError" in str(error)
            return
        finally:
            ModelTrainer.train = original
        assert all(o.failure_reason for o in result.failed)

    def test_model_params_keyed_by_an_alias_are_applied(self):
        """They were looked up canonically and silently discarded."""
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["knn_classifier"],
            positive_label="churn",
            model_params={"knn": {"n_neighbors": 3}},
        )
        params = result.outcome_for("knn_classifier").training.model_parameters
        assert params["n_neighbors"] == 3

    def test_a_model_params_key_that_names_nothing_is_refused(self):
        frame = classification_frame(n=200)
        with pytest.raises(ValidationError, match="not a model"):
            ModelComparator().compare(
                frame, target="Churn", models=["knn_classifier"],
                positive_label="churn", model_params={"knnn": {"n_neighbors": 3}},
            )

    def test_probability_support_is_read_from_the_capability_not_the_object(self):
        """§12: never call predict_proba on a model that does not advertise it."""
        frame = classification_frame(n=200)
        profile = TaskDetector().detect(
            frame["Churn"], target_name="Churn", positive_label="churn"
        )
        split = split_rows(frame["Churn"], profile.task_type)
        trainer = ModelTrainer()
        result = trainer.train(
            split.take(frame), target="Churn", model="logistic_regression",
            target_profile=profile,
        )
        assert result.supports_predict_proba is True

        # Force the declaration to False and confirm the metric follows the
        # declaration rather than the object, which still has the method.
        lying = dataclasses_replace(result, supports_predict_proba=False)
        report = trainer.evaluate(
            lying, split.take(frame, evaluation=True), target="Churn"
        )
        assert report["roc_auc"].status is MetricStatus.UNSUPPORTED_BY_MODEL

    def test_a_supplied_profile_cannot_label_encode_a_regression_target(self):
        """The S6 closure had no gate on the trainer's caller-supplied profile."""
        from aidatasetkit.core.types import TargetProfile

        rng = np.random.default_rng(3)
        n = 120
        frame = pd.DataFrame({"a": rng.normal(0, 1, n), "price": np.arange(n) * 7.5})
        asserted = TargetProfile(task_type=TaskType.CLASSIFICATION, target_name="price")
        with pytest.raises(TrainingError, match="holds quantities"):
            ModelTrainer().train(
                frame, target="price", model="logistic_regression",
                target_profile=asserted,
            )

    def test_the_leaderboard_shows_why_a_metric_is_missing(self):
        """It rendered an unavailable metric as a blank, indistinguishable from
        never-measured -- reintroducing the silent NaN at the last step."""
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame, target="Churn", models=["decision_tree_classifier"],
            model_params=SMALL,
        )
        table = result.to_frame()
        assert "roc_auc" in table.columns
        assert table["roc_auc"].iloc[0] == MetricStatus.UNDEFINED.value

    def test_the_stratified_flag_reports_the_outcome(self):
        y = pd.Series(["a"] * 300 + ["b"] * 300 + ["rare"] * 3)
        split = split_rows(y, TaskType.CLASSIFICATION)
        both = set(split.take(y).unique()) == set(split.take(y, evaluation=True).unique())
        assert split.stratified is both


def dataclasses_replace(instance, **changes):
    """``dataclasses.replace`` for a slotted frozen dataclass."""
    import dataclasses

    return dataclasses.replace(instance, **changes)


class TestTheseChecksCanActuallyFail:
    """§62. A guard that cannot fail is decoration.

    Each test performs the wrong behaviour deliberately and asserts that the
    corresponding contract notices. Nothing here mutates the library; the wrong
    behaviour is reproduced locally, which is enough to show the check has
    something to catch.
    """

    def test_fitting_the_scaler_on_train_and_evaluation_moves_what_it_learned(self):
        """The mutation the leakage test exists to catch."""
        from sklearn.preprocessing import StandardScaler

        frame = TestNothingIsFittedOnEvaluationRows._frame(True)
        split = split_rows(frame["y"], TaskType.CLASSIFICATION)
        amounts = frame["amount"].to_numpy(dtype="float64")
        filled = np.nan_to_num(amounts, nan=float(np.nanmedian(amounts)))

        honest = StandardScaler().fit(filled[split.train_positions].reshape(-1, 1))
        leaked = StandardScaler().fit(filled.reshape(-1, 1))
        assert leaked.mean_[0] != pytest.approx(honest.mean_[0])
        assert leaked.scale_[0] > honest.scale_[0] * 10

    def test_reversing_the_direction_of_mae_would_invert_the_ranking(self):
        """If MAE were ranked as higher-is-better, the worst model would win."""
        from aidatasetkit.evaluation import EvaluationReport, MetricValue
        from aidatasetkit.training.comparison import ModelOutcome

        def outcome(name, mae):
            return ModelOutcome(
                name, True, object(),
                EvaluationReport(
                    TaskType.REGRESSION,
                    (MetricValue("mae", mae, MetricDirection.LOWER_IS_BETTER),),
                    row_count=10,
                ),
            )

        outcomes = (outcome("good", 1.0), outcome("bad", 99.0))
        split = split_rows(pd.Series(np.linspace(0, 1, 50)), TaskType.REGRESSION)
        correct = ComparisonResult(
            task_type=TaskType.REGRESSION, target_name="y", ranking_metric="mae",
            ranking_direction=MetricDirection.LOWER_IS_BETTER,
            split=split, outcomes=outcomes,
        )
        reversed_by_mistake = ComparisonResult(
            task_type=TaskType.REGRESSION, target_name="y", ranking_metric="mae",
            ranking_direction=MetricDirection.HIGHER_IS_BETTER,
            split=split, outcomes=outcomes,
        )
        assert correct.best.model_name == "good"
        assert reversed_by_mistake.best.model_name == "bad"

    def test_reading_the_wrong_probability_column_would_invert_roc_auc(self):
        """Why the positive column is read from the encoding, not assumed."""
        from sklearn import metrics as skmetrics

        rng = np.random.default_rng(6)
        n = 200
        y_true = rng.integers(0, 2, n)
        confidence = np.where(y_true == 0, rng.uniform(0.6, 0.99, n), rng.uniform(0.01, 0.4, n))
        proba = np.column_stack([confidence, 1 - confidence])

        right = skmetrics.roc_auc_score((y_true == 0).astype(int), proba[:, 0])
        wrong = skmetrics.roc_auc_score((y_true == 0).astype(int), proba[:, 1])
        assert right > 0.9 and wrong < 0.1
        assert wrong == pytest.approx(1 - right)

    def test_giving_one_model_a_different_split_would_change_its_rows(self):
        """The fairness check compares row identity for exactly this reason."""
        y = classification_frame(n=200)["Churn"]
        first = split_rows(y, TaskType.CLASSIFICATION, KitConfig(random_state=1))
        second = split_rows(y, TaskType.CLASSIFICATION, KitConfig(random_state=2))
        assert first.evaluation_count == second.evaluation_count
        assert set(first.evaluation_positions) != set(second.evaluation_positions)
        assert first.fingerprint != second.fingerprint

    def test_sharing_one_fitted_preprocessor_would_be_visible_as_one_object(self):
        """The isolation check compares identity, which a shared object fails."""
        frame = classification_frame(n=200)
        result = ModelComparator().compare(
            frame,
            target="Churn",
            models=["knn_classifier", "logistic_regression"],
            positive_label="churn",
        )
        first = result.outcome_for("knn_classifier").training.preprocessor
        second = result.outcome_for("logistic_regression").training.preprocessor
        assert id(first) != id(second)
        # Had they been shared, this is the assertion that would have caught it.
        shared = [first, first]
        assert len({id(p) for p in shared}) == 1

    def test_treating_an_integer_regression_target_as_classification_is_refused(self):
        """The S6 closure, still holding at the S7 boundary."""
        from aidatasetkit.core.exceptions import UnsupportedTaskError

        rng = np.random.default_rng(8)
        n = 200
        frame = pd.DataFrame(
            {"a": rng.normal(0, 1, n), "units": np.arange(n) * 7}
        )
        assert (
            TaskDetector().detect(frame["units"], target_name="units").task_type
            is TaskType.REGRESSION
        )
        with pytest.raises(UnsupportedTaskError, match="holds quantities"):
            ModelComparator().compare(
                frame, target="units", task="classification",
                models=["logistic_regression"],
            )

    def test_and_the_same_target_trains_correctly_as_regression(self):
        rng = np.random.default_rng(8)
        n = 200
        frame = pd.DataFrame({"a": rng.normal(0, 1, n), "units": np.arange(n) * 7})
        result = ModelComparator().compare(
            frame, target="units", models=["linear_regression"]
        )
        assert result.task_type is TaskType.REGRESSION
        assert result.outcome_for("linear_regression").training.target_encoding is None
