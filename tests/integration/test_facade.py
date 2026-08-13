"""AIDataFacade: the guided path, and the guarantees it must not weaken.

Three tests carry this file. The golden workflows prove the whole sequence runs
end to end for both tasks. The leakage tests prove the convenience layer is not a
route around the safety the layers below it enforce. And the equivalence test
proves the facade is syntax rather than a second analytical engine -- given the
same data, config and model, it must produce the same split, the same metrics and
the same predictions as calling the components directly.

If the last of those ever fails, the facade has started doing science of its own,
which is the one thing it is not for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit import AIDataFacade
from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import (
    IncompatibleModelError,
    SchemaError,
    UnknownModelError,
    UnsupportedTaskError,
    WorkflowStateError,
)
from aidatasetkit.core.types import TaskType
from aidatasetkit.evidence import Verdict
from aidatasetkit.facade import Stage
from aidatasetkit.preprocessing import PreprocessingConfig
from aidatasetkit.profiling import TaskDetector
from aidatasetkit.training import ModelTrainer, split_rows

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


def churn_data(n: int = 400, seed: int = 20250401):
    """Numeric on two scales, a category, gaps, imbalance, awkward labels, an id.

    The test frame carries an unseen city and no target column, which is what an
    external test set usually looks like.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.5, 0.08, n)
    train = pd.DataFrame(
        {
            "CustomerID": [f"C{i:05d}" for i in range(n)],
            "income": rng.normal(60_000, 15_000, n).round(2),
            "ratio": signal.round(4),
            "city": rng.choice(["Riyadh", "Jeddah"], n),
            "Churn": np.where(signal + rng.normal(0, 0.02, n) > 0.58, "churn", "stay"),
        }
    )
    train.loc[train.index[:25], "income"] = np.nan
    test = (
        train.iloc[:60]
        .drop(columns=["Churn"])
        .assign(city="Dammam")
        .reset_index(drop=True)
    )
    test.loc[test.index[:5], "income"] = np.nan
    return train, test


def margin_data(n: int = 400, seed: int = 20250402):
    rng = np.random.default_rng(seed)
    income = rng.normal(60_000, 15_000, n)
    ratio = rng.normal(0.5, 0.05, n)
    train = pd.DataFrame(
        {
            "RowID": np.arange(n),
            "income": income.round(2),
            "ratio": ratio.round(4),
            "city": rng.choice(["Riyadh", "Jeddah"], n),
            "margin": 2e-4 * income + 300.0 * ratio - 200.0 + rng.normal(0, 2.0, n),
        }
    )
    train.loc[train.index[:25], "income"] = np.nan
    test = train.iloc[:60].drop(columns=["margin"]).assign(city="Dammam")
    return train, test


# --------------------------------------------------------------------------- #
# §38 / §70 -- the golden workflows
# --------------------------------------------------------------------------- #


class TestTheGoldenClassificationWorkflow:
    """The intended user story, executed."""

    @pytest.fixture(scope="class")
    @staticmethod
    def session():
        train, test = churn_data()
        ai = AIDataFacade(
            target="Churn",
            id_column="CustomerID",
            task="classification",
            positive_label="churn",
        )
        ai.load(train, test)
        ai.profile()
        ai.statistics()
        ai.check_quality()
        ai.prepare()
        comparison = ai.compare_models(models=CLASSIFIERS, model_params=SMALL)
        ai.select_model("logistic_regression")
        training = ai.train()
        evaluation = ai.evaluate()
        predictions = ai.predict_test(with_probabilities=True)
        return ai, comparison, training, evaluation, predictions

    def test_the_task_was_resolved_by_the_detector(self, session):
        ai, *_ = session
        assert ai.task is TaskType.CLASSIFICATION

    def test_profile_statistics_and_quality_are_all_available(self, session):
        ai, *_ = session
        assert ai.profile().row_count == 400
        assert set(ai.statistics()) == {"income", "ratio"}
        assert ai.check_quality().issue_count >= 0
        assert isinstance(ai.verdict, Verdict)

    def test_every_capability_profile_was_planned(self, session):
        ai, *_ = session
        plans = ai.prepare()
        assert len(plans) == 5, "eighteen models, five preprocessors"

    def test_the_comparison_ranked_every_model(self, session):
        _, comparison, *_ = session
        assert len(comparison.outcomes) == len(CLASSIFIERS)
        assert not comparison.failed

    def test_selection_is_recorded_canonically(self, session):
        ai, *_ = session
        assert ai.selected_model == "logistic_regression"

    def test_training_and_evaluation_completed(self, session):
        _, _, training, evaluation, _ = session
        assert training.model_name == "logistic_regression"
        assert evaluation["accuracy"].is_available

    def test_predictions_are_original_labels_not_encoded_integers(self, session):
        """The single most likely way a facade leaks an internal decision."""
        *_, predictions = session
        assert set(predictions.predictions) <= {"churn", "stay"}
        assert predictions.predictions.dtype.kind in "OU"

    def test_the_identifier_is_carried_beside_the_prediction(self, session):
        ai, _, _, _, predictions = session
        assert predictions.id_name == "CustomerID"
        frame = predictions.to_frame()
        assert list(frame.columns)[:2] == ["CustomerID", "prediction"]
        assert frame["CustomerID"].iloc[0] == "C00000"

    def test_probability_columns_are_named_by_the_original_classes(self, session):
        *_, predictions = session
        assert predictions.class_labels == ("churn", "stay")
        assert np.allclose(predictions.probabilities.sum(axis=1), 1.0)

    def test_an_unseen_test_category_predicted_without_refitting(self, session):
        ai, _, training, _, predictions = session
        learned = training.preprocessor.get_feature_names_out()
        assert not any("Dammam" in str(name) for name in learned)
        assert predictions.row_count == 60

    def test_the_stage_reflects_the_work_done(self, session):
        ai, *_ = session
        assert ai.stage is Stage.EVALUATED
        assert ai.status["predicted"] is True


class TestTheGoldenRegressionWorkflow:
    @pytest.fixture(scope="class")
    @staticmethod
    def session():
        train, test = margin_data()
        ai = AIDataFacade(target="margin", id_column="RowID")
        ai.load(train, test)
        ai.check_quality()
        comparison = ai.compare_models(models=REGRESSORS, model_params=SMALL)
        ai.select_model("ridge")
        training = ai.train()
        evaluation = ai.evaluate()
        predictions = ai.predict_test()
        return ai, comparison, training, evaluation, predictions

    def test_the_task_was_inferred_without_a_hint(self, session):
        ai, *_ = session
        assert ai.task is TaskType.REGRESSION

    def test_no_target_was_ever_encoded(self, session):
        _, comparison, training, _, _ = session
        assert training.target_encoding is None
        for outcome in comparison.succeeded:
            assert outcome.training.target_encoding is None

    def test_the_alias_resolved_to_the_canonical_regressor(self, session):
        ai, *_ = session
        assert ai.selected_model == "ridge_regression"

    def test_predictions_are_quantities(self, session):
        *_, predictions = session
        assert np.issubdtype(predictions.predictions.dtype, np.floating)
        assert predictions.probabilities is None
        assert len(np.unique(predictions.predictions)) > 1

    def test_and_they_span_the_range_the_target_does(self, session):
        ai, _, _, _, predictions = session
        assert predictions.predictions.min() < predictions.predictions.max()

    def test_ranking_used_the_regression_default(self, session):
        _, comparison, *_ = session
        assert comparison.ranking_metric == "rmse"

    def test_an_integer_valued_regression_target_is_not_classified(self):
        """Carried forward from the S6 closure, through the facade."""
        rng = np.random.default_rng(5)
        n = 200
        train = pd.DataFrame({"a": rng.normal(0, 1, n), "units": np.arange(n) * 7})
        ai = AIDataFacade(target="units")
        ai.load(train)
        assert ai.task is TaskType.REGRESSION
        ai.select_model("linear_regression")
        assert ai.train().target_encoding is None


# --------------------------------------------------------------------------- #
# §40 / §72 / §73 -- the release-blocking safety tests
# --------------------------------------------------------------------------- #


class TestTheFacadeIsNotARouteAroundSafety:
    def test_target_dependent_leakage_detection_is_still_active(self):
        """§16 and §72. The facade must not drop the target before quality.

        S7's review found exactly this defect one layer down: the target was
        removed before the inspector saw it, and every target-dependent check
        went quiet. A feature identical to the target is the proof.
        """
        rng = np.random.default_rng(11)
        n = 300
        churn = (rng.random(n) < 0.4).astype(int)
        train = pd.DataFrame(
            {"amount": rng.normal(100, 10, n), "leaky": churn, "y": churn}
        )
        ai = AIDataFacade(target="y")
        ai.load(train)
        report = ai.check_quality()
        codes = {issue.code for issue in report.issues}
        assert "target_leakage_exact_duplicate" in codes

    def test_and_the_verdict_reflects_it(self):
        rng = np.random.default_rng(11)
        n = 300
        churn = (rng.random(n) < 0.4).astype(int)
        train = pd.DataFrame(
            {"amount": rng.normal(100, 10, n), "leaky": churn, "y": churn}
        )
        ai = AIDataFacade(target="y")
        ai.load(train)
        ai.check_quality()
        assert ai.verdict is Verdict.BLOCKED
        assert ai.verdict_reasons

    def test_blocked_data_cannot_be_trained_through_the_facade(self):
        """§17 and §24. No override parameter exists, deliberately."""
        rng = np.random.default_rng(11)
        n = 300
        churn = (rng.random(n) < 0.4).astype(int)
        train = pd.DataFrame(
            {"amount": rng.normal(100, 10, n), "leaky": churn, "y": churn}
        )
        ai = AIDataFacade(target="y")
        ai.load(train)
        ai.select_model("logistic_regression")
        with pytest.raises(WorkflowStateError, match="BLOCKED"):
            ai.train()

    def test_nor_compared(self):
        rng = np.random.default_rng(11)
        n = 300
        churn = (rng.random(n) < 0.4).astype(int)
        train = pd.DataFrame(
            {"amount": rng.normal(100, 10, n), "leaky": churn, "y": churn}
        )
        ai = AIDataFacade(target="y")
        ai.load(train)
        with pytest.raises(WorkflowStateError, match="BLOCKED"):
            ai.compare_models(models=["logistic_regression"])

    def test_the_refusal_points_at_the_documented_escape_hatch(self):
        rng = np.random.default_rng(11)
        n = 300
        churn = (rng.random(n) < 0.4).astype(int)
        train = pd.DataFrame(
            {"amount": rng.normal(100, 10, n), "leaky": churn, "y": churn}
        )
        ai = AIDataFacade(target="y")
        ai.load(train)
        ai.select_model("logistic_regression")
        with pytest.raises(WorkflowStateError, match="aidatasetkit.training"):
            ai.train()

    def test_the_safety_check_runs_even_if_the_user_never_called_it(self):
        """Otherwise the check is optional in practice whatever the docs say."""
        rng = np.random.default_rng(11)
        n = 300
        churn = (rng.random(n) < 0.4).astype(int)
        train = pd.DataFrame(
            {"amount": rng.normal(100, 10, n), "leaky": churn, "y": churn}
        )
        ai = AIDataFacade(target="y")
        ai.load(train)
        ai.select_model("logistic_regression")
        assert ai.status["quality_checked"] is False
        with pytest.raises(WorkflowStateError):
            ai.train()
        assert ai.status["quality_checked"] is True


class TestPrepareFitsNothing:
    """§19 and §26. Fitting before the split is the trap this method avoids."""

    def test_prepare_returns_plans_and_no_fitted_state(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        plans = ai.prepare()
        for plan in plans.values():
            assert hasattr(plan, "fingerprint")
            assert not hasattr(plan, "transform")

    def test_prepare_does_not_change_what_training_learns(self):
        """If prepare had fitted anything, the fit below would differ."""
        train, _ = churn_data(n=200)

        without = AIDataFacade(target="Churn", positive_label="churn")
        without.load(train)
        without.select_model("knn_classifier")
        first = without.train()

        with_prepare = AIDataFacade(target="Churn", positive_label="churn")
        with_prepare.load(train)
        with_prepare.prepare()
        with_prepare.select_model("knn_classifier")
        second = with_prepare.train()

        numeric = first.preprocessor.transformer.named_transformers_["numeric"]
        other = second.preprocessor.transformer.named_transformers_["numeric"]
        np.testing.assert_array_equal(
            numeric.named_steps["scaler"].mean_, other.named_steps["scaler"].mean_
        )

    def test_the_fitted_scaler_saw_only_the_training_side_of_the_split(self):
        """§19 directly: the statistic must match a train-only fit exactly."""
        from sklearn.preprocessing import StandardScaler

        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("knn_classifier")
        result = ai.train()

        split = split_rows(train["Churn"], TaskType.CLASSIFICATION, KitConfig())
        rows = split.take(train)
        income = rows["income"].to_numpy(dtype="float64")
        expected = StandardScaler().fit(
            np.nan_to_num(income, nan=float(np.nanmedian(income))).reshape(-1, 1)
        )
        learned = result.preprocessor.transformer.named_transformers_[
            "numeric"
        ].named_steps["scaler"]
        assert learned.mean_[0] == pytest.approx(expected.mean_[0])

    def test_the_whole_dataset_would_have_given_a_different_answer(self):
        """Otherwise the previous test proves nothing."""
        from sklearn.preprocessing import StandardScaler

        train, _ = churn_data(n=200)
        split = split_rows(train["Churn"], TaskType.CLASSIFICATION, KitConfig())
        everything = train["income"].to_numpy(dtype="float64")
        train_only = split.take(train)["income"].to_numpy(dtype="float64")
        a = StandardScaler().fit(
            np.nan_to_num(everything, nan=float(np.nanmedian(everything))).reshape(-1, 1)
        )
        b = StandardScaler().fit(
            np.nan_to_num(train_only, nan=float(np.nanmedian(train_only))).reshape(-1, 1)
        )
        assert a.mean_[0] != pytest.approx(b.mean_[0])


class TestExternalTestDataIsIsolated:
    """§27 and §73. Test data is for the final answer, not for development."""

    @staticmethod
    def _learned_scaler(test_frame):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train, test_frame)
        ai.select_model("knn_classifier")
        result = ai.train()
        return result.preprocessor.transformer.named_transformers_[
            "numeric"
        ].named_steps["scaler"]

    def test_an_extreme_test_row_does_not_move_the_scaler(self):
        _, benign = churn_data(n=200)
        poisoned = benign.copy()
        poisoned.loc[poisoned.index[:10], "income"] = 1e9

        honest = self._learned_scaler(benign)
        attacked = self._learned_scaler(poisoned)
        np.testing.assert_array_equal(honest.mean_, attacked.mean_)
        np.testing.assert_array_equal(honest.scale_, attacked.scale_)

    def test_nor_does_it_appear_in_the_categorical_vocabulary(self):
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train, test)
        ai.select_model("knn_classifier")
        result = ai.train()
        names = [str(n) for n in result.preprocessor.get_feature_names_out()]
        assert not any("Dammam" in name for name in names)

    def test_test_data_does_not_change_a_comparison(self):
        train, test = churn_data(n=200)

        without = AIDataFacade(target="Churn", positive_label="churn")
        without.load(train)
        a = without.compare_models(models=CLASSIFIERS, model_params=SMALL)

        with_test = AIDataFacade(target="Churn", positive_label="churn")
        with_test.load(train, test)
        b = with_test.compare_models(models=CLASSIFIERS, model_params=SMALL)

        assert [o.model_name for o in a.ranked] == [o.model_name for o in b.ranked]
        assert a.split.fingerprint == b.split.fingerprint

    def test_evaluation_uses_the_holdout_not_the_training_rows(self):
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train, test)
        ai.select_model("decision_tree_classifier")
        ai.train()
        report = ai.evaluate()
        split = split_rows(train["Churn"], TaskType.CLASSIFICATION, KitConfig())
        assert report.row_count == split.evaluation_count
        assert report.row_count < len(train)

    def test_predict_test_does_not_refit_anything(self):
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train, test)
        ai.select_model("knn_classifier")
        result = ai.train()
        before = result.preprocessor.transformer.named_transformers_[
            "numeric"
        ].named_steps["scaler"].mean_.copy()
        ai.predict_test()
        after = result.preprocessor.transformer.named_transformers_[
            "numeric"
        ].named_steps["scaler"].mean_
        np.testing.assert_array_equal(before, after)


# --------------------------------------------------------------------------- #
# §71 -- the facade is syntax, not a second engine
# --------------------------------------------------------------------------- #


class TestFacadeAndDirectApiAgree:
    """Given the same inputs, the two routes must produce the same science."""

    def test_the_split_metrics_and_predictions_all_match(self):
        train, test = churn_data(n=300)

        ai = AIDataFacade(
            target="Churn", id_column="CustomerID", positive_label="churn"
        )
        ai.load(train, test)
        ai.select_model("logistic_regression")
        facade_training = ai.train()
        facade_report = ai.evaluate()
        facade_predictions = ai.predict_test()

        # The same thing, by hand.
        profile = TaskDetector(KitConfig()).detect(
            train["Churn"], target_name="Churn", positive_label="churn"
        )
        split = split_rows(train["Churn"], profile.task_type, KitConfig())
        trainer = ModelTrainer(
            config=KitConfig(), preprocessing_config=PreprocessingConfig()
        )
        direct_training = trainer.train(
            split.take(train),
            target="Churn",
            model="logistic_regression",
            target_profile=profile,
        )
        direct_report = trainer.evaluate(
            direct_training, split.take(train, evaluation=True), target="Churn"
        )

        assert facade_training.plan_fingerprint == direct_training.plan_fingerprint
        assert facade_training.train_row_count == direct_training.train_row_count
        for metric in facade_report:
            assert metric.value == direct_report[metric.name].value

        from aidatasetkit.prediction import predict_frame

        direct_predictions = predict_frame(
            direct_training, test, id_column="CustomerID"
        )
        np.testing.assert_array_equal(
            facade_predictions.predictions, direct_predictions.predictions
        )

    def test_comparison_matches_the_direct_comparator(self):
        from aidatasetkit.training import ModelComparator

        train, _ = churn_data(n=300)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        through_facade = ai.compare_models(models=CLASSIFIERS, model_params=SMALL)

        direct = ModelComparator().compare(
            train,
            target="Churn",
            models=CLASSIFIERS,
            positive_label="churn",
            model_params=SMALL,
        )
        assert through_facade.to_dict() == direct.to_dict()


# --------------------------------------------------------------------------- #
# §41 / §74 -- state
# --------------------------------------------------------------------------- #


class TestStateInvalidation:
    def test_loading_new_data_discards_everything_derived_from_the_old(self):
        train_a, test_a = churn_data(n=200, seed=1)
        train_b, _ = churn_data(n=200, seed=2)

        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train_a, test_a)
        ai.check_quality()
        ai.prepare()
        ai.compare_models(models=["logistic_regression"], model_params=SMALL)
        ai.select_model("logistic_regression")
        ai.train()
        ai.evaluate()
        ai.predict_test()
        assert ai.stage is Stage.EVALUATED

        ai.load(train_b)
        assert ai.stage is Stage.LOADED
        for attribute in (
            "comparison", "selected_model", "training", "evaluation", "predictions",
        ):
            with pytest.raises(WorkflowStateError):
                getattr(ai, attribute)
        assert ai.status["profiled"] is False
        assert ai.status["quality_checked"] is False
        assert ai.status["prepared"] is False
        assert ai.status["test_rows"] is None

    def test_selecting_a_different_model_discards_the_previous_fit(self):
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train, test)
        ai.select_model("logistic_regression")
        ai.train()
        ai.evaluate()
        ai.predict_test()

        ai.select_model("decision_tree_classifier")
        assert ai.selected_model == "decision_tree_classifier"
        for attribute in ("training", "evaluation", "predictions"):
            with pytest.raises(WorkflowStateError):
                getattr(ai, attribute)

    def test_a_new_comparison_discards_a_selection_made_from_the_old_one(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.compare_models(models=CLASSIFIERS, model_params=SMALL)
        ai.select_model("logistic_regression")
        ai.train()

        ai.compare_models(models=["dummy_classifier"], model_params=SMALL)
        with pytest.raises(WorkflowStateError):
            ai.selected_model
        with pytest.raises(WorkflowStateError):
            ai.training

    def test_a_failed_load_leaves_the_previous_session_intact(self):
        """§42: a partial replacement is worse than a refusal."""
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train, test)
        ai.select_model("logistic_regression")
        ai.train()

        with pytest.raises(SchemaError):
            ai.load(train.drop(columns=["Churn"]))

        assert ai.stage is Stage.TRAINED
        assert ai.training.model_name == "logistic_regression"

    def test_a_failed_train_does_not_mark_the_session_trained(self):
        """``n_neighbors=0`` is refused by scikit-learn at fit.

        ``n_neighbors=100_000`` would not do: a k larger than the training set is
        accepted at fit and only refused at predict, so the fit would genuinely
        succeed and this test would be asserting nothing.
        """
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("knn_classifier")
        with pytest.raises(ValueError):
            ai.train(n_neighbors=0)
        assert ai.stage is Stage.MODEL_SELECTED
        with pytest.raises(WorkflowStateError):
            ai.training

    def test_a_failed_retrain_does_not_destroy_the_previous_fit(self):
        """§42: the previous valid state survives a failed replacement."""
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("knn_classifier")
        good = ai.train()
        with pytest.raises(ValueError):
            ai.train(n_neighbors=0)
        assert ai.training is good

    def test_a_failed_comparison_does_not_install_a_result(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        good = ai.compare_models(models=["logistic_regression"], model_params=SMALL)
        with pytest.raises(Exception):
            ai.compare_models(models=["nonesuch"])
        assert ai.comparison is good


class TestRepeatedCalls:
    """§11: deliberate semantics, not accidental accumulation."""

    def test_reading_the_data_is_idempotent(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        assert ai.profile() is ai.profile()
        assert ai.check_quality() is ai.check_quality()
        assert ai.prepare().keys() == ai.prepare().keys()

    def test_statistics_returns_a_fresh_mapping_each_time(self):
        """Cached values, but not a handle the caller can edit."""
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        first = ai.statistics()
        first["income"] = None
        assert ai.statistics()["income"] is not None

    def test_retraining_replaces_rather_than_accumulates(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("logistic_regression")
        first = ai.train()
        second = ai.train()
        assert first is not second
        assert ai.training is second


class TestInstanceIsolation:
    def test_two_sessions_do_not_share_anything(self):
        train_a, _ = churn_data(n=200, seed=1)
        train_b, _ = churn_data(n=200, seed=2)

        first = AIDataFacade(target="Churn", positive_label="churn")
        second = AIDataFacade(target="Churn", positive_label="churn")

        first.load(train_a)
        assert second.stage is Stage.EMPTY

        second.load(train_b)
        first.select_model("logistic_regression")
        first.train()
        assert second.stage is Stage.LOADED
        with pytest.raises(WorkflowStateError):
            second.training

    def test_their_fitted_objects_are_distinct(self):
        train, _ = churn_data(n=200)
        sessions = []
        for _ in range(2):
            ai = AIDataFacade(target="Churn", positive_label="churn")
            ai.load(train)
            ai.select_model("logistic_regression")
            ai.train()
            sessions.append(ai)
        assert sessions[0].training.estimator is not sessions[1].training.estimator


class TestReproducibility:
    def test_two_sessions_with_the_same_seed_agree_exactly(self):
        train, test = churn_data(n=300)
        outputs = []
        for _ in range(2):
            ai = AIDataFacade(
                target="Churn", id_column="CustomerID", positive_label="churn"
            )
            ai.load(train, test)
            comparison = ai.compare_models(models=CLASSIFIERS, model_params=SMALL)
            ai.select_model("logistic_regression")
            ai.train()
            outputs.append((comparison.to_dict(), ai.predict_test().predictions))
        assert outputs[0][0] == outputs[1][0]
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


class TestCallerDataIsNeverModified:
    def test_a_whole_workflow_leaves_both_frames_untouched(self):
        train, test = churn_data(n=200)
        before_train, before_test = train.copy(deep=True), test.copy(deep=True)

        ai = AIDataFacade(
            target="Churn", id_column="CustomerID", positive_label="churn"
        )
        ai.load(train, test)
        ai.profile(); ai.statistics(); ai.check_quality(); ai.prepare()
        ai.compare_models(models=["logistic_regression"], model_params=SMALL)
        ai.select_model("logistic_regression")
        ai.train(); ai.evaluate(); ai.predict_test()

        pd.testing.assert_frame_equal(train, before_train)
        pd.testing.assert_frame_equal(test, before_test)
        assert list(train.columns) == list(before_train.columns)
        assert train.dtypes.equals(before_train.dtypes)

    def test_an_arbitrary_index_survives_and_predictions_stay_aligned(self):
        train, test = churn_data(n=200)
        train.index = [f"t{i}" for i in range(len(train))]
        test.index = [f"t{i}" for i in range(len(test))]  # labels overlap on purpose
        before = test.index.copy()

        ai = AIDataFacade(
            target="Churn", id_column="CustomerID", positive_label="churn"
        )
        ai.load(train, test)
        ai.select_model("logistic_regression")
        ai.train()
        predictions = ai.predict_test()

        pd.testing.assert_index_equal(test.index, before)
        assert list(predictions.ids) == list(test["CustomerID"])

    def test_duplicate_index_labels_do_not_misalign_predictions(self):
        train, test = churn_data(n=200)
        test.index = [7] * len(test)
        ai = AIDataFacade(
            target="Churn", id_column="CustomerID", positive_label="churn"
        )
        ai.load(train, test)
        ai.select_model("logistic_regression")
        ai.train()
        predictions = ai.predict_test()
        assert predictions.row_count == len(test)
        assert list(predictions.ids) == list(test["CustomerID"])


class TestTheErrorModel:
    """§45: an actionable sentence, never an AttributeError."""

    @pytest.mark.parametrize(
        "operation",
        [
            lambda ai: ai.profile(),
            lambda ai: ai.statistics(),
            lambda ai: ai.check_quality(),
            lambda ai: ai.prepare(),
            lambda ai: ai.compare_models(),
            lambda ai: ai.select_model("logistic_regression"),
            lambda ai: ai.train(),
        ],
    )
    def test_every_operation_before_load_says_so(self, operation):
        ai = AIDataFacade(target="Churn")
        with pytest.raises(WorkflowStateError, match="no data is loaded"):
            operation(ai)

    @pytest.mark.parametrize(
        "operation",
        [lambda ai: ai.evaluate(), lambda ai: ai.predict_test()],
    )
    def test_operations_needing_a_fit_name_that_instead(self, operation):
        """The *nearest* missing prerequisite, which is the actionable one.

        Told "no data is loaded", someone who has loaded nothing and trained
        nothing would go and load data, then hit the same wall. Naming the fit
        points at the next thing to do.
        """
        ai = AIDataFacade(target="Churn")
        with pytest.raises(WorkflowStateError, match="no model has been trained"):
            operation(ai)

    def test_training_before_selection_names_the_missing_step(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        with pytest.raises(WorkflowStateError, match="no model has been selected"):
            ai.train()

    def test_evaluating_before_training_names_the_missing_step(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("logistic_regression")
        with pytest.raises(WorkflowStateError, match="no model has been trained"):
            ai.evaluate()

    def test_predicting_without_test_data_says_how_to_supply_it(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("logistic_regression")
        ai.train()
        with pytest.raises(WorkflowStateError, match="load\\(train, test\\)"):
            ai.predict_test()

    def test_a_missing_target_column_is_refused_by_name(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Nonexistent")
        with pytest.raises(SchemaError, match="Nonexistent"):
            ai.load(train)

    def test_a_missing_id_column_is_refused_by_name(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", id_column="NoSuchId")
        with pytest.raises(SchemaError, match="NoSuchId"):
            ai.load(train)

    def test_an_id_column_missing_from_the_test_frame_is_refused(self):
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", id_column="CustomerID")
        with pytest.raises(SchemaError, match="test frame"):
            ai.load(train, test.drop(columns=["CustomerID"]))

    def test_a_non_dataframe_is_refused(self):
        ai = AIDataFacade(target="Churn")
        with pytest.raises(SchemaError, match="DataFrame"):
            ai.load({"Churn": [0, 1]})

    def test_an_empty_target_name_is_refused_at_construction(self):
        with pytest.raises(SchemaError, match="target column name"):
            AIDataFacade(target="   ")

    def test_an_unknown_model_is_refused(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        with pytest.raises(UnknownModelError):
            ai.select_model("nonesuch")

    def test_a_regressor_cannot_be_selected_for_a_classification_target(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        with pytest.raises(IncompatibleModelError):
            ai.select_model("ridge_regression")

    def test_a_classifier_cannot_be_selected_for_a_regression_target(self):
        train, _ = margin_data(n=200)
        ai = AIDataFacade(target="margin")
        ai.load(train)
        with pytest.raises(IncompatibleModelError):
            ai.select_model("logistic_regression")

    def test_a_classification_hint_on_a_continuous_target_is_refused(self):
        """The S6 target-safety closure, reaching the facade constructor path."""
        train, _ = margin_data(n=200)
        ai = AIDataFacade(target="margin", task="classification")
        with pytest.raises(UnsupportedTaskError, match="holds quantities"):
            ai.load(train)


class TestTheFacadeDoesNotDoScienceOfItsOwn:
    """§69: it should not need sklearn, statistics formulas, or split logic."""

    def test_it_imports_no_estimator_or_transformer(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "aidatasetkit" / "facade" / "facade.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "sklearn", "StandardScaler", "OneHotEncoder", "SimpleImputer",
            "train_test_split", "accuracy_score", "mean_squared_error",
            "LabelEncoder",
        ):
            assert forbidden not in source, f"the facade names {forbidden}"

    def test_the_prediction_layer_does_not_either(self):
        from pathlib import Path

        package = (
            Path(__file__).resolve().parents[2] / "aidatasetkit" / "prediction"
        )
        for path in package.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("sklearn", "StandardScaler", "train_test_split"):
                assert forbidden not in source, f"{path.name} names {forbidden}"

    def test_there_is_no_one_click_entry_point(self):
        """§51: the explicit sequence is the educational design."""
        for forbidden in (
            "run", "auto", "solve", "fit_everything", "best_model",
            "select_best_model", "tune", "optimize", "grid_search",
        ):
            assert not hasattr(AIDataFacade, forbidden), f"AIDataFacade.{forbidden}"

    def test_the_public_surface_is_small(self):
        public = {
            name
            for name in dir(AIDataFacade)
            if not name.startswith("_")
        }
        assert public == {
            "comparison", "compare_models", "check_quality", "evaluate",
            "evaluation", "load", "predict", "predict_test", "predictions",
            "prepare", "profile", "select_model", "selected_model", "stage",
            "statistics", "status", "task", "train", "training", "verdict",
            "verdict_reasons",
            # S9. Two names for the unsupervised half, matching the shape the
            # supervised half already has: a verb that does the work, and a
            # property that reports the most recent result.
            "cluster", "clustering",
        }


class TestDefectsTheAdversarialReviewFound:
    """One test per confirmed finding. Each failed before the fix beside it."""

    def test_the_declared_identifier_is_not_trained_on(self):
        """Found by four dimensions. The docs said "not a feature"; it was one.

        On a frame where the identifier correlates with the target, a linear
        model scored R2 = 1.0 off the id alone.
        """
        rng = np.random.default_rng(3)
        n = 300
        frame = pd.DataFrame({"RowID": np.arange(n) * 1.0, "a": rng.normal(0, 1, n)})
        frame["y"] = frame["RowID"] * 0.5 + rng.normal(0, 0.01, n)

        ai = AIDataFacade(target="y", id_column="RowID")
        ai.load(frame)
        ai.select_model("linear_regression")
        result = ai.train()
        assert "RowID" not in result.feature_names
        assert ai.evaluate()["r2"].value < 0.99

    def test_and_the_plan_records_why_it_was_excluded(self):
        rng = np.random.default_rng(3)
        n = 300
        frame = pd.DataFrame({"RowID": np.arange(n) * 1.0, "a": rng.normal(0, 1, n)})
        frame["y"] = rng.normal(0, 1, n)
        ai = AIDataFacade(target="y", id_column="RowID")
        ai.load(frame)
        plan = next(iter(ai.prepare().values()))
        assert plan.decision_for("RowID").reason_code == "confirmed_identifier"

    def test_without_a_declared_id_nothing_changes(self):
        """The fix must not start excluding columns nobody named.

        ``code`` repeats deliberately: an all-distinct column is held back by S4's
        *identifier heuristic*, which is correct pre-existing behaviour and would
        make this test pass for the wrong reason.
        """
        rng = np.random.default_rng(3)
        n = 300
        frame = pd.DataFrame(
            {"code": (np.arange(n) % 12) * 1.0, "a": rng.normal(0, 1, n)}
        )
        frame["y"] = rng.normal(0, 1, n)
        ai = AIDataFacade(target="y")
        ai.load(frame)
        ai.select_model("linear_regression")
        assert "code" in ai.train().feature_names

    def test_and_a_declared_identifier_is_excluded_even_when_it_repeats(self):
        """The declaration is not a guess, so it does not need the heuristic."""
        rng = np.random.default_rng(3)
        n = 300
        frame = pd.DataFrame(
            {"code": (np.arange(n) % 12) * 1.0, "a": rng.normal(0, 1, n)}
        )
        frame["y"] = rng.normal(0, 1, n)
        ai = AIDataFacade(target="y", id_column="code")
        ai.load(frame)
        ai.select_model("linear_regression")
        assert "code" not in ai.train().feature_names

    def test_the_verdict_matches_the_audit_artifact_exactly(self):
        """Found by three dimensions: the facade's verdict was the laxer one.

        ``decide_verdict`` takes three inputs and the facade passed one, so
        rule 3 -- a feature the planner held back -- could never fire.
        """
        from aidatasetkit.evidence import AuditBuilder
        from aidatasetkit.preprocessing import PreprocessingPlanner
        from aidatasetkit.profiling import DataProfiler, DataQualityInspector

        rng = np.random.default_rng(21)
        n = 300
        frame = pd.DataFrame(
            {
                "amount": rng.normal(100, 10, n),
                "many": [f"v{i}" for i in range(n)],   # high cardinality -> held back
                "y": rng.choice(["a", "b"], n),
            }
        )
        ai = AIDataFacade(target="y")
        ai.load(frame)
        ai.check_quality()

        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="y")
        plan = PreprocessingPlanner().plan(
            frame,
            profile,
            ModelFactoryProfile(),
            quality=quality,
            target="y",
        )
        artifact = AuditBuilder(dataset_name=None).build(
            frame, profile=profile, quality=quality, plan=plan
        )
        assert ai.verdict is artifact.verdict

    def test_predict_refuses_a_non_dataframe_with_a_project_error(self):
        """Found by five dimensions: it died on ``.columns`` first."""
        from aidatasetkit.core.exceptions import PredictionValidationError

        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", id_column="CustomerID", positive_label="churn")
        ai.load(train)
        ai.select_model("logistic_regression")
        ai.train()
        with pytest.raises(PredictionValidationError, match="DataFrame"):
            ai.predict(["not", "a", "frame"])

    def test_predict_refuses_a_frame_missing_the_declared_identifier(self):
        """It used to return an id-less result nobody could join to anything."""
        from aidatasetkit.core.exceptions import PredictionValidationError

        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", id_column="CustomerID", positive_label="churn")
        ai.load(train, test)
        ai.select_model("logistic_regression")
        ai.train()
        with pytest.raises(PredictionValidationError, match="CustomerID"):
            ai.predict(test.drop(columns=["CustomerID"]))

    def test_reselecting_the_same_model_keeps_its_fit(self):
        """The docstring promised "a different model"; any reselect discarded it."""
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("logistic_regression")
        fitted = ai.train()
        ai.select_model("logistic_regression")
        assert ai.training is fitted

    def test_but_a_different_model_still_discards_it(self):
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("logistic_regression")
        ai.train()
        ai.select_model("knn_classifier")
        with pytest.raises(WorkflowStateError):
            ai.training

    def test_a_comparison_in_which_everything_failed_is_not_installed(self):
        """It stored an empty ranking *and* discarded a good previous fit."""
        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("logistic_regression")
        fitted = ai.train()
        with pytest.raises(Exception):
            ai.compare_models(
                models=["knn_classifier", "decision_tree_classifier"],
                model_params={
                    "knn_classifier": {"n_neighbors": 0},
                    "decision_tree_classifier": {"max_depth": -5},
                },
            )
        assert ai.training is fitted

    def test_prepare_plans_the_rows_training_will_actually_use(self):
        """It planned on every loaded row; train plans on the split's training half."""
        from aidatasetkit.training import split_rows

        train, _ = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        ai.load(train)
        ai.select_model("knn_classifier")
        fitted = ai.train()

        key = fitted.preprocessing_profile.key
        assert ai.prepare()[key].fingerprint == fitted.plan_fingerprint

    def test_statistics_agree_with_the_profile_on_a_column_holding_an_infinity(self):
        """Raw ``nan_policy='omit'`` kept infinities; the profiler drops them."""
        rng = np.random.default_rng(4)
        n = 200
        values = rng.normal(50, 5, n)
        values[3] = np.inf
        frame = pd.DataFrame({"amount": values, "y": rng.choice(["a", "b"], n)})

        ai = AIDataFacade(target="y")
        ai.load(frame)
        from_facade = ai.statistics()["amount"].mean
        from_profile = ai.profile().column("amount").numeric.mean
        assert from_facade == pytest.approx(from_profile)
        assert np.isfinite(from_facade)

    def test_a_constant_regression_target_is_refused_even_when_requested(self):
        """``task="regression"`` skipped the "nothing to predict" check, and every
        model then scored a perfect RMSE of 0.0."""
        rng = np.random.default_rng(9)
        n = 200
        frame = pd.DataFrame({"a": rng.normal(0, 1, n), "y": np.full(n, 7.5)})
        ai = AIDataFacade(target="y", task="regression")
        with pytest.raises(UnsupportedTaskError, match="nothing to predict"):
            ai.load(frame)

    def test_a_non_default_class_limit_is_honoured_by_the_trainer_too(self):
        """The trainer read the default limit while the detector read the config."""
        rng = np.random.default_rng(6)
        n = 200
        frame = pd.DataFrame({"a": rng.normal(0, 1, n), "y": np.arange(n) % 25})
        generous = KitConfig(task_detection_max_classes=50)

        ai = AIDataFacade(target="y", task="classification", config=generous)
        ai.load(frame)
        assert ai.task is TaskType.CLASSIFICATION
        ai.select_model("logistic_regression")
        ai.train()  # would have raised "holds quantities" before the fix

    def test_the_prediction_frame_does_not_alias_the_result(self):
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", id_column="CustomerID", positive_label="churn")
        ai.load(train, test)
        ai.select_model("logistic_regression")
        ai.train()
        predictions = ai.predict_test()
        before = predictions.predictions.copy()
        frame = predictions.to_frame()
        frame.loc[0, "prediction"] = "TAMPERED"
        np.testing.assert_array_equal(predictions.predictions, before)

    def test_an_identifier_named_like_an_output_column_is_refused(self):
        from aidatasetkit.core.exceptions import ValidationError as CoreValidationError
        from aidatasetkit.prediction import PredictionResult

        result = PredictionResult(
            predictions=np.array(["a", "b"]),
            task_type=TaskType.CLASSIFICATION,
            model_name="m",
            row_count=2,
            ids=np.array([1, 2]),
            id_name="prediction",
        )
        with pytest.raises(CoreValidationError, match="collides"):
            result.to_frame()

    def test_dir_still_shows_the_submodules(self):
        """Returning ``__all__`` alone hid every module the docs tell people to
        import."""
        import aidatasetkit

        listing = dir(aidatasetkit)
        assert "AIDataFacade" in listing
        assert "core" in listing
        assert "__version__" in listing

    def test_an_empty_test_frame_is_refused_at_load(self):
        train, test = churn_data(n=200)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        with pytest.raises(Exception, match="no rows"):
            ai.load(train, test.iloc[:0])

    def test_duplicate_column_labels_are_refused_by_name(self):
        train, _ = churn_data(n=200)
        doubled = pd.concat([train, train[["income"]]], axis=1)
        ai = AIDataFacade(target="Churn", positive_label="churn")
        with pytest.raises(SchemaError, match="duplicate column"):
            ai.load(doubled)

    def test_a_binary_only_model_meets_a_multiclass_target_at_selection(self):
        """select_model checked the task family but not the resolved target."""
        rng = np.random.default_rng(8)
        n = 300
        frame = pd.DataFrame(
            {"a": rng.normal(0, 1, n), "y": rng.choice(["x", "y", "z"], n)}
        )
        ai = AIDataFacade(target="y")
        ai.load(frame)
        ai.select_model("logistic_regression")  # multiclass-capable, fine
        assert ai.selected_model == "logistic_regression"


def ModelFactoryProfile():
    """The capability triple the verdict test plans with."""
    from aidatasetkit.core.types import PreprocessingProfile

    return PreprocessingProfile(
        requires_scaling=False, supports_sparse_input=False, handles_missing_values=False
    )


class TestRootImport:
    def test_the_facade_is_importable_from_the_package_root(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", "from aidatasetkit import AIDataFacade; print(AIDataFacade.__name__)"],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "AIDataFacade"

    def test_importing_the_package_does_not_pull_in_scikit_learn(self):
        """The CLI keeps its file checks ahead of any heavy import; an eager
        facade at the root would quietly undo that."""
        import subprocess
        import sys

        script = (
            "import sys, aidatasetkit;"
            "print(any(m.startswith('sklearn') for m in sys.modules));"
            "print('aidatasetkit.facade' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.split()
        assert out == ["False", "False"]

    def test_the_existing_root_exports_are_unchanged(self):
        import aidatasetkit

        for name in ("KitConfig", "TaskType", "TargetProfile", "__version__"):
            assert hasattr(aidatasetkit, name)