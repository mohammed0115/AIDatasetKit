"""G0-01: the final test takes no part in choosing the model it scores.

Three partitions, three jobs. The *training* side of the internal split fits the
preprocessor and every estimator. The *validation* side ranks the comparison and
scores :meth:`AIDataFacade.evaluate` -- so once a comparison has ranked on it, it
has chosen the model, and its score is a development number. The external *test*
frame is scored once, by :meth:`AIDataFacade.evaluate_final`, after which nothing
may choose or refit a model.

Most tests here are negative. A test frame is poisoned -- labels flipped, values
wildly out of range, categories the training rows never had -- and everything
development produces is asserted identical to a session with no test frame at
all. If any path read the test rows, the poison would show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit import AIDataFacade
from aidatasetkit.core.exceptions import TrainingError, WorkflowStateError
from aidatasetkit.facade import Stage
from aidatasetkit.training import ModelTrainer

MODELS = ["logistic_regression", "decision_tree_classifier", "random_forest_classifier"]


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(60_000, 15_000, n).round(2)
    ratio = rng.normal(0.5, 0.05, n).round(4)
    plan = rng.choice(["basic", "plus", "pro"], n)
    signal = (income - 60_000) / 15_000 + (ratio - 0.5) / 0.05 + (plan == "pro") * 0.8
    churn = np.where(signal + rng.normal(0, 0.7, n) > 0, "yes", "no")
    return pd.DataFrame({"income": income, "ratio": ratio, "plan": plan, "churn": churn})


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return _frame(400, seed=1)


@pytest.fixture(scope="module")
def test() -> pd.DataFrame:
    return _frame(150, seed=2)


@pytest.fixture(scope="module")
def poisoned(test) -> pd.DataFrame:
    """Everything that would move a fit, a ranking or a verdict if it were read."""
    bad = test.copy()
    bad["churn"] = np.where(bad["churn"] == "yes", "no", "yes")
    bad["income"] = bad["income"] * 1e6
    bad["plan"] = "only_in_test"
    return bad


def _session(train, test=None) -> AIDataFacade:
    return AIDataFacade(target="churn", positive_label="yes").load(train, test)


def _developed(train, test=None):
    """Run the whole development path and return everything it produced."""
    ai = _session(train, test)
    quality = ai.check_quality().to_dict()
    plans = {key: plan.to_dict() for key, plan in ai.prepare().items()}
    comparison = ai.compare_models(MODELS).to_dict()
    ai.select_model(comparison["ranked_order"][0])
    ai.train()
    validation = ai.evaluate().to_dict()
    predictions = ai.training.predict(train.drop(columns=["churn"]))
    return ai, {
        "quality": quality,
        "plans": plans,
        "comparison": comparison,
        "validation": validation,
        "predictions": predictions.tolist(),
    }


class TestTheTestFrameInfluencesNothing:
    @pytest.fixture(scope="class")
    @staticmethod
    def without(train):
        return _developed(train)[1]

    @pytest.mark.parametrize("key", ["quality", "plans", "comparison", "validation", "predictions"])
    def test_a_real_test_frame_changes_nothing_in_development(self, train, test, without, key):
        assert _developed(train, test)[1][key] == without[key]

    @pytest.mark.parametrize("key", ["quality", "plans", "comparison", "validation", "predictions"])
    def test_a_poisoned_test_frame_changes_nothing_in_development(
        self, train, poisoned, without, key
    ):
        assert _developed(train, poisoned)[1][key] == without[key]

    def test_the_poison_would_have_shown(self, train, test, poisoned):
        """Otherwise the two tests above could pass for a frame nobody could see."""
        clean = _developed(train, test)[0].evaluate_final().report["accuracy"].value
        bad = _developed(train, poisoned)[0].evaluate_final().report["accuracy"].value
        assert clean - bad > 0.3

    def test_a_category_only_the_test_has_never_joins_the_vocabulary(self, train, poisoned):
        ai = _developed(train, poisoned)[0]
        fixed = train.drop(columns=["churn"]).head(20)
        before = ai.training.preprocessor.transform(fixed)
        ai.evaluate_final()
        after = ai.training.preprocessor.transform(fixed)
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
        assert "only_in_test" not in str(ai.training.preprocessor.lineage())


class TestTheFinalEvaluation:
    @pytest.fixture
    @staticmethod
    def trained(train, test):
        return _developed(train, test)[0]

    def test_it_is_the_trainer_s_measurement_on_the_test_rows(self, trained, test):
        final = trained.evaluate_final()
        direct = ModelTrainer().evaluate(trained.training, test, target="churn")
        assert final.report.to_dict() == direct.to_dict()
        assert final.test_rows == len(test)
        assert final.model_name == trained.selected_model

    def test_it_is_measured_once(self, trained):
        assert trained.evaluate_final() is trained.evaluate_final()
        assert trained.final_evaluation is trained.evaluate_final()

    def test_it_modifies_neither_frame(self, train, test):
        train_copy, test_copy = train.copy(), test.copy()
        _developed(train, test)[0].evaluate_final()
        pd.testing.assert_frame_equal(train, train_copy)
        pd.testing.assert_frame_equal(test, test_copy)

    def test_the_stage_reports_it(self, trained):
        trained.evaluate_final()
        assert trained.stage is Stage.FINAL_EVALUATED
        assert trained.status["final_evaluated"] is True

    def test_two_identical_sessions_agree(self, train, test):
        first = _developed(train, test)[0].evaluate_final().to_dict()
        second = _developed(train, test)[0].evaluate_final().to_dict()
        assert first == second

    def test_it_needs_a_trained_model(self, train, test):
        with pytest.raises(WorkflowStateError, match="no model has been trained"):
            _session(train, test).evaluate_final()

    def test_it_needs_a_test_frame(self, train):
        ai = _developed(train)[0]
        with pytest.raises(WorkflowStateError, match="no test frame was loaded"):
            ai.evaluate_final()

    def test_it_needs_the_answer_in_the_test_frame(self, train, test):
        ai = _developed(train, test.drop(columns=["churn"]))[0]
        with pytest.raises(TrainingError, match="carry the answer"):
            ai.evaluate_final()

    def test_a_failed_final_evaluation_does_not_freeze(self, train, test):
        ai = _developed(train, test.drop(columns=["churn"]))[0]
        with pytest.raises(TrainingError):
            ai.evaluate_final()
        ai.train()
        assert ai.status["final_evaluated"] is False


class TestTheExperimentFreezes:
    @pytest.fixture
    @staticmethod
    def finished(train, test):
        ai = _developed(train, test)[0]
        ai.evaluate_final()
        return ai

    @pytest.mark.parametrize(
        "step",
        [
            lambda ai: ai.compare_models(MODELS),
            lambda ai: ai.select_model("decision_tree_classifier"),
            lambda ai: ai.select_model(ai.selected_model),
            lambda ai: ai.train(),
            lambda ai: ai.train(max_iter=5),
        ],
        ids=["compare", "select_other", "select_same", "retrain", "retune"],
    )
    def test_nothing_that_could_choose_or_refit_is_allowed(self, finished, step):
        with pytest.raises(WorkflowStateError, match="final evaluation has been run"):
            step(finished)

    def test_the_frozen_result_is_left_intact_by_a_refusal(self, finished):
        before = finished.final_evaluation.to_dict()
        with pytest.raises(WorkflowStateError):
            finished.train()
        assert finished.final_evaluation.to_dict() == before
        assert finished.stage is Stage.FINAL_EVALUATED

    def test_reading_is_still_allowed(self, finished):
        finished.evaluate()
        finished.predict_test()

    def test_loading_new_data_starts_a_new_experiment(self, finished, train, test):
        finished.load(train, test)
        assert finished.status["final_evaluated"] is False
        finished.compare_models(MODELS)
        finished.select_model("logistic_regression")
        finished.train()


class TestSelectionIsReported:
    def test_a_ranking_on_the_same_rows_is_flagged(self, train, test):
        ai = _developed(train, test)[0]
        assert ai.status["validation_used_for_selection"] is True
        assert ai.evaluate_final().validation_used_for_selection is True

    def test_training_without_a_comparison_is_not(self, train, test):
        ai = _session(train, test)
        ai.select_model("logistic_regression")
        ai.train()
        assert ai.status["validation_used_for_selection"] is False
        assert ai.evaluate_final().validation_used_for_selection is False


class TestCopiesOfTrainingRows:
    def test_disjoint_frames_share_nothing(self, train, test):
        assert _developed(train, test)[0].evaluate_final().rows_also_in_training == 0

    def test_copied_rows_are_counted(self, train, test):
        leaky = pd.concat([test, train.iloc[:7]], ignore_index=True)
        assert _developed(train, leaky)[0].evaluate_final().rows_also_in_training == 7

    def test_a_copy_with_a_different_dtype_is_not_a_copy(self, train, test):
        """Exact means exact: dtype is part of a row's identity here."""
        leaky = pd.concat([test, train.iloc[:3]], ignore_index=True)
        leaky["ratio"] = leaky["ratio"].astype("float32")
        assert _developed(train, leaky)[0].evaluate_final().rows_also_in_training == 0


class TestNoOtherRouteExists:
    @pytest.mark.parametrize(
        "name",
        ["tune", "grid_search", "cross_validate", "select_threshold", "optimize_threshold"],
    )
    def test_the_facade_has_no_search(self, name):
        assert not hasattr(AIDataFacade, name)
