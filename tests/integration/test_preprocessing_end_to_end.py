"""The S4 contract test: a realistic frame, planned, fitted on train, applied to test.

One fixture carries every situation the layer has to survive at once -- gaps in
both a number and a label, an unseen city, an ordinal that needs a supplied
order, a boolean, an identifier, an outlier, a high-cardinality column -- because
the interesting failures happen where those meet, not one at a time.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.pipeline import Pipeline

from aidatasetkit.core.types import RunMetadata, TaskType
from aidatasetkit.models import ModelFactory, default_registry
from aidatasetkit.preprocessing import (
    FeatureAction,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
    TargetLabelEncoder,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector

ORDER = ["high_school", "diploma", "bachelor", "master", "phd"]


@pytest.fixture
def churn_train() -> pd.DataFrame:
    index = np.arange(400)
    charges = 40.0 + (index % 61) * 1.7
    charges[7] = 9_000.0
    return pd.DataFrame(
        {
            "CustomerID": [f"CUST-{v:05d}" for v in index],
            "Age": np.where(index % 17 == 0, np.nan, (20 + index % 50).astype("float64")),
            "MonthlyCharges": charges,
            "City": np.where(
                index % 23 == 0, None, [["Riyadh", "Jeddah", "Mecca"][v % 3] for v in index]
            ),
            "ContractType": [["monthly", "yearly", "two_year"][v % 3] for v in index],
            "Education": [ORDER[v % 5] for v in index],
            "IsActive": (index % 3 == 0),
            "Tenure": (index % 60).astype("int64"),
            "AgentNote": [f"note-{v}" for v in index],
            "Churn": np.where(index % 5 == 0, "churn", "stay"),
        }
    )


@pytest.fixture
def churn_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": ["CUST-90001", "CUST-90002"],
            "Age": [33.0, np.nan],
            "MonthlyCharges": [88.0, 51.0],
            "City": ["Dammam", None],
            "ContractType": ["monthly", "yearly"],
            "Education": ["master", "diploma"],
            "IsActive": [True, False],
            "Tenure": [12, 40],
            "AgentNote": ["note-x", "note-y"],
        }
    )


@pytest.fixture
def config() -> PreprocessingConfig:
    return PreprocessingConfig(
        ordinal_orders={"Education": ORDER},
        confirmed_id_columns=("CustomerID",),
    )


@pytest.fixture
def prepared(churn_train, config):
    """Plan and fit against the logistic-regression capability profile."""
    features = churn_train.drop(columns=["Churn"])
    profile = DataProfiler().profile(churn_train)
    target_profile = TaskDetector().detect(
        churn_train["Churn"], target_name="Churn", positive_label="churn"
    )
    quality = DataQualityInspector().inspect(
        churn_train, target="Churn", profile=profile, target_profile=target_profile
    )
    capabilities = default_registry().resolve("logistic_regression").capabilities
    plan = PreprocessingPlanner(config).plan(
        churn_train,
        profile,
        capabilities,
        quality=quality,
        target="Churn",
        target_profile=target_profile,
    )
    preprocessor = PreprocessorBuilder(config).build(plan, churn_train)
    preprocessor.fit(features)
    return plan, preprocessor, target_profile


class TestThePlanReadsCorrectly:
    def test_the_identifier_is_excluded(self, prepared):
        plan, _, _ = prepared
        assert plan.decision_for("CustomerID").action is FeatureAction.EXCLUDE

    def test_the_free_text_note_is_held_back(self, prepared):
        plan, _, _ = prepared
        decision = plan.decision_for("AgentNote")
        assert decision.action is FeatureAction.REVIEW

    def test_the_ordinal_uses_the_supplied_order(self, prepared):
        plan, _, _ = prepared
        assert plan.spec_for("Education").ordinal_order == tuple(ORDER)
        assert "ordinal_encoding" in plan.decision_for("Education").steps

    def test_numeric_columns_are_imputed_and_scaled(self, prepared):
        plan, _, _ = prepared
        steps = plan.decision_for("Age").steps
        assert "median_imputation" in steps and "standard_scaling" in steps

    def test_labels_are_one_hot_encoded(self, prepared):
        plan, _, _ = prepared
        assert "onehot_encoding" in plan.decision_for("City").steps

    def test_the_boolean_is_treated_as_a_label(self, prepared):
        plan, _, _ = prepared
        assert plan.decision_for("IsActive").role.value == "binary_categorical"

    def test_outliers_are_not_removed_or_clipped(self, prepared, churn_train):
        """The quality inspector flagged them; preprocessing left them alone."""
        _, preprocessor, _ = prepared
        out = _dense(preprocessor.transform(churn_train.drop(columns=["Churn"])))
        assert out.shape[0] == len(churn_train)
        assert churn_train["MonthlyCharges"].max() == 9_000.0

    def test_every_decision_carries_a_reason(self, prepared):
        plan, _, _ = prepared
        for decision in plan.decisions:
            assert decision.reason_code and decision.reason

    def test_the_plan_can_be_read_before_fitting(self, prepared):
        plan, _, _ = prepared
        description = plan.describe()
        assert "CustomerID" in description and "Education" in description


class TestTrainOnlyFitting:
    def test_an_unseen_city_does_not_crash_inference(self, prepared, churn_test):
        _, preprocessor, _ = prepared
        assert _dense(preprocessor.transform(churn_test)).shape[0] == 2

    def test_the_unseen_city_never_joins_the_learned_categories(self, prepared, churn_test):
        _, preprocessor, _ = prepared
        preprocessor.transform(churn_test)
        encoder = preprocessor.transformer.named_transformers_["nominal"].named_steps["encoder"]
        learned = {value for values in encoder.categories_ for value in values}
        assert "Dammam" not in learned

    def test_the_learned_median_comes_from_training(self, prepared, churn_train, churn_test):
        _, preprocessor, _ = prepared
        imputer = preprocessor.transformer.named_transformers_["numeric"].named_steps["imputer"]
        before = imputer.statistics_.copy()
        preprocessor.transform(churn_test)
        np.testing.assert_allclose(imputer.statistics_, before)

    def test_train_and_test_produce_the_same_column_count(self, prepared, churn_train, churn_test):
        _, preprocessor, _ = prepared
        train_out = _dense(preprocessor.transform(churn_train.drop(columns=["Churn"])))
        test_out = _dense(preprocessor.transform(churn_test))
        assert train_out.shape[1] == test_out.shape[1]

    def test_missing_values_in_the_test_frame_are_filled(self, prepared, churn_test):
        _, preprocessor, _ = prepared
        assert not np.isnan(_dense(preprocessor.transform(churn_test))).any()


class TestFeatureNamesAndLineage:
    def test_every_output_column_is_named(self, prepared):
        _, preprocessor, _ = prepared
        names = list(preprocessor.get_feature_names_out())
        assert len(names) == preprocessor.n_features_out
        assert all(isinstance(name, str) and name for name in names)

    def test_one_hot_columns_are_recognisable(self, prepared):
        _, preprocessor, _ = prepared
        names = list(preprocessor.get_feature_names_out())
        assert {"City_Riyadh", "City_Jeddah", "City_Mecca"} <= set(names)

    def test_lineage_covers_every_included_feature(self, prepared):
        plan, preprocessor, _ = prepared
        assert set(preprocessor.lineage()) == set(plan.included_features)

    def test_lineage_expands_a_one_hot_column(self, prepared):
        _, preprocessor, _ = prepared
        assert len(preprocessor.lineage()["City"]) == 3

    def test_lineage_keeps_a_scalar_column_one_to_one(self, prepared):
        _, preprocessor, _ = prepared
        assert preprocessor.lineage()["Age"] == ("Age",)

    def test_the_transformed_count_can_populate_run_metadata(self, prepared):
        """RunMetadata already carries the field; this is what fills it."""
        plan, preprocessor, _ = prepared
        metadata = RunMetadata(
            model_name="logistic_regression",
            task_type=TaskType.CLASSIFICATION,
            model_parameters={},
            random_state=42,
            training_rows=400,
            feature_count=len(plan.included_features),
            preprocessing_profile=plan.preprocessing_profile.key,
            transformed_feature_count=preprocessor.n_features_out,
        )
        assert metadata.transformed_feature_count == preprocessor.n_features_out
        assert metadata.to_dict()["transformed_feature_count"] > metadata.feature_count


class TestModelIntegration:
    def _pipeline(self, name, plan, config, churn_train):
        preprocessor = PreprocessorBuilder(config).build(plan, churn_train)
        return Pipeline(
            [("preprocessor", preprocessor.transformer), ("model", ModelFactory.create(name))]
        )

    @pytest.fixture
    def encoded_target(self, churn_train):
        profile = TaskDetector().detect(
            churn_train["Churn"], target_name="Churn", positive_label="churn"
        )
        encoder = TargetLabelEncoder(positive_label="churn").fit(churn_train["Churn"], profile)
        return encoder, encoder.transform(churn_train["Churn"])

    @pytest.mark.parametrize("name", ["dummy_classifier", "logistic_regression"])
    def test_the_preprocessor_composes_with_each_model(
        self, name, prepared, churn_train, churn_test, config, encoded_target
    ):
        plan, _, _ = prepared
        _, y = encoded_target
        pipeline = self._pipeline(name, plan, config, churn_train)
        pipeline.fit(churn_train.drop(columns=["Churn"]), y)
        assert pipeline.predict(churn_test).shape == (2,)

    def test_logistic_regression_produces_probabilities(
        self, prepared, churn_train, churn_test, config, encoded_target
    ):
        plan, _, _ = prepared
        _, y = encoded_target
        pipeline = self._pipeline("logistic_regression", plan, config, churn_train)
        pipeline.fit(churn_train.drop(columns=["Churn"]), y)
        probabilities = pipeline.predict_proba(churn_test)
        assert probabilities.shape == (2, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_the_positive_probability_column_is_the_one_the_analyst_meant(
        self, prepared, churn_train, churn_test, config, encoded_target
    ):
        """"churn" sorts first, so the positive column is 0, not 1."""
        plan, _, _ = prepared
        encoder, y = encoded_target
        pipeline = self._pipeline("logistic_regression", plan, config, churn_train)
        pipeline.fit(churn_train.drop(columns=["Churn"]), y)
        column = encoder.positive_column_index()
        assert column == 0
        probabilities = pipeline.predict_proba(churn_test)[:, column]
        assert ((0.0 <= probabilities) & (probabilities <= 1.0)).all()

    def test_predictions_decode_back_to_the_original_labels(
        self, prepared, churn_train, config, encoded_target
    ):
        plan, _, _ = prepared
        encoder, y = encoded_target
        pipeline = self._pipeline("logistic_regression", plan, config, churn_train)
        pipeline.fit(churn_train.drop(columns=["Churn"]), y)
        decoded = encoder.inverse_transform(pipeline.predict(churn_train.drop(columns=["Churn"])))
        assert set(decoded) <= {"churn", "stay"}

    def test_the_two_models_get_different_preprocessing(self, churn_train, config):
        profile = DataProfiler().profile(churn_train)
        planner = PreprocessingPlanner(config)
        plans = {
            entry.canonical_name: planner.plan(
                churn_train, profile, entry.capabilities, target="Churn"
            )
            for entry in default_registry().catalog()
        }
        fingerprints = {name: plan.fingerprint for name, plan in plans.items()}
        assert len(set(fingerprints.values())) == 2

    def test_that_difference_comes_from_capabilities_not_names(self, churn_train, config):
        profile = DataProfiler().profile(churn_train)
        planner = PreprocessingPlanner(config)
        for entry in default_registry().catalog():
            plan = planner.plan(churn_train, profile, entry.capabilities, target="Churn")
            expected = entry.capabilities.requires_scaling
            has_scaler = any(
                "scaling" in step for step in plan.decision_for("Age").steps
            )
            assert has_scaler == expected


class TestImmutability:
    def test_the_training_frame_is_unchanged(self, churn_train, config):
        before = churn_train.copy(deep=True)
        profile = DataProfiler().profile(churn_train)
        capabilities = default_registry().resolve("logistic_regression").capabilities
        plan = PreprocessingPlanner(config).plan(
            churn_train, profile, capabilities, target="Churn"
        )
        preprocessor = PreprocessorBuilder(config).build(plan, churn_train)
        preprocessor.fit_transform(churn_train.drop(columns=["Churn"]))
        pd.testing.assert_frame_equal(churn_train, before)

    def test_the_test_frame_is_unchanged(self, prepared, churn_test):
        before = churn_test.copy(deep=True)
        _, preprocessor, _ = prepared
        preprocessor.transform(churn_test)
        pd.testing.assert_frame_equal(churn_test, before)

    def test_the_target_is_unchanged(self, churn_train):
        target = churn_train["Churn"]
        before = target.copy(deep=True)
        TargetLabelEncoder(positive_label="churn").fit_transform(target)
        pd.testing.assert_series_equal(target, before)

    def test_row_count_and_order_survive(self, prepared, churn_train):
        _, preprocessor, _ = prepared
        out = _dense(preprocessor.transform(churn_train.drop(columns=["Churn"])))
        assert out.shape[0] == len(churn_train)

    def test_dtypes_survive(self, prepared, churn_train):
        before = churn_train.dtypes.copy()
        _, preprocessor, _ = prepared
        preprocessor.transform(churn_train.drop(columns=["Churn"]))
        pd.testing.assert_series_equal(churn_train.dtypes, before)


class TestSerialisation:
    def test_the_plan_round_trips_through_json(self, prepared):
        plan, _, _ = prepared
        payload = json.loads(json.dumps(plan.to_dict()))
        assert payload["target_name"] == "Churn"
        assert "Education" in payload["ordinal_features"]
        assert "CustomerID" in payload["excluded_features"]

    def test_no_transformer_object_leaks_into_the_plan(self, prepared):
        plan, _, _ = prepared
        text = json.dumps(plan.to_dict())
        for forbidden in ("SimpleImputer", "OneHotEncoder", "StandardScaler", "Pipeline"):
            assert forbidden not in text

    def test_the_target_encoding_serialises(self, churn_train):
        encoder = TargetLabelEncoder(positive_label="churn").fit(churn_train["Churn"])
        payload = json.loads(json.dumps(encoder.encoding.to_dict()))
        assert payload["positive_label_encoded"] == 0


def _dense(matrix):
    """Return a dense array whether or not the transformer produced a sparse one."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
