"""Compare a transparent manual sklearn workflow with AIDatasetKit on UCI 468."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aidatasetkit import AIDataFacade
from aidatasetkit.core import KitConfig, TaskType
from aidatasetkit.preprocessing import PreprocessingConfig
from aidatasetkit.training import split_rows

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "online_shoppers_intention.csv"
TARGET = "Revenue"
NOMINAL_FEATURES = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
)
MODEL_NAMES = ("logistic_regression", "decision_tree_classifier", "random_forest_classifier")

frame = pd.read_csv(DATA)
X = frame.drop(columns=TARGET)
y = frame[TARGET]
config = KitConfig(random_state=42, validation_size=0.2)
preprocessing_config = PreprocessingConfig(nominal_features=NOMINAL_FEATURES)
split = split_rows(y, TaskType.CLASSIFICATION, config)
X_train, X_eval = split.take(X), split.take(X, evaluation=True)
y_train, y_eval = split.take(y), split.take(y, evaluation=True)
numeric_features = [column for column in X.columns if column not in NOMINAL_FEATURES]

manual_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            ),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            list(NOMINAL_FEATURES),
        ),
    ],
    remainder="drop",
)
manual_model = Pipeline(
    [("preprocess", manual_preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=42))]
)
manual_model.fit(X_train, y_train)
manual_pred = manual_model.predict(X_eval)
manual_prob = manual_model.predict_proba(X_eval)[:, list(manual_model.classes_).index(True)]
manual_metrics = {
    "accuracy": accuracy_score(y_eval, manual_pred),
    "precision": precision_score(y_eval, manual_pred, pos_label=True, zero_division=0),
    "recall": recall_score(y_eval, manual_pred, pos_label=True, zero_division=0),
    "f1": f1_score(y_eval, manual_pred, pos_label=True, zero_division=0),
    "roc_auc": roc_auc_score(y_eval, manual_prob),
}

ai = AIDataFacade(
    target=TARGET,
    task="classification",
    positive_label=True,
    config=config,
    preprocessing_config=preprocessing_config,
)
ai.load(frame)
quality = ai.check_quality()
plans = ai.prepare()
comparison = ai.compare_models(
    models=MODEL_NAMES,
    model_params={"random_forest_classifier": {"n_estimators": 60}},
)
ai.select_model("logistic_regression")
training = ai.train()
evaluation = ai.evaluate()
kit_metrics = {name: evaluation[name].value for name in ("accuracy", "precision", "recall", "f1", "roc_auc")}

result = {
    "data": {"rows": len(frame), "features": X.shape[1], "positive_rate": float(y.mean())},
    "shared_split": {
        "fingerprint": split.fingerprint,
        "train_rows": split.train_count,
        "evaluation_rows": split.evaluation_count,
        "stratified": split.stratified,
        "library_fingerprint": comparison.split.fingerprint,
        "same_positions": split.fingerprint == comparison.split.fingerprint,
    },
    "manual": {
        "model": "sklearn LogisticRegression",
        "metrics": manual_metrics,
        "feature_count_after_manual_preprocessing": int(manual_model.named_steps["preprocess"].fit_transform(X_train).shape[1]),
    },
    "aidatasetkit": {
        "selected_model": training.model_name,
        "metrics": kit_metrics,
        "quality_verdict": ai.verdict.value,
        "quality_issue_count": quality.issue_count,
        "preprocessing_plan_count": len(plans),
        "feature_count_after_preprocessing": int(len(training.preprocessor.get_feature_names_out())),
        "comparison_models": [outcome.model_name for outcome in comparison.outcomes],
        "comparison_ranking_metric": comparison.ranking_metric,
        "comparison_order": [outcome.model_name for outcome in comparison.ranked],
    },
}
print(json.dumps(result, indent=2, sort_keys=True, default=float))
