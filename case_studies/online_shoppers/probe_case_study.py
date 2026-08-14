"""Inspect the UCI Online Shoppers data and the planned AIDatasetKit workflow."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aidatasetkit import AIDataFacade
from aidatasetkit.core import KitConfig
from aidatasetkit.preprocessing import PreprocessingConfig

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

frame = pd.read_csv(DATA)
config = KitConfig(random_state=42, validation_size=0.2)
preprocessing = PreprocessingConfig(nominal_features=NOMINAL_FEATURES)
ai = AIDataFacade(
    target=TARGET,
    task="classification",
    positive_label=True,
    config=config,
    preprocessing_config=preprocessing,
)
ai.load(frame)
profile = ai.profile()
quality = ai.check_quality()
plans = ai.prepare()

summary = {
    "shape": list(frame.shape),
    "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    "target_counts": {str(key): int(value) for key, value in frame[TARGET].value_counts().items()},
    "missing_cells": int(frame.isna().sum().sum()),
    "duplicate_rows": int(frame.duplicated().sum()),
    "task": ai.task.value,
    "profile_rows": profile.row_count,
    "quality_verdict": ai.verdict.value,
    "quality_issue_count": quality.issue_count,
    "preprocessing_plan_count": len(plans),
    "plans": {key: plan.to_dict() for key, plan in plans.items()},
}
print(json.dumps(summary, indent=2, default=str, sort_keys=True))
