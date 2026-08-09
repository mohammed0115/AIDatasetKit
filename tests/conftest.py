"""Shared fixtures.

Fixtures here are deterministic. Nothing in the test suite depends on an
unseeded random number generator or on the wall clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import KitConfig

RANDOM_SEED = 20240101


@pytest.fixture
def config() -> KitConfig:
    """The default configuration."""
    return KitConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator."""
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """A frame containing one column of every structural kind."""
    return pd.DataFrame(
        {
            "int_col": pd.Series([1, 2, 3, 4], dtype="int64"),
            "float_col": pd.Series([1.5, 2.5, np.nan, 4.5], dtype="float64"),
            "nullable_int_col": pd.Series([1, 2, None, 4], dtype="Int64"),
            "bool_col": pd.Series([True, False, True, False], dtype="bool"),
            "string_col": pd.Series(["a", "b", "a", "c"], dtype="string"),
            "object_col": pd.Series(["x", "y", "x", None], dtype="object"),
            "category_col": pd.Categorical(["low", "high", "low", "high"]),
            "datetime_col": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "timedelta_col": pd.to_timedelta([1, 2, 3, 4], unit="D"),
        }
    )
