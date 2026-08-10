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


#: Rows in the planted-problem fixture before duplicates are appended.
PROBLEM_ROWS = 200

#: Rows duplicated at the end of the planted-problem fixture.
PROBLEM_DUPLICATES = 4


@pytest.fixture
def problematic_frame() -> pd.DataFrame:
    """A frame with one deliberately planted instance of every quality issue.

    Every value is deterministic. The cycle lengths are chosen so that no issue
    fires by coincidence -- ``city`` cycles with period 61 against a churn pattern
    of period 10, so no city maps to a single class and the deterministic-mapping
    heuristic stays quiet for it.

    ============================  ==============================================
    Column                        Planted issue
    ============================  ==============================================
    ``customer_id``               identifier-like, high cardinality
    ``age``                       outlier candidates
    ``income``                    missing values above the warning threshold
    ``country``                   constant
    ``plan``                      near-constant
    ``city``                      high cardinality, legitimately not an id
    ``income_text``               numeric values stored as text
    ``zip_code``                  text code that must NOT be called numeric
    ``ratio``                     positive and negative infinities
    ``signup_date``               real datetime, must not be called numeric text
    ``temperature``               unique measurement, must NOT be called an id
    ``moderate_signal``           strong association that is NOT leakage
    ``risk_score``                association above the leakage threshold
    ``churn_reason``              post-outcome name, deterministic mapping
    ``churn_flag``                exactly equal to the target
    ``Churn``                     imbalanced binary target
    (whole rows)                  duplicated rows
    ============================  ==============================================
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = PROBLEM_ROWS
    index = np.arange(n)
    churn = (index % 10 == 0).astype(int)

    income = rng.normal(50_000, 12_000, n)
    income[index % 4 == 0] = np.nan

    age = 20 + (index % 41)
    age[5] = 900
    age[105] = 950

    ratio = rng.normal(1.0, 0.2, n)
    ratio[7] = np.inf
    ratio[8] = -np.inf

    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{value:05d}" for value in index],
            "age": age.astype("int64"),
            "income": income,
            "country": ["SA"] * n,
            "plan": ["basic"] * (n - 1) + ["pro"],
            "city": [f"city_{value % 61}" for value in index],
            "income_text": [["10", "20", "30", "40", "unknown"][v % 5] for v in index],
            "zip_code": [["02134", "90210", "01002", "10001"][v % 4] for v in index],
            "ratio": ratio,
            "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(index, unit="D"),
            "temperature": 15.0 + index * 0.137,
            "moderate_signal": churn * 10.0 + rng.normal(0.0, 3.0, n),
            "risk_score": churn * 100.0 + rng.normal(0.0, 0.5, n),
            "churn_reason": np.where(churn == 1, "price", "none"),
            "churn_flag": churn,
            "Churn": churn,
        }
    )

    duplicated = frame.iloc[:PROBLEM_DUPLICATES]
    return pd.concat([frame, duplicated], ignore_index=True)


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    """A frame with nothing wrong with it, used to catch false positives.

    Every cycle length is odd and coprime with the target's period of two, so no
    feature determines the target. An earlier version used ``index % 24`` for
    tenure, which does fix ``index % 2`` -- the leakage check caught it, which is
    the behaviour these fixtures exist to keep honest.
    """
    n = 120
    index = np.arange(n)
    return pd.DataFrame(
        {
            "age": (20 + index % 41).astype("int64"),
            "segment": [["A", "B", "C"][value % 3] for value in index],
            "measurement": (index % 37) * 1.5,
            "tenure_months": (index % 23).astype("int64"),
            "target": (index % 2).astype("int64"),
        }
    )


#: Golden datasets for the visualization layer. Each isolates one situation the
#: recommendation rules have to get right, and every value is deterministic.


@pytest.fixture
def binary_frame() -> pd.DataFrame:
    """Numeric and categorical features against a binary classification target."""
    index = np.arange(300)
    return pd.DataFrame(
        {
            "age": (20 + index % 45).astype("int64"),
            "income": (30_000 + (index % 71) * 900).astype("float64"),
            "segment": [["A", "B", "C"][value % 3] for value in index],
            "churn": (index % 7 == 0).astype("int64"),
        }
    )


@pytest.fixture
def multiclass_frame() -> pd.DataFrame:
    """A three-class target with a numeric and a categorical feature."""
    index = np.arange(300)
    return pd.DataFrame(
        {
            "score": (index % 53).astype("float64"),
            "region": [["north", "south", "east"][value % 3] for value in index],
            "grade": [["low", "medium", "high"][value % 3] for value in index],
        }
    )


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    """A continuous target with two numeric features."""
    index = np.arange(300)
    return pd.DataFrame(
        {
            "rooms": (1 + index % 6).astype("int64"),
            "area": (40.0 + (index % 97) * 2.5),
            "price": (100_000.0 + (index % 89) * 3_137.0),
        }
    )


@pytest.fixture
def high_cardinality_frame() -> pd.DataFrame:
    """A legitimate categorical feature with far more levels than any chart wants."""
    index = np.arange(2_000)
    return pd.DataFrame(
        {
            "city": [f"city_{value % 500}" for value in index],
            "amount": (index % 37).astype("float64"),
        }
    )


@pytest.fixture
def missing_heavy_frame() -> pd.DataFrame:
    """Two columns riddled with gaps, and one without any."""
    index = np.arange(200)
    sparse = np.where(index % 3 == 0, np.nan, index.astype("float64"))
    sparser = np.where(index % 2 == 0, np.nan, index.astype("float64"))
    return pd.DataFrame(
        {"complete": index.astype("float64"), "gappy": sparse, "gappier": sparser}
    )


@pytest.fixture
def outlier_frame() -> pd.DataFrame:
    """A numeric column with values far outside the Tukey fences."""
    index = np.arange(200)
    values = (index % 40).astype("float64")
    values[5] = 5_000.0
    values[150] = -4_000.0
    return pd.DataFrame({"measurement": values, "steady": (index % 11).astype("float64")})


@pytest.fixture
def constant_frame() -> pd.DataFrame:
    """One constant column, one near-constant, one ordinary."""
    index = np.arange(200)
    near = ["basic"] * 199 + ["pro"]
    return pd.DataFrame(
        {
            "country": ["SA"] * 200,
            "plan": near,
            "age": (20 + index % 40).astype("int64"),
        }
    )


@pytest.fixture
def identifier_frame() -> pd.DataFrame:
    """Numeric and textual identifiers beside a genuine measurement."""
    index = np.arange(300)
    return pd.DataFrame(
        {
            "row_id": index.astype("int64"),
            "customer_id": [f"CUST-{value:05d}" for value in index],
            "temperature": 15.0 + index * 0.137,
        }
    )


@pytest.fixture
def wide_frame() -> pd.DataFrame:
    """Two hundred numeric columns: the pairwise-explosion case."""
    index = np.arange(200)
    return pd.DataFrame(
        {
            f"f{position:03d}": ((index * (position + 1)) % (37 + position)).astype(
                "float64"
            )
            for position in range(200)
        }
    )


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
