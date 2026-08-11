"""Build the synthetic churn dataset used by the golden audit test.

Every value is generated from a fixed seed and nothing here resembles a real
person. The frame is deliberately full of the problems an audit should catch, so
that a change in what AIDatasetKit reports shows up as a test failure rather than
as a surprise in somebody's notebook.

============================  ==================================================
Column                        Planted problem
============================  ==================================================
``CustomerID``                identifier-like, unique per row
``Age``                       missing values
``MonthlyCharges``            ordinary numeric, nothing wrong
``City``                      low-cardinality nominal, one level unseen in train
``ContractType``              nominal
``Education``                 ordinal, but only if an order is supplied
``IsActive``                  boolean
``HighCardinalityFeature``    far too many levels to encode
``Churn``                     the binary target
``Churn_Copy``                exactly equal to the target
============================  ==================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Fixed so the file on disk is byte-identical on every machine.
SEED = 20250201
ROWS = 600


def build() -> pd.DataFrame:
    """Return the synthetic frame."""
    rng = np.random.default_rng(SEED)
    index = np.arange(ROWS)
    churn = (index % 5 == 0).astype(int)

    age = (22 + index % 47).astype("float64")
    age[index % 11 == 0] = np.nan

    return pd.DataFrame(
        {
            "CustomerID": [f"CUST-{value:06d}" for value in index],
            "Age": age,
            "MonthlyCharges": (30.0 + (index % 83) * 1.37).round(2),
            "City": [["Riyadh", "Jeddah", "Dammam"][value % 3] for value in index],
            "ContractType": [
                ["monthly", "annual", "two_year"][value % 3] for value in index
            ],
            "Education": [["primary", "secondary", "tertiary"][value % 3] for value in index],
            "IsActive": (index % 2 == 0),
            "HighCardinalityFeature": [f"sku_{value % 220}" for value in index],
            "Churn": churn,
            "Churn_Copy": churn,
        }
    )


def main() -> None:
    frame = build()
    path = Path(__file__).with_name("train.csv")
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame)} rows x {frame.shape[1]} columns)")


if __name__ == "__main__":
    main()
