"""Build the public-alpha demo dataset.

Synthetic, seeded, and deliberately imperfect: a customer table with the four
problems an audit should catch before anybody trains on it. No real person's
data appears here, and none should ever be committed to this repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20250301
ROWS = 500


def build() -> pd.DataFrame:
    """Return the demo frame."""
    rng = np.random.default_rng(SEED)
    index = np.arange(ROWS)
    churn = (index % 6 == 0).astype(int)

    tenure = (1 + index % 71).astype("float64")
    tenure[index % 13 == 0] = np.nan  # a real gap, in a real feature

    return pd.DataFrame(
        {
            # unique per row: looks like a key, must not become a feature
            "CustomerID": [f"CUST-{value:06d}" for value in index],
            "TenureMonths": tenure,
            "MonthlyCharges": (25.0 + (index % 79) * 1.15).round(2),
            "Plan": [["basic", "standard", "premium"][value % 3] for value in index],
            "Region": [["north", "south", "east", "west"][value % 4] for value in index],
            "AutoRenew": (index % 2 == 0),
            # far too many levels to one-hot encode safely
            "SupportTicketID": [f"TKT-{value % 380}" for value in index],
            "Churn": churn,
            # exactly the target, under another name
            "ChurnedFlag": churn,
        }
    )


def main() -> None:
    path = Path(__file__).with_name("sample.csv")
    build().to_csv(path, index=False)
    print(f"wrote {path} ({ROWS} rows x {build().shape[1]} columns)")


if __name__ == "__main__":
    main()
