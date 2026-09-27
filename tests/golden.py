"""Which committed semantic fixture the running environment must reproduce.

The semantic artifact records each column's pandas dtype by name, and pandas 3
names a text column ``str`` where pandas 2 names it ``object``. That is a true
difference in what the library observed, not noise, so it is not normalised away;
the artifact identity is therefore tied to the pandas major version, as
``docs/limitations.md`` states.

Both supported majors are held to an exact fixture rather than one of them being
skipped. ``expected_audit_semantic.json`` is the reference fixture (pandas 3, the
reference environment); ``expected_audit_semantic.pandas2.json`` is the same
audit under pandas 2. ``tests/unit/test_golden_fixtures.py`` asserts the two
differ in dtype names and the digests derived from them, and in nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "audit_churn"

REFERENCE_FIXTURE = EXAMPLE / "expected_audit_semantic.json"
PANDAS2_FIXTURE = EXAMPLE / "expected_audit_semantic.pandas2.json"

#: The pandas major the reference fixture was generated under.
REFERENCE_PANDAS_MAJOR = 3


def pandas_major() -> int:
    return int(pd.__version__.split(".")[0])


def semantic_fixture_path() -> Path:
    """The fixture this environment's pandas must reproduce byte for byte."""
    major = pandas_major()
    if major == REFERENCE_PANDAS_MAJOR:
        return REFERENCE_FIXTURE
    if major == 2:
        return PANDAS2_FIXTURE
    raise RuntimeError(
        f"pandas {pd.__version__} is outside the supported majors (2 and 3); "
        "there is no fixture to compare against."
    )
