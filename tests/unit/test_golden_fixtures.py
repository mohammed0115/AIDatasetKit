"""The two semantic fixtures differ where pandas differs, and nowhere else.

``expected_audit_semantic.json`` (pandas 3) and its pandas-2 sibling describe the
same audit of the same file. Keeping two fixtures is only honest if the gap
between them is exactly what pandas changed: the dtype name it reports for a text
column, and the dataset and schema digests computed from pandas' own row hasher
and dtype names. Anything else differing would be an environment dependence in
the library, hidden behind a second fixture.
"""

from __future__ import annotations

import json
import re

from tests.golden import PANDAS2_FIXTURE, REFERENCE_FIXTURE

#: Every path allowed to differ, as a regular expression over the walk below.
ALLOWED = (
    re.compile(r"^\.columns\[\d+\]\.pandas_dtype$"),
    re.compile(r"^\.dataset\.fingerprint$"),
    re.compile(r"^\.dataset\.schema_fingerprint$"),
)


def _differences(left, right, path=""):
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        found = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                found.append((f"{path}.{key}", left.get(key), right.get(key)))
            else:
                found += _differences(left[key], right[key], f"{path}.{key}")
        return found
    if isinstance(left, list):
        if len(left) != len(right):
            return [(path, len(left), len(right))]
        found = []
        for index, (a, b) in enumerate(zip(left, right)):
            found += _differences(a, b, f"{path}[{index}]")
        return found
    return [] if left == right else [(path, left, right)]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheTwoFixtures:
    def test_both_are_committed(self):
        assert REFERENCE_FIXTURE.exists()
        assert PANDAS2_FIXTURE.exists()

    def test_they_differ_only_where_pandas_differs(self):
        differences = _differences(_load(REFERENCE_FIXTURE), _load(PANDAS2_FIXTURE))
        unexpected = [d for d in differences if not any(p.match(d[0]) for p in ALLOWED)]
        assert unexpected == []

    def test_the_dtype_difference_is_the_text_dtype_rename(self):
        for path, reference, pandas2 in _differences(
            _load(REFERENCE_FIXTURE), _load(PANDAS2_FIXTURE)
        ):
            if path.endswith("pandas_dtype"):
                assert (reference, pandas2) == ("str", "object"), path

    def test_they_do_differ(self):
        """Otherwise one fixture would be enough and this file would be ceremony."""
        assert _differences(_load(REFERENCE_FIXTURE), _load(PANDAS2_FIXTURE))

    def test_they_record_the_same_configuration(self):
        assert _load(REFERENCE_FIXTURE)["config"] == _load(PANDAS2_FIXTURE)["config"]
