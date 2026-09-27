"""Structural description of a dataset.

:class:`DataProfiler` answers "what am I holding?". It reads the frame and
returns measurements. It never fills, drops, casts, sorts, or reorders anything,
and the caller's frame is byte-for-byte unchanged afterwards.

Column kinds come from :func:`~aidatasetkit.core.schema.detect_column_kinds`,
which is the only dtype classifier in the library. Numeric statistics come from
:class:`~aidatasetkit.statistics.engine.StatisticsEngine`, so no formula is
written twice.

Complexity is linear in cells: each column is scanned a bounded number of times
and the duplicate-row scan is a single hash pass. Nothing here compares rows
pairwise.
"""

from __future__ import annotations

import re
from collections.abc import Hashable

import numpy as np
import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.counting import value_counts
from aidatasetkit.core.exceptions import DomainError, SchemaError, ValidationError
from aidatasetkit.core.schema import detect_column_kinds
from aidatasetkit.core.types import (
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    NumericSummary,
)
from aidatasetkit.statistics.engine import StatisticsEngine

__all__ = ["DataProfiler", "ID_NAME_PATTERN"]

#: Column names that suggest an identifier. Word-boundary anchored so that
#: ``video`` does not match ``id`` and ``code_quality`` does not match ``code``.
ID_NAME_PATTERN = re.compile(
    r"(^|[^a-z])(id|ids|uuid|guid|key|code|no|num|number|identifier|index|ref|"
    r"reference|serial|sku|isbn)([^a-z]|$)",
    re.IGNORECASE,
)

#: A value shaped like a machine-generated identifier rather than a label.
_ID_VALUE_PATTERN = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|[0-9a-f]{16,}"
    r"|[A-Za-z]{1,5}[-_]?\d{4,})$",
    re.IGNORECASE,
)

#: How many values are inspected when testing value structure.
_STRUCTURE_SAMPLE_SIZE = 200


class DataProfiler:
    """Measures the structure and contents of a dataframe.

    Args:
        config: Thresholds used for the ``is_near_constant``,
            ``is_high_cardinality``, and ``is_id_like`` flags. Defaults to
            :class:`~aidatasetkit.core.config.KitConfig`.

    Example:
        >>> import pandas as pd
        >>> profile = DataProfiler().profile(pd.DataFrame({"a": [1, 2, 2]}))
        >>> profile.row_count
        3
    """

    def __init__(self, config: KitConfig | None = None) -> None:
        self._config = config if config is not None else KitConfig()

    @property
    def config(self) -> KitConfig:
        """The configuration these measurements are taken against."""
        return self._config

    def profile(self, frame: pd.DataFrame) -> DatasetProfile:
        """Describe ``frame`` without modifying it.

        Args:
            frame: The dataframe to inspect.

        Returns:
            A :class:`~aidatasetkit.core.types.DatasetProfile`.

        Raises:
            ValidationError: If ``frame`` is not a :class:`pandas.DataFrame`.
            SchemaError: If column labels are duplicated.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError(
                f"A pandas DataFrame is required, got {type(frame).__name__}."
            )

        kinds = detect_column_kinds(frame)
        row_count = int(len(frame))
        memory = frame.memory_usage(deep=True)

        column_profiles = tuple(
            self._profile_column(
                frame[name], name, kinds.of(name), row_count, int(memory.get(name, 0))
            )
            for name in frame.columns
        )

        duplicates = self._count_duplicate_rows(frame)
        total_missing = sum(profile.missing_count for profile in column_profiles)
        cells = row_count * len(frame.columns)

        return DatasetProfile(
            row_count=row_count,
            column_count=int(len(frame.columns)),
            duplicate_row_count=duplicates,
            duplicate_row_ratio=_ratio(duplicates, row_count),
            total_missing_count=total_missing,
            total_missing_ratio=_ratio(total_missing, cells),
            memory_usage_bytes=int(memory.sum()),
            column_profiles=column_profiles,
        )

    # ------------------------------------------------------------------ #
    # Column measurement
    # ------------------------------------------------------------------ #

    def _profile_column(
        self,
        series: pd.Series,
        name: Hashable,
        kind: ColumnKind,
        row_count: int,
        memory_bytes: int,
    ) -> ColumnProfile:
        """Measure one column."""
        missing_count = int(series.isna().sum())
        present = series.dropna()
        count = int(len(present))
        unique_count = _count_distinct(present, name) if count else 0

        dominant_value, dominant_ratio = self._dominant(present, count, name)
        is_constant = count > 0 and unique_count == 1
        is_near_constant = (
            not is_constant
            and dominant_ratio is not None
            and dominant_ratio >= self._config.near_constant_threshold
        )
        unique_ratio = _ratio(unique_count, count)

        return ColumnProfile(
            name=name,
            detected_kind=kind,
            pandas_dtype=str(series.dtype),
            count=count,
            missing_count=missing_count,
            missing_ratio=_ratio(missing_count, row_count),
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            is_constant=is_constant,
            is_near_constant=is_near_constant,
            is_high_cardinality=self._is_high_cardinality(kind, unique_count),
            is_id_like=self._is_id_like(present, name, kind, unique_ratio, missing_count),
            dominant_value=dominant_value,
            dominant_ratio=dominant_ratio,
            infinite_count=self._count_infinite(present, kind),
            memory_usage_bytes=memory_bytes,
            numeric=self._numeric_summary(present, kind),
        )

    @staticmethod
    def _dominant(
        present: pd.Series, count: int, name: Hashable
    ) -> tuple[object, float | None]:
        """Return the most frequent value and the share of rows it occupies."""
        if count == 0:
            return None, None
        counted = _value_counts(present, name)
        if counted.empty:
            return None, None
        label = counted.index[0]
        if isinstance(label, np.generic):
            label = label.item()
        return label, float(counted.iloc[0]) / count

    def _is_high_cardinality(self, kind: ColumnKind, unique_count: int) -> bool:
        """Whether a label-like column has more distinct values than configured."""
        if kind is not ColumnKind.CATEGORICAL:
            return False
        return unique_count > self._config.high_cardinality_threshold

    @staticmethod
    def _count_infinite(present: pd.Series, kind: ColumnKind) -> int:
        """Count non-finite numeric values, which pandas does not treat as missing."""
        if kind is not ColumnKind.NUMERIC or present.empty:
            return 0
        try:
            values = present.to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError):
            return 0
        return int(np.isinf(values).sum())

    @staticmethod
    def _numeric_summary(present: pd.Series, kind: ColumnKind) -> NumericSummary | None:
        """Summarise a numeric column, leaving undefined statistics as ``None``.

        Delegates every formula to :class:`StatisticsEngine`; nothing is
        recomputed here.
        """
        if kind is not ColumnKind.NUMERIC or present.empty:
            return None

        finite = present[np.isfinite(present.to_numpy(dtype="float64", na_value=np.nan))]
        if finite.empty:
            return NumericSummary()

        engine = StatisticsEngine(finite)
        quartiles = engine.quartiles()
        return NumericSummary(
            minimum=engine.min(),
            maximum=engine.max(),
            mean=engine.mean(),
            median=quartiles.q2,
            std=_optional(engine.std),
            q25=quartiles.q1,
            q75=quartiles.q3,
            iqr=quartiles.q3 - quartiles.q1,
        )

    # ------------------------------------------------------------------ #
    # Identifier heuristic
    # ------------------------------------------------------------------ #

    def _is_id_like(
        self,
        present: pd.Series,
        name: Hashable,
        kind: ColumnKind,
        unique_ratio: float,
        missing_count: int,
    ) -> bool:
        """Whether a column behaves like a record identifier.

        Near-uniqueness alone is deliberately not enough. A continuous
        measurement is unique in every row too, and calling it an identifier
        would be a false positive an analyst has to argue with. Uniqueness is
        treated as a necessary condition, and at least one corroborating signal
        is then required: an identifier-shaped name, an identifier-shaped value,
        or a sequential integer counter.
        """
        if kind in (ColumnKind.BOOLEAN, ColumnKind.DATETIME, ColumnKind.OTHER):
            return False
        if missing_count or present.empty:
            return False
        if unique_ratio < self._config.id_uniqueness_threshold:
            return False

        if ID_NAME_PATTERN.search(str(name)):
            return True
        if kind is ColumnKind.CATEGORICAL:
            return self._values_look_like_identifiers(present)
        return self._looks_like_a_counter(present)

    @staticmethod
    def _values_look_like_identifiers(present: pd.Series) -> bool:
        """Whether sampled text values are shaped like machine-generated keys."""
        sample = present.head(_STRUCTURE_SAMPLE_SIZE).astype(str)
        matches = sum(1 for value in sample if _ID_VALUE_PATTERN.match(value))
        return matches == len(sample) and len(sample) > 0

    @staticmethod
    def _looks_like_a_counter(present: pd.Series) -> bool:
        """Whether an integral numeric column is a dense ascending run.

        A row counter covers its range almost exactly. A unique measurement such
        as a price or a temperature does not.
        """
        try:
            values = present.to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError):
            return False
        if values.size < 2 or not np.all(np.isfinite(values)):
            return False
        if not np.all(values == np.floor(values)):
            return False
        span = float(values.max() - values.min()) + 1.0
        return span > 0 and values.size / span >= 0.99

    # ------------------------------------------------------------------ #
    # Dataset measurement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_duplicate_rows(frame: pd.DataFrame) -> int:
        """Count rows that repeat an earlier row, in a single hash pass."""
        if frame.empty or not len(frame.columns):
            return 0
        try:
            return int(frame.duplicated().sum())
        except TypeError:
            # Unhashable cell values (lists, dicts) make duplicate detection
            # impossible without a row-by-row comparison, which is quadratic.
            return 0


def _unhashable_column_error(name: Hashable) -> SchemaError:
    """Explain why a column of unhashable values cannot be profiled."""
    return SchemaError(
        f"Column {name!r} holds unhashable values such as lists or dicts. "
        "Counting distinct values and duplicate rows is impossible for such a "
        "column, and no model in this library can consume it. Flatten or encode "
        "it before profiling."
    )


def _count_distinct(present: pd.Series, name: Hashable) -> int:
    """Count distinct values, explaining clearly when the values are unhashable."""
    try:
        return int(present.nunique())
    except TypeError as error:
        raise _unhashable_column_error(name) from error


def _value_counts(present: pd.Series, name: Hashable) -> pd.Series:
    """Tally values, explaining clearly when the values are unhashable."""
    try:
        return value_counts(present)
    except TypeError as error:
        raise _unhashable_column_error(name) from error


def _ratio(part: int, whole: int) -> float:
    """Return ``part / whole``, or ``0.0`` when there is nothing to divide."""
    return float(part) / float(whole) if whole else 0.0


def _optional(measure) -> float | None:
    """Return the measurement, or ``None`` when undefined for this sample."""
    try:
        return measure()
    except DomainError:
        return None
