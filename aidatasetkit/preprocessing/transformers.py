"""The two transformers scikit-learn does not provide.

Both exist because the alternative was a pandas operation on the caller's frame,
which would neither survive a train/test split nor leave the caller's data alone.

They follow scikit-learn's conventions -- parameters stored unchanged in
``__init__``, learned state in trailing-underscore attributes -- so they clone,
compose in a pipeline, and report their output names like any other step.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from aidatasetkit.core.exceptions import PreprocessingError

__all__ = [
    "CategoricalCaster",
    "ExplicitMappingEncoder",
    "NumericCaster",
    "NumericTextConverter",
    "OrdinalDomainGuard",
]


def _non_numeric_examples(series: pd.Series, limit: int = 3) -> list[Any]:
    """The first few values in ``series`` that cannot be read as a number."""
    coerced = pd.to_numeric(series, errors="coerce")
    offending = series[coerced.isna() & series.notna()]
    return [value for value in offending.unique()[:limit]]


class NumericCaster(BaseEstimator, TransformerMixin):
    """Puts numeric columns into plain ``float64``.

    pandas nullable dtypes (``Int64``, ``Float64``) carry ``pd.NA``, which
    scikit-learn rejects outright -- and a model that consumes ``NaN`` natively
    has no imputer in front of it to convert them. Casting here means an
    extension dtype behaves like any other number, and missing stays missing.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NumericCaster":
        """Record the incoming column names."""
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return a ``float64`` array, missing values preserved as ``NaN``.

        Raises:
            PreprocessingError: If a column planned as numeric holds a value that
                is not a number. pandas reports that as "could not convert string
                to float", which names neither the column nor the row, and reads
                like a bug in the library rather than in the data.
        """
        frame = _as_frame(X)
        columns = []
        for column in frame.columns:
            try:
                columns.append(frame[column].to_numpy(dtype="float64", na_value=np.nan))
            except (TypeError, ValueError) as error:
                offenders = _non_numeric_examples(frame[column])
                raise PreprocessingError(
                    f"Column {column!r} is planned as numeric but holds value(s) that are "
                    f"not numbers, for example {offenders}. Correct the values, or re-plan "
                    "with the column treated as categorical."
                ) from error
        return np.column_stack(columns) if columns else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return one output name per input column."""
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray([str(f) for f in input_features], dtype=object)
        )
        return np.asarray([str(name) for name in names], dtype=object)


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Puts label columns into the one dtype the imputer accepts.

    ``SimpleImputer`` refuses a plain ``bool`` column outright, and does not
    recognise ``pd.NA`` as missing in a nullable one. Both are ordinary in real
    data and neither is the caller's problem, so every label column is cast to
    object here and its missing markers normalised to ``NaN`` first.

    The cast happens on a copy. The caller's frame keeps its dtypes.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoricalCaster:
        """Record the incoming column names."""
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a text-valued copy with a single missing marker.

        Values become strings so that a column holding several Python types --
        ordinary after a CSV read -- can still be sorted into categories. An
        encoder asked to order ``1`` against ``"a"`` raises instead.
        """
        frame = _as_frame(X)
        for column in frame.columns:
            _require_no_text_collision(frame[column], column)
        cast = frame.astype(object).map(lambda v: v if pd.isna(v) else str(v))
        return cast.where(frame.notna(), np.nan)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return one output name per input column."""
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray([str(f) for f in input_features], dtype=object)
        )
        return np.asarray([str(name) for name in names], dtype=object)


class ExplicitMappingEncoder(BaseEstimator, TransformerMixin):
    """Applies an analyst-supplied value-to-number mapping.

    The educational form of this is ``df.replace(...)`` on the frame itself. That
    mutates the caller's data, and applied separately to train and test it can
    produce two different encodings of the same column. Here the mapping is a
    fitted transformer: declared once, applied identically wherever it runs, and
    the source frame is untouched.

    Args:
        mappings: One value-to-number mapping per column, keyed by column label.
        unknown_value: What an unmapped value becomes. ``None`` -- the default --
            makes an unmapped value an error rather than a silent number.
    """

    def __init__(
        self,
        mappings: Mapping[Hashable, Mapping[Any, Any]] | None = None,
        unknown_value: float | None = None,
    ) -> None:
        self.mappings = mappings
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> ExplicitMappingEncoder:
        """Record the columns and check every mapping is usable."""
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.n_features_in_ = frame.shape[1]

        resolved = dict(self.mappings or {})
        self.mappings_ = {}
        for column in frame.columns:
            mapping = resolved.get(column, resolved.get(str(column)))
            if mapping is None:
                raise PreprocessingError(
                    f"No explicit mapping was supplied for {column!r}, but this "
                    "transformer was asked to encode it."
                )
            _require_finite_targets(mapping, column)
            self.mappings_[column] = dict(mapping)

        if self.unknown_value is not None:
            _require_finite_number(self.unknown_value, "unknown_value", "the mapping")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the mapping, leaving the input frame unchanged.

        Raises:
            PreprocessingError: If a value has no mapping and no ``unknown_value``
                was configured.
        """
        frame = _as_frame(X)
        columns = []
        for column in frame.columns:
            mapping = self.mappings_.get(column, self.mappings_.get(str(column)))
            if mapping is None:
                raise PreprocessingError(
                    f"Column {column!r} was not present when this mapping was fitted."
                )
            mapped = frame[column].map(mapping)
            unmapped = mapped.isna() & frame[column].notna()
            if unmapped.any():
                if self.unknown_value is None:
                    offenders = sorted({str(v) for v in frame[column][unmapped].unique()})[:5]
                    raise PreprocessingError(
                        f"Column {column!r} contains value(s) {offenders} with no entry "
                        "in the mapping you supplied. Add them, or set unknown_value "
                        "to give them an explicit encoding."
                    )
                mapped = mapped.where(~unmapped, self.unknown_value)
            columns.append(mapped.to_numpy(dtype="float64", na_value=np.nan))
        return np.column_stack(columns) if columns else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return one output name per input column."""
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray([str(f) for f in input_features], dtype=object)
        )
        return np.asarray([str(name) for name in names], dtype=object)


class NumericTextConverter(BaseEstimator, TransformerMixin):
    """Turns text that holds numbers into numbers, on request only.

    Never applied by default. Text that mostly parses as numbers is reported by
    the quality inspector and held for review, because converting it turns every
    value that fails to parse into a missing one -- and a column of sentinels
    quietly becomes a column of imputed medians.

    Args:
        errors: ``"raise"`` -- the default -- names the values that will not
            parse. ``"missing"`` routes them to whatever imputation follows,
            which is a decision the caller has to take deliberately.
    """

    def __init__(self, errors: str = "raise") -> None:
        self.errors = errors

    def fit(self, X: pd.DataFrame, y: Any = None) -> NumericTextConverter:
        """Record the columns and validate the policy."""
        if self.errors not in ("raise", "missing"):
            raise PreprocessingError(
                f'errors must be "raise" or "missing", got {self.errors!r}.'
            )
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Parse each column, leaving the input frame unchanged."""
        frame = _as_frame(X)
        columns = []
        for column in frame.columns:
            source = frame[column]
            parsed = pd.to_numeric(source, errors="coerce")
            unparsed = parsed.isna() & source.notna()
            if unparsed.any() and self.errors == "raise":
                offenders = sorted({str(v) for v in source[unparsed].unique()})[:5]
                raise PreprocessingError(
                    f"Column {column!r} holds value(s) {offenders} that are not "
                    'numbers. Set errors="missing" to treat them as missing, having '
                    "decided that is what they mean."
                )
            values = parsed.to_numpy(dtype="float64", na_value=np.nan)
            infinite = int(np.isinf(values).sum())
            if infinite:
                # Text such as "inf" parses to an infinity the profiler never saw,
                # so the plan could not hold the column back. Stopping here keeps
                # the promise that no infinity reaches an estimator.
                raise PreprocessingError(
                    f"Parsing column {column!r} produced {infinite} infinite value(s) "
                    "from text such as 'inf'. This version does not support infinity "
                    "as a trainable value; remove or cap those rows first."
                )
            columns.append(values)
        return np.column_stack(columns) if columns else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return one output name per input column."""
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray([str(f) for f in input_features], dtype=object)
        )
        return np.asarray([str(name) for name in names], dtype=object)


def _require_finite_number(value: Any, label: Any, where: str) -> None:
    """Reject a mapping destination that is not a finite real number.

    Raises:
        PreprocessingError: Naming the offending value and where it came from.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PreprocessingError(
            f"{where} maps {label!r} to {value!r}, which is not a number. An explicit "
            "mapping must give each category a numeric value."
        )
    if not np.isfinite(float(value)):
        raise PreprocessingError(
            f"{where} maps {label!r} to {value!r}. This version does not support "
            "infinity or NaN as a trainable value; choose a finite number."
        )


def _require_finite_targets(mapping: Mapping[Any, Any], column: Any) -> None:
    """Check every destination of one column's mapping."""
    for key, value in mapping.items():
        _require_finite_number(value, key, f"The explicit mapping for {column!r}")


def _require_no_text_collision(series: pd.Series, column: Any) -> None:
    """Refuse a column whose distinct values share a text form.

    ``1`` and ``"1"`` are different categories; ``True`` and ``"True"`` are
    different categories. Casting them both to text would merge them into one
    encoded level, and no later step could tell them apart again.

    Raises:
        PreprocessingError: Naming the values that would collide.
    """
    # Distinctness by type and text, not Series.unique(): that de-duplicates by
    # ``==``, so ``True`` swallows the integer ``1`` and hides its collision with
    # the string "1".
    values = {(type(value).__name__, value) for value in series.dropna()}
    if len(values) < 2:
        return
    by_text: dict[str, list[Any]] = {}
    for _, value in sorted(values, key=lambda item: (item[0], str(item[1]))):
        by_text.setdefault(str(value), []).append(value)
    collisions = {text: found for text, found in by_text.items() if len(found) > 1}
    if collisions:
        example = next(iter(collisions.items()))
        raise PreprocessingError(
            f"Column {column!r} holds distinct categories that share a text form: "
            f"{example[1]!r} would all become {example[0]!r}, merging categories that "
            "are not the same. Convert the column to a single consistent type before "
            "preprocessing."
        )


def _as_frame(X: Any) -> pd.DataFrame:
    """Accept a frame or anything pandas can wrap, without copying data."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


class OrdinalDomainGuard(BaseEstimator, TransformerMixin):
    """Refuses an ordinal value that has no place in the analyst's ordering.

    This is what :attr:`UnknownOrdinalPolicy.ERROR` promises, said where the
    column name is still known. ``OrdinalEncoder`` raises the same refusal one
    step later, by then knowing the column only as a position in an array.

    Args:
        orders: The permitted values of each ordinal column, lowest rank first.
    """

    def __init__(self, orders: Mapping[Any, Sequence[Any]] | None = None) -> None:
        self.orders = orders

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OrdinalDomainGuard":
        """Record the incoming column names. The permitted values are given."""
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` unchanged once every value is one the ordering ranks.

        Raises:
            PreprocessingError: If a column holds a value the ordering omits.
        """
        frame = _as_frame(X)
        permitted = {
            str(column): [str(value) for value in order]
            for column, order in (self.orders or {}).items()
        }
        for column in frame.columns:
            allowed = permitted.get(str(column))
            if allowed is None:
                continue
            values = frame[column]
            unknown = sorted(
                {str(value) for value in values[values.notna()].unique()} - set(allowed)
            )
            if unknown:
                raise PreprocessingError(
                    f"Column {column!r} holds value(s) {unknown[:5]} that the ordinal order "
                    f"{allowed} does not rank. An unranked value has no position on "
                    "the axis. Extend ordinal_orders for this column, or set "
                    "unknown_ordinal_policy to ENCODE to give unknowns a reserved code."
                )
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return one output name per input column; this step renames nothing."""
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray([str(f) for f in input_features], dtype=object)
        )
        return np.asarray(names, dtype=object)
