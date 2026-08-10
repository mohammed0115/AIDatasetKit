"""The two transformers scikit-learn does not provide.

Both exist because the alternative was a pandas operation on the caller's frame,
which would neither survive a train/test split nor leave the caller's data alone.

They follow scikit-learn's conventions -- parameters stored unchanged in
``__init__``, learned state in trailing-underscore attributes -- so they clone,
compose in a pipeline, and report their output names like any other step.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
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
]


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
        """Return a ``float64`` array, missing values preserved as ``NaN``."""
        frame = _as_frame(X)
        columns = [
            frame[column].to_numpy(dtype="float64", na_value=np.nan)
            for column in frame.columns
        ]
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
            self.mappings_[column] = dict(mapping)
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
            columns.append(parsed.to_numpy(dtype="float64", na_value=np.nan))
        return np.column_stack(columns) if columns else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return one output name per input column."""
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray([str(f) for f in input_features], dtype=object)
        )
        return np.asarray([str(name) for name in names], dtype=object)


def _require_no_text_collision(series: pd.Series, column: Any) -> None:
    """Refuse a column whose distinct values share a text form.

    ``1`` and ``"1"`` are different categories; ``True`` and ``"True"`` are
    different categories. Casting them both to text would merge them into one
    encoded level, and no later step could tell them apart again.

    Raises:
        PreprocessingError: Naming the values that would collide.
    """
    values = series.dropna().unique()
    if len(values) < 2:
        return
    by_text: dict[str, list[Any]] = {}
    for value in values:
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
