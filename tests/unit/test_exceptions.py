"""Tests for the exception hierarchy.

Callers rely on catching a category rather than a leaf, so the parent
relationships are part of the public contract.
"""

from __future__ import annotations

import inspect

import pytest

from aidatasetkit.core import exceptions
from aidatasetkit.core.exceptions import (
    AIDatasetKitError,
    AmbiguousModelAliasError,
    AmbiguousTaskError,
    DomainError,
    EmptyDataError,
    IncompatibleModelError,
    MissingDependencyError,
    MissingValueError,
    ModelError,
    NonFiniteValueError,
    NonNumericDataError,
    ShapeError,
    TaskError,
    UnknownModelError,
    UnsupportedTaskError,
    ValidationError,
)

ALL_EXCEPTIONS = [
    obj
    for _, obj in inspect.getmembers(exceptions, inspect.isclass)
    if issubclass(obj, BaseException) and obj.__module__ == exceptions.__name__
]


def test_every_exported_name_exists():
    assert sorted(exceptions.__all__) == sorted(cls.__name__ for cls in ALL_EXCEPTIONS)


@pytest.mark.parametrize("error_type", ALL_EXCEPTIONS, ids=lambda cls: cls.__name__)
def test_everything_derives_from_the_library_base(error_type):
    assert issubclass(error_type, AIDatasetKitError)


@pytest.mark.parametrize(
    ("error_type", "parent"),
    [
        (EmptyDataError, ValidationError),
        (ShapeError, ValidationError),
        (NonNumericDataError, ValidationError),
        (MissingValueError, ValidationError),
        (NonFiniteValueError, ValidationError),
        (DomainError, ValidationError),
        (AmbiguousTaskError, TaskError),
        (UnsupportedTaskError, TaskError),
        (IncompatibleModelError, TaskError),
        (UnknownModelError, ModelError),
        (AmbiguousModelAliasError, ModelError),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_categories_can_be_caught_as_a_group(error_type, parent):
    assert issubclass(error_type, parent)
    with pytest.raises(parent):
        raise error_type("boom")


def test_a_missing_optional_package_is_not_a_model_failure():
    """It is raised for an absent plotting library too, so it hangs off the base."""
    assert issubclass(MissingDependencyError, AIDatasetKitError)
    assert not issubclass(MissingDependencyError, ModelError)


def test_the_library_base_does_not_swallow_unrelated_errors():
    assert not issubclass(AIDatasetKitError, ValueError)
    with pytest.raises(ValueError):
        try:
            raise ValueError("unrelated")
        except AIDatasetKitError:  # pragma: no cover - must not catch
            pytest.fail("AIDatasetKitError caught an unrelated exception")
