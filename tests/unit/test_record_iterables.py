"""Bounded synchronous iterables of mapping records through ``load_table``.

Lists and tuples stay on their existing path. Generators, iterators and
re-iterable containers are consumed once, under the records limits, into one
DataFrame. ``AIDataFacade.load`` stays DataFrame-only.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict

import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    EmptyInputError,
    InvalidIngestionOptionsError,
    KeyLimitError,
    MalformedInputError,
    RecordLimitError,
    SchemaError,
    UnsupportedFormatError,
)
from aidatasetkit.evidence import ARTIFACT_SCHEMA_VERSION, AuditBuilder
from aidatasetkit.evidence.fingerprint import dataset_fingerprint
from aidatasetkit.facade import AIDataFacade
from aidatasetkit.ingestion import IngestionLimits, LoadOptions, SourceKind, load_table
from aidatasetkit.profiling import DataProfiler, DataQualityInspector


def _rows():
    return [
        {"b": 1, "a": "x"},
        {"a": "y", "c": 3},
        {"b": 4},
    ]


def _audit(source):
    loaded = load_table(source)
    frame = loaded.frame
    profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=profile)
    artifact = AuditBuilder().build(
        frame, profile=profile, quality=quality, ingestion=loaded.metadata,
    )
    return loaded, artifact


class TestSources:
    def test_a_generator_expression_of_mappings(self):
        loaded = load_table({"a": index} for index in range(2))
        assert list(loaded.frame["a"]) == [0, 1]
        assert loaded.metadata.source_kind is SourceKind.RECORDS
        assert loaded.selector is None
        assert loaded.metadata.format is None

    def test_a_generator_function_of_mappings(self):
        def gen():
            yield {"a": 1}
            yield {"a": 2}

        assert load_table(gen()).frame.shape == (2, 1)

    def test_iter_of_a_list(self):
        loaded = load_table(iter([{"a": 1}, {"a": 2}]))
        assert list(loaded.frame["a"]) == [1, 2]

    def test_a_custom_iterator(self):
        class Cursor:
            def __init__(self):
                self._rows = [{"a": 1}, {"a": 2}]
                self._index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._index >= len(self._rows):
                    raise StopIteration
                row = self._rows[self._index]
                self._index += 1
                return row

        assert load_table(Cursor()).frame.shape == (2, 1)

    def test_a_custom_reiterable_is_accepted(self):
        class Box:
            def __iter__(self):
                return iter([{"a": 1}, {"a": 2}])

        try:
            loaded = load_table(Box())
        except UnsupportedFormatError:
            raise AssertionError("custom re-iterable refused") from None
        assert list(loaded.frame["a"]) == [1, 2]

    def test_ordered_dict_records(self):
        loaded = load_table(
            iter([OrderedDict([("z", 1), ("a", 2)]), OrderedDict([("a", 3), ("m", 4)])])
        )
        assert list(loaded.frame.columns) == ["z", "a", "m"]

    def test_a_mapping_subclass(self):
        class Sub(dict):
            pass

        loaded = load_table(iter([Sub(a=1, b=2)]))
        assert list(loaded.frame.columns) == ["a", "b"]
        assert loaded.frame.iloc[0]["a"] == 1

    def test_an_existing_list_is_unchanged(self):
        loaded = load_table([{"name": "A", "score": 10}, {"name": "B", "score": 20}])
        assert list(loaded.frame.columns) == ["name", "score"]
        assert loaded.metadata.source_kind is SourceKind.RECORDS

    def test_an_existing_tuple_is_unchanged(self):
        assert load_table(({"a": 1}, {"a": 2})).frame.shape == (2, 1)

    def test_an_over_long_list_still_reports_its_full_length(self):
        with pytest.raises(RecordLimitError, match=r"5 > 2"):
            load_table([{"a": 1}] * 5, limits=IngestionLimits(max_records=2))


class TestEquivalence:
    def test_list_generator_iterator_and_box_match(self):
        rows = _rows()

        def gen():
            yield from rows

        class Box:
            def __iter__(self):
                return iter(rows)

        loaded = {
            "list": load_table(rows),
            "tuple": load_table(tuple(rows)),
            "generator": load_table(gen()),
            "iterator": load_table(iter(rows)),
            "box": load_table(Box()),
        }
        reference = loaded["list"]
        for name, other in loaded.items():
            pd.testing.assert_frame_equal(reference.frame, other.frame)
            assert other.metadata.warnings == reference.metadata.warnings, name
            assert other.metadata.to_dict() == reference.metadata.to_dict(), name
            assert dataset_fingerprint(reference.frame) == dataset_fingerprint(other.frame), name

    def test_list_and_iterator_fingerprints_match(self):
        rows = [
            {"created_at": f"2024-01-{day:02d}", "city": "a" if day % 2 else "b", "label": day % 2}
            for day in range(1, 9)
        ]
        _, listed = _audit(rows)
        _, streamed = _audit(iter([dict(row) for row in rows]))
        assert listed.schema_version == streamed.schema_version == ARTIFACT_SCHEMA_VERSION == "1.4"
        assert listed.semantic_fingerprint == streamed.semantic_fingerprint, "source type salts a fingerprint"
        assert listed.config.fingerprint == streamed.config.fingerprint
        assert listed.verdict == streamed.verdict
        assert listed.dataset.fingerprint == streamed.dataset.fingerprint
        assert listed.ingestion.to_dict() == streamed.ingestion.to_dict()
        assert listed.date_inference.to_dict() == streamed.date_inference.to_dict()
        assert listed.source_selector is None and streamed.source_selector is None

    def test_evidence_has_no_iterator_identity(self):
        def gen():
            yield {"a": 1}

        loaded = load_table(gen())
        blob = str(loaded.metadata.to_dict())
        assert "generator object" not in blob, "iterator identity enters evidence"
        assert "0x" not in blob


class TestLimits:
    def test_an_empty_generator(self):
        def gen():
            if False:
                yield {"a": 1}

        with pytest.raises(EmptyInputError, match="iterable is empty"):
            load_table(gen())

    def test_one_record(self):
        assert load_table(iter([{"a": 1}])).frame.shape == (1, 1)

    def test_exactly_max_records(self):
        loaded = load_table(({"a": index} for index in range(2)), limits=IngestionLimits(max_records=2))
        assert list(loaded.frame["a"]) == [0, 1]

    def test_max_records_is_enforced(self):
        pulls = {"n": 0}

        class It:
            def __iter__(self):
                return self

            def __next__(self):
                pulls["n"] += 1
                if pulls["n"] > 6:
                    raise StopIteration
                return {"a": pulls["n"]}

        try:
            load_table(It(), limits=IngestionLimits(max_records=2))
        except RecordLimitError as exc:
            assert exc.observed_value == 3
            assert "a" not in str(exc)
        else:
            raise AssertionError("max_records ignored")
        assert pulls["n"] == 3, "max_records ignored"

    def test_the_pull_stops_at_max_records_plus_one(self):
        pulls = {"n": 0}

        class It:
            def __iter__(self):
                return self

            def __next__(self):
                pulls["n"] += 1
                if pulls["n"] > 6:
                    raise StopIteration
                return {"a": 1}

        with pytest.raises(RecordLimitError):
            load_table(It(), limits=IngestionLimits(max_records=2))
        assert pulls["n"] == 3, "iterator continues beyond max_records + 1"

    def test_materializing_with_list_is_not_how_the_limit_is_enforced(self):
        pulls = {"n": 0}

        class It:
            def __iter__(self):
                return self

            def __next__(self):
                pulls["n"] += 1
                if pulls["n"] > 6:
                    raise StopIteration
                return {"a": 1}

        try:
            load_table(It(), limits=IngestionLimits(max_records=2))
            refused = False
        except RecordLimitError:
            refused = True
        except Exception:
            refused = False
        assert refused and pulls["n"] == 3, "iterator materialized with list() before limits"

    def test_an_infinite_generator_stops_at_the_bound(self):
        pulls = {"n": 0}

        def infinite():
            while True:
                pulls["n"] += 1
                if pulls["n"] > 5:
                    raise AssertionError("iterator continues beyond max_records + 1")
                yield {"a": pulls["n"]}

        with pytest.raises(RecordLimitError):
            load_table(infinite(), limits=IngestionLimits(max_records=3))
        assert pulls["n"] == 4

    def test_max_keys_per_record(self):
        with pytest.raises(KeyLimitError):
            load_table(iter([{"a": 1, "b": 2}]), limits=IngestionLimits(max_keys_per_record=1))

    def test_later_row_columns_stop_the_pull(self):
        pulls = {"n": 0}

        def gen():
            rows = [{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}, {"a": 4}, {"a": 5}]
            for row in rows:
                pulls["n"] += 1
                yield row

        with pytest.raises(ColumnLimitError):
            load_table(gen(), limits=IngestionLimits(max_columns=2))
        assert pulls["n"] == 2, "max_columns ignored"

    def test_later_row_cells_stop_the_pull(self):
        pulls = {"n": 0}

        def gen():
            for index in range(6):
                pulls["n"] += 1
                yield {"a": index, "b": index}

        with pytest.raises(CellLimitError):
            load_table(gen(), limits=IngestionLimits(max_cells=5))
        assert pulls["n"] == 3, "max_cells ignored"

    def test_record_character_budget_applies_to_an_iterable(self):
        try:
            load_table(({"a": "abcd"} for _ in range(1)), limits=IngestionLimits(max_record_chars=3))
        except CellLimitError as exc:
            assert "abcd" not in str(exc)
        else:
            raise AssertionError("max_record_chars ignored")

    def test_bytes_use_the_same_character_budget_as_a_list(self):
        limits = IngestionLimits(max_record_chars=4)
        with pytest.raises(CellLimitError):
            load_table([{"a": b"abcd"}], limits=limits)
        with pytest.raises(CellLimitError):
            load_table(iter([{"a": b"abcd"}]), limits=limits)

    def test_max_rows_and_max_field_length_do_not_split_list_from_iterable(self):
        rows = [{"a": "x" * 10}, {"a": "y"}]
        limits = IngestionLimits(max_rows=1, max_field_length=3)
        listed = load_table(rows, limits=limits)
        streamed = load_table(iter(rows), limits=limits)
        pd.testing.assert_frame_equal(listed.frame, streamed.frame)

    def test_refusal_does_not_return_a_table(self):
        def gen():
            yield {"a": 1}
            yield {"a": "x" * 10}

        with pytest.raises(CellLimitError):
            load_table(gen(), limits=IngestionLimits(max_record_chars=4))


class TestMappingRules:
    def test_a_later_row_can_add_a_column(self):
        loaded = load_table(iter([{"a": 1}, {"b": 2}]))
        assert list(loaded.frame.columns) == ["a", "b"], "later-row columns ignored"
        assert bool(loaded.frame.iloc[0]["b"] != loaded.frame.iloc[0]["b"])

    def test_missing_keys_match_the_list_warning(self):
        rows = [{"a": 1, "b": 2}, {"a": 3}, {"b": 4}]
        assert load_table(iter(rows)).metadata.warnings == load_table(rows).metadata.warnings

    def test_reordered_later_keys_do_not_reorder_columns(self):
        loaded = load_table(iter([{"b": 1, "a": 2}, {"a": 3, "b": 4}]))
        assert list(loaded.frame.columns) == ["b", "a"]

    def test_an_empty_mapping_among_populated_rows(self):
        loaded = load_table(iter([{"a": 1}, {}, {"a": 3}]))
        assert loaded.frame["a"].isna().tolist() == [False, True, False]
        assert "3 records" in loaded.metadata.warnings[0]

    def test_all_empty_mappings(self):
        with pytest.raises(EmptyInputError, match="no columns"):
            load_table(iter([{}, {}]))

    def test_a_non_mapping_item_is_refused(self):
        try:
            load_table(iter([[1, 2], {"a": 1}]))
        except MalformedInputError as exc:
            assert "list" in str(exc)
            assert "[1, 2]" not in str(exc)
        except Exception:
            raise AssertionError("non-mapping yielded item accepted") from None
        else:
            raise AssertionError("non-mapping yielded item accepted")

    def test_a_non_mapping_after_valid_items(self):
        with pytest.raises(MalformedInputError, match="Record 1"):
            load_table(iter([{"a": 1}, "nope"]))

    def test_a_non_string_key(self):
        with pytest.raises(MalformedInputError, match="int key"):
            load_table(iter([{1: "x"}]))

    def test_the_non_string_key_message_omits_the_value(self):
        secret = "secret-cell"
        with pytest.raises(MalformedInputError) as caught:
            load_table(iter([{1: secret}]))
        assert secret not in str(caught.value)

    def test_the_yielded_mapping_is_not_retained(self):
        row = {"a": 1, "b": 2}

        def gen():
            yield row

        loaded = load_table(gen())
        row["a"] = 99
        row["c"] = 7
        assert loaded.frame.iloc[0]["a"] == 1, "yielded mapping retained by reference"
        assert list(loaded.frame.columns) == ["a", "b"]

    def test_nested_aliasing_matches_a_list(self):
        nested = [1]
        listed = load_table([{"a": nested}]).frame
        streamed = load_table(iter([{"a": nested}])).frame
        nested.append(2)
        assert listed.iloc[0, 0] == streamed.iloc[0, 0] == [1, 2]


class TestOnePass:
    def test_iter_is_called_once(self):
        class Box:
            def __init__(self):
                self.calls = 0

            def __iter__(self):
                self.calls += 1
                return iter([{"a": 1}, {"a": 2}])

        box = Box()
        load_table(box)
        assert box.calls == 1, "iterator consumed twice"

    def test_len_and_repr_are_not_called(self):
        class Box:
            def __iter__(self):
                return iter([{"a": 1}])

            def __len__(self):
                raise AssertionError("len(source) was called")

            def __repr__(self):
                raise AssertionError("repr(source) was called")

        assert load_table(Box()).frame.iloc[0, 0] == 1

    def test_file_options_and_selectors_do_not_apply(self):
        with pytest.raises(InvalidIngestionOptionsError):
            load_table(iter([{"a": 1}]), options=LoadOptions(delimiter=";"))
        with pytest.raises(InvalidIngestionOptionsError):
            load_table(iter([{"a": 1}]), sheet="Data")
        with pytest.raises(InvalidIngestionOptionsError):
            load_table(iter([{"a": 1}]), table="people")


class TestCleanup:
    def test_a_generator_is_closed_on_success(self):
        def gen():
            yield {"a": 1}

        source = gen()
        load_table(source)
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED

    def test_a_generator_is_closed_on_empty_input(self):
        def gen():
            if False:
                yield {"a": 1}

        source = gen()
        with pytest.raises(EmptyInputError):
            load_table(source)
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED

    def test_a_generator_is_closed_when_the_record_limit_refuses_it(self):
        def gen():
            yield {"a": 1}
            yield {"a": 2}
            yield {"a": 3}

        source = gen()
        with pytest.raises(RecordLimitError):
            load_table(source, limits=IngestionLimits(max_records=2))
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED, "GeneratorType not closed"

    def test_a_generator_is_closed_on_a_malformed_item(self):
        def gen():
            yield {"a": 1}
            yield "nope"
            yield {"a": 3}

        source = gen()
        with pytest.raises(MalformedInputError):
            load_table(source)
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED

    def test_a_generator_is_closed_on_a_resource_refusal(self):
        def gen():
            yield {"a": 1, "b": 2}
            yield {"c": 3}
            yield {"d": 4}

        source = gen()
        with pytest.raises(ColumnLimitError):
            load_table(source, limits=IngestionLimits(max_columns=2))
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED

    def test_a_generator_is_closed_after_its_own_exception(self):
        def gen():
            yield {"a": 1}
            raise RuntimeError("boom-without-row")

        source = gen()
        with pytest.raises(RuntimeError, match="boom-without-row"):
            load_table(source)
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED

    def test_an_iterator_exception_is_not_published(self):
        def gen():
            yield {"a": 1}
            raise RuntimeError("boom-without-row")

        try:
            load_table(gen())
        except RuntimeError as exc:
            assert str(exc) == "boom-without-row"
            assert "{'a': 1}" not in str(exc)
        else:
            raise AssertionError("iterator exception publishes partial output")

    def test_a_generator_is_closed_when_dataframe_construction_fails(self, monkeypatch):
        class Boom(pd.DataFrame):
            def __new__(cls, *args, **kwargs):
                raise RuntimeError("frame failed")

        monkeypatch.setattr("aidatasetkit.ingestion.loader.pd.DataFrame", Boom)

        def gen():
            yield {"a": 1}

        source = gen()
        with pytest.raises(RuntimeError, match="frame failed"):
            load_table(source)
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED

    def test_a_general_iterator_is_not_arbitrarily_closed(self):
        class It:
            def __init__(self):
                self.closed = False
                self._rows = iter([{"a": 1}])

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._rows)

            def close(self):
                self.closed = True

        source = It()
        load_table(source)
        assert source.closed is False


class TestRefusals:
    def test_a_bare_mapping_is_not_a_record_iterable(self):
        for source in ({"name": "A", "score": 1}, "abcd", b"abcd"):
            try:
                load_table(source)
            except UnsupportedFormatError:
                continue
            except Exception:
                raise AssertionError("mapping or string misread as records") from None
            raise AssertionError("mapping or string misread as records")

    def test_str_and_bytes_keep_the_path_refusal(self):
        with pytest.raises(UnsupportedFormatError, match="pathlib.Path"):
            load_table("abc")
        with pytest.raises(UnsupportedFormatError, match="pathlib.Path"):
            load_table(b"abc")

    @pytest.mark.parametrize(
        "source",
        [bytearray(b"ab"), {1, 2}, frozenset([1, 2]), range(3)],
    )
    def test_unordered_or_binary_iterables_are_refused(self, source):
        with pytest.raises(UnsupportedFormatError):
            load_table(source)

    def test_an_async_generator_is_refused_before_iteration(self):
        async def agen():
            yield {"a": 1}

        source = agen()
        try:
            load_table(source)
        except UnsupportedFormatError as exc:
            if "async" not in str(exc):
                raise AssertionError("async iterable accepted") from None
        except Exception:
            raise AssertionError("async iterable accepted") from None
        else:
            raise AssertionError("async iterable accepted")
        assert source.ag_running is False

    def test_an_async_iterable_is_refused_before_iteration(self):
        class AIter:
            def __init__(self):
                self.started = False

            def __aiter__(self):
                self.started = True
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        source = AIter()
        with pytest.raises(UnsupportedFormatError, match="async"):
            load_table(source)
        assert source.started is False

    def test_a_generator_of_sequences_is_refused(self):
        def gen():
            yield [1, 2]
            yield [3, 4]

        source = gen()
        with pytest.raises(MalformedInputError, match="not a mapping"):
            load_table(source)
        assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED


class TestFacadeBoundary:
    def test_the_facade_still_requires_a_dataframe(self):
        facade = AIDataFacade(target="label", task="classification")
        try:
            facade.load(({"label": 0} for _ in range(1)))
        except SchemaError as exc:
            assert "DataFrame" in str(exc)
        except Exception:
            raise AssertionError("facade contract silently broadened") from None
        else:
            raise AssertionError("facade contract silently broadened")
        assert facade.stage.value == "empty"


class TestRegressions:
    def test_a_dataframe_is_still_the_same_object(self):
        frame = pd.DataFrame({"a": [1]})
        assert load_table(frame).frame is frame

    def test_the_artifact_schema_stays_1_4(self):
        assert ARTIFACT_SCHEMA_VERSION == "1.4"
