"""Every time the example artifact's identity moved, and exactly why.

The identity is ``config_fingerprint(semantic_dict())`` of the audit_churn
example. It is allowed to move -- the fingerprint exists to report that the
record changed -- but never silently. Each move is a step below, and each step
carries the literal identity on either side of it and a *rollback*: the edit that
undoes that change and nothing else.

What is asserted is that, starting from the committed fixture and undoing the
steps newest first, every rollback reproduces the previous identity exactly. A
step that quietly carried a second change would leave a fingerprint that no
longer matches, and the chain would break at that step, naming it.

The chain runs on the committed fixture, not on a freshly built artifact. The
fixture records pandas dtype names, which differ between pandas majors, so a
rebuilt artifact's identity depends on the environment; the history of what
changed does not. Whether the current code reproduces the fixture is asserted
separately, against the fixture for the running pandas major.

**S9 -- clustering landed.** ``supports_out_of_sample_assignment`` joined
:class:`ModelCapabilities`, and the ``known_limitations`` scope sentence stopped
calling clustering out of scope. Neither was special-cased out of the builder to
keep the old number stable: hiding a capability, or leaving a scope statement the
library has outgrown, to preserve an identity would be arranging the record to
match the fingerprint.

**G0 -- the multicollinearity check.** ``KitConfig`` gained
``multicollinearity_vif_threshold``. Every threshold is recorded under
``config.settings.thresholds``, so the new one is too, and ``config.fingerprint``
-- the digest of ``config.settings`` -- moved with it. No finding changed: the
example has no numeric feature with a VIF at or above 10. The schema version does
not move, because no existing field changed shape or meaning; an artifact written
before this step is recognisable by the absent threshold and by its identity.

**G0 -- cv_folds removed.** The setting was declared, validated and recorded
under ``config.settings.thresholds`` while no code path read it: the package runs
no cross-validation. It is gone from ``KitConfig``, so it is gone from the record,
and ``config.fingerprint`` moved again. Nothing else changed, and the schema
version stays: an artifact that records ``cv_folds`` recorded a value that
steered nothing, which is exactly what this step stops asserting.

**G1-W1 -- how the table was read.** The artifact gained a top-level
``ingestion`` record, filled when the table came through
:func:`aidatasetkit.ingestion.load_table` and ``null`` otherwise. The example is
built from a frame, so its record is ``null``: the key is the whole change. The
schema version stays 1.0 by the same rule as before -- a field was added, none
changed shape or meaning -- and an artifact written before this step is
recognisable by the absent key.

**G1-W1 closure -- artifact schema 1.0 to 1.1.** The paragraph above is the
reasoning that shipped, and it was wrong: a new top-level key changes the shape
of the record, and the contract is that a change of shape is identifiable from
``schema_version`` alone, without comparing fingerprints. The version is now
1.1, a minor bump because the change is additive. It is its own step rather than
a rewrite of the previous one, because an artifact with ``ingestion`` and schema
1.0 was published on ``main`` (90ecfae) and has to stay explicable. Nothing else
moved: undoing the step is setting the version back, and that alone reproduces
the previous identity.

**G1-W5 -- artifact schema 1.1 to 1.2.** The artifact gained a top-level
``chunked_profiling`` record. The example is built from a frame and does not
opt into chunked profiling, so the record is ``null``. The version moves with
the key: undoing the step deletes the key and sets the version back to 1.1,
and that alone reproduces the previous identity. Publication schema 1.0 and
the package version are not part of this fingerprint.

**G1-W6 -- artifact schema 1.2 to 1.3.** The artifact gained a top-level
``source_selector`` record. The example is built from a frame, so the record is
``null``. Undoing the step deletes the key and sets the version back to 1.2,
and that alone reproduces the previous identity. The selected SQLite table is
recorded on SQLite artifacts and in their config fingerprint; this fixture has
no table to select.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from aidatasetkit.evidence import canonical_json
from aidatasetkit.evidence.fingerprint import config_fingerprint

EXAMPLE = Path(__file__).resolve().parents[1].parent / "examples" / "audit_churn"

#: The reference fixture: the one every recorded identity was measured on.
REFERENCE_FIXTURE = EXAMPLE / "expected_audit_semantic.json"

#: The scope sentence the artifact carried before clustering was verified.
SCOPE_BEFORE = (
    "Only the classification readiness verdict has been verified end to end. "
    "Regression models and their preprocessing are verified; clustering, time "
    "series, text, and images remain out of scope."
)


def _rollback_s9(semantic: dict) -> None:
    del semantic["model"]["capabilities"]["supports_out_of_sample_assignment"]
    semantic["known_limitations"] = [
        SCOPE_BEFORE if entry.startswith("Only the classification") else entry
        for entry in semantic["known_limitations"]
    ]


def _restore_threshold(name: str, value: object) -> Callable[[dict], None]:
    """Undo the removal of one recorded threshold, and nothing else."""

    def rollback(semantic: dict) -> None:
        settings = semantic["config"]["settings"]
        settings["thresholds"][name] = value
        semantic["config"]["fingerprint"] = config_fingerprint(settings)

    return rollback


def _rollback_ingestion(semantic: dict) -> None:
    del semantic["ingestion"]


def _rollback_schema_1_1(semantic: dict) -> None:
    semantic["schema_version"] = "1.0"


def _rollback_schema_1_2(semantic: dict) -> None:
    semantic["schema_version"] = "1.1"
    del semantic["chunked_profiling"]


def _rollback_schema_1_3(semantic: dict) -> None:
    semantic["schema_version"] = "1.2"
    del semantic["source_selector"]


def _rollback_threshold(name: str) -> Callable[[dict], None]:
    """Undo the addition of one recorded threshold, and nothing else.

    ``config.fingerprint`` is recomputed from ``config.settings`` exactly as the
    builder computes it, because it is a digest of those settings and would
    otherwise still describe the newer ones.
    """

    def rollback(semantic: dict) -> None:
        settings = semantic["config"]["settings"]
        del settings["thresholds"][name]
        semantic["config"]["fingerprint"] = config_fingerprint(settings)

    return rollback


@dataclass(frozen=True)
class Step:
    """One recorded move of the identity."""

    name: str
    before: str
    after: str
    rollback: Callable[[dict], None]


#: Oldest first. Each ``before`` is the previous step's ``after``; the literals
#: are kept so a change to the current code cannot quietly move both sides.
#: Regenerated by ``examples/audit_churn/generate_artifacts.py`` on the
#: reference environment.
CHAIN: tuple[Step, ...] = (
    Step(
        "S9 clustering capability and scope sentence",
        before="4d3139d4a04e34bd34243c80d8232475e30edbfe9daa9f660ef2038f37d95bbc",
        after="b8581966475da37125414a12afc3144d6b1468258ff3dcb3149368406b884852",
        rollback=_rollback_s9,
    ),
    Step(
        "G0 multicollinearity_vif_threshold recorded",
        before="b8581966475da37125414a12afc3144d6b1468258ff3dcb3149368406b884852",
        after="1c449f10c3f9b826e6c166522122f77cc602348a349d11b662ddfe540af5f470",
        rollback=_rollback_threshold("multicollinearity_vif_threshold"),
    ),
    Step(
        "G0 cv_folds removed from the recorded configuration",
        before="1c449f10c3f9b826e6c166522122f77cc602348a349d11b662ddfe540af5f470",
        after="0f0fd1d5e9785d46351d7d28dc15d2d4eda4cc7fbc54bf0509a2a76eb3b471a5",
        rollback=_restore_threshold("cv_folds", 5),
    ),
    Step(
        "G1-W1 ingestion record added to the artifact",
        before="0f0fd1d5e9785d46351d7d28dc15d2d4eda4cc7fbc54bf0509a2a76eb3b471a5",
        after="007838931330c91af0d430a3e90c8b5c98305f76d1f482a511302939d85e2c71",
        rollback=_rollback_ingestion,
    ),
    Step(
        "G1-W1 closure: artifact schema 1.0 -> 1.1",
        before="007838931330c91af0d430a3e90c8b5c98305f76d1f482a511302939d85e2c71",
        after="3e93dd5e11b75f51d99c16bee263723627b4f93bebeb10c31e2bf31c92ed6be0",
        rollback=_rollback_schema_1_1,
    ),
    Step(
        "G1-W5 artifact schema 1.1 -> 1.2 and chunked_profiling record",
        before="3e93dd5e11b75f51d99c16bee263723627b4f93bebeb10c31e2bf31c92ed6be0",
        after="39f4c43867f1b30f47a42986187c3cce4b8dbad1540e5e7a369e0e142d5d5d47",
        rollback=_rollback_schema_1_2,
    ),
    Step(
        "G1-W6 artifact schema 1.2 -> 1.3 and source_selector record",
        before="39f4c43867f1b30f47a42986187c3cce4b8dbad1540e5e7a369e0e142d5d5d47",
        after="2a33aff562bf32116d2f48587eda19af85e9a2db956a7e6c00b8a89c1bcf7c75",
        rollback=_rollback_schema_1_3,
    ),
)

#: The identity the committed reference fixture must carry today.
CURRENT = CHAIN[-1].after


def _stored() -> dict:
    return json.loads(REFERENCE_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def semantic() -> dict:
    import sys

    root = str(EXAMPLE.resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from examples.audit_churn.generate_artifacts import build_artifact

    return build_artifact(pd.read_csv(EXAMPLE / "train.csv")).semantic_dict()


class TestTheChain:
    def test_every_step_moves_the_identity(self):
        """Otherwise a step is asserting that nothing happened."""
        for step in CHAIN:
            assert step.before != step.after, step.name

    def test_the_steps_are_contiguous(self):
        for earlier, later in zip(CHAIN, CHAIN[1:]):
            assert earlier.after == later.before, later.name

    def test_the_committed_fixture_carries_the_current_identity(self):
        assert config_fingerprint(_stored()) == CURRENT

    def test_undoing_each_step_reproduces_the_identity_before_it(self):
        """A migration that quietly carried something else would be worse than none."""
        semantic = _stored()
        for step in reversed(CHAIN):
            assert config_fingerprint(semantic) == step.after, step.name
            step.rollback(semantic)
            assert config_fingerprint(semantic) == step.before, step.name


class TestTheCurrentCode:
    def test_it_reproduces_the_fixture_for_this_pandas_major(self, semantic):
        from tests.golden import semantic_fixture_path

        stored = json.loads(semantic_fixture_path().read_text(encoding="utf-8"))
        assert config_fingerprint(semantic) == config_fingerprint(stored)

    def test_the_new_fingerprint_is_deterministic(self, semantic):
        """Rebuilt from scratch, twice, and byte-identical both times.

        A fingerprint that moved once is a migration. A fingerprint that moves
        every time is a broken contract, and the difference is worth proving
        rather than assuming.
        """
        import sys

        root = str(EXAMPLE.resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from examples.audit_churn.generate_artifacts import build_artifact

        frame = pd.read_csv(EXAMPLE / "train.csv")
        again = build_artifact(frame).semantic_dict()
        assert canonical_json(again) == canonical_json(semantic)
        assert config_fingerprint(again) == config_fingerprint(semantic)


class TestS9:
    def test_the_field_is_present_in_the_artifact(self, semantic):
        """The point of the migration: the capability is now on the record."""
        capabilities = semantic["model"]["capabilities"]
        assert capabilities["supports_out_of_sample_assignment"] is False

    def test_the_scope_statement_no_longer_calls_clustering_out_of_scope(
        self, semantic
    ):
        """An artifact travels further than any other document this library writes.

        Leaving it saying clustering is unverified, when six clusterers are held
        to the same executed contracts as the eighteen supervised models, would
        be shipping a false statement to keep a number stable.
        """
        scope = [
            entry
            for entry in semantic["known_limitations"]
            if entry.startswith("Only the classification")
        ]
        assert len(scope) == 1
        sentence = scope[0]
        verified, _, out_of_scope = sentence.partition("remain out of scope")
        assert out_of_scope == "."
        # Clustering has moved from one half of the sentence to the other. The
        # word is still present -- it now says clustering *is* verified -- so
        # asserting its absence would pass for a sentence that never mentioned it.
        assert "clustering models and their preprocessing are verified" in verified
        assert "clustering, time series" not in verified
        # The verdict itself is still classification-only, and that has not been
        # narrowed, because it has not been verified for anything else.
        assert sentence.startswith(
            "Only the classification readiness verdict has been verified end to end."
        )

    def test_the_builder_does_not_hide_the_field_for_supervised_models(self, semantic):
        """No special case, which was an explicit requirement of the change.

        The example audits a logistic regression, for which the answer is a
        trivial ``False`` -- and it is emitted anyway, because a key that appears
        only on clustering rows would make every consumer branch on its absence.
        """
        assert (
            "supports_out_of_sample_assignment"
            in semantic["model"]["capabilities"]
        )
        assert semantic["model"]["task_type"] == "classification"


class TestTheIngestionStep:
    def test_the_key_is_present_and_null_for_a_frame_built_artifact(self, semantic):
        assert "ingestion" in semantic
        assert semantic["ingestion"] is None

    def test_nothing_but_the_key_moved(self):
        stored = _stored()
        _rollback_schema_1_3(stored)
        _rollback_schema_1_2(stored)
        _rollback_schema_1_1(stored)
        del stored["ingestion"]
        assert config_fingerprint(stored) == _step("G1-W1 ingestion").before


def _step(prefix: str) -> Step:
    (step,) = [step for step in CHAIN if step.name.startswith(prefix)]
    return step


class TestTheSchemaStep:
    """The version says what the record is, without comparing fingerprints."""

    def test_the_current_code_writes_1_3(self, semantic):
        from aidatasetkit.evidence import ARTIFACT_SCHEMA_VERSION

        assert semantic["schema_version"] == ARTIFACT_SCHEMA_VERSION == "1.3"

    def test_the_fixture_declares_1_3(self):
        assert _stored()["schema_version"] == "1.3"

    def test_setting_the_version_back_reproduces_the_intermediate_artifact(self):
        """The 1.0-with-ingestion artifact that 90ecfae published."""
        stored = _stored()
        _rollback_schema_1_3(stored)
        _rollback_schema_1_2(stored)
        _rollback_schema_1_1(stored)
        assert "ingestion" in stored
        assert config_fingerprint(stored) == _step("G1-W1 ingestion").after

    def test_then_removing_the_key_reproduces_the_pre_g1_artifact(self):
        stored = _stored()
        _rollback_schema_1_3(stored)
        _rollback_schema_1_2(stored)
        _rollback_schema_1_1(stored)
        del stored["ingestion"]
        assert config_fingerprint(stored) == _step("G1-W1 ingestion").before

    def test_only_the_version_moved(self):
        """Data, findings, decisions and lineage are what they were in 1.0."""
        current, previous = _stored(), _stored()
        _rollback_schema_1_3(current)
        _rollback_schema_1_3(previous)
        _rollback_schema_1_2(current)
        _rollback_schema_1_2(previous)
        _rollback_schema_1_1(previous)
        changed = {key for key in current if current[key] != previous[key]}
        assert changed == {"schema_version"}

    def test_a_1_0_artifact_is_recognisable_as_the_older_contract(self):
        older = _stored()
        _rollback_schema_1_3(older)
        _rollback_schema_1_2(older)
        _rollback_schema_1_1(older)
        assert older["schema_version"] == "1.0" != _stored()["schema_version"]
        assert "chunked_profiling" not in older


class TestTheChunkedSchemaStep:
    def test_the_key_is_present_and_null_when_chunking_was_not_requested(self, semantic):
        assert "chunked_profiling" in semantic
        assert semantic["chunked_profiling"] is None

    def test_deleting_the_key_and_the_version_reproduces_1_1(self):
        stored = _stored()
        _rollback_schema_1_3(stored)
        _rollback_schema_1_2(stored)
        assert stored["schema_version"] == "1.1"
        assert "chunked_profiling" not in stored
        assert config_fingerprint(stored) == _step("G1-W1 closure").after

    def test_only_the_record_and_the_version_moved(self):
        current, previous = _stored(), _stored()
        _rollback_schema_1_3(current)
        _rollback_schema_1_3(previous)
        _rollback_schema_1_2(previous)
        changed = set(current) ^ set(previous)
        changed.update(
            key for key in current.keys() & previous.keys() if current[key] != previous[key]
        )
        assert changed == {"schema_version", "chunked_profiling"}


class TestTheSourceSelectorSchemaStep:
    def test_the_key_is_present_and_null_for_a_frame_built_artifact(self, semantic):
        assert "source_selector" in semantic
        assert semantic["source_selector"] is None

    def test_deleting_the_key_and_the_version_reproduces_1_2(self):
        stored = _stored()
        _rollback_schema_1_3(stored)
        assert stored["schema_version"] == "1.2"
        assert "source_selector" not in stored
        assert config_fingerprint(stored) == _step("G1-W5 artifact schema").after

    def test_only_the_record_and_the_version_moved(self):
        current, previous = _stored(), _stored()
        _rollback_schema_1_3(previous)
        changed = set(current) ^ set(previous)
        changed.update(
            key for key in current.keys() & previous.keys() if current[key] != previous[key]
        )
        assert changed == {"schema_version", "source_selector"}


class TestTheCvFoldsStep:
    def test_the_setting_is_no_longer_recorded(self, semantic):
        assert "cv_folds" not in semantic["config"]["settings"]["thresholds"]

    def test_the_config_fingerprint_describes_the_recorded_settings(self, semantic):
        config = semantic["config"]
        assert config["fingerprint"] == config_fingerprint(config["settings"])


class TestTheMulticollinearityStep:
    def test_the_threshold_is_recorded_with_its_default(self, semantic):
        thresholds = semantic["config"]["settings"]["thresholds"]
        assert thresholds["multicollinearity_vif_threshold"] == 10.0

    def test_the_config_fingerprint_describes_the_recorded_settings(self, semantic):
        config = semantic["config"]
        assert config["fingerprint"] == config_fingerprint(config["settings"])

    def test_no_finding_moved_with_it(self, semantic):
        """Only the recorded configuration changed; the evidence did not."""
        codes = {finding["code"] for finding in semantic["findings"]}
        assert "possible_multicollinearity" not in codes
        assert "multicollinearity_not_assessed" not in codes

    def test_the_schema_version_did_not_move(self):
        """At this step, not today: the G1-W1 closure moved it to 1.1 later."""
        older = _stored()
        for step in reversed(CHAIN):
            before = older["schema_version"]
            step.rollback(older)
            if step.name.startswith("G0 multicollinearity"):
                assert before == older["schema_version"] == "1.0"
                break

    def test_an_older_artifact_is_recognisable(self):
        """A reader holding a pre-G0 artifact can tell, from the file alone."""
        older = _stored()
        for step in reversed(CHAIN):
            step.rollback(older)
            if step.name.startswith("G0 multicollinearity"):
                break
        assert "multicollinearity_vif_threshold" not in older["config"]["settings"]["thresholds"]
        assert config_fingerprint(older) != CURRENT
