"""One entry point over the whole library, and nothing of its own.

Every method here delegates. The task comes from the detector, the summary from
the profiler, the statistics from the statistics engine, the findings from the
quality inspector, the plan from the planner, the split and the fit and the
metrics from the training layer, the labels from the target encoder. Not one line
of this module decides anything about data.

That is the point. A facade that reimplemented a check "for convenience" would be
a second AIDatasetKit hiding behind a friendlier name, and the two would disagree
eventually -- quietly, and in the direction nobody was watching.

What it does own is *sequence and state*: which results exist, which are stale,
and what a method needs before it can run. Loading a new frame discards
everything derived from the old one. Selecting a different model discards the fit
of the previous one. Those are the failures a session object exists to prevent.

.. rubric:: What it deliberately does not do

There is no ``run()``, no ``auto()``, no ``best_model()``. Comparison ranks;
choosing is a person's job, and a facade that quietly trained rank one would be
teaching that the ranking is an answer rather than a measurement. There is no
tuning, no cross-validation, and no persistence -- those are absent from the
library, and a convenience method cannot conjure them.

.. rubric:: The escape hatch

This class is optional. Every component it calls is public and composable, and an
analyst who wants the plan before the fit, or three preprocessors side by side,
should use them directly. The facade exists to shorten the syntax of the common
path, not to become the only path.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import (
    SchemaError,
    TrainingError,
    WorkflowStateError,
)
from aidatasetkit.core.types import (
    DatasetProfile,
    QualityReport,
    TargetProfile,
    TaskType,
)
from aidatasetkit.evidence import Verdict, decide_verdict
from aidatasetkit.facade.state import FinalEvaluation, Stage
from aidatasetkit.ingestion import IngestionLimits, load_table
from aidatasetkit.models import ModelFactory
from aidatasetkit.preprocessing import (
    BlueprintCache,
    PreprocessingConfig,
    PreprocessingPlanner,
)
from aidatasetkit.prediction import PredictionResult, predict_frame
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector
from aidatasetkit.profiling.task_detector import UNRESOLVED
from aidatasetkit.statistics import DescriptiveSummary, StatisticsEngine
from aidatasetkit.training import ComparisonResult, ModelComparator, ModelTrainer

__all__ = ["AIDataFacade"]


class AIDataFacade:
    """A guided path through the library, for one dataset and one question.

    Args:
        target: The column to predict. Required for every supervised task, and
            refused for ``task="clustering"``, which has no target at all.
        task: ``"classification"``, ``"regression"``, ``"clustering"``, or
            ``None`` to let the task detector decide from the target. A
            supervised hint is *checked* against the data, never accepted over it
            -- asking for classification on a column of prices is refused.

            ``"clustering"`` is different in kind: there is no target for a
            detector to read, so it cannot be inferred and must be asked for.
            Naming it changes which half of this object is available -- see
            :meth:`cluster`.
        id_column: A column identifying each row. It is not a feature and not an
            index; it is carried through to prediction output so an answer can be
            joined back to the row it belongs to.
        positive_label: Which class is the event of interest in a binary problem.
            Only ``0``/``1`` and ``False``/``True`` carry a conventional meaning;
            for ``"churn"``/``"stay"`` nothing in the data says which side
            matters, and precision, recall and ROC-AUC all invert if it is
            guessed.
        config: Shared thresholds and the seed. One authoritative configuration
            -- there is deliberately no ``AIDataFacade(random_state=...)``
            shortcut competing with :class:`~aidatasetkit.core.config.KitConfig`.
        preprocessing_config: Strategies for preparation itself.

    Example:
        >>> from aidatasetkit import AIDataFacade
        >>> ai = AIDataFacade(target="Churn", task="classification")
        >>> # ai.load(train_df, test_df)
        >>> # ai.check_quality()
        >>> # results = ai.compare_models()
        >>> # ai.select_model("logistic_regression")
        >>> # ai.train(); ai.evaluate(); ai.predict_test()
    """

    def __init__(
        self,
        *,
        target: str | None = None,
        task: TaskType | str | None = None,
        id_column: str | None = None,
        positive_label: Any = UNRESOLVED,
        config: KitConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self._is_clustering = self._asks_for_clustering(task)
        if self._is_clustering:
            if target is not None:
                raise SchemaError(
                    f"task='clustering' was asked for together with "
                    f"target={target!r}, and clustering has no target. Rather "
                    "than quietly ignoring the column or quietly clustering on "
                    "it -- two different results, neither of them what was "
                    "asked for -- this is refused. Drop target= to cluster on "
                    f"every column, or drop {target!r} from the frame first if "
                    "it should not take part."
                )
        elif not isinstance(target, str) or not target.strip():
            raise SchemaError(
                f"A target column name is required, got {target!r}. The facade "
                "is built around one question about one column. The single "
                "exception is task='clustering', which has no target."
            )
        self._target = target
        self._task_hint = task
        self._id_column = id_column
        self._positive_label = positive_label
        self._config = config if config is not None else KitConfig()
        self._preprocessing = self._with_identifier_declared(
            preprocessing_config
            if preprocessing_config is not None
            else PreprocessingConfig(),
            id_column,
        )
        self._reset_everything()

    @staticmethod
    def _asks_for_clustering(task: TaskType | str | None) -> bool:
        """Whether the caller named clustering, without judging any other value.

        An unrecognised ``task`` is deliberately *not* rejected here. Refusing it
        at construction would move an error that :class:`TaskDetector` already
        raises, with a better message, from ``load`` to ``__init__`` -- so this
        answers only the one question it was asked and leaves every other value
        to the layer that owns it.
        """
        if task is None:
            return False
        try:
            return TaskType.coerce(task) is TaskType.CLUSTERING
        except Exception:  # noqa: BLE001 - not this method's error to raise
            return False

    @staticmethod
    def _with_identifier_declared(
        config: PreprocessingConfig, id_column: str | None
    ) -> PreprocessingConfig:
        """Tell the preprocessing layer which column is the identifier.

        Declaring ``id_column`` and then saying nothing to anybody was a hole with
        teeth. S4 has a contract for exactly this -- ``confirmed_id_columns`` --
        and the facade was not using it, so the column the caller had *named* as a
        record key was profiled, encoded and trained on like any other feature.
        On a frame where the id correlates with the target that is not a subtle
        loss of accuracy: a linear model scored R2 = 1.0 off the identifier alone.

        The heuristic in the quality layer catches many identifiers on its own,
        but it is a heuristic and it is allowed to be wrong. A column the caller
        has declared is not a guess, and it belongs in the field that says so.
        """
        if id_column is None:
            return config
        import dataclasses

        already = tuple(config.confirmed_id_columns)
        if id_column in already:
            return config
        return dataclasses.replace(
            config, confirmed_id_columns=already + (id_column,)
        )

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def _reset_everything(self) -> None:
        """Return to the state of a freshly constructed session."""
        self._train: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None
        self._target_profile: TargetProfile | None = None
        self._reset_from_load()

    def _reset_from_load(self) -> None:
        """Discard everything derived from a loaded frame."""
        self._profile: DatasetProfile | None = None
        self._statistics: dict[str, DescriptiveSummary] | None = None
        self._quality: QualityReport | None = None
        self._verdict: Verdict | None = None
        self._verdict_reasons: tuple[str, ...] = ()
        self._plans: dict[str, Any] | None = None
        self._comparison: ComparisonResult | None = None
        # Set by evaluate_final and cleared only here, by a new load. See
        # _refuse_if_frozen for what it forbids and why.
        self._frozen = False
        self._reset_from_selection()

    def _reset_from_selection(self) -> None:
        """Discard everything derived from a chosen model."""
        self._selected: str | None = None
        self._reset_from_training()

    def _reset_from_training(self) -> None:
        """Discard everything derived from a fit.

        ``_split`` and ``_trainer`` belong here too. They were written by
        ``train`` and cleared by nothing, so a session that loaded a second frame
        kept the first one's split object alive -- harmless only because every
        path that reads them also requires ``_training``, which is exactly the
        kind of "safe by accident" that stops being safe when somebody adds a
        method.
        """
        self._training: Any = None
        self._evaluation: Any = None
        self._predictions: PredictionResult | None = None
        self._split: Any = None
        self._trainer: Any = None
        self._clustering: Any = None
        self._final: FinalEvaluation | None = None

    @property
    def stage(self) -> Stage:
        """Roughly where this session has got to. Reporting, not gatekeeping."""
        if self._train is None:
            return Stage.EMPTY
        # A clustering run fits and measures in one call, so it arrives at
        # EVALUATED directly. There is no held-out score behind that word here --
        # ClusteringRunner's docstring is explicit that the metrics are
        # in-sample -- it means the partition has been found and measured.
        if self._clustering is not None:
            return Stage.EVALUATED
        if self._final is not None:
            return Stage.FINAL_EVALUATED
        if self._evaluation is not None:
            return Stage.EVALUATED
        if self._training is not None:
            return Stage.TRAINED
        if self._selected is not None:
            return Stage.MODEL_SELECTED
        if self._comparison is not None:
            return Stage.COMPARED
        if self._plans is not None:
            return Stage.PREPARED
        return Stage.LOADED

    @property
    def status(self) -> dict[str, Any]:
        """A read-only summary of what this session holds.

        Deliberately free of estimators, preprocessors and data. It answers
        "where am I and what can I do next", which is what a person asks; it is
        not a handle onto internal state.
        """
        return {
            "stage": self.stage.value,
            "target": self._target,
            "id_column": self._id_column,
            "task": self._reported_task(),
            "train_rows": None if self._train is None else int(len(self._train)),
            "test_rows": None if self._test is None else int(len(self._test)),
            "profiled": self._profile is not None,
            "statistics_computed": self._statistics is not None,
            "quality_checked": self._quality is not None,
            "verdict": None if self._verdict is None else self._verdict.value,
            "prepared": self._plans is not None,
            "compared": self._comparison is not None,
            "selected_model": self._selected,
            "trained": self._training is not None,
            "evaluated": self._evaluation is not None,
            "predicted": self._predictions is not None,
            "validation_used_for_selection": self._validation_used_for_selection(),
            "final_evaluated": self._final is not None,
        }

    def _reported_task(self) -> str | None:
        """The task for :attr:`status`, without requiring data to be loaded.

        ``status`` is the one method that must answer at any stage, including
        before ``load``, so it cannot go through :attr:`task`.
        """
        if self._is_clustering:
            return TaskType.CLUSTERING.value
        if self._target_profile is None:
            return None
        return self._target_profile.task_type.value

    def _require(self, what: Any, operation: str, prerequisite: str) -> Any:
        """Refuse an operation whose prerequisite is missing, and say which."""
        if what is None:
            raise WorkflowStateError(
                f"Cannot {operation} yet: {prerequisite}. Current stage is "
                f"{self.stage.value!r}."
            )
        return what

    def _require_supervised(self, operation: str) -> None:
        """Refuse a supervised operation on a clustering session.

        Raised as a workflow-state error rather than allowed to fail deeper: with
        ``self._target`` set to ``None``, ``self._train[self._target]`` raises a
        ``KeyError`` naming ``None``, which tells a caller nothing about what
        they did wrong.
        """
        if self._is_clustering:
            raise WorkflowStateError(
                f"Cannot {operation}: this session was created with "
                "task='clustering', which has no target and therefore nothing "
                "to predict, rank against, or score. Use cluster() instead. For "
                "a supervised run, create a session with a target."
            )

    def _require_clustering(self, operation: str) -> None:
        """Refuse a clustering operation on a supervised session."""
        if not self._is_clustering:
            raise WorkflowStateError(
                f"Cannot {operation}: this session was created with "
                f"target={self._target!r}, so it is a supervised run. Clustering "
                "ignores labels entirely, and running it here would quietly "
                "cluster on the target column as though it were a feature. "
                "Create a session with task='clustering' and no target."
            )

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None = None,
        *,
        limits: IngestionLimits | None = None,
    ) -> AIDataFacade:
        """Take ownership of a training frame, and optionally a test frame.

        Loading means loading. Nothing is profiled, prepared, fitted or measured
        here; the only work done is resolving the task, because every later step
        needs one answer to that and re-deriving it per step is how two parts of a
        run come to disagree.

        **This invalidates everything.** A session that kept a model trained on
        the previous frame would answer questions about data it had never seen.

        Args:
            train: Features and target together. Stored by reference and never
                modified; every internal operation reads or copies.
            test: Optional external rows for final prediction. It may or may not
                carry the target -- inference does not need the answer. It never
                takes part in fitting, comparison, ranking or evaluation.

        Returns:
            ``self``, so a load can be chained.

        Raises:
            SchemaError: If a frame is not a DataFrame, the target is absent from
                the training frame, or a declared identifier column is missing.
            TrainingError: If the training frame has no rows.
        """
        if limits is not None:
            load_table(train, limits=limits)
            if test is not None:
                load_table(test, limits=limits)
        self._require_frame(train, "train")
        if not self._is_clustering and self._target not in train.columns:
            raise SchemaError(
                f"The target {self._target!r} is not a column of the training "
                f"frame. Available: {[str(c) for c in list(train.columns)[:20]]}."
            )
        if not len(train):
            raise TrainingError("The training frame has no rows.")
        if self._id_column is not None and self._id_column not in train.columns:
            raise SchemaError(
                f"The identifier column {self._id_column!r} is not in the training "
                "frame. Declare the one that is there, or drop id_column."
            )
        if test is not None:
            self._require_frame(test, "test")
            if not len(test):
                raise TrainingError(
                    "The test frame has no rows. A session that accepted it would "
                    "report a test set it could never predict for."
                )
            self._require_unique_columns(test, "test")
            if self._id_column is not None and self._id_column not in test.columns:
                raise SchemaError(
                    f"The identifier column {self._id_column!r} is not in the test "
                    "frame, so a prediction could not be joined back to its row."
                )
        self._require_unique_columns(train, "train")

        # Resolved before anything is stored, so a frame whose target cannot be
        # typed leaves the previous session intact rather than half-replaced.
        #
        # A clustering session has no target to resolve and keeps `None` here.
        # That is the honest record: a TargetProfile invented to fill the field
        # would put a task type nobody detected into the audit artifact, and
        # every consumer of it would be reading a fabrication.
        target_profile = (
            None
            if self._is_clustering
            else TaskDetector(self._config).detect(
                train[self._target],
                hint=self._task_hint,
                target_name=self._target,
                positive_label=self._positive_label,
            )
        )

        self._reset_everything()
        self._train = train
        self._test = test
        self._target_profile = target_profile
        return self

    @staticmethod
    def _require_frame(frame: Any, role: str) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise SchemaError(
                f"The {role} data must be a pandas DataFrame, got "
                f"{type(frame).__name__}."
            )

    @staticmethod
    def _require_unique_columns(frame: pd.DataFrame, role: str) -> None:
        """Refuse duplicate labels here rather than deep inside a transformer.

        ``frame[label]`` returns a frame rather than a series when the label
        repeats, and every component downstream misbehaves differently. Left to
        surface on its own it arrived as a third-party message naming no column.
        """
        labels = list(frame.columns)
        duplicated = sorted({str(l) for l in labels if labels.count(l) > 1})
        if duplicated:
            raise SchemaError(
                f"The {role} frame has duplicate column label(s) {duplicated}. "
                "Rename them before loading -- selecting one of them returns two "
                "columns, and every step after that reads the wrong thing."
            )

    @property
    def task(self) -> TaskType:
        """The task this session is running.

        For a supervised session this is what the detector resolved from the
        target column. For a clustering session it is what the caller asked for,
        because there is no column to resolve it from.
        """
        self._require(self._train, "report the task", "no data is loaded")
        if self._is_clustering:
            return TaskType.CLUSTERING
        return self._require(
            self._target_profile, "report the task", "no data is loaded"
        ).task_type

    # ------------------------------------------------------------------ #
    # Reading the data
    # ------------------------------------------------------------------ #

    def profile(self) -> DatasetProfile:
        """Describe the shape of the training data.

        Delegates to :class:`~aidatasetkit.profiling.profiler.DataProfiler`.
        Computed once and returned unchanged on repeat calls -- it is a reading of
        a frame that cannot change while it is loaded, so recomputing it would
        cost time and return the same answer.

        Does not check quality, prepare anything, or fit anything.
        """
        self._require(self._train, "profile", "no data is loaded")
        if self._profile is None:
            self._profile = DataProfiler(self._config).profile(self._train)
        return self._profile

    def statistics(self) -> dict[str, DescriptiveSummary]:
        """Describe every numeric column quantitatively.

        A thin adapter over :class:`~aidatasetkit.statistics.engine.StatisticsEngine`,
        which works one column at a time. The facade chooses *which* columns --
        the numeric ones, from the profile -- and computes nothing itself.

        The values handed to the engine are prepared exactly as the profiler
        prepares them: missing dropped, non-finite dropped. An earlier version
        passed the raw column with ``nan_policy="omit"``, which kept infinities --
        so a column holding one produced a mean of ``inf`` here and a finite mean
        in the profile, two different answers to one question from one library.

        Cached like :meth:`profile`.
        """
        self._require(self._train, "compute statistics", "no data is loaded")
        if self._statistics is None:
            import numpy as np

            from aidatasetkit.core.types import ColumnKind

            summaries: dict[str, DescriptiveSummary] = {}
            for name in self.profile().columns_of_kind(ColumnKind.NUMERIC):
                present = self._train[name].dropna()
                finite = present[
                    np.isfinite(present.to_numpy(dtype="float64", na_value=np.nan))
                ]
                if finite.empty:
                    continue
                summaries[str(name)] = StatisticsEngine(finite).describe()
            self._statistics = summaries
        return dict(self._statistics)

    def check_quality(self) -> QualityReport:
        """Look for the problems that quietly ruin a model.

        Delegates to
        :class:`~aidatasetkit.profiling.quality.DataQualityInspector`, **with the
        target named**. That is not incidental: the target-dependent checks --
        leakage above all -- go silent without it, and a feature exactly equal to
        the target would pass unnoticed. S7's adversarial review found precisely
        that defect one layer down, and a regression test holds this door shut.

        Also records the run's verdict through the existing
        :func:`~aidatasetkit.evidence.policy.decide_verdict`, so the facade does
        not grow a second opinion about what counts as a blocker.

        Cached like :meth:`profile`.
        """
        self._require(self._train, "check quality", "no data is loaded")
        if self._quality is None:
            report = DataQualityInspector(self._config).inspect(
                self._train, profile=self.profile(), target=self._target
            )
            self._quality = report
            self._verdict, self._verdict_reasons = self._decide(report)
        return self._quality

    def _decide(self, report: QualityReport) -> tuple[Verdict, tuple[str, ...]]:
        """Ask the evidence layer for the verdict it would record.

        Built through :class:`~aidatasetkit.evidence.builder.AuditBuilder` rather
        than by calling :func:`decide_verdict` directly, because the policy takes
        three inputs and an earlier version passed one. Rule 3 -- *a feature the
        planner held back is REVIEW_REQUIRED* -- reads the preprocessing
        decisions, so omitting them made the facade's verdict quietly **laxer**
        than the audit artifact's on identical data. Two verdicts in one library
        disagreeing about the same frame is precisely the failure this project
        exists to make impossible.

        Now there is one path to a verdict, and it is the one the CLI writes into
        ``audit.json``.
        """
        from aidatasetkit.evidence import AuditBuilder

        artifact = AuditBuilder(dataset_name=None).build(
            self._train,
            profile=self.profile(),
            quality=report,
            target=self._target_profile,
            plan=self._plan_for_verdict(),
        )
        return artifact.verdict, tuple(artifact.verdict_reasons)

    def _plan_for_verdict(self) -> Any:
        """A plan for the verdict to read held-back features from.

        Any capability profile will do: the decisions that reach the verdict --
        a held-back identifier, an infinity, a many-levelled category -- are taken
        before the profile is consulted, so every profile agrees about them.
        """
        planner = PreprocessingPlanner(self._preprocessing)
        from aidatasetkit.core.types import PreprocessingProfile

        try:
            return planner.plan(
                self._train,
                self.profile(),
                PreprocessingProfile(
                    requires_scaling=False,
                    supports_sparse_input=False,
                    handles_missing_values=False,
                ),
                quality=self._quality,
                target=self._target,
            )
        except Exception:  # noqa: BLE001 - a plan that cannot be built is not a verdict input
            return None

    @property
    def verdict(self) -> Verdict:
        """The readiness verdict, once quality has been checked."""
        self._require(
            self._verdict, "report the verdict", "quality has not been checked"
        )
        return self._verdict  # type: ignore[return-value]

    @property
    def verdict_reasons(self) -> tuple[str, ...]:
        """Why the verdict is what it is."""
        self._require(
            self._verdict, "report the verdict", "quality has not been checked"
        )
        return self._verdict_reasons

    # ------------------------------------------------------------------ #
    # Preparation
    # ------------------------------------------------------------------ #

    def prepare(self) -> dict[str, Any]:
        """Check that this data *can* be prepared, and work out how.

        **It fits nothing.** That is not a limitation, it is the contract: a
        preprocessor fitted here would have seen every row, and the training layer
        has not drawn its train/evaluation split yet. Fitting before that split is
        the single most effective way to destroy a model's evaluation, and a
        convenience method must not be the thing that does it.

        What it does is real work all the same. Preprocessing depends on model
        *capabilities*, and the twenty-four built-in models need five distinct
        pipelines between them, so there is no one prepared matrix to hand back.
        This derives the plan for each distinct capability profile the task's
        models require, which surfaces any planning failure now rather than in the
        middle of a comparison, and records what would happen to each column.

        For a supervised task the plans are derived from the **training side of
        the split**, which is the frame ``train()`` and ``compare_models()`` will
        actually plan against. Planning on every loaded row instead would report
        decisions that differ from the ones taken later -- a column constant in
        the training half but varying in the held-out half is decided differently
        by the two -- and a preview that does not match what happens is worse than
        no preview. A clustering run draws no split, so its preview is derived
        from every loaded row, for the same reason: that is what it will do.

        Returns:
            A mapping of capability-profile key to
            :class:`~aidatasetkit.preprocessing.types.PreprocessingPlan`.

        Raises:
            WorkflowStateError: If no data is loaded.
        """
        self._require(self._train, "prepare", "no data is loaded")
        if self._plans is not None:
            return dict(self._plans)

        # A clustering run draws no split -- three of the six clusterers cannot
        # label a row they were not fitted on, so there is no held-out side for
        # one to attach to -- and its runner fits on every loaded row. Planning
        # here on the same rows is therefore not a laxer preview, it is the
        # matching one; planning on a split half would report decisions that the
        # run never takes.
        if self._is_clustering:
            rows = self._train
        else:
            from aidatasetkit.training import split_rows

            rows = split_rows(
                self._train[self._target], self.task, self._config
            ).take(self._train)

        planner = PreprocessingPlanner(self._preprocessing)
        profile = DataProfiler(self._config).profile(rows)
        quality = DataQualityInspector(self._config).inspect(
            rows, profile=profile, target=self._target
        )

        plans: dict[str, Any] = {}
        for name in ModelFactory.available(task=self.task):
            capabilities = ModelFactory.registration(name).capabilities
            key = capabilities.preprocessing_profile().key
            if key in plans:
                continue
            plans[key] = planner.plan(
                rows,
                profile,
                capabilities.preprocessing_profile(),
                quality=quality,
                target=self._target,
            )
        self._plans = plans
        return dict(plans)

    # ------------------------------------------------------------------ #
    # Modelling
    # ------------------------------------------------------------------ #

    def compare_models(
        self,
        models: list[str] | tuple[str, ...] | None = None,
        *,
        ranking_metric: str | None = None,
        model_params: dict[str, dict[str, Any]] | None = None,
    ) -> ComparisonResult:
        """Run several models across one experimental boundary.

        A thin call into :class:`~aidatasetkit.training.comparison.ModelComparator`.
        The split, the capability-driven preparation, the metrics, the ranking and
        the failure isolation all belong to it; this method chooses nothing.

        **This is not hyperparameter optimization.** Comparing ten models is
        exactly ten fits at their published defaults.

        **It does not select a model.** A ranking is a measurement under one
        dataset, one split and one metric; deciding what to do about it is yours.

        Repeat calls re-run the comparison and replace the previous result, and
        invalidate any model selected from it -- a leaderboard from a comparison
        that no longer exists is worse than none.

        Raises:
            WorkflowStateError: If no data is loaded, or quality has not been
                checked and the data is blocked.
            TrainingError: If the run cannot proceed.
        """
        self._require(self._train, "compare models", "no data is loaded")
        self._require_supervised("compare models")
        self._refuse_if_frozen("compare models")
        self._refuse_if_blocked("compare models")

        result = ModelComparator(
            config=self._config,
            preprocessing_config=self._preprocessing,
        ).compare(
            self._train,
            target=self._target,
            models=models,
            task=self.task,
            positive_label=self._positive_label,
            ranking_metric=ranking_metric,
            model_params=model_params,
        )
        if not result.succeeded:
            # The comparator raises when every model fails for one shared reason,
            # and returns normally when they fail for different ones. That second
            # case is still a comparison of nothing: storing it would install a
            # result with an empty ranking *and* discard a perfectly good previous
            # fit, on the strength of a run in which nothing was measured.
            reasons = "; ".join(
                f"{o.model_name}: {o.failure_reason}" for o in result.failed
            )
            raise TrainingError(
                f"All {len(result.outcomes)} model(s) failed, so there is nothing "
                f"to compare -- {reasons}"
            )

        # Stored only after it succeeds. A failed comparison leaves the previous
        # one in place rather than installing a half-built result.
        self._comparison = result
        self._reset_from_selection()
        return result

    @property
    def comparison(self) -> ComparisonResult:
        """The most recent comparison."""
        return self._require(
            self._comparison, "report a comparison", "no comparison has been run"
        )

    def select_model(self, model: str) -> str:
        """Choose the model to train. A human decision, recorded.

        Resolves aliases and validates the choice against the resolved target
        through :class:`~aidatasetkit.models.factory.ModelFactory`, so a regressor
        cannot be selected for a classification target.

        **Selecting does not train.** The two are separate because they are
        separate decisions, and because a facade that fitted on selection would
        make "have a look at that model" cost a fit.

        Choosing a different model discards the previous fit, its evaluation and
        its predictions.

        Args:
            model: A canonical name or an alias. Most short aliases name a family
                and need no task here -- the facade already knows it.

        Returns:
            The canonical name, which is what every later step records.
        """
        self._require_supervised("select a model")
        self._require(self._target_profile, "select a model", "no data is loaded")
        self._refuse_if_frozen("select a model")
        entry = ModelFactory.registration(model, task=self.task)
        entry.require_available()
        # Checked against the resolved target, not merely the task family, so a
        # binary-only model meets a multiclass target here rather than inside a
        # fit several steps later.
        entry.capabilities.validate_for(self._target_profile)  # type: ignore[arg-type]

        if entry.canonical_name == self._selected:
            # Re-affirming the current choice is not a change, and discarding a
            # fit for it would punish saying the same thing twice. The docstring
            # promises "a different model"; this is what makes that true.
            return self._selected
        self._reset_from_selection()
        self._selected = entry.canonical_name
        return self._selected

    @property
    def selected_model(self) -> str:
        """The canonical name of the chosen model."""
        return self._require(
            self._selected, "report the selected model", "no model has been selected"
        )

    def train(self, **model_params: Any) -> Any:
        """Fit the selected model on the training rows.

        Delegates to :class:`~aidatasetkit.training.trainer.ModelTrainer`, which
        draws the split, prepares according to the model's capabilities, fits the
        preprocessor on training rows alone and then fits the estimator.

        Repeat calls refit and replace, discarding the previous evaluation and
        predictions along with the previous fit.

        Args:
            **model_params: Estimator parameters, passed through unchanged.
                Nothing here searches or tunes.

        Raises:
            WorkflowStateError: If no model has been selected, or the data is
                blocked.
        """
        self._require(self._train, "train", "no data is loaded")
        self._require_supervised("train")
        self._require(self._selected, "train", "no model has been selected")
        self._refuse_if_frozen("train")
        self._refuse_if_blocked("train")

        from aidatasetkit.training import split_rows

        split = split_rows(self._train[self._target], self.task, self._config)
        trainer = ModelTrainer(
            config=self._config, preprocessing_config=self._preprocessing
        )
        training = trainer.train(
            split.take(self._train),
            target=self._target,
            model=self._selected,
            target_profile=self._target_profile,  # type: ignore[arg-type]
            model_params=model_params or None,
        )
        # Only now. A fit that raised must not leave a session marked trained.
        self._reset_from_training()
        self._training = training
        self._split = split
        self._trainer = trainer
        return training

    @property
    def training(self) -> Any:
        """The most recent training result."""
        return self._require(
            self._training, "report a training result", "no model has been trained"
        )

    def evaluate(self) -> Any:
        """Measure the trained model on the validation rows.

        The validation rows are the held-out side of the split the training layer
        drew -- **not** the training rows, and **not** the external test frame.
        The model was not fitted on them. But if :meth:`compare_models` ranked
        models on the same split -- the same seed and fraction draw the same rows
        -- then these rows also *chose* the model, and this score is a
        development measurement, optimistic by however much the ranking
        exploited them. :attr:`status` reports that as
        ``validation_used_for_selection``. The independent estimate is
        :meth:`evaluate_final`, on an external test frame nothing else touched.

        Repeat calls recompute against the same held-out rows and return the same
        answer.

        Raises:
            WorkflowStateError: If no model has been trained.
        """
        self._require_supervised("evaluate")
        self._require(self._training, "evaluate", "no model has been trained")
        evaluation = self._trainer.evaluate(
            self._training,
            self._split.take(self._train, evaluation=True),
            target=self._target,
        )
        self._evaluation = evaluation
        return evaluation

    @property
    def evaluation(self) -> Any:
        """The most recent evaluation report."""
        return self._require(
            self._evaluation, "report an evaluation", "no evaluation has been run"
        )

    def evaluate_final(self) -> FinalEvaluation:
        """Measure the trained model, once, on the external test frame.

        This is the one independent estimate a session produces. The test frame
        has taken part in nothing before this call: not in profiling or the
        quality verdict (both read the training frame), not in planning or
        fitting a preprocessor (fitted on the training side of the split), not in
        comparison or ranking (validation rows only), not in selection (a
        person's choice from that ranking), and not in fitting the estimator.
        The library performs no hyperparameter search, no threshold search and
        no cross-validation, so there is no other route by which it could.

        **Calling this freezes the experiment.** A score seen on the test rows
        that could still send you back to compare, select or retrain would make
        those rows part of development, and the next score on them would no
        longer be a test. After this call :meth:`compare_models`,
        :meth:`select_model` and :meth:`train` refuse until new data is loaded.
        Repeat calls return the same result without measuring again.

        The fitted preprocessor is applied, never refitted. Test rows that are
        exact copies of training rows are counted and reported, because a model
        scored on rows it may have learned is not being tested on them.

        Raises:
            WorkflowStateError: If no model has been trained, or no test frame
                was loaded.
            TrainingError: If the test frame does not carry the target.
        """
        self._require_supervised("run the final evaluation")
        self._require(self._training, "run the final evaluation", "no model has been trained")
        self._require(
            self._test,
            "run the final evaluation",
            "no test frame was loaded -- pass one to load(train, test)",
        )
        if self._final is not None:
            return self._final

        report = self._trainer.evaluate(self._training, self._test, target=self._target)
        final = FinalEvaluation(
            model_name=self._selected,  # type: ignore[arg-type]
            report=report,
            test_rows=int(len(self._test)),
            rows_also_in_training=_rows_shared_with(self._test, self._train),
            validation_used_for_selection=self._validation_used_for_selection(),
        )
        # Only now: an evaluation that raised leaves the session unfrozen.
        self._final = final
        self._frozen = True
        return final

    @property
    def final_evaluation(self) -> FinalEvaluation:
        """The final evaluation, once it has been run."""
        return self._require(
            self._final, "report the final evaluation", "evaluate_final has not been run"
        )

    def _validation_used_for_selection(self) -> bool:
        """Whether the trained model's validation rows also ranked the comparison.

        Row identity, not configuration, decides it: the two splits are compared
        by fingerprint, so a comparison run under a different seed or fraction is
        correctly reported as not having seen these rows.
        """
        if self._comparison is None or self._split is None:
            return False
        return self._comparison.split.fingerprint == self._split.fingerprint

    def _refuse_if_frozen(self, operation: str) -> None:
        """Stop anything that could choose or refit a model after the final test."""
        if self._frozen:
            raise WorkflowStateError(
                f"Cannot {operation}: the final evaluation has been run, so the "
                "test rows have been seen. Choosing or refitting a model now would "
                "make them part of development, and a later score on them would "
                "not be a test. Load new data to start a new experiment."
            )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_test(self, *, with_probabilities: bool = False) -> PredictionResult:
        """Predict for the external test frame.

        Delegates to :func:`~aidatasetkit.prediction.predictor.predict_frame`. The
        fitted preprocessor is *applied*, never refitted, so an unseen category in
        the test data follows the vocabulary learned in training and does not join
        it. Nothing here can change what the model knows.

        Classification predictions come back as the original labels. Regression
        predictions are quantities and pass through no encoder at all.

        Args:
            with_probabilities: Attach class probabilities when the model declares
                it can produce them, with the original class labels as column
                names.

        Raises:
            WorkflowStateError: If no model has been trained, or no test frame was
                loaded.
        """
        self._require_supervised("predict for the test frame")
        self._require(self._training, "predict", "no model has been trained")
        self._require(
            self._test,
            "predict for test data",
            "no test frame was loaded -- pass one to load(train, test)",
        )
        predictions = predict_frame(
            self._training,
            self._test,
            id_column=self._id_column,
            with_probabilities=with_probabilities,
        )
        self._predictions = predictions
        return predictions

    def predict(
        self, frame: pd.DataFrame, *, with_probabilities: bool = False
    ) -> PredictionResult:
        """Predict for any frame, without storing the result.

        The same delegation as :meth:`predict_test`, for rows that are not the
        loaded test set. Nothing is recorded on the session, because a prediction
        for an arbitrary frame is not part of this session's story.
        """
        self._require_supervised("predict")
        self._require(self._training, "predict", "no model has been trained")
        # Delegated whole, rather than pre-inspecting the frame here. An earlier
        # version wrote ``self._id_column in frame.columns``, which did two wrong
        # things at once: a non-DataFrame died on ``.columns`` with a bare
        # AttributeError before the prediction layer's own typed guard could
        # speak, and a frame missing the declared identifier silently came back
        # with no ids at all -- a result nobody could join to anything, reported
        # as a success. Both are the prediction layer's decisions to make.
        return predict_frame(
            self._training,
            frame,
            id_column=self._id_column,
            with_probabilities=with_probabilities,
        )

    @property
    def predictions(self) -> PredictionResult:
        """The most recent stored test predictions."""
        return self._require(
            self._predictions, "report predictions", "predict_test has not been run"
        )

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #

    def cluster(self, model: str, **model_params: Any) -> Any:
        """Find a partition of the loaded rows, and measure its geometry.

        Available only on a session created with ``task="clustering"``. A thin
        call into :class:`~aidatasetkit.training.clustering.ClusteringRunner`,
        which owns the preparation, the fit and the metrics; this method chooses
        nothing.

        **The model is named, never chosen.** There is no ``cluster()`` without
        an argument that would pick one, and no search over ``k``. Both would be
        the library answering the question the analyst came to ask.

        **The metrics it returns are in-sample**, computed on the same rows the
        model was fitted on. That is a real limitation and the runner's docstring
        explains why it is not the leakage this library exists to prevent. Take a
        high silhouette as a description of this partition, not as evidence that
        it will reproduce on the next batch.

        Repeat calls refit and replace the previous result.

        Args:
            model: A canonical name or an alias -- ``kmeans``, ``dbscan``,
                ``birch`` and so on. Must be a clustering model.
            **model_params: Estimator parameters, passed through unchanged.
                ``n_clusters`` belongs here, and its default is inherited from
                scikit-learn rather than chosen for the data.

        Returns:
            A :class:`~aidatasetkit.training.clustering.ClusteringResult`.

        Raises:
            WorkflowStateError: If this is a supervised session, no data is
                loaded, or the data is blocked.
            IncompatibleModelError: If the named model is not a clusterer.
        """
        self._require_clustering("cluster")
        self._require(self._train, "cluster", "no data is loaded")
        # The same gate every supervised fit passes. Clustering does not make
        # BLOCKED data safe: a column of infinities or a frame that is 90%
        # missing wrecks a distance exactly as it wrecks a coefficient, and a
        # convenience method that skipped the check for the unsupervised half
        # would be a bypass in all but name.
        self._refuse_if_blocked("cluster")

        from aidatasetkit.training import ClusteringRunner

        runner = ClusteringRunner(
            config=self._config, preprocessing_config=self._preprocessing
        )
        result = runner.run(
            self._train,
            model=model,
            model_params=model_params or None,
            established=(self.profile(), self._quality),
        )
        # Only now, exactly as ``train`` does it. Resetting first would mean a
        # call that raised -- a misnamed model, a frame with no usable features --
        # discarded a perfectly good previous partition on its way out.
        self._reset_from_training()
        self._clustering = result
        return self._clustering

    @property
    def clustering(self) -> Any:
        """The most recent clustering result."""
        return self._require(
            self._clustering, "report a clustering", "cluster() has not been run"
        )

    # ------------------------------------------------------------------ #
    # Safety
    # ------------------------------------------------------------------ #

    def _refuse_if_blocked(self, operation: str) -> None:
        """Stop a run whose data the existing safety contract calls blocked.

        The verdict is recomputed here rather than read from the cache. The
        session holds the caller's frame by reference, so a frame edited after
        ``check_quality()`` -- a leaky column added, a constant introduced -- would
        otherwise be trained on under a verdict describing the data as it used to
        be. Re-inspecting costs a few milliseconds and removes the only route by
        which BLOCKED data could reach a fit.

        Quality is checked here even if the user never asked, because a facade
        that trained without looking would make the safety step optional in
        practice whatever the documentation said.

        There is deliberately **no override parameter**. The library has never had
        one -- nothing in it trained until now -- and inventing a bypass in the
        convenience layer would make the easy path the one that skips the check.
        An analyst who has read the findings and decided to proceed anyway can use
        the training layer directly; that route is public, documented, and
        requires saying so explicitly.
        """
        # The profile is refreshed first, because the quality report and the
        # verdict are both read from it -- a stale profile would make a fresh
        # inspection describe the old frame anyway.
        self._profile = DataProfiler(self._config).profile(self._train)
        self._quality = DataQualityInspector(self._config).inspect(
            self._train, profile=self._profile, target=self._target
        )
        self._verdict, self._verdict_reasons = self._decide(self._quality)
        if self._verdict is not Verdict.BLOCKED:
            return
        reasons = "; ".join(self._verdict_reasons) or "see check_quality()"
        raise WorkflowStateError(
            f"Cannot {operation}: this data is BLOCKED by the quality checks -- "
            f"{reasons}. Resolve the findings, or use aidatasetkit.training "
            "directly if you have read them and intend to proceed anyway. There "
            "is no override here, because a bypass on the convenient path is a "
            "bypass everybody takes."
        )


def _rows_shared_with(test: pd.DataFrame, train: pd.DataFrame) -> int | None:
    """How many test rows are exact copies of some training row.

    Compared on the columns the two frames share, in the training frame's order,
    by pandas' row hash -- values and dtypes both. A test row identical to a
    training row is one the model may have memorised, so the count bounds how
    optimistic a final score can be for that reason. ``None`` when the frames
    share no column or hold a value that cannot be hashed; the count is then
    unknown rather than zero.
    """
    shared = [column for column in train.columns if column in test.columns]
    if not shared:
        return None
    try:
        seen = set(pd.util.hash_pandas_object(train[shared], index=False).tolist())
        hashes = pd.util.hash_pandas_object(test[shared], index=False).tolist()
    except TypeError:
        return None
    return sum(1 for value in hashes if value in seen)
