"""Comparing models fairly, and saying exactly what "fairly" meant.

Fair does not mean identical. Every model here sees the *same raw rows* and is
judged on the *same evaluation rows*, and each one receives the preparation its
own capabilities call for -- a scaler where scaling is required, gaps left in
place where the model consumes them, a sparse matrix where sparse is accepted.
Flattening those differences would make the comparison tidier and less true: a
k-nearest-neighbour model handed unscaled features is not a worse algorithm, it
is a badly prepared one, and a table that recorded it as the former would be
lying about what happened.

So the boundary that is held constant is the *experiment*, not the matrix:

    one split, drawn once
        -> the same training rows for every model
        -> the same evaluation rows for every model
        -> capability-driven preparation per model
        -> an independent fit per model
        -> one evaluation policy
        -> one ranking metric, in one direction

What this does not produce is a best model. It produces an ordering under one
dataset, one split, one preprocessing contract, one configuration and one
metric, and the result carries all six so the sentence can be written out in
full. "Random forest ranked first" is a claim about an experiment. "Random forest
is the best model" is a claim about the world, and nothing here measured that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import (
    AIDatasetKitError,
    IncompatibleModelError,
    TrainingError,
    ValidationError,
)
from aidatasetkit.core.types import TargetProfile, TaskType
from aidatasetkit.evaluation import (
    DEFAULT_RANKING_METRIC,
    EvaluationReport,
    MetricDirection,
    metric_direction,
)
from aidatasetkit.models import ModelFactory, ModelRegistry
from aidatasetkit.preprocessing import BlueprintCache, PreprocessingConfig
from aidatasetkit.profiling import TaskDetector
from aidatasetkit.profiling.task_detector import UNRESOLVED
from aidatasetkit.training.split import DataSplit, split_rows
from aidatasetkit.training.trainer import ModelTrainer, TrainingResult

__all__ = ["ModelOutcome", "ComparisonResult", "ModelComparator"]


def _readable(error: Exception) -> str:
    """A failure reason a person can act on.

    Whitespace-collapsed rather than first-line-only. Backend refusals are
    routinely several lines with the actionable half below the fold, and keeping
    only the first line threw away the remedy. An exception carrying no message
    at all is named by its type instead -- ``"".splitlines()`` is the empty list,
    and indexing it raised ``IndexError`` *inside* the handler, taking down the
    whole comparison over one model's silent failure.
    """
    text = " ".join(str(error).split())
    if not text:
        return f"{type(error).__name__} was raised with no message."
    return text[:400]


@dataclass(frozen=True, slots=True)
class ModelOutcome:
    """What happened to one model in a comparison.

    Either it trained and was measured, or it did not and the reason is here.
    Both are outcomes; a comparison that reported only the first would be a
    shorter table with no indication that it was shorter.
    """

    model_name: str
    succeeded: bool
    training: TrainingResult | None = None
    evaluation: EvaluationReport | None = None
    failure_reason: str | None = None
    failure_type: str | None = None

    def __post_init__(self) -> None:
        if self.succeeded and (self.training is None or self.evaluation is None):
            raise ValidationError(
                f"{self.model_name} is marked successful without a training result "
                "and an evaluation report."
            )
        if not self.succeeded and not self.failure_reason:
            raise ValidationError(
                f"{self.model_name} is marked failed without saying why."
            )

    def ranking_value(self, metric: str) -> float | None:
        """This model's score on the ranking metric, or ``None`` if it has none.

        ``None`` covers three situations that a ranking treats identically and a
        reader should not: the model failed, the metric was never attempted, or
        it was attempted and came back unavailable. The distinction survives in
        :attr:`evaluation` and in :attr:`failure_reason`.
        """
        if not self.succeeded or self.evaluation is None:
            return None
        if metric not in self.evaluation:
            return None
        entry = self.evaluation[metric]
        return entry.value if entry.is_available else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the outcome."""
        return {
            "model_name": self.model_name,
            "succeeded": self.succeeded,
            "training": None if self.training is None else self.training.to_dict(),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "failure_reason": self.failure_reason,
            "failure_type": self.failure_type,
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """One comparison run, and the context that makes it interpretable.

    Attributes:
        task_type: The task every model was held to.
        target_name: The column being predicted.
        ranking_metric: The metric that decided the order.
        ranking_direction: Which way that metric's scale runs.
        split: The one split every model shared.
        outcomes: Every model attempted, in canonical-name order -- which is
            execution order too, so a run is repeatable step for step.
        config_summary: The configuration in force, for the record.
    """

    task_type: TaskType
    target_name: str
    ranking_metric: str
    ranking_direction: MetricDirection
    split: DataSplit
    outcomes: tuple[ModelOutcome, ...]
    config_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> tuple[ModelOutcome, ...]:
        """The models that trained and were measured."""
        return tuple(outcome for outcome in self.outcomes if outcome.succeeded)

    @property
    def failed(self) -> tuple[ModelOutcome, ...]:
        """The models that did not, each with its reason."""
        return tuple(outcome for outcome in self.outcomes if not outcome.succeeded)

    @property
    def partial(self) -> bool:
        """Whether some models are missing from the ranking."""
        return bool(self.failed) or bool(self.unranked)

    @property
    def unranked(self) -> tuple[ModelOutcome, ...]:
        """Models that trained but have no value for the ranking metric.

        They are not failures and they are not ranked. Placing them anywhere in
        the order would be inventing a score; leaving them out silently would let
        a reader think every model was measured the same way.
        """
        return tuple(
            outcome
            for outcome in self.succeeded
            if outcome.ranking_value(self.ranking_metric) is None
        )

    @property
    def ranked(self) -> tuple[ModelOutcome, ...]:
        """The models that carry a ranking score, best first.

        Ordered by the metric in its own direction, ties broken by canonical
        name. The tie-break is alphabetical rather than incidental: registration
        order, dict order and hash order are all invisible to a reader and all
        capable of changing between runs.
        """
        scored = [
            (outcome, outcome.ranking_value(self.ranking_metric))
            for outcome in self.succeeded
        ]
        scored = [(outcome, value) for outcome, value in scored if value is not None]
        higher = self.ranking_direction is MetricDirection.HIGHER_IS_BETTER
        return tuple(
            outcome
            for outcome, _ in sorted(
                scored,
                key=lambda pair: (-pair[1] if higher else pair[1], pair[0].model_name),
            )
        )

    @property
    def best(self) -> ModelOutcome | None:
        """The model that ranked first, or ``None`` if none could be ranked.

        Named ``best`` for the position it holds in this table, not for a quality
        it possesses. See the module docstring.
        """
        ranked = self.ranked
        return ranked[0] if ranked else None

    def outcome_for(self, model_name: str) -> ModelOutcome:
        """Return one model's outcome by canonical name."""
        for outcome in self.outcomes:
            if outcome.model_name == model_name:
                return outcome
        raise KeyError(
            f"{model_name!r} was not part of this comparison. Compared: "
            f"{[o.model_name for o in self.outcomes]}."
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the leaderboard as a dataframe, best first, then the rest.

        Every model attempted gets a row, and every metric it computed gets a
        cell -- including the models that could not be ranked, whose other
        metrics were previously dropped from the table entirely.

        An unavailable metric shows its status rather than a blank. This is the
        table a person actually reads, and rendering ``roc_auc`` as an empty cell
        made "this model publishes no probabilities" indistinguishable from "this
        was never measured" -- reintroducing at the last step exactly the silence
        :class:`~aidatasetkit.evaluation.types.MetricStatus` exists to prevent.
        """
        def row_for(outcome: ModelOutcome, rank: int | None, status: str) -> dict:
            row: dict[str, Any] = {
                "rank": rank,
                "model": outcome.model_name,
                "status": status,
            }
            for metric in outcome.evaluation or ():
                row[metric.name] = (
                    metric.value if metric.is_available else metric.status.value
                )
            return row

        rows = [
            row_for(outcome, position, "ranked")
            for position, outcome in enumerate(self.ranked, start=1)
        ]
        rows += [
            row_for(outcome, None, f"no {self.ranking_metric}")
            for outcome in self.unranked
        ]
        rows += [row_for(outcome, None, "failed") for outcome in self.failed]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole run.

        Carries no data rows, no estimator, and no timing -- timing is
        observational and would make two identical runs compare unequal.
        """
        return {
            "task_type": self.task_type.value,
            "target_name": self.target_name,
            "ranking_metric": self.ranking_metric,
            "ranking_direction": self.ranking_direction.value,
            "split": self.split.to_dict(),
            "model_count": len(self.outcomes),
            "ranked_order": [outcome.model_name for outcome in self.ranked],
            "unranked": [outcome.model_name for outcome in self.unranked],
            "failed": [outcome.model_name for outcome in self.failed],
            "partial": self.partial,
            "config": dict(self.config_summary),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


class ModelComparator:
    """Runs several models across one experimental boundary.

    Args:
        config: Shared thresholds, the seed, and the evaluation fraction.
        preprocessing_config: Strategies for the preparation itself.
        registry: Registry to resolve model names against.

    Example:
        >>> from aidatasetkit.training import ModelComparator
        >>> comparator = ModelComparator()
        >>> # result = comparator.compare(frame, target="Churn")
        >>> # result.to_frame()
    """

    def __init__(
        self,
        config: KitConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._config = config if config is not None else KitConfig()
        self._preprocessing = (
            preprocessing_config if preprocessing_config is not None else PreprocessingConfig()
        )
        self._registry = registry

    def compare(
        self,
        frame: pd.DataFrame,
        *,
        target: str,
        models: list[str] | tuple[str, ...] | None = None,
        task: TaskType | str | None = None,
        positive_label: Any = UNRESOLVED,
        ranking_metric: str | None = None,
        model_params: dict[str, dict[str, Any]] | None = None,
    ) -> ComparisonResult:
        """Train and evaluate several models across one split.

        Args:
            frame: Features and target together. Read, never modified.
            target: The label column.
            models: Canonical names or aliases. ``None`` means every registered
                model compatible with the resolved task. Duplicates that resolve
                to one canonical name are trained once.
            task: Optional task hint, passed to the detector, which validates it
                against the data rather than accepting it.
            positive_label: Which class is the event of interest in a binary
                problem. Only ``0``/``1`` and ``False``/``True`` carry a
                conventional meaning; for ``"churn"``/``"stay"`` nothing in the
                data says which side matters, and precision, recall and ROC-AUC
                all invert if it is guessed wrong. Left unresolved, those metrics
                report macro averaging or an explicit reason rather than a number
                about the wrong class.
            ranking_metric: Which metric decides the order. Defaults to ``f1``
                for classification and ``rmse`` for regression.
            model_params: Optional per-model estimator parameters, keyed by the
                name as supplied. Passed through unchanged; nothing is searched.

        Returns:
            A :class:`ComparisonResult`.

        Raises:
            TrainingError: If the shared preconditions fail -- an absent target,
                too few rows, an unsplittable frame, or every model failing for
                one shared reason.
        """
        target_profile = self._resolve_target(frame, target, task, positive_label)
        chosen = self._resolve_models(target_profile.task_type, models)
        parameters = self._resolve_params(
            model_params, target_profile.task_type, chosen
        )
        metric = ranking_metric or DEFAULT_RANKING_METRIC[target_profile.task_type]
        direction = self._direction_of(metric, target_profile.task_type)

        # Drawn once, before any model is touched. This single object is what
        # makes the comparison fair; a per-model split would make every number
        # in the table incomparable with every other.
        split = split_rows(frame[target], target_profile.task_type, self._config)
        train_frame = split.take(frame)
        evaluation_frame = split.take(frame, evaluation=True)

        # One cache across the run, so capability-identical models are shown to
        # resolve to one blueprint. Each build still produces fresh, unfitted
        # transformers -- the cache holds shapes, never learned state.
        cache = BlueprintCache()
        # One trainer, and one set of established facts about the training rows.
        # The profile and the quality report -- the leakage scan among them --
        # describe the data, not the model, so computing them per model gave the
        # same answer nine times over.
        trainer = ModelTrainer(
            config=self._config,
            preprocessing_config=self._preprocessing,
            cache=cache,
            registry=self._registry,
        )
        established = trainer.establish(train_frame, target)

        outcomes = [
            self._run_one(
                trainer,
                name,
                train_frame,
                evaluation_frame,
                target=target,
                target_profile=target_profile,
                established=established,
                params=parameters.get(name),
            )
            for name in chosen
        ]

        self._refuse_a_shared_failure(outcomes)

        return ComparisonResult(
            task_type=target_profile.task_type,
            target_name=str(target),
            ranking_metric=metric,
            ranking_direction=direction,
            split=split,
            outcomes=tuple(outcomes),
            config_summary={
                "random_state": self._config.random_state,
                "validation_size": self._config.validation_size,
                "shuffle": self._config.shuffle,
                "preprocessing": self._preprocessing.to_dict(),
            },
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_target(
        self,
        frame: pd.DataFrame,
        target: str,
        task: TaskType | str | None,
        positive_label: Any,
    ) -> TargetProfile:
        """Ask the shared authority what this target is. Never decide it here."""
        if target not in frame.columns:
            raise TrainingError(
                f"The target {target!r} is not a column of this frame. Available: "
                f"{[str(c) for c in list(frame.columns)[:20]]}."
            )
        if frame.shape[1] < 2:
            raise TrainingError(
                "The frame holds only the target column, so there are no features "
                "to learn from."
            )
        return TaskDetector(self._config).detect(
            frame[target],
            hint=task,
            target_name=target,
            positive_label=positive_label,
        )

    def _resolve_models(
        self, task_type: TaskType, models: list[str] | tuple[str, ...] | None
    ) -> tuple[str, ...]:
        """Turn what the caller typed into canonical names, in a stable order.

        An alias and its canonical name are the same model, and a caller who
        wrote both meant one fit rather than two. The order is alphabetical by
        canonical name so that execution order is reproducible and visible,
        rather than inherited from registration or from a dictionary.
        """
        if models is None:
            every = ModelFactory.available(task=task_type, registry=self._registry)
            if not every:
                raise TrainingError(
                    f"No {task_type.value} model is registered, so there is "
                    "nothing to compare. An empty leaderboard would report no "
                    "failures and no results, which reads as success."
                )
            return every

        if isinstance(models, str):
            raise ValidationError(
                f"models must be a sequence of names, not the single string "
                f"{models!r}. Pass [{models!r}] to compare one model."
            )
        if not models:
            raise TrainingError(
                "No models were named, so there is nothing to compare. Pass "
                "models=None to compare every model compatible with this task."
            )

        resolved: dict[str, None] = {}
        for name in models:
            entry = ModelFactory.registration(
                name, task=task_type, registry=self._registry
            )
            entry.require_available()
            resolved[entry.canonical_name] = None
        return tuple(sorted(resolved))

    def _resolve_params(
        self,
        model_params: dict[str, dict[str, Any]] | None,
        task_type: TaskType,
        chosen: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        """Key the caller's parameters by canonical name, and refuse strays.

        ``models=`` accepts aliases, so ``model_params`` must too -- an earlier
        version looked the parameters up under the canonical name only, so
        ``model_params={"knn": {...}}`` was silently discarded and the model was
        trained at its defaults while the record showed nothing amiss. Silently
        ignoring configuration is worse than refusing it: the run looks like the
        one that was asked for.

        A key naming no model in this run is an error for the same reason. A typo
        that changes nothing is the hardest kind to notice.
        """
        if not model_params:
            return {}

        resolved: dict[str, dict[str, Any]] = {}
        for name, params in model_params.items():
            try:
                entry = ModelFactory.registration(
                    name, task=task_type, registry=self._registry
                )
            except IncompatibleModelError:
                # Names a real model of the *other* task family. One mapping
                # covering both families is an ordinary thing to keep, and a
                # regressor's entry is simply not this run's business.
                continue
            except AIDatasetKitError as error:
                # Resolves to nothing at all. That is a typo, and a typo that
                # changed nothing while looking as though it had is the failure
                # this check exists for.
                raise ValidationError(
                    f"model_params names {name!r}, which is not a model: {error}"
                ) from None

            canonical = entry.canonical_name
            if canonical not in chosen:
                # A real model, just not one this run compares. Keeping a single
                # parameter mapping across several runs is an ordinary thing to
                # do, and refusing it would make that pattern impossible without
                # catching anything: the key names a model, so it is not a typo.
                continue
            if canonical in resolved and resolved[canonical] != params:
                raise ValidationError(
                    f"model_params gives {canonical} two different sets of "
                    "parameters under different names. Only one can be applied, "
                    "and choosing would be guessing."
                )
            resolved[canonical] = params
        return resolved

    @staticmethod
    def _direction_of(metric: str, task_type: TaskType) -> MetricDirection:
        """Check the metric exists *and* belongs to this task's policy.

        Knowing the name was not enough. ``r2`` is a real metric with a real
        direction, and asking a classification run to rank on it passed
        validation, produced a report that never contained it, and returned a
        leaderboard in which every model was silently unranked -- a table with no
        order and no error explaining why.
        """
        from aidatasetkit.evaluation import CLASSIFICATION_METRICS, REGRESSION_METRICS

        allowed = {
            TaskType.CLASSIFICATION: CLASSIFICATION_METRICS,
            TaskType.REGRESSION: REGRESSION_METRICS,
        }[task_type]
        if metric not in allowed:
            raise ValidationError(
                f"{metric!r} cannot rank a {task_type.value} run. That task "
                f"measures {list(allowed)}, and the default is "
                f"{DEFAULT_RANKING_METRIC[task_type]!r}."
            )
        return metric_direction(metric)

    def _run_one(
        self,
        trainer: ModelTrainer,
        name: str,
        train_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        *,
        target: str,
        target_profile: TargetProfile,
        established: tuple[Any, Any],
        params: dict[str, Any] | None,
    ) -> ModelOutcome:
        """Train and evaluate one model, keeping its failure to itself.

        Only :class:`AIDatasetKitError` and a backend's own ``ValueError`` are
        turned into a recorded failure. ``TypeError`` is deliberately *not*
        caught: it is what a mistake in this file raises, and converting it into
        "model failed" would turn a bug in the orchestration into a
        plausible-looking data problem, sending the reader into their dataset
        looking for something that was never there. An earlier version of this
        method caught it anyway, contradicting this very paragraph.

        ``MemoryError`` is not caught either. A run that exhausted memory has not
        told you anything about the model, and continuing to fit the next one is
        unlikely to go better.
        """
        try:
            training = trainer.train(
                train_frame,
                target=target,
                model=name,
                target_profile=target_profile,
                model_params=params,
                established=established,
            )
            evaluation = trainer.evaluate(training, evaluation_frame, target=target)
        except (AIDatasetKitError, ValueError) as error:
            return ModelOutcome(
                model_name=name,
                succeeded=False,
                failure_reason=_readable(error),
                failure_type=type(error).__name__,
            )
        return ModelOutcome(
            model_name=name, succeeded=True, training=training, evaluation=evaluation
        )

    @staticmethod
    def _refuse_a_shared_failure(outcomes: list[ModelOutcome]) -> None:
        """Surface one shared cause rather than N copies of it.

        If every model failed the same way, the dataset violated something none
        of them owns, and reporting it once as a run-level failure says that.
        Nine identical rows would invite the reader to look for nine problems.

        At least two models are required before "they all agree" means anything.
        With one model compared, the set of reasons is trivially a set of one,
        and an earlier version therefore reported that single model's own failure
        as a property of the data -- blaming the dataset for something only that
        model had said.
        """
        if len(outcomes) < 2 or any(outcome.succeeded for outcome in outcomes):
            return
        reasons = {outcome.failure_reason for outcome in outcomes}
        if len(reasons) > 1:
            return
        raise TrainingError(
            f"Every one of the {len(outcomes)} models failed for the same reason, "
            "so this is a property of the data or the configuration rather than "
            f"of any model: {reasons.pop()}"
        )
