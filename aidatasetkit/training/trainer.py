"""Training one model correctly, by composing the layers that already exist.

This module owns an *order*, not an algorithm. Every decision it makes was made
somewhere else: the task by the detector, the model by the registry, the
preprocessing by the capability profile, the encoding by the target contract.
What is new here is the sequence, and the guarantee that runs through it.

The guarantee is the one this project exists for:

    Nothing is fitted on evaluation rows. Ever.

That is not a comment. The preprocessor is fitted on the training frame alone and
then *applied* to the evaluation frame; the target encoder learns its vocabulary
from training labels alone. An evaluation row that would have moved a median, a
scaler's centre, or a one-hot vocabulary does not move it, and the test suite
proves that by comparing the learned state with and without such a row present.

The result is a record, not a handle. It carries the fitted estimator so a caller
can predict with it, and everything else it carries is metadata that survives
serialisation -- because a training run that cannot be described afterwards is a
number without a provenance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import TrainingError
from aidatasetkit.core.types import (
    Estimator,
    PreprocessingProfile,
    TargetProfile,
    TaskType,
)
from aidatasetkit.models import ModelFactory, ModelRegistry
from aidatasetkit.preprocessing import (
    BlueprintCache,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
    TargetLabelEncoder,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

__all__ = ["TrainingResult", "ModelTrainer"]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """One model, fitted, and everything needed to say how.

    Attributes:
        model_name: The canonical name, never the alias the caller typed.
        task_type: The task the target resolved to.
        estimator: The fitted estimator. Present so a caller can predict; it is
            never serialised, and :meth:`to_dict` does not reach it.
        preprocessor: The fitted preprocessor, likewise.
        target_encoding: The label mapping for a classification run, carrying the
            positive class. ``None`` for regression, where nothing is encoded.
        preprocessing_profile: The capability triple that chose the pipeline.
        plan_fingerprint: Identity of the preprocessing plan.
        feature_names: The transformed feature names, from S4.
        lineage: Each input column and what it became, from S4.
        train_row_count: How many rows were learned from.
        feature_count: Raw feature columns offered.
        transformed_feature_count: Columns the model actually received.
        model_parameters: The estimator's resolved parameters.
        random_state: The seed the fitted estimator actually carries -- read back
            from it rather than from the strategy's defaults, so an override
            passed by the caller is what appears here.
        supports_predict_proba: The model's *declared* capability, consulted
            before probabilities are ever requested.
        review_features: Columns the plan held back for a human decision. They
            are not in the matrix, and a training record that did not name them
            would let a column vanish between the audit and the fit.
        excluded_features: Columns the plan dropped outright, with the same
            reasoning.
        fit_seconds: Observational. Never part of a ranking or a fingerprint.
    """

    model_name: str
    task_type: TaskType
    estimator: Estimator
    preprocessor: Any
    target_encoding: Any
    preprocessing_profile: PreprocessingProfile
    plan_fingerprint: str
    feature_names: tuple[str, ...]
    lineage: dict[Any, tuple[str, ...]]
    train_row_count: int
    feature_count: int
    transformed_feature_count: int
    model_parameters: dict[str, Any] = field(default_factory=dict)
    random_state: int | None = None
    supports_predict_proba: bool = False
    review_features: tuple[str, ...] = ()
    excluded_features: tuple[str, ...] = ()
    fit_seconds: float | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for new rows, transforming them with the fitted preprocessor.

        The preprocessor is *applied*, never refitted -- which is what makes a
        prediction on unseen rows honest, and what keeps an unseen category
        following the vocabulary learned in training.
        """
        return np.asarray(self.estimator.predict(self.preprocessor.transform(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return class probabilities, or ``None`` if this model has none.

        Asked of the fitted object rather than of the model's name. A caller
        receives ``None`` instead of an ``AttributeError``, because "this model
        cannot do that" is an answer and a crash is not.
        """
        if not hasattr(self.estimator, "predict_proba"):
            return None
        return np.asarray(self.estimator.predict_proba(self.preprocessor.transform(X)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the run.

        No estimator, no preprocessor, no row of data. What is here is what would
        let somebody repeat the run: the model, the seed, the preprocessing
        identity, and the shapes.
        """
        from aidatasetkit.core.types import jsonable

        return {
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "preprocessing_profile": self.preprocessing_profile.key,
            "plan_fingerprint": self.plan_fingerprint,
            "train_row_count": self.train_row_count,
            "feature_count": self.feature_count,
            "transformed_feature_count": self.transformed_feature_count,
            "feature_names": list(self.feature_names),
            "lineage": {
                str(source): list(produced) for source, produced in self.lineage.items()
            },
            "model_parameters": {
                str(key): jsonable(value)
                for key, value in sorted(self.model_parameters.items())
            },
            "random_state": self.random_state,
            "supports_predict_proba": self.supports_predict_proba,
            "review_features": list(self.review_features),
            "excluded_features": list(self.excluded_features),
            "target_encoding": (
                None
                if self.target_encoding is None
                else self.target_encoding.encoding.to_dict()
            ),
        }


class ModelTrainer:
    """Fits one model on one training frame, in the order the layers require.

    Args:
        config: Shared thresholds and the seed.
        preprocessing_config: Strategies for the preparation itself.
        cache: Blueprint cache, shared across a comparison so that
            capability-identical models are shown to resolve to one blueprint.
            Only pipeline *shapes* are cached; nothing fitted is.
        registry: Model registry to resolve against. Defaults to the built-in one.

    Example:
        >>> from aidatasetkit.training import ModelTrainer
        >>> trainer = ModelTrainer()
        >>> # result = trainer.train(frame, target="Churn", model="logistic_regression")
    """

    def __init__(
        self,
        config: KitConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        cache: BlueprintCache | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._config = config if config is not None else KitConfig()
        self._preprocessing = (
            preprocessing_config if preprocessing_config is not None else PreprocessingConfig()
        )
        self._cache = cache if cache is not None else BlueprintCache()
        self._registry = registry

    @property
    def config(self) -> KitConfig:
        """The shared configuration in force."""
        return self._config

    @property
    def cache(self) -> BlueprintCache:
        """The blueprint cache, keyed by capability profile and plan shape."""
        return self._cache

    def train(
        self,
        train_frame: pd.DataFrame,
        *,
        target: str,
        model: str,
        target_profile: TargetProfile,
        model_params: dict[str, Any] | None = None,
        established: tuple[Any, Any] | None = None,
    ) -> TrainingResult:
        """Fit one model on ``train_frame`` and nothing else.

        Args:
            train_frame: Training rows, features and target together. Read, never
                modified.
            target: The label column.
            model: A canonical name or an alias, resolved through the registry.
            target_profile: What the task detector concluded. Required rather
                than re-derived, so that every model in a comparison is judged
                against one conclusion about the data.
            model_params: Optional estimator parameters. Passed through
                unchanged; nothing here searches, tunes, or suggests.
            established: An already-computed ``(DatasetProfile, QualityReport)``
                for this exact training frame. They describe the *data*, not the
                model, so every model in a comparison would otherwise recompute
                the same answer -- including the leakage scan, which is the
                expensive one. Supplying them once per run is a speed decision
                and a correctness one: one conclusion about the data, shared.

        Returns:
            A :class:`TrainingResult`.

        Raises:
            TrainingError: If the target is absent, or the frame holds no rows.
            IncompatibleModelError: If the model cannot serve this target.
        """
        if target not in train_frame.columns:
            raise TrainingError(
                f"The target {target!r} is not a column of the training frame. "
                f"Available: {[str(c) for c in list(train_frame.columns)[:20]]}."
            )
        if not len(train_frame):
            raise TrainingError(
                "The training frame has no rows. A model cannot be fitted on "
                "nothing."
            )
        self._require_the_profile_matches_the_column(train_frame[target], target_profile)

        # Resolved against the target *before* anything is built, so an
        # incompatible choice fails here rather than inside a fit several steps
        # later. This is the same check ModelFactory.create performs.
        strategy = ModelFactory.strategy(
            model,
            target=target_profile,
            config=self._config,
            registry=self._registry,
        )
        capabilities = strategy.capabilities

        labels = train_frame[target]
        plan, preprocessor = self._prepare(
            train_frame, target, capabilities.preprocessing_profile(), established
        )
        features = train_frame.drop(columns=[target])

        started = time.perf_counter()
        matrix = preprocessor.fit_transform(features)
        y, encoding = self._encode(labels, target_profile)

        estimator = strategy.build(**(model_params or {}))
        estimator.fit(matrix, y)
        elapsed = time.perf_counter() - started

        return TrainingResult(
            model_name=strategy.name,
            task_type=target_profile.task_type,
            estimator=estimator,
            preprocessor=preprocessor,
            target_encoding=encoding,
            preprocessing_profile=capabilities.preprocessing_profile(),
            plan_fingerprint=plan.fingerprint,
            feature_names=tuple(str(n) for n in preprocessor.get_feature_names_out()),
            lineage=preprocessor.lineage(),
            train_row_count=int(len(train_frame)),
            feature_count=int(features.shape[1]),
            transformed_feature_count=int(preprocessor.n_features_out),
            model_parameters=dict(estimator.get_params()),
            random_state=self._seed_in_force(estimator),
            supports_predict_proba=bool(capabilities.supports_predict_proba),
            review_features=tuple(str(c) for c in plan.review_features),
            excluded_features=tuple(str(c) for c in plan.excluded_features),
            fit_seconds=float(elapsed),
        )

    def evaluate(
        self, result: TrainingResult, evaluation_frame: pd.DataFrame, *, target: str
    ) -> Any:
        """Measure a fitted model against held-out rows.

        Args:
            result: What :meth:`train` returned.
            evaluation_frame: Rows the model has not seen. Read, never modified,
                and never fitted on.
            target: The label column, which must be present here too.

        Returns:
            An :class:`~aidatasetkit.evaluation.types.EvaluationReport`.

        Raises:
            TrainingError: If the target column is missing, or the frame is empty.
            SchemaError: If the columns do not match what training saw.
        """
        from aidatasetkit.evaluation import evaluate_classification, evaluate_regression

        if target not in evaluation_frame.columns:
            raise TrainingError(
                f"The target {target!r} is not a column of the evaluation frame. "
                "The rows a model is judged on have to carry the answer."
            )
        if not len(evaluation_frame):
            raise TrainingError(
                "The evaluation frame has no rows. A model cannot be judged on "
                "nothing."
            )

        features = evaluation_frame.drop(columns=[target])
        truth = evaluation_frame[target]
        # transform, never fit_transform. The distinction is the whole contract.
        matrix = result.preprocessor.transform(features)
        predictions = np.asarray(result.estimator.predict(matrix))

        if result.task_type is TaskType.REGRESSION:
            return evaluate_regression(np.asarray(truth, dtype="float64"), predictions)

        encoder = result.target_encoding
        y_true = encoder.transform(truth)
        # Asked of the declared capability first, not of the object. A model that
        # records supports_predict_proba=False is not called, even if the fitted
        # estimator happens to expose the method -- the declaration is the
        # contract, and calling past it would make the metric's stated reason
        # ("its capabilities record supports_predict_proba=False") a lie.
        probabilities = None
        if result.supports_predict_proba and hasattr(result.estimator, "predict_proba"):
            probabilities = np.asarray(result.estimator.predict_proba(matrix))
        return evaluate_classification(
            y_true,
            predictions,
            probabilities=probabilities,
            encoding=encoder,
            class_count=len(encoder.encoding.classes),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def establish(self, train_frame: pd.DataFrame, target: str) -> tuple[Any, Any]:
        """Profile and inspect a training frame once, for every model to share.

        These answers are about the data. Recomputing them per model gave the
        same result each time and paid for the leakage scan once per model.
        """
        dataset_profile = DataProfiler(self._config).profile(train_frame)
        quality = DataQualityInspector(self._config).inspect(
            train_frame, profile=dataset_profile, target=target
        )
        return dataset_profile, quality

    def _prepare(
        self,
        train_frame: pd.DataFrame,
        target: str,
        profile: PreprocessingProfile,
        established: tuple[Any, Any] | None = None,
    ):
        """Plan and build a preprocessor for one capability profile.

        The **whole training frame** is profiled, inspected and planned, with the
        target named -- not a copy with the target already removed. An earlier
        version dropped it first and passed ``target=None``, which silently
        switched off every target-dependent check the library has, leakage
        detection among them. A feature exactly equal to the target would have
        been trained on without a word, in the one layer whose entire purpose is
        to notice that.

        The planner excludes the target from the features itself, so naming it is
        the only thing needed to keep those checks alive.

        Every call builds *fresh, unfitted* transformers. The cache holds pipeline
        shapes so two capability-identical models can be shown to resolve to one
        blueprint; it holds nothing learned, so they cannot share a median.
        """
        dataset_profile, quality = (
            established
            if established is not None
            else self.establish(train_frame, target)
        )
        plan = PreprocessingPlanner(self._preprocessing).plan(
            train_frame, dataset_profile, profile, quality=quality, target=target
        )
        features = train_frame.drop(columns=[target])
        builder = PreprocessorBuilder(self._preprocessing, cache=self._cache)
        return plan, builder.build(plan, features)

    def _require_the_profile_matches_the_column(
        self, labels: pd.Series, target_profile: TargetProfile
    ) -> None:
        """Check the caller's profile against the values it claims to describe.

        ``train`` takes the :class:`TargetProfile` from its caller so that every
        model in a comparison is judged against one conclusion about the data.
        That is right, and it opened a hole: a profile saying ``classification``
        handed alongside a column of quantities would reach
        :class:`TargetLabelEncoder`, whose own value-level guard is skipped
        precisely *because* a profile was supplied. A caller could turn prices
        into class indices through this door -- the exact corruption the shared
        target authority exists to prevent.

        So the authority is consulted here too. It is the same function the task
        detector uses, so the two cannot disagree.
        """
        from aidatasetkit.core.schema import target_holds_quantities

        if target_profile.task_type is TaskType.REGRESSION:
            return
        # Threaded the configuration through. Called with none, this defaulted to
        # KitConfig() and read the *default* class limit, while whoever resolved
        # the task read theirs -- so a session with a non-default
        # task_detection_max_classes could resolve a target one way and have the
        # trainer refuse it on the other.
        if target_holds_quantities(labels, self._config):
            raise TrainingError(
                "The supplied TargetProfile says this is a "
                f"{target_profile.task_type.value} target, but the column holds "
                "quantities. Training on it would encode every measurement as a "
                "class index and destroy the thing being predicted. Detect the "
                "task from this column rather than asserting it, or name the "
                "label column if you meant a different one."
            )

    def _encode(self, labels: pd.Series, target_profile: TargetProfile):
        """Encode a classification target; pass a regression target through.

        The regression branch does nothing at all, deliberately. The shared target
        authority refuses to label encode a quantity, and the correct number of
        transformations to apply to one is none.
        """
        if target_profile.task_type is TaskType.REGRESSION:
            return np.asarray(labels, dtype="float64"), None

        encoder = TargetLabelEncoder(
            positive_label=(
                target_profile.positive_label
                if target_profile.positive_label_resolved
                else None
            )
        )
        return encoder.fit_transform(labels, target_profile), encoder

    @staticmethod
    def _seed_in_force(estimator: Any) -> int | None:
        """The seed the fitted estimator actually carries.

        Read back from the estimator rather than from the strategy's published
        defaults. A caller who passed ``random_state=7`` through ``model_params``
        would otherwise see the default recorded here while
        ``model_parameters`` said 7 -- one record contradicting the other, and
        the wrong one is the field a reader would trust to repeat the run.
        """
        seed = estimator.get_params().get("random_state")
        return None if seed is None else int(seed)
