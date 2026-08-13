"""Running one clusterer, in the order the layers require and with no target.

Clustering is not supervised learning with the labels left out, and this module
is separate from :mod:`aidatasetkit.training.trainer` because pretending
otherwise would require lying to three layers at once. There is no target column,
so nothing is encoded and no leakage check about a target can run; there is no
ground truth, so no metric here says whether an answer is right; and three of the
six built-in clusterers cannot label a row they were not fitted on, so the split
that makes supervised evaluation honest has nothing to attach to.

.. rubric:: The metrics are in-sample, and that is not hidden

This is the single most important sentence in the module. The partition is
scored on **the same rows it was fitted on**. In supervised training that would
be the leakage this library exists to prevent, and the reason it is not the same
failure here is narrow and worth stating exactly:

- there is no target, so a score cannot be inflated by having seen the answer --
  there is no answer;
- the three metrics are *internal*, measuring the geometry of a partition rather
  than agreement with anything held back;
- and for ``dbscan``, ``optics`` and ``agglomerative`` there is no out-of-sample
  assignment at all, so a held-out set could not be scored even in principle.

What it does mean is that these numbers describe how tidily this model divided
**this** data, and they do not predict how it will divide the next batch. A high
silhouette is not evidence that the clusters will reproduce. For the three models
that *can* assign out of sample, :meth:`ClusteringResult.assign` exists so a
caller can check that for themselves on rows they held back deliberately.

.. rubric:: What is still enforced

Everything that does not depend on a target. The frame is profiled and inspected,
the plan comes from the model's capability profile through the same planner the
supervised path uses, the preprocessor is fitted once and kept so new rows can be
transformed rather than re-fitted, and nothing is dropped or imputed that the
plan did not say to drop or impute.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import TrainingError
from aidatasetkit.core.types import Fittable, PreprocessingProfile, TaskType
from aidatasetkit.evaluation import NOISE_LABEL, evaluate_clustering
from aidatasetkit.models import ModelFactory, ModelRegistry
from aidatasetkit.preprocessing import (
    BlueprintCache,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

__all__ = ["ClusteringResult", "ClusteringRunner"]


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """One fitted clusterer, the partition it found, and how it got there.

    Attributes:
        model_name: The canonical name, never the alias the caller typed.
        estimator: The fitted estimator. Present so a caller can inspect
            centroids or assign new rows; never serialised.
        preprocessor: The fitted preprocessor, likewise.
        labels: One cluster label per input row, in input order. Read-only, and
            ``-1`` means the algorithm declined to assign that row.
        preprocessing_profile: The capability triple that chose the pipeline.
        plan_fingerprint: Identity of the preprocessing plan.
        feature_names: The transformed feature names, from S4.
        lineage: Each input column and what it became, from S4.
        row_count: How many rows were clustered.
        feature_count: Raw feature columns offered.
        transformed_feature_count: Columns the model actually received.
        evaluation: The three internal metrics, each with a number or a reason.
            Computed on the fitted rows -- see the module docstring.
        model_parameters: The estimator's resolved parameters.
        random_state: The seed the fitted estimator carries, or ``None`` for the
            four clusterers whose constructors take none.
        supports_out_of_sample_assignment: Whether :meth:`assign` will work.
        review_features: Columns the plan held back for a human decision.
        excluded_features: Columns the plan dropped outright.
        fit_seconds: Observational. Never part of a ranking or a fingerprint.
    """

    model_name: str
    estimator: Fittable
    preprocessor: Any
    labels: np.ndarray
    preprocessing_profile: PreprocessingProfile
    plan_fingerprint: str
    feature_names: tuple[str, ...]
    lineage: dict[Any, tuple[str, ...]]
    row_count: int
    feature_count: int
    transformed_feature_count: int
    evaluation: Any
    model_parameters: dict[str, Any] = field(default_factory=dict)
    random_state: int | None = None
    supports_out_of_sample_assignment: bool = False
    review_features: tuple[str, ...] = ()
    excluded_features: tuple[str, ...] = ()
    fit_seconds: float | None = None

    #: Clustering is the task family, stated so callers can branch on one field
    #: across every result type this library produces.
    task_type: TaskType = TaskType.CLUSTERING

    @property
    def noise_count(self) -> int:
        """How many rows the algorithm declined to assign to any cluster.

        Never zero for a density method on real data, and always zero for the
        four that partition. Reported separately from the cluster count because
        ``-1`` is not a cluster.
        """
        return int(np.count_nonzero(self.labels == NOISE_LABEL))

    @property
    def n_clusters(self) -> int:
        """How many clusters were found, not counting noise.

        This is the *found* count, which for a density method is not a parameter
        anybody set, and for the others may still be lower than the ``n_clusters``
        requested -- an empty cluster is not reported as one.
        """
        return int(np.unique(self.labels[self.labels != NOISE_LABEL]).size)

    @property
    def cluster_sizes(self) -> dict[int, int]:
        """How many rows fell in each cluster, noise excluded, largest first.

        The distribution matters as much as the count: eight clusters where one
        holds 97% of the rows is a different result from eight even ones, and a
        bare ``n_clusters=8`` reports them identically.
        """
        assigned = self.labels[self.labels != NOISE_LABEL]
        found, counts = np.unique(assigned, return_counts=True)
        pairs = sorted(
            ((int(label), int(count)) for label, count in zip(found, counts)),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return dict(pairs)

    def assign(self, X: pd.DataFrame) -> np.ndarray:
        """Assign rows the model never saw to the clusters it found.

        The preprocessor is *applied*, never refitted, so a new row is placed in
        the same space the clusters were found in.

        Raises:
            TrainingError: If this algorithm cannot assign out of sample. That is
                a property of the algorithm rather than of this implementation:
                ``DBSCAN``, ``OPTICS`` and ``AgglomerativeClustering`` produce a
                labelling of the rows they were fitted on and define no rule for
                any other row. Refitting on the new rows would produce a
                *different partition*, not an assignment into this one, so it is
                refused rather than approximated.
        """
        if not self.supports_out_of_sample_assignment:
            raise TrainingError(
                f"{self.model_name!r} cannot assign rows it was not fitted on. "
                "The algorithm produces a labelling of the fitted rows and "
                "defines no rule for a new one, so there is nothing to apply. "
                "Refitting on the new rows would give a different partition "
                "rather than an assignment into this one. Choose a model whose "
                "capabilities report supports_out_of_sample_assignment -- "
                "kmeans, minibatch_kmeans or birch -- if new rows must be "
                "assigned."
            )
        return np.asarray(self.estimator.predict(self.preprocessor.transform(X)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the run.

        No estimator, no preprocessor, and **no per-row labels**. A label per row
        is data about a record, and the row-level output of a model belongs to
        whoever ran it, not in an artifact that gets committed and shared. The
        aggregate shape of the partition is here instead, which is what a reader
        needs to judge it.
        """
        from aidatasetkit.core.types import jsonable

        return {
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "preprocessing_profile": self.preprocessing_profile.key,
            "plan_fingerprint": self.plan_fingerprint,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "transformed_feature_count": self.transformed_feature_count,
            "feature_names": list(self.feature_names),
            "lineage": {
                str(source): list(produced) for source, produced in self.lineage.items()
            },
            "n_clusters": self.n_clusters,
            "noise_count": self.noise_count,
            "cluster_sizes": {str(k): v for k, v in self.cluster_sizes.items()},
            "model_parameters": {
                str(key): jsonable(value)
                for key, value in sorted(self.model_parameters.items())
            },
            "random_state": self.random_state,
            "supports_out_of_sample_assignment": (
                self.supports_out_of_sample_assignment
            ),
            "review_features": list(self.review_features),
            "excluded_features": list(self.excluded_features),
            "evaluation": self.evaluation.to_dict(),
        }


class ClusteringRunner:
    """Fits one clusterer on a frame with no target column.

    Args:
        config: Shared thresholds and the seed.
        preprocessing_config: Strategies for the preparation itself.
        cache: Blueprint cache. Only pipeline *shapes* are cached; nothing
            fitted is.
        registry: Model registry to resolve against. Defaults to the built-in one.

    Example:
        >>> from aidatasetkit.training import ClusteringRunner
        >>> runner = ClusteringRunner()
        >>> # result = runner.run(frame, model="kmeans", model_params={"n_clusters": 3})
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
            preprocessing_config
            if preprocessing_config is not None
            else PreprocessingConfig()
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

    def run(
        self,
        frame: pd.DataFrame,
        *,
        model: str,
        model_params: dict[str, Any] | None = None,
        established: tuple[Any, Any] | None = None,
        silhouette_row_limit: int | None = None,
    ) -> ClusteringResult:
        """Cluster ``frame`` with one model and measure the partition.

        Args:
            frame: The rows to cluster. Every column is a feature; there is no
                target to name and none is removed. Read, never modified.
            model: A canonical name or an alias, resolved through the registry
                and required to be a clustering model.
            model_params: Optional estimator parameters. Passed through
                unchanged; nothing here searches, tunes, or picks ``k``.
            established: An already-computed ``(DatasetProfile, QualityReport)``
                for this exact frame, so several models can share one conclusion
                about the data.
            silhouette_row_limit: Overrides the point above which the silhouette
                is reported rather than computed.

        Returns:
            A :class:`ClusteringResult`.

        Raises:
            TrainingError: If the frame holds no rows.
            IncompatibleModelError: If the named model is not a clusterer.
        """
        if not len(frame):
            raise TrainingError(
                "The frame has no rows. There is nothing to cluster."
            )

        # Resolved with task= rather than target=, because there is no target to
        # resolve against. This is what makes naming a classifier here fail now,
        # with a message about the task family, instead of failing inside `fit`
        # with a message about a missing argument.
        strategy = ModelFactory.strategy(
            model,
            task=TaskType.CLUSTERING,
            config=self._config,
            registry=self._registry,
        )
        # The explicit no-target check approved as conflict D. `task=` above
        # already narrowed the registry lookup; this asserts the capability
        # contract itself, so the two cannot drift apart.
        strategy.validate_for(None)
        capabilities = strategy.capabilities

        plan, preprocessor = self._prepare(
            frame, capabilities.preprocessing_profile(), established
        )

        started = time.perf_counter()
        matrix = preprocessor.fit_transform(frame)
        estimator = strategy.build(**(model_params or {}))
        labels = self._fit_labels(estimator, matrix)
        elapsed = time.perf_counter() - started

        limit = (
            {}
            if silhouette_row_limit is None
            else {"silhouette_row_limit": silhouette_row_limit}
        )
        evaluation = evaluate_clustering(matrix, labels, **limit)

        return ClusteringResult(
            model_name=strategy.name,
            estimator=estimator,
            preprocessor=preprocessor,
            labels=labels,
            preprocessing_profile=capabilities.preprocessing_profile(),
            plan_fingerprint=plan.fingerprint,
            feature_names=tuple(str(n) for n in preprocessor.get_feature_names_out()),
            lineage=preprocessor.lineage(),
            row_count=int(len(frame)),
            feature_count=int(frame.shape[1]),
            transformed_feature_count=int(preprocessor.n_features_out),
            evaluation=evaluation,
            model_parameters=dict(estimator.get_params()),
            random_state=self._seed_in_force(estimator),
            supports_out_of_sample_assignment=bool(
                capabilities.supports_out_of_sample_assignment
            ),
            review_features=tuple(str(c) for c in plan.review_features),
            excluded_features=tuple(str(c) for c in plan.excluded_features),
            fit_seconds=float(elapsed),
        )

    def establish(self, frame: pd.DataFrame) -> tuple[Any, Any]:
        """Profile and inspect a frame once, for several models to share.

        Called with no target, so the target-dependent checks -- leakage above
        all -- do not run. They cannot: there is no target for a feature to leak.
        Every check that describes the data alone still does.
        """
        dataset_profile = DataProfiler(self._config).profile(frame)
        quality = DataQualityInspector(self._config).inspect(
            frame, profile=dataset_profile
        )
        return dataset_profile, quality

    def _prepare(
        self,
        frame: pd.DataFrame,
        profile: PreprocessingProfile,
        established: tuple[Any, Any] | None = None,
    ):
        """Plan and build a preprocessor for one capability profile, with no target.

        ``target=None`` is passed to the planner explicitly rather than omitted.
        In the supervised path an absent target silently disables leakage
        detection, which is why the trainer was fixed to always name it; here
        there is genuinely no target, and passing ``None`` on purpose is the
        difference between "no target exists" and "somebody forgot".
        """
        dataset_profile, quality = (
            established if established is not None else self.establish(frame)
        )
        plan = PreprocessingPlanner(self._preprocessing).plan(
            frame, dataset_profile, profile, quality=quality, target=None
        )
        builder = PreprocessorBuilder(self._preprocessing, cache=self._cache)
        return plan, builder.build(plan, frame)

    @staticmethod
    def _fit_labels(estimator: Any, matrix: Any) -> np.ndarray:
        """Fit and return the partition, as a read-only integer array.

        ``fit_predict`` is preferred where it exists because for several of these
        algorithms it is the only supported path -- and for ``MiniBatchKMeans``
        it is not merely a shorthand for ``fit`` then ``predict``. The array is
        frozen because a caller mutating it would silently invalidate
        :attr:`ClusteringResult.n_clusters`, ``noise_count`` and every metric
        already computed from it.
        """
        if hasattr(estimator, "fit_predict"):
            labels = estimator.fit_predict(matrix)
        else:  # pragma: no cover - every built-in clusterer has fit_predict
            estimator.fit(matrix)
            labels = estimator.labels_
        frozen = np.array(labels, dtype="int64", copy=True)
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _seed_in_force(estimator: Any) -> int | None:
        """The seed the fitted estimator actually carries.

        Four of the six clusterers take no ``random_state`` at all, so ``None``
        here is an ordinary answer meaning "this algorithm is deterministic
        given its input", not a missing record.
        """
        seed = estimator.get_params().get("random_state")
        return None if seed is None else int(seed)
