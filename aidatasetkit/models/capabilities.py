"""What a model can do and what it needs.

:class:`ModelCapabilities` is a record of *facts about an algorithm*, verified
against the installed library rather than assumed. It is not training
configuration: no epochs, no learning rate, no optimiser, no hyperparameters.
Those belong to the strategy's parameters and, later, to backend-specific
training configuration.

Three of these facts -- and only three -- change how a pipeline is assembled, and
:meth:`ModelCapabilities.preprocessing_profile` extracts them into the
:class:`~aidatasetkit.core.types.PreprocessingProfile` established in S0. That
profile is what preprocessing caches on, so every model declaring the same three
answers shares one preprocessor.

Those three describe **one data contract, not three independent facts**, and the
difference is load-bearing. scikit-learn's tree family accepts a sparse matrix,
accepts ``NaN``, and refuses a sparse matrix containing ``NaN`` -- and a model
declaring ``handles_missing_values`` is never handed imputed data, so the sparse
matrix it would actually receive is precisely the one it rejects. Each field
therefore answers "would this model accept what preprocessing would build for it
given the others", which is why a tree can accept a clean ``csr_matrix`` in
isolation and still, correctly, declare ``supports_sparse_input=False``. A
declaration that were true only in isolation would let the pipeline build
something no model could consume.

Declared capabilities are enforced by contract tests that actually fit the
estimator. Metadata that lies is worse than no metadata, because every later
decision trusts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aidatasetkit.core.exceptions import IncompatibleModelError, ValidationError
from aidatasetkit.core.types import (
    Backend,
    Interpretability,
    PreprocessingProfile,
    TargetProfile,
    TaskType,
)

__all__ = ["ModelCapabilities"]


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Verified facts about one model.

    Attributes:
        task_type: The task family this model serves.
        backend: The library that implements it.
        supports_predict_proba: Whether a fitted estimator exposes and can
            execute ``predict_proba``. Drives whether ROC-AUC can be computed and
            whether a probability column can be produced.
        requires_scaling: Whether the algorithm is sensitive to feature scale,
            either through a distance metric or a scale-dependent penalty. Adds a
            scaler to the numeric branch of the pipeline.
        supports_sparse_input: Whether the estimator accepts the sparse matrix
            *this library would hand it*. When false, one-hot encoding must
            produce a dense array. Read together with ``handles_missing_values``,
            not separately -- see the note below.
        supports_multiclass: Whether more than two classes can be fitted directly.
            Only consulted when ``task_type`` is classification.
        handles_missing_values: Whether the estimator consumes ``NaN`` without
            raising. When true the numeric imputation step may be skipped, so the
            matrix reaching the model still holds its gaps.
        interpretability_level: How readable a fitted model is to a human. Purely
            descriptive metadata for the comparison table -- it never influences
            preprocessing or selection, and it makes no claim to be a measurement.
            ``HIGH`` means the fitted model can be read directly (a coefficient
            per feature, a rule, a class prior); ``MEDIUM`` means it yields
            importances but no directly readable decision function; ``LOW`` means
            neither.
        is_baseline: Whether this model exists to establish a floor rather than to
            compete. Reported separately in comparisons so that an improvement can
            be read against it.
        supports_out_of_sample_assignment: Whether a *fitted* model can assign a
            row it never saw. Meaningful only for clustering, and false by
            default, so every model that predates this field keeps the answer it
            already had.

            A supervised model answers this trivially -- predicting unseen rows
            is what it is for -- so declaring it there would be a field that is
            ``True`` for all eighteen and carries no information. Clustering is
            where it divides: ``KMeans`` keeps centroids and can assign one,
            while ``DBSCAN`` and ``AgglomerativeClustering`` produce a labelling
            *of the fitted rows only* and expose no ``predict`` at all. Verified
            by fitting and then calling, not by reading a class hierarchy.

            This is the difference between a model that can be deployed and a
            model that has described one dataset, which is why it is recorded
            rather than discovered at prediction time.
    """

    task_type: TaskType
    backend: Backend
    supports_predict_proba: bool
    requires_scaling: bool
    supports_sparse_input: bool
    supports_multiclass: bool
    handles_missing_values: bool
    interpretability_level: Interpretability
    is_baseline: bool = False
    supports_out_of_sample_assignment: bool = False

    def __post_init__(self) -> None:
        if (
            self.task_type is not TaskType.CLUSTERING
            and self.supports_out_of_sample_assignment
        ):
            raise ValidationError(
                f"supports_out_of_sample_assignment is meaningless for a "
                f"{self.task_type.value} model; a supervised model assigns unseen "
                "rows by definition, so declaring it would state nothing. It "
                "divides clustering algorithms, and only those."
            )
        if self.task_type is not TaskType.CLASSIFICATION:
            if self.supports_predict_proba:
                raise ValidationError(
                    f"supports_predict_proba is meaningless for a "
                    f"{self.task_type.value} model; class probabilities exist only "
                    "for classification."
                )
            if self.supports_multiclass:
                raise ValidationError(
                    f"supports_multiclass is meaningless for a "
                    f"{self.task_type.value} model; classes exist only for "
                    "classification."
                )

    def preprocessing_profile(self) -> PreprocessingProfile:
        """Extract the capabilities that materially change pipeline assembly.

        Only three answers affect the pipeline, so only three take part in the
        cache identity. Two models agreeing on all three share one preprocessor
        however different they are otherwise.
        """
        return PreprocessingProfile(
            requires_scaling=self.requires_scaling,
            supports_sparse_input=self.supports_sparse_input,
            handles_missing_values=self.handles_missing_values,
        )

    def validate_for(self, target: TargetProfile | None) -> None:
        """Check this model against a resolved target, before any fitting.

        Args:
            target: What the task detector concluded about the target, or
                ``None`` when there is no target at all. ``None`` is a real
                answer, not a missing argument: clustering has no target by
                definition, and the alternative -- a caller inventing a
                placeholder ``TargetProfile`` to get past this check -- would put
                a fabricated task type into the audit record.

        Raises:
            IncompatibleModelError: If the model serves a different task family,
                if a supervised model is offered no target at all, or if the
                target is multiclass and the model is binary-only.
        """
        if target is None:
            if self.task_type is not TaskType.CLUSTERING:
                raise IncompatibleModelError(
                    f"This model serves {self.task_type.value} tasks and was "
                    "offered no target. A supervised model has nothing to learn "
                    "from unlabelled rows. Name the label column, or choose a "
                    "clustering model if the data has no labels."
                )
            return
        if target.task_type is not self.task_type:
            raise IncompatibleModelError(
                f"This model serves {self.task_type.value} tasks, but the target "
                f"was resolved as {target.task_type.value}."
            )
        if (
            self.task_type is TaskType.CLASSIFICATION
            and target.n_classes is not None
            and target.n_classes > 2
            and not self.supports_multiclass
        ):
            raise IncompatibleModelError(
                f"This model supports two classes only, but the target has "
                f"{target.n_classes}."
            )

    def is_compatible_with(self, target: TargetProfile | None) -> bool:
        """Whether this model can be used for ``target``, ``None`` meaning none."""
        try:
            self.validate_for(target)
        except IncompatibleModelError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the capabilities."""
        return {
            "task_type": self.task_type.value,
            "backend": self.backend.value,
            "supports_predict_proba": self.supports_predict_proba,
            "requires_scaling": self.requires_scaling,
            "supports_sparse_input": self.supports_sparse_input,
            "supports_multiclass": self.supports_multiclass,
            "handles_missing_values": self.handles_missing_values,
            "interpretability_level": self.interpretability_level.value,
            "is_baseline": self.is_baseline,
            # Emitted for every model, not only the clustering ones. A field
            # present on some rows and absent on others is a field a reader has
            # to guess the meaning of, and hiding it from the artifact to keep an
            # old fingerprint stable would be arranging the evidence to match a
            # number. The fingerprint moved; CHANGELOG.md records why.
            "supports_out_of_sample_assignment": self.supports_out_of_sample_assignment,
            "preprocessing_profile": self.preprocessing_profile().key,
        }
