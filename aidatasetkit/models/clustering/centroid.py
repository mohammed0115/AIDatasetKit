"""Clustering by distance to a moving centre.

Both models here keep ``cluster_centers_`` -- one readable point per cluster, in
the feature space -- which is what lets them assign a row they never saw and what
makes them the two most interpretable clusterers in the catalog.

.. rubric:: The capabilities, measured identically for both

``requires_scaling=True``
    Three separated blobs with one feature multiplied by ``1e5`` -- an income
    beside a ratio -- scored ARI ``0.656`` unscaled against ``1.000``
    standardised for KMeans, and ``0.631`` against ``1.000`` for MiniBatch. A
    Euclidean distance is a sum over features, so the largest unit wins.
``supports_sparse_input=True``
    Fits on a ``scipy.sparse.csr_matrix`` and assigns from one.
``handles_missing_values=False``
    Raises ``ValueError("Input X contains NaN")``. A centroid is a mean, and a
    mean has no answer for an absent coordinate.
``supports_out_of_sample_assignment=True``
    Verified by fitting and then calling ``predict`` on held-out rows, and by
    checking that ``predict`` on the training rows reproduces ``labels_``
    exactly.

.. rubric:: ``n_clusters`` is the caller's decision and nothing here helps

``k`` is not a tuning knob, it is the question being asked, and a library that
picked it by running an elbow or a silhouette sweep would be choosing the answer
on the analyst's behalf. So the scikit-learn default of ``8`` is restated and
left alone. **It is an inherited default, not a recommendation, and it is almost
certainly wrong for any particular dataset.** Pass ``n_clusters`` deliberately.

.. rubric:: ``n_init=10`` deviates from the scikit-learn default, on evidence

scikit-learn 1.9 defaults ``n_init`` to ``"auto"``, which resolves to a single
initialisation for k-means++. That single restart makes the *quality* of the
result a lottery on the seed. Measured on three overlapping Gaussians, 300 rows,
eight seeds:

=================  ========  ====================  ================
Model              n_init    distinct partitions   inertia spread
=================  ========  ====================  ================
KMeans             ``auto``  2 of 8                ``0.0970``
KMeans             ``10``    4 of 8                ``0.0519``
MiniBatchKMeans    ``auto``  8 of 8                ``130.77``
MiniBatchKMeans    ``10``    8 of 8                ``17.29``
=================  ========  ====================  ================

Read the second column before the third, because it is the one that surprises.
More restarts produced *more* distinct partitions, not fewer. Ten restarts do
**not** make clustering seed-independent, and nothing does -- what they do is
narrow how bad the unlucky seed is allowed to be. MiniBatchKMeans is the case
that matters: an 87% cut in the spread of the objective the algorithm is
minimising. That is why the deviation is here, and stating it in the catalog is
the point, since ``default_params`` is what an analyst reads to find out what
they are actually getting.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.cluster import KMeans, MiniBatchKMeans

from aidatasetkit.core.types import Backend, Fittable, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["KMeansStrategy", "MiniBatchKMeansStrategy"]


@register_model(aliases=("kmeans",))
class KMeansStrategy(ModelStrategy):
    """Lloyd's algorithm: ``k`` centroids, alternating assignment and update.

    See the module docstring for the measurements behind every capability.
    """

    name: ClassVar[str] = "kmeans_clustering"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLUSTERING,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: ARI 0.656 unscaled against 1.000 standardised, on three
        # separated blobs with one feature recorded in a unit 1e5 larger.
        requires_scaling=True,
        # Verified: fits on a csr_matrix and assigns from one.
        supports_sparse_input=True,
        # Verified: raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        # Verified: predict() on held-out rows returns labels, and on the
        # training rows reproduces labels_ exactly.
        supports_out_of_sample_assignment=True,
        # A fitted model is `cluster_centers_`: one point per cluster, in the
        # units of the features it was given. That is directly readable in the
        # sense this field means -- print it and you can say what each cluster
        # is. The caveat is that requires_scaling=True makes those coordinates
        # per standard deviation, exactly as for the linear models.
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``n_clusters=8`` restates the scikit-learn default and is written out
        because it is the one parameter that decides what question is being
        asked. Nothing here looked at the data to choose it.

        ``n_init=10`` deviates from scikit-learn's ``"auto"``; the module
        docstring carries the measurement.
        """
        return {
            "n_clusters": 8,
            "n_init": 10,
            "random_state": self.config.random_state,
        }

    def build(self, **params: Any) -> Fittable:
        """Construct a new, unfitted :class:`~sklearn.cluster.KMeans`."""
        return self._construct(KMeans, self.resolve_params(**params), self.name)


@register_model(aliases=("minibatch_kmeans",))
class MiniBatchKMeansStrategy(ModelStrategy):
    """KMeans on random mini-batches: far cheaper, and measurably less stable.

    Eight seeds produced eight different partitions on overlapping data where
    full KMeans produced four, and the spread in the objective was ``17.29``
    against ``0.0519``. The speed is real and so is the cost; both belong in the
    record rather than only the first one.
    """

    name: ClassVar[str] = "minibatch_kmeans_clustering"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLUSTERING,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: ARI 0.631 unscaled against 1.000 standardised.
        requires_scaling=True,
        supports_sparse_input=True,
        handles_missing_values=False,
        supports_out_of_sample_assignment=True,
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        The same two deliberate values as :class:`KMeansStrategy`, for the same
        reasons. ``batch_size`` is left at the scikit-learn default: choosing it
        trades speed against stability, and that trade belongs to whoever knows
        how large their data is.
        """
        return {
            "n_clusters": 8,
            "n_init": 10,
            "random_state": self.config.random_state,
        }

    def build(self, **params: Any) -> Fittable:
        """Construct a new, unfitted :class:`~sklearn.cluster.MiniBatchKMeans`."""
        return self._construct(
            MiniBatchKMeans, self.resolve_params(**params), self.name
        )
