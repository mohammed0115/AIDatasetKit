"""Clustering by merging, either over the rows or over a summary of them.

Two models that both build a tree, and that differ on the one capability this
release added a field for.

:class:`AgglomerativeClusteringStrategy` merges the rows themselves, bottom-up,
and keeps ``children_`` -- the merge order. When it finishes it has a labelling
of the rows it saw and no mechanism for a row it did not: there is no centre, no
boundary, no rule. It exposes no ``predict``, and that is not an omission in
scikit-learn, it is what the algorithm is.

:class:`BirchStrategy` first compresses the data into a tree of subcluster
summaries and then clusters those. Because the summaries persist as
``subcluster_centers_``, a new row can be routed down the tree, so Birch *can*
assign out of sample. It is the interesting row in the capability table: a
hierarchical method that deploys, because it kept a summary rather than the rows.

Neither accepts a missing value, both need scaling, and they split on sparse
input -- Birch accepts a ``csr_matrix``, Agglomerative raises. All four facts
were measured, and the sparse split is the one that would be easiest to get
wrong by reasoning from the family name.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.cluster import AgglomerativeClustering, Birch

from aidatasetkit.core.types import Backend, Fittable, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["AgglomerativeClusteringStrategy", "BirchStrategy"]


@register_model(aliases=("agglomerative", "hierarchical"))
class AgglomerativeClusteringStrategy(ModelStrategy):
    """Bottom-up merging of the rows themselves, under Ward linkage.

    Describes one dataset and cannot be deployed: it has no ``predict`` and no
    rule that would give one.
    """

    name: ClassVar[str] = "agglomerative_clustering"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLUSTERING,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: ARI 0.637 unscaled against 1.000 standardised. Ward linkage
        # merges on a variance criterion, which reads the unit a column was
        # recorded in exactly as a Euclidean distance does.
        requires_scaling=True,
        # Verified: raises TypeError("Sparse data was passed for X, but dense
        # data is required"). The linkage computation needs a dense array.
        supports_sparse_input=False,
        # Verified: raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        # Verified: no `predict` attribute. There is no centre and no boundary,
        # so there is nothing to assign a new row against.
        supports_out_of_sample_assignment=False,
        # `children_` is the full merge order, which a dendrogram can be drawn
        # from. Informative about structure, not readable per feature.
        interpretability_level=Interpretability.MEDIUM,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        All three restate scikit-learn's. ``n_clusters=2`` is written out for
        the same reason KMeans writes out its ``8``: it decides what question is
        being asked, it was not chosen by looking at any data, and two is very
        unlikely to be the number you want.

        ``linkage="ward"`` and ``metric="euclidean"`` are pinned together
        because they are not independent -- Ward is defined only for a Euclidean
        metric, and scikit-learn raises if they are separated. Recording the
        pair keeps a future default change from silently altering every fit.

        ``random_state`` is deliberately absent: the constructor does not accept
        one, and the merge order is deterministic.
        """
        return {"n_clusters": 2, "linkage": "ward", "metric": "euclidean"}

    def build(self, **params: Any) -> Fittable:
        """Construct a new, unfitted :class:`~sklearn.cluster.AgglomerativeClustering`."""
        return self._construct(
            AgglomerativeClustering, self.resolve_params(**params), self.name
        )


@register_model(aliases=("birch",))
class BirchStrategy(ModelStrategy):
    """Incremental tree of subcluster summaries, then a clustering of those.

    The hierarchical method that can be deployed: the summaries survive fitting,
    so a new row can be routed down the tree and assigned.
    """

    name: ClassVar[str] = "birch_clustering"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLUSTERING,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: ARI 0.637 unscaled against 1.000 standardised. `threshold`
        # is an absolute radius, so it has the same unit problem `eps` has.
        requires_scaling=True,
        # Verified: fits on a csr_matrix and assigns from one.
        supports_sparse_input=True,
        # Verified: raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        # Verified: predict() on held-out rows returns labels, and on the
        # training rows reproduces labels_ exactly.
        supports_out_of_sample_assignment=True,
        # `subcluster_centers_` is a summary in feature space, but there are far
        # more subclusters than clusters and the mapping between them is the CF
        # tree. Informative, not directly readable -- so MEDIUM rather than the
        # HIGH the centroid models earn.
        interpretability_level=Interpretability.MEDIUM,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        All three restate scikit-learn's. ``threshold=0.5`` is written out
        because, like DBSCAN's ``eps``, it is an absolute radius in the scaled
        feature space and the result cannot be reasoned about without it.

        ``random_state`` is deliberately absent: the constructor does not accept
        one, and the tree is built deterministically from the row order.
        """
        return {"n_clusters": 3, "threshold": 0.5, "branching_factor": 50}

    def build(self, **params: Any) -> Fittable:
        """Construct a new, unfitted :class:`~sklearn.cluster.Birch`."""
        return self._construct(Birch, self.resolve_params(**params), self.name)
