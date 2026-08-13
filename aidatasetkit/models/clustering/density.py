"""Clustering by density, where some rows belong to nothing.

These two differ from every other model in this library in a way that is easy to
miss and expensive to miss: **they do not partition the data**. A row they judge
to be in no dense region is labelled ``-1``, and ``-1`` is not a cluster. It is
the algorithm declining to assign. Any code that counts distinct labels, or
computes a metric over them, has to hold that apart or it will report a noise
bucket as a group of customers.

That is handled where it can be measured -- in the clustering result and the
metric policy, which record the noise count and exclude noise rows before
scoring -- rather than declared here, because how much noise appears is a fact
about a *run*, not a property of the algorithm.

.. rubric:: Scaling is not a refinement here, it is the difference between
   working and not working

``eps`` is an absolute distance. It is ``0.5`` by default, and ``0.5`` means
nothing until you know what the columns are measured in. On the same three
separated blobs the other modules use, with one feature in a unit ``1e5``
larger:

===========  ==============  ==============
Model        ARI unscaled    ARI scaled
===========  ==============  ==============
DBSCAN       ``0.000``       ``1.000``
OPTICS       ``0.051``       ``0.061``
===========  ==============  ==============

DBSCAN unscaled scored ARI ``0.000`` -- it recovered nothing at all, because
every point was further apart than ``eps`` along the large-unit axis. This is the
starkest ``requires_scaling`` measurement in the whole catalog, and it is why
these models sit behind the same scaler as the linear ones.

.. rubric:: OPTICS is not a drop-in DBSCAN, and its default is not good

The second row above is the honest one. OPTICS at its published defaults did not
recover the structure scaled *or* unscaled. Measured on three separated blobs
drawn with ``default_rng(0)``, and again with thirty uniform outliers from
``default_rng(99)`` added:

=========================  =========================  =========================
Frame                      OPTICS                     DBSCAN
=========================  =========================  =========================
300 clean rows             13 clusters, 186 noise     3 clusters, 17 noise
330 rows, 30 outliers      13 clusters, 215 noise     3 clusters, 47 noise
=========================  =========================  =========================

OPTICS called well over half the rows noise on data DBSCAN partitioned exactly.
It builds a reachability ordering with ``min_samples=5`` and then extracts
clusters with ``xi``, and that extraction is sensitive in a way the default does
not advertise.

Nothing here corrects it, because correcting it would mean choosing ``xi`` or
``min_samples`` by looking at the data. It is recorded instead, in the place an
analyst reads before choosing a model. OPTICS earns its place by not requiring a
single global ``eps`` -- it can find clusters of differing density, which DBSCAN
cannot -- and that strength is worth the defaults being close to useless.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.cluster import DBSCAN, OPTICS

from aidatasetkit.core.types import Backend, Fittable, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["DBSCANStrategy", "OPTICSStrategy"]


@register_model(aliases=("dbscan",))
class DBSCANStrategy(ModelStrategy):
    """Density-based clustering with one global radius, and explicit noise.

    Finds however many clusters the density supports -- ``n_clusters`` is not a
    parameter -- and labels everything sparse ``-1``.
    """

    name: ClassVar[str] = "dbscan_clustering"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLUSTERING,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified, and more strongly than for any other model in the catalog:
        # ARI 0.000 unscaled against 1.000 standardised. `eps` is an absolute
        # distance, so an unscaled column decides the neighbourhood by itself.
        requires_scaling=True,
        # Verified: fits on a csr_matrix.
        supports_sparse_input=True,
        # Verified: raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        # Verified: no `predict` attribute at all. A fitted DBSCAN holds
        # `core_sample_indices_` and `components_`, which describe the rows it
        # saw; assigning a new row would require a rule the algorithm does not
        # define.
        supports_out_of_sample_assignment=False,
        # `components_` is the set of core points -- the fitted model is a subset
        # of the training rows, not a summary of them. That yields evidence
        # about the clusters without a readable decision function, which is what
        # MEDIUM means here.
        interpretability_level=Interpretability.MEDIUM,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        Both restate scikit-learn's. ``eps=0.5`` is written out because it is
        the model: it is an absolute distance in the *scaled* feature space this
        library hands over, and an analyst who does not know its value cannot
        reason about the result at all.

        ``random_state`` is deliberately absent: the constructor does not accept
        one, and the algorithm is deterministic given its inputs.
        """
        return {"eps": 0.5, "min_samples": 5}

    def build(self, **params: Any) -> Fittable:
        """Construct a new, unfitted :class:`~sklearn.cluster.DBSCAN`."""
        return self._construct(DBSCAN, self.resolve_params(**params), self.name)


@register_model(aliases=("optics",))
class OPTICSStrategy(ModelStrategy):
    """Density clustering without a single global radius.

    Finds clusters of differing density, which DBSCAN cannot. Read the module
    docstring before using it: at these defaults it called 186 of 300 rows noise
    on data DBSCAN split into three clusters with 17.
    """

    name: ClassVar[str] = "optics_clustering"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLUSTERING,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: the partition changes under standardisation. Note that the
        # measured ARI is poor either way (0.051 against 0.061) -- scaling is
        # necessary here and, at these defaults, not sufficient.
        requires_scaling=True,
        # Verified: raises TypeError("scipy distance metrics do not support
        # sparse matrices"). The default metric="minkowski" routes through
        # scipy, which has no sparse path.
        supports_sparse_input=False,
        # Verified: raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        # Verified: no `predict` attribute.
        supports_out_of_sample_assignment=False,
        # A fitted OPTICS holds `reachability_`, an ordering over the training
        # rows. It is genuinely informative to plot and tells you nothing per
        # feature, and there is no decision function to read.
        interpretability_level=Interpretability.LOW,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``min_samples=5`` restates scikit-learn's. ``xi`` and ``max_eps`` are
        left alone too, deliberately: the module docstring records that these
        defaults performed badly on ordinary separated blobs, and the correct
        response to that is to report it, not to pick better numbers by looking
        at data.

        ``random_state`` is deliberately absent: the constructor does not accept
        one.
        """
        return {"min_samples": 5}

    def build(self, **params: Any) -> Fittable:
        """Construct a new, unfitted :class:`~sklearn.cluster.OPTICS`."""
        return self._construct(OPTICS, self.resolve_params(**params), self.name)
