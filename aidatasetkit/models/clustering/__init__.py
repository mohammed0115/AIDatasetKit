"""Clustering strategies.

Importing this package registers every clusterer it contains. The strategy
modules are imported explicitly below rather than discovered, so which models
exist never depends on import order or on a filesystem scan.

Clustering is not a third modelling subsystem. These six use the same
:class:`~aidatasetkit.models.base.ModelStrategy`, the same
:class:`~aidatasetkit.models.capabilities.ModelCapabilities`, the same registry
and the same factory as the classifiers and regressors, and they reach S4 through
the same three-answer :class:`~aidatasetkit.core.types.PreprocessingProfile`.
Nothing in the preprocessing layer asks whether a model has a target.

.. rubric:: Every one of the six requires scaling

That is not a family-level assumption, it is six separate measurements that
happened to agree, and the reason they agree is that all six decide membership by
a distance and none of them has a way to know what a column was measured in.
Measured on three separated blobs with one feature recorded in a unit ``1e5``
larger, as adjusted Rand index against the truth:

=========================  ==============  ============
Model                      ARI unscaled    ARI scaled
=========================  ==============  ============
``dbscan``                 ``0.000``       ``1.000``
``minibatch_kmeans``       ``0.631``       ``1.000``
``agglomerative``          ``0.637``       ``1.000``
``birch``                  ``0.637``       ``1.000``
``kmeans``                 ``0.656``       ``1.000``
``optics``                 ``0.051``       ``0.061``
=========================  ==============  ============

DBSCAN recovered *nothing* unscaled. OPTICS recovered nothing either way, which
is a fact about its published defaults and is documented where an analyst will
meet it, in :mod:`aidatasetkit.models.clustering.density`.

.. rubric:: The six occupy two profile keys, and both already existed

=========================================  ========  =======  ===========
Profile                                    Scaling   Sparse   Native NaN
=========================================  ========  =======  ===========
kmeans, minibatch kmeans, dbscan, birch    yes       yes      no
optics, agglomerative                      yes       no       no
=========================================  ========  =======  ===========

The first key is the one ridge, linear regression and k-nearest-neighbours
already share; the second is Gaussian naive Bayes's. So twenty-four built-in
models still need only **five** preprocessors between them. As with regression,
that is not an arrangement made here -- it is what happens when the cache key is
a capability triple rather than a model name.

.. rubric:: Out-of-sample assignment is where they actually divide

The capability this release added, and the only one that splits the six into
groups a family name would not predict:

=================================================  ==================
Can assign a row it never saw                      Cannot
=================================================  ==================
``kmeans``, ``minibatch_kmeans``, ``birch``        ``dbscan``,
                                                   ``optics``,
                                                   ``agglomerative``
=================================================  ==================

Birch is the row worth reading twice. It is a hierarchical method, sitting beside
Agglomerative in every textbook, and it can be deployed while Agglomerative
cannot -- because it kept a tree of summaries rather than the rows. Three of
these six describe one dataset and have no mechanism for the next one, and a
caller who assumed otherwise would find out at prediction time. That is exactly
why the fact is recorded rather than inferred.
"""

from aidatasetkit.models.clustering.centroid import (
    KMeansStrategy,
    MiniBatchKMeansStrategy,
)
from aidatasetkit.models.clustering.density import DBSCANStrategy, OPTICSStrategy
from aidatasetkit.models.clustering.hierarchical import (
    AgglomerativeClusteringStrategy,
    BirchStrategy,
)

__all__ = [
    "AgglomerativeClusteringStrategy",
    "BirchStrategy",
    "DBSCANStrategy",
    "KMeansStrategy",
    "MiniBatchKMeansStrategy",
    "OPTICSStrategy",
]
