"""The clustering catalog: every declared fact, executed against scikit-learn.

The generic contracts in ``test_model_contracts.py`` already run over the live
registry, so these six models are subjected to all of them the moment they are
registered. What is here is the part those cannot express: the facts that are
specific to clustering, and the two claims S9 added to the library's vocabulary.

Nothing here reads a docstring or a class hierarchy. Every assertion fits an
estimator and looks at what happened.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from aidatasetkit.core.exceptions import IncompatibleModelError, ValidationError
from aidatasetkit.core.types import (
    ClusterEstimator,
    Estimator,
    Fittable,
    Interpretability,
    TargetProfile,
    TaskType,
)
from aidatasetkit.models import ModelCapabilities, default_registry
from tests.conftest import BUILT_IN_CLUSTERERS

CLUSTERERS = [
    entry
    for entry in default_registry().catalog()
    if entry.task_type is TaskType.CLUSTERING
]


def _identify(entry) -> str:
    return entry.canonical_name


def _blobs(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated spherical clusters, and the truth behind them."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 7.0]])
    which = rng.integers(0, 3, n)
    return centres[which] + rng.normal(0, 0.7, (n, 2)), which


def _labels(entry, X, **params) -> np.ndarray:
    return np.asarray(entry.strategy_type().build(**params).fit_predict(X))


class TestTheCatalogIsWhatItSaysItIs:
    def test_exactly_six_clusterers_are_registered(self):
        assert tuple(sorted(e.canonical_name for e in CLUSTERERS)) == (
            BUILT_IN_CLUSTERERS
        )

    def test_none_of_them_claims_a_supervised_capability(self):
        for entry in CLUSTERERS:
            assert not entry.capabilities.supports_predict_proba
            assert not entry.capabilities.supports_multiclass


@pytest.mark.parametrize("entry", CLUSTERERS, ids=_identify)
class TestEveryDeclaredCapabilityIsExecuted:
    """One test per declared fact, each of which fits the real estimator."""

    def test_requires_scaling_is_true_and_the_data_agrees(self, entry):
        """Every one of the six declares it, so every one of the six must earn it.

        The frame is three separated blobs with one feature recorded in a unit
        1e5 larger -- an income beside a ratio, which is an ordinary pair of
        columns and not a contrived one. The claim is that standardising changes
        the answer, so the two partitions are compared directly.
        """
        assert entry.capabilities.requires_scaling

        points, truth = _blobs(seed=5)
        stretched = points * np.array([1.0, 1e5])
        scaled = StandardScaler().fit_transform(stretched)

        raw = _labels(entry, stretched)
        standardised = _labels(entry, scaled)
        assert adjusted_rand_score(truth, raw) != adjusted_rand_score(
            truth, standardised
        ), "standardising this frame changed nothing, so the declaration is unearned"

    def test_sparse_support_is_exactly_what_the_estimator_does(self, entry):
        X, _ = _blobs()
        sparse = csr_matrix(np.abs(X))
        if entry.capabilities.supports_sparse_input:
            assert _labels(entry, sparse).shape == (X.shape[0],)
        else:
            with pytest.raises((TypeError, ValueError)):
                _labels(entry, sparse)

    def test_no_clusterer_accepts_a_missing_value(self, entry):
        """All six declare ``handles_missing_values=False``; all six must raise."""
        assert not entry.capabilities.handles_missing_values
        X, _ = _blobs()
        holed = X.copy()
        holed[:30, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            _labels(entry, holed)

    def test_out_of_sample_assignment_matches_the_declaration(self, entry):
        """The claim S9 added, checked by calling rather than by looking.

        ``hasattr`` alone would not do: a method that exists and raises would
        pass it while failing the caller. So the declaring models are made to
        assign real rows, and the rest are required to have nothing to call.
        """
        X, _ = _blobs()
        fitted = entry.strategy_type().build()
        fitted.fit(X)

        if entry.capabilities.supports_out_of_sample_assignment:
            assigned = np.asarray(fitted.predict(X[:10]))
            assert assigned.shape == (10,)
            # On the rows it was fitted on, an assignment that disagreed with
            # the fitted labelling would mean the two describe different
            # partitions -- and `assign` would be answering a different question
            # from the one `labels` reports.
            assert (np.asarray(fitted.predict(X)) == fitted.labels_).all()
        else:
            assert not hasattr(fitted, "predict")

    def test_the_protocols_agree_with_the_declaration(self, entry):
        built = entry.strategy_type().build()
        assert isinstance(built, Fittable)
        assert isinstance(built, ClusterEstimator)
        assert isinstance(built, Estimator) is (
            entry.capabilities.supports_out_of_sample_assignment
        )

    def test_interpretability_rests_on_an_attribute_that_exists(self, entry):
        """HIGH is claimed only where a readable per-cluster summary is produced.

        The field is descriptive metadata, but "descriptive" is not "unfalsifiable":
        the two models claiming HIGH do so because they keep ``cluster_centers_``,
        one point per cluster in feature space, and that is checkable.
        """
        X, _ = _blobs()
        fitted = entry.strategy_type().build(n_clusters=3) if _takes_k(entry) else (
            entry.strategy_type().build()
        )
        fitted.fit(X)
        if entry.capabilities.interpretability_level is Interpretability.HIGH:
            assert hasattr(fitted, "cluster_centers_")
            assert fitted.cluster_centers_.shape == (3, X.shape[1])
        else:
            assert not hasattr(fitted, "cluster_centers_")

    def test_it_is_deterministic_when_run_twice(self, entry):
        """Same seed, same data, same partition. Twice, not once."""
        X, _ = _blobs()
        assert (_labels(entry, X) == _labels(entry, X)).all()


def _takes_k(entry) -> bool:
    return "n_clusters" in entry.strategy_type().build().get_params()


class TestNoiseIsNotACluster:
    """``-1`` is the algorithm declining to assign, and only two models emit it."""

    @staticmethod
    def _noisy() -> np.ndarray:
        X, _ = _blobs()
        rng = np.random.default_rng(99)
        return np.vstack([X, rng.uniform(-25, 25, (30, 2))])

    def test_the_density_models_do_label_some_rows_noise(self):
        """Otherwise every noise-handling test below passes on data with none."""
        scaled = StandardScaler().fit_transform(self._noisy())
        produced = {
            name: int((_labels(_entry(name), scaled) == -1).sum())
            for name in ("dbscan_clustering", "optics_clustering")
        }
        assert all(count > 0 for count in produced.values()), produced

    def test_the_documented_optics_and_dbscan_table_still_holds(self):
        """The table in ``density.py`` and ``docs/clustering.md``, re-derived.

        Pinned because an earlier draft of that docstring quoted figures
        measured on a generator earlier lines had already advanced, so they
        could not be reproduced from the seeds they named. A documented number
        nothing re-runs is a number that quietly stops being true.

        Both generators are constructed here rather than shared, which is
        exactly the mistake being guarded against.
        """
        blobs = np.random.default_rng(0)
        centres = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 7.0]])
        clean = centres[blobs.integers(0, 3, 300)] + blobs.normal(0, 0.7, (300, 2))
        outliers = np.random.default_rng(99).uniform(-25, 25, (30, 2))
        noisy = np.vstack([clean, outliers])

        expected = {
            ("optics_clustering", "clean"): (13, 186),
            ("dbscan_clustering", "clean"): (3, 17),
            ("optics_clustering", "noisy"): (13, 215),
            ("dbscan_clustering", "noisy"): (3, 47),
        }
        for (name, which), (clusters, noise) in expected.items():
            labels = _labels(_entry(name), clean if which == "clean" else noisy)
            assert (
                len(set(labels.tolist()) - {-1}),
                int((labels == -1).sum()),
            ) == (clusters, noise), f"{name} on the {which} frame"

    def test_the_partitioning_models_never_do(self):
        scaled = StandardScaler().fit_transform(self._noisy())
        for name in (
            "kmeans_clustering",
            "minibatch_kmeans_clustering",
            "birch_clustering",
            "agglomerative_clustering",
        ):
            assert (_labels(_entry(name), scaled, n_clusters=3) != -1).all()


def _entry(name):
    return default_registry().resolve(name)


class TestTheNewCapabilityField:
    """``supports_out_of_sample_assignment``: default, meaning, and refusals."""

    @staticmethod
    def _clustering(**overrides) -> ModelCapabilities:
        base = dict(
            task_type=TaskType.CLUSTERING,
            backend="sklearn",
            supports_predict_proba=False,
            requires_scaling=True,
            supports_sparse_input=True,
            supports_multiclass=False,
            handles_missing_values=False,
            interpretability_level=Interpretability.HIGH,
        )
        from aidatasetkit.core.types import Backend

        base["backend"] = Backend.SKLEARN
        return ModelCapabilities(**{**base, **overrides})

    def test_it_defaults_to_false(self):
        """So every model written before S9 keeps the answer it already had."""
        assert self._clustering().supports_out_of_sample_assignment is False

    def test_every_supervised_model_leaves_it_at_the_default(self):
        for entry in default_registry().catalog():
            if entry.task_type is not TaskType.CLUSTERING:
                assert not entry.capabilities.supports_out_of_sample_assignment

    def test_a_supervised_model_cannot_declare_it(self):
        """It would state nothing: predicting unseen rows is what they are for."""
        with pytest.raises(
            ValidationError, match="supports_out_of_sample_assignment is meaningless"
        ):
            ModelCapabilities(
                task_type=TaskType.REGRESSION,
                backend=__import__(
                    "aidatasetkit.core.types", fromlist=["Backend"]
                ).Backend.SKLEARN,
                supports_predict_proba=False,
                requires_scaling=False,
                supports_sparse_input=True,
                supports_multiclass=False,
                handles_missing_values=False,
                interpretability_level=Interpretability.HIGH,
                supports_out_of_sample_assignment=True,
            )

    def test_it_reaches_the_serialised_capabilities(self):
        """Not special-cased out of the artifact to keep an old fingerprint."""
        payload = self._clustering(
            supports_out_of_sample_assignment=True
        ).to_dict()
        assert payload["supports_out_of_sample_assignment"] is True

    def test_it_is_emitted_for_supervised_models_too(self):
        """A field present on some rows and absent on others cannot be read."""
        payload = default_registry().resolve("logistic_regression").capabilities.to_dict()
        assert payload["supports_out_of_sample_assignment"] is False

    def test_it_does_not_take_part_in_the_preprocessing_profile(self):
        """Only three answers change how a pipeline is assembled, and this is not one.

        Had it joined the cache key, the six clusterers would have split into
        two more profiles and every capability-identical model would have stopped
        sharing a preprocessor for no reason at all.
        """
        assert (
            self._clustering(supports_out_of_sample_assignment=True)
            .preprocessing_profile()
            == self._clustering(supports_out_of_sample_assignment=False)
            .preprocessing_profile()
        )


class TestValidateForWithNoTarget:
    """Conflict D: ``validate_for(None)`` is an answer, not an AttributeError."""

    def test_a_clusterer_accepts_no_target(self):
        for entry in CLUSTERERS:
            entry.capabilities.validate_for(None)  # must not raise

    def test_a_supervised_model_refuses_no_target(self):
        capabilities = default_registry().resolve("logistic_regression").capabilities
        with pytest.raises(IncompatibleModelError, match="offered no target"):
            capabilities.validate_for(None)

    def test_it_is_a_refusal_rather_than_an_attribute_error(self):
        """The latent defect S9 found: `target.task_type` on None.

        A bare ``AttributeError`` names the wrong thing and cannot be caught by
        anybody handling model incompatibility, so the type of the exception is
        part of the contract.
        """
        capabilities = default_registry().resolve("ridge_regression").capabilities
        with pytest.raises(IncompatibleModelError):
            capabilities.validate_for(None)

    def test_is_compatible_with_answers_rather_than_raising(self):
        clusterer = _entry("kmeans_clustering").capabilities
        supervised = _entry("gaussian_nb").capabilities
        assert clusterer.is_compatible_with(None) is True
        assert supervised.is_compatible_with(None) is False

    def test_a_clusterer_still_refuses_a_supervised_target(self):
        profile = TargetProfile(task_type=TaskType.CLASSIFICATION, n_classes=2)
        with pytest.raises(IncompatibleModelError):
            _entry("kmeans_clustering").capabilities.validate_for(profile)
