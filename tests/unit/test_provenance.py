"""Tests for environment capture."""

from __future__ import annotations

import json

from aidatasetkit.core.provenance import (
    EnvironmentVersions,
    capture_environment,
    package_version,
)


class TestPackageVersion:
    def test_installed_distribution_reports_a_version(self):
        assert package_version("numpy") != "unknown"

    def test_missing_distribution_reports_unknown_instead_of_raising(self):
        assert package_version("a-distribution-that-does-not-exist") == "unknown"


class TestCaptureEnvironment:
    def test_the_whole_numerical_stack_is_recorded(self):
        environment = capture_environment()
        assert isinstance(environment, EnvironmentVersions)
        for value in environment.to_dict().values():
            assert isinstance(value, str) and value

    def test_the_library_itself_is_recorded(self):
        import aidatasetkit

        assert capture_environment().aidatasetkit == aidatasetkit.__version__

    def test_result_is_cached(self):
        assert capture_environment() is capture_environment()

    def test_record_is_json_serialisable(self):
        payload = json.loads(json.dumps(capture_environment().to_dict()))
        assert "scikit_learn" in payload
