"""The environment the determinism tests hand to a child interpreter.

Those tests exist to show that an answer does not depend on the hash seed, so
the child has to inherit almost nothing. It also has to be able to start. On
Windows the two pulled against each other: an environment of only the seed and
an empty PATH could not import scikit-learn, and twelve tests errored in setup
before measuring anything. These tests hold both halves of the contract.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.conftest import WINDOWS_REQUIRED_ENV, isolated_env


class TestItStaysIsolated:
    def test_it_carries_the_seed(self):
        assert isolated_env("12345")["PYTHONHASHSEED"] == "12345"

    def test_the_path_is_empty(self):
        assert isolated_env("0")["PATH"] == ""

    def test_nothing_else_leaks_from_the_parent(self):
        allowed = {"PYTHONHASHSEED", "PATH", *WINDOWS_REQUIRED_ENV}
        assert set(isolated_env("0")) <= allowed

    @pytest.mark.skipif(sys.platform == "win32", reason="the Windows minimum applies")
    def test_off_windows_it_is_exactly_the_seed_and_path(self):
        assert set(isolated_env("0")) == {"PYTHONHASHSEED", "PATH"}


class TestAChildCanStart:
    def test_a_child_under_it_imports_scikit_learn(self):
        """The failure this helper was written for, reproduced as a check."""
        result = subprocess.run(
            [sys.executable, "-c", "import sklearn, sys; print(sys.flags.hash_randomization)"],
            capture_output=True,
            text=True,
            env=isolated_env("0"),
        )
        assert result.returncode == 0, result.stderr
        # PYTHONHASHSEED=0 disables randomisation: proof the seed arrived.
        assert result.stdout.strip() == "0"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only requirement")
    def test_without_the_windows_minimum_it_cannot(self):
        """Evidence that the one variable passed is necessary, not decorative."""
        env = {k: v for k, v in isolated_env("0").items() if k not in WINDOWS_REQUIRED_ENV}
        result = subprocess.run(
            [sys.executable, "-c", "import sklearn"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "10106" in result.stderr
