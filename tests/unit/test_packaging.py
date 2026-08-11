"""What the published package promises about itself.

These check the contract a user meets before any analysis happens: the version
they installed, the command they can type, the metadata a package index will
read, and the promise that a core install does not drag in an optional plotting
library. None of it needs the repository, which is the point.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestVersionHasOneSource:
    def test_the_package_reports_a_version(self):
        import aidatasetkit

        assert aidatasetkit.__version__

    def test_it_is_the_alpha_we_are_releasing(self, pyproject):
        import aidatasetkit

        assert aidatasetkit.__version__ == pyproject["project"]["version"] == "0.1.0a1"

    def test_it_is_read_from_installed_metadata_not_hardcoded(self):
        """One authoritative source: change pyproject and the package follows."""
        source = (ROOT / "aidatasetkit" / "__init__.py").read_text(encoding="utf-8")
        assert "version(" in source

    def test_the_artifact_schema_is_versioned_separately(self):
        import aidatasetkit
        from aidatasetkit.evidence import ARTIFACT_SCHEMA_VERSION

        assert ARTIFACT_SCHEMA_VERSION == "1.0"
        assert ARTIFACT_SCHEMA_VERSION != aidatasetkit.__version__


class TestTheConsoleCommand:
    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from aidatasetkit.cli.main import main; raise SystemExit(main())",
                *args,
            ],
            capture_output=True,
            text=True,
        )

    def test_the_entry_point_is_declared(self, pyproject):
        scripts = pyproject["project"]["scripts"]
        assert scripts["aidatasetkit"] == "aidatasetkit.cli.main:main"

    def test_version_prints_cleanly(self):
        import aidatasetkit

        result = self._run("--version")
        assert result.returncode == 0
        assert aidatasetkit.__version__ in result.stdout
        assert "Traceback" not in result.stderr

    def test_help_explains_the_exit_codes(self):
        result = self._run("--help")
        assert result.returncode == 0
        for token in ("0", "1", "2", "3", "blocked", "threshold"):
            assert token in result.stdout

    def test_audit_help_explains_every_argument(self):
        result = self._run("audit", "--help")
        for flag in (
            "--target",
            "--task",
            "--model",
            "--output",
            "--fail-on",
            "--include-values",
        ):
            assert flag in result.stdout, flag

    def test_audit_help_says_nothing_is_trained(self):
        result = self._run("audit", "--help")
        assert "trained" in result.stdout.lower()

    def test_audit_help_warns_before_sharing_an_artifact(self):
        """Checks the warning, not one word of it.

        An earlier version asserted the word "privacy", which pointed installed
        users at a docs file no wheel contains. The help now names what an
        artifact holds instead, and the test asks whether a user is warned rather
        than whether a particular noun survived.
        """
        text = self._run("audit", "--help").stdout.lower()
        assert "sharing" in text
        assert "column names" in text


class TestDependenciesAreClassified:
    def test_the_runtime_dependencies_are_the_four_we_verified(self, pyproject):
        names = {
            dependency.split(">")[0].split("=")[0].strip()
            for dependency in pyproject["project"]["dependencies"]
        }
        assert names == {"numpy", "pandas", "scipy", "scikit-learn"}

    def test_no_test_or_build_tool_is_a_runtime_dependency(self, pyproject):
        text = " ".join(pyproject["project"]["dependencies"]).lower()
        for tool in ("pytest", "build", "twine", "setuptools", "wheel", "black", "ruff"):
            assert tool not in text, tool

    def test_matplotlib_is_optional_only(self, pyproject):
        extras = pyproject["project"]["optional-dependencies"]
        assert any("matplotlib" in item for item in extras["viz"])
        assert not any(
            "matplotlib" in item for item in pyproject["project"]["dependencies"]
        )

    def test_importing_the_package_does_not_import_matplotlib(self):
        """A core install has no matplotlib; importing must not assume one."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, aidatasetkit; print('matplotlib' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"

    def test_the_evidence_layer_does_not_import_matplotlib_either(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, aidatasetkit.evidence, aidatasetkit.cli.main;"
                " print('matplotlib' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"


class TestPackageMetadata:
    def test_the_python_requirement_matches_what_was_tested(self, pyproject):
        assert pyproject["project"]["requires-python"] == ">=3.11"

    def test_the_running_interpreter_satisfies_it(self):
        assert sys.version_info >= (3, 11)

    def test_the_readme_is_the_long_description(self, pyproject):
        assert pyproject["project"]["readme"] == "README.md"

    def test_no_invented_project_urls(self, pyproject):
        """A placeholder URL in package metadata is a promise nobody can keep."""
        for url in pyproject["project"].get("urls", {}).values():
            assert "OWNER" not in url and "example.com" not in url

    def test_the_manifest_ships_docs_and_tests_in_the_sdist(self):
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        for wanted in ("docs", "tests", "examples", "README.md"):
            assert wanted in manifest, wanted


class TestReleaseDocumentsExist:
    @pytest.mark.parametrize(
        "path",
        [
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "RELEASE_NOTES_0.1.0a1.md",
            "docs/getting-started.md",
            "docs/audit-artifact.md",
            "docs/lineage.md",
            "docs/privacy.md",
            "docs/limitations.md",
            "docs/safety-model.md",
            "docs/book/OUTLINE.md",
            "examples/public_alpha/README.md",
            "scripts/release_smoke_test.sh",
        ],
    )
    def test_it_is_present(self, path):
        assert (ROOT / path).exists(), path

    def test_the_release_notes_name_this_version(self):
        notes = (ROOT / "RELEASE_NOTES_0.1.0a1.md").read_text(encoding="utf-8")
        assert "0.1.0a1" in notes

    def test_the_changelog_opens_with_this_version(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert changelog.index("0.1.0a1") < 400

    #: Words a release document may only use while denying them. Naming the word
    #: is not the problem -- "this is not certified" is exactly the sentence a
    #: reader deserves. Asserting it is.
    LOADED = ("production-ready", "certified", "compliant", "leakage-free", "guaranteed")

    @pytest.mark.parametrize(
        "name",
        [
            "README.md",
            "CHANGELOG.md",
            "RELEASE_NOTES_0.1.0a1.md",
            "docs/limitations.md",
            "docs/safety-model.md",
        ],
    )
    def test_every_loaded_word_appears_only_inside_a_denial(self, name):
        import re

        text = (ROOT / name).read_text(encoding="utf-8").lower()
        for word in self.LOADED:
            for match in re.finditer(re.escape(word), text):
                # A sentence's worth of context: the denial that governs a list
                # ("no output will ever tell you your data is: a, b, c") sits
                # further back than the last comma.
                before = text[max(0, match.start() - 240) : match.start()]
                denied = any(
                    marker in before
                    for marker in (
                        "not ",
                        "no ",
                        "never",
                        "does not",
                        "avoid",
                        "without",
                        "rather than",
                        "instead of",
                        "will ever tell",
                        "do not use",
                    )
                )
                assert denied, (
                    f"{name} uses {word!r} as a claim: ...{text[match.start()-70:match.end()+30]!r}"
                )

    def test_the_guard_can_actually_fail(self):
        """A sentence that asserts one of these words must be caught."""
        import re

        text = "aidatasetkit is production-ready and fully compliant."
        word = "production-ready"
        match = re.search(re.escape(word), text)
        before = text[: match.start()]
        assert not any(m in before for m in ("not ", "no ", "never", "does not"))


class TestOneCanonicalListOfLimitations:
    """docs/limitations.md is the single list, and the artifact is a subset of it.

    Two lists that drift apart is the failure mode here: a reader trusts whichever
    they found first, and the artifact's copy is the one that travels furthest.
    """

    @staticmethod
    def _significant(text: str) -> set[str]:
        import re

        stop = {
            "the", "and", "that", "with", "this", "from", "have", "been", "does",
            "not", "for", "are", "its", "it", "a", "an", "of", "in", "is", "to",
            "which", "than", "any", "all", "one", "into", "them", "they", "here",
        }
        words = re.findall(r"[a-z]{4,}", text.lower())
        return {word for word in words if word not in stop}

    @pytest.fixture(scope="class")
    @staticmethod
    def canonical() -> str:
        return (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8").lower()

    def test_the_document_declares_itself_canonical(self, canonical):
        assert "canonical list" in canonical

    def test_every_limitation_the_artifact_carries_is_covered(self, canonical):
        from aidatasetkit.evidence import KNOWN_LIMITATIONS

        vocabulary = self._significant(canonical)
        for limitation in KNOWN_LIMITATIONS:
            words = self._significant(limitation)
            overlap = words & vocabulary
            assert len(overlap) >= max(3, len(words) // 3), (
                f"docs/limitations.md does not cover: {limitation[:70]!r}"
            )

    def test_the_artifact_still_carries_them_so_a_reader_always_has_them(self):
        from aidatasetkit.evidence import KNOWN_LIMITATIONS

        assert len(KNOWN_LIMITATIONS) >= 7

    def test_no_document_promises_more_than_the_canonical_list(self):
        """A release document may summarise the list; it may not contradict it."""
        for name in ("README.md", "RELEASE_NOTES_0.1.0a1.md"):
            text = (ROOT / name).read_text(encoding="utf-8").lower()
            assert "limitations.md" in text or "known limitations" in text, name


class TestTheTypedClassifierIsHonest:
    """The metadata claimed "Typing :: Typed" with no PEP 561 marker.

    Without ``py.typed`` a type checker ignores every annotation in the installed
    package, so the classifier advertised something the distribution did not
    deliver. The library is annotated; it was the marker that was missing.
    """

    def test_the_marker_exists(self):
        assert (ROOT / "aidatasetkit" / "py.typed").exists()

    def test_it_is_declared_as_package_data(self, pyproject):
        data = pyproject["tool"]["setuptools"]["package-data"]
        assert "py.typed" in data["aidatasetkit"]

    def test_the_classifier_is_still_claimed(self, pyproject):
        assert "Typing :: Typed" in pyproject["project"]["classifiers"]

    def test_the_library_really_is_annotated(self):
        """The claim has to be true in substance, not only in metadata."""
        modules = list((ROOT / "aidatasetkit").rglob("*.py"))
        annotated = [
            path
            for path in modules
            if "from __future__ import annotations" in path.read_text(encoding="utf-8")
        ]
        assert len(annotated) / len(modules) > 0.7


class TestTheLongDescriptionRendersOnAPackageIndex:
    """README is the PyPI long description, where a relative link is a dead link."""

    @pytest.fixture(scope="class")
    @staticmethod
    def readme() -> str:
        return (ROOT / "README.md").read_text(encoding="utf-8")

    def test_no_relative_markdown_links(self, readme):
        import re

        relative = sorted(
            {
                target
                for target in re.findall(r"\]\((?!https?://|#)([^)]+)\)", readme)
            }
        )
        assert not relative, f"these become dead links on PyPI: {relative}"

    def test_no_relative_images(self, readme):
        import re

        assert not re.findall(r"!\[[^\]]*\]\((?!https?://)([^)]+)\)", readme)

    def test_no_raw_html_that_indexes_strip(self, readme):
        import re

        assert not re.search(r"<(div|img|table|details|br)\b", readme)

    def test_the_content_type_is_declared(self, pyproject):
        assert pyproject["project"]["readme"] == "README.md"

    def test_documentation_is_still_reachable_by_path(self, readme):
        """Dropping the links must not drop the reader's way to the docs."""
        for path in ("docs/privacy.md", "docs/getting-started.md"):
            assert path in readme
