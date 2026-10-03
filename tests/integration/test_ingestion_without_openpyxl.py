"""Proof that ingestion does not need openpyxl until an .xlsx file is read.

Same discipline as the pyarrow-absence tests: the package imports, and every
other format loads, on a machine where openpyxl is not installed; only reading
an .xlsx file asks for it, and that ask is a structured MissingDependencyError
naming the extra, not an ImportError traceback.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aidatasetkit

REPOSITORY_ROOT = Path(aidatasetkit.__file__).resolve().parent.parent

#: Installed ahead of the code under test: any import of openpyxl fails, exactly
#: as it would on a machine without the excel extra.
_BLOCKER = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "openpyxl" or name.startswith("openpyxl."):
            raise ImportError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _Blocked())
for module in [name for name in sys.modules if name.startswith("openpyxl")]:
    del sys.modules[module]
"""


def run_without_openpyxl(body: str) -> subprocess.CompletedProcess:
    """Execute ``body`` in a subprocess where openpyxl cannot be imported."""
    script = _BLOCKER + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPOSITORY_ROOT),
        timeout=180,
    )


@pytest.fixture(scope="module")
def blocker_works():
    """Confirm the import hook actually blocks openpyxl before relying on it."""
    result = run_without_openpyxl(
        """
        try:
            import openpyxl
        except ImportError:
            print("BLOCKED")
        else:
            print("NOT BLOCKED")
        """
    )
    assert "BLOCKED" in result.stdout, result.stderr
    assert "NOT BLOCKED" not in result.stdout
    return True


@pytest.fixture(scope="module")
def xlsx_file(tmp_path_factory):
    """A real .xlsx file, written here where openpyxl exists."""
    pytest.importorskip("openpyxl", reason="the excel extra is not installed")
    import openpyxl

    folder = tmp_path_factory.mktemp("excel")
    path = folder / "data.xlsx"
    wb = openpyxl.Workbook()
    wb.active.append(["a"])
    wb.active.append([1])
    wb.save(path)
    return path


@pytest.mark.usefixtures("blocker_works")
class TestIngestionWithoutOpenpyxl:
    def test_the_package_imports(self):
        result = run_without_openpyxl("import aidatasetkit; print('IMPORTS')")
        assert "IMPORTS" in result.stdout, result.stderr

    def test_csv_still_loads(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        result = run_without_openpyxl(
            f"""
            from pathlib import Path
            from aidatasetkit.ingestion import load_table
            print("SHAPE", load_table(Path({str(path)!r})).frame.shape)
            """
        )
        assert "SHAPE (1, 2)" in result.stdout, result.stderr

    def test_an_xlsx_read_names_the_extra(self, xlsx_file):
        result = run_without_openpyxl(
            f"""
            from pathlib import Path
            from aidatasetkit.core.exceptions import MissingDependencyError
            from aidatasetkit.ingestion import load_table
            try:
                load_table(Path({str(xlsx_file)!r}))
            except MissingDependencyError as error:
                print("REFUSED", str(error))
            else:
                print("NOT REFUSED")
            """
        )
        assert "REFUSED" in result.stdout, result.stderr
        assert "aidatasetkit[excel]" in result.stdout
        assert "NOT REFUSED" not in result.stdout
