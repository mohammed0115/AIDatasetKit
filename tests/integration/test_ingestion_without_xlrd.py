"""Proof that ingestion does not need xlrd until an .xls file is read.

The package imports, and CSV and .xlsx still load, where xlrd cannot be
imported. Only reading an .xls file asks for it, and that ask is a
MissingDependencyError naming the excel extra.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aidatasetkit

REPOSITORY_ROOT = Path(aidatasetkit.__file__).resolve().parent.parent

_BLOCKER = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "xlrd" or name.startswith("xlrd."):
            raise ImportError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _Blocked())
for module in [name for name in sys.modules if name == "xlrd" or name.startswith("xlrd.")]:
    del sys.modules[module]
"""


def run_without_xlrd(body: str) -> subprocess.CompletedProcess:
    """Execute ``body`` in a subprocess where xlrd cannot be imported."""
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
    result = run_without_xlrd(
        """
        try:
            import xlrd
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
def xls_file(tmp_path_factory):
    """A real .xls file, written here where xlwt exists."""
    pytest.importorskip("xlwt", reason="the dev extra is not installed")
    import xlwt

    folder = tmp_path_factory.mktemp("xls")
    path = folder / "data.xls"
    book = xlwt.Workbook()
    sheet = book.add_sheet("Data")
    sheet.write(0, 0, "a")
    sheet.write(1, 0, 1)
    book.save(str(path))
    return path


@pytest.mark.usefixtures("blocker_works")
class TestIngestionWithoutXlrd:
    def test_the_package_imports(self):
        result = run_without_xlrd("import aidatasetkit; print('IMPORTS')")
        assert "IMPORTS" in result.stdout, result.stderr

    def test_csv_still_loads(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        result = run_without_xlrd(
            f"""
            from pathlib import Path
            from aidatasetkit.ingestion import load_table
            print("SHAPE", load_table(Path({str(path)!r})).frame.shape)
            """
        )
        assert "SHAPE (1, 2)" in result.stdout, result.stderr

    def test_an_xls_read_names_the_extra(self, xls_file):
        result = run_without_xlrd(
            f"""
            from pathlib import Path
            from aidatasetkit.core.exceptions import MissingDependencyError
            from aidatasetkit.ingestion import load_table
            try:
                load_table(Path({str(xls_file)!r}))
            except MissingDependencyError as error:
                print("REFUSED", str(error))
            else:
                print("NOT REFUSED")
            """
        )
        assert "REFUSED" in result.stdout, result.stderr
        assert "aidatasetkit[excel]" in result.stdout
        assert "NOT REFUSED" not in result.stdout

    def test_xlsx_still_uses_openpyxl(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl", reason="the excel extra is not installed")
        path = tmp_path / "data.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active.append(["a"])
        workbook.active.append([1])
        workbook.save(path)
        result = run_without_xlrd(
            f"""
            from pathlib import Path
            from aidatasetkit.ingestion import load_table
            loaded = load_table(Path({str(path)!r}))
            print("XLSX", loaded.metadata.format.value, loaded.frame.iloc[0, 0])
            """
        )
        assert "XLSX xlsx 1" in result.stdout, result.stderr
