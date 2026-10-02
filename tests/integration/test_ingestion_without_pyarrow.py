"""Proof that ingestion does not need pyarrow until a columnar file is read.

The claim is architectural: the package imports, and CSV/TSV/JSON/records all
load, on a machine where pyarrow is not installed; only reading a Parquet or
Feather file asks for it, and that ask is a structured MissingDependencyError
naming the extra, not an ImportError traceback.

Asserting that from inside this process would prove nothing, because pyarrow
*is* installed here for the columnar tests. Each test therefore runs in a fresh
subprocess with an import hook that makes ``pyarrow`` unimportable, which is
what the absence would look like.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aidatasetkit

REPOSITORY_ROOT = Path(aidatasetkit.__file__).resolve().parent.parent

#: Installed ahead of the code under test: any import of pyarrow fails, exactly
#: as it would on a machine without the parquet extra.
_BLOCKER = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _Blocked())
for module in [name for name in sys.modules if name.startswith("pyarrow")]:
    del sys.modules[module]
"""


def run_without_pyarrow(body: str) -> subprocess.CompletedProcess:
    """Execute ``body`` in a subprocess where pyarrow cannot be imported."""
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
    """Confirm the import hook actually blocks pyarrow before relying on it."""
    result = run_without_pyarrow(
        """
        try:
            import pyarrow
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
def parquet_file(tmp_path_factory):
    """A real Parquet file, written here where pyarrow exists."""
    pytest.importorskip("pyarrow", reason="the parquet extra is not installed")
    import pandas as pd

    folder = tmp_path_factory.mktemp("columnar")
    path = folder / "data.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path, index=False)
    return path


@pytest.mark.usefixtures("blocker_works")
class TestIngestionWithoutPyarrow:
    def test_the_package_imports(self):
        result = run_without_pyarrow("import aidatasetkit; print('IMPORTS')")
        assert "IMPORTS" in result.stdout, result.stderr

    def test_json_still_loads(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('[{"a": 1}]', encoding="utf-8")
        result = run_without_pyarrow(
            f"""
            from pathlib import Path
            from aidatasetkit.ingestion import load_table
            print("ROWS", load_table(Path({str(path)!r})).frame.shape[0])
            """
        )
        assert "ROWS 1" in result.stdout, result.stderr

    @pytest.mark.parametrize("suffix", [".parquet", ".feather"])
    def test_a_columnar_read_names_the_extra(self, parquet_file, suffix):
        path = parquet_file if suffix == ".parquet" else parquet_file.with_suffix(".feather")
        if suffix == ".feather":
            import pandas as pd

            pd.read_parquet(parquet_file).to_feather(path)
        result = run_without_pyarrow(
            f"""
            from pathlib import Path
            from aidatasetkit.core.exceptions import MissingDependencyError
            from aidatasetkit.ingestion import load_table
            try:
                load_table(Path({str(path)!r}))
            except MissingDependencyError as error:
                print("REFUSED", str(error))
            else:
                print("NOT REFUSED")
            """
        )
        assert "REFUSED" in result.stdout, result.stderr
        assert "aidatasetkit[parquet]" in result.stdout
        assert "NOT REFUSED" not in result.stdout
