"""Proof that planning does not need a plotting library.

The claim is architectural: recommendation, scoring, redundancy filtering, and
serialisation must all work on a machine where matplotlib is not installed, and
only an actual render request should need it.

Asserting that from inside this process would prove nothing, because matplotlib
*is* installed here for the renderer tests. Each test therefore runs in a fresh
subprocess with an import hook that makes ``matplotlib`` unimportable, which is
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

#: Installed ahead of the code under test: any import of matplotlib fails, exactly
#: as it would on a machine without the viz extra.
_BLOCKER = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _Blocked())
for module in [name for name in sys.modules if name.startswith("matplotlib")]:
    del sys.modules[module]
"""


def run_without_matplotlib(body: str) -> subprocess.CompletedProcess:
    """Execute ``body`` in a subprocess where matplotlib cannot be imported."""
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
    """Confirm the import hook actually blocks matplotlib before relying on it."""
    result = run_without_matplotlib(
        """
        try:
            import matplotlib
        except ImportError:
            print("BLOCKED")
        else:
            print("NOT BLOCKED")
        """
    )
    assert "BLOCKED" in result.stdout, result.stderr
    assert "NOT BLOCKED" not in result.stdout
    return True


class TestPlanningWithoutMatplotlib:
    def test_the_package_imports(self, blocker_works):
        result = run_without_matplotlib(
            """
            import aidatasetkit.visualization as viz
            print("IMPORTED", sorted(viz.__all__)[0])
            """
        )
        assert "IMPORTED" in result.stdout, result.stderr

    def test_importing_the_package_pulls_in_no_plotting_library(self, blocker_works):
        result = run_without_matplotlib(
            """
            import sys
            import aidatasetkit.visualization  # noqa: F401
            loaded = [name for name in sys.modules if name.startswith("matplotlib")]
            print("MATPLOTLIB_MODULES", loaded)
            """
        )
        assert "MATPLOTLIB_MODULES []" in result.stdout, result.stderr

    def test_a_full_plan_is_produced_and_serialised(self, blocker_works):
        result = run_without_matplotlib(
            """
            import json
            import numpy as np
            import pandas as pd
            from aidatasetkit.visualization import VisualizationService

            index = np.arange(300)
            frame = pd.DataFrame({
                "age": (20 + index % 45).astype("int64"),
                "income": (30000 + (index % 71) * 900).astype("float64"),
                "segment": [["A", "B", "C"][v % 3] for v in index],
                "churn": (index % 7 == 0).astype("int64"),
            })
            plan = VisualizationService().recommend(frame, target="churn")
            payload = json.dumps(plan.to_dict())
            print("CHARTS", len(plan.charts))
            print("TOP", plan.charts[0].chart_type.value)
            print("SERIALISED", len(payload) > 0)
            """
        )
        assert "CHARTS" in result.stdout, result.stderr
        assert "TOP target_distribution" in result.stdout
        assert "SERIALISED True" in result.stdout

    def test_preparation_works_without_a_renderer(self, blocker_works):
        result = run_without_matplotlib(
            """
            import numpy as np
            import pandas as pd
            from aidatasetkit.visualization import VisualizationService

            frame = pd.DataFrame({"x": (np.arange(200) % 37).astype("float64")})
            prepared = VisualizationService().histogram(frame, "x")
            print("PREPARED", len(prepared.data["values"]))
            """
        )
        assert "PREPARED 200" in result.stdout, result.stderr

    def test_manual_validation_still_reports_library_errors(self, blocker_works):
        result = run_without_matplotlib(
            """
            import numpy as np
            import pandas as pd
            from aidatasetkit.core.exceptions import InvalidVisualizationRequest
            from aidatasetkit.visualization import VisualizationService

            frame = pd.DataFrame({"c": ["a", "b"] * 50})
            try:
                VisualizationService().histogram(frame, "c")
            except InvalidVisualizationRequest as error:
                print("REFUSED", "numeric" in str(error))
            """
        )
        assert "REFUSED True" in result.stdout, result.stderr


class TestRenderingWithoutMatplotlib:
    def test_asking_to_render_explains_how_to_install_it(self, blocker_works):
        result = run_without_matplotlib(
            """
            import numpy as np
            import pandas as pd
            from aidatasetkit.core.exceptions import MissingDependencyError
            from aidatasetkit.visualization import VisualizationService

            frame = pd.DataFrame({"x": (np.arange(200) % 37).astype("float64")})
            service = VisualizationService()
            spec = service.recommend(frame).charts[0]
            try:
                service.render(frame, spec)
            except MissingDependencyError as error:
                print("HINT", "pip install aidatasetkit[viz]" in str(error))
                print("REASSURED", "planning do not need it" in str(error))
            """
        )
        assert "HINT True" in result.stdout, result.stderr
        assert "REASSURED True" in result.stdout

    def test_the_failure_is_a_library_error_not_an_import_error(self, blocker_works):
        result = run_without_matplotlib(
            """
            import numpy as np
            import pandas as pd
            from aidatasetkit.core.exceptions import AIDatasetKitError
            from aidatasetkit.visualization import VisualizationService

            frame = pd.DataFrame({"x": (np.arange(200) % 37).astype("float64")})
            service = VisualizationService()
            spec = service.recommend(frame).charts[0]
            try:
                service.render(frame, spec)
            except AIDatasetKitError as error:
                print("LIBRARY_ERROR", type(error).__name__)
            except ImportError:
                print("RAW_IMPORT_ERROR")
            """
        )
        assert "LIBRARY_ERROR MissingDependencyError" in result.stdout, result.stderr
        assert "RAW_IMPORT_ERROR" not in result.stdout
