"""Validate the research notebook's current public AIDatasetKit references without network access."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import nbformat

from aidatasetkit import AIDataFacade
from aidatasetkit.models import ModelFactory

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "AIDatasetKit_Real_World_Comparative_Study.ipynb"
PROTOCOL = ROOT.parent.parent / "docs" / "research" / "AIDatasetKit_Research_Protocol.md"
README = ROOT / "README.md"

notebook = nbformat.read(NOTEBOOK, as_version=4)
assert all(cell.get("id") for cell in notebook.cells), "Every notebook cell requires an id."
source = "\n".join(cell.source for cell in notebook.cells)
public_methods = {
    name
    for name, value in inspect.getmembers(AIDataFacade, inspect.isfunction)
    if not name.startswith("_")
}
method_mentions = set(re.findall(r"\bai_[A-Za-z_]+\.([a-z_]+)\(", source))
unknown_methods = method_mentions - public_methods
assert not unknown_methods, f"Unknown AIDataFacade methods: {sorted(unknown_methods)}"

catalog = {
    family: set(ModelFactory.available(task=family))
    for family in (
        "classification",
        "regression",
        "clustering",
        "anomaly_detection",
        "dimensionality_reduction",
    )
}
assert "kmeans_clustering" in catalog["clustering"]
assert 'select_model("kmeans")' not in source
assert 'cluster("kmeans_clustering", n_clusters=3)' in source
for expected in (
    "AIDatasetKit vs. a Conventional Expert Workflow",
    "Controlled adversarial experiments",
    "When the conventional expert workflow is better",
    "When AIDatasetKit may help a learner",
    "Threats to validity",
):
    assert expected in source, f"Notebook section missing: {expected}"
for path in (NOTEBOOK, PROTOCOL, README):
    assert path.is_file(), f"Missing research artifact: {path}"
print("Research notebook API and structure validation passed.")
