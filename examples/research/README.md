# Research Examples

## Real-World Comparative Study

[Open the executable notebook](AIDatasetKit_Real_World_Comparative_Study.ipynb) to compare a competent manual `pandas`/`scikit-learn` workflow with AIDatasetKit on published UCI tabular datasets. The companion [research protocol](../../docs/research/AIDatasetKit_Research_Protocol.md) defines the questions, fairness rules, controlled stress tests, sources, limitations, and threats to validity.

The notebook studies **trade-offs** in safety, transparency, reproducibility, workflow complexity, flexibility, and runtime. It does not claim that AIDatasetKit is universally better than scikit-learn, and it does not make clinical, privacy, deployment, or causal claims.

## Execute interactively

The notebook uses `ucimlrepo` to download public datasets at runtime. This interactive dependency is intentional and is **not** part of the core test suite.

```bash
pip install -e .
pip install ucimlrepo jupyter
jupyter notebook examples/research/AIDatasetKit_Real_World_Comparative_Study.ipynb
```

It records its package versions, seed, split policy, model choices, and public dataset identifiers. No source data is embedded in the notebook. Review each public-data license and cite the UCI sources before redistributing any data or derived material.

## Read before interpreting

Begin with [Getting Started](../../docs/getting-started.md). The lab uses current contracts described in [Facade](../../docs/facade.md), [Training and Comparison](../../docs/training-and-comparison.md), [Clustering](../../docs/clustering.md), [Dimensionality Reduction](../../docs/dimensionality-reduction.md), [Privacy](../../docs/privacy.md), and [Limitations](../../docs/limitations.md).
