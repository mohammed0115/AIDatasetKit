# AIDatasetKit Real-World Comparative Research Protocol

## Objective
Compare a competent manual pandas/scikit-learn workflow with AIDatasetKit on real public datasets, focusing on safety, transparency, reproducibility, workflow burden, scientific correctness, flexibility, and runtime — not merely predictive performance.

## Public datasets

The interactive notebook may retrieve these public datasets through `ucimlrepo`; normal `pytest` must never make a network request. Each execution records the UCI ID, local shape, target interpretation, source URL, environment versions, seed, and model parameters.

| Task used in the study | Official dataset | UCI ID / DOI | Methodological interpretation | License |
|---|---|---|---|---|
| Classification and dimensionality reduction | Breast Cancer Wisconsin (Diagnostic) | [17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) / [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | `Diagnosis` is the benchmark target and `ID` is excluded. This is not a clinical study or diagnostic system. | CC BY 4.0 |
| Regression | Wine Quality | [186](https://archive.ics.uci.edu/dataset/186/wine+quality) / [10.24432/C56S3T](https://doi.org/10.24432/C56S3T) | UCI supports classification and regression; this protocol explicitly treats sensory `quality` as a numeric regression target. | CC BY 4.0 |
| Clustering | Wholesale customers | [292](https://archive.ics.uci.edu/dataset/292/wholesale+customers) / [10.24432/C5030X](https://doi.org/10.24432/C5030X) | Exclude UCI's `Region` target from the unsupervised feature matrix. Internal diagnostics are not proof of real customer segments. | CC BY 4.0 |
| Anomaly detection | Statlog (Shuttle) | [148](https://archive.ics.uci.edu/dataset/148/statlog+shuttle) / [10.24432/C5WS31](https://doi.org/10.24432/C5WS31) | Labels are absent during fitting and used only afterward as an external benchmark convention; a minority label is not a universal definition of anomaly. | CC BY 4.0 |

UCI reports 569 rows/30 features for Breast Cancer Wisconsin (Diagnostic), 4,898 rows/11 features for Wine Quality, 440 rows for Wholesale customers, and 58,000 rows for Statlog Shuttle.[1] [2] [3] [4]

## Central research question

> What changes when recurrent analytical responsibilities are moved from analyst-authored orchestration into explicit library contracts?

The study does **not** ask whether AIDatasetKit is universally better than scikit-learn. AIDatasetKit uses the scientific Python ecosystem; the credible comparison concerns trade-offs in safety, transparency, reproducibility, complexity, flexibility, and runtime.

## Research questions

| ID | Question | Discipline |
|---|---|---|
| RQ1 | Do matched workflows reach scientifically comparable results? | Match raw rows, split, target, model family, material configuration, and metrics. |
| RQ2 | Which workflow detects or responds to controlled safety defects? | Report blocks, reviews, false positives, false negatives, and manual code required. |
| RQ3 | Which makes preprocessing, lineage, evidence, and findings more inspectable? | Inspectability is not a claim of scientific truth. |
| RQ4 | Which records reproducibility inputs more completely? | Do not promise bit-identical results across untested environments. |
| RQ5 | How does orchestration burden differ? | Lines/components are descriptive, never a quality score. |
| RQ6 | What end-to-end runtime overhead is observed? | Include the work each workflow actually performs. |
| RQ7 | Where is manual composition more flexible? | Treat customization and bespoke workflows as valid advantages. |
| RQ8 | What does each approach make clearer to a learner? | Shorter code must not be confused with understanding. |

## Two layers
1. Clean-data parity.
2. Controlled safety stress tests on top of real published data.

## Stress tests
- target leakage,
- ID leakage,
- evaluation leakage,
- invalid infinity / numeric text,
- unknown categories,
- task contradiction.

## Advantages to test
- safety contracts,
- capability-driven preprocessing,
- lineage/evidence,
- reproducibility,
- reduced boilerplate,
- educational clarity,
- cautious unsupervised semantics.

## Disadvantages to test
- runtime overhead,
- reduced flexibility,
- opinionated policies,
- no CV/tuning core,
- dependence on correctness of library contracts,
- smaller algorithm surface,
- fingerprint/version migration complexity,
- no domain certification implied.

## Threats to validity

Because the study evaluates its own library, **author bias** is primary. Predefine experiments, preserve raw results, publish the notebook, report negative findings, and invite independent reproduction. Additional threats are manual-baseline quality, limited datasets and representativeness, domain validity, synthetic-perturbation realism, unsupervised metric comparability, AIDatasetKit/scikit-learn version dependence, default-policy dependence, learning-effect bias, and external validity.

No result implies medical approval, privacy certification, financial suitability, employment suitability, production readiness, causal explanation, or domain certification. Dimensionality reduction is not anonymization and an anomaly score/flag is not a probability, confidence, calibrated risk, or proof of cause.

## Clean-data parity design

Clean-data parity always comes before controlled safety tests. The conventional baseline must use `Pipeline`, `ColumnTransformer`, training/evaluation separation, correct imputation and encoding, scaling only where appropriate, deterministic seeds, and task-appropriate metrics. It must not be weakened or made leaky to favor the library.

| Dimension | Conventional expert workflow | AIDatasetKit workflow | Fairness criterion |
|---|---|---|---|
| Split | Explicit `train_test_split` | One shared documented split | Verify row positions when public APIs permit it. |
| Preprocessing | Manually composed per-model pipeline | Capability-driven plan and fitted preprocessor | Match declared roles and material preprocessing choices. |
| Model | Same family and material parameters | Canonical registered name | No CV or tuning on either side. |
| Metrics | Explicit target/positive-label/averaging policy | Evaluation report and comparison metric | Use the same target and evaluation rows. |
| Runtime | Full requested manual workflow | Full requested library workflow | Include checks, planning, lineage, and metadata when requested. |

A mismatch is a research result to investigate, not a reason to choose the prettier score. For clustering, internal metrics are diagnostics rather than semantic truth. For anomaly detection, score is not probability, confidence, calibrated risk, or cause. For reduction, components are not universally important features, latent truth, or anonymization.

## Controlled safety stress tests

Every stress case must state: **This defect was intentionally introduced for the experiment. It is not a defect claimed to exist in the original public dataset.**

| Experiment | Manual question | AIDatasetKit question | Report discipline |
|---|---|---|---|
| Exact target leakage | Is an explicit inspection/check present? | Is it detected, excluded, blocked, or reviewed? | Record outcome and error clarity. |
| ID leakage | Does analyst logic identify an identifier? | Does declared/heuristic ID handling respond? | Include false positives and exclusions. |
| Evaluation-only extreme | Is fitting limited to training rows? | Are learned preprocessing statistics training-fitted only? | Compare learned state, not just scores. |
| Infinity | Is explicit non-finite validation present? | Is the safety response project-owned and actionable? | Do not mislabel backend failure as a data diagnosis. |
| Numeric text | Is a deliberate conversion policy used? | Does current policy review/convert it explicitly? | Record policy and impact. |
| Unknown category | Is encoder behavior declared? | Does external transformation honor its contract? | Keep result reproducible. |
| Task contradiction | Does application code guard task/target use? | Does facade contract reject it? | Preserve actionable task-level error. |

The notebook publishes a safety outcome table. “Detected” alone is not better: false positives, false negatives, and reduced flexibility are equally reportable.

## Measurement, transparency, and evidence

Record Python, pandas, NumPy, SciPy, scikit-learn, AIDatasetKit, seed, split policy, model parameters, dataset IDs, local shapes, source URLs, sampling, and run time. Runtime is a within-environment sanity observation, not a cross-machine benchmark promise. Compare non-comment source lines, manually configured components, explicit safety checks, and manually assembled result objects only as descriptive burden measures; do not calculate an overall winner score.

Where current contracts expose them, inspect preprocessing plans, feature lineage, result metadata, split fingerprints, and evidence. Evidence and semantic fingerprints support reproducibility under the library contract; they do not prove a scientific conclusion and must not serialize raw rows or runtime estimators.

## Candidate strengths and costs

| Candidate observation | Counter-hypothesis or cost |
|---|---|
| Explicit leakage, ID, finiteness, and task boundaries | Policies may generate false positives or block unusual valid expert workflows. |
| Capability-driven preprocessing and feature lineage | Correctness depends on the model-capability and task-detection contracts. |
| Reproducibility metadata and evidence | Fingerprint migration creates versioning complexity and is not scientific proof. |
| Guided orchestration may reduce repeated boilerplate | The abstraction has a learning curve and cannot replace core analytical knowledge. |
| Cautious unsupervised semantics | No automatic winner is less convenient for exploratory work. |
| Stable comparison lifecycle | Current core deliberately has no CV orchestration, tuning, AutoML, or unrestricted custom-estimator surface. |

### When the conventional expert workflow is better

Manual composition is usually stronger for unusual preprocessing, custom scikit-learn estimators/metrics, bespoke feature engineering, research prototypes, cross-validation experiments, hyperparameter tuning, and uncommon scientific workflows. It is a legitimate advanced interface, not a workaround.

### When AIDatasetKit may help a learner

AIDatasetKit can make stages, safety findings, preprocessing plans, lineage, evidence, and results easier to inspect. Students must still learn pandas, scikit-learn, metrics, preprocessing, leakage, and statistics. The library should simplify composition, not replace understanding.

## Documentation, book, and paper bridge

Use [Getting Started](../getting-started.md) before this lab, then consult [facade](../facade.md), [training and comparison](../training-and-comparison.md), [clustering](../clustering.md), [dimensionality reduction](../dimensionality-reduction.md), [lineage](../lineage.md), [audit artifact](../audit-artifact.md), [privacy](../privacy.md), and [limitations](../limitations.md). The research lab can later support book-outline topics on quality, leakage, preprocessing, task families, reproducibility, evidence, and limitations; see [Book Outline](../book/OUTLINE.md).

Working title for a later paper:

> **Safety, Transparency, and Workflow Abstraction in Tabular Machine Learning: A Comparative Evaluation of AIDatasetKit and Conventional scikit-learn Pipelines**

A later paper may contain abstract, introduction, related work, system design, research questions, datasets, experiments, clean-data results, stress tests, runtime/workflow burden, discussion, threats to validity, limitations, and conclusion. This protocol is not that paper.

## References

[1] [UCI — Breast Cancer Wisconsin (Diagnostic), ID 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

[2] [UCI — Wine Quality, ID 186](https://archive.ics.uci.edu/dataset/186/wine+quality)

[3] [UCI — Wholesale customers, ID 292](https://archive.ics.uci.edu/dataset/292/wholesale+customers)

[4] [UCI — Statlog (Shuttle), ID 148](https://archive.ics.uci.edu/dataset/148/statlog+shuttle)

[5] [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
