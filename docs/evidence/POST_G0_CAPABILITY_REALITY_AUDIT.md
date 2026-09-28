# POST-G0 Capability Reality Audit — AIDatasetKit

```text
CURRENT_GATE                 = G0 closed; G1 audit
CURRENT_ACTIVITY             = main integration + capability reality audit (no implementation)
G0_STATUS                    = PASS (16/16)
G0_MAIN_INTEGRATION          = PASS
GATES_FULLY_PASSED           = 1
TOTAL_GATES                  = 13
OVERALL_CERTIFIED_PROGRESS   = 40.95%   (95 proven / 232 applicable, G0–G12)
AUDIT_GATE                   = PASS     (the audit is complete with evidence; G1–G12 are not)
READY_FOR_G1_IMPLEMENTATION  = YES      (updated after G0.1: P0-1 closed; was NO at audit time, §9)
NEXT_AUTHORIZED_ACTION       = owner authorisation of G1-W1 (G0.1 closed, §0)
```

**حدود هذا المستند:**

- هذا تدقيق فقط. لم يُصلَح أي عيب ولم يُكتب أي كود في المكتبة.
- استُخدمت اختبارات استكشافية مؤقتة في scratchpad الجلسة، ولم يُضَف أي منها إلى commit. نتائجها ووصف أوامرها في §8. حُذفت بعد حفظ نتائجها هنا.
- **الفحص على:** `main` = `9e82bd79f34c992dd1b183ac8bc3971c1eb3b8c6`، أي G0 المعتمد.
- **مصدر الحقيقة:** الكود والاختبارات. التوثيق لا يُعتد به إلا حين يطابقهما.
- **التصنيف:** يتبع سُلَّم §5 من التكليف حرفيًا.

---

## 0. Updates after the audit (append-only)

**تصحيح 1 — عدد الصفوف:**

```text
AUDIT_CAPABILITY_ROWS = 222
COMMIT_MESSAGE_237    = HISTORICAL_TYPO
AUDIT_DOCUMENTS       = CORRECT
```

- رسالة الـcommit `2dc8ece` تقول «237 capability rows». العدد الصحيح في جدول §5 هو 222 صفًا عبر G1–G12: 216 بندًا منطبقًا، و6 بين NOT_APPLICABLE وOUT_OF_SCOPE.
- المستندات كانت صحيحة من البداية، والنسبة `95/232` لم تتغير بسبب هذا الخطأ.
- لم يُعدَّل التاريخ (لا amend ولا rebase)؛ التصحيح هنا فقط.

**تحديث 2 — G0.1 (profiling performance hotfix):**

- **P0-1: CLOSED.** الإثبات في `docs/evidence/G0_1_PROFILING_PERFORMANCE_HOTFIX_REPORT.md`.
- **P0-2** (فاصل CSV): ما زال **OPEN**، ويبقى ضمن G1.
- **بنود G11:** انتقلت أربعة بنود من `MISSING` إلى `PARTIAL`: G11-02، G11-04، G11-11، G11-21. **لم ينتقل أي بند إلى `SUPPORTED_AND_TESTED`**، لأن benchmark واحدًا وحراسة مسار واحد لا يغلقان فجوات الأداء.
- **النسب:** G11 بقي 5/21 = 23.81%، والنسبة الكلية بقيت 95/232 = **40.95%**.
- **G12-14:** خُفِّضت شدته من P0 إلى P1.

## 1. Main integration

| البند | القيمة |
|---|---|
| `origin/main` قبل الدمج | `59bb28639b01982e3a1fa882317e1054acb14bea` |
| `origin/g0-certification` | `9e82bd79f34c992dd1b183ac8bc3971c1eb3b8c6` |
| divergence | لا يوجد: `main...g0-certification` = `0 20`، و`main` سلف مباشر |
| طريقة الدمج | `git merge --ff-only origin/g0-certification` (كان `main` المحلي عنده أصلًا: "Already up to date") |
| push | `59bb286..9e82bd7  main -> main`، push عادي بلا `+` وبلا force |
| `origin/main` بعد الدمج | `9e82bd79f34c992dd1b183ac8bc3971c1eb3b8c6` = SHA المعتمد |
| CI على main | **run 36337340274، success، 10/10**: https://github.com/mohammed0115/AIDatasetKit/actions/runs/36337340274 |
| الفرع `g0-certification` | باقٍ ولم يُحذف |
| شجرة العمل | نظيفة |

| Job | Job id | Result | Collected | Final line |
|---|---|---|---:|---|
| ubuntu-latest / build, install and smoke | 108670596729 | success | — | Release smoke test passed |
| ubuntu-latest / minimum | 108670596935 | success | 4113 | 4067 passed, 47 skipped |
| ubuntu-latest / minimum + extras | 108670597008 | success | 4135 | 4091 passed, 44 skipped, 1 warning |
| ubuntu-latest / reference | 108670596930 | success | 4113 | 4067 passed, 47 skipped |
| ubuntu-latest / reference + extras | 108670596907 | success | 4135 | 4091 passed, 44 skipped |
| windows-latest / build, install and smoke | 108670596996 | success | — | Release smoke test passed |
| windows-latest / minimum | 108670596977 | success | 4113 | 4068 passed, 46 skipped |
| windows-latest / minimum + extras | 108670596943 | success | 4135 | 4092 passed, 43 skipped, 1 warning |
| windows-latest / reference | 108670596986 | success | 4113 | 4068 passed, 46 skipped |
| windows-latest / reference + extras | 108670596988 | success | 4135 | 4092 passed, 43 skipped |

```text
G0_MAIN_INTEGRATION = PASS
```

---

## 2. Repository reality

- **الحزمة:** 90 وحدة Python في 12 حزمة فرعية. `core` هي القاعدة، وفوقها statistics ثم profiling ثم preprocessing ثم models ثم training وevaluation وprediction ثم visualization ثم evidence، وفي الأعلى facade وcli. الاتجاه مفروض باختبار: `tests/integration/test_architecture_boundaries.py`.
- **الاختبارات:** 4113 اختبارًا (4135 مع extras). تمر على Windows وLinux، على الحد الأدنى وعلى البيئة المرجعية.
- **القارئ الوحيد للملفات:** `aidatasetkit/cli/main.py::_audit`، عبر `pd.read_csv(path)` لامتداد `.csv` فقط.
- **Python API:** يقبل `pandas.DataFrame` فقط.
- **لا توجد** حزمة ingestion، ولا حزمة تحويلات عامة، ولا وحدة trends أو اتجاهات زمنية، ولا دعم لعدة datasets.

## 3. Public API map

| الحزمة | الرموز العامة | الدور |
|---|---:|---|
| `aidatasetkit` (الجذر، lazy) | 17 | `AIDataFacade`, `KitConfig`، الأنواع الأساسية |
| `core` | 52 | الأنواع، الإعداد، الأخطاء، `detect_column_kinds`، `to_float_array`، provenance |
| `statistics` | 13 | `StatisticsEngine` (27 طريقة)، `FrequencyTable`، `correlation` (pearson/spearman/kendall)، `covariance`، moments |
| `profiling` | 7 | `DataProfiler`، `DataQualityInspector` (12 فحصًا)، `TaskDetector` |
| `preprocessing` | 25 | Planner/Builder/transformers مرتبطة بقدرات النموذج |
| `models` | 7 | Registry وFactory: 9 مصنِّفات، 9 regressors، 6 clusterers |
| `training` | 9 | Trainer، Comparator، `split_rows`، ClusteringRunner |
| `evaluation` | 15 | مقاييس التصنيف والانحدار والتجميع |
| `prediction` | 2 | `predict_frame` |
| `visualization` | 15 | Advisor، Service، 9 أنواع رسوم، renderer لـmatplotlib (اختياري) |
| `evidence` | 34 | AuditBuilder، canonical JSON، fingerprints، HTML report، `publish_run`/`read_current` |
| `facade` | 2 | `AIDataFacade`، `Stage` |
| `cli` | 3 | `aidatasetkit audit` |

## 4. Dependency map

- **النواة:** numpy ≥1.26.4، pandas ≥2.1.4، scipy ≥1.11.4، scikit-learn ≥1.6.1.
- **الاختيارية:**
  - `viz`: matplotlib ≥3.9.0.
  - `boosting`: xgboost ≥2.0.0، lightgbm ≥4.0.0، catboost ≥1.2.
  - `dev`: pytest ≥8.0.0، build، twine.
- **ما لا توجد بين الاعتماديات:** openpyxl، pyarrow، PyYAML، SQLAlchemy، أي مكتبة لقراءة المستندات، وأي عميل شبكة. هذا يؤكد من جهة الاعتماديات أن Excel وParquet وYAML وقواعد البيانات **غير موجودة**، ويؤكد أن النواة **لا تتصل بالشبكة**.

## 5. Capability map (الجدول المرجعي)

**كيف تُقرأ الأعمدة:**

- "Code Evidence" يشير إلى الملف والدالة.
- "Test Evidence" يشير إلى ملف اختبار موجود، أو إلى سكربت استكشافي في هذه الجلسة (§8).
- وجود اسم دالة وحده لم يُعتبر دليلًا.
- كل صف `PARTIAL` يذكر ما ينقصه تحديدًا.

| ID | Gate | Capability | Classification | Code Evidence | Test Evidence | Gap | Severity |
|---|---|---|---|---|---|---|---|
| G1-01 | G1 | CSV, comma-delimited, UTF-8 (CLI) | `SUPPORTED_AND_TESTED` | cli/main.py::_audit `pd.read_csv(path)` | tests/integration/test_audit_end_to_end.py (TestTheConsoleCommand, golden) | — | — |
| G1-02 | G1 | CSV delimiter detection (`;`, tab) | `MISSING` | cli/main.py: `read_csv` with pandas' default `,` | probe_ingestion.py (exploratory): `a;b;label` -> 1 column, exit 0 | Silent misread: audit succeeds on a wrong table | P0 |
| G1-03 | G1 | TSV | `MISSING` | cli/main.py: suffix check refuses `.tsv` | probe_ingestion.py (exploratory): exit 1 'not supported yet' | No reader | P1 |
| G1-04 | G1 | XLSX | `MISSING` | no reader | probe_ingestion.py (exploratory): refused | No reader | P1 |
| G1-05 | G1 | XLS | `MISSING` | no reader | probe_ingestion.py (exploratory): refused | No reader | P2 |
| G1-06 | G1 | Parquet | `MISSING` | no reader | probe_ingestion.py (exploratory): refused | No reader | P1 |
| G1-07 | G1 | Feather / Arrow tables | `MISSING` | no reader; pyarrow not a dependency | probe_ingestion.py (exploratory): refused | No reader | P2 |
| G1-08 | G1 | JSON | `MISSING` | no reader | probe_ingestion.py (exploratory): refused | No reader | P1 |
| G1-09 | G1 | JSONL | `MISSING` | no reader | probe_ingestion.py (exploratory): refused | No reader | P1 |
| G1-10 | G1 | XML | `MISSING` | no reader | probe_ingestion.py (exploratory): refused | No reader | P3 |
| G1-11 | G1 | YAML | `MISSING` | no reader; PyYAML not a dependency | probe_ingestion.py (exploratory): refused | No reader | P3 |
| G1-12 | G1 | pandas DataFrame in memory (Python API) | `SUPPORTED_AND_TESTED` | DataProfiler.profile, DataQualityInspector.inspect, AIDataFacade.load | tests/unit/test_profiler.py, test_quality.py, integration/test_facade.py | — | — |
| G1-13 | G1 | list of dicts / records / iterables | `MISSING` | profiler.py:95 raises ValidationError for non-DataFrame | probe_profile_perf.py: `ValidationError: A pandas DataFrame is required, got list.` | No adapter | P1 |
| G1-14 | G1 | SQLite | `MISSING` | no reader | — | No reader | P2 |
| G1-15 | G1 | PostgreSQL / SQLAlchemy / query results | `MISSING` | no reader; no DB dependency | — | No reader; connection handling undecided | P2 |
| G1-16 | G1 | Document extraction (TXT, Markdown, HTML, PDF, DOCX) | `OUT_OF_SCOPE` | no reader, by design of a tabular library | probe_ingestion.py (exploratory): all refused | Belongs upstream (MWIE/connectors): extract to a table first | — |
| G1-17 | G1 | Non-UTF-8 encodings / encoding detection | `MISSING` | read_csv without `encoding=` | probe_ingestion.py (exploratory): latin-1 -> `UnicodeDecodeError`, exit 1 | No detection, no encoding option, raw error text | P1 |
| G1-18 | G1 | UTF-8 with BOM | `SUPPORTED_NOT_SUFFICIENTLY_TESTED` | pandas strips the BOM | probe_ingestion.py (exploratory): header `a` read correctly; no test | Behaviour untested | P3 |
| G1-19 | G1 | Duplicate header detection | `SUPPORTED_AND_TESTED` | cli/main.py raw-header check | tests/integration/test_audit_end_to_end.py::test_duplicate_headers_are_refused_rather_than_renamed | — | — |
| G1-20 | G1 | Empty input | `PARTIAL` | pandas EmptyDataError surfaces as exit 1; header-only accepted | probe_ingestion.py (exploratory): empty -> `EmptyDataError: No columns to parse` | No library error type; header-only file produces an audit of zero rows | P2 |
| G1-21 | G1 | Malformed rows | `PARTIAL` | pandas ParserError surfaces via CLI catch-all | probe_ingestion.py (exploratory): `ParserError: ... Expected 2 fields in line 3` | Raw pandas message; no row-level report, no tolerant mode | P1 |
| G1-22 | G1 | Type inference incl. dates from text | `PARTIAL` | core/schema.py detects DATETIME only for datetime64 dtype | probe_ingestion.py (exploratory): ISO dates in CSV -> `categorical`, dtype object | Dates never inferred from files | P1 |
| G1-23 | G1 | Row/column counts and memory estimate | `SUPPORTED_AND_TESTED` | DatasetProfile.row_count/column_count/memory_usage_bytes | tests/unit/test_profiler.py | — | — |
| G1-24 | G1 | Excel sheets | `MISSING` | no spreadsheet reader | — | No reader | P2 |
| G1-25 | G1 | Large files: chunking / streaming | `MISSING` | whole file read into memory | — | No chunked path | P1 |
| G1-26 | G1 | Resource limits (size, rows, columns) | `MISSING` | none on input (G0-05 of the original plan not done) | — | Oversized input can exhaust memory | P1 |
| G1-27 | G1 | Unsupported format refused with a clear message | `SUPPORTED_AND_TESTED` | cli/main.py suffix check | tests/integration/test_audit_end_to_end.py (not really parquet) | — | — |
| G2-01 | G2 | Missing values per column and dataset | `SUPPORTED_AND_TESTED` | profiler.py _profile_column; checks.check_missing_values | test_profiler.py, test_quality.py | — | — |
| G2-02 | G2 | Duplicate rows | `SUPPORTED_AND_TESTED` | profiler._count_duplicate_rows; check_duplicate_rows | test_profiler.py, test_quality.py | — | — |
| G2-03 | G2 | Unique counts, cardinality, high cardinality | `SUPPORTED_AND_TESTED` | profiler._count_distinct; check_high_cardinality | test_profiler.py, test_quality.py | — | — |
| G2-04 | G2 | Min / max / mean / median | `SUPPORTED_AND_TESTED` | NumericSummary via StatisticsEngine | test_profiler.py, test_statistics_engine.py, test_extreme_values.py | — | — |
| G2-05 | G2 | Mode / dominant value | `SUPPORTED_AND_TESTED` | profiler._dominant; StatisticsEngine.mode | test_profiler.py, test_counting.py, test_statistics_engine.py | — | — |
| G2-06 | G2 | Standard deviation | `SUPPORTED_AND_TESTED` | StatisticsEngine.std | test_statistics_engine.py, test_extreme_values.py | — | — |
| G2-07 | G2 | Quantiles | `SUPPORTED_AND_TESTED` | StatisticsEngine.quartiles/percentile | test_statistics_engine.py | — | — |
| G2-08 | G2 | Distribution shape (skew, kurtosis, bins) in the profile | `PARTIAL` | engine has skewness/kurtosis; profile carries none | test_statistics_engine.py (engine only) | Not part of the profile | P2 |
| G2-09 | G2 | Zero / negative value counts | `MISSING` | — | probe: no field, no finding | Not measured | P2 |
| G2-10 | G2 | Invalid values against a domain rule | `MISSING` | only infinities and numbers-as-text | — | No rule mechanism | P2 |
| G2-11 | G2 | Infinities | `SUPPORTED_AND_TESTED` | profiler._count_infinite; check_infinite_values | test_quality.py | — | — |
| G2-12 | G2 | Constant and near-constant columns | `SUPPORTED_AND_TESTED` | check_constant_columns, check_near_constant_columns | test_quality.py | — | — |
| G2-13 | G2 | Outliers (IQR, z-score) | `SUPPORTED_AND_TESTED` | check_outliers | test_quality.py | — | — |
| G2-14 | G2 | Text lengths | `MISSING` | — | probe: no field | Not measured | P2 |
| G2-15 | G2 | Empty strings | `MISSING` | '' counted as a present value | probe: `txt` ['', 'abc', ''] -> missing 0, no finding | Blank text invisible to quality | P2 |
| G2-16 | G2 | Datetime range / frequency / gaps | `MISSING` | DATETIME kind only; `numeric` is None | probe: datetime column -> no range | Not measured | P1 |
| G2-17 | G2 | Multicollinearity (VIF) | `SUPPORTED_AND_TESTED` | check_multicollinearity | test_quality.py (Multicollinearity + IsRobust) | — | — |
| G2-18 | G2 | Structured findings with severity and review flag | `SUPPORTED_AND_TESTED` | core/types.QualityIssue; evidence FindingEvidence | test_quality.py, test_evidence_artifact.py | — | — |
| G2-19 | G2 | Affected rows and ratios in every finding | `PARTIAL` | counts/ratios in some details (missing, outliers, duplicates) | test_quality.py | No uniform affected-row contract | P2 |
| G2-20 | G2 | Evidence payload with redaction | `SUPPORTED_AND_TESTED` | evidence/builder.py _SAFE_TEXT_DETAIL_KEYS | test_evidence_artifact.py::test_every_text_key_is_either_vocabulary_or_redacted | — | — |
| G2-21 | G2 | Numbers stored as text, identifier-like columns | `SUPPORTED_AND_TESTED` | check_numeric_stored_as_text, check_id_like_columns | test_quality.py, test_profiler.py | — | — |
| G2-22 | G2 | Target-aware checks (leakage, imbalance) | `SUPPORTED_AND_TESTED` | check_target_leakage, check_class_imbalance | test_quality.py, integration tests | — | — |
| G3-01 | G3 | Missing-value handling | `PARTIAL` | preprocessing imputers inside a fitted model pipeline | test_preprocessing_builder.py | Produces a model matrix, not a cleaned table | P1 |
| G3-02 | G3 | Duplicate removal | `PARTIAL` | PreprocessingConfig drop of exact target duplicates | test_preprocessing_plan.py | No general de-duplication operation | P1 |
| G3-03 | G3 | Type conversion | `PARTIAL` | NumericTextConverter / casters in the pipeline | test_preprocessing_regressions.py | Pipeline-only | P1 |
| G3-04 | G3 | Date normalisation | `MISSING` | — | — | None | P1 |
| G3-05 | G3 | Numeric normalisation / scaling | `PARTIAL` | NumericScaler in the pipeline | test_preprocessing_builder.py | Pipeline-only | P2 |
| G3-06 | G3 | String normalisation | `MISSING` | — | — | None | P2 |
| G3-07 | G3 | Category normalisation | `PARTIAL` | CategoricalCaster, ExplicitMappingEncoder | test_preprocessing_builder.py | Pipeline-only | P2 |
| G3-08 | G3 | Column rename | `MISSING` | — | — | None | P2 |
| G3-09 | G3 | Filtering | `MISSING` | — | — | None | P1 |
| G3-10 | G3 | Sorting | `MISSING` | — | — | None | P3 |
| G3-11 | G3 | Joins and merges | `MISSING` | — | — | None | P1 |
| G3-12 | G3 | Group-by | `MISSING` | — | — | None | P1 |
| G3-13 | G3 | Aggregation | `MISSING` | — | — | None | P1 |
| G3-14 | G3 | Pivot / reshape | `MISSING` | — | — | None | P2 |
| G3-15 | G3 | Derived columns | `MISSING` | — | — | None | P1 |
| G3-16 | G3 | Source immutability | `SUPPORTED_AND_TESTED` | every layer copies or reads | tests '*caller_frame*' in preprocessing, facade, visualization | — | — |
| G3-17 | G3 | Transformation history (lineage) | `SUPPORTED_AND_TESTED` | FittedPreprocessor.lineage(); evidence FeatureLineage | test_audit_end_to_end.py (lineage), test_preprocessing_closure.py | Covers preprocessing only | — |
| G3-18 | G3 | Reversibility | `NOT_APPLICABLE` | not promised by any API | — | — | — |
| G4-01 | G4 | Univariate descriptive summary | `SUPPORTED_AND_TESTED` | StatisticsEngine.describe | test_statistics_engine.py | — | — |
| G4-02 | G4 | Frequency tables | `SUPPORTED_AND_TESTED` | statistics/frequency.FrequencyTable | test_frequency.py | — | — |
| G4-03 | G4 | Moments, skewness, kurtosis | `SUPPORTED_AND_TESTED` | statistics/moments.py, engine | test_moments.py | — | — |
| G4-04 | G4 | Correlation (pearson, spearman, kendall) | `SUPPORTED_AND_TESTED` | statistics/bivariate.correlation | test_bivariate.py | — | — |
| G4-05 | G4 | Covariance | `SUPPORTED_AND_TESTED` | statistics/bivariate.covariance | test_bivariate.py | — | — |
| G4-06 | G4 | Correlation matrix | `SUPPORTED_AND_TESTED` | visualization/preparation heatmap delegating to statistics | test_visualization_preparation.py::test_values_agree_with_the_statistics_package | — | — |
| G4-07 | G4 | Categorical association (contingency, chi-square, Cramér's V) | `MISSING` | — | — | None | P2 |
| G4-08 | G4 | Multivariate summaries | `PARTIAL` | VIF, correlation matrix | test_quality.py | No multivariate summary API | P2 |
| G4-09 | G4 | Statistical comparison tests (t, Mann-Whitney, ...) | `MISSING` | — | — | None | P2 |
| G4-10 | G4 | Input validation and degenerate-case refusal | `SUPPORTED_AND_TESTED` | statistics/guards.py | test_statistics_engine.py, test_bivariate.py | — | — |
| G4-11 | G4 | Deterministic output | `SUPPORTED_AND_TESTED` | pure functions | test_statistics_engine.py, determinism tests | — | — |
| G4-12 | G4 | Programmatic output (dataclasses, to_dict) | `SUPPORTED_AND_TESTED` | DescriptiveSummary, FrequencyTable.to_dict/to_frame | test_statistics_engine.py, test_frequency.py | — | — |
| G4-13 | G4 | Human-readable EDA output | `PARTIAL` | HTML report shows profile and findings | test_evidence_artifact.py | No EDA/statistics report | P2 |
| G5-01 | G5 | Period-over-period comparison | `MISSING` | — | — | None | P1 |
| G5-02 | G5 | Absolute change | `MISSING` | — | — | None | P1 |
| G5-03 | G5 | Percentage change | `MISSING` | — | — | None | P1 |
| G5-04 | G5 | Rolling metrics | `MISSING` | — | — | None | P2 |
| G5-05 | G5 | Moving averages | `MISSING` | — | — | None | P2 |
| G5-06 | G5 | Time aggregation (resample) | `MISSING` | — | — | None | P1 |
| G5-07 | G5 | Trend detection | `MISSING` | — | — | None | P2 |
| G5-08 | G5 | Anomaly candidates in time | `MISSING` | outlier check is not time-aware | — | None | P2 |
| G5-09 | G5 | Seasonality indicators | `MISSING` | — | — | None | P3 |
| G5-10 | G5 | Dataset A vs B | `MISSING` | only dataset fingerprints differ/equal | — | No comparison report | P1 |
| G5-11 | G5 | Segment A vs B | `PARTIAL` | visualization grouped_box / grouped_bar preparation | test_visualization_regressions.py (grouped) | Visual only; no numeric comparison | P2 |
| G5-12 | G5 | Before vs after | `MISSING` | — | — | None | P2 |
| G5-13 | G5 | Actual vs target | `MISSING` | — | — | None | P2 |
| G5-14 | G5 | Division by zero and missing periods | `MISSING` | — | — | None | P1 |
| G5-15 | G5 | Forecasting | `MISSING` | not implemented, not claimed (README: time series not supported) | — | None | P3 |
| G6-01 | G6 | Categorical segmentation (group rows by a column) | `MISSING` | — | — | None | P1 |
| G6-02 | G6 | Numeric binning | `MISSING` | histogram bins are internal to visualization | — | No binning API | P2 |
| G6-03 | G6 | Rule-based segmentation | `MISSING` | — | — | None | P2 |
| G6-04 | G6 | Clustering | `SUPPORTED_AND_TESTED` | models/clustering (6 clusterers), training/clustering.ClusteringRunner | test_clustering_catalog.py, test_clustering_end_to_end.py, test_clustering_metrics.py | — | — |
| G6-05 | G6 | Group profiling | `PARTIAL` | ClusteringResult.cluster_sizes | test_clustering_end_to_end.py | No per-segment profile | P2 |
| G6-06 | G6 | Segment comparison | `MISSING` | — | — | None | P2 |
| G6-07 | G6 | Normalisation (z-scores) | `SUPPORTED_AND_TESTED` | StatisticsEngine.z_scores | test_statistics_engine.py | — | — |
| G6-08 | G6 | Scaling as a standalone primitive | `PARTIAL` | NumericScaler only inside the pipeline | test_preprocessing_builder.py | Not exposed as a primitive | P2 |
| G6-09 | G6 | Percentile ranks | `MISSING` | percentile() returns values, not ranks | — | None | P2 |
| G6-10 | G6 | Weighted metrics | `PARTIAL` | FrequencyTable.weighted_mean | test_frequency.py | Only a weighted mean of a frequency table | P2 |
| G6-11 | G6 | Composite metrics | `MISSING` | — | — | None | P2 |
| G6-12 | G6 | Deterministic scoring primitives | `MISSING` | — | — | None | P1 |
| G7-01 | G7 | Joins | `MISSING` | — | — | None | P1 |
| G7-02 | G7 | Key validation | `MISSING` | — | — | None | P1 |
| G7-03 | G7 | One-to-one / one-to-many validation | `MISSING` | — | — | None | P1 |
| G7-04 | G7 | Schema compatibility between datasets | `PARTIAL` | prediction validates columns against training; schema fingerprints | test_facade.py, test_evidence_fingerprint.py | Train/predict only; no general contract | P1 |
| G7-05 | G7 | Required columns | `PARTIAL` | target / id_column presence checks | test_facade.py | No declared schema | P1 |
| G7-06 | G7 | Optional columns | `MISSING` | — | — | None | P2 |
| G7-07 | G7 | Expected types | `MISSING` | — | — | None | P1 |
| G7-08 | G7 | Relationship discovery | `MISSING` | — | — | None | P3 |
| G7-09 | G7 | Cross-dataset aggregation | `MISSING` | — | — | None | P2 |
| G7-10 | G7 | Cross-dataset comparison | `MISSING` | — | — | None | P2 |
| G7-11 | G7 | Malformed records report | `MISSING` | — | — | None | P2 |
| G7-12 | G7 | Empty inputs across datasets | `PARTIAL` | facade refuses an empty train or test frame | test_facade.py | Single-dataset only | P2 |
| G7-13 | G7 | Structured, actionable validation errors | `PARTIAL` | typed exceptions with messages | test_exceptions.py | No machine-readable codes or field paths | P2 |
| G8-01 | G8 | Bar chart | `SUPPORTED_AND_TESTED` | ChartType.BAR; VisualizationService.bar | test_visualization_service.py | — | — |
| G8-02 | G8 | Line chart | `MISSING` | no ChartType | — | None | P1 |
| G8-03 | G8 | Scatter | `SUPPORTED_AND_TESTED` | ChartType.SCATTER | test_visualization_service.py | — | — |
| G8-04 | G8 | Histogram | `SUPPORTED_AND_TESTED` | ChartType.HISTOGRAM | test_visualization_service.py | — | — |
| G8-05 | G8 | Box plot | `SUPPORTED_AND_TESTED` | ChartType.BOX_PLOT | test_visualization_service.py | — | — |
| G8-06 | G8 | Correlation heatmap | `SUPPORTED_AND_TESTED` | ChartType.CORRELATION_HEATMAP | test_visualization_preparation.py | — | — |
| G8-07 | G8 | Target distribution | `SUPPORTED_AND_TESTED` | ChartType.TARGET_DISTRIBUTION | test_visualization_advisor.py | — | — |
| G8-08 | G8 | Time series | `MISSING` | datetime columns are not charted (advisor) | test_visualization_advisor.py::test_datetime_columns_are_not_charted_in_this_version | None | P1 |
| G8-09 | G8 | Category comparisons (grouped bar/box) | `SUPPORTED_AND_TESTED` | ChartType.GROUPED_BAR / GROUPED_BOX | test_visualization_regressions.py | — | — |
| G8-10 | G8 | Missingness chart | `SUPPORTED_AND_TESTED` | ChartType.MISSING_VALUES | test_visualization_service.py | — | — |
| G8-11 | G8 | Rendering to an image file | `PARTIAL` | renderer returns a matplotlib Figure | test_visualization_renderer.py (extras jobs) | No save/export API | P2 |
| G8-12 | G8 | JSON report (audit.json) | `SUPPORTED_AND_TESTED` | evidence canonical_json | test_evidence_serialization.py, golden tests | — | — |
| G8-13 | G8 | Markdown report | `MISSING` | — | — | None | P2 |
| G8-14 | G8 | HTML report | `SUPPORTED_AND_TESTED` | evidence/report.render_report | test_evidence_artifact.py, golden report test | — | — |
| G8-15 | G8 | CSV exports | `MISSING` | only to_frame() DataFrames | — | No writer | P2 |
| G8-16 | G8 | Excel export | `MISSING` | — | — | None | P3 |
| G8-17 | G8 | PDF export | `MISSING` | — | — | None | P3 |
| G8-18 | G8 | Charts embedded in or referenced by a report | `MISSING` | HTML report has no charts | — | None | P2 |
| G8-19 | G8 | Report consistency (HTML from the same payload as JSON) | `SUPPORTED_AND_TESTED` | render_report(payload) | test_evidence_artifact.py::test_the_report_leaks_nothing_the_json_does_not | — | — |
| G8-20 | G8 | Limitations in reports | `SUPPORTED_AND_TESTED` | KNOWN_LIMITATIONS in artifact | test_capability_fingerprint_migration.py (scope sentence) | — | — |
| G8-21 | G8 | Evidence in reports | `SUPPORTED_AND_TESTED` | findings, decisions, lineage | test_evidence_artifact.py | — | — |
| G8-22 | G8 | Deterministic serialisation | `SUPPORTED_AND_TESTED` | canonical_json | golden tests; determinism across interpreters | — | — |
| G8-23 | G8 | Atomic publication | `SUPPORTED_AND_TESTED` | evidence/publication.publish_run/read_current | test_publication.py (Windows and Linux CI) | — | — |
| G9-01 | G9 | One structured artifact for a dataset | `PARTIAL` | AuditArtifact.to_dict: dataset, columns, findings, decisions, lineage, verdict, limitations | test_evidence_artifact.py | No statistics, trends, segments, comparisons or visualization sections | P1 |
| G9-02 | G9 | Bounded size | `SUPPORTED_NOT_SUFFICIENTLY_TESTED` | size grows with columns (16-29 KB measured) | probe_profile_perf.py; no size test | No declared bound | P2 |
| G9-03 | G9 | Determinism | `SUPPORTED_AND_TESTED` | canonical JSON, fingerprints | golden tests, 3-seed determinism run | — | — |
| G9-04 | G9 | Schema version | `SUPPORTED_AND_TESTED` | ARTIFACT_SCHEMA_VERSION | test_evidence_artifact.py | — | — |
| G9-05 | G9 | Provenance | `SUPPORTED_AND_TESTED` | environment versions, created_at, config | test_provenance.py, test_evidence_artifact.py | — | — |
| G9-06 | G9 | Evidence references (finding -> column, rule, numbers) | `SUPPORTED_AND_TESTED` | FindingEvidence | test_evidence_artifact.py | — | — |
| G9-07 | G9 | Limitations | `SUPPORTED_AND_TESTED` | KNOWN_LIMITATIONS, warnings | test_evidence_artifact.py | — | — |
| G9-08 | G9 | Raw sensitive data excluded by default | `SUPPORTED_AND_TESTED` | redact_values default; --include-values opt-in | test_evidence_artifact.py (redaction) | — | — |
| G9-09 | G9 | Usable without an LLM | `SUPPORTED_AND_TESTED` | plain JSON and HTML | every test | — | — |
| G9-10 | G9 | No automatic external transmission | `SUPPORTED_AND_TESTED` | no network code (scan) | test_evidence_artifact.py::test_it_is_a_standalone_page_with_no_external_requests | — | — |
| G9-11 | G9 | Machine-checkable schema (JSON Schema) for the context | `MISSING` | docs/audit-artifact.md is prose | — | None | P2 |
| G9-12 | G9 | Visualization plan in the context | `PARTIAL` | VisualizationPlan.to_dict exists separately | test_visualization_advisor.py (serialisation) | Not part of the artifact | P2 |
| G10-01 | G10 | Source dataset identity | `SUPPORTED_AND_TESTED` | dataset_fingerprint | test_evidence_fingerprint.py | — | — |
| G10-02 | G10 | Source columns (lineage) | `SUPPORTED_AND_TESTED` | FeatureLineage | test_audit_end_to_end.py (lineage) | — | — |
| G10-03 | G10 | Configuration record | `SUPPORTED_AND_TESTED` | config fingerprint and settings | test_capability_fingerprint_migration.py | — | — |
| G10-04 | G10 | Transformation history | `PARTIAL` | preprocessing lineage | lineage tests | No history for general operations (none exist) | P2 |
| G10-05 | G10 | Operation, parameters, result record | `PARTIAL` | audit stage record; TrainingResult/ComparisonResult.to_dict | test_training_and_comparison.py | No uniform run record across APIs | P2 |
| G10-06 | G10 | Warnings | `SUPPORTED_AND_TESTED` | artifact.warnings | test_audit_end_to_end.py | — | — |
| G10-07 | G10 | Limitations | `SUPPORTED_AND_TESTED` | KNOWN_LIMITATIONS | test_evidence_artifact.py | — | — |
| G10-08 | G10 | Hashes and run ids | `SUPPORTED_AND_TESTED` | fingerprints; publication run_id and manifest | test_publication.py | — | — |
| G10-09 | G10 | Single error hierarchy | `SUPPORTED_AND_TESTED` | core/exceptions.AIDatasetKitError | test_exceptions.py | — | — |
| G10-10 | G10 | Unsupported-format error | `PARTIAL` | CLI message + exit 1 | tests/integration/test_audit_end_to_end.py | No exception type for library callers | P2 |
| G10-11 | G10 | Malformed-input error | `PARTIAL` | raw pandas ParserError via CLI catch-all | probe_ingestion.py (exploratory) | Not a library error | P1 |
| G10-12 | G10 | Schema errors | `SUPPORTED_AND_TESTED` | SchemaError | test_facade.py, test_training_and_comparison.py | — | — |
| G10-13 | G10 | Data-quality blocking (BLOCKED verdict) | `SUPPORTED_AND_TESTED` | evidence/policy.decide_verdict; facade refuses | test_facade.py, test_audit_end_to_end.py | — | — |
| G10-14 | G10 | Resource-limit error | `MISSING` | — | — | None | P1 |
| G10-15 | G10 | Configuration errors | `SUPPORTED_AND_TESTED` | ConfigurationError | test_config.py | — | — |
| G10-16 | G10 | Internal-error classification | `PARTIAL` | CLI catch-all prints type name, exit 1 | tests/integration/test_audit_end_to_end.py | No internal-error category | P3 |
| G10-17 | G10 | Path traversal (published names, CURRENT run id) | `SUPPORTED_AND_TESTED` | publication._FILE_NAME/_RUN_ID | test_publication.py (traversal cases) | — | — |
| G10-18 | G10 | No unsafe deserialisation | `SUPPORTED_NOT_SUFFICIENTLY_TESTED` | scan: no pickle/eval/yaml.load | no guard test | Unguarded against regression | P3 |
| G10-19 | G10 | Archives | `NOT_APPLICABLE` | no archive input | — | — | — |
| G10-20 | G10 | Temporary files cleaned (staging) | `SUPPORTED_AND_TESTED` | publication._discard | test_publication.py::_no_debris | — | — |
| G10-21 | G10 | Spreadsheet formulas | `NOT_APPLICABLE` | no spreadsheet output | probe: formula text inert in HTML | Becomes applicable with CSV/Excel export | — |
| G10-22 | G10 | No external links in the report | `SUPPORTED_AND_TESTED` | standalone HTML | test_it_is_a_standalone_page_with_no_external_requests | — | — |
| G10-23 | G10 | Oversized inputs | `MISSING` | — | — | None | P1 |
| G10-24 | G10 | Malformed files | `PARTIAL` | refused with pandas' message | probe_ingestion.py (exploratory) | No hardened reader | P1 |
| G10-25 | G10 | Sensitive data in logs | `SUPPORTED_NOT_SUFFICIENTLY_TESTED` | NullHandler; logs carry exc_info only | no test | Unverified | P3 |
| G10-26 | G10 | Secret handling | `NOT_APPLICABLE` | the library handles no secrets | — | — | — |
| G10-27 | G10 | HTML escaping | `SUPPORTED_AND_TESTED` | report._escape (html.escape) | test_evidence_artifact.py:505 | — | — |
| G10-28 | G10 | No network calls | `SUPPORTED_NOT_SUFFICIENTLY_TESTED` | scan: no urllib/requests/socket/http | no guard test | Unguarded against regression | P2 |
| G10-29 | G10 | No LLM integration / telemetry | `SUPPORTED_NOT_SUFFICIENTLY_TESTED` | scan: none | no guard test | Unguarded against regression | P2 |
| G10-30 | G10 | Value redaction, opt-in disclosure | `SUPPORTED_AND_TESTED` | AuditBuilder(redact_values=True) default | test_evidence_artifact.py | — | — |
| G10-31 | G10 | Offline deterministic core | `SUPPORTED_AND_TESTED` | no I/O beyond given paths | determinism tests | — | — |
| G11-01 | G11 | Benchmark suite | `MISSING` | — | — | None | P1 |
| G11-02 | G11 | Small / medium / large timings on record | `PARTIAL` | G0.1: 10k and 100k timings committed in G0_1_PROFILING_PERFORMANCE_HOTFIX_REPORT.md | scripts/benchmark_profiling.py | No 1M or large-file record; not per release | P2 |
| G11-03 | G11 | Ingestion time | `MISSING` | — | — | None | P2 |
| G11-04 | G11 | Profiling time | `PARTIAL` | G0.1: counting hot path guarded; 100k x 20 = 1.50 s (fd4cdd2 1.08 s, 79e7cc8 16.39 s) | test_counting_performance.py (call growth, ratio to pandas) | No guard on whole-frame profiling time | P2 |
| G11-05 | G11 | Report time | `MISSING` | measured here only | probe_profile_perf.py | None | P2 |
| G11-06 | G11 | Memory usage | `MISSING` | measured here only (tracemalloc) | probe_profile_perf.py | None | P2 |
| G11-07 | G11 | Chunking | `MISSING` | — | — | None | P1 |
| G11-08 | G11 | Sampling where budgets demand it | `SUPPORTED_AND_TESTED` | visualization sampling; VIF row cap | test_visualization_preparation.py, test_quality.py | — | — |
| G11-09 | G11 | Timeouts / resource limits | `MISSING` | — | — | None | P1 |
| G11-10 | G11 | Graceful degradation | `SUPPORTED_AND_TESTED` | VIF declines past 50 columns; silhouette row limit | test_quality.py, test_clustering_metrics.py | — | — |
| G11-11 | G11 | Deterministic benchmarks | `PARTIAL` | scripts/benchmark_profiling.py (seeded, warm-up, median, digest) | run by hand, recorded in the G0.1 report | Not run in CI | P2 |
| G11-12 | G11 | CI platforms (Windows, Linux) | `SUPPORTED_AND_TESTED` | .github/workflows/ci.yml | runs 36331952431, 36337340274 (10/10) | — | — |
| G11-13 | G11 | Dependency matrix (minimum, reference, extras) | `SUPPORTED_AND_TESTED` | constraints/*.txt | CI 8 cells; test_dependency_contract.py | — | — |
| G11-14 | G11 | Package build and install | `SUPPORTED_AND_TESTED` | scripts/release_smoke_test.sh | CI smoke jobs | — | — |
| G11-15 | G11 | Release readiness | `PARTIAL` | 0.1.0a1 unreleased; smoke passes | test_packaging.py | Not published; install hints point to an index | P3 |
| G11-16 | G11 | macOS | `MISSING` | not in the matrix | — | Not run | P2 |
| G11-17 | G11 | Python 3.13 / 3.14 | `MISSING` | not in the matrix | — | Not run | P2 |
| G11-18 | G11 | Warnings clean (-W error) | `MISSING` | one external warning seen (matplotlib 3.9.0) | — | Not enforced | P3 |
| G11-19 | G11 | Lint | `MISSING` | no tool configured | — | None | P3 |
| G11-20 | G11 | Type checking | `PARTIAL` | py.typed shipped, annotations present | — | No type check run | P2 |
| G11-21 | G11 | Performance regression guard | `PARTIAL` | test_counting_performance.py | fails on 79e7cc8 (576,002 extra calls; ratio 65-75), passes on the fix, all CI cells | Counting/profiling path only | P1 |
| G12-01 | G12 | Q1 Structured opportunity data can be analysed | `PARTIAL` | profiling, quality, statistics on a DataFrame | G2/G4 tests | Needs G1 records adapter and G3 group-by | P1 |
| G12-02 | G12 | Q2 History and pattern discovery | `MISSING` | no time trends | — | G5 | P1 |
| G12-03 | G12 | Q3 Successful vs unsuccessful comparison | `PARTIAL` | classification training, leakage and imbalance on an outcome label | test_training_and_comparison.py | No descriptive segment comparison (G5/G6) | P1 |
| G12-04 | G12 | Q4 Required-skills analysis | `MISSING` | no multi-valued text handling | — | Needs explode/tokenise upstream or in G3 | P2 |
| G12-05 | G12 | Q5 Deterministic scoring inputs without an LLM | `PARTIAL` | z-scores, statistics | test_statistics_engine.py | No composite/percentile primitives (G6) | P1 |
| G12-06 | G12 | Q6 Revenue, cost, margin trends | `MISSING` | no derived columns or time aggregation | — | G3 + G5 | P1 |
| G12-07 | G12 | Q7 Professional profile data once tabular | `PARTIAL` | profiling works on any table | G2 tests | Needs G1 records adapter | P2 |
| G12-08 | G12 | Q8 Content performance | `MISSING` | no time trends or comparisons | — | G5 | P2 |
| G12-09 | G12 | Q9 Charts and reports for a command centre | `PARTIAL` | chart plans, HTML audit | G8 tests | No line/time-series, no chart export, no dashboard format | P1 |
| G12-10 | G12 | Q10 AI-ready structured context | `PARTIAL` | audit artifact | G9 tests | Missing statistics/trends/segments sections | P1 |
| G12-11 | G12 | Q11 Results traceable to evidence | `SUPPORTED_AND_TESTED` | findings, lineage, fingerprints | test_evidence_artifact.py | — | — |
| G12-12 | G12 | Q12 Works without sending data to an LLM | `SUPPORTED_AND_TESTED` | no network code | test_it_is_a_standalone_page_with_no_external_requests | — | — |
| G12-13 | G12 | Q13 Several datasets safely | `MISSING` | one frame per call | — | G7 | P1 |
| G12-14 | G12 | Q14 Realistic data sizes | `PARTIAL` | measured, see Performance; P0-1 closed in G0.1 | probe_profile_perf.py, G0.1 report | No limits, no chunking | P1 |
| G12-15 | G12 | Q15 What must stay in Masari / MWIE | `NOT_APPLICABLE` | boundary definition, not a capability | docs/MASARI_INTEGRATION.md | — | — |

## 6. Format support matrix

| الصيغة | القراءة | الدليل |
|---|---|---|
| CSV (`,`، UTF-8، مع BOM أو بدونه) | `SUPPORTED_AND_TESTED` عبر CLI (BOM: `SUPPORTED_NOT_SUFFICIENTLY_TESTED`) | e2e، probe |
| CSV بفاصل `;` أو tab | **يُقرأ خطأً دون تحذير**، كعمود واحد، وexit 0 | probe: `['a;b;label']` |
| CSV بترميز غير UTF-8 | يرفض بـ`UnicodeDecodeError` خام | probe |
| TSV، XLSX، XLS، Parquet، Feather، JSON، JSONL، XML، YAML | `MISSING`: يرفضها CLI برسالة «not supported yet» | probe |
| DataFrame في الذاكرة | `SUPPORTED_AND_TESTED` | tests |
| list of dicts / records | `MISSING`: `ValidationError` | probe |
| SQLite، PostgreSQL، SQLAlchemy | `MISSING` | — |
| TXT، MD، HTML، PDF، DOCX | `OUT_OF_SCOPE`: استخراج قبل المكتبة | — |

## 7. Output support matrix

| المخرَج | الحالة | الدليل |
|---|---|---|
| `audit.json` (canonical، مُرقَّم بإصدار، deterministic) | `SUPPORTED_AND_TESTED` | golden، determinism |
| `lineage.json` | `SUPPORTED_AND_TESTED` | e2e |
| `report.html` (مستقل، مُهرَّب، بلا شبكة) | `SUPPORTED_AND_TESTED` | test_evidence_artifact |
| `manifest.json` و`CURRENT` (نشر ذري) | `SUPPORTED_AND_TESTED` | test_publication، CI على Linux وWindows |
| `VisualizationPlan.to_dict()` (JSON) | `SUPPORTED_AND_TESTED` | test_visualization_advisor |
| رسم matplotlib | `PARTIAL`: يعيد `Figure` ولا يصدّر ملفًا | test_visualization_renderer |
| `to_frame()` (DataFrame) | موجود، لكنه ليس تصديرًا إلى ملف | — |
| Markdown، CSV، Excel، PDF | `MISSING` | — |

## 8. Exploratory evidence

السكربتات مؤقتة، شُغّلت على `main` = `9e82bd7` محليًا، وحُذفت بعد تسجيل نتائجها هنا.

**Ingestion probe.** أمر CLI حقيقي في عملية فرعية لكل ملف: `aidatasetkit audit <file> --output <dir> --fail-on never`، ثم قراءة الأعمدة عبر `read_current`.

| المدخل | exit | الأعمدة المقروءة | stderr |
|---|---:|---|---|
| csv comma utf-8 | 0 | `a`, `b`, `label` | — |
| csv semicolon | 0 | `a;b;label` | — |
| tsv | 1 | — | error: .tsv is not supported yet. The alpha reads CSV only. |
| tab-separated named .csv | 0 | `a\tb` (one column) | — |
| csv latin-1 | 1 | — | error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 10: invalid continuation byte |
| csv utf-8 BOM | 0 | `a`, `b` | — |
| empty file | 1 | — | error: EmptyDataError: No columns to parse from file |
| header only | 0 | `a`, `b` | — |
| malformed row (extra field) | 1 | — | error: ParserError: Error tokenizing data. C error: Expected 2 fields in line 3, saw 3 |
| iso dates in csv | 0 | `d`, `v` | — |
| xlsx | 1 | — | error: .xlsx is not supported yet. The alpha reads CSV only. |
| parquet | 1 | — | error: .parquet is not supported yet. The alpha reads CSV only. |
| json | 1 | — | error: .json is not supported yet. The alpha reads CSV only. |
| jsonl | 1 | — | error: .jsonl is not supported yet. The alpha reads CSV only. |
| xml | 1 | — | error: .xml is not supported yet. The alpha reads CSV only. |
| yaml | 1 | — | error: .yaml is not supported yet. The alpha reads CSV only. |
| feather | 1 | — | error: .feather is not supported yet. The alpha reads CSV only. |
| txt | 1 | — | error: .txt is not supported yet. The alpha reads CSV only. |
| pdf | 1 | — | error: .pdf is not supported yet. The alpha reads CSV only. |
| docx | 1 | — | error: .docx is not supported yet. The alpha reads CSV only. |
| html | 1 | — | error: .html is not supported yet. The alpha reads CSV only. |
| md | 1 | — | error: .md is not supported yet. The alpha reads CSV only. |
| xls | 1 | — | error: .xls is not supported yet. The alpha reads CSV only. |
| db | 1 | — | error: .db is not supported yet. The alpha reads CSV only. |

**Profiling probe:**

- `DataProfiler().profile([{"a": 1}, {"a": 2}])` → `ValidationError: A pandas DataFrame is required, got list.`
- عمود `datetime64` → kind `datetime`، و`numeric` = None، فلا مدى ولا تردد.
- عمود نصي `['', 'abc', '']` → missing = 0، ولا finding.
- عمود فيه `-1.0, 0.0, 5.0` → لا عدّ للأصفار أو القيم السالبة.
- إطار صغير → لا findings.

**Security probe:** اسم عمود `<img src=x onerror=alert(1)>` يظهر في HTML مُهرَّبًا (`&lt;img src=x`)، ولا يظهر الوسم الخام. `=HYPERLINK(...)` يظهر نصًا خاملًا.

**Performance probe:** `DataProfiler` → `DataQualityInspector` → `AuditBuilder` → `canonical_json`، بأعمدة نصفها أرقام عشوائية ونصفها فئات من 8 قيم، على Windows 11 وPython 3.12.3 وpandas 2.3.3، بكود `main`:

| الحجم | إطار (MB) | profile (s) | quality (s) | artifact (s) | ذروة Python (MB، tracemalloc) | حجم artifact (KB) |
|---|---:|---:|---:|---:|---:|---:|
| 10000x21 | 5.9 | 8.35 | 1.13 | 0.65 | 5.2 | 28.6 |
| 100000x21 | 58.8 | 67.4 | 1.53 | 4.57 | 25.5 | 28.6 |
| 1000000x11 | 298.0 | 298.15 | 2.47 | 15.64 | 245.8 | 16.4 |

**Regression probe:** الـprofile نفسه لـ100k صف × 20 عمودًا، بالمفسّر نفسه، على شجرتين معزولتين (`PYTHONPATH` يشير إلى كل شجرة):

| Commit | 100k × 20 | عمود رقمي واحد كل قيمه متميزة | عمود فئوي بـ8 قيم |
|---|---:|---:|---:|
| `fd4cdd2` (قبل إصلاح ترتيب المتعادلات) | **2.34 s** | 0.07 s | 0.133 s |
| `79e7cc8` (إصلاح ترتيب المتعادلات، موجود في `main`) | **21.48 s** | 2.15 s | 0.102 s |

## 9. Gaps

**العدّ:** الجدول يحتوي على مستوى الصفوف P0 = 3، وP1 = 55، وP2 = 64، وP3 = 15. صفوف G12 تكرر فجوات بوابات أخرى، لذلك تُجمَع أدناه **فجوات متميزة**.

### P0 (متميزتان)

1. **P0-1: تراجع أداء الـprofiling بنحو 9× (و30× على الأعمدة المتميزة القيم).** **الحالة: CLOSED في G0.1** (انظر §0 و`G0_1_PROFILING_PERFORMANCE_HOTFIX_REPORT.md`).
   - أدخله `79e7cc8`، وهو إصلاح ترتيب المتعادلات الذي أغلق G0.
   - `core/counting._ties_in_order_of_appearance` يمر على كل الصفوف في حلقة Python، ويستدعي `counted.index[i]` و`counted.iloc[i]` لكل عنصر.
   - في عمود رقمي متصل كل القيم متعادلة، فيصبح كل عمود من هذا النوع مسار O(n) بطيئًا.
   - لم يلتقطه أي اختبار، لعدم وجود حماية أداء (G11).
   - معايير G0 الـ16 لم تشمل الأداء، لذلك يبقى G0 = PASS بمعاييره. لكن هذا عيب حقيقي في الكود المعتمد، ويُسجَّل هنا ولا يُخفى.
2. **P0-2: قراءة CSV خاطئة بصمت.**
   - ملف مفصول بـ`;` أو tab ويحمل امتداد `.csv` يُقرأ عمودًا واحدًا، ويُنتج audit كاملًا بـexit 0.
   - لمكتبة غرضها إنتاج الأدلة، هذا دليل خاطئ دون أي إشارة.
   - يقع داخل نطاق G1.

### P1 (أهم الفجوات المتميزة)

- **G1:**
  - لا قارئ سوى CSV (TSV، XLSX، Parquet، JSON/JSONL)، ولا records adapter.
  - لا ترميز غير UTF-8، ولا دعم للأسطر المشوهة برسالة منظمة.
  - لا استنتاج للتواريخ من النص، ولا حدود موارد، ولا chunking.
- **G2:** لا ملف وصف للتواريخ (range، frequency، gaps).
- **G3:** لا عمليات تنظيف أو تحويل على الجداول (filter، group-by، aggregate، join، derived columns). الموجود مرتبط بخط أنابيب النموذج ويُخرج مصفوفة.
- **G5:** لا شيء زمني ولا مقارنات بين datasets.
- **G6:** لا segmentation بالأعمدة، ولا scoring primitives.
- **G7:** لا joins، ولا تحقق من المفاتيح أو الأنواع أو المخطط.
- **G8:** لا line chart ولا time series.
- **G9:** الـcontext الحالي لا يحتوي statistics أو trends أو segments أو comparisons.
- **G10:** لا خطأ لحدود الموارد، والمدخلات المشوهة تظهر كأخطاء pandas خام.
- **G11:** لا benchmarks، ولا حماية ضد تراجع الأداء، ولا timeouts أو حدود.

### P2 و P3

كما في الجدول:

- **P2** (عينة): الأصفار والقيم السالبة، أطوال النص، السلاسل الفارغة، اختبارات الارتباط الفئوي، تصدير Markdown وCSV، تصدير الرسوم، percentile ranks وcomposite metrics، JSON Schema للـcontext، macOS، Python 3.13 و3.14، type checking.
- **P3:** lint، و`-W error`، وتلميحات التثبيت التي تشير إلى index، وXML وYAML.

## 10. Contradictions between docs and code

| # | التوثيق | الواقع | الأثر |
|---|---|---|---|
| C1 | `README.md:306` و`docs/limitations.md:69`: «Datetime columns are profiled» | الـprofile لا يسجّل للتاريخ إلا النوع والعدد والقيم المفقودة؛ لا مدى ولا تردد. وأي CSV لا يُنتج عمود تاريخ أصلًا (probe) | ادعاء أوسع من الواقع (P2) |
| C2 | `docs/limitations.md:48`: «The CLI reads CSV» | صحيح، لكن لا شيء يقول إن الفاصل يجب أن يكون `,` والترميز UTF-8، والبدائل تُقرأ خطأً بصمت | P0-2 غير موثق |
| C3 | تقرير G0 النهائي: «لا regression جديد» و«A tally with no ties is returned untouched» | الجملة الثانية صحيحة. لكن الإصلاح أبطأ الـprofiling 9× ولم يُقَس | تصحيح يُسجَّل هنا؛ التقرير لم يدّعِ قياس أداء |
| C4 | `AIDatasetKit_ARCHITECTURE_OVERVIEW.md` يصف تصميمًا أوسع («original design, kept as written») | مُعلَن صراحة كتصميم لا كتنفيذ، وفيه status block | لا تناقض فعلي، لكن القارئ قد يخلط بينهما |

## 11. Duplicate authorities, dead capabilities, security and privacy

**Duplicate authorities:**

- لم يُعثر على سلطتين متنافستين. تصنيف الأنواع سلطة واحدة (`core.schema.detect_column_kinds`، ويعيد `FeatureDetector` استخدامه).
- الارتباط يُحسب في `statistics`، ويستدعيه visualization وleakage (مُختبَر: `test_the_package_contains_no_second_profiler`، `test_correlation_is_delegated_to_the_statistics_package`).
- العدّ يمر كله عبر `core.counting`.
- الاستثناء الوحيد أن VIF يحل `lstsq` بنفسه. هذا متعمَّد وموثق، وليس مصدرًا ثانيًا لقيمة موجودة.

**Dead or unreachable capabilities:**

- `core.types.RunMetadata` مُصدَّر في جذر الحزمة، ولا يستخدمه أي مسار في المكتبة؛ يستخدمه اختبار واحد فقط.
- `KitConfig.cv_folds` أُزيل في G0.
- لم يُعثر على غيرهما.

**Security:**

- لا `pickle` ولا `eval` ولا `exec` ولا `yaml.load` (فحص الكود).
- لا قراءة للأرشيفات ولا `tempfile`.
- HTML مُهرَّب، ومُختبَر.
- أسماء الملفات المنشورة ومعرّف الـrun محمية من path traversal، ومُختبَرة.
- **فجوات:** لا حدود لحجم المدخل (P1)؛ الملفات المشوهة تمر إلى pandas مباشرة (P1).
- لا يوجد اختبار حراسة يمنع لاحقًا إدخال استيراد شبكي أو إلغاء تسلسل غير آمن (P2/P3).
- الـlogging يستخدم `NullHandler`، ولا اختبار يثبت خلو السجلات من القيم (P3).

**Privacy:**

- لا شبكة، ولا LLM، ولا telemetry، ولا قراءة لمتغيرات البيئة (فحص الكود).
- تنقيح القيم مفعَّل افتراضيًا، والكشف عنها بـ`--include-values` فقط، ومُختبَر.
- التقرير HTML لا يطلب أي مورد خارجي، ومُختبَر.

## 12. Recommendation

| المجال | القرار | السبب |
|---|---|---|
| profiling، quality، statistics، evidence، publication، CI | **REUSE** | مُختبَرة على منصتين وبيئتين، ولها golden وdeterminism |
| visualization | **EXTEND** | أنواع الرسوم الموجودة مُختبَرة؛ ينقص line/time series والتصدير |
| ingestion | **EXTEND** (حزمة جديدة فوق `core`) | لا يوجد شيء لإعادة كتابته؛ CLI يستهلك الطبقة الجديدة لاحقًا |
| cleaning/transformation، trends، segmentation، multi-dataset | **EXTEND** | غير موجودة؛ تُبنى فوق `statistics` و`profiling` دون كسر الحدود |
| preprocessing/training | **REUSE** كما هي | ليست بديلًا عن التنظيف العام، ولا ينبغي تحميلها بذلك |
| `core.counting` | **REUSE**، مع إصلاح P0-1 | العقد صحيح، والتنفيذ بطيء |

لا شيء يستدعي **REBUILD** أو **EXTRACT**.

**الموجة التالية المقترحة: G0.1 hotfix (حجم S).** قبل أي كود في G1:

1. ترتيب المتعادلات بعملية vectorized، مثلًا `pd.factorize` مع `np.argsort(kind="stable")`، بالعقد نفسه، ودون تغيير أي fixture.
2. اختبار حراسة أداء: profile لـ100k × 20 تحت حد مُعلَن ومتسامح.
3. تشغيل CI.

**لماذا `READY_FOR_G1_IMPLEMENTATION = NO`:**

- الشروط الأربعة الأولى متحققة: G0 في main، وCI على main ناجح، وتدقيق G1 مكتمل، وفجوات G1 ومعايير قبولها محددة في الـroadmap.
- الشرط الخامس، «لا P0 معماري»، متحقق حرفيًا أيضًا، لأن P0-1 ليس معماريًا.
- مع ذلك، معايير قبول G1 تشمل سلوك الملفات الكبيرة وزمن الإدخال، ولا يمكن إثباتها فوق تراجع أداء بـ9× في المسار نفسه الذي يغذيه G1.
- لذلك التوصية: اعتماد G0.1 أولًا. قرار البدء في النهاية للمالك.

## 13. Masari integration questions

الأسئلة الخمسة عشر مصنّفة في الجدول (G12-01..15). الحدود المعمارية في `docs/MASARI_INTEGRATION.md`، والجدول الكامل للإجابات في §4 من ذلك الملف.

## 14. PROJECT PLAN PROGRESS

| Gate | المجال | Proven | Applicable | Certified % | الحالة | الخطوة التالية |
|---|---|---:|---:|---:|---|---|
| G0 | Baseline stability | 16 | 16 | 100.00% | PASS | مغلق |
| G0.1 | Profiling performance hotfix | — | — | — | PASS | P0-1 مغلق (انظر §0) |
| G1 | Ingestion | 5 | 26 | 19.23% | AUDIT | الموجة 1 بعد G0.1 (الـroadmap) |
| G2 | Profiling & Quality | 15 | 22 | 68.18% | AUDIT | تواريخ ونص وقيم |
| G3 | Cleaning & Transformation | 2 | 17 | 11.76% | AUDIT | عمليات جدولية |
| G4 | EDA & Statistics | 9 | 13 | 69.23% | AUDIT | ارتباط فئوي واختبارات |
| G5 | Trends & Comparisons | 0 | 15 | 0.00% | AUDIT | بعد G3 |
| G6 | Segmentation & Scoring | 2 | 12 | 16.67% | AUDIT | بعد G3 |
| G7 | Multi-dataset & Schema | 0 | 13 | 0.00% | AUDIT | بعد G1 وG3 |
| G8 | Visualization & Reporting | 15 | 23 | 65.22% | AUDIT | line/time series وتصدير |
| G9 | AI-ready Context | 8 | 12 | 66.67% | AUDIT | أقسام context |
| G10 | Provenance/Security/Privacy/Errors | 16 | 28 | 57.14% | AUDIT | أخطاء الموارد والمدخلات |
| G11 | Performance & Certification | 5 | 21 | 23.81% | AUDIT | 4 بنود MISSING→PARTIAL في G0.1؛ benchmarks في CI |
| G12 | Masari Consumer Integration | 2 | 14 | 14.29% | AUDIT | بعد G1–G9 |
| **All** | G0–G12 | **95** | **232** | **40.95%** | — | — |

**تفصيل الحالات لكل بوابة** (حتى لا يختفي العمل الموجود غير المكتمل):

| Gate | SUPPORTED_AND_TESTED | SUPPORTED_NOT_SUFFICIENTLY_TESTED | PARTIAL | MISSING | BLOCKED | NOT_APPLICABLE | OUT_OF_SCOPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| G1 | 5 | 1 | 3 | 17 | 0 | 0 | 1 |
| G2 | 15 | 0 | 2 | 5 | 0 | 0 | 0 |
| G3 | 2 | 0 | 5 | 10 | 0 | 1 | 0 |
| G4 | 9 | 0 | 2 | 2 | 0 | 0 | 0 |
| G5 | 0 | 0 | 1 | 14 | 0 | 0 | 0 |
| G6 | 2 | 0 | 3 | 7 | 0 | 0 | 0 |
| G7 | 0 | 0 | 4 | 9 | 0 | 0 | 0 |
| G8 | 15 | 0 | 1 | 7 | 0 | 0 | 0 |
| G9 | 8 | 1 | 2 | 1 | 0 | 0 | 0 |
| G10 | 16 | 4 | 6 | 2 | 0 | 3 | 0 |
| G11 | 5 | 0 | 6 | 10 | 0 | 0 | 0 |
| G12 | 2 | 0 | 7 | 5 | 0 | 1 | 0 |

```text
CURRENT_GATE                 = G0 closed; G1 audit complete
CURRENT_ACTIVITY             = audit report
G0_STATUS                    = PASS
G0_MAIN_INTEGRATION          = PASS
GATES_FULLY_PASSED           = 1
TOTAL_GATES                  = 13
OVERALL_CERTIFIED_PROGRESS   = 40.95%
READY_FOR_G1_IMPLEMENTATION  = YES after G0.1 (P0-1 closed; P0-2 is a G1 item)
NEXT_AUTHORIZED_ACTION       = owner authorisation of G0.1, then G1 wave 1
```

**التمييز المطلوب:**

- **IMPLEMENTED:** الصفوف `SUPPORTED_*` و`PARTIAL`.
- **TESTED:** `SUPPORTED_AND_TESTED`.
- **INTEGRATED:** ما يمر عبر facade أو CLI أو artifact.
- **VERIFIED:** ما مرّ في CI على المنصتين.
- **CERTIFIED:** G0 وحده.
- **MISSING** و**OUT_OF_SCOPE:** كما في الجدول.
