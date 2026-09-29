# G1-W1 — Ingestion Foundation: تقرير الدليل

```text
AUTHORIZED_WAVE        = G1-W1
G1_W1_STATUS           = PASS (branch CI 10/10؛ main CI يُسجَّل في الرد النهائي بعد الـfast-forward)
                         → أعادته مراجعة CTO إلى CONDITIONAL_PASS بسبب إصدار الـschema؛ الإغلاق في §17
PRE_WAVE_MAIN_SHA      = 69303b35e30c8dda1b51490360e65bc22ff920ed
BRANCH                 = g1-w1-ingestion
TESTED_BRANCH_SHA      = 4d0eb6c2a23014ee7600a184224b51f8ca539052 (آخر commit كود/اختبار/وثائق قبل هذا التقرير)
POST_WAVE_SHA          = commit هذا التقرير (رأس الفرع عند الدمج؛ مسجل في الرد النهائي)
FINAL_MAIN_SHA         = يساوي POST_WAVE_SHA بعد fast-forward فقط
P0-2                   = CLOSED
G1                     = 10/26 = 38.46%   (كان 5/26 = 19.23%)
OVERALL                = 102/232 = 43.97% (كان 95/232 = 40.95%)
G1_W2_AUTHORIZATION    = NO_GO_PENDING_CTO_REVIEW
```

## 1. النطاق

- **منفّذ:** `aidatasetkit.ingestion` كسلطة القراءة الوحيدة: CSV وTSV وpandas.DataFrame وlist/tuple of mappings.
- **لم يُنفَّذ (بقرار):** اكتشاف الترميز التلقائي، حدود الموارد، استنتاج التواريخ، chunking، JSON/JSONL، Parquet/Feather، XLS/XLSX، SQLite/PostgreSQL، XML/YAML، URLs، HTTP، scraping، MWIE، Masari.
- `AIDataFacade.load()` لم يُوسَّع.
- لا اعتماد جديد: المكتبة تستخدم `csv` و`codecs` من stdlib، وpandas الموجود أصلاً.

## 2. Commits

| SHA | الرسالة |
|---|---|
| `5759522` | feat(ingestion): add typed table loading contracts |
| `b8065c3` | feat(ingestion): detect and validate delimited text; load frames and records |
| `6353985` | feat(evidence): record how the audited table was read |
| `ed91a89` | feat(cli): route table loading through ingestion |
| `2c46646` | fix(ingestion): validate an explicit delimiter against the content |
| `bd87aab` | test(ingestion): cover delimiters encodings and malformed inputs |
| `312668a` | perf(ingestion): benchmark load_table against a bare read_csv |
| `2c27926` | docs(ingestion): document what is read, how, and the code path |
| `4d0eb6c` | docs(audit): record G1-W1 in the capability audit and roadmap |

`2c46646` إصلاح لعيب وجدته أثناء التطوير: فاصل صريح `,` على ملف مفصول بـ`;` كان
يُقرأ عموداً واحداً، أي P0-2 نفسه عبر باب آخر. لم يُدمج قبل إصلاحه.

## 3. تصميم الـAPI

```python
from pathlib import Path
from aidatasetkit.ingestion import load_table, LoadOptions

loaded = load_table(Path("data.csv"), options=LoadOptions(encoding="cp1256", delimiter=None, header=True))
loaded.frame      # pandas.DataFrame
loaded.metadata   # LoadMetadata (frozen) -> .to_dict()
```

- **المصادر المقبولة:** `os.PathLike` (امتداد `.csv`/`.tsv`)، `pandas.DataFrame`، `list`/`tuple` of `Mapping`.
- **مرفوض:** `str` و`bytes` (غموض مسار/محتوى/رابط)؛ أي نوع آخر → `UnsupportedFormatError`.
- **`LoadOptions`:** frozen، يُتحقق منه عند الإنشاء (`InvalidIngestionOptionsError`). الأسماء البديلة (`utf8`, `latin1`, `iso-8859-1`) تُحوَّل إلى اسمها القانوني عبر `codecs.lookup`.
- **`LoadMetadata`:** `source_kind`, `format`, `encoding`, `delimiter`, `delimiter_source`, `header`, `row_count`, `column_count`, `memory_bytes`, `warnings`. ما لا ينطبق = `None`. لا مسار ولا قيم خلايا.
- **DataFrame:** يُعاد الكائن نفسه (`loaded.frame is frame`) دون تعديل، ويُرفض الفارغ والمكرر الأعمدة، ولا profiling.
- **records:** الأعمدة = اتحاد المفاتيح بترتيب أول ظهور؛ المفتاح الغائب = خلية ناقصة + تحذير بعدد السجلات؛ المفاتيح غير النصية والعناصر غير Mapping مرفوضة؛ المدخل لا يتغير.

## 4. خوارزمية اكتشاف الفاصل (`ingestion/delimited.py`)

1. **Probe** أول 64 KiB بايت: 0 بايت → `EmptyInputError`؛ NUL → `MalformedInputError` (ثنائي/UTF-16)؛ BOM مع `utf-8` → `utf-8-sig` + تحذير.
2. **Sample** أول `SAMPLE_CHARS = 64 KiB` حرف بعد الفك. إن لم تكن العيّنة الملف كله يُحذف آخر سجل (قد يكون مقطوعاً)، وخطأ اقتباس في نهايتها لا يُعد عيباً.
3. لكل فاصل من `, ; \t |`: `csv.reader(strict=True)` على العيّنة → عرض كل سجل غير فارغ. **مؤهَّل** = كل السجلات بعرض واحد ≥ 2.
4. القرار:
   - فاصل صريح: يُستخدم، إلا إذا كان الملف تحته عموداً واحداً وفاصل آخر مؤهَّل → `MalformedInputError`.
   - `.tsv`: tab مؤهَّل → `format`؛ tab عمود واحد وفاصل آخر مؤهَّل → `MalformedInputError`؛ tab عمود واحد فقط → عمود واحد + تحذير.
   - `.csv`: مؤهَّل واحد → `detected`؛ أكثر → `AmbiguousDelimiterError` بعدد الأعمدة لكل مرشح؛ الكل عمود واحد → عمود واحد + تحذير و`delimiter=None`؛ الكل مكسور الاقتباس → `MalformedInputError`؛ غير ذلك → `MalformedInputError` (حقول غير متسقة).
5. **Validate** تمريرة كاملة متدفقة بـ`csv` تحت الفاصل المختار: كل سجل بعرض السجل الأول (السطر الفارغ تماماً يُتجاهل كما في pandas؛ سطر مسافات في جدول متعدد الأعمدة يُرفض)، العناوين المكررة → `DuplicateHeadersError`، عنوان بلا صفوف → `EmptyInputError`، بايت لا يُفك → `EncodingError`.
6. **Parse** `pd.read_csv(sep, encoding, header)` مرة واحدة، ثم مطابقة عدد الصفوف والأعمدة مع خطوة التحقق؛ الاختلاف → `MalformedInputError`.

**الحتمية:** لا عشوائية ولا اعتماد على المنصة؛ ترتيب المرشحين ثابت (`SUPPORTED_DELIMITERS`). `test_detection_is_deterministic` واختبارات CI على Windows وLinux.

**الكلفة:** عيّنة محدودة + تمريرتان كاملتان (csv ثم pandas). انظر §10.

## 5. عقد الترميز

| الترميز | السلوك |
|---|---|
| `utf-8` | الافتراضي؛ مع BOM يُقرأ كـ`utf-8-sig` + تحذير |
| `utf-8-sig` | عند الطلب |
| `latin-1` | عند الطلب |
| `cp1256` | عند الطلب (عربي Windows) |
| أي ترميز آخر (UTF-16، cp1252، …) | `InvalidIngestionOptionsError` عند إنشاء `LoadOptions` |
| بايت لا يُفك | `EncodingError` مع offset، وليس `UnicodeDecodeError` |

`AUTO_ENCODING_DETECTION = NO`.

## 6. أنواع الأخطاء (`core/exceptions.py`)

كلها تحت `IngestionError(AIDatasetKitError)`، ومتوافقة مع الفئات القائمة:

| الخطأ | يرث أيضاً |
|---|---|
| `InputNotFoundError` | — |
| `UnsupportedFormatError` | — |
| `InvalidIngestionOptionsError` | `ConfigurationError` |
| `EncodingError` | — |
| `MalformedInputError` | — |
| `AmbiguousDelimiterError` | `MalformedInputError` |
| `EmptyInputError` | `EmptyDataError` |
| `DuplicateHeadersError` | `SchemaError` |

الـCLI: كل `IngestionError` → سطر واحد `error: TypeName: message` على stderr، exit 1، بلا traceback (إلا مع `--debug`)، ولا نشر.

## 7. الفرق الدلالي في الـartifact وترحيل الـfingerprint

- `AuditArtifact.ingestion: IngestionEvidence | None`، ومفتاح `"ingestion"` في `semantic_dict()` بعد `"dataset"`.
- `AuditBuilder.build(..., ingestion=LoadMetadata | None)`: يرفض (`EvidenceError`) metadata لا تطابق أعداد الإطار.
- `report.html`: قسم *Input* (يُحذف إن لم يوجد سجل).
- `schema_version` بقي `1.0` (المفتاح إضافي). **[مُصحَّح في §17: هذا الحكم خاطئ، والإصدار الآن `1.1`.]**
- **الـgolden:** الفرق الوحيد في `examples/audit_churn/`: سطر `"ingestion": null` في `expected_audit_semantic.json` و`expected_audit_semantic.pandas2.json` و`audit.json`، وسطر الـfingerprint في `audit.json` و`report.html`. القيمة `null` لأن الـgolden يُبنى من DataFrame عبر `AuditBuilder` مباشرة، لا عبر `load_table`. لا تغيّر في `lineage.json` ولا في config fingerprint ولا في أي قياس.
- **Fingerprint:** `0f0fd1d5e9785d46351d7d28dc15d2d4eda4cc7fbc54bf0509a2a76eb3b471a5` → `007838931330c91af0d430a3e90c8b5c98305f76d1f482a511302939d85e2c71`. خطوة `"G1-W1 ingestion record added to the artifact"` في `tests/unit/test_capability_fingerprint_migration.py` مع rollback (حذف المفتاح) يعيد البصمة السابقة حرفياً (`TestTheIngestionStep`).

## 8. الاختبارات المحلية

كل الأرقام من سجلات pytest في scratchpad الجلسة (الأمر: `python -m pytest -p no:cacheprovider -q <files>` مع `PYTHONPATH=.`).

**Focused** (`test_ingestion.py`, `test_cli_ingestion.py`, `test_capability_fingerprint_migration.py`, `test_counting_performance.py`, `test_architecture_boundaries.py`, `test_audit_end_to_end.py`):

| البيئة | Python | pandas | numpy | النتيجة |
|---|---|---|---|---|
| L (global) | 3.12.3 | 2.3.3 | 2.4.1 | 371 passed |
| MIN (`venv_min`) | 3.11.16 | 2.1.4 | 1.26.4 | 371 passed |
| V (`venv_verified`) | 3.12.3 | 3.0.5 | 2.5.2 | 371 passed |

- حراسات G0.1 (`tests/unit/test_counting_performance.py`) ضمن الـ371 ولم تُعدَّل.
- **عدد الاختبارات:** collect-only على `69303b3` = 4219، وعلى `4d0eb6c` = 4366؛ لا اختبار محذوف؛ الزيادة +147 = 112 (`test_ingestion.py`) + 19 (`test_cli_ingestion.py`) + 5 (layer لكل module جديد) + 2 (`TestTheIngestionStep`) + 9 (`test_exceptions.py` للأخطاء التسعة الجديدة).
- **الـsuite الكامل محلياً** (MIN، git archive لـ`4d0eb6c`): انظر الملحق §15.

## 9. CI

**الفرع:** run `36404737361` على `4d0eb6c` — **10/10 success**.

| Job | النتيجة | JUnit tests | failures | errors | skipped |
|---|---|---:|---:|---:|---:|
| ubuntu-latest / minimum | success | 4367 | 0 | 0 | 47 |
| ubuntu-latest / minimum + extras | success | 4388 | 0 | 0 | 44 |
| ubuntu-latest / reference | success | 4367 | 0 | 0 | 47 |
| ubuntu-latest / reference + extras | success | 4388 | 0 | 0 | 44 |
| windows-latest / minimum | success | 4367 | 0 | 0 | 46 |
| windows-latest / minimum + extras | success | 4388 | 0 | 0 | 43 |
| windows-latest / reference | success | 4367 | 0 | 0 | 46 |
| windows-latest / reference + extras | success | 4388 | 0 | 0 | 43 |
| ubuntu-latest / build, install and smoke | success | — | — | — | — |
| windows-latest / build, install and smoke | success | — | — | — | — |

JUnit يعدّ 4367 مقابل collect 4366، وهو الفرق نفسه (+1) المسجل في G0.1 (4220 مقابل 4219). الزيادة عن G0.1 في JUnit: 4220 → 4367 = +147.

**main:** بعد fast-forward؛ الـrun ID والنتيجة في الرد النهائي.

## 10. الأداء (§21)

`scripts/benchmark_ingestion.py 100000 5`: 100k صف × 10 أعمدة (7.9 MB)، warm-up ثم 5 تشغيلات متتالية، الوسيط؛ الذاكرة بـtracemalloc في تشغيل منفصل. الأساس: `pd.read_csv(path, sep=<الفاصل الصحيح>)`.

| pandas | الفاصل | read_csv | load_table | النسبة | peak read_csv | peak load_table |
|---|---|---:|---:|---:|---:|---:|
| 2.3.3 | `,` | 0.124 s | 0.480 s | 3.88× | 19.2 MB | 19.2 MB |
| 2.3.3 | `;` | 0.180 s | 0.526 s | 2.92× | 19.2 MB | 19.2 MB |
| 3.0.5 | `,` | 0.196 s | 0.594 s | 3.03× | 12.2 MB | 12.2 MB |
| 3.0.5 | `;` | 0.221 s | 0.601 s | 2.71× | 12.2 MB | 12.2 MB |

- **الذاكرة:** دون زيادة (1.00×): التحقق متدفق ولا يحتفظ بنسخة ثانية.
- **الزمن:** ~+0.35–0.40 s لكل 100k صف. السبب معروف: تمريرة `csv` الكاملة في Python، وهي الطريقة الوحيدة لرؤية الصف القصير الذي يملؤه pandas بـNaN بصمت. للمقارنة، profiling لـ100k × 20 في G0.1 يستغرق ثوانٍ، فالقراءة جزء صغير من الـaudit. **مُفسَّر ومقبول**؛ لا حراسة أداء جديدة للقراءة في هذه الموجة (G11-03 بقي `MISSING`).
- حراسات G0.1 دون تغيير وتمر.
- عيّنة الاكتشاف: 64 KiB حرف؛ probe: 64 KiB بايت.

## 11. الاختبارات العدائية (§19)

`scripts/ingestion_mutations.py`: كل تعديل يُطبَّق على نسخة من الشجرة (لا على الـworktree) ويُشغَّل الاختبار المسمّى.

| # | الإضعاف | النتيجة |
|---|---|---|
| 1 | `pd.read_csv` بالفاصل الافتراضي `,` | 2 failed |
| 2 | الفاصل من الامتداد فقط | 1 failed |
| 3 | اختيار صامت عند الغموض | 1 failed |
| 4 | ترك pandas يعيد تسمية المكرر | 2 failed |
| 5 | تمرير `UnicodeDecodeError` الخام | 1 failed |
| 6 | حذف metadata من الدليل | 2 failed |
| 7 | مسار مطلق في الـartifact | 1 failed |
| 8 | مخرجات جزئية قبل ingestion | 6 failed |

**8/8 قُتلت.** إضافة لذلك تُظهر الاختبارات السلوك الخاطئ مباشرة: `pd.read_csv` على ملف `;` يعطي `['a;b;label']`، وعلى عناوين مكررة يعطي `a.1`، وعلى صف قصير يقبله بـNaN.

## 12. الأمن والخصوصية (§20)

- `TestTheBoundary`: فحص AST لكل ملف في `aidatasetkit/ingestion/`: لا `socket`/`urllib`/`http`/`requests`/`httpx`/`pickle`/`shelve`/`marshal`/`subprocess`، ولا `eval`/`exec`/`compile`/`__import__`.
- لا شبكة ولا telemetry ولا LLM. `str` لا يُفسَّر كمسار أو رابط.
- `load_table` لا يكتب أي ملف (`test_it_writes_nothing`) ولا يُجري profiling (`test_it_does_not_profile`).
- رسائل الخطأ تستخدم `path.name` فقط، ولا تقتبس قيم خلايا (`TestTheErrorContract`).
- لا مسار مطلق في `audit.json` ولا `report.html` ولا `manifest.json` ولا `CURRENT` (`test_no_artifact_carries_the_absolute_path`، بأربعة أشكال للمسار).
- سجل ingestion بلا قيم خلايا (`test_the_ingestion_record_holds_no_cell_value`).
- الرفض لا ينشر شيئاً، والتشغيل السابق يبقى سليماً (`TestRefusalsPublishNothing`).
- لا force push ولا tag ولا release ولا PyPI. ملفات المالك غير المتتبعة لم تُلمس.

## 13. غير المنفّذ

```text
SUPPORTED_NOW              = CSV, TSV, DataFrame, records
EXPLICIT_ENCODINGS         = utf-8, utf-8-sig, latin-1, cp1256
AUTO_ENCODING_DETECTION    = NO
RESOURCE_LIMITS            = NOT_YET_IMPLEMENTED
DATE_INFERENCE             = NOT_YET_IMPLEMENTED
CHUNKING                   = NOT_YET_IMPLEMENTED
JSON_XLSX_PARQUET          = NOT_YET_IMPLEMENTED
```

## 14. P0-2 قبل وبعد

| | قبل (`69303b3`) | بعد |
|---|---|---|
| `a;b;label` في `.csv` | عمود واحد `a;b;label`، audit كامل، exit 0 | 3 أعمدة، `delimiter=";"`, `delimiter_source="detected"` في `audit.json` وقسم Input |
| tab في `.csv` | عمود واحد | عمودان |
| غامض (`,` و`;` متسقان) | يُقرأ بـ`,` بصمت | `AmbiguousDelimiterError`، exit 1، لا نشر |
| `--delimiter ,` على ملف `;` | (لا خيار) | `MalformedInputError` |

## 15. ملحق: الـsuite الكامل محلياً

- **الأمر:** `run_suite.sh g1w1_min 4d0eb6c venv_min/Scripts/python.exe`: `git archive` للـcommit في شجرة معزولة، ثم `pytest -p no:cacheprovider -q -rfE --junitxml`.
- **البيئة:** Windows 11 (10.0.26200)، locale cp1252، Python 3.11.16، numpy 1.26.4، pandas 2.1.4، scipy 1.11.4، scikit-learn 1.6.1، pytest 8.0.0؛ `import_path` داخل الشجرة المعزولة.
- **collect:** 4366 tests.
- **النتيجة:** `4321 passed, 46 skipped in 8159.74s (2:15:59)`، exit 0، لا FAILED ولا ERROR.
- 4321 + 46 = 4367، أي +1 عن الـcollect، وهو الفرق نفسه في JUnit على CI (§9).
- المدة الطويلة سببها ذاكرة الجهاز المحدودة (≈1.5 GB حرة)، لا الكود: الخلايا نفسها على CI استغرقت 2–6 دقائق.

## 16. PROJECT PLAN PROGRESS

| Gate | Proven | Applicable | Certified % | Status | التغيير في هذه الموجة |
|---|---:|---:|---:|---|---|
| G0 | 16 | 16 | 100.00% | PASS | — |
| G0.1 | — | — | — | PASS | — |
| G1 Ingestion | 10 | 26 | 38.46% | IN PROGRESS (W1 PASS) | +5 SAT (G1-02, 03, 18, 20, 21)؛ G1-13 وG1-17 → PARTIAL |
| G2 | 15 | 22 | 68.18% | AUDIT | — |
| G3 | 2 | 17 | 11.76% | AUDIT | — |
| G4 | 9 | 13 | 69.23% | AUDIT | — |
| G5 | 0 | 15 | 0.00% | AUDIT | — |
| G6 | 2 | 12 | 16.67% | AUDIT | — |
| G7 | 0 | 13 | 0.00% | AUDIT | — |
| G8 | 15 | 23 | 65.22% | AUDIT | — |
| G9 | 8 | 12 | 66.67% | AUDIT | — |
| G10 | 18 | 28 | 64.29% | AUDIT | +2 SAT (G10-11, G10-24) |
| G11 | 5 | 21 | 23.81% | AUDIT | — (G11-03 بقي MISSING) |
| G12 | 2 | 14 | 14.29% | AUDIT | — |
| **All** | **102** | **232** | **43.97%** | — | +7 |

محسوبة من جدول §5 في `POST_G0_CAPABILITY_REALITY_AUDIT.md` بعدّ الصفوف آلياً، لا يدوياً.

---

## 17. Addendum — G1-W1 artifact schema closure (append-only)

```text
PREVIOUS_FINAL_MAIN_SHA    = 90ecfae778a5983d03f4b4fa664327c62b92cff0
PREVIOUS_MAIN_CI           = 36421870660, 10/10
SCHEMA_DEFECT              = ingestion changed the artifact shape while schema_version stayed 1.0
CORRECTION                 = artifact schema 1.1 (minor: additive)
PUBLICATION_SCHEMA         = unchanged at 1.0
PACKAGE_VERSION_CHANGED    = NO
INGESTION_BEHAVIOR_CHANGED = NO
BRANCH                     = g1-w1-schema-closure
IMPLEMENTATION_SHA         = 21e62b4 (fix(evidence): bump the artifact schema to 1.1 …)
DOCS_SHA                   = cdca988
```

### 17.1 العيب

كتبتُ في §7 أن `schema_version` «بقي `1.0` (المفتاح إضافي)»، وكذلك في الـCHANGELOG وفي
docstring خطوة الترحيل. هذا خلاف العقد في `evidence/types.py` و`docs/audit-artifact.md`:
تغيّر شكل السجل يجب أن يُعرف من `schema_version` وحده. قبل التصحيح لم يكن يميّز
artifact الـG1-W1 عن السابق له إلا مقارنة البصمات. §7 وبقية النص أعلاه تُركت كما كُتبت،
وأُشير إلى التصحيح في موضعه.

### 17.2 التصحيح

- `ARTIFACT_SCHEMA_VERSION = "1.1"` مع سجل للإصدارات في تعليق الثابت. القاعدة: إضافة
  حقل أو حذفه أو تغيير معناه ترفع الإصدار؛ الإضافي minor، وغيره major.
- `lineage.json` يحمل الإصدار نفسه (`lineage_dict()` يستخدم `schema_version` ذاته،
  واختبار e2e يشترط التساوي)، فأصبح `1.1` أيضاً.
- `PUBLICATION_SCHEMA_VERSION` بقي `1.0` (اختبار جديد `test_the_publication_layout_is_versioned_separately`).
- لا تغيير في ingestion ولا الاكتشاف ولا الـAPI العام ولا profiling/quality/preprocessing ولا النشر الذري.

### 17.3 سلسلة الترحيل

| الخطوة | before | after | rollback |
|---|---|---|---|
| G1-W1 ingestion record added | `0f0fd1d5…b471a5` | `00783893…c2e71` | حذف `ingestion` |
| G1-W1 closure: schema 1.0 → 1.1 | `00783893…c2e71` | `3e93dd5e11b75f51d99c16bee263723627b4f93bebeb10c31e2bf31c92ed6be0` | `schema_version = "1.0"` |

- الخطوة السابقة لم تُعدَّل؛ الجديدة بعدها (`TestTheChain` يثبت الاتصال والعكس لكل خطوة).
- `TestTheSchemaStep`: إعادة الإصدار إلى `1.0` تعيد `00783893…` (الـartifact الوسيط الذي نشره `90ecfae`)؛ ثم حذف المفتاح يعيد `0f0fd1d5…`؛ والمفتاح الوحيد الذي يختلف بين الحالتين هو `schema_version`، أي أن القياسات والـfindings والـdecisions والـlineage لم تتحرك.
- `TestTheMulticollinearityStep::test_the_schema_version_did_not_move` كان يتحقق من الإصدار *الحالي*، فصار يتحقق منه *عند تلك الخطوة* بالرجوع في السلسلة. الادعاء نفسه (تلك الخطوة لم تغيّر الإصدار) ما زال مُثبَتاً.

### 17.4 الفرق الدلالي للـGolden (مقابل `90ecfae`)

مولَّد آلياً بمقارنة JSON عنصراً عنصراً (`git show 90ecfae:<file>` مقابل الملف الجديد):

| الملف | الفرق |
|---|---|
| `expected_audit_semantic.json` (pandas 3) | `/schema_version: "1.0" → "1.1"` فقط |
| `expected_audit_semantic.pandas2.json` | `/schema_version: "1.0" → "1.1"` فقط |
| `audit.json` | `/schema_version` و`/provenance/semantic_fingerprint` فقط |
| `lineage.json` | `/schema_version` فقط |
| `report.html` | سطران: خلية semantic fingerprint، وتذييل «Artifact schema 1.1». dataset/schema/config fingerprints متطابقة |

لا تغيّر في قيم profiling ولا أنواع الأعمدة ولا findings ولا verdict ولا decisions ولا ingestion metadata ولا dataset/config fingerprints.

### 17.5 ملاحظة إجرائية

أول commit للتصحيحين كُتب برسالة تبدأ بـBOM (PowerShell 5.1 `Set-Content -Encoding utf8`).
كانا محليين ولم يُدفعا؛ أُعيد إنشاؤهما بـ`git reset --soft 90ecfae` ورسائل بلا BOM، والشجرة
الناتجة مطابقة بايتاً ببايت (`git diff befd912 cdca988` فارغ). لا شيء مدفوع أُعيدت كتابته.

### 17.6 الاختبارات والـCI

```text
PRE_CLOSURE_MAIN_SHA       = 90ecfae778a5983d03f4b4fa664327c62b92cff0
FIRST_SCHEMA_FIX_SHA       = 21e62b4f3952bbc12a773c2ba64a10e1481c5242
PACKAGING_SMOKE_FIX_SHA    = e491eb6a422eff3a925436f7513a4155e2b434cf
FINAL_LOCAL_TESTED_SHA     = e491eb6a422eff3a925436f7513a4155e2b434cf
FINAL_MAIN_CODE_SHA        = e7db88c4b26f90c0495f440ac30f4a601f394601 (fast-forward only; later report-only commit is identified below)
PYTHON                     = 3.12.3
NUMPY                      = 2.5.2
PANDAS                     = 3.0.5
SCIPY                      = 1.18.0
SCIKIT_LEARN               = 1.9.0
PYTEST                     = 8.3.3
PACKAGE_VERSION            = 0.1.0a1 (unchanged)
ARTIFACT_SCHEMA            = 1.0 -> 1.1
PUBLICATION_SCHEMA         = 1.0 (unchanged)
FOCUSED_TESTS              = 736 passed (on cdca988; before the smoke-script-only follow-up)
FULL_PYTEST                = 4329 passed, 46 skipped, 0 failed, 0 errors; 424.79 s
JUNIT                      = 4375 tests, 0 failures, 0 errors, 46 skipped
BRANCH_CI                  = run 36534163130, SUCCESS, 10/10, SHA bd775acd65bcb9c67d441a5c76ace6cda4b4668a
FINAL_EVIDENCE_SHA         = e56df2c9fad9d3bedb4e54db5701771225be4104 (evidence-only follow-up)
FINAL_EVIDENCE_CI          = run 36534982952, SUCCESS, 10/10, 0 failures, 0 errors
FINAL_EVIDENCE_CI_URL      = https://github.com/mohammed0115/AIDatasetKit/actions/runs/36534982952
MAIN_CI                    = run 36536466851, SUCCESS, 10/10, SHA e7db88c4b26f90c0495f440ac30f4a601f394601
FINAL_REPORT_UPDATE        = documentation-only; exact resulting main SHA and its CI are returned in the final response
G1_W2                      = NOT STARTED; NO_GO_PENDING_CTO_REVIEW
```

- The candidate was tested from a separate detached worktree at the exact committed SHA above. CI-equivalent commands: `python -m pytest --collect-only -q` and `python -m pytest -rfE --junitxml=junit.xml`. Collection reported `4374 items / 1 skipped`; the full run and JUnit report agree on 4329 passed, 46 skipped, zero failures, and zero errors (4375 JUnit cases including the collection skip).
- The focused command covered artifact/evidence schemas, fingerprints and migration rollback, golden fixtures, serialization, publication, ingestion, packaging, G0.1 performance guards, architecture boundaries, audit end-to-end, and CLI ingestion: 736 passed. The full suite above was rerun after the packaging-smoke correction.
- `scripts/release_smoke_test.sh` initially failed because its installed-artifact assertion hard-coded schema `1.0`. The follow-up changes only that assertion to compare against exported `ARTIFACT_SCHEMA_VERSION == "1.1"`. The CI release smoke then passed for both wheel and sdist: build, twine metadata check, clean installs, CLI, audit, and published artifact checks. No distribution was published.
- Programmatic Golden diff against `90ecfae`: both semantic fixtures changed only `/schema_version`; `audit.json` changed only `/schema_version` and `/provenance/semantic_fingerprint`; `lineage.json` changed only `/schema_version`; HTML changed only the semantic fingerprint display and artifact-schema footer. Dataset, schema, config and plan fingerprints remained identical. The migration tests prove `3e93dd5e…` rolls back to `00783893…` by setting schema to `1.0`, then to `0f0fd1d5…` by removing `ingestion`.
- Branch CI run `36534163130` passed all 10 jobs on exact pushed SHA `bd775acd65bcb9c67d441a5c76ace6cda4b4668a`; final evidence-commit CI run `36534982952` passed all 10 jobs on exact SHA `e56df2c9fad9d3bedb4e54db5701771225be4104`; the subsequent report-only commit `e7db88c4b26f90c0495f440ac30f4a601f394601` also passed 10/10 in run `36535690233`. The branch was fast-forwarded to local `main` and pushed normally; main CI run `36536466851` passed 10/10 on exact SHA `e7db88c4b26f90c0495f440ac30f4a601f394601`. Each matrix was 2 package + 8 test jobs, with zero failures/errors. `gh auth status` reports no authenticated CLI host; these results were verified from public GitHub Actions pages. A final documentation-only update is being added after that verified main SHA; its resulting SHA/CI are reported separately.
- Remaining G1 gaps from the capability audit: G1-04 XLSX, G1-05 XLS, G1-06 Parquet, G1-07 Feather/Arrow, G1-08 JSON, G1-09 JSONL, G1-10 XML, G1-11 YAML, G1-14 SQLite, G1-15 PostgreSQL/query results, G1-24 Excel sheets, G1-25 chunked/streamed reads, and G1-26 resource limits remain missing; G1-13 still refuses generators/other iterables and is not wired through `AIDataFacade.load`; G1-17 supports only named encodings, without automatic detection; G1-22 does not infer dates from file text. G1-16 document extraction remains explicitly out of scope.

### 17.7 الحالة

```text
G1_W1_FINAL_GATE           = PASS (all code, branch, evidence, fast-forward, and main CI criteria verified; following report update is documentation-only)
P0_1_STATUS                = CLOSED (unchanged; G0.1 evidence)
P0_2_STATUS                = CLOSED (G1-W1 ingestion evidence)
READY_FOR_G1_W2            = NO; separate CTO authorization required
G1_CERTIFIED_PROGRESS      = 38.46% (10/26)   — بلا تغيير
OVERALL_CERTIFIED_PROGRESS = 43.97% (102/232) — بلا تغيير
G1_W2_AUTHORIZATION        = NO_GO_PENDING_CTO_REVIEW
```
