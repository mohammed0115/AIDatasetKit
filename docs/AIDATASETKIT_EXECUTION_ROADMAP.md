# AIDatasetKit — Execution roadmap (G0.1 → G12)

**المصدر:**

- مبني على `docs/evidence/POST_G0_CAPABILITY_REALITY_AUDIT.md` (commit `9e82bd7` على `main`).
- المعرّفات `Gx-NN` تشير إلى صفوف جدول التدقيق.
- لا تقديرات زمنية؛ الحجم فقط: `S` / `M` / `L` / `XL`.

**قواعد لكل موجة:**

- موجة واحدة في كل تفويض.
- commits صغيرة وأحادية الغرض.
- لا skip لإخفاء فشل.
- CI على Windows وLinux لكل SHA يُعتمد.
- الموجة لا تُغلق إلا بمعيار قبول مُثبت بتشغيل.

**نسبة البوابة** = بنود `SUPPORTED_AND_TESTED` ÷ البنود المنطبقة. تُعاد حسابها من جدول التدقيق بعد كل موجة، ولا تُقدَّر.

---

## G0.1 — Hotfix قبل G1 (**منجز: PASS**، انظر `docs/evidence/G0_1_PROFILING_PERFORMANCE_HOTFIX_REPORT.md`)

| البند | المحتوى |
|---|---|
| Objective | إزالة P0-1: تراجع أداء الـprofiling الذي أدخله `79e7cc8` |
| Proven | عقد ترتيب المتعادلات صحيح ومُختبَر (`TestTieOrderDoesNotDependOnTheMachine`) |
| Gap | `_ties_in_order_of_appearance` حلقة Python بوصول pandas لكل عنصر؛ 100k × 20 = 21.48 s مقابل 2.34 s قبل الإصلاح |
| Wave | vectorize: `pd.factorize(series, use_na_sentinel=...)`، ثم `np.bincount`، ثم `np.argsort(-counts, kind="stable")` على ترتيب الظهور. عداد exact_tally للأعداد الصحيحة الضخمة كما هو |
| Tests | الاختبارات الحالية دون تعديل؛ اختبار حراسة أداء جديد (profile لـ100k × 20 تحت حد متسامح مُعلَن، مثل 10 s على CI)؛ fixtures لا تتغير |
| Acceptance | `test_counting` كله يمر؛ Golden دون تغيير؛ 100k × 20 ≤ 1.5 × زمن `fd4cdd2` على الجهاز نفسه؛ CI 10/10 |
| Stop | أي تغيير في fixture، أو في ترتيب مُعلَن، يوقف الموجة |
| Complexity | **S** |

---

## G1 — Ingestion (بعد G1-W1: 38.46%، 10 من 26؛ كان 19.23%)

**Objective:** إدخال عام وآمن للجداول، مع كشف صريح لبنية الملف، ودون أي قراءة صامتة خاطئة.

**Proven:** CSV بفاصلة وUTF-8 عبر CLI؛ DataFrame في الذاكرة؛ رفض الأعمدة المكررة؛ عدد الصفوف والأعمدة والذاكرة؛ رسالة واضحة للصيغة غير المدعومة.

**Gaps:**

- ~~**P0-2:** فاصل غير `,` يُقرأ خطأً دون تحذير.~~ **مغلق في G1-W1.**
- **P1:** XLSX وParquet وJSON/JSONL؛ استنتاج التواريخ؛ حدود الموارد؛ chunking. (الترميزات الصريحة والأسطر المشوهة وTSV وrecords أُنجزت في G1-W1؛ اكتشاف الترميز التلقائي غير مخطط.)
- **P2:** XLS وsheets وFeather وSQLite وPostgreSQL.
- **P3:** XML وYAML.
- المستندات: `OUT_OF_SCOPE`.

**Waves:**

| Wave | المحتوى | Depends | Complexity |
|---|---|---|---|
| G1-W1 ✅ **منجزة** (`docs/evidence/G1_W1_INGESTION_FOUNDATION_REPORT.md`) | حزمة `aidatasetkit.ingestion` (طبقة فوق `core`): `load_table(path \| frame \| records, *, options) -> LoadedTable(frame, metadata)`. metadata فيها الصيغة، والترميز، والفاصل، والرأس، والصفوف والأعمدة، والذاكرة، والتحذيرات. CSV مع كشف الفاصل (رفض صريح عند الغموض)، وترميز صريح، وTSV، وrecords (list of dicts). خطأ منظم لكل من الصيغة غير المدعومة والمدخل المشوه والمدخل الفارغ. CLI يستخدم الطبقة الجديدة | G0.1 | M |
| G1-W2 (غير مُفوَّضة؛ تنتظر مراجعة CTO) | حدود موارد مُعلَنة (bytes، rows، columns) وخطأ `ResourceLimitError`؛ استنتاج التواريخ من النص بسياسة معلنة، دون تخمين صامت | W1 | M |
| G1-W3 ✅ **منجزة** (`docs/evidence/G1_W3_FORMAT_READERS_REPORT.md`) | Parquet وFeather (pyarrow كـextra اختياري)؛ JSON وJSONL | W1 | M |
| G1-W4 (PASS؛ CI الفرع وCI على main 10/10، انظر `docs/evidence/G1_W4_XLSX_READER_REPORT.md`) | XLSX (openpyxl كـextra) مع sheets واختيار صريح للـsheet؛ رفض ملفات الماكرو | W1 | M |
| G1-W5 (PASS؛ CI الفرع وCI على main 10/10، انظر `docs/evidence/G1_W5_CHUNKED_PROFILING_REPORT.md`) | chunked profiling للملفات الكبيرة أو sampling مُعلَن، مسجّل في الـartifact | W2 | L |
| G1-W6 (مُنفَّذ على الفرع؛ غير معتمد. انظر `docs/evidence/G1_W6_SQLITE_READER_REPORT.md`) | SQLite read-only عبر `sqlite3` من stdlib؛ جدول عادي واحد، بلا SQL من المتصل | W1 | S |

**Tests:**

- fixtures صغيرة مُولَّدة داخل الاختبار: فواصل `,` و`;` وtab و`|`، ترميزات utf-8 وutf-8-sig وlatin-1 وcp1256، ملف فارغ، رأس فقط، أسطر مشوهة، صيغ خاطئة الامتداد.
- لكل صيغة: قراءة، ثم profile، ثم مقارنة مع DataFrame مرجعي.
- اختبارات حدود الموارد.
- CI على المنصتين، مع extras وبدونها.

**Acceptance:**

- كل مدخل غامض إما يُقرأ صحيحًا أو يُرفض بخطأ منظم؛ **لا قراءة صامتة خاطئة**. الـprobe يُحوَّل إلى اختبار.
- كل صيغة مُعلنة لها اختبار قراءة ذهابًا وإيابًا.
- كل رفض يحمل صنف خطأ من `AIDatasetKitError`.
- metadata الإدخال تظهر في `audit.json`.

**Stop:** أي صيغة تحتاج اعتمادًا إلزاميًا جديدًا في النواة تتوقف لقرار المالك (تبقى extra).

---

## G2 — Profiling & Quality (68.18%: 15 من 22)

**Proven:** القيم المفقودة، التكرار، التفرد، الإحصاءات الوصفية، القيم اللانهائية، القيم الثابتة، القيم الشاذة، VIF، الـfindings وأدلتها، الفحوص المرتبطة بالهدف.

**Gaps:**

- **P1:** ملف وصف للتواريخ (min وmax وfrequency وgaps).
- **P2:** الأصفار والقيم السالبة، أطوال النص، السلاسل الفارغة، شكل التوزيع في الـprofile، قيم خارج قاعدة مجال، عقد موحد للصفوف المتأثرة.

**Waves:**

- **G2-W1 (M):** `DatetimeSummary` وفحص فجوات زمنية.
- **G2-W2 (S):** `TextSummary` (الطول، السلاسل الفارغة) وفحص السلاسل الفارغة.
- **G2-W3 (S):** عدّ الأصفار والقيم السالبة، وskew وkurtosis في `NumericSummary`، مع migration للـfingerprint.

**Acceptance:** لكل حقل جديد: اختبار، وmigration مسجّل في السلسلة، وfixture لكل pandas major.

**Stop:** أي finding جديد يغيّر verdict الـGolden يتوقف لمراجعة.

---

## G3 — Cleaning & Transformation (11.76%: 2 من 17)

**Objective:** عمليات جدولية عامة **منفصلة** عن خط أنابيب النموذج، تعيد جدولًا جديدًا مع سجل عمليات.

**Gaps (P1):** filter، group-by، aggregate، join، derived columns، de-duplication، type conversion، date normalisation. **(P2):** rename، reshape، normalisation للنصوص والفئات. **(P3):** sort.

**Waves:**

- **G3-W1 (L):** `aidatasetkit.transform` مع `Operation` بدون حالة ومسجَّلة. كل عملية تنتج `(frame, OperationRecord)`، والمصدر لا يتغير. تبدأ بـ filter وselect وrename وcast وdedupe.
- **G3-W2 (M):** group-by وaggregate، مع سياسة NaN معلنة.
- **G3-W3 (M):** derived columns بتعبيرات آمنة: دوال مُسجَّلة، **لا `eval`**.
- **G3-W4 (M):** date normalisation وreshape.

**Tests:** الثبات (المصدر لا يتغير)، سجل العمليات قابل للتسلسل ومُفَهرس في الـprovenance، الحالات الحدية (فارغ، NaN، أنواع مختلطة).

**Stop:** لا تُعاد كتابة preprocessing؛ G3 لا يغيّر مسار النموذج.

---

## G4 — EDA & Statistics (69.23%: 9 من 13)

**Gaps (P2):** الارتباط الفئوي (contingency، chi-square، Cramér's V)؛ اختبارات المقارنة (t، Mann-Whitney) مع الافتراضات مُعلنة؛ ملخص متعدد المتغيرات؛ مخرَج مقروء لـEDA.

**Waves:**

- **G4-W1 (M):** `association` في `statistics`.
- **G4-W2 (M):** `compare_groups` مع القيود.
- **G4-W3 (S):** قسم statistics في التقرير.

**Acceptance:** مطابقة scipy المرجعية؛ رفض الحالات المنحلة بـ`DomainError`؛ مخرجات deterministic.

---

## G5 — Trends & Comparisons (0%: 0 من 15)

**Depends:** G1-W2 (التواريخ)، G3-W2 (التجميع).

**Waves:**

- **G5-W1 (M):** time aggregation، التغير المطلق والنسبي مع سياسة القسمة على صفر والفترات المفقودة.
- **G5-W2 (M):** rolling metrics والمتوسطات المتحركة.
- **G5-W3 (M):** مقارنات dataset A مقابل B، وsegment مقابل segment، وقبل مقابل بعد، والفعلي مقابل المستهدف.
- **G5-W4 (L):** trend detection، وanomaly candidates زمنية، ومؤشرات موسمية. كلها «candidates» لا «حقائق».

**Stop:** forecasting خارج G5 إلا بتفويض منفصل.

---

## G6 — Segmentation & Scoring primitives (16.67%: 2 من 12)

**Depends:** G3-W2.

**Waves:**

- **G6-W1 (M):** segmentation بالأعمدة، والتقسيم الرقمي بسياسة معلنة، وsegmentation بالقواعد (قواعد يمررها المستدعي).
- **G6-W2 (M):** percentile ranks، وscaling كـprimitive، وweighted وcomposite metrics بأوزان يمررها المستدعي.
- **G6-W3 (S):** profile لكل segment ومقارنة بين segments.

**Stop:** أي تسمية أعمال (مثل «فرصة جيدة») أو أوزان مجال خارج المكتبة.

---

## G7 — Multi-dataset & Schema (0%: 0 من 13)

**Waves:**

- **G7-W1 (M):** `SchemaContract` (أعمدة مطلوبة واختيارية، أنواع، مفاتيح) مع تقرير انتهاكات منظم بمسارات الحقول.
- **G7-W2 (M):** joins مع تحقق 1:1 و1:N وتقرير المفاتيح المفقودة.
- **G7-W3 (M):** تجميع ومقارنة عبر datasets.

---

## G8 — Visualization & Reporting (65.22%: 15 من 23)

**Gaps:**

- **P1:** line وtime series.
- **P2:** تصدير الرسوم إلى ملف، رسوم داخل التقرير، Markdown، CSV exports.
- **P3:** Excel وPDF.

**Waves:**

- **G8-W1 (M):** line وtime series (بعد G2-W1).
- **G8-W2 (S):** `render_to_file(png/svg)`، ونشر الرسوم ضمن الـrun الذري.
- **G8-W3 (M):** تقرير Markdown، وتصدير CSV للجداول. CSV يحتاج **تحييد الصيغ** (`=`، `+`، `-`، `@`).

---

## G9 — AI-ready context (66.67%: 8 من 12)

**Waves:**

- **G9-W1 (M):** أقسام statistics وvisualizations في الـartifact، مع migration.
- **G9-W2 (S):** JSON Schema منشور واختبار تحقق عليه.
- **G9-W3:** trends وsegments وcomparisons، بعد G5 وG6.

**Acceptance:** الحجم محدود ومُختبَر؛ determinism؛ لا قيم خام افتراضيًا.

---

## G10 — Provenance / Security / Privacy / Errors (57.14%: 16 من 28)

**Waves:**

- **G10-W1 (S):** أخطاء منظمة للصيغة غير المدعومة والمدخل المشوه وحدود الموارد (تتقاطع مع G1).
- **G10-W2 (S):** اختبارات حراسة: لا استيراد شبكي أو LLM أو telemetry، ولا pickle أو eval (فحص AST في الاختبارات)، ولا قيم في السجلات.
- **G10-W3 (M):** سجل تشغيل موحد (operation، parameters، result) عبر G3 وG5.

---

## G11 — Performance & Certification (23.81%: 5 من 21)

**Waves:**

- **G11-W1 (M):** benchmarks حتمية (بيانات مُولَّدة بـseed، أحجام 10k و100k و1M)، تُسجَّل مع الـSHA. حراسة تراجع في CI بحدود متسامحة.
- **G11-W2 (S):** macOS وPython 3.13 في المصفوفة.
- **G11-W3 (S):** `-W error` ثم lint ثم type check، تدريجيًا.

**Stop:** لا تشديد لحدود الأداء بحيث يصبح CI متذبذبًا.

---

## G12 — Masari consumer certification (14.29%: 2 من 14)

**Depends:** G1 وG3 وG5 وG6 وG7 وG8 وG9.

**Wave:** مجموعة سيناريوهات تكامل عامة بلا منطق Masari. جداول مُولَّدة تمثل «فرص»، «إيرادات»، «محتوى»، وكل سيناريو يستدعي API عامًا فقط.

**Acceptance:** الأسئلة الخمسة عشر في `docs/MASARI_INTEGRATION.md` تنتقل إلى `SUPPORTED_AND_TESTED` أو `NOT_APPLICABLE`، مع اختبار لكل منها.

---

## ترتيب مقترح

```text
G0.1 → G1-W1 → G1-W2 → G10-W1 → G2-W1 → G3-W1 → G3-W2 → G5-W1 → G6-W1 → G7-W1 → G8-W1 → G9-W1 → G11-W1 → (the rest) → G12
```

الترتيب يتبع الاعتماديات المذكورة في كل بوابة. كل خطوة تحتاج تفويضًا منفصلًا.
