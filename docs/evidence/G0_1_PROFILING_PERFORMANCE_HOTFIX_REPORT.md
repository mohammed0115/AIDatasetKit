# G0.1 — Profiling performance hotfix report

```text
CURRENT_WAVE                 = G0.1
G0_STATUS                    = PASS
G0_1_STATUS                  = PASS on the branch; main integration recorded in §8
P0_1_STATUS                  = CLOSED
P0_2_STATUS                  = OPEN (CSV delimiter; a G1 item, not touched here)
G0_1_PROGRESS_PERCENT        = 100% (15/15 acceptance criteria, §9)
OVERALL_CERTIFIED_PROGRESS   = 40.95% (95/232; unchanged, §7)
READY_FOR_G1_IMPLEMENTATION  = YES, pending the owner's authorisation of G1-W1
NEXT_AUTHORIZED_ACTION       = G1-W1 needs its own authorisation
```

## 1. Repository

| Field | Value |
|---|---|
| `PRE_SHA` | `2dc8ece5d4cf9923f20d0d2f275632d953a90c4e` (`main` = `origin/main`، شجرة نظيفة، 0/0) |
| `HOTFIX_SHA` | `9065f25ca9b322be2dd86ecdbe768e74c2bbc577`: آخر commit كود/اختبار؛ CI run 36388055537، 10/10 |
| `FINAL_MAIN_SHA` | الـcommit الذي يضيف هذا التقرير (توثيق فقط فوق `HOTFIX_SHA`)، ثم addendum §8 |
| `BRANCH` | `g0.1-profiling-performance` (دُفع؛ لم يُحذف) |
| `WORKTREE_CLEAN` | YES |

**Commits:**

```text
7bc55ba test(perf): reproduce the profiling tie-order regression
3905320 fix(profiling): restore vectorized deterministic counting
0e724e9 test(perf): pass dropna by keyword in the call-growth guard
7a2c52e perf(profiling): order ties from factorize codes without row positions
9065f25 test(perf): add a reproducible profiling benchmark script
<this>  docs(g0.1): record performance recovery evidence
```

**تصحيح عن `7bc55ba`:**

- الاختبار الـparametrised في حارس الخوارزمية مرّر `dropna` موضعيًا، بينما `value_counts(series, *, dropna)` يأخذه كـkeyword فقط.
- لذلك فشلت حالاته الثماني على الكود المتراجع بـ`TypeError`، لا بسبب عدد الاستدعاءات. ادعاء الـcommit أنها «تعيد إنتاج» P0-1 لم يكن صحيحًا بالنسبة لها.
- الحالات الأربع الأخرى (profiler وratio) فشلت لسببها الصحيح.
- `0e724e9` صحّح ذلك. بعد التصحيح فشلت الثماني على الكود المتراجع بالـassertion نفسها (576,002 استدعاء زائد)، ونجحت على الإصلاح.

## 2. Root cause

**القياس** (`cProfile`، استدعاء `value_counts` واحد على 100,000 float متميزة، كود `2dc8ece`): **5,000,675 function calls** في 4.95 s. الأثقل:

| الدالة | عدد الاستدعاءات |
|---|---:|
| `rank()` | 100,000 |
| `counted.iloc[i]` (`indexing.__getitem__` → `_getitem_axis` → `_validate_integer`) | 100,001 |
| `counted.index[i]` | 100,000 |
| `_is_missing` | 200,000 |
| `isinstance` | 900,154 |

**سبب التراجع:**

- `79e7cc8` جعل ترتيب التعادلات حتميًا بحلقة Python لكل عنصر: مرور `series.tolist()` كامل، ثم `sorted` بمفتاح `rank` يصل إلى pandas scalar لكل label.
- التعقيد خطي، لكن بمعامل يقارب 50 استدعاء Python لكل صف.
- في عمود كل قيمه متميزة تتعادل **كل** القيم (العدد 1)، فيدفع كل عمود رقمي متصل هذه الكلفة.

**لماذا اختلف عن `fd4cdd2`:** `fd4cdd2` لم يكن يرتّب التعادلات إطلاقًا؛ كان يعيد ترتيب pandas كما هو. ذلك الترتيب يعتمد على CPU في pandas 2.1، وهذا هو العيب الذي كشفه Linux وأصلحه `79e7cc8`.

**`exact_tally`** (الأعداد الصحيحة الضخمة) مسار منفصل ولم يتأثر: `profile_10k_huge_integers` ‏0.023 s في الـcommits الثلاثة.

## 3. Algorithm

**قبل (`79e7cc8`):**

```text
for each row: key = MISSING if isna(value) else value; first_seen.setdefault(key, row)
sorted(range(k), key=lambda i: (-counted.iloc[i], first_seen[counted.index[i]], i))
```

**بعد (`7a2c52e`):**

```text
codes, uniques = pd.factorize(series)            # codes in order of first appearance; NA = -1
m = max(codes[:first_missing_row]) + 1           # distinct values seen before the first NA
where = Index(uniques).get_indexer(counted.index)
key = where (< m) | m for NA labels | where + 1 (>= m) | k + 1 for unobserved
order = np.lexsort((arange(k), key, -counts))    # stable; count desc, then key, then pandas' order
```

**لماذا هو deterministic ومطابق للعقد:**

- `pd.factorize` يعطي الأكواد بترتيب الظهور الأول، وهو موثَّق كذلك في pandas حين `sort=False`، القيمة الافتراضية. إذن الكود نفسه هو ترتيب الظهور لكل قيمة غير مفقودة.
- القيم المفقودة تشترك في مفتاح واحد يقع بعد القيم التي ظهرت قبل أول صف مفقود بالضبط، أي الأكواد `0 .. m-1`.
- `lexsort` مستقر ولا يعتمد على ترتيب pandas للتعادلات، فالنتيجة لا تعتمد على CPU.
- أعداد pandas وlabels الخاصة به وdtypes لا تتغير؛ يُعاد ترتيبها فقط.
- إذا تعذّر تحديد موقع أي label مرصود، يُستخدم المسار السابق الدقيق (`_ties_by_python_loop`). هذا بطيء لكنه غير خاطئ، ومُختبَر بـ`test_an_unmatchable_label_falls_back_to_the_exact_loop`.
- لا تحويل إلى string، فـ`1` يبقى مختلفًا عن `"1"` (مُختبَر).
- لا cache.

## 4. Benchmarks

**البيئة:**

- Windows 11 (build 26200)، Intel Core i7-8665U (8 logical CPUs)، 15.7 GB RAM (نحو 1.5 GB حرة وقت القياس).
- Python 3.12.3، pandas 2.3.3، numpy 2.4.1.

**المنهج:**

- لكل حالة ولكل commit: عملية مستقلة (`PYTHONPATH` يشير إلى شجرة `git archive` معزولة)، warm-up مستبعد، ثم 7 تشغيلات مقاسة، ويُعتمد الـmedian.
- **Peak memory:** تشغيل منفصل تحت `tracemalloc`، ذاكرة Python فقط.
- **تشغيل متسلسل ومتداخل حسب الحالة** (fd4cdd2 ثم 79e7cc8 ثم الإصلاح لكل حالة، قبل الانتقال إلى الحالة التالية)، لإلغاء انجراف سرعة الجهاز. هذا الانجراف لوحظ في الـbaseline الأول: الكود نفسه في `79e7cc8` و`2dc8ece` أعطى 23.7 s و10.9 s في جلستين مختلفتين.
- لا اختبارات أخرى تعمل أثناء القياس.
- **البيانات:** seed `20260927`، مولَّدة داخل العملية. `scripts/benchmark_profiling.py` هو النسخة المُضافة إلى المستودع من المنهج نفسه.
- **الأمر:** `PYTHONPATH=<tree> python -W ignore bench.py <label> <out.json> 7 <case>`، لكل حالة ولكل commit.

| Case | fd4cdd2 median | 79e7cc8 median | Before hotfix (2dc8ece, جلسة baseline) | After hotfix (`7a2c52e`) | Improvement vs 79e7cc8 | Peak memory fd4cdd2 → after |
|---|---:|---:|---:|---:|---:|---:|
| profile 10k × 20 | 0.087 s | 1.235 s | 1.419 s | 0.177 s | 7.0× | 2.03 → 2.03 MB |
| **profile 100k × 20** | **1.084 s** | **16.386 s** | 10.896 s | **1.498 s** | **10.9×** | 20.12 → 20.12 MB |
| profile 100k unique numeric | 0.038 s | 1.121 s | 3.584 s | 0.063 s | 17.9× | 7.31 → 8.48 MB (+16%) |
| profile 100k low cardinality | 0.062 s | 0.050 s | 0.160 s | 0.053 s | — | 4.52 → 4.52 MB |
| profile 100k all tied pairs | 0.023 s | 0.580 s | 2.390 s | 0.030 s | 19.3× | 4.37 → 5.57 MB (**+27%**) |
| profile 10k huge integers | 0.023 s | 0.023 s | 0.116 s | 0.028 s | — (exact_tally) | 0.43 → 0.43 MB |
| profile 100k near 1e308 | 0.049 s | 1.088 s | 3.234 s | 0.063 s | 17.2× | 7.31 → 8.48 MB (+16%) |
| profile 100k NaN/inf | 0.041 s | 1.025 s | 1.387 s | 0.071 s | 14.4× | 6.89 → 8.00 MB (+16%) |
| profile 100k unique strings | 0.181 s | 1.308 s | 1.315 s | 0.297 s | 4.4× | 7.41 → 9.11 MB (+23%) |
| profile 100k nullable Int64 | 0.023 s | 0.089 s | 0.126 s | 0.020 s | 4.4× | 4.76 → 4.76 MB |
| value_counts 100k unique numeric | 0.014 s | 1.045 s | 1.919 s | 0.028 s | 37.9× | 5.71 → 6.87 MB (+20%) |
| value_counts 100k unique strings | 0.050 s | 1.129 s | 1.298 s | 0.105 s | 10.7× | 5.80 → 7.50 MB (**+29%**) |

- كل digest متطابق بين الـcommits الثلاثة في كل الحالات الـ12، فالنتائج واحدة وظيفيًا.
- عمود «Before hotfix» من جلسة baseline غير متداخلة، ويُعرض للتوثيق فقط. المقارنة الملزمة هي الأعمدة المتداخلة.

**شروط الأداء:**

| الشرط | الحد | النتيجة |
|---|---|---|
| 100k × 20 ≤ 1.5 × fd4cdd2 | ≤ 1.626 s | **1.498 s = 1.38×** ✓ |
| 100k × 20 ≤ 0.30 × 79e7cc8 | ≤ 4.916 s | **1.498 s = 0.091×** ✓ |
| لا تراجع قريب من 2.15 s على العمود الفريد | — | **0.063 s** ✓ |

**الذاكرة:**

- الـprofile الكامل 100k × 20 لم تتغير ذروته (20.12 MB).
- على الأعمدة المفردة زادت الذروة 16–23%، **وفي حالتين تجاوزت 25%**: tied pairs بـ+27% (+1.2 MB)، و`value_counts` على strings فريدة بـ+29% (+1.7 MB).
- **التفسير:** الزيادة تساوي ناتج `pd.factorize` نفسه، أي مصفوفة الأكواد (n × 8 bytes = 0.8 MB لـ100k صف) ومصفوفة الـuniques (k × 8 bytes). هذا الحد الأدنى لأي حساب vectorized لترتيب الظهور. `fd4cdd2` لم يدفعه لأنه لم يكن يحسم التعادلات أصلًا، ولذلك لم يكن حتميًا.
- **النسخة الأولى من الإصلاح** (`3905320`) كانت أثقل (10.81 MB مقابل 7.31، +48%)، بسبب `np.unique(return_index)` و`flatnonzero` ومصفوفة مواقع بطول الصفوف. أُزيل ذلك في `7a2c52e`.
- مقارنة بالمسار المتراجع، الإصلاح أقل بكثير: 7.50 مقابل 19.30 MB.
- الحكم أن الزيادة ضرورية ومفسَّرة. أُبلغ عنها للمراجعة ولم تُخفَ.

**10k × 20:** 0.177 s مقابل 0.087 s في `fd4cdd2`. الكلفة ثابتة لكل عمود (factorize وget_indexer وlexsort، نحو 4 ms للعمود الواحد). لا يشملها شرط الـ1.5× (المحدد لـ100k × 20)، وتُسجَّل هنا.

## 5. Correctness

| البند | الدليل |
|---|---|
| tie ordering حسب أول ظهور | `TestTheVectorisedOrderEqualsTheExactOne`: مطابقة `_ties_by_python_loop` سلسلة بسلسلة، على 15 مدخلًا × `dropna` True/False |
| عكس ترتيب pandas للتعادلات | نفس الصنف، مع `pd.Series.value_counts` يعيد التعادلات معكوسة؛ و`TestTieOrderDoesNotDependOnTheMachine` السابق |
| الاستقرار عبر التكرار | `test_repeated_calls_agree` |
| strings وnumbers متعادلة، mixed `1 / 1.0 / True / "1"` | ضمن الـ15 مدخلًا؛ `test_one_is_not_the_string_one` |
| NaN، None مع NaN، nullable Int64/boolean، datetimes مع NaT | ضمن الـ15 |
| أعداد أكبر من float64 | `test_huge_integers_still_take_the_exact_path`، و`TestIntegersBeyondFloat64` |
| floats قرب 1e308، inf/-inf | ضمن الـ15، و`test_extreme_values.py` |
| empty، single value، all-distinct، low cardinality، categories غير مرصودة | ضمن الـ15 |
| عدم mutation | `test_the_input_is_not_modified` |
| Golden / fingerprint / artifact schema | ملفات `examples/` لم تُمس (`git diff --stat -- examples/` فارغ)؛ `TestTheGoldenSemanticArtifact` و`test_golden_fixtures` و`test_capability_fingerprint_migration` و`test_evidence_artifact` تمر دون تعديل؛ `ARTIFACT_SCHEMA_VERSION` لم يتغير |
| public API | `value_counts(series, *, dropna)` كما هو؛ الدوال الجديدة خاصة (`_first_positions`، `_ties_by_python_loop`) |

## 6. Tests

```text
FOCUSED_TESTS    = counting + performance: 147 + 13 passed; golden/fingerprint/evidence/profiling/
                   statistics/quality/task/frequency/split: 634 passed — each on pandas 2.1.4 (Py 3.11),
                   2.3.3 and 3.0.5 (Py 3.12), locally on Windows
ALGORITHMIC_GUARD = fails on 2dc8ece (576,002 / 545,291 / 315,002 extra calls), passes on the fix
RELATIVE_GUARD    = fails on 2dc8ece (ratio 65 / 20 / 75 > 6), passes on the fix (≈ 2)
FULL_TESTS        = CI run 36388055537 on 9065f25: 10/10 success
CI_RUN_BRANCH     = 36388055537 — https://github.com/mohammed0115/AIDatasetKit/actions/runs/36388055537
CI_RUN_MAIN       = §8
FAILURES          = 0
ERRORS            = 0
SKIPPED           = Linux 47 / 44 with extras; Windows 46 / 43 (as before G0.1)
```

| Job | Job id | Collected | Result |
|---|---|---:|---|
| ubuntu-latest / minimum | 108817567320 | 4219 | 4173 passed, 47 skipped |
| ubuntu-latest / minimum + extras | 108817567386 | 4241 | 4197 passed, 44 skipped, 1 warning |
| ubuntu-latest / reference | 108817567385 | 4219 (JUnit 4220) | 4173 passed, 47 skipped (من JUnit؛ السجل لم يُقرأ) |
| ubuntu-latest / reference + extras | 108817567478 | 4241 | 4197 passed, 44 skipped |
| ubuntu-latest / build, install and smoke | 108817567071 | — | smoke passed |
| windows-latest / minimum | 108817567331 | 4219 | 4174 passed, 46 skipped |
| windows-latest / minimum + extras | 108817567304 | 4241 | 4198 passed, 43 skipped, 1 warning |
| windows-latest / reference | 108817567341 | 4219 | 4174 passed, 46 skipped |
| windows-latest / reference + extras | 108817567503 | 4241 | 4198 passed, 43 skipped |
| windows-latest / build, install and smoke | 108817567355 | — | smoke passed |

- **من تقارير JUnit للخلايا الثماني:** `test_counting_performance` 13/13 passed، و`TestTheVectorisedOrderEqualsTheExactOne` 93/93 passed. صفر failures، صفر errors، صفر skipped.
- **الحراس:** تعمل ضمن الاختبارات العادية في كل خلية، ولم يلزم job أداء منفصل. حارس النسبة يقارن بـpandas في العملية نفسها، ولم يتذبذب.
- **التحذير الواحد** في خلايا الـextras هو `PyparsingDeprecationWarning` من matplotlib 3.9.0، كما قبل G0.1.

## 7. Audit and progress update

- **P0-1 → CLOSED.**
- **P0-2 → OPEN.**
- **G11:** انتقلت G11-02 وG11-04 وG11-11 وG11-21 من `MISSING` إلى `PARTIAL`، مع ذكر النطاق الضيق لكل منها في جدول التدقيق. لم ينتقل أي بند إلى `SUPPORTED_AND_TESTED`، لأن حارس مسار واحد وbenchmark غير مُشغَّل في CI لا يغلقان فجوات الأداء.
- **النسب:** G11 = 5/21 = 23.81%، بلا تغيير؛ والكلية = 95/232 = **40.95%**، بلا تغيير.
- **G12-14:** شدته من P0 إلى P1.
- **تصحيح:** `AUDIT_CAPABILITY_ROWS = 222`، و`COMMIT_MESSAGE_237 = HISTORICAL_TYPO`، و`AUDIT_DOCUMENTS = CORRECT` (§0 في التدقيق).

**Findings recorded, not fixed (خارج نطاق G0.1):**

- `aidatasetkit/visualization/preparation.py:463` يستدعي `.astype(str).value_counts()` مباشرة، متجاوزًا `core.counting`. ترتيب التعادلات فيه يعتمد على pandas وCPU. أثره على بيانات الرسوم فقط (ترتيب الأعمدة عند الاقتطاع)، لا على `audit.json`.
- `profiler.py:290` (`_looks_like_a_counter`) يُصدر `RuntimeWarning: overflow encountered in scalar subtract` على قيم قرب 1e308. هذا موجود منذ `fd4cdd2`، ولا يغيّر النتيجة (`span` يصبح inf فلا يُعَد counter).

## 8. Main integration

يُسجَّل في addendum توثيقي بعد CI على `main`.

## 9. Gate

| # | شرط | الحالة |
|---|---|---|
| 1 | Root cause مثبت | ✓ (§2، cProfile) |
| 2 | الأداء ضمن الحد | ✓ (§4) |
| 3 | algorithmic guard يمر، ويفشل على 79e7cc8 | ✓ |
| 4 | relative performance guard يمر، ويفشل على 79e7cc8 | ✓ |
| 5 | اختبارات correctness تمر | ✓ |
| 6 | Golden لم يتغير | ✓ |
| 7 | fingerprints لم تتغير | ✓ |
| 8 | CI الفرع 10/10 | ✓ (36388055537) |
| 9 | main محدث بـfast-forward | §8 |
| 10 | CI main 10/10 | §8 |
| 11 | P0-1 مغلق | ✓ |
| 12 | P0-2 مفتوح | ✓ |
| 13 | worktree نظيف | ✓ |
| 14 | التقرير committed | ✓ |
| 15 | لم يبدأ G1 | ✓ |

## 10. Scope

```text
G1_STARTED                 = NO
CSV_DELIMITER_P0_2         = OPEN
MASARI_CHANGED             = NO
MWIE_CHANGED               = NO
PUBLIC_API_BREAKING_CHANGE = NO
NEW_DEPENDENCIES           = NO
GOLDEN_CHANGED             = NO
FORCE_PUSH / HISTORY_REWRITE = NO
```

## PROJECT PLAN PROGRESS

| Gate | المجال | Proven | Applicable | Certified % | الحالة | الخطوة التالية |
|---|---|---:|---:|---:|---|---|
| G0 | Baseline stability | 16 | 16 | 100.00% | PASS | مغلق |
| G0.1 | Profiling performance hotfix | 15 | 15 | 100.00% (معايير القبول) | PASS بعد §8 | مغلق |
| G1 | Ingestion | 5 | 26 | 19.23% | AUDIT | G1-W1 بتفويض |
| G2 | Profiling & Quality | 15 | 22 | 68.18% | AUDIT | بعد G1 |
| G3 | Cleaning & Transformation | 2 | 17 | 11.76% | AUDIT | — |
| G4 | EDA & Statistics | 9 | 13 | 69.23% | AUDIT | — |
| G5 | Trends & Comparisons | 0 | 15 | 0.00% | AUDIT | — |
| G6 | Segmentation & Scoring | 2 | 12 | 16.67% | AUDIT | — |
| G7 | Multi-dataset & Schema | 0 | 13 | 0.00% | AUDIT | — |
| G8 | Visualization & Reporting | 15 | 23 | 65.22% | AUDIT | — |
| G9 | AI-ready Context | 8 | 12 | 66.67% | AUDIT | — |
| G10 | Provenance/Security/Privacy/Errors | 16 | 28 | 57.14% | AUDIT | — |
| G11 | Performance & Certification | 5 | 21 | 23.81% | AUDIT | 4 بنود صارت PARTIAL |
| G12 | Masari Consumer Integration | 2 | 14 | 14.29% | AUDIT | — |

```text
CURRENT_WAVE                 = G0.1
G0_STATUS                    = PASS
G0_1_STATUS                  = PASS (after §8)
P0_1_STATUS                  = CLOSED
P0_2_STATUS                  = OPEN
G0_1_PROGRESS_PERCENT        = 100%
OVERALL_CERTIFIED_PROGRESS   = 40.95%
READY_FOR_G1_IMPLEMENTATION  = YES (owner authorisation required)
NEXT_AUTHORIZED_ACTION       = none; G1-W1 awaits authorisation
```

G0.1 ليس بوابة من البوابات الـ13، ولا يدخل في `OVERALL_CERTIFIED_PROGRESS`. نسبته تُحسب من شروط قبوله الـ15.
