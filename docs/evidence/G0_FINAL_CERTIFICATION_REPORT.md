# AIDatasetKit — G0 Final Certification Report

> **الحكم في سطر:** كل ما طُلب إصلاحه أُصلح وأُثبت بتشغيل فعلي على **Windows**. البوابة تبقى **`BLOCKED`** لشرطين إلزاميين لا يمكن تحقيقهما من هذا الجهاز دون قرار من المالك:
> 1. لا يوجد تشغيل فعلي على **Linux**.
> 2. شجرة العمل تحتوي ملفات **untracked** يملكها المالك.
>
> لا يوجد «PARTIAL PASS».

كل رقم في هذا التقرير مستخرج من سجل تشغيل. نسخة مختصرة من كل سجل نهائي محفوظة في `docs/evidence/runs/`. تحتوي النسخة على البيئة ومسار الاستيراد والأمر وسطر الجمع وسطر النتيجة وقائمة الفشل. سجلات JUnit الكاملة بقيت في scratchpad الجلسة ولم تُضَف إلى المستودع.

---

## A. Repository reality

| Field | Value |
|---|---|
| `PRE_SHA` | `59bb28639b01982e3a1fa882317e1054acb14bea` |
| `POST_SHA` | الـcommit الذي يضيف هذا التقرير. لا يعدّل هذا الـcommit إلا ملفات تحت `docs/evidence/`. الكود المُختبَر هو أبوه `fd4cdd2eddc561866e5e43b852fdcfffda9eb982`. |
| `BRANCH` | `main` |
| `ORIGIN_MAIN` | `59bb28639b01982e3a1fa882317e1054acb14bea`. لم يحدث أي push. |
| `AHEAD_BEHIND` | 18 commit أمام `origin/main` (17 commit قبل هذا التقرير، ثم هذا التقرير)، و0 خلفه. |
| `WORKTREE_CLEAN` | **لا.** الملفات المتتبَّعة نظيفة. تبقى ملفات untracked يملكها المالك وكانت موجودة قبل G0: أربعة ملفات في `docs/` (`AIDatasetKit_ARCHITECTURE_OVERVIEW (1).md`، `AIDatasetKit_Execution_Plan_AR.md`، `CHANGELOG (2).md`، `gaps1.md`)، و35 ملف PDF في `docs/book/`. لم تُضَف إلى commit ولم تُحذَف ولم تُخفَ، لأن القرار فيها للمالك. |
| `COMMITS_CREATED` | 18 (القائمة أدناه) |
| `FILES_CHANGED` | `59bb286..fd4cdd2`: 72 ملفًا، 4864 إضافة و306 حذف. داخل `aidatasetkit/`: 21 ملفًا (+1158/−56). داخل `tests/`: 24 ملفًا (+2108/−169). |
| `git diff --check 59bb286 fd4cdd2` | نظيف |

**التحقق من الـWIP:**

- قبل أي تعديل، سُجّلت قيمة `sha256(git diff)` للـWIP: `e190292e…41461`.
- أُعيد بناء الـWIP حرفيًا، ثم طابق diff إعادة البناء هذه القيمة بايتًا ببايت قبل الـcommit `0afd53f`.
- إذن: الـWIP لم يُحذَف ولم يُعدَّل قبل حفظه، ثم جاءت التقوية في commit منفصل.

**قائمة الـcommits:**

```text
173f3fc docs(g0): record pre-execution state and the baseline reference
64098a5 test(windows): preserve the one variable a child interpreter needs
3700cad test(encoding): read committed fixtures and artifacts as UTF-8
0afd53f feat(quality): add a multicollinearity (VIF) check          ← WIP المالك حرفيًا
a5e8656 fix(quality): bound and harden the multicollinearity check
798e21b feat(vif): record the config fingerprint migration and pin a fixture per pandas major
6d57e8d fix(profiling): count integers beyond float64 exactly on pandas 2
4852539 fix(statistics): measure finite values near the float64 ceiling without overflow
3362ff9 test(compat): assert the library's protection, not one scikit-learn's failure
e64fa29 fix(validation): isolate the final test from model selection (G0-01)
e6f704b fix(compat): call scipy.stats.moment positionally so SciPy 1.11 works
5ae0a87 feat(output): publish each audit run atomically behind a CURRENT pointer
17d2baf test(output): prove old-or-new under failure, kill, contention and tamper
48999fa fix(config): remove cv_folds, a setting nothing ever read (G0-06)
448a244 docs(release): align publication and signing claims with the evidence (G0-03)
d9e13c8 ci(matrix): pin the tested environments and declare exactly what was run
fd4cdd2 docs(changelog): record the G0 changes, breaking ones first
<this>  docs(g0): add final certification evidence
```

---

## B. Baseline comparison

كل التشغيلات على Windows 11 (build 26200)، بـlocale `cp1252`.

- **العزل:** كل تشغيل على شجرة معزولة من `git archive <commit>`. في كل venv تُثبَّت الحزمة `--no-deps` حتى توجد metadata التوزيع.
- **مسار الاستيراد:** سُجّل في كل تشغيل، وكان داخل الشجرة المعزولة في كل مرة.
- **الأمر:** `python -m pytest -p no:cacheprovider -q -rfE --junitxml=…` مع `PYTHONDONTWRITEBYTECODE=1`.

| Environment | Before (`59bb286`) collected / passed / failed / skipped / errors | After (`fd4cdd2`) collected / passed / failed / skipped / errors | Duration after | Evidence |
|---|---|---|---|---|
| **MIN**: Py 3.11.16, numpy 1.26.4, pandas 2.1.4, scipy 1.11.4, sklearn 1.6.1, pytest 8.0.0 | لم يُقَس على الـbaseline. أول قياس كان على `3362ff9` وأعطى 52 فشلًا. | **4097 / 4052 / 0 / 46 / 0** | 6693 s (تحت حمل؛ انظر الملاحظة) | `runs/final_MIN.md` |
| **REF**: Py 3.12.3, numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, sklearn 1.9.0, pytest 8.3.3 | HEAD: 3821 / 3762 / 3 / 45 / 12. WIP: 3834 / 3773 / 5 / 45 / 12 | **4097 / 4052 / 0 / 46 / 0** | 6852 s (تحت حمل) | `runs/final_REF.md` |
| **L**: Py 3.12.3، بيئة المطوّر المحلية: numpy 2.4.1، pandas 2.3.3، scipy 1.17.0، sklearn 1.8.0 | HEAD: 3821 / 3751 / 14 / 45 / 12. WIP: 3834 / 3764 / 14 / 45 / 12 | **4097 / 4052 / 0 / 46 / 0** | 411.56 s | `runs/final_L_alone.md` |
| **MIN + extras**: MIN مع matplotlib 3.9.0، xgboost 2.0.0، lightgbm 4.0.0، catboost 1.2 | لم يُقَس | **4119 / 4076 / 0 / 43 / 0** | 412.61 s | `runs/final_MIN_EXTRAS.md` |
| **REF + extras**: REF مع matplotlib 3.11.2، xgboost 3.4.1، lightgbm 4.7.0، catboost 1.2.10 | لم يُقَس | **4119 / 4076 / 0 / 43 / 0** | 494.00 s | `runs/final_REF_EXTRAS.md` |
| **Linux**: أي بيئة | — | **BLOCKED**: لا يوجد WSL أو Docker أو Podman على الجهاز، والـpush ممنوع فلا يعمل CI | — | — |
| Packaging (Windows) | — | **PASSED**: wheel وsdist، `twine check` PASSED لكليهما، تثبيت كل منهما في venv نظيف، `--version`، ثم audit يُقرأ عبر `read_current` من الحزمة المثبتة | — | `runs/final_smoke.md` |

**ملاحظة عن تشغيل L تحت الحمل (`runs/final_L.md`):**

- شُغّلت MIN وREF وL بالتوازي، فاستغرق كل منها قرابة 1h55 بدل ~7–9 دقائق.
- فشل في تشغيل L اختبار واحد: `test_visualization_without_matplotlib::test_a_full_plan_is_produced_and_serialised`، بعد 5608 s.
- العملية الفرعية انتهت بـ`returncode: 0`، أي أنها نجحت. الخطأ `TimeoutExpired` جاء من مؤقت `communicate()` الذي صار سالبًا، وليس من الكود.
- الاختبار نفسه نجح في MIN وREF في الدفعة نفسها.
- أُعيد تشغيل L وحده، فأعطى 4052/0/46/0 خلال 411 s.
- التصنيف: **توقف في الجهاز تحت الحمل**، وليس عيبًا. يُسجَّل هنا ولا يُخفى.

**الـ46 skipped:**

- **39 إنشائية:** اختبارات parametrized على كل نموذج، تُتخطى للنماذج التي لا تعلن القدرة المختبَرة («does not claim both»، «accepts sparse»، «no native NaN»، «no signal to recover»).
- **4 تتطلب matplotlib:** تُشغَّل وتنجح في بيئتي extras.
- **1** يُشغَّل على غير Windows فقط.
- **في بيئتي extras (43):** لا يبقى إلا الإنشائية، إضافة إلى اختبار يتطلب **غياب** xgboost، وهذا يُشغَّل في البيئات الأساسية.

**المقارنة مع التقرير المرجعي (`G0_BASELINE_CORRECTION_REPORT.md`):** الأرقام المرجعية أُعيد إنتاجها هنا ولم يُفترض أنها ما زالت صحيحة. `WIP/V` كان 5 failed و12 errors، ونقاط التفتيش أعادت إنتاج الأسباب الأربعة.

---

## C. Gap closure matrix

| Gap ID | Root cause | Files changed | Tests added | Focused result | Regression result | Status | Evidence |
|---|---|---|---|---|---|---|---|
| **WIN-SUBPROC** | الأبناء يبدؤون بـ`{PYTHONHASHSEED, PATH:""}`. بغياب `SystemRoot` يفشل Winsock (`asyncio.windows_events`) عند استيراد sklearn، فيظهر `WinError 10106` | `tests/conftest.py` و4 ملفات اختبار | `test_isolated_env.py` (6 اختبارات) | 12 error → 0؛ 261 passed | 0 errors في كل البيئات | **CLOSED** | `64098a5` |
| **WIN-UTF8** | `read_text()` بلا encoding يعمل مع `cp1252` | `test_audit_end_to_end.py`، `test_capability_fingerprint_migration.py` | — (تصحيح 32 قراءة و4 كتابات) | `committed_report_renders` يمر على cp1252 | ✓ | **CLOSED** | `3700cad` |
| **VIF-WIP** | الإعداد الجديد يحرّك config fingerprint؛ لا حد للكلفة؛ NaN عند القيم الكبيرة جدًا؛ عتبة تقبل NaN/inf | `profiling/checks.py`، `core/config.py`، fixtures، migration | 24 (16 robustness + 8 config) + chain | 15 اختبارًا جديدًا تفشل على WIP الحرفي وتنجح بعد التقوية | ✓ | **CLOSED** | `0afd53f`، `a5e8656`، `798e21b` |
| **OVF-INT** | `value_counts` في pandas 2.x يحوّل int ضخمًا إلى float؛ و`_sort_key` في task detector يستدعي `float(label)` | `core/counting.py` (جديد)، profiler، task_detector، frequency، split | `test_counting.py` (38) | الاختبارات الثلاثة التاريخية تمر على pandas 2.1.4 و2.3.3 و3.0.5 | ✓ | **CLOSED** | `6d57e8d` |
| **OVF-FLOAT** (مُكتشف أثناء G0، موجود في HEAD) | mean=nan وmedian=−inf وiqr=inf لقيم قرب 1e308، فالـartifact لا يُسلسَل | `statistics/engine.py`، `guards.py`، `moments.py`، profiler | `test_extreme_values.py` (16) | 12 من 16 تفشل على الكود السابق | ✓ | **CLOSED** | `4852539` |
| **SKL-COMPAT** | 7 اختبارات تفترض سلوكًا خاصًا بـsklearn 1.9 أو 1.8 | 3 ملفات اختبار و3 docstrings | — (إعادة كتابة تؤكد الفرعين) | تمر على 1.6.1 و1.8.0 و1.9.0 | ✓ | **CLOSED** | `3362ff9`، `e6f704b` |
| **SCIPY-1.11** (مُكتشف أثناء G0) | `stats.moment(order=)` غير موجود قبل SciPy 1.12، فتفشل 51 حالة على الحد المعلن | `statistics/moments.py`، `test_moments.py` | — | MIN: 52 فشلًا → 0 | ✓ | **CLOSED** | `e6f704b` |
| **G0-01** Data leakage | `evaluate()` كان يقيس على صفوف اختارت النموذج، ويصفها بأنها مستقلة؛ ولا تقدير مستقل على الإطلاق | `facade/facade.py`، `facade/state.py`، docs | `test_final_evaluation.py` (39) | 39 passed؛ mutants: 8 و5 فشلًا | ✓ | **CLOSED** | `e64fa29` |
| **G0-04** Atomic output | ثلاث `write_text` متتالية في `--output` | `evidence/publication.py` (جديد)، CLI، exceptions، smoke، docs | `test_publication.py` (73) و5 اختبارات CLI | 73 passed؛ mutants: 30 و2 فشلًا | ✓ | **CLOSED on Windows** | `5ae0a87`، `17d2baf` |
| **G0-06** `cv_folds` | إعداد موثّق ومُتحقَّق منه ومسجّل في البصمة، ولا يقرؤه أي مسار | `core/config.py`، fixtures، chain | `TestCvFoldsIsGone` (4) و`TestTheCvFoldsStep` (2) | ✓ | ✓ | **CLOSED** (أُزيل) | `48999fa` |
| **G0-03** «signed-off» وادعاءات النشر | وصف الحزمة يعد بـartifact موقَّع؛ «public alpha»؛ `pip install aidatasetkit` | pyproject، RELEASE_NOTES، CHANGELOG، README، CONTRIBUTING، SECURITY، ARCH | `TestReleaseClaimsMatchTheEvidence` (18) | يلتقط كل ادعاء في النص السابق | ✓ | **CLOSED** | `448a244` |
| **G0-02 / DEP** CI ومصفوفة الاعتماديات | حدود معلنة لم تُختبر (`sklearn>=1.4`)؛ لا CI | pyproject، `constraints/*`، `.github/workflows/ci.yml`، docs، MANIFEST | `test_dependency_contract.py` (26) | ✓ | MIN وREF وextras تمر | **Windows: CLOSED — Linux: BLOCKED** | `d9e13c8` |
| **G0-05** حدود حجم CSV والذاكرة | خارج موجات هذا التكليف | — | — | — | — | **NOT ADDRESSED** (خارج النطاق المفوَّض؛ مسجّل) | — |
| **G0-07** typing/lint/perf baseline | لا توجد أدوات static مهيأة في المشروع | — | — | — | — | **NOT ADDRESSED** (لا توجد أداة موجودة لتشغيلها) | — |

---

## D. Dependency contract

| | Minimum = الحدود المعلنة | Reference |
|---|---|---|
| Python | 3.11 (مُشغَّل على 3.11.16) | 3.12 (مُشغَّل على 3.12.3) |
| NumPy | 1.26.4 | 2.5.2 |
| pandas | 2.1.4 | 3.0.5 |
| SciPy | 1.11.4 | 1.18.0 |
| scikit-learn | **1.6.1** (كان `>=1.4` دون اختبار) | 1.9.0 |
| pytest (`dev`) | 8.0.0 | 8.3.3 |
| `viz`: matplotlib | 3.9.0 | 3.11.2 |
| `boosting`: xgboost / lightgbm / catboost | 2.0.0 / 4.0.0 / 1.2 | 3.4.1 / 4.7.0 / 1.2.10 |
| build / twine (`dev`، أدوات إصدار) | غير مثبتة الحد؛ مُشغَّلة: build 1.6.1، twine 7.0.0 | — |

**مبرر الحدود:**

- **sklearn 1.6.1:** على 1.5.2 يرفض ExtraTrees قيم NaN، فيصبح الإعلان `handles_missing_values=True` كاذبًا، ويفشل 43 اختبارًا في ستة ملفات.
- **SciPy 1.11:** يعمل بعد إصلاح `moment`.
- **pandas 2:** أُبقي مدعومًا بإصلاحين بدل الهروب إلى pandas 3.
- **الحدود العليا:** لا توجد، و`test_dependency_contract` يمنع ظهورها.

**المنصات المختبرة فعلًا:**

- **Windows 11:** نعم، أربع بيئات إضافة إلى packaging.
- **Linux:** لا؛ الـworkflow معرَّف لكنه لم يُشغَّل.
- **macOS:** لا.

**Static checks:** لا يهيئ المشروع أي أداة lint أو type (لا mypy ولا ruff ولا flake8 في `pyproject.toml` أو `CONTRIBUTING.md`). لا يوجد ما «يوجد أصلًا» ليُشغَّل. flake8 وmypy مثبتان عرضيًا في interpreter الجهاز، ولم يُستخدما كبوابة.

---

## E. WIP closure

- **الوصف:** `check_multicollinearity` هو الفحص الثاني عشر الافتراضي. يحسب لكل عمود رقمي `VIF = 1/(1−R²)` بـ`lstsq` مقابل الأعمدة الرقمية الأخرى. العتبة `multicollinearity_vif_threshold=10`.
- **ما أضافته التقوية:**
  - حد الكلفة: 50 عمودًا، وإلا يُصدر INFO `multicollinearity_not_assessed`. و20,000 صف بعينة deterministic متساوية التباعد، تُسجَّل في `rows_used`/`rows_available`.
  - توسيط وتحجيم قبل أي جمع، فلا overflow، والنتيجة مستقلة عن الوحدات.
  - استبعاد عمود لا يُمثَّل float64 أو فيه inf.
  - رفض عتبة NaN أو inf أو من نوع غير رقمي.
- **Fingerprint migration:** سلسلة من ثلاث خطوات: S9، ثم VIF، ثم cv_folds. لكل خطوة بصمتان حرفيتان وrollback لا يلغي إلا تغييرها. التراجع من الأحدث إلى الأقدم على الـfixture المحفوظ يعيد كل بصمة سابقة بالضبط (`test_undoing_each_step_reproduces_the_identity_before_it`). `schema_version` بقي `1.0`، لأن حقلًا موجودًا لم يتغير معناه، ويمكن تمييز artifact قديم من الملف وحده.
- **تغييرات Golden:**
  - VIF: سطران في كل fixture، هما العتبة الجديدة وconfig fingerprint.
  - cv_folds: سطران، هما حذف `cv_folds` وconfig fingerprint.
  - لا finding تغيّر؛ `lineage.json` لم يتغير؛ توليدان متتاليان متطابقان بايتًا ببايت.
  - أُضيف fixture لـpandas 2 (`expected_audit_semantic.pandas2.json`). `test_golden_fixtures` يثبت أن الفرق بين الـfixture ين هو خمسة `pandas_dtype` (`str`↔`object`) والبصمتان المشتقتان منها فقط.
- **عدم وجود regression:** على REF في نقطة التفتيش `798e21b`: 3830 passed و0 failed و0 errors. والتشغيلات النهائية نظيفة (القسم B).

---

## F. Atomicity evidence (`tests/unit/test_publication.py`، و5 اختبارات CLI)

- **Failure injection:** مع نسخة سابقة وبدونها، على كل خطوة:
  - كل واحدة من الكتابات الخمس (3 ملفات، manifest، مسودة المؤشر).
  - كل `fsync`.
  - مزامنة المجلد.
  - `rename` من staging.
  - `os.replace` لـ`CURRENT`.
  - فشل مزامنة بعد الـreplace (يبقى النشر، لأن الـreplace هو نقطة الالتزام).
  - ملف staged لا يطابق المقصود (يُكشف بالتحقق قبل النقل).
  - بعد كل حالة: القارئ يرى النسخة القديمة كاملة، أو `NoPublishedRunError`، ولا يبقى staging أو مسودة.
- **Process kill:**
  - قتل أثناء الكتابة بعد نصف الملف الثاني.
  - قتل قبل الـcommit، فيبقى run كامل غير مُشار إليه ويُتجاهل.
  - `kill` في أربع لحظات عشوائية أثناء نشر متكرر لملفات حجم كل منها 300 KB.
- **Concurrent writers:**
  - أربع عمليات حقيقية تنشر في المجلد نفسه، والأب يقرأ باستمرار.
  - لوحظ على الأقل ثلاثة كتّاب مختلفين صاروا current.
  - كل قراءة كانت متسقة، و**30 run** نُشرت كلها كاملة وبـhashes صحيحة.
  - CLI: عمليتا audit حقيقيتان بالتزامن نشرتا run ين كاملين.
- **Tamper detection:** كلها تُرفض بـ`CorruptPublicationError`:
  - byte معدّل، ملف مبتور، ملف مفقود، ملف زائد.
  - manifest معدّل.
  - ملف وmanifest مزوّران معًا (يكشفهما digest المؤشر).
  - `CURRENT` فارغ أو غير JSON أو مصفوفة، أو يحتوي path traversal، أو يشير إلى run غير موجود، أو بلا digest.
  - run نُقل باسم آخر.
- **Windows PermissionError:**
  - قفل عابر يُعاد، وقفل دائم يفشل نظيفًا والنسخة القديمة current.
  - اختبار **حقيقي غير محاكى**: العملية تفتح `CURRENT` فيرفض Windows الـreplace، والنشر ينتظر ثم ينجح.
- **ثبات old-or-new:** كل اختبار ينتهي بـ`_generation()`، الذي يتحقق أن الملفات الثلاثة تحمل الجيل نفسه.
- **Mutation:**
  - كتابة `CURRENT` قبل كتابة الملفات → 30 فشلًا.
  - حذف التحقق قبل النقل → فشلان.
- **قرار التزامن:** «آخر commit كامل يفوز» بلا قفل. القفل كان سيضيف حالة فشل (عملية مقتولة تمسك القفل) ولا يمنع أي حالة يمكن أن يراها القارئ.

---

## G. Leakage evidence (`tests/integration/test_final_evaluation.py`، 39 اختبارًا)

**الحدود:**

| القسم | fit | ranking | يُقاس بـ |
|---|---|---|---|
| training | نعم | لا | — |
| validation | لا | نعم | `evaluate()` |
| external test | لا | لا | `evaluate_final()` مرة واحدة |

- **Selection path:** `compare_models` → `select_model` → `train` يرسم الـsplit نفسه. `status["validation_used_for_selection"]` يقارن بصمتي صفوف الـsplit (لا الإعداد) ويقول إن كانت صفوف validation قد اختارت النموذج.
- **Final evaluation path:** `evaluate_final()` يقيس النموذج على الـtest الخارجي بـ`transform` فقط، ويعدّ صفوف الـtest المطابقة حرفيًا لصفوف training، ثم **يجمّد التجربة**. بعدها يرفض `compare_models` و`select_model` (حتى للنموذج نفسه) و`train` (حتى بمعاملات جديدة) إلى أن يُحمَّل `load()` جديد. القراءة تبقى مسموحة، وفشل `evaluate_final` لا يجمّد.
- **الاختبارات السلبية:**
  - test frame مسموم: labels مقلوبة، قيم مضروبة في 10⁶، فئة لا توجد إلا فيه.
  - يترك quality وplans وcomparison وvalidation والتنبؤات مطابقة تمامًا لجلسة بلا test.
  - وفي الجلسة نفسها تنهار دقته النهائية بأكثر من 0.3، دليلًا على أن السم كان مرئيًا لو قُرئ.
  - الفئة المسمومة لا تدخل الـvocabulary، ومخرَج الـpreprocessor لإطار ثابت لا يتغير.
  - Mutants تسرّب الـtest إلى training أو إلى comparison تُفشل 8 و5 اختبارات.
- **لا مسارات أخرى:** لا يوجد hyperparameter search ولا threshold search ولا cross-validation في الحزمة. `cv_folds` أُزيل، والـfacade لا يملك `tune` أو `grid_search` أو `cross_validate` أو `select_threshold` (مُختبَر).
- **Determinism:** 344 اختبارًا حساسًا شُغّلت تحت ثلاثة `PYTHONHASHSEED` (0 و1 و987654)، وأعطت نتائج متطابقة وكلها passed. جلستان متطابقتان تعطيان `evaluate_final` متطابقًا.

---

## H. Historical failures

| الفشل في التقرير المرجعي | المصير |
|---|---|
| **Golden** `evidence_still_matches` و`canonical_bytes` (2) | على V كان سببها WIP؛ أُغلقت بالـmigration وإعادة التوليد. على L كان السبب dtype pandas 2؛ أُغلقت بـfixture pandas 2 مع قيد على الفرق. تمر في كل البيئات. |
| **Migration** (2) | صارت سلسلة من ثلاث خطوات على الـfixture المحفوظ. تمر. |
| **Encoding** `committed_report_renders` (1) | `encoding="utf-8"`. تمر على cp1252. |
| **LR truncation** (4) | الحماية تُختبر دون شرط؛ سلوك sklearn يُختبر بفرعيه. تمر على 1.6.1 و1.8.0 و1.9.0. |
| **HGB window shape** (2) | الشيء نفسه. تمر. |
| **Overflow** (3) | `core.counting` وإصلاح `_sort_key`. تمر على pandas 2.1.4 و2.3.3 و3.0.5. |
| **subprocess WinError 10106** (12 error) | `isolated_env()`. صفر errors. |
| **packaging/provenance artifact** (2، على V فقط) | كان أثرًا من أداة الفحص (شجرة بلا metadata). أداة التشغيل تثبّت الشجرة `--no-deps` في كل venv. تمر، ولم يكن عيبًا في المكتبة. |

**جرد الاختبارات:**

- الـbaseline فيه 3821 ID، والنهائي 4097 (بلا extras).
- 13 ID من الـbaseline لم تعد موجودة، ولا أحد منها حُذف بلا بديل:
  - `test_rerunning_overwrites_cleanly`: أُعيدت تسميته `test_rerunning_publishes_a_second_complete_run` لأن العقد تغيّر.
  - 8 من `TestTheMigration::*`: أُعيد توزيعها على `TestTheChain` و`TestTheCurrentCode` و`TestS9`، بالتأكيدات نفسها أو أقوى.
  - 3 من `test_cross_validation_needs_at_least_two_folds[...]`: الحقل أُزيل، وحلّت محلها `TestCvFoldsIsGone`.
  - `TestNumericTextThreshold::test_the_threshold_is_configurable`: أعاد WIP المالك تسميته إلى `test_a_stricter_threshold_still_fires`.
- لا اختبار تحوّل إلى skip، ولم يُضف أي `skipif` لإخفاء فشل. الـ`skipif` الوحيدان الجديدان (غير Windows / Windows فقط) يختاران المنصة في `test_isolated_env` و`test_publication`.

---

## I. Final gate

```text
G0_FINAL_GATE = BLOCKED

PASS_CONDITIONS_MET:
- صفر errors في كل البيئات المُشغَّلة (MIN، REF، L، MIN+extras، REF+extras)
- صفر failures غير مقبولة (فشل التوقف تحت الحمل مُفسَّر، ومُعاد في تشغيل منفرد: 0 failed)
- لا regression جديد؛ جرد الاختبارات مُفسَّر ID بـID
- Windows مثبت بتشغيل فعلي
- بيئة الحد الأدنى تمر (Py 3.11 / numpy 1.26.4 / pandas 2.1.4 / scipy 1.11.4 / sklearn 1.6.1)
- البيئة المرجعية تمر
- WIP VIF مغلق (حرفيًا + تقوية + migration)
- Overflow مغلق (أعداد صحيحة ضخمة + floats قرب 1e308)
- Data Leakage مغلق (G0-01)
- atomic grouped publication مثبت (Windows)
- concurrency مثبت (Windows، عمليات حقيقية)
- cv_folds محسوم (أُزيل مع migration)
- ادعاءات الإصدار صحيحة ومحمية باختبار
- اختبارات visualization شُغّلت فعلًا (matplotlib 3.9.0 و3.11.2)
- كل التغييرات committed
- لا push ولا merge ولا release ولا tag

OPEN_BLOCKERS:
- Linux: لا يوجد تشغيل فعلي. الجهاز بلا WSL أو Docker أو Podman، والـpush ممنوع فلا يعمل
  .github/workflows/ci.yml. مسار _fsync_directory على POSIX لم يُنفَّذ فعليًا قط.
- Worktree: 4 ملفات untracked في docs/ و35 PDF في docs/book/، يملكها المالك وكانت موجودة
  قبل G0. تحتاج قرارًا منه: commit أو .gitignore أو نقل.
- (مسجّل، خارج شروط هذا التكليف) G0-05 (حدود حجم CSV) وG0-07 (typing/lint/perf) من خطة
  التنفيذ لم تُنفَّذ.

READY_FOR_G1 = NO
```
