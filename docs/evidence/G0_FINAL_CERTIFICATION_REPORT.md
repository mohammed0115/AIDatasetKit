# AIDatasetKit — G0 Final Certification Report

```text
CURRENT_GATE           = G0
CURRENT_GATE_STATUS    = PASS
G0_PROGRESS_PERCENT    = 100%   (16/16 PASS)
OVERALL_GATES_PASSED   = 1
OVERALL_GATES_TOTAL    = 13
READY_FOR_NEXT_GATE    = YES — G1 may start once the owner authorises it; nothing of G1 has started
NEXT_AUTHORIZED_ACTION = none. G0 is closed; G1 needs a separate authorisation
```

هذا التقرير يحل محل نسخة `9d96e89`، التي انتهت بـ`BLOCKED` لمانعين: لا تشغيل على Linux، وملفات untracked في شجرة العمل. أُغلق المانعان في هذه الجولة، ولم يُمس غيرهما إلا عطل حقيقي كشفه Linux وأُصلح (القسم 5).

كل رقم هنا مستخرج من سجل تشغيل:

- سجلات CI: سجلات GitHub Actions وتقارير JUnit المرفوعة.
- السجلات المحلية: نسخ مختصرة في `docs/evidence/runs/`.
- التفاصيل الكاملة للموجات السابقة (الأخطاء التاريخية، الذرية، التسرب) موجودة في نسخة `9d96e89` من هذا الملف، ولم يتغير منها شيء إلا ما يذكره هذا التقرير.

---

## 1. Repository

| Field | Value |
|---|---|
| `PRE_SHA` | `59bb28639b01982e3a1fa882317e1054acb14bea` |
| `CODE_TESTED_SHA` | `79e7cc82374b468ed58fa4896a1dc2bad7448a46` (CI run 36328760194، ومحليًا على Windows) |
| `FINAL_SHA` | الـcommit الذي يضيف هذا التقرير. الفرق بينه وبين `79e7cc8` توثيق فقط (القسم 2). |
| `ORIGIN_MAIN` | `59bb28639b01982e3a1fa882317e1054acb14bea`، لم يتغير أثناء العمل ولم يُدفع إليه شيء |
| `BRANCH` | محليًا `main`. دُفع إلى الفرع المؤقت `origin/g0-certification` فقط. |
| `AHEAD_BEHIND` | أمام `origin/main` بـ20 commit، وخلفه بـ0 |
| `WORKTREE_CLEAN` | **YES**: `git status --short` فارغ؛ لا تغييرات متتبَّعة ولا ملفات untracked |
| `COMMITS_CREATED_THIS_CLOSURE` | 2: `79e7cc8` (إصلاح عطل Linux) والـcommit الذي يضيف هذا التقرير |
| Draft PR | لم يُفتح. الـworkflow يعمل على `push` لأي فرع، فكفى دفع الفرع المؤقت. |
| merge / tag / release / PyPI | لا شيء |

---

## 2. SHA delta

**`fd4cdd2..9d96e89`، التوثيق فقط.** `git diff --name-status` أعطى:

| File | Change Type | Code/Tests/Docs/CI | Affects Tested Behavior? |
|---|---|---|---|
| `docs/evidence/G0_FINAL_CERTIFICATION_REPORT.md` | A | Docs | No |
| `docs/evidence/runs/final_L.md` | A | Docs | No |
| `docs/evidence/runs/final_L_alone.md` | A | Docs | No |
| `docs/evidence/runs/final_MIN.md` | A | Docs | No |
| `docs/evidence/runs/final_MIN_EXTRAS.md` | A | Docs | No |
| `docs/evidence/runs/final_REF.md` | A | Docs | No |
| `docs/evidence/runs/final_REF_EXTRAS.md` | A | Docs | No |
| `docs/evidence/runs/final_smoke.md` | A | Docs | No |

- الإجمالي: 8 ملفات، 596 إضافة، 0 حذف، ولا ملف خارج `docs/evidence/`.
- الملفات التي تفحصها الاختبارات من `docs/` معروفة بالاسم، وليس بينها ملف مما سبق. تشغيل `test_packaging` و`test_dependency_contract` على الشجرة بعد الإضافة أعطى 99 passed.
- الحكم: docs-only.

**`9d96e89..79e7cc8`، كود واختبارات.** هذا هو إصلاح عطل Linux (القسم 5)، ولذلك أُعيدت **كل** الاختبارات عليه: CI كاملًا على Windows وLinux، إضافة إلى MIN وREF محليًا.

| File | Change Type | Code/Tests/Docs/CI | Affects Tested Behavior? |
|---|---|---|---|
| `aidatasetkit/core/counting.py` | M | Code | Yes: ترتيب القيم المتعادلة فقط؛ الأعداد نفسها لا تتغير |
| `tests/unit/test_counting.py` | M | Tests | Yes |

**`79e7cc8..FINAL_SHA`، التوثيق فقط.**

| File | Change Type | Code/Tests/Docs/CI | Affects Tested Behavior? |
|---|---|---|---|
| `docs/evidence/G0_FINAL_CERTIFICATION_REPORT.md` | M | Docs | No |
| `docs/evidence/G0_UNTRACKED_BACKUP_INVENTORY.md` | A | Docs | No |
| `docs/evidence/runs/ci_github_actions.md` | A | Docs | No |
| `docs/evidence/runs/closure_MIN.md` | A | Docs | No |
| `docs/evidence/runs/closure_REF.md` | A | Docs | No |
| `docs/supported-environments.md` | M | Docs | No: جدول المنصات يذكر التشغيل الفعلي |
| `RELEASE_NOTES_0.1.0a1.md` | M | Docs | No: فقرة المنصات. `test_packaging` و`test_dependency_contract`، اللذان يقرآن هذا الملف، مرّا (99 passed) |

---

## 3. Untracked files

```text
ORIGINAL_COUNT   = 39   (4 × docs/*.md, 35 × docs/book/*.pdf; 263,140,445 bytes)
BACKUP_COUNT     = 39
HASH_MISMATCHES  = 0
SOURCE_REMAINING = 0
BACKUP_PATH      = D:\Flash\AIDatasetKit_owner_untracked_backup_20260927T145943Z\
MANIFEST_PATH    = D:\Flash\AIDatasetKit_owner_untracked_backup_20260927T145943Z\MANIFEST.json
MANIFEST_SHA256  = cdf64be1174cc5509860b769e7fdb875956d7f2cab6aa2fc3a7f259c061d0b8b
RECOVERABLE      = YES — copy each MANIFEST files[].relative_path back to the same path
```

**الطريقة:**

- الـinventory مأخوذ من `git ls-files --others --exclude-standard`، لكل ملف: المسار والحجم وsha256 وآخر تعديل والنوع (المكتشف من الـmagic bytes). محفوظ في `docs/evidence/G0_UNTRACKED_BACKUP_INVENTORY.md`.
- قبل نقل أي ملف، تحقق السكربت من أنه غير متتبَّع ومن أن hash المصدر لم يتغير منذ الـinventory.
- ثم لكل ملف على حدة: `copy2` (يحفظ وقت التعديل)، ثم `fsync`، ثم إعادة حساب sha256 والحجم في الوجهة، ثم حذف المصدر فقط بعد التطابق. أول عدم تطابق يوقف كل شيء دون حذف.
- بعد انتهاء النقل، جرى تحقق مستقل ثانٍ على الملفات الـ39 كلها.
- لم يُستخدم wildcard ولا `git clean`، ولم يُمس أي ملف متتبَّع. المجلد الاحتياطي خارج المستودع ولم يُضَف إلى Git.
- بقي المجلد الفارغ `docs/book/`. Git لا يتتبع المجلدات الفارغة، فلم يُحذف.

---

## 4. Windows

| Run | SHA | Python | Dependencies | Collected | Passed | Failed | Errors | Skipped | Duration | Evidence |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Local MIN | `79e7cc8` | 3.11.16 | numpy 1.26.4، pandas 2.1.4، scipy 1.11.4، sklearn 1.6.1 | 4113 | 4068 | 0 | 0 | 46 | 381.81 s | `runs/closure_MIN.md` |
| Local REF | `79e7cc8` | 3.12.3 | numpy 2.5.2، pandas 3.0.5، scipy 1.18.0، sklearn 1.9.0 | 4113 | 4068 | 0 | 0 | 46 | 386.59 s | `runs/closure_REF.md` |
| CI `windows-latest` ×4 + smoke | `79e7cc8` | 3.11.9 / 3.12.10 | انظر القسم 5 | 4113 / 4135 | انظر القسم 5 | 0 | 0 | — | — | `runs/ci_github_actions.md` |

- الأدلة المحلية السابقة على `fd4cdd2` محفوظة كما هي (`runs/final_*.md`).
- **لم أُعِد** تشغيل الـextras المحلية ولا smoke المحلي على `79e7cc8`. غطتهما وظائف CI على `windows-latest` للـSHA نفسه، ولا أدّعي تشغيلًا محليًا لم يحدث.

---

## 5. Linux (GitHub Actions)

**Run 36328760194**، commit `79e7cc82374b468ed58fa4896a1dc2bad7448a46`، الحصيلة **success** (10/10 وظائف).
https://github.com/mohammed0115/AIDatasetKit/actions/runs/36328760194

| JOB_NAME | Job id | OS | PYTHON | DEPENDENCIES | COLLECTED | PASSED | FAILED | ERRORS | SKIPPED | DURATION |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ubuntu-latest / minimum | 108646519617 | Ubuntu 24.04 | 3.11.16 | numpy 1.26.4، pandas 2.1.4، scipy 1.11.4، sklearn 1.6.1، pytest 8.0.0 | 4113 | 4067 | 0 | 0 | 47 | 129 s |
| ubuntu-latest / minimum + extras | 108646519605 | Ubuntu 24.04 | 3.11.16 | ما سبق مع matplotlib 3.9.0، xgboost 2.0.0، lightgbm 4.0.0، catboost 1.2 | 4135 | 4091 | 0 | 0 | 44 | 152 s |
| ubuntu-latest / reference | 108646519530 | Ubuntu 24.04 | 3.12.14 | numpy 2.5.2، pandas 3.0.5، scipy 1.18.0، sklearn 1.9.0، pytest 8.3.3 | 4113 | 4067 | 0 | 0 | 47 | 186 s |
| ubuntu-latest / reference + extras | 108646519507 | Ubuntu 24.04 | 3.12.14 | ما سبق مع matplotlib 3.11.2، xgboost 3.4.1، lightgbm 4.7.0، catboost 1.2.10 | 4135 | 4091 | 0 | 0 | 44 | 234 s |
| ubuntu-latest / build, install and smoke | 108646519376 | Ubuntu 24.04 | 3.12.14 | build 1.6.1، twine 7.0.0 | — | wheel ✓، sdist ✓، audit من الحزمة المثبتة ✓ | 0 | — | — | 60 s |
| windows-latest / minimum | 108646519603 | Windows | 3.11.9 | MIN | 4113 | 4068 | 0 | 0 | 46 | 236 s |
| windows-latest / minimum + extras | 108646519588 | Windows | 3.11.9 | MIN + extras | 4135 | 4092 | 0 | 0 | 43 | 261 s |
| windows-latest / reference | 108646519557 | Windows | 3.12.10 | REF | 4113 | 4068 | 0 | 0 | 46 | 292 s |
| windows-latest / reference + extras | 108646519629 | Windows | 3.12.10 | REF + extras | 4135 | 4092 | 0 | 0 | 43 | 308 s |
| windows-latest / build, install and smoke | 108646519479 | Windows | 3.12.10 | build 1.6.1، twine 7.0.0 | — | `twine check` PASSED ×2، wheel ✓، sdist ✓ | 0 | — | — | 145 s |

- **المدة:** مدة الوظيفة من بدئها إلى انتهائها، وتشمل التثبيت. الأعداد مأخوذة من سطر pytest الأخير في السجل، ومطابقة لتقرير JUnit المرفوع.
- **الـskip الزائد على Linux:** Linux يتخطى 47 مقابل 46 على Windows. السبب: اختباران لـWindows فقط يُتخطيان هناك (القفل الحقيقي على `CURRENT`، و`WinError 10106`)، بينما يُشغَّل اختبار خاص بغير Windows.

**تحقق بالاسم من تقارير JUnit لوظائف ubuntu الأربع:**

| المجال | ما مرّ |
|---|---|
| publication | 72 passed، و1 skip خاص بـWindows |
| failure injection | 30 |
| directory-sync | 4 بالحقن. ومسار `_fsync_directory` الحقيقي على POSIX نُفِّذ في كل نشر آخر؛ **أول تنفيذ فعلي له** |
| process kill | 6 |
| concurrent writers | 1 |
| tampering | 15 |
| CLI all-or-nothing وaudit متزامن | 7 |
| Golden (fixture pandas 2 على MIN، وpandas 3 على REF) | 7 + 5 + 6 |
| tie order | 7 |
| final evaluation / leakage | 39 |
| packaging | 73 |

صفر failures وصفر errors في كل تقرير.

### Linux failures and repairs

**Run 36327967662** على `9d96e89`، الحصيلة failure، 9/10 وظائف ناجحة.

- الوظيفة التي فشلت: `ubuntu-latest / minimum + extras` (job 108644293697): `3 failed, 4072 passed, 44 skipped`.
- الاختبارات الثلاثة: `test_the_evidence_still_matches_it_exactly`، `test_the_canonical_bytes_match_too`، `test_it_reproduces_the_fixture_for_this_pandas_major`.
- الفرق الوحيد: `dominant_value_digest` للعمود `CustomerID`.

**التشخيص:**

- وظيفة `ubuntu-latest / minimum` نجحت بالإصدارات نفسها تمامًا (numpy 1.26.4، pandas 2.1.4)، فالـextras ليست السبب.
- `CustomerID` فريد في كل صف، فكل قيمه متعادلة بعدد 1.
- في pandas 2.1.4 يستدعي `value_counts` الدالة `sort_values(ascending=...)` دون `kind`، أي quicksort غير مستقر.
- NumPy 1.25+ يوجّه هذا الترتيب وقت التشغيل إلى تنفيذ خاص بالمعالج (AVX-512 حيث يتوفر؛ جهاز التطوير بلا AVX-512 حسب `np.show_runtime()`).
- إذن «القيمة الغالبة»، أي أول عنصر، كانت تعتمد على نوع معالج الـrunner.

**التصنيف:** library defect، لأن الـartifact لم يكن deterministic على pandas 2.1.

**الإصلاح (`79e7cc8`):** `core.counting.value_counts` يأخذ كل الأعداد من pandas كما هي. عند وجود تعادل فقط، يرتّب المتعادلات بترتيب الظهور الأول، وهو ما يعيده pandas 3 وما يحتويه الـfixture ان المحفوظان أصلًا، فلم يتغير أي fixture.

**Regression test:** `TestTieOrderDoesNotDependOnTheMachine` يجعل pandas يعيد المتعادلات معكوسة كما قد يفعل ترتيب غير مستقر، ويشترط ثبات الـtally وقيمة `dominant_value` وهوية الـGolden. على الكود السابق فشل 4 من 7، بينها هوية الـGolden، أي أن عطل CI أعيد إنتاجه.

**التحقق:** focused على MIN وL وREF محليًا (54 passed)، ثم المصفوفة كاملة في run 36328760194، ثم MIN وREF محليًا بكاملهما.

لا يوجد أي skip أو xfail جديد.

---

## 6. Gate decision

```text
G0_FINAL_GATE = PASS
READY_FOR_G1  = YES — subject to the owner's separate authorisation; G1 not started
```

| شرط PASS | الدليل |
|---|---|
| Linux ناجح فعليًا | run 36328760194: 4 وظائف ubuntu للاختبار ووظيفة smoke، كلها success |
| Windows ما زال ناجحًا | 4 وظائف windows-latest وsmoke في CI، إضافة إلى MIN وREF محليًا على `79e7cc8` |
| CI اختبر FINAL_SHA أو SHA مطابقًا وظيفيًا | CI اختبر `79e7cc8`، و`79e7cc8..FINAL_SHA` توثيق فقط (القسم 2) |
| صفر failures | كل التشغيلات على `79e7cc8` |
| صفر errors | كل التشغيلات على `79e7cc8` |
| لا regression جديد | العطل الوحيد الذي ظهر أُصلح بـregression test؛ لم يتغير أي fixture؛ لا skip جديد |
| worktree نظيف | `git status --short` فارغ |
| الملفات غير المتتبعة محفوظة وقابلة للاستعادة | 39/39، و0 hash mismatch، والـmanifest موجود |
| التقرير committed | هذا الملف في FINAL_SHA |
| لا push إلى main، ولا merge، ولا release | `origin/main` = `59bb286`؛ لا PR؛ لا tag |

---

## 7. PROJECT PLAN PROGRESS

### 7.1 تقدم G0 التفصيلي

| ID | المهمة | التنفيذ | الاختبار | الحالة | الدليل/المانع |
|---|---|---|---|---|---|
| G0-0 | تثبيت Pre-state وBaseline | `G0_PRE_EXECUTION_REPORT.md` و`G0_BASELINE_CORRECTION_REPORT.md` | hash الـWIP أُعيد إنتاجه بايتًا ببايت | PASS | `173f3fc`، `0afd53f` |
| G0-1 | بيئة مرجعية واعتماديات مقفلة | `constraints/minimum.txt` و`reference.txt` | `test_dependency_contract` (26)؛ CI يثبّت منهما | PASS | `d9e13c8`، run 36328760194 |
| G0-2 | CI فعلي على Windows وLinux | `.github/workflows/ci.yml`: 8 خلايا و2 smoke | 10/10 success على `79e7cc8` | PASS | run 36328760194 |
| G0-3 | إصلاح subprocess على Windows | `isolated_env()` مع `SystemRoot` فقط | 12 errors → 0؛ `test_isolated_env` (6) | PASS | `64098a5` |
| G0-4 | فرض UTF-8 في fixtures وGolden | `encoding="utf-8"` صريح | يمر على cp1252 محليًا وعلى Linux | PASS | `3700cad` |
| G0-5 | إغلاق VIF WIP وfingerprint migration | WIP حرفيًا، ثم تقوية، ثم سلسلة migration | 15 اختبارًا تفشل على WIP وتمر بعد التقوية؛ Golden على المنصتين | PASS | `0afd53f`، `a5e8656`، `798e21b` |
| G0-6 | إصلاح Overflow | `core/counting.py`؛ التحجيم في `statistics` | `test_counting` و`test_extreme_values` على pandas 2 و3، Windows وLinux | PASS | `6d57e8d`، `4852539`، `79e7cc8` |
| G0-7 | تثبيت عقد الاعتماديات | الحدود = الحد الأدنى المُختبَر، بلا حدود عليا | MIN وREF ومعهما extras، على المنصتين | PASS | `d9e13c8`، run 36328760194 |
| G0-8 | تصحيح اختبارات scikit-learn | الحماية دون شرط، وسلوك الاعتماد بفرعيه | تمر على 1.6.1 و1.8.0 و1.9.0 | PASS | `3362ff9`، `e6f704b` |
| G0-9 | منع Data Leakage | `evaluate_final()` مع تجميد، و`validation_used_for_selection` | 39 اختبارًا؛ mutants تفشل 8 و5؛ تمر على Linux | PASS | `e64fa29` |
| G0-10 | Atomic grouped output | `publish_run` و`read_current` و`CURRENT` | 73 اختبارًا (منها 30 failure injection، و6 kill، و15 tamper)؛ Windows وLinux | PASS | `5ae0a87`، `17d2baf`، run 36328760194 |
| G0-11 | Concurrent writers contract | آخر commit كامل يفوز، بلا قفل | 4 كتّاب حقيقيون، و30 run؛ audit CLI متزامن؛ القفل الحقيقي على Windows | PASS | `17d2baf`، run 36328760194 |
| G0-12 | حسم `cv_folds` | أُزيل، مع خطوة migration | `TestCvFoldsIsGone` وخطوة السلسلة | PASS | `48999fa` |
| G0-13 | تصحيح ادعاءات الإصدار | أُزيلت «signed-off» و«public alpha» والتثبيت من index | `TestReleaseClaimsMatchTheEvidence` يلتقط كل ادعاء قديم | PASS | `448a244` |
| G0-14 | Full certification matrix | Windows محليًا، وCI على Windows وLinux | 10/10 وظائف، ومحليًا MIN وREF، والكل 0 failed و0 errors | PASS | run 36328760194، `runs/closure_*.md` |
| G0-15 | ختم G0 النهائي | شجرة نظيفة، ملفات محفوظة، تقرير committed | القسم 6 | PASS | FINAL_SHA |

```text
G0_ITEMS_PASS       = 16/16
G0_ITEMS_BLOCKED    = 0/16
G0_PROGRESS_PERCENT = 100%
```

هناك بندان من خطة التنفيذ الأصلية (`AIDatasetKit_Execution_Plan_AR.md`)، أصبحت هذه الخطة الآن في النسخة الاحتياطية:

- **G0-05** (حدود حجم CSV والذاكرة)
- **G0-07** (typing/lint/perf baseline)

لم يكونا ضمن الـ16 بندًا المعتمدة لهذا التكليف، ولم يُنفَّذا. يُذكران هنا حتى لا يُفهم الختم على أنه غطاهما.

### 7.2 تقدم الخطة العامة للمكتبة

| Gate | المجال | الحالة | نسبة التقدم المثبتة | المانع/الخطوة التالية |
|---|---|---|---:|---|
| G0 | الثبات والاختبارات والعقود الأساسية | PASS | 100% (16/16) | مغلق |
| G1 | إدخال الملفات والبيانات العامة | NOT_STARTED | 0% | ينتظر تفويض المالك |
| G2 | Profiling وجودة البيانات | NOT_STARTED | 0% | ينتظر G1 |
| G3 | Cleaning وTransformation | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G4 | EDA والتحليل الإحصائي | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G5 | Trends وComparisons | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G6 | Segmentation وScoring primitives | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G7 | Multi-dataset وSchema validation | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G8 | Visualization وReporting | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G9 | AI-ready structured context | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G10 | Provenance وSecurity وPrivacy وErrors | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G11 | Performance وHardening وCertification | NOT_STARTED | 0% | ينتظر الخطة المعتمدة |
| G12 | شهادة الاستهلاك من Masari | NOT_STARTED | 0% | ينتظر جميع البوابات المطلوبة |

- قدرات موجودة في الكود (profiling، quality، visualization planning، evidence) لا تُحتسب تقدمًا لـG2 أو G8 أو G10: تلك البوابات لم تُعرَّف لها معايير قبول معتمدة بعد، والنسبة لا تُرفع بالانطباع.
- Scraping وجمع بيانات المنصات خارج AIDatasetKit، وتقعان ضمن MWIE/Connectors.

### 7.3 الملخص الرقمي

```text
CURRENT_GATE           = G0
CURRENT_GATE_STATUS    = PASS
G0_PROGRESS_PERCENT    = 100%
OVERALL_GATES_PASSED   = 1
OVERALL_GATES_TOTAL    = 13
READY_FOR_NEXT_GATE    = YES (بعد تفويض المالك)
NEXT_AUTHORIZED_ACTION = لا شيء؛ G1 يحتاج تفويضًا منفصلًا
```

---

## 8. المخاطر المتبقية

- **macOS** لم يُشغَّل ولا يُدَّعى. Python 3.13 و3.14 لم يُشغَّلا.
- **Linux runner:** GitHub أعلن أن `ubuntu-latest` سينتقل إلى Ubuntu 26 ابتداءً من 2026-10-19، وأن actions التي تستهدف Node.js 20 تُشغَّل قسرًا على Node.js 24 (annotations في run 36327967662). الخطر الأول يتعلق باستقرار الـCI مستقبلًا لا بالكود؛ ثبّت الصورة `ubuntu-24.04` إن أردت نتائج قابلة للمقارنة.
- **الترتيب deterministic الآن داخل `core.counting`:** أي كود جديد يستدعي `Series.value_counts` مباشرة ثم يعتمد على ترتيب المتعادلات سيعيد العطل. الاستدعاءات الأربعة الموجودة تمر كلها عبر `core.counting`.
- الـruns المنشورة تتراكم دون حذف، ومجلدات staging المتروكة لا تُكنس (قرار موثّق).
- تلميحات `pip install aidatasetkit[...]` في الكود تشير إلى index لا توجد فيه الحزمة بعد.
- الـsuite لم يُشغَّل بـ`-W error`.
- في `ubuntu-latest` ظهر تحذير واحد من matplotlib 3.9.0 (`PyparsingDeprecationWarning`)، مصدره خارجي.
