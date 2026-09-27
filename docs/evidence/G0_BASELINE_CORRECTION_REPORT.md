# تقرير تصحيح وتثبيت Baseline — AIDatasetKit

> نسخة محفوظة من التقرير الذي سُلّم في المحادثة بتاريخ 2026-09-27 قبل بدء G0. لم
> يكن مكتوبًا على القرص، فحُفظ هنا ليصبح المرجع قابلًا للتدقيق من المستودع. لم يُعدَّل
> مضمونه؛ ما صحّحته مرحلة G0 لاحقًا مسجل في `G0_FINAL_CERTIFICATION_REPORT.md`.

## 1. Baseline لـHEAD النظيف وWIP

- `HEAD = 59bb28639b01982e3a1fa882317e1054acb14bea` على `main`؛ لا stash ولا tags.
- WIP: 8 ملفات، `216 insertions(+), 7 deletions(-)`؛ يضيف فحص multicollinearity
  (VIF) كفحص جودة ثانٍ عشر، وإعداد `KitConfig.multicollinearity_vif_threshold`، و13
  اختبارًا.
- العزل: `git archive 59bb286… | tar -x` إلى scratchpad؛ تحقق من
  `aidatasetkit.__file__` في كل تشغيل.
- الأمر: `python -m pytest -p no:cacheprovider -q -rfE --junitxml=<x>.xml` مع
  `PYTHONDONTWRITEBYTECODE=1`.
- **L (المحلية):** Python 3.12.3، numpy 2.4.1، pandas 2.3.3، scipy 1.17.0،
  scikit-learn 1.8.0، pytest 8.3.3.
- **V (الموثقة في RELEASE_NOTES):** numpy 2.5.2، pandas 3.0.5، scipy 1.18.0،
  scikit-learn 1.9.0، في venv معزول.

| التشغيل | collected | سطر pytest النهائي |
|---|---|---|
| HEAD / L | 3821 | `14 failed, 3751 passed, 45 skipped, 12 errors in 431.52s` |
| WIP / L | 3834 | `14 failed, 3764 passed, 45 skipped, 12 errors in 425.63s` |
| HEAD / V | 3821 | `3 failed, 3762 passed, 45 skipped, 12 errors in 500.80s` |
| WIP / V | 3834 | `5 failed, 3773 passed, 45 skipped, 12 errors in 496.89s` |

## 2. تفسير 3834 مقابل 3835

من نفس التشغيل (WIP / L). `test_visualization_renderer.py` يُتخطّى على مستوى الملف
(matplotlib غير مثبتة)، فيُحسب ضمن الـ45 skipped ولا يدخل في collected. JUnit
`tests=3835`؛ الحالة الوحيدة غير المجموعة هي `tests.unit.test_visualization_renderer`
(`collection skipped`). نفس النمط في HEAD: 3821 مقابل 3822.

## 3. مقارنة الفشلات

| المجموعة (العدد) | HEAD/L | WIP/L | HEAD/V | WIP/V | السبب | التصنيف |
|---|---|---|---|---|---|---|
| Golden `evidence_still_matches` + `canonical_bytes` (2) | FAIL | FAIL | PASS | FAIL | L: dtype `object` مقابل `str` (pandas 3). V: `config.fingerprint` تغيّر بسبب الإعداد الجديد | سببه WIP |
| `capability_fingerprint_migration` (2) | FAIL | FAIL | PASS | FAIL | نفس السبب | سببه WIP |
| Golden `committed_report_renders` (1) | FAIL | FAIL | FAIL | FAIL | `read_text()` بلا encoding مع cp1252 | HEAD؛ عيب اختبار Windows |
| LR `rank_` truncation (4) | FAIL | FAIL | PASS | PASS | يتطلب scikit-learn 1.9.0 | اختبار يعتمد على سلوك خارجي |
| HGB `window shape` (2) | FAIL | FAIL | PASS | PASS | يتطلب scikit-learn 1.9.0 | اختبار يعتمد على سلوك خارجي |
| `OversizedNumbers…OverflowError` (3) | FAIL | FAIL | PASS | PASS | `profiler.py:328` `value_counts` على pandas 2.2.3/2.3.3 | خلل مكتبة ضمن `pandas>=2.1` |
| subprocess `WinError 10106` (12 ERROR) | ERROR | ERROR | ERROR | ERROR | غياب `SystemRoot` | HEAD؛ عيب اختبار Windows |
| `packaging` + `provenance` (2) | PASS | PASS | FAIL | PASS | نسخة `git archive` بلا metadata | أثر أداة الفحص، ليس finding |

## 4. الاعتماديات وWindows

- LR/HGB: scikit-learn 1.9.0 وحده هو المحدِّد (إبدال 1.8.0 يُزيل السلوكين؛ إبدال
  scipy 1.17.0 أو numpy 2.4.1 لا يغيّر). لا truncation في 1.5.2، 1.6.1، 1.7.0–1.7.2،
  1.8.0. البيئة المحلية **أقدم** من الموثقة.
- على sklearn 1.5.2 / pandas 2.2.3 / numpy 2.1.3 / scipy 1.14.1: 43 فشلًا في 6 ملفات؛
  الحد الأدنى المعلن `scikit-learn>=1.4` غير مدعوم فعليًا.
- subprocess: `{PYTHONHASHSEED, PATH:""}` → `WinError 10106`؛ إضافة `SystemRoot`
  وحده → نجاح. السلسلة `sklearn → joblib.memory → asyncio.windows_events →
  _overlapped`. على مستوى الاختبار: `249 passed, 12 errors` → `261 passed`. المكتبة لا
  تستخدم `subprocess` ولا `os.environ`.

## 5. الكتابة الذرية

`_write` يكتب ثلاثة ملفات متتالية بـ`write_text` (يمسح ثم يكتب). `os.replace` لكل ملف
يحل ذرية الملف الواحد فقط، لا اتساق المجموعة. العقد المقترح: staging خاص بالتشغيل +
manifest بـsha256 + نقل إلى `runs/<run_id>` + نشر مؤشر `CURRENT` بـ`os.replace`؛ القارئ
يتحقق من الـmanifest والـhashes؛ التزامن بقفل أو «آخر commit كامل يفوز».

## 6. أحكام

- النشر: PyPI وTestPyPI → HTTP 404؛ GitHub Releases → `[]`؛ لا tags على origin.
  «غير منشور على هذه القنوات؛ القنوات الأخرى غير مثبتة».
- G1 ingestion: ABSENT (القارئ الوحيد `read_csv` في CLI). G3: لم يُعَد التحقق.
  اختبارات matplotlib: NOT_RUN.
- `cv_folds` معرّف دون استخدام تنفيذي.
- `G0 = FAIL`؛ لا Gate حصل على PASS.
