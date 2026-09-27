# Masari as a consumer of AIDatasetKit — integration boundaries

AIDatasetKit مكتبة عامة ومستقلة لتحليل البيانات الجدولية. Masari AI **مستهلك محتمل واحد** لها، ويُستخدم هنا لاختبار عمومية المكتبة فقط. هذا الملف يرسم الحدود، ولا يضيف أي منطق.

## 1. Pipeline

```text
MWIE / Connectors   → جمع البيانات، scraping، استخراج المستندات، تطبيعها إلى جداول
AIDatasetKit        → أدلة تحليلية عامة: profiling، جودة، إحصاء، رسوم، artifact قابل للتتبع
OpenAI Agent        → الاستدلال على الأدلة (خارج المكتبة، ولا تستدعيه المكتبة)
Masari              → التنسيق، السياسات، الموافقات، منطق الأعمال
```

## 2. What stays out of AIDatasetKit

**لا يدخل المكتبة أبدًا:**

- أسماء أو منطق خاص بـMasari أو LinkedIn أو مستقل أو Upwork أو التقديم على الوظائف.
- scraping أو جمع بيانات المنصات أو المصادقة عليها. هذا من عمل MWIE/Connectors.
- استخراج النص من PDF أو DOCX أو HTML. يحدث قبل المكتبة، والمكتبة تستقبل جدولًا.
- تسميات أعمال (مثل «فرصة واعدة»)، أو أوزان scoring خاصة بمجال، أو عتبات قرار. تبقى في Masari. المكتبة توفر primitives عامة فقط (percentiles، z-scores، composite metrics) بأوزان يمررها المستدعي.
- استدعاء LLM أو أي خدمة شبكية. المكتبة offline، وهذا مُثبت بفحص الكود واختبار «standalone page».
- الموافقات والسياسات وقرار التنفيذ.

## 3. The contract between them

- **إلى المكتبة:** جداول منظمة. اليوم: `pandas.DataFrame` في Python، أو CSV بفاصلة وUTF-8 عبر CLI. بعد G1: صيغ أخرى وrecords.
- **من المكتبة:** `audit.json`، أي artifact مُرقَّم بإصدار، deterministic، فيه الأدلة والقيود، ومنشور ذريًا خلف `CURRENT`. إضافة إلى كائنات Python قابلة للتسلسل (`to_dict()`).
- **البيانات الخام:** لا تغادر المكتبة إلا حين يطلب المستدعي ذلك صراحة. تنقيح القيم مفعَّل افتراضيًا.

## 4. Answers to the fifteen questions

التصنيف بسُلَّم التدقيق. الأدلة التفصيلية في `docs/evidence/POST_G0_CAPABILITY_REALITY_AUDIT.md` (الصفوف G12-01..15).

| # | السؤال | التصنيف | الجواب بالأدلة |
|---|---|---|---|
| 1 | تمرير بيانات فرص منظمة وتحليلها | `PARTIAL` | profiling والجودة والإحصاء تعمل على أي DataFrame. تنقص قراءة records وملفات غير CSV (G1)، وgroup-by (G3) |
| 2 | تحليل تاريخ الفرص واكتشاف الأنماط | `MISSING` | لا مسار زمني: التواريخ لا تُستنتج من CSV، ولا trends (G5) |
| 3 | مقارنة الفرص الناجحة وغير الناجحة | `PARTIAL` | تدريب تصنيفي، وleakage وimbalance على عمود النتيجة. لا مقارنة وصفية بين segments (G5/G6) |
| 4 | تحليل المهارات المطلوبة | `MISSING` | لا تعامل مع القيم النصية متعددة القيم. تفكيكها إلى صفوف مهمة تطبيع (MWIE)، أو عملية explode عامة في G3 |
| 5 | مدخلات scoring حتمية دون LLM | `PARTIAL` | z-scores وإحصاءات حتمية موجودة. لا percentile ranks ولا composite metrics (G6) |
| 6 | الإيرادات والتكاليف والهوامش والاتجاهات | `MISSING` | لا derived columns ولا time aggregation (G3 وG5) |
| 7 | بيانات الملف المهني بعد تمثيلها جدوليًا | `PARTIAL` | الـprofiling يعمل على أي جدول. التمثيل الجدولي نفسه مسؤولية MWIE |
| 8 | أداء المحتوى | `MISSING` | يحتاج trends ومقارنات زمنية (G5) |
| 9 | رسوم وتقارير لـCommand Center | `PARTIAL` | خطط رسوم (9 أنواع) وHTML audit. لا line أو time series، ولا تصدير صور (G8) |
| 10 | AI-ready structured context | `PARTIAL` | `audit.json` فيه الأدلة والقيود والـprovenance. تنقصه أقسام statistics وtrends وsegments وcomparisons (G9) |
| 11 | تتبع النتائج إلى أدلتها | `SUPPORTED_AND_TESTED` | findings فيها العمود والقاعدة والأرقام، إضافة إلى lineage وfingerprints وmanifest |
| 12 | العمل دون إرسال البيانات إلى LLM | `SUPPORTED_AND_TESTED` | لا كود شبكي، والتقرير HTML مستقل (مُختبَر) |
| 13 | عدة datasets بأمان | `MISSING` | كل استدعاء يعالج إطارًا واحدًا. لا joins ولا تحقق من المفاتيح (G7) |
| 14 | أحجام البيانات الواقعية | `PARTIAL` | مُقاس محليًا: profile لـ100k × 20 = 2.34 s قبل `79e7cc8`، و21.48 s بعده (P0-1). ‏1M × 11 = 298 s اليوم. لا حدود ولا chunking |
| 15 | ما يبقى داخل Masari أو MWIE | `NOT_APPLICABLE` | القسم 2 أعلاه |

## 5. Practical guidance before G1

- حوّل البيانات في MWIE إلى CSV بفاصلة وUTF-8، أو إلى DataFrame. ملف بفاصل `;` يُقرأ اليوم خطأً دون تحذير (P0-2 في التدقيق).
- التواريخ تُحوَّل إلى `datetime64` قبل التمرير، وإلا عوملت كفئات.
- الأحجام فوق 100k صف تستحق الانتظار حتى إصلاح P0-1.
