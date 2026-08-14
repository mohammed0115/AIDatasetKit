# Online Shoppers: المسار اليدوي مقابل AIDatasetKit

هذه دراسة حالة تعليمية قابلة لإعادة التشغيل على **Online Shoppers Purchasing Intention Dataset**، وهي بيانات عامة حقيقية من مستودع UCI. تقارن الدراسة بناء `scikit-learn` يدوي شفاف بمسار `AIDataFacade` على **المهمة والصفوف والنموذج والإعدادات نفسها**؛ وليست منافسة أداء أو ادعاء أن المكتبة تختار نموذجًا «أفضل» تلقائيًا.

## المحتويات

| المسار | الغرض |
|---|---|
| `data/online_shoppers_intention.csv` | نسخة CSV الرسمية التي حُملت من UCI؛ SHA-256 هو `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` |
| `online_shoppers_manual_vs_aidatasetkit.ipynb` | الدفتر التنفيذي العربي مع النتائج والرسوم |
| `probe_case_study.py` | فحص مختصر للبيانات وخطة الجودة/المعالجة |
| `run_comparison.py` | تجربة قابلة للتنفيذ تنتج ملخص JSON للمقارنة |

## مصدر البيانات وترخيصها

تتكون البيانات من 12,330 جلسة لتوقع `Revenue` الثنائي. يذكر UCI أن 1,908 جلسة انتهت بشراء و10,422 لم تنتهِ به، وأن الملف لا يحمل قيَمًا مفقودة وفق وصف المصدر. البيانات مرخصة بموجب **CC BY 4.0**؛ لذا يلزم الإسناد عند إعادة استخدامها.[1] [2]

```text
Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset.
UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
```

## إعادة التنفيذ

شغّل من جذر المستودع:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  case_studies/online_shoppers/online_shoppers_manual_vs_aidatasetkit.ipynb \
  --ExecutePreprocessor.timeout=180
```

يبني الدفتر تقسيمًا stratified بنسبة 80/20 وبذرة `42`، ثم يتحقق أن مواقع صفوف التدريب والتقييم في المسار اليدوي تطابق مسار AIDatasetKit. لا توجد cross-validation أو hyperparameter tuning أو AutoML، ولا ينبغي تفسير النموذج على أنه إثبات سببي أو توصية تجارية جاهزة للتنفيذ.

## نقطة التعلّم الأساسية

على الخبير اليدوي تحديد أدوار الخصائص، وبناء `ColumnTransformer` و`Pipeline`، وحماية حدود التدريب/التقييم، وتوثيق البذرة والمقياس والمعلمات. تسجل AIDatasetKit profile وفحص الجودة وخطط preprocessing المعتمدة على capabilities وبصمة التقسيم والمقارنة؛ لكنها لا تستبدل الحكم البشري على معنى الأعمدة أو تكلفة الأخطاء أو صلاحية النتيجة خارج هذه البيانات.

## المراجع

[1] [UCI — Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention)

[2] [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
