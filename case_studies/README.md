# دراسات حالة AIDatasetKit

يحتوي هذا المجلد على تطبيقات عملية قابلة لإعادة التشغيل تستخدم بيانات عامة، وتركز على **ما الذي يفعله المحلل البشري** وما الذي تنسقه AIDatasetKit بصورة قابلة للفحص. لا تمثل هذه الدراسات تقييمات إنتاجية أو توصيات قرار آلية.

| دراسة الحالة | المهمة | مصدر البيانات | ما الذي ستتعلمه؟ |
|---|---|---|---|
| [Online Shoppers: خبير البيانات التقليدي مقابل AIDatasetKit](online_shoppers/README.md) | تصنيف ثنائي | [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention) | مقارنة `scikit-learn` اليدوية بمسار AIDatasetKit على الصفوف والمعلمات نفسها |

## نقطة الانطلاق

افتح دفتر [التطبيق العملي](online_shoppers/online_shoppers_manual_vs_aidatasetkit.ipynb) مباشرة من GitHub لمراجعة النتائج المنفذة والرسوم، أو نزّل المستودع وأعد التنفيذ محليًا وفق تعليمات الدراسة. يوضح الدفتر أن المكتبة تسجل profile والجودة وخطط preprocessing وبصمة التقسيم ومقارنة النماذج، بينما يبقى اختيار معنى الأعمدة والمقياس وقرار الاستخدام مسؤولية بشرية.

> لا يوجد في هذه الدراسات cross-validation أو hyperparameter tuning أو AutoML أو استنتاج سببي أو قرار عمل تلقائي، ما لم تنص الدراسة صراحةً على خلاف ذلك.
