# AIDatasetKit — Architecture Overview

> ## Status — unreleased alpha `0.1.0a1` (not published to any index; no release, no tag)
>
> **AIDatasetKit is the safety and audit layer for tabular machine learning.**
> *Prove what happened between your data and your model.*
>
> This document is the original design, kept as written. The status block below
> records what has actually been built against it, so a reader can tell the plan
> from the delivery. Sections further down that describe `facade.py`,
> `datasets.py`, training, evaluation, or prediction are **design, not code** —
> they are planned, not shipped.
>
> ### Delivered and verified
>
> | Stage | Package | State |
> |---|---|---|
> | S0 Foundation | `core/` | done |
> | S1 Statistics | `statistics/` | done |
> | S2 Profiling + data quality | `profiling/` | done |
> | S2.5 Smart visualization | `visualization/` | done |
> | S3 Model architecture | `models/` | done |
> | S4 Intelligent preprocessing | `preprocessing/` | verified done |
> | S5 Classification algorithms | `models/classification/` | verified done |
> | S5.5 Evidence and provenance | `evidence/`, `cli/` | verified done |
>
> ### The layer added after this document was written
>
> `evidence/` aggregates what every other layer established into one canonical,
> versioned, deterministic audit artifact, and `cli/` exposes it as
> `aidatasetkit audit`. Evidence recomputes nothing, and **nothing depends on
> evidence** — the import direction is enforced by
> `tests/integration/test_architecture_boundaries.py`:
>
> ```
> core → statistics → profiling → preprocessing / models / visualization
>                                          ↓
>                                      evidence → cli
> ```
>
> ### Still planned
>
> S6 regression · S7 training and evaluation · S8 the `AIDataFacade` this
> document opens with · S9 clustering · S10 anomaly detection and dimensionality
> reduction · external model backends · deep learning.
>
> None of these are cancelled. The order changed: an audit layer real users try is
> worth more now than a larger catalogue nobody has.

---


## الفكرة العامة

الهيكل المقترح للمشروع هو: **مكتبة واحدة من الخارج، لكن داخليًا مقسمة إلى محركات مستقلة**.

المستخدم العادي يتعامل غالبًا مع `AIDataFacade` فقط، بينما يستطيع المتخصص استخدام أي طبقة مباشرة عندما يحتاج تحكمًا أدق.

---

## Project Structure

```text
aidatasetkit/
│
├── __init__.py
├── facade.py                    # الواجهة الرئيسية للمستخدم
├── datasets.py                  # datasets تجريبية للاختبارات والأمثلة
│
├── core/                        # الأساس المشترك للمكتبة
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── types.py
│   ├── schema.py
│   └── arrays.py
│
├── statistics/
│   ├── __init__.py
│   ├── engine.py
│   ├── moments.py
│   ├── frequency.py
│   └── bivariate.py
│
├── profiling/
│   ├── __init__.py
│   ├── profiler.py
│   ├── quality.py
│   ├── checks.py
│   └── task_detector.py
│
├── preprocessing/
│   ├── __init__.py
│   ├── feature_detector.py
│   ├── plan.py
│   └── builder.py
│
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── capabilities.py
│   ├── registry.py
│   ├── factory.py
│   ├── assembly.py
│   │
│   ├── classification/
│   │   ├── dummy.py
│   │   ├── logistic.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── extra_trees.py
│   │   ├── gradient_boosting.py
│   │   ├── hist_gradient_boosting.py
│   │   ├── knn.py
│   │   └── gaussian_nb.py
│   │
│   ├── regression/
│   │   ├── dummy.py
│   │   ├── linear.py
│   │   ├── ridge.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── extra_trees.py
│   │   └── gradient_boosting.py
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── kmeans.py
│   │   ├── mini_batch_kmeans.py
│   │   ├── dbscan.py
│   │   ├── optics.py
│   │   ├── agglomerative.py
│   │   └── birch.py
│   │
│   ├── anomaly_detection/
│   │   ├── __init__.py
│   │   ├── isolation_forest.py
│   │   ├── local_outlier_factor.py
│   │   ├── one_class_svm.py
│   │   └── elliptic_envelope.py
│   │
│   ├── dimensionality_reduction/
│   │   ├── __init__.py
│   │   ├── pca.py
│   │   ├── truncated_svd.py
│   │   ├── fast_ica.py
│   │   └── nmf.py
│   │
│   └── optional/
│       └── xgboost.py
│
├── training/
│   ├── __init__.py
│   └── trainer.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── comparison.py
│   ├── splitters.py
│   └── metrics/
│       ├── __init__.py
│       ├── classification.py
│       ├── regression.py
│       ├── clustering.py
│       ├── anomaly_detection.py
│       └── dimensionality_reduction.py
│
├── prediction/
│   ├── __init__.py
│   ├── builder.py
│   └── validator.py
│
└── future/
    └── deep_learning/
        ├── pytorch/
        └── tensorflow/
```

---

## Data Flow

```text
             CSV / DataFrame
                    │
                    ▼
             AIDataFacade
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
   Profiler     Statistics     Quality
       │            │             │
       └────────────┼─────────────┘
                    ▼
             Task Detector
                    │
                    ▼
             FeatureDetector
                    │
                    ▼
          PreprocessingPlan
                    │
                    ▼
            Model Registry
                    │
     ┌──────────────┼──────────────┬──────────────────┐
     ▼              ▼              ▼                  ▼
Classification   Regression     Clustering      Anomaly Detection
     │              │              │                  │
 Logistic        Linear         KMeans          IsolationForest
 RandomForest    Ridge          DBSCAN          LocalOutlierFactor
 GradientBoost   RandomForest   OPTICS          OneClassSVM
 KNN             ...            Agglomerative   ...
     │              │              │                  │
     └──────────────┴──────┬───────┴──────────────────┘
                           ▼
                  Dimensionality Reduction
                           │
                     PCA / SVD / ICA / NMF
                           │
                           ▼
                        Trainer
                           │
                           ▼
                       Evaluator
                           │
                           ▼
                    Model Comparison
                           │
                    قرار المتخصص
                           │
                           ▼
                       Final Fit
                           │
                           ▼
                      Prediction /
                  Cluster Assignment /
                   Anomaly Scores /
                  Reduced Features
```

الفكرة الأساسية هنا أن المكتبة تساعد المحلل على الفحص والتحليل والتجهيز والمقارنة، لكنها **لا تختار النموذج النهائي أو تغيّر البيانات بصمت**.

---

## Statistics Engine

المحرك الإحصائي أحد المحركات الأساسية للمكتبة.

الاستخدام من خلال الـFacade:

```python
ai.statistics("MonthlyCharges")
```

وداخليًا:

```text
AIDataFacade
      ↓
StatisticsEngine
      ├── mean
      ├── median
      ├── mode
      ├── variance
      ├── std
      ├── quartiles
      ├── IQR
      ├── skewness
      ├── kurtosis
      ├── z-score
      └── moments
```

### Bivariate Statistics

```text
Bivariate Statistics
      ├── covariance
      └── correlation
```

### Frequency Statistics

```text
FrequencyTable
      ├── counts
      ├── cumulative
      ├── relative
      └── percentage
```

---

## Facade — الطريق السريع

```python
from aidatasetkit import AIDataFacade

ai = AIDataFacade(
    target="Churn",
    id_column="CustomerID",
    task="classification"
)

ai.load(train_df, test_df)

ai.profile()
ai.statistics()
ai.check_quality()

ai.prepare()

results = ai.compare_models()
```

ثم:

```python
ai.select_model("random_forest")
ai.train()

report = ai.evaluate()

prediction_df = ai.predict_test()
```

الـFacade لا تنفذ الخوارزميات بنفسها. مهمتها تنسيق الخدمات الداخلية وتقديم API بسيطة.

---

## Direct Expert Access

```python
from aidatasetkit.statistics import StatisticsEngine

stats = StatisticsEngine(df["MonthlyCharges"])

stats.mean()
stats.std()
stats.skewness()
stats.kurtosis()
stats.iqr()
```

أو:

```python
from aidatasetkit.models import ModelFactory

model = ModelFactory.create(
    "random_forest_classifier"
)
```

الـFacade ليست سجنًا؛ هي طريق سريع فقط.

---

## Unsupervised Learning

المكتبة لا تقتصر على البيانات التي تحتوي على `target`. يجب أن تدعم أيضًا السيناريوهات التي لا توجد فيها Labels.

### Clustering

الاستخدام المستهدف:

```python
ai = AIDataFacade(task="clustering")

ai.load(df)

ai.profile()
ai.statistics()
ai.check_quality()
ai.prepare()

results = ai.compare_models([
    "kmeans",
    "dbscan",
    "agglomerative"
])
```

الخوارزميات المخطط لها:

```text
Clustering
    ├── KMeans
    ├── MiniBatchKMeans
    ├── DBSCAN
    ├── OPTICS
    ├── AgglomerativeClustering
    └── Birch
```

تقييم Clustering مختلف عن Classification، لذلك لا نستخدم Accuracy أو F1.

المقاييس المناسبة تشمل:

```text
Silhouette Score
Davies-Bouldin Index
Calinski-Harabasz Score
Number of Clusters
Noise Ratio        # مهم مع DBSCAN و OPTICS
```

### Anomaly Detection

```text
Anomaly Detection
    ├── IsolationForest
    ├── LocalOutlierFactor
    ├── OneClassSVM
    └── EllipticEnvelope
```

الاستخدام المستقبلي:

```python
ai = AIDataFacade(task="anomaly_detection")
ai.load(df)
ai.prepare()
ai.select_model("isolation_forest")
ai.train()

scores = ai.anomaly_scores()
```

### Dimensionality Reduction

```text
Dimensionality Reduction
    ├── PCA
    ├── TruncatedSVD
    ├── FastICA
    └── NMF
```

هذه الطبقة تساعد في:

- تقليل عدد Features.
- Visualization قبل النمذجة.
- إزالة بعض الضوضاء.
- ضغط البيانات.
- تجهيز المدخلات لبعض خوارزميات ML.
- اكتشاف البنية الداخلية للبيانات.

الاستخدام المستهدف:

```python
ai = AIDataFacade(task="dimensionality_reduction")
ai.load(df)
ai.prepare()

reduced = ai.transform(
    method="pca",
    n_components=10
)
```

### Evaluation by Task

التقييم يجب أن يكون حسب نوع المهمة:

```text
Evaluator
    ├── Classification Metrics
    ├── Regression Metrics
    ├── Clustering Metrics
    ├── Anomaly Detection Metrics
    └── Dimensionality Reduction Diagnostics
```

هذا يمنع افتراض أن كل Model له `predict_proba()` أو أن كل Dataset لها `target`.

---

## Neural Networks — Future Extension

```text
deep_learning/
│
├── base.py
│
├── pytorch/
│   ├── mlp_classifier.py
│   ├── mlp_regressor.py
│   └── trainer.py
│
└── tensorflow/
    ├── mlp_classifier.py
    └── mlp_regressor.py
```

ومستقبلًا يمكن أن يصبح الاستخدام:

```python
ai.select_model("pytorch_mlp_classifier")
```

بينما تبقى `AIDataFacade` كما هي.

---

## Three Levels of Usage

```text
Level 1 — Data Analyst
        │
        └── AIDataFacade

Level 2 — Data Scientist
        │
        ├── StatisticsEngine
        ├── DataProfiler
        ├── ModelComparator
        ├── ModelEvaluator
        ├── Clustering
        ├── Anomaly Detection
        └── Dimensionality Reduction

Level 3 — AI / ML Engineer
        │
        ├── ModelStrategy
        ├── ModelCapabilities
        ├── ModelRegistry
        ├── PreprocessingPlan
        ├── Custom Models
        └── Future Deep Learning Backends
```

### Level 1 — Data Analyst

يريد Workflow واضحة وسريعة:

```python
ai.profile()
ai.statistics()
ai.check_quality()
ai.prepare()
ai.compare_models()
```

### Level 2 — Data Scientist

يحتاج تحكمًا أكبر في:
- الإحصاء
- Profiling
- مقارنة النماذج
- Evaluation
- تحليل النتائج

### Level 3 — AI / ML Engineer

يحتاج إلى:
- إضافة Models جديدة
- تعريف capabilities
- بناء preprocessing مخصص
- إنشاء Strategies
- إضافة backends مستقبلية مثل PyTorch

---

## Architectural Principles

- **SOLID**: كل مكوّن له مسؤولية واضحة ومحددة.
- **Facade Pattern**: تبسيط التعامل مع المكتبة عبر `AIDataFacade`.
- **Strategy Pattern**: تبديل الخوارزميات دون تغيير الـworkflow.
- **Factory / Registry**: إضافة Models جديدة دون if/elif ضخمة.
- **DRY**: عدم تكرار preprocessing والتحليل المشترك.
- **KISS**: لا نضيف abstraction إلا عندما تحل مشكلة حقيقية.

---

## Product Boundary

AIDatasetKit ليست AutoML.

المكتبة تساعد المتخصص في:

```text
Inspect
   ↓
Profile
   ↓
Statistics
   ↓
Quality Checks
   ↓
Prepare
   ↓
Select Task Family
   ↓
Compare Models / Methods
   ↓
Engineer Decision
   ↓
Train / Fit / Transform
   ↓
Evaluate
   ↓
Prediction / Clusters / Anomaly Scores / Reduced Features
```

لكنها لا تقوم بصمت بـ:
- اختيار أفضل model
- حذف features
- حذف outliers
- معالجة leakage
- تعديل البيانات
- اختيار hyperparameters

يمكنها تقديم:
- Warnings
- Rankings
- Diagnostics
- Recommendations
- Metadata

لكن القرار النهائي يبقى للمتخصص.

---

## الهدف النهائي

> مكتبة موحدة تساعد محللي البيانات وعلماء البيانات ومهندسي الذكاء الاصطناعي على الانتقال من Dataset خام إلى بيانات مفهومة ومفحوصة ومجهزة، ثم مقارنة وتدريب وتقييم النماذج، أو تنفيذ Clustering وAnomaly Detection وDimensionality Reduction، من خلال معمارية قابلة للتوسع، دون إخفاء القرارات المهمة عن المتخصص.

هذه البنية تجعل المكتبة بسيطة للمحلل، قوية لعالم البيانات، وقابلة للتوسع لمهندس الذكاء الاصطناعي.
