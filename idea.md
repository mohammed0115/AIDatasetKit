                 AIDataFacade
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   DataProfiler   Statistics    DataQuality
        │
        ▼
   Preprocessor
        │
        ▼
   ModelFactory
        │
 ┌──────┼────────┬──────────┐
 ▼      ▼        ▼          ▼
Dummy Logistic Tree     GradientBoost
                         ExtraTrees
        │
        ▼
     Trainer
        │
        ▼
    Evaluator
        │
        ▼
 AIAnalysisReport