"""Prediction: applying a fitted model to rows it has never seen.

The narrowest layer in the library, and deliberately so. It transforms with an
already-fitted preprocessor, calls an already-fitted estimator, and turns the
answer back into the caller's own vocabulary. It fits nothing, learns nothing,
and cannot change what any model knows.

That is why inference lives here rather than as a method on something else: a
frame arriving for prediction must be incapable of influencing the model, and
the clearest way to guarantee that is to give the operation no code that could.
"""

from aidatasetkit.prediction.predictor import predict_frame
from aidatasetkit.prediction.types import PredictionResult

__all__ = ["PredictionResult", "predict_frame"]
