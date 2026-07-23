import numpy as np
import pandas as pd
import pytest

from src.predict import predict_h, predict_p


class _FakeModel:
    def __init__(self, log_pred):
        self._log_pred = log_pred

    def predict(self, X):
        return np.array([self._log_pred])


def test_predict_h_combines_weighted_ensemble_and_reverses_log1p():
    row = pd.Series({"feat_a": 1.0, "feat_b": 2.0})
    meta = {"features": ["feat_a", "feat_b"], "xgb_weight": 0.6, "lgb_weight": 0.4}
    xgb_m = _FakeModel(log_pred=np.log1p(10.0))
    lgb_m = _FakeModel(log_pred=np.log1p(10.0))

    result = predict_h(row, xgb_m, lgb_m, meta)

    assert result == pytest.approx(10.0, rel=1e-6)


def test_predict_p_combines_weighted_ensemble_and_reverses_log1p():
    row = pd.Series({"feat_a": 1.0, "feat_b": 2.0})
    meta = {"features": ["feat_a", "feat_b"], "xgb_weight": 0.5, "rf_weight": 0.5}
    xgb_m = _FakeModel(log_pred=np.log1p(8.0))
    rf_m = _FakeModel(log_pred=np.log1p(6.0))

    result = predict_p(row, xgb_m, rf_m, meta)
    expected = np.expm1(0.5 * np.log1p(8.0) + 0.5 * np.log1p(6.0))

    assert result == pytest.approx(expected, rel=1e-6)
