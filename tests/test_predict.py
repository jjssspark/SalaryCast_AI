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


class _FakeStackModel:
    """models/pitcher_meta_model.pkl은 sklearn Ridge(). [xgb_pred, lgb_pred, rf_pred] 3개를
    입력받아 최종 log1p 예측을 반환하는 스태킹 메타모델 — predict_p는 이 인터페이스를 따라야 함."""

    def predict(self, X):
        xgb_p, lgb_p, rf_p = X[0]
        return np.array([0.5 * xgb_p + 0.3 * lgb_p + 0.2 * rf_p])


def test_predict_p_stacks_three_base_models_and_reverses_log1p():
    # 실제 pitcher_model_meta.pkl은 {'method': 'Stacking', 'features': [...]}만 담고
    # xgb_weight/rf_weight가 없다 — predict_p는 xgb/lgb/rf 예측을 Ridge 스태커에 넣어야 한다.
    row = pd.Series({"feat_a": 1.0, "feat_b": 2.0})
    meta = {"features": ["feat_a", "feat_b"]}
    xgb_m = _FakeModel(log_pred=np.log1p(8.0))
    lgb_m = _FakeModel(log_pred=np.log1p(9.0))
    rf_m = _FakeModel(log_pred=np.log1p(6.0))
    stack_m = _FakeStackModel()

    result = predict_p(row, xgb_m, lgb_m, rf_m, stack_m, meta)
    expected = np.expm1(0.5 * np.log1p(8.0) + 0.3 * np.log1p(9.0) + 0.2 * np.log1p(6.0))

    assert result == pytest.approx(expected, rel=1e-6)
