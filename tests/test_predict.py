"""src/predict.py — 번들 예측과 표본 검사.

v7의 predict_h/predict_p는 모델 개수와 결합 방식을 코드에 박아뒀다.
지금은 meta가 구성을 들고 있어 predict_salary 하나로 처리한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.predict import NotEnoughRecord, check_sample, predict_salary, to_frame


class _Fixed:
    """항상 같은 로그 예측을 내는 가짜 모델."""

    def __init__(self, log_pred):
        self._log = log_pred

    def predict(self, X):
        return np.array([self._log])


class _Echo:
    """입력 피처 순서를 확인하려고 첫 컬럼 값을 그대로 돌려준다."""

    def predict(self, X):
        return np.array([X.iloc[0, 0]])


def _bundle(method, models, weight=1.0, first=None, second=None):
    return {
        "meta": {
            "features": ["feat_a", "feat_b"],
            "method": method,
            "blend_first": first,
            "blend_second": second,
            "weight": weight,
        },
        "models": models,
        "reference": {},
    }


def test_predict_salary_reverses_log1p_for_single_model():
    bundle = _bundle("XGBoost", {"XGBoost": _Fixed(np.log1p(10.0))})

    assert predict_salary({"feat_a": 1.0, "feat_b": 2.0}, bundle) == pytest.approx(10.0, rel=1e-6)


def test_predict_salary_blends_in_log_space_with_meta_weight():
    bundle = _bundle(
        "blend:LightGBM+Ridge",
        {"LightGBM": _Fixed(np.log1p(8.0)), "Ridge": _Fixed(np.log1p(18.0))},
        weight=0.75, first="LightGBM", second="Ridge",
    )

    expected = np.expm1(0.75 * np.log1p(8.0) + 0.25 * np.log1p(18.0))

    assert predict_salary({"feat_a": 1.0, "feat_b": 2.0}, bundle) == pytest.approx(expected, rel=1e-6)


def test_predict_salary_never_returns_negative():
    bundle = _bundle("XGBoost", {"XGBoost": _Fixed(-5.0)})  # expm1(-5) < 0

    assert predict_salary({"feat_a": 0.0, "feat_b": 0.0}, bundle) == 0.0


def test_to_frame_follows_training_feature_order_and_fills_missing():
    frame = to_frame({"feat_b": 2.0}, ["feat_a", "feat_b"])

    assert list(frame.columns) == ["feat_a", "feat_b"]
    assert np.isnan(frame.iloc[0]["feat_a"])
    assert frame.iloc[0]["feat_b"] == 2.0


def test_predict_salary_reorders_row_to_match_training():
    # dict 순서가 학습 순서와 달라도 피처가 뒤바뀌면 안 된다.
    bundle = _bundle("Echo", {"Echo": _Echo()})

    result = predict_salary({"feat_b": np.log1p(99.0), "feat_a": np.log1p(7.0)}, bundle)

    assert result == pytest.approx(7.0, rel=1e-6)


def test_check_sample_blocks_hitter_with_too_few_at_bats():
    with pytest.raises(NotEnoughRecord):
        check_sample(pd.DataFrame({"ab": [10, 15]}), is_hitter=True)


def test_check_sample_blocks_pitcher_with_too_few_innings():
    with pytest.raises(NotEnoughRecord):
        check_sample(pd.DataFrame({"innings": [5.0, 3.0]}), is_hitter=False)


def test_check_sample_blocks_when_no_season_remains():
    with pytest.raises(NotEnoughRecord):
        check_sample(pd.DataFrame({"ab": []}), is_hitter=True)


def test_check_sample_passes_for_a_regular():
    check_sample(pd.DataFrame({"ab": [400, 450, 380]}), is_hitter=True)
    check_sample(pd.DataFrame({"innings": [60.0, 55.0, 70.0]}), is_hitter=False)
