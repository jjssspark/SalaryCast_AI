"""StoveLens AI — 연봉 예측.

호출 위치: src/ui/pages.py
데이터 파일 없음. src.data_loader.load_models()가 준 번들을 인자로 받는다.

타깃이 log1p(연평균 계약금)이라 예측값을 expm1로 되돌린다.
모델 구성(단일/블렌드)은 meta에 적혀 있어 이 파일이 알 필요가 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 표본이 이만큼도 안 되면 예측하지 않는다.
# 대타로 몇 경기 나온 선수에게 연봉을 붙이면 숫자만 그럴듯하고 근거가 없다.
MIN_SEASONS = 1
MIN_HITTER_AB = 50
MIN_PITCHER_INNINGS = 20.0


class NotEnoughRecord(Exception):
    """표본이 부족해 예측을 내지 않는 경우."""


def to_frame(row, features: list[str]) -> pd.DataFrame:
    """모델이 학습 때 본 순서 그대로 한 행짜리 입력을 만든다."""
    source = row if isinstance(row, dict) else row.to_dict()
    return pd.DataFrame([{name: source.get(name, np.nan) for name in features}])[features]


def predict_salary(row, bundle: dict) -> float:
    """연평균 계약금 예측값(억 원)."""
    meta = bundle["meta"]
    models = bundle["models"]
    X = to_frame(row, meta["features"]).astype(float)

    if meta["method"].startswith("blend:"):
        weight = float(meta["weight"])
        first = models[meta["blend_first"]].predict(X)[0]
        second = models[meta["blend_second"]].predict(X)[0]
        log_pred = weight * first + (1 - weight) * second
    else:
        log_pred = models[meta["method"]].predict(X)[0]

    return float(max(0.0, np.expm1(log_pred)))


def check_sample(seasons: pd.DataFrame, is_hitter: bool) -> None:
    """예측을 낼 만한 표본인지. 부족하면 NotEnoughRecord를 던진다."""
    if len(seasons) < MIN_SEASONS:
        raise NotEnoughRecord("최근 기록이 없습니다.")

    if is_hitter:
        at_bats = pd.to_numeric(seasons.get("ab"), errors="coerce").fillna(0).sum()
        if at_bats < MIN_HITTER_AB:
            raise NotEnoughRecord(
                f"최근 3시즌 타수가 {int(at_bats)}타석뿐이라 예측을 내기 어렵습니다."
            )
        return

    innings = pd.to_numeric(seasons.get("innings"), errors="coerce").fillna(0).sum()
    if innings < MIN_PITCHER_INNINGS:
        raise NotEnoughRecord(
            f"최근 3시즌 투구 이닝이 {innings:.0f}이닝뿐이라 예측을 내기 어렵습니다."
        )
