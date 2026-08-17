"""StoveLens AI — 리그 기준값.

호출 위치: src/ui/player.py (스탯을 리그와 비교해 보여줄 때)
데이터 파일을 직접 읽지 않는다. data_loader가 읽은 시즌 스탯 DataFrame을 받는다.

화면에 '상위 15%'라고 쓰려면 무엇 대비 15%인지가 정해져 있어야 한다.
전체 선수를 모수로 잡으면 1군에 잠깐 올라온 선수까지 섞여서
주전이면 누구나 상위 10%로 나온다. 규정타석/이닝에 준하는 선을 넘긴
최근 두 시즌 선수만 모수로 쓰고, 그 기준을 화면에 같이 적는다.

src/features.py에도 백분위 계산이 있지만 모수가 다르다.
그쪽은 FA 계약자 93명(모델 학습 표본), 여기는 리그 주전 전체다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 모수 기준. 규정타석(144경기 x 3.1)에는 못 미치지만
# '한 시즌을 주전으로 뛴 선수'를 거르는 선이다.
QUALIFY_AB = 300
QUALIFY_INNINGS = 30.0

# 최근 몇 시즌을 모수로 삼을지. 너무 길면 리그 환경 변화가 섞인다.
REFERENCE_SEASONS = 2

# 값이 낮을수록 좋은 스탯. 백분위를 뒤집어야 한다.
LOWER_IS_BETTER = {"era", "whip", "bb9"}


def _completed_years(stats: pd.DataFrame, count: int) -> list[int]:
    """진행 중인 시즌은 모수에서 뺀다. 누적 스탯이 아직 덜 쌓여 있다."""
    completion = stats.groupby("collect_year")["season_completion"].max()
    completed = completion[completion >= 0.999].index
    return sorted(int(year) for year in completed)[-count:]


def build_reference(stats: pd.DataFrame, is_hitter: bool) -> dict:
    """리그 비교용 표본. {'years','n','qualifier','sample','mean'}"""
    years = _completed_years(stats, REFERENCE_SEASONS)
    pool = stats[stats["collect_year"].isin(years)]

    if is_hitter:
        pool = pool[pd.to_numeric(pool["ab"], errors="coerce").fillna(0) >= QUALIFY_AB]
        columns = ["war", "ops", "hr", "rbi", "obp", "slg", "wrc_plus", "sb"]
    else:
        pool = pool[
            pd.to_numeric(pool["innings"], errors="coerce").fillna(0) >= QUALIFY_INNINGS
        ]
        columns = ["war", "era", "whip", "innings", "so", "save", "hold", "k9"]

    sample, mean = {}, {}
    for column in columns:
        if column not in pool.columns:
            continue
        values = pd.to_numeric(pool[column], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        sample[column] = values
        mean[column] = float(values.mean())

    return {
        "years": years,
        "n": int(len(pool)),
        "qualifier": f"{QUALIFY_AB}타수" if is_hitter else f"{QUALIFY_INNINGS:.0f}이닝",
        "sample": sample,
        "mean": mean,
    }


def percentile(value, sample: np.ndarray, lower_is_better: bool) -> float:
    """표본 안에서 value의 위치. 0~1, 클수록 좋다."""
    if value is None or pd.isna(value) or sample is None or len(sample) == 0:
        return 0.5
    share = float((sample <= float(value)).mean())
    return 1.0 - share if lower_is_better else share


def compare(reference: dict, stat: str, value) -> dict | None:
    """스탯 하나를 리그와 견준 결과.

    bar    — 막대 길이(0~1). 백분위를 그대로 쓴다.
    avg_at — 리그 평균이 그 막대 위에서 놓이는 자리(0~1).
    """
    sample = reference["sample"].get(stat)
    if sample is None:
        return None

    lower_is_better = stat in LOWER_IS_BETTER
    pct = percentile(value, sample, lower_is_better)
    league_mean = reference["mean"][stat]

    return {
        "value": value,
        "pct": pct,
        "bar": pct,
        "avg": league_mean,
        "avg_at": percentile(league_mean, sample, lower_is_better),
        "rank_text": rank_text(pct),
        "lower_is_better": lower_is_better,
    }


def format_value(value, fmt: str | None) -> str:
    """야구 표기로 숫자를 찍는다. 타율·OPS는 앞의 0을 떼서 .935로 쓴다."""
    if value is None or pd.isna(value):
        return "-"

    text = (fmt or "{:.1f}").format(float(value))
    if fmt == "{:.3f}" and text.startswith("0."):
        return text[1:]
    return text


def rank_text(pct: float) -> str:
    """0.85 -> '상위 15%'."""
    top = max(1, min(99, round((1.0 - pct) * 100)))
    return f"상위 {top}%"


def basis_text(reference: dict, is_hitter: bool) -> str:
    years = reference["years"]
    if len(years) > 1:
        span = f"{years[0]}~{years[-1]}"
    elif years:
        span = str(years[0])
    else:
        span = "-"

    unit = "타수" if is_hitter else "이닝"
    threshold = QUALIFY_AB if is_hitter else int(QUALIFY_INNINGS)
    return (
        f"<b>비교 기준</b> — {span}시즌 {threshold}{unit} 이상 "
        f"<b>{reference['n']}명</b>의 시즌 기록.<br/>"
        f"막대 = 이 {reference['n']}명 중 이 선수의 위치 · "
        f"흰 눈금 = 리그 평균이 놓이는 자리."
    )
