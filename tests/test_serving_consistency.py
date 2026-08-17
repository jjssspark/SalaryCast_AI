"""학습 경로와 서빙 경로가 같은 입력에 같은 피처를 만드는가.

설계 스펙 §8 "피처 일관성"을 고정한다.

v7까지는 학습 피처를 노트북이, 서빙 피처를 src/features.py가 따로 만들었다.
같은 선수인데 두 경로의 값이 달라져도 아무도 몰랐고, 그게 누수(P2)와
3년 평균 버그(P5)가 오래 살아남은 이유였다.

지금은 두 경로가 같은 build_*_row를 부른다. 서빙은 그 위에 백분위만 덧붙인다.
그 관계가 깨지면 여기서 걸린다.
"""

import numpy as np
import pandas as pd

from src.features import (
    HITTER_PCT_COLS,
    PITCHER_PCT_COLS,
    build_hitter_row,
    build_pitcher_row,
    build_reference_dist,
    build_serving_row,
)

STAR = {"star_score": 3, "mvp_count": 0, "golden_glove_count": 1,
        "allstar_count": 0, "national_team": 2, "postseason_experience": 1}

HITTER_SEASONS = pd.DataFrame(
    [
        (2023, 141, 524, 0.856, 6.0, 1.0),
        (2024, 139, 524, 0.857, 4.8, 1.0),
        (2025, 130, 480, 0.812, 3.9, 1.0),
    ],
    columns=["collect_year", "games", "ab", "ops", "war", "season_completion"],
)

PITCHER_SEASONS = pd.DataFrame([{
    "collect_year": 2025, "games": 60, "innings": 60.0, "era": 2.50,
    "whip": 1.05, "win": 3, "lose": 2, "save": 30, "hold": 0, "qs": 0,
    "so": 70, "bb": 20, "hit": 45, "wpa": 2.0, "war": 2.5,
    "k9": 10.5, "bb9": 3.0, "k_bb": 3.5, "ip_per_game": 1.0,
    "season_completion": 1.0,
}])


def _hitter_reference() -> dict:
    rows = [
        {"position": "OF", "war_3yr_avg": v, "war_3yr_sum": v * 3,
         "ops_3yr_avg": 0.70 + v / 50, "wrc_plus_3yr_avg": 90 + v,
         "hr_3yr_avg": 10 + v, "rbi_3yr_avg": 50 + v, "woba_3yr_avg": 0.32 + v / 100}
        for v in range(1, 12)
    ]
    return build_reference_dist(pd.DataFrame(rows), HITTER_PCT_COLS, "position")


def _pitcher_reference() -> dict:
    rows = [
        {"pitcher_role": "CL", "war_3yr_sum": v, "era_3yr_avg": 5.0 - v / 5,
         "innings_3yr_avg": 40 + v * 3}
        for v in range(1, 12)
    ]
    return build_reference_dist(pd.DataFrame(rows), PITCHER_PCT_COLS, "pitcher_role")


def _same(trained: dict, served: dict) -> None:
    for key, value in trained.items():
        assert key in served, f"서빙 경로에서 {key}가 사라졌다"
        both_nan = (
            isinstance(value, float) and isinstance(served[key], float)
            and np.isnan(value) and np.isnan(served[key])
        )
        assert both_nan or served[key] == value, \
            f"{key}: 학습 {value} vs 서빙 {served[key]}"


def test_hitter_serving_row_keeps_every_training_value_untouched():
    trained = build_hitter_row(HITTER_SEASONS, 2026, 30, "OF", STAR, 12.0)
    served = build_serving_row(
        HITTER_SEASONS, 2026, 30, "OF", STAR, 12.0, _hitter_reference(), is_hitter=True)

    assert trained is not None and served is not None
    _same(trained, served)


def test_pitcher_serving_row_keeps_every_training_value_untouched():
    trained = build_pitcher_row(PITCHER_SEASONS, 2026, 31, "CL", STAR, 8.0)
    served = build_serving_row(
        PITCHER_SEASONS, 2026, 31, "CL", STAR, 8.0, _pitcher_reference(), is_hitter=False)

    assert trained is not None and served is not None
    _same(trained, served)


def test_serving_row_only_adds_percentiles_and_season_label():
    """서빙이 덧붙이는 것은 백분위와 사용 시즌 표기뿐이어야 한다.

    다른 이름이 끼어들면 학습에 없던 피처가 예측에 들어간 것이다.
    """
    trained = build_hitter_row(HITTER_SEASONS, 2026, 30, "OF", STAR, 12.0)
    served = build_serving_row(
        HITTER_SEASONS, 2026, 30, "OF", STAR, 12.0, _hitter_reference(), is_hitter=True)

    added = set(served) - set(trained)
    unexpected = {
        key for key in added
        if key != "seasons_used" and not key.endswith(("_all_pct", "_pos_pct"))
    }
    assert not unexpected, f"서빙에만 있는 예상 밖 피처: {unexpected}"


def test_percentiles_are_measured_against_the_saved_reference_not_the_input():
    """백분위는 저장된 학습 분포에 대고 매긴다.

    입력 한 건으로 다시 순위를 매기면 누구를 넣어도 100%가 나온다.
    """
    weak = build_serving_row(
        HITTER_SEASONS.assign(war=[0.1, 0.1, 0.1]),
        2026, 30, "OF", STAR, 12.0, _hitter_reference(), is_hitter=True)

    assert weak is not None
    assert weak["war_3yr_sum_all_pct"] < 50, "약한 성적인데 상위로 잡혔다"


def test_both_paths_return_none_for_empty_seasons():
    empty = HITTER_SEASONS.iloc[0:0]
    assert build_hitter_row(empty, 2026, 30, "OF", STAR, 12.0) is None
    assert build_serving_row(
        empty, 2026, 30, "OF", STAR, 12.0, _hitter_reference(), is_hitter=True) is None
