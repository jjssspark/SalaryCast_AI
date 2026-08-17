"""src/features.py — 시즌 선택과 집계 규칙.

v7까지는 engineer_* 함수가 파생 피처를 전부 계산했지만, 지금은 학습과 서빙이
같은 build_*_row를 쓰고 engineer_*는 상대 순위만 붙인다. 검증 대상도
'어느 시즌을 고르는가'와 '그 시즌을 어떻게 접는가'로 옮겼다.
"""

import numpy as np
import pandas as pd

from src.features import (
    age_terms,
    aggregate_seasons,
    build_hitter_row,
    build_pitcher_row,
    market_level_prior,
    resolve_player_id,
    select_recent_seasons,
)

COUNTING = {"games": "games", "ab": "ab", "hr": "hr", "war": "war"}
RATE = {"ops": "ops"}


def _seasons(rows):
    """rows: (연도, 경기, 타수, OPS, WAR, 시즌진행률)"""
    return pd.DataFrame(
        rows, columns=["collect_year", "games", "ab", "ops", "war", "season_completion"]
    )


def test_select_recent_seasons_fills_a_missing_year_from_further_back():
    # 연속된 세 해가 아니라 '존재하는' 세 시즌이다. 2025년 기록이 통째로 없으면
    # 2024·2023·2022로 채운다.
    picked = select_recent_seasons(_seasons([
        (2022, 140, 520, 0.840, 4.5, 1.0),
        (2023, 141, 524, 0.856, 6.0, 1.0),
        (2024, 139, 524, 0.857, 4.8, 1.0),
    ]), base_year=2025)

    assert sorted(picked["collect_year"]) == [2022, 2023, 2024]


def test_select_recent_seasons_keeps_injury_shortened_year():
    # 부상으로 짧게 뛴 해도 뺀 게 아니라 남긴다. 결장 자체가 계약금에 반영되기
    # 때문에 제외하면 예측이 나빠졌다. 짧은 시즌의 영향은 출전량 가중으로 줄인다.
    picked = select_recent_seasons(_seasons([
        (2023, 141, 524, 0.856, 6.0, 1.0),
        (2024, 139, 524, 0.857, 4.8, 1.0),
        (2025, 51, 174, 0.727, 1.2, 1.0),
        (2026, 97, 338, 0.674, 1.2, 0.75),
    ]), base_year=2026)

    assert sorted(picked["collect_year"]) == [2024, 2025, 2026]


def test_select_recent_seasons_respects_lookback_limit():
    # 은퇴 직전 선수가 10년 전 전성기 기록으로 평가되면 안 된다.
    picked = select_recent_seasons(_seasons([
        (2014, 144, 550, 0.950, 7.0, 1.0),
        (2015, 144, 550, 0.950, 7.0, 1.0),
        (2025, 100, 350, 0.700, 1.0, 1.0),
    ]), base_year=2025)

    assert sorted(picked["collect_year"]) == [2025]


def test_aggregate_seasons_divides_by_actual_season_count():
    # 시즌이 2개뿐인데 3으로 나누면 성적이 3분의 2로 축소된다.
    out = aggregate_seasons(_seasons([
        (2024, 100, 400, 0.800, 3.0, 1.0),
        (2025, 100, 400, 0.800, 3.0, 1.0),
    ]), COUNTING, RATE)

    assert out["games_3yr_avg"] == 100.0
    assert out["active_seasons"] == 2


def test_aggregate_seasons_ignores_missing_war_seasons():
    # 2013~2016 타자 WAR은 원본이 결측이다. 0으로 세면 기여가 없었던 것이 된다.
    out = aggregate_seasons(_seasons([
        (2015, 140, 500, 0.800, np.nan, 1.0),
        (2016, 140, 500, 0.800, 4.0, 1.0),
    ]), COUNTING, RATE)

    assert out["war_3yr_avg"] == 4.0
    assert out["war_seasons_valid"] == 1


def test_aggregate_seasons_weights_rate_stats_by_playing_time():
    # 174타수짜리 시즌의 OPS가 524타수 시즌과 같은 무게를 가지면 안 된다.
    out = aggregate_seasons(_seasons([
        (2024, 140, 600, 0.900, 5.0, 1.0),
        (2025, 50, 200, 0.600, 1.0, 1.0),
    ]), COUNTING, RATE)

    simple_mean = (0.900 + 0.600) / 2
    weighted = (0.900 * 600 + 0.600 * 200) / 800

    assert out["ops_3yr_avg"] == weighted
    assert out["ops_3yr_avg"] > simple_mean


def test_aggregate_seasons_annualizes_counting_but_not_rate_stats():
    out = aggregate_seasons(_seasons([(2026, 75, 300, 0.800, 2.0, 0.75)]), COUNTING, RATE)

    assert out["games_3yr_avg"] == 100.0   # 75 / 0.75
    assert out["ops_3yr_avg"] == 0.800     # 비율은 환산하지 않는다


def test_market_level_prior_uses_only_earlier_years():
    # v7은 같은 해 계약의 중앙값을 넣어 정답을 그대로 흘렸다.
    fa = pd.DataFrame({
        "fa_year": [2022, 2023, 2023, 2024],
        "annual_avg_salary": [10.0, 20.0, 30.0, 999.0],
    })

    assert market_level_prior(fa, 2024) == 20.0
    assert pd.isna(market_level_prior(fa, 2022))


def test_age_terms_are_missing_when_birth_year_is_unknown():
    # 추측한 나이를 넣으면 prime_years_left가 잘못된 방향으로 강하게 작동한다.
    assert all(pd.isna(value) for value in age_terms(None).values())


def test_age_terms_clip_prime_years_left():
    assert age_terms(40)["prime_years_left"] == 0
    assert age_terms(34)["prime_years_left"] == 1
    assert age_terms(20)["prime_years_left"] == 10


def test_resolve_player_id_prefers_the_regular_over_the_namesake():
    # '김현수'는 매년 140경기 뛴 선수와 1경기 뛴 선수가 함께 잡힌다.
    seasons = pd.DataFrame({
        "player_name": ["김현수"] * 4,
        "player_id": [76290, 76290, 69516, 69516],
        "collect_year": [2023, 2024, 2023, 2024],
        "team": ["LG", "LG", "KIA", "KIA"],
        "games": [140, 140, 1, 1],
    })

    assert resolve_player_id(seasons, "김현수", "LG", 2025) == 76290


def test_build_hitter_row_encodes_position_and_star_interaction():
    star = {"star_score": 3, "mvp_count": 0, "golden_glove_count": 1,
            "allstar_count": 0, "national_team": 0, "postseason_experience": 0}

    row = build_hitter_row(_seasons([(2025, 140, 500, 0.900, 5.0, 1.0)]),
                           2026, 30, "C", star, 12.0)

    assert row["position_enc"] == 0                    # C
    assert row["star_x_war"] == 3 * row["war_3yr_sum"]
    assert row["age_squared"] == 900
    assert row["market_level"] == 12.0


def test_build_pitcher_row_gates_role_specific_features():
    seasons = pd.DataFrame([{
        "collect_year": 2025, "games": 60, "innings": 60.0, "era": 2.50,
        "whip": 1.05, "win": 3, "lose": 2, "save": 30, "hold": 0, "qs": 0,
        "so": 70, "bb": 20, "hit": 45, "wpa": 2.0, "war": 2.5,
        "k9": 10.5, "bb9": 3.0, "k_bb": 3.5, "ip_per_game": 1.0,
        "season_completion": 1.0,
    }])

    row = build_pitcher_row(seasons, 2026, 31, "CL", {"star_score": 0}, 8.0)

    assert row["role_x_save"] == row["save_3yr_avg"]
    assert row["role_x_hold"] == 0.0
    assert row["role_x_inn"] == 0.0
    assert row["role_enc"] == 0                        # CL


def test_build_row_returns_none_for_empty_seasons():
    assert build_hitter_row(_seasons([]), 2026, 30, "OF", {"star_score": 0}, 10.0) is None
