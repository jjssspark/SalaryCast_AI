import pandas as pd

from src.features import engineer_hitter_features, engineer_pitcher_features


def _hitter_df():
    return pd.DataFrame({
        "fa_year": [2023, 2023, 2024],
        "annual_avg_salary": [10.0, 5.0, 8.0],
        "age_at_fa": [30, 34, 28],
        "star_score": [3, 0, 5],
        "position": ["OF", "C", "OF"],
        "war_3yr_avg": [2.0, 1.0, 3.0],
        "war_3yr_sum": [6.0, 3.0, 9.0],
        "ops_3yr_avg": [0.850, 0.700, 0.900],
        "hr_3yr_avg": [20.0, 5.0, 25.0],
        "rbi_3yr_avg": [70.0, 40.0, 90.0],
        "woba_3yr_avg": [0.360, 0.310, 0.380],
        "wrc_plus_3yr_avg": [120.0, 90.0, 140.0],
    })


def _pitcher_df():
    return pd.DataFrame({
        "fa_year": [2023, 2024],
        "annual_avg_salary": [7.0, 12.0],
        "age_at_fa": [31, 29],
        "star_score": [2, 4],
        "war_3yr_sum": [5.0, 8.0],
        "games_3yr_avg": [50.0, 30.0],
        "pitcher_role": ["SU", "SP"],
        "innings_3yr_avg": [60.0, 150.0],
        "era_3yr_avg": [3.50, 3.00],
        "whip_3yr_avg": [1.30, 1.10],
        "win_3yr_avg": [3.0, 12.0],
        "lose_3yr_avg": [2.0, 8.0],
        "save_3yr_avg": [0.0, 0.0],
        "hold_3yr_avg": [20.0, 0.0],
        "hit_allowed_3yr_avg": [55.0, 140.0],
    })


def test_engineer_hitter_features_adds_expected_columns():
    result = engineer_hitter_features(_hitter_df())

    assert "market_level" in result.columns
    assert result.loc[0, "age_squared"] == 30 ** 2
    assert result.loc[1, "prime_years_left"] == 1  # 35 - 34, clipped to [0, 10]
    assert result.loc[0, "star_x_war"] == 3 * 6.0
    assert result.loc[0, "war_sum_sq"] == 6.0 ** 2
    assert result.loc[1, "position_enc"] == 0  # "C" -> 0


def test_engineer_hitter_features_computes_wrc_plus_percentiles():
    # Regression test: the hitter model expects wrc_plus_3yr_avg_all_pct/_pos_pct
    # columns (models/hitter_model_meta.pkl "features"), but the percentile loop
    # previously omitted "wrc_plus_3yr_avg", causing a KeyError at prediction time.
    result = engineer_hitter_features(_hitter_df())

    assert "wrc_plus_3yr_avg_all_pct" in result.columns
    assert "wrc_plus_3yr_avg_pos_pct" in result.columns


def test_engineer_hitter_features_prime_years_left_is_clipped_to_zero():
    df = _hitter_df()
    df.loc[0, "age_at_fa"] = 40  # 35 - 40 = -5, should clip to 0
    result = engineer_hitter_features(df)
    assert result.loc[0, "prime_years_left"] == 0


def test_engineer_hitter_features_does_not_mutate_input():
    original = _hitter_df()
    engineer_hitter_features(original)
    assert "market_level" not in original.columns


def test_engineer_pitcher_features_matches_trained_model_feature_list():
    # 회귀 테스트: models/pitcher_model_meta.pkl의 "features"에 있는 모든 컬럼이
    # 생성돼야 한다. 예전 구현은 role_SP/role_SU/war_per_game 등 학습 때 쓰이지
    # 않은 이름을 만들어 배포 후 KeyError가 났었다.
    result = engineer_pitcher_features(_pitcher_df())

    required = [
        "hit_per_inn", "whip_era_ratio", "win_rate", "star_x_war",
        "role_x_save", "role_x_hold", "role_x_inn",
        "war_3yr_sum_pos_pct", "war_3yr_sum_all_pct",
        "era_3yr_avg_pos_pct", "era_3yr_avg_all_pct",
        "innings_3yr_avg_pos_pct", "innings_3yr_avg_all_pct",
        "prime_years_left", "age_squared", "role_enc", "market_level",
    ]
    for col in required:
        assert col in result.columns, f"missing column: {col}"

    assert result.loc[0, "age_squared"] == 31 ** 2
    assert result.loc[0, "role_x_hold"] == 20.0  # SU 역할이라 hold_3yr_avg 그대로
    assert result.loc[1, "role_x_inn"] == 150.0  # SP 역할이라 innings_3yr_avg 그대로
    assert result.loc[0, "role_enc"] == 2  # "SU" -> 2 (CL=0, SP=1, SU=2)
    assert result.loc[1, "role_enc"] == 1  # "SP" -> 1
