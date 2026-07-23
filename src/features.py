"""StoveLens AI — 피처 엔지니어링 (학습 데이터 가공 + 미래 FA 선수 피처 생성).

호출 위치: src/data_loader.py load_data()가 engineer_hitter_features/engineer_pitcher_features를,
app/app.py의 src.ui.pages.render_search가 build_future_hitter_row/build_future_pitcher_row를 호출.
데이터 파일: 직접 read/write 없음 — 호출부에서 넘긴 DataFrame(hitter_training_v7.csv,
pitcher_training_v7.csv, hitter/pitcher_season_stats_2015_2026_v2.csv, star_features_*.csv
기반 컬럼)을 가공만 함.
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: features.py".
"""
import pandas as pd

from src.constants import POS_CODE


def engineer_hitter_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_level"] = df.groupby("fa_year")["annual_avg_salary"].transform("median")
    df["age_squared"] = df["age_at_fa"] ** 2
    df["prime_years_left"] = (35 - df["age_at_fa"]).clip(0, 10)
    df["star_x_war"] = df["star_score"] * df["war_3yr_sum"]
    df["star_x_ops"] = df["star_score"] * df["ops_3yr_avg"]
    df["war_sum_sq"] = df["war_3yr_sum"] ** 2
    pos_map = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5, "IF": 6, "DH": 7}
    df["position_enc"] = df["position"].map(pos_map).fillna(6).astype(int)
    for col in ["war_3yr_avg", "war_3yr_sum", "ops_3yr_avg", "wrc_plus_3yr_avg", "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg"]:
        df[f"{col}_all_pct"] = df[col].rank(pct=True)
        df[f"{col}_pos_pct"] = df.groupby("position")[col].rank(pct=True)
    return df


def engineer_pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_level"] = df.groupby("fa_year")["annual_avg_salary"].transform("median")
    df["age_squared"] = df["age_at_fa"] ** 2
    df["prime_left"] = (35 - df["age_at_fa"]).clip(0, 10)
    df["war_per_game"] = df["war_3yr_sum"] / (df["games_3yr_avg"] * 3 + 1)
    df["star_x_war"] = df["star_score"] * df["war_3yr_sum"]
    df["role_SP"] = (df["pitcher_role"] == "SP").astype(int)
    df["role_SU"] = (df["pitcher_role"] == "SU").astype(int)
    df["role_CL"] = (df["pitcher_role"] == "CL").astype(int)
    return df


def build_future_hitter_row(player_name, fr, h_stats_df, h_star_df, h_df):
    rows = h_stats_df[h_stats_df["playerName"] == player_name].sort_values("collect_year", ascending=False)
    if rows.empty:
        return None
    year_col_s = "collect_year" if "collect_year" in rows.columns else "year"
    base_yr = int(rows[year_col_s].max())
    target_yrs = [base_yr - i for i in range(3)]
    recent3 = rows[rows[year_col_s].isin(target_yrs)]
    if recent3.empty:
        recent3 = rows.head(3)
    n_yrs = 3
    last   = rows.iloc[0]
    avg3   = recent3.sum(numeric_only=True).fillna(0) / n_yrs
    active_seasons_h = len(recent3)

    existing = h_df[h_df["player_name"] == player_name]
    if not existing.empty:
        old = existing.sort_values("fa_year").iloc[-1]
        age_at_fa = int(old["age_at_fa"]) + (int(fr["fa_year_expected"]) - int(old["fa_year"]))
    else:
        age_at_fa = 32

    star_row = h_star_df[h_star_df["player_name"] == player_name]
    star = star_row.iloc[-1] if not star_row.empty else pd.Series({
        "mvp_count": 0, "golden_glove_count": 0, "allstar_count": 0,
        "national_team": 0, "postseason_experience": 0, "star_score": 0,
    })

    pos_code   = POS_CODE.get(fr.get("position", "외야수"), "OF")
    enc_map    = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5, "IF": 6, "DH": 7}
    war_sum    = float(recent3["hitterWar"].fillna(0).sum())
    war_avg    = war_sum / n_yrs
    ops_avg    = float(avg3.get("hitterOps", 0))
    hr_avg     = float(avg3.get("hitterHr", 0))
    rbi_avg    = float(avg3.get("hitterRbi", 0))
    woba_avg   = float(avg3.get("hitterWoba", 0))
    wrc_plus_avg = float(avg3.get("hitterWrcPlus", 0))
    star_score = int(star.get("star_score", 0))

    recent_market = h_df[h_df["fa_year"] == h_df["fa_year"].max()]["annual_avg_salary"].median()
    if pd.isna(recent_market):
        recent_market = float(h_df["annual_avg_salary"].median())

    ref = h_df[["position", "war_3yr_avg", "war_3yr_sum", "ops_3yr_avg", "wrc_plus_3yr_avg",
                 "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg"]].copy()
    ref["position"] = ref["position"].fillna("OF")
    new_val = {"position": pos_code, "war_3yr_avg": war_avg, "war_3yr_sum": war_sum,
               "ops_3yr_avg": ops_avg, "wrc_plus_3yr_avg": wrc_plus_avg,
               "hr_3yr_avg": hr_avg, "rbi_3yr_avg": rbi_avg, "woba_3yr_avg": woba_avg}
    ref = pd.concat([ref, pd.DataFrame([new_val])], ignore_index=True)
    pct = {}
    for col in ["war_3yr_avg", "war_3yr_sum", "ops_3yr_avg", "wrc_plus_3yr_avg", "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg"]:
        pct[f"{col}_all_pct"] = float(ref[col].rank(pct=True).iloc[-1])
        pct[f"{col}_pos_pct"] = float(ref.groupby("position")[col].rank(pct=True).iloc[-1])

    return pd.Series({
        "fa_year": int(fr["fa_year_expected"]),
        "age_at_fa": age_at_fa,
        "games_3yr_avg":     float(avg3.get("hitterGameCount", 0)),
        "ab_3yr_avg":        float(avg3.get("hitterAb", 0)),
        "avg_3yr_avg":       float(avg3.get("hitterHra", 0)),
        "obp_3yr_avg":       float(avg3.get("hitterObp", 0)),
        "slg_3yr_avg":       float(avg3.get("hitterSlg", 0)),
        "ops_3yr_avg":       ops_avg,
        "isop_3yr_avg":      float(avg3.get("hitterIsop", 0)),
        "babip_3yr_avg":     float(avg3.get("hitterBabip", 0)),
        "woba_3yr_avg":      woba_avg,
        "wrc_plus_3yr_avg":  float(avg3.get("hitterWrcPlus", 0)),
        "hr_3yr_avg":        hr_avg,
        "rbi_3yr_avg":       rbi_avg,
        "run_3yr_avg":       float(avg3.get("hitterRun", 0)),
        "hit_3yr_avg":       float(avg3.get("hitterHit", 0)),
        "double_3yr_avg":    float(avg3.get("hitterH2", 0)),
        "triple_3yr_avg":    float(avg3.get("hitterH3", 0)),
        "bb_3yr_avg":        float(avg3.get("hitterBb", 0)),
        "hp_3yr_avg":        float(avg3.get("hitterHp", 0)),
        "kk_3yr_avg":        float(avg3.get("hitterKk", 0)),
        "sb_3yr_avg":        float(avg3.get("hitterSb", 0)),
        "cs_3yr_avg":        float(avg3.get("hitterCs", 0)),
        "gd_3yr_avg":        float(avg3.get("hitterGd", 0)),
        "wpa_3yr_avg":       float(avg3.get("hitterWpa", 0)),
        "war_3yr_avg":       war_avg,
        "war_3yr_sum":       war_sum,
        "war_last_year":     float(last["hitterWar"]) if not pd.isna(last["hitterWar"]) else 0.0,
        "ops_last_year":     float(last["hitterOps"]) if not pd.isna(last["hitterOps"]) else 0.0,
        "wrc_plus_last_year": float(last["hitterWrcPlus"]) if not pd.isna(last["hitterWrcPlus"]) else 0.0,
        "mvp_count":          int(star.get("mvp_count", 0)),
        "golden_glove_count": int(star.get("golden_glove_count", 0)),
        "allstar_count":      int(star.get("allstar_count", 0)),
        "national_team":      int(star.get("national_team", 0)),
        "postseason_experience": int(star.get("postseason_experience", 0)),
        "star_score":         star_score,
        **pct,
        "market_level":     float(recent_market),
        "age_squared":      age_at_fa ** 2,
        "prime_years_left": max(0, min(10, 35 - age_at_fa)),
        "star_x_war":       star_score * war_sum,
        "star_x_ops":       star_score * ops_avg,
        "war_sum_sq":       war_sum ** 2,
        "position_enc":     enc_map.get(pos_code, 5),
        "active_seasons":   active_seasons_h,
    })


def build_future_pitcher_row(player_name, fr, p_stats_df, p_star_df, p_df):
    rows = p_stats_df[p_stats_df["playerName"] == player_name].sort_values("collect_year", ascending=False)
    if rows.empty:
        return None
    year_col_s = "collect_year" if "collect_year" in rows.columns else "year"
    base_yr = int(rows[year_col_s].max())
    target_yrs = [base_yr - i for i in range(3)]
    recent3 = rows[rows[year_col_s].isin(target_yrs)]
    if recent3.empty:
        recent3 = rows.head(3)
    n_yrs = 3
    last   = rows.iloc[0]
    avg3   = recent3.sum(numeric_only=True).fillna(0) / n_yrs
    active_seasons_p = len(recent3)

    save_avg = float(avg3.get("pitcherSave", 0))
    hold_avg = float(avg3.get("pitcherHold", 0))
    role = "CL" if save_avg >= 10 else ("SU" if hold_avg >= 10 else "SP")

    existing = p_df[p_df["player_name"] == player_name]
    if not existing.empty:
        old = existing.sort_values("fa_year").iloc[-1]
        age_at_fa = int(old["age_at_fa"]) + (int(fr["fa_year_expected"]) - int(old["fa_year"]))
    else:
        age_at_fa = 32

    star_row = p_star_df[p_star_df["player_name"] == player_name]
    star = star_row.iloc[-1] if not star_row.empty else pd.Series({
        "mvp_count": 0, "golden_glove_count": 0, "allstar_count": 0,
        "national_team": 0, "postseason_experience": 0, "star_score": 0,
    })

    star_score   = int(star.get("star_score", 0))
    war_sum      = float(recent3["pitcherWar"].fillna(0).sum())
    war_avg      = war_sum / n_yrs
    games_avg    = float(avg3.get("pitcherGameCount", 0))

    recent_market = p_df[p_df["fa_year"] == p_df["fa_year"].max()]["annual_avg_salary"].median()
    if pd.isna(recent_market):
        recent_market = float(p_df["annual_avg_salary"].median())

    return pd.Series({
        "age_at_fa":           age_at_fa,
        "games_3yr_avg":       games_avg,
        "innings_3yr_avg":     float(avg3.get("pitcherInning", 0)),
        "era_3yr_avg":         float(avg3.get("pitcherEra", 0)),
        "whip_3yr_avg":        float(avg3.get("pitcherWhip", 0)),
        "win_3yr_avg":         float(avg3.get("pitcherWin", 0)),
        "lose_3yr_avg":        float(avg3.get("pitcherLose", 0)),
        "save_3yr_avg":        save_avg,
        "hold_3yr_avg":        hold_avg,
        "hit_allowed_3yr_avg": float(avg3.get("pitcherHit", 0)),
        "war_3yr_avg":         war_avg,
        "war_3yr_sum":         war_sum,
        "war_last_year":       float(last["pitcherWar"]) if not pd.isna(last["pitcherWar"]) else 0.0,
        "era_last_year":       float(last["pitcherEra"]) if not pd.isna(last["pitcherEra"]) else 0.0,
        "whip_last_year":      float(last["pitcherWhip"]) if not pd.isna(last["pitcherWhip"]) else 0.0,
        "mvp_count":           int(star.get("mvp_count", 0)),
        "golden_glove_count":  int(star.get("golden_glove_count", 0)),
        "allstar_count":       int(star.get("allstar_count", 0)),
        "national_team":       int(star.get("national_team", 0)),
        "postseason_experience": int(star.get("postseason_experience", 0)),
        "star_score":          star_score,
        "market_level":        float(recent_market),
        "age_squared":         age_at_fa ** 2,
        "prime_left":          max(0, min(10, 35 - age_at_fa)),
        "war_per_game":        war_sum / (games_avg * 3 + 1),
        "star_x_war":          star_score * war_sum,
        "role_SP":             1 if role == "SP" else 0,
        "role_SU":             1 if role == "SU" else 0,
        "role_CL":             1 if role == "CL" else 0,
        "pitcher_role":        role,
        "active_seasons":      active_seasons_p,
    })
