"""StoveLens AI — 데이터/모델 파일 로드.

호출 위치: app/app.py main()에서 load_data(), load_models(), load_season_stats() 호출.
데이터 파일: data/*.csv (hitter_training_v7.csv, pitcher_training_v7.csv, teams.csv,
position_need.csv, fa_contracts_v3.csv, hitter/pitcher_season_stats_2015_2026_v2.csv,
star_features_hitter/pitcher.csv), models/*.pkl. 모두 utf-8-sig 인코딩.
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: data_loader.py" +
"데이터/모델 파일 로드 실패 시 사용자 친화적 에러 메시지 추가".
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import engineer_hitter_features, engineer_pitcher_features

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"


class DataLoadError(Exception):
    """데이터/모델 파일 로드 실패 시 발생. 사용자 친화적 메시지를 담는다."""


def get_future_fa_candidates(h_season_df, p_season_df):
    candidates = []
    for df, player_type in [(h_season_df, "hitter"), (p_season_df, "pitcher")]:
        year_col = "collect_year" if "collect_year" in df.columns else "year"
        team_col = "teamName" if "teamName" in df.columns else "team"

        debut = df.groupby("playerName")[year_col].min().reset_index()
        debut.columns = ["player_name", "debut_year"]
        debut["fa_year_expected"] = debut["debut_year"] + 10
        fa = debut[debut["fa_year_expected"].between(2027, 2030)].copy()
        fa["player_type"] = player_type

        season_cnt = df.groupby("playerName")[year_col].count()
        fa["_cnt"] = fa["player_name"].map(season_cnt).fillna(0)

        latest = (df.sort_values(year_col)
                    .groupby("playerName").last()
                    .reset_index()[["playerName", team_col]])
        latest.columns = ["player_name", "team_2026"]
        fa = fa.merge(latest, on="player_name", how="left")
        candidates.append(fa)

    combined = pd.concat(candidates, ignore_index=True)
    combined = (combined.sort_values("_cnt", ascending=False)
                        .drop_duplicates("player_name", keep="first")
                        .drop(columns="_cnt")
                        .reset_index(drop=True))
    combined["position"] = "외야수"
    combined["fa_grade"] = "B"
    return combined


@st.cache_data
def load_data():
    try:
        h = pd.read_csv(DATA / "hitter_training_v7.csv", encoding="utf-8-sig")
        p = pd.read_csv(DATA / "pitcher_training_v7.csv", encoding="utf-8-sig")
        teams = pd.read_csv(DATA / "teams.csv", encoding="utf-8-sig")
        pn = pd.read_csv(DATA / "position_need.csv", encoding="utf-8-sig")
        fa = pd.read_csv(DATA / "fa_contracts_v3.csv", encoding="utf-8-sig")
        h_stats = pd.read_csv(DATA / "hitter_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
        p_stats = pd.read_csv(DATA / "pitcher_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
    except FileNotFoundError as e:
        raise DataLoadError(f"데이터 파일을 찾을 수 없습니다: {Path(e.filename).name}") from e
    future = get_future_fa_candidates(h_stats, p_stats)
    h = engineer_hitter_features(h)
    p = engineer_pitcher_features(p)
    return h, p, teams, pn, future, fa


@st.cache_resource
def load_models():
    try:
        hm = joblib.load(MODELS / "hitter_model_meta.pkl")
        hx = joblib.load(MODELS / "hitter_xgb_model.pkl")
        hl = joblib.load(MODELS / "hitter_lgb_model.pkl")
        pm = joblib.load(MODELS / "pitcher_model_meta.pkl")
        px = joblib.load(MODELS / "pitcher_xgb_model.pkl")
        pr = joblib.load(MODELS / "pitcher_rf_model.pkl")
    except FileNotFoundError as e:
        raise DataLoadError(f"모델 파일을 찾을 수 없습니다: {Path(e.filename).name}") from e
    return hm, hx, hl, pm, px, pr


@st.cache_data
def load_season_stats():
    try:
        h_stats = pd.read_csv(DATA / "hitter_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
        p_stats = pd.read_csv(DATA / "pitcher_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
        h_star  = pd.read_csv(DATA / "star_features_hitter.csv", encoding="utf-8-sig")
        p_star  = pd.read_csv(DATA / "star_features_pitcher.csv", encoding="utf-8-sig")
    except FileNotFoundError as e:
        raise DataLoadError(f"데이터 파일을 찾을 수 없습니다: {Path(e.filename).name}") from e
    return h_stats, p_stats, h_star, p_star
