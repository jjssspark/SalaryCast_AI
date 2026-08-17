"""StoveLens AI — 데이터/모델 파일 로드.

호출 위치: app/app.py main()에서 load_data(), load_models(), load_season_stats() 호출.
데이터 파일: data/*.csv, models/*.pkl. 모두 utf-8-sig 인코딩.

이전 구현에서 고친 것:
1. data/future_fa_candidates.csv(42명, 실제 조사 결과)를 읽지 않고
   '데뷔 연도 + 10년'으로 후보를 추정한 뒤 포지션을 전원 "외야수",
   FA 등급을 전원 "B"로 하드코딩했다. 이제 그 파일을 그대로 쓴다.
2. 시즌 스탯이 상위 100여 명만 담긴 v2였다. 전 선수가 담긴 v3로 바꿨다.
3. 선수 사진을 화면에서 매번 네이버 API로 조회했다. 미리 수집한 CSV를 쓴다.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"


class DataLoadError(Exception):
    """데이터/모델 파일 로드 실패 시 발생. 사용자 친화적 메시지를 담는다."""


def _read(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(DATA / name, encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise DataLoadError(f"데이터 파일을 찾을 수 없습니다: {name}") from exc


@st.cache_data
def load_data():
    """구단·FA 계약 등 화면이 참고하는 기준 데이터.

    학습 테이블(*_training_v8.csv)은 여기서 읽지 않는다. 화면은 학습 표본이
    아니라 시즌 스탯에서 그때그때 입력을 만들어 쓴다. 학습 표본 안에서의
    상대 위치가 필요한 곳은 모델과 함께 저장해 둔 reference_dist_*.pkl을 쓴다.
    """
    return (
        _read("teams.csv"),
        _read("position_need.csv"),
        _read("future_fa_candidates.csv"),
        _read("fa_contracts_v4.csv"),
    )


@st.cache_data
def load_season_stats():
    return (
        _read("hitter_season_stats_2013_2026_v3.csv"),
        _read("pitcher_season_stats_2013_2026_v3.csv"),
    )


@st.cache_data
def load_awards():
    """MVP · 골든글러브 · 국가대표. 연도별 수상 한 건이 한 행이다.

    이전에는 star_features_hitter/pitcher.csv를 썼는데 FA 계약자 140명분만
    손으로 적어 둔 값이었고 틀린 것도 있었다. 지금은 연도별 수상자 명단에서
    만들어 전 선수를 덮는다 (scripts/collect_star_features.py).
    """
    return _read("star_features_v2.csv")


@st.cache_data
def load_birth_lookup() -> dict[int, int]:
    """선수 생년. player_id -> 연도. 없으면 빈 dict.

    네이버가 생년월일을 주지 않아 scripts/collect_player_birth.py가
    위키데이터에서 받아 채운다. 이름이 아니라 player_id를 키로 쓴다.
    같은 이름이 96쌍이라 이름으로 붙이면 다른 사람의 생년이 들어온다.
    """
    path = DATA / "player_birth_manual.csv"
    if not path.exists():
        return {}

    table = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["birth_year"])
    return {int(row.player_id): int(row.birth_year) for row in table.itertuples()}


@st.cache_data
def load_search_index():
    """검색용 선수 마스터와 사진.

    사진은 선수 이름이 아니라 player_id로 붙인다. 같은 이름이 96쌍 있어
    이름으로 붙이면 다른 사람 얼굴이 나온다.
    """
    return _read("player_master.csv"), _read("player_photos.csv")


def _load_bundle(label: str) -> dict:
    """meta에 적힌 멤버 모델만 읽는다. 블렌드 구성이 바뀌어도 로더는 그대로다."""
    try:
        meta = joblib.load(MODELS / f"{label}_v8_meta.pkl")
        models = {
            name: joblib.load(MODELS / f"{label}_v8_{name.lower()}.pkl")
            for name in meta["members"]
        }
        reference = joblib.load(MODELS / f"reference_dist_{label}_v8.pkl")
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"모델 파일을 찾을 수 없습니다: {Path(exc.filename).name}"
        ) from exc
    return {"meta": meta, "models": models, "reference": reference}


@st.cache_resource
def load_models():
    return _load_bundle("hitter"), _load_bundle("pitcher")
