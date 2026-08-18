"""StoveLens AI — 데이터/모델 파일 로드.

호출 위치: app/app.py main()에서 load_data(), load_models(), load_season_stats() 호출.
데이터 파일: data/*.csv, models/*.pkl. 모두 utf-8-sig 인코딩.

이전 구현에서 고친 것:
1. data/future_fa_candidates_v2.csv(42명, 실제 조사 결과)를 읽지 않고
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

    FA 계약과 FA 예정 목록에는 player_id가 박혀 있다. 이름으로 붙이면 같은
    이름 61쌍이 서로의 계약을 가져간다 (scripts/add_player_ids.py).

    fa_eligibility_estimated.csv는 조사값이 아니라 규칙으로 뽑은 추정 자격
    연도다. 조사한 42명과 섞지 않으려고 파일을 나눠 둔다.

    학습 테이블(*_training_v8.csv)은 여기서 읽지 않는다. 화면은 학습 표본이
    아니라 시즌 스탯에서 그때그때 입력을 만들어 쓴다. 학습 표본 안에서의
    상대 위치가 필요한 곳은 모델과 함께 저장해 둔 reference_dist_*.pkl을 쓴다.
    """
    return (
        _read("teams.csv"),
        _read("position_need.csv"),
        _read("future_fa_candidates_v2.csv"),
        _read("fa_contracts_v6.csv"),
        _read("fa_eligibility_estimated.csv"),
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
def load_extensions() -> pd.DataFrame:
    """비FA 다년계약. FA 계약과 섞지 않는다.

    FA를 거치지 않고 구단과 장기 계약을 맺은 선수는 계약 기간 동안 시장에 나오지
    않는다. fa_contracts_v6.csv에 넣으면 FA가 아닌 계약이 학습 정답에 섞이므로
    파일을 따로 둔다. 화면에서만 쓴다.
    """
    path = DATA / "non_fa_extensions.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["player_name", "team", "sign_date", "annual_avg_salary", "through_year"]
        )
    return pd.read_csv(path, encoding="utf-8-sig")


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

    우선순위는 수기 > API다. 네이버에 사진이 없는 선수를 손으로 채울 수 있게
    열어 둔 자리다 (output/reports/missing_photos.csv가 대상 목록).
    앱은 CSV만 읽는다. 실행 중 외부 호출은 하지 않는다.
    """
    return _read("player_master.csv"), _merge_manual_photos(_read("player_photos.csv"))


def _merge_manual_photos(photos: pd.DataFrame) -> pd.DataFrame:
    """data/player_photos_manual.csv를 API 수집분 위에 덮는다.

    photo_url이 있으면 그대로 쓰고, 없고 photo_file만 있으면 assets/photos/ 아래
    파일을 가리킨다. 둘 다 비면 그 행은 버린다 — 채울 자리만 적어 둔 것이기 때문이다.
    """
    path = DATA / "player_photos_manual.csv"
    if not path.exists():
        return photos

    manual = pd.read_csv(path, encoding="utf-8-sig")
    if manual.empty:
        return photos

    def source_of(row) -> str | None:
        url = str(row.get("photo_url") or "").strip()
        if url and url.lower() != "nan":
            return url
        name = str(row.get("photo_file") or "").strip()
        if name and name.lower() != "nan":
            return f"assets/photos/{name}"
        return None

    manual = manual.assign(photo_url=[source_of(row) for _, row in manual.iterrows()])
    manual = manual.dropna(subset=["photo_url", "player_id"])
    if manual.empty:
        return photos

    manual = manual.assign(source="manual")[["player_id", "player_name", "photo_url", "source"]]
    kept = photos[~photos["player_id"].isin(manual["player_id"])]
    return pd.concat([kept, manual], ignore_index=True)


def _load_bundle(label: str) -> dict:
    """meta에 적힌 멤버 모델만 읽는다. 블렌드 구성이 바뀌어도 로더는 그대로다."""
    try:
        meta = joblib.load(MODELS / f"{label}_v9_meta.pkl")
        models = {
            name: joblib.load(MODELS / f"{label}_v9_{name.lower()}.pkl")
            for name in meta["members"]
        }
        reference = joblib.load(MODELS / f"reference_dist_{label}_v9.pkl")
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"모델 파일을 찾을 수 없습니다: {Path(exc.filename).name}"
        ) from exc
    return {"meta": meta, "models": models, "reference": reference}


@st.cache_resource
def load_models():
    return _load_bundle("hitter"), _load_bundle("pitcher")
