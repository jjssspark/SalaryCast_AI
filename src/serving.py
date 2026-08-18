"""StoveLens AI — 검색된 선수 하나를 예측 한 건으로 조립한다.

호출 위치: src/ui/player.py, src/ui/home.py
데이터 파일을 직접 읽지 않는다. app/app.py가 data_loader로 만든 Context를 받는다.

화면 코드가 '어느 시즌을 쓸지, 나이를 어디서 가져올지, 시장 수준을 어떻게 잡을지'를
직접 정하면 학습 때와 어긋난다. 그 판단을 여기 모아두고 학습 스크립트
(scripts/build_training_v8.py)와 같은 src.features 함수만 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import (
    SEASON_WINDOW,
    build_serving_row,
    classify_pitcher_role,
    format_seasons_used,
    market_level_prior,
    resolve_age_at,
    select_recent_seasons,
    star_counts,
)
from src.predict import NotEnoughRecord, check_sample, predict_salary

# 네이버가 주는 포지션은 굵은 분류뿐이다. 모델은 학습 때 FA 계약서의
# 세부 포지션 코드를 봤으므로 가장 가까운 코드로 옮긴다.
POSITION_TO_CODE = {
    "포수": "C", "외야수": "OF", "내야수": "IF", "유격수": "SS",
    "야수": "IF", "지명타자": "DH", "1루수": "1B", "2루수": "2B", "3루수": "3B",
}
CODE_TO_KOREAN = {
    "C": "포수", "1B": "1루수", "2B": "2루수", "3B": "3루수", "SS": "유격수",
    "OF": "외야수", "IF": "내야수", "DH": "지명타자",
}
ROLE_TO_KOREAN = {"SP": "선발", "SU": "셋업", "CL": "마무리", "RP": "불펜"}


@dataclass(frozen=True)
class Context:
    """앱이 한 번 읽어두고 계속 쓰는 것들."""

    master: pd.DataFrame
    photos: pd.DataFrame
    hitter_seasons: pd.DataFrame
    pitcher_seasons: pd.DataFrame
    awards: pd.DataFrame
    fa: pd.DataFrame
    future: pd.DataFrame
    eligibility: pd.DataFrame
    extensions: pd.DataFrame
    birth: dict
    teams: pd.DataFrame
    position_need: pd.DataFrame
    hitter_bundle: dict
    pitcher_bundle: dict
    hitter_league: dict
    pitcher_league: dict
    photo_by_id: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Card:
    """화면 한 장에 필요한 모든 값."""

    player_id: int
    name: str
    team: str
    is_hitter: bool
    mode: str                 # past | extension | future | active
    fa_year_estimated: bool   # fa_year가 조사값이 아니라 규칙 추정일 때 True
    position_code: str | None
    position_label: str
    role: str | None
    fa_year: int
    base_year: int
    age: int | None
    seasons: pd.DataFrame
    seasons_used: str
    row: dict
    predicted: float
    low: float
    high: float
    actual: float | None
    star_score: int
    mvp_count: int
    golden_glove_count: int
    national_team: int
    photo_url: str | None


def _seasons_of(context: Context, is_hitter: bool) -> pd.DataFrame:
    return context.hitter_seasons if is_hitter else context.pitcher_seasons


def bundle_of(context: Context, is_hitter: bool) -> dict:
    return context.hitter_bundle if is_hitter else context.pitcher_bundle


def league_of(context: Context, is_hitter: bool) -> dict:
    return context.hitter_league if is_hitter else context.pitcher_league




def market_level_for(fa: pd.DataFrame, fa_year: int, is_hitter: bool) -> float:
    """직전 3년 FA 계약의 연평균 중앙값. 타자와 투수를 나눠서 센다.

    학습이 그렇게 만든다 — scripts/build_training_v8.py가 fa를 fa_hitters와
    fa_pitchers로 쪼개 넘긴다. 투수 시장이 타자보다 눅어서 한 덩어리로 세면
    2025년 기준 타자 11.5억 / 투수 6.1억이 둘 다 9.75억이 된다.

    직전 3년이 비는 경우가 둘인데 서로 다르게 다뤄야 한다.

    하나는 가장 이른 FA 연도(2018)다. 과거가 없어서 비는 것이라 그해 계약의
    중앙값을 쓴다. 여기서 최신 시장 수준을 쓰면 2018년 계약을 2027년 시세로
    환산하게 된다.

    다른 하나는 아직 오지 않은 FA 연도다. 미래라서 비는 것이므로 우리가 아는
    가장 최근 3개 연도를 쓴다.

    v9는 이 값이 예측의 곱셈 계수라, 어긋나면 예측이 그만큼 밀린다.
    """
    same_type = fa[(fa["position"] == "P") != is_hitter]

    level = market_level_prior(same_type, fa_year)
    if not pd.isna(level):
        return float(level)

    same_year = same_type[same_type["fa_year"] == fa_year]["annual_avg_salary"]
    if len(same_year):
        return float(same_year.median())

    return float(market_level_prior(same_type, int(same_type["fa_year"].max()) + 1))


def _is_hitter(context: Context, player_id: int) -> bool:
    row = context.master[context.master["player_id"] == player_id]
    return bool(len(row)) and str(row.iloc[0]["player_type"]) == "hitter"


def _past_contract(context: Context, player_id: int) -> pd.Series | None:
    """이 선수가 실제로 맺은 FA 계약 중 가장 최근 건.

    계약 파일(fa_contracts_v6.csv)에 player_id가 박혀 있어 그대로 맞춘다.
    예전에는 이름으로 찾고 출전 경기 수·소속팀으로 동명이인을 갈랐는데,
    같은 이름이 61쌍이라 시즌 기록이 한 줄만 바뀌어도 다른 사람의 계약이
    붙을 수 있었다. 확정한 값을 파일에 적어두는 쪽으로 옮겼다
    (scripts/add_player_ids.py).
    """
    same = context.fa[context.fa["player_id"] == player_id]
    return None if same.empty else same.sort_values("fa_year").iloc[-1]


def _extension_of(context: Context, name: str, master_row: pd.Series) -> pd.Series | None:
    """이 선수의 비FA 다년계약. 이름과 소속팀이 둘 다 맞을 때만 인정한다.

    같은 이름이 96쌍 있어 이름만으로 붙이면 다른 사람 계약이 딸려온다.
    """
    table = context.extensions
    if table.empty:
        return None

    team = str(master_row.get("team_latest") or "")
    found = table[(table["player_name"] == name) & (table["team"] == team)]
    return None if found.empty else found.sort_values("sign_date").iloc[-1]


def build_card(context: Context, player_id: int) -> Card:
    """선수 한 명의 예측 카드. 표본이 부족하면 NotEnoughRecord를 던진다."""
    found = context.master[context.master["player_id"] == player_id]
    if found.empty:
        raise NotEnoughRecord("등록되지 않은 선수입니다.")
    master_row = found.iloc[0]

    name = str(master_row["player_name"])
    is_hitter = str(master_row["player_type"]) == "hitter"
    seasons_all = _seasons_of(context, is_hitter)
    player_seasons = seasons_all[seasons_all["player_id"] == player_id]
    if player_seasons.empty:
        raise NotEnoughRecord("최근 기록이 없습니다.")

    future_row = context.future[context.future["player_id"] == player_id]
    estimated_row = context.eligibility[context.eligibility["player_id"] == player_id]
    contract = _past_contract(context, player_id)
    extension = _extension_of(context, name, master_row)

    # 비FA 다년계약을 맺은 선수는 그 기간 동안 FA 시장에 나오지 않는다.
    # 다음 FA가 얼마일지를 묻는 것 자체가 틀린 질문이라 가장 먼저 걸러낸다.
    fa_year_estimated = False
    if extension is not None:
        mode = "extension"
        fa_year = int(str(extension["sign_date"])[:4])
        base_year = fa_year - 1
        actual = float(extension["annual_avg_salary"])
    # 앞으로 FA가 예정된 선수라면 그쪽이 우선이다. 이미 끝난 계약보다
    # 다음 계약이 얼마일지가 이 서비스에서 궁금한 값이다.
    elif not future_row.empty:
        mode = "future"
        fa_year = int(future_row.iloc[0]["fa_year_expected"])
        base_year = int(player_seasons["collect_year"].max())
        actual = None
    elif contract is not None:
        mode = "past"
        fa_year = int(contract["fa_year"])
        base_year = fa_year - 1
        actual = float(contract["annual_avg_salary"])
    # 조사한 목록에 없는 현역은 데뷔 연도와 뛴 시즌 수로 자격 연도를 추정한다.
    # 추정임을 카드에 달아 보내고, 화면이 그대로 표시한다
    # (scripts/estimate_fa_eligibility.py).
    elif not estimated_row.empty:
        mode = "future"
        fa_year_estimated = True
        fa_year = int(estimated_row.iloc[0]["fa_year_expected"])
        base_year = int(player_seasons["collect_year"].max())
        actual = None
    else:
        mode = "active"
        base_year = int(player_seasons["collect_year"].max())
        fa_year = base_year + 1
        actual = None

    seasons = select_recent_seasons(player_seasons, base_year, SEASON_WINDOW)
    if seasons.empty:
        raise NotEnoughRecord("해당 시점의 시즌 기록이 없습니다.")
    check_sample(seasons, is_hitter)

    if mode == "past":
        position_code = str(contract["position"])
    else:
        position_code = POSITION_TO_CODE.get(str(master_row.get("position") or ""), "IF")

    role = None if is_hitter else classify_pitcher_role(seasons)
    group_value = position_code if is_hitter else role

    star = star_counts(context.awards, player_id, fa_year)
    bundle = bundle_of(context, is_hitter)

    row = build_serving_row(
        seasons=seasons,
        fa_year=fa_year,
        age_at_fa=resolve_age_at(name, fa_year, context.fa, context.birth, player_id),
        group_value=group_value,
        star=star,
        market_level=market_level_for(context.fa, fa_year, is_hitter),
        reference=bundle["reference"],
        is_hitter=is_hitter,
    )
    if row is None:
        raise NotEnoughRecord("예측에 필요한 값을 만들 수 없습니다.")

    predicted = predict_salary(row, bundle)
    margin = float(bundle["meta"]["mae_억"])
    age = row.get("age_at_fa")

    return Card(
        player_id=int(player_id),
        name=name,
        team=str(master_row.get("team_latest") or "무소속"),
        is_hitter=is_hitter,
        mode=mode,
        fa_year_estimated=fa_year_estimated,
        position_code=position_code if is_hitter else None,
        position_label=(
            CODE_TO_KOREAN.get(position_code, position_code)
            if is_hitter
            else ROLE_TO_KOREAN.get(role, "투수")
        ),
        role=role,
        fa_year=fa_year,
        base_year=base_year,
        age=None if age is None or pd.isna(age) else int(age),
        seasons=seasons,
        seasons_used=format_seasons_used(seasons),
        row=row,
        predicted=predicted,
        low=max(0.0, predicted - margin),
        high=predicted + margin,
        actual=actual,
        star_score=int(star["star_score"]),
        mvp_count=int(star["mvp_count"]),
        golden_glove_count=int(star["golden_glove_count"]),
        national_team=int(star["national_team"]),
        photo_url=context.photo_by_id.get(int(player_id)),
    )


# --------------------------------------------------------------------------
# 화면에 바로 얹을 수 있게 다듬은 값들
# --------------------------------------------------------------------------

def season_table(card: Card) -> pd.DataFrame:
    """최근 3시즌 원본 기록."""
    if card.is_hitter:
        columns = [
            ("collect_year", "시즌"), ("team", "팀"), ("games", "경기"),
            ("ab", "타수"), ("avg", "타율"), ("obp", "출루"), ("slg", "장타"),
            ("ops", "OPS"), ("hr", "홈런"), ("rbi", "타점"), ("sb", "도루"),
            ("war", "WAR"),
        ]
    else:
        columns = [
            ("collect_year", "시즌"), ("team", "팀"), ("games", "경기"),
            ("innings", "이닝"), ("era", "ERA"), ("whip", "WHIP"),
            ("win", "승"), ("lose", "패"), ("save", "세이브"),
            ("hold", "홀드"), ("so", "삼진"), ("war", "WAR"),
        ]

    present = [(src, label) for src, label in columns if src in card.seasons.columns]
    table = card.seasons[[src for src, _ in present]].copy()
    table.columns = [label for _, label in present]
    return table.sort_values("시즌", ascending=False)


def stat_panel(card: Card) -> list[dict]:
    """전광판에 띄울 스탯 4개. 투수는 역할에 따라 다르게 고른다."""
    row = card.row

    if card.is_hitter:
        return [
            {"key": "war", "label": "WAR", "value": row.get("war_3yr_avg"), "fmt": "{:.1f}"},
            {"key": "ops", "label": "OPS", "value": row.get("ops_3yr_avg"), "fmt": "{:.3f}"},
            {"key": "hr", "label": "홈런", "value": row.get("hr_3yr_avg"), "fmt": "{:.1f}"},
            {"key": "rbi", "label": "타점", "value": row.get("rbi_3yr_avg"), "fmt": "{:.1f}"},
        ]

    if card.role == "CL":
        third = {"key": "save", "label": "세이브", "value": row.get("save_3yr_avg"), "fmt": "{:.0f}"}
    elif card.role == "SU":
        third = {"key": "hold", "label": "홀드", "value": row.get("hold_3yr_avg"), "fmt": "{:.0f}"}
    else:
        third = {"key": "innings", "label": "이닝", "value": row.get("innings_3yr_avg"), "fmt": "{:.0f}"}

    return [
        {"key": "war", "label": "WAR", "value": row.get("war_3yr_avg"), "fmt": "{:.1f}"},
        {"key": "era", "label": "ERA", "value": row.get("era_3yr_avg"), "fmt": "{:.2f}"},
        third,
        {"key": "whip", "label": "WHIP", "value": row.get("whip_3yr_avg"), "fmt": "{:.2f}"},
    ]


def verdict(card: Card) -> dict:
    """예측과 실제 계약의 차이를 한 줄로. 실제 계약이 있는 past·extension 전용."""
    gap = card.actual - card.predicted
    error = abs(gap) / card.actual if card.actual else 0.0

    if error <= 0.15:
        title, tone = "적정 계약", "even"
    elif gap > 0:
        title, tone = "고평가 계약", "over"
    else:
        title, tone = "저평가 계약", "under"

    return {
        "title": title,
        "tone": tone,
        "text": (
            f"실제가 예측보다 {abs(gap):.1f}억 "
            f"{'높음' if gap > 0 else '낮음'} (오차 {error * 100:.0f}%)."
        ),
    }


def salary_scale(context: Context) -> float:
    """예상 범위 막대의 오른쪽 끝. 역대 FA 최고 연평균."""
    return float(context.fa["annual_avg_salary"].max())


def position_median(context: Context, card: Card) -> tuple[float, int] | None:
    """같은 포지션 FA 계약의 연평균 중앙값과 건수."""
    key = card.position_code if card.is_hitter else "P"
    same = context.fa[context.fa["position"] == key]
    if same.empty:
        return None
    return float(same["annual_avg_salary"].median()), int(len(same))


def overall_median(context: Context, is_hitter: bool) -> float:
    mask = context.fa["position"] != "P" if is_hitter else context.fa["position"] == "P"
    return float(context.fa[mask]["annual_avg_salary"].median())


def team_offers(context: Context, card: Card) -> pd.DataFrame:
    """구단별 예상 제시가. 포지션 필요도·우승 의지·재정 여력을 곱한다."""
    key = card.position_code if card.is_hitter else "P"
    matched = context.position_need
    matched = matched[matched["position"] == key][["team_name", "need_score"]]

    merged = context.teams.merge(matched, on="team_name", how="left")
    merged["need_score"] = merged["need_score"].fillna(0.5)
    merged["offer"] = card.predicted * (
        1
        + merged["need_score"] * 0.20
        + merged["win_now_score"] * 0.15
        + merged["cap_space_score"] * 0.10
    )
    merged["low"] = (merged["offer"] * 0.9).round(1)
    merged["high"] = (merged["offer"] * 1.1).round(1)
    return merged.sort_values("offer", ascending=False).reset_index(drop=True)


def resolve_by_name(context: Context, name: str) -> int | None:
    """이름으로 대표 선수 하나. 동명이인이면 시즌 수가 많은 쪽이 그 선수다."""
    found = context.master[context.master["player_name"] == name]
    if found.empty:
        return None
    found = found.sort_values(["season_count", "latest_year"], ascending=False)
    return int(found.iloc[0]["player_id"])


def headline_candidates(context: Context, limit: int = 5) -> list[dict]:
    """홈 화면 타일. 예측이 나오는 미래 FA 후보 중 예상 연봉 상위."""
    rows = []
    for _, candidate in context.future.iterrows():
        player_id = resolve_by_name(context, str(candidate["player_name"]))
        if player_id is None:
            continue
        try:
            card = build_card(context, player_id)
        except NotEnoughRecord:
            continue

        rows.append({
            "player_id": player_id,
            "name": card.name,
            "team": str(candidate["team_2026"]),
            "position": str(candidate["position"]),
            "fa_year": int(candidate["fa_year_expected"]),
            "predicted": card.predicted,
            "age": card.age,
        })

    rows.sort(key=lambda item: item["predicted"], reverse=True)
    return rows[:limit]


def ticker_rows(context: Context, limit: int = 12) -> list[dict]:
    """상단 흐르는 띠. 예측 여부와 무관하게 후보 명단 그대로."""
    upcoming = context.future.sort_values(["fa_year_expected", "player_name"])
    return [
        {
            "year": int(row["fa_year_expected"]),
            "name": str(row["player_name"]),
            "team": str(row["team_2026"]),
            "position": str(row["position"]),
        }
        for _, row in upcoming.head(limit).iterrows()
    ]


def counters(context: Context) -> list[tuple[str, int]]:
    return [
        ("학습 계약", int(len(context.fa))),
        ("검색 선수", int(len(context.master))),
        ("구단", int(len(context.teams))),
    ]


def safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(result) else result
