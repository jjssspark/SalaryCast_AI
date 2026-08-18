"""StoveLens AI — 피처 엔지니어링 (학습·서빙 공용).

호출 위치:
  src/data_loader.py load_data()          -> engineer_hitter_features / engineer_pitcher_features
  src/ui/pages.py   render_search()       -> build_future_hitter_row / build_future_pitcher_row
  scripts/build_training_v8.py            -> build_hitter_row / build_pitcher_row

데이터 파일을 직접 read/write하지 않는다. 호출부가 넘긴 DataFrame만 가공한다.
입력은 data/*_season_stats_2010_2026_v4.csv 스키마를 따른다.

이전 구현에서 고친 것:
1. market_level이 groupby(fa_year)로 타깃 중앙값을 넣어 정답을 흘렸다. 직전 연도들만 쓰도록 바꿨다.
2. 최근 3시즌 합계를 시즌 개수와 무관하게 항상 3으로 나눴다. 실제 시즌 수로 나눈다.
3. 최근 3시즌을 연속된 세 해로 잡아, 중간에 기록이 빠지면 그만큼 손해였다.
   존재하는 최근 3시즌을 쓴다. 2025가 없으면 2026·2024·2023이 된다.
4. 진행 중인 시즌(2026)의 누적 스탯이 완료 시즌과 같은 취급을 받았다. 시즌 환산해서 쓴다.
5. 백분위를 매번 새로 계산해 기준 분포가 흔들렸다. 학습 시점 분포를 고정해서 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_WINDOW = 3
MARKET_WINDOW = 3

# 기록이 빈 해는 더 과거로 내려가 채우되, 여기까지만 본다.
# 상한이 없으면 은퇴 직전 선수가 10년 전 전성기 기록으로 평가된다.
LOOKBACK_LIMIT = 5

# 타자 누적/비율 스탯. 진행 중 시즌은 누적만 환산한다.
HITTER_COUNTING = {
    "games": "games", "ab": "ab", "hr": "hr", "rbi": "rbi", "run": "run",
    "hit": "hit", "h2": "double", "h3": "triple", "bb": "bb", "hbp": "hp",
    "so": "kk", "sb": "sb", "wpa": "wpa", "war": "war",
}
HITTER_RATE = {
    "avg": "avg", "obp": "obp", "slg": "slg", "ops": "ops", "isop": "isop",
    "babip": "babip", "woba": "woba", "wrc_plus": "wrc_plus",
}

PITCHER_COUNTING = {
    "games": "games", "innings": "innings", "win": "win", "lose": "lose",
    "save": "save", "hold": "hold", "qs": "qs", "so": "so", "bb": "bb",
    "hit": "hit_allowed", "wpa": "wpa", "war": "war",
}
PITCHER_RATE = {
    "era": "era", "whip": "whip", "k9": "k9", "bb9": "bb9",
    "k_bb": "k_bb", "ip_per_game": "ip_per_game",
}

POSITION_ENC = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5, "IF": 6, "DH": 7}
PITCHER_ROLE_ENC = {"CL": 0, "RP": 1, "SP": 2, "SU": 3}

HITTER_PCT_COLS = [
    "war_3yr_avg", "war_3yr_sum", "ops_3yr_avg",
    "wrc_plus_3yr_avg", "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg",
]
PITCHER_PCT_COLS = ["war_3yr_sum", "era_3yr_avg", "innings_3yr_avg"]

EMPTY_STAR = {
    "mvp_count": 0, "golden_glove_count": 0, "national_team": 0, "star_score": 0,
}

# star_score 가중치. 받기 어려운 상일수록 크게 둔다.
STAR_WEIGHT = {"mvp_count": 5, "golden_glove_count": 3, "national_team": 2}

# 이전 스타성 데이터에는 올스타 선정 횟수와 포스트시즌 경험도 있었지만 뺐다.
# 올스타는 한국어 위키백과에 2015~2024 중 9개 연도만 있고 2014년 이전이 없어
# 옛 선수가 체계적으로 과소 집계된다. 포스트시즌은 140명 중 대부분이 1이라
# 사실상 상수였다. 근거 없는 값을 넣느니 항목을 빼는 편이 낫다.


def star_counts(awards: pd.DataFrame, player_id, before_year: int) -> dict:
    """before_year 이전까지 쌓인 수상 이력.

    FA 계약은 그 시점까지의 경력을 보고 맺으므로 이후 수상은 세면 안 된다.
    2023년 FA인 선수의 2024년 골든글러브를 넣으면 미래를 보는 셈이다.

    national_team은 0/1이 아니라 선발 횟수다. 대회가 잦았던 시기의 선수가
    유리해질까 봐 0/1로 시작했는데, 재보니 횟수 쪽이 더 잘 맞았다
    (억 원 OOF R² — 타자 0.538 -> 0.545, 투수 0.649 -> 0.679).
    여러 대회에 계속 뽑히는 것 자체가 시장이 값을 매기는 신호로 보인다.
    """
    if awards is None or player_id is None or awards.empty:
        return dict(EMPTY_STAR)

    mine = awards[
        (awards["player_id"] == int(player_id)) & (awards["year"] < int(before_year))
    ]
    counts = {
        "mvp_count": int((mine["award"] == "MVP").sum()),
        "golden_glove_count": int((mine["award"] == "GG").sum()),
        "national_team": int((mine["award"] == "NT").sum()),
    }
    counts["star_score"] = sum(counts[key] * weight for key, weight in STAR_WEIGHT.items())
    return counts


def age_terms(age_at_fa: int | None) -> dict[str, float]:
    """나이 파생 항. 생년을 모르는 선수는 셋 다 결측으로 둔다.

    추측한 나이를 넣으면 prime_years_left가 잘못된 방향으로 강하게 작동한다.
    결측이면 트리 모델이 그 분기를 쓰지 않는다.
    """
    if age_at_fa is None or pd.isna(age_at_fa):
        return {"age_at_fa": np.nan, "age_squared": np.nan, "prime_years_left": np.nan}

    age = int(age_at_fa)
    return {
        "age_at_fa": age,
        "age_squared": age ** 2,
        "prime_years_left": max(0, min(10, 35 - age)),
    }


# --------------------------------------------------------------------------
# 시즌 선택 · 집계
# --------------------------------------------------------------------------

def select_recent_seasons(
    rows: pd.DataFrame,
    base_year: int,
    window: int = SEASON_WINDOW,
    lookback: int = LOOKBACK_LIMIT,
) -> pd.DataFrame:
    """base_year 이하에서 기록이 존재하는 최근 window개 시즌.

    연속된 세 해가 아니라 '존재하는' 세 시즌이다. 한 해가 통째로 비면
    그만큼 더 과거로 내려가 채우되 lookback년 전까지만 본다.

    부상으로 시즌을 '못 채운' 해까지 건너뛰는 안을 재봤지만 뺐다.
    홍창기 2025년(51경기)처럼 짧은 시즌을 제외하면 성적 평균은 실력에
    가까워지지만, '한 해를 부상으로 날렸다'는 사실 자체가 사라진다.
    시장은 그걸 계약금에 반영하기 때문에 제외하면 예측이 나빠졌다.
    (억 원 OOF R² — 타자 0.541 -> 0.537, 투수 0.718 -> 0.636)
    짧은 시즌의 영향은 대신 aggregate_seasons에서 출전량 가중으로 줄인다.
    """
    eligible = rows[rows["collect_year"].between(base_year - lookback, base_year)]
    selected = eligible.sort_values("collect_year", ascending=False).head(window)

    # 최근 window개 시즌 안에 한 번도 안 뛴 선수는 아무것도 돌려주지 않는다.
    # lookback이 5년이라 그 사이 한 시즌만 걸려도 행이 만들어지는데, 그 기록으로
    # 지금의 몸값을 설명할 수는 없다. 시즌 데이터를 2010년까지 넓히자 이대호
    # 2017년 계약(37.5억)이 2011년 시즌 하나로 학습에 들어왔다. 그전에는 2011년
    # 데이터가 없어서 우연히 걸러졌을 뿐, 규칙이 막고 있던 게 아니다.
    #
    # 해외에 다녀온 선수는 남는다. 김현수 2018년 계약은 2013~2015 세 시즌이
    # 붙는데, 실제로 시장도 그 성적을 보고 값을 매겼다.
    if selected.empty or int(selected["collect_year"].max()) < base_year - (window - 1):
        return selected.iloc[0:0]
    return selected


def resolve_player_id(
    seasons: pd.DataFrame, name: str, team: str, fa_year: int
) -> int | None:
    """동명이인 중 실제 계약 당사자를 고른다.

    KBO에는 같은 이름이 96쌍 있고 FA 계약 140건 중 11건이 여기 걸린다.
    김현수는 2015~2026을 매년 140경기 뛴 76290과 2025년 1경기 뛴 69516이 함께 잡힌다.
    FA 직전 3시즌 출전 경기 수로 고르고, 소속팀이 계약 팀과 겹치면 우선한다.

    학습(scripts/build_training_v8.py)과 서빙(src/serving.py)이 이 함수를 같이 쓴다.
    두 곳이 다른 규칙으로 고르면 같은 선수가 학습 때와 다른 사람이 된다.
    """
    same_name = seasons[seasons["player_name"] == name]
    if same_name.empty:
        return None

    window = same_name[
        same_name["collect_year"].between(fa_year - SEASON_WINDOW, fa_year - 1)
    ]
    pool = window if not window.empty else same_name
    if pool["player_id"].nunique() == 1:
        return int(pool["player_id"].iloc[0])

    scores = []
    for player_id, sub in pool.groupby("player_id"):
        games = pd.to_numeric(sub["games"], errors="coerce").fillna(0).sum()
        team_match = 1 if (sub["team"] == team).any() else 0
        scores.append((team_match, games, int(player_id)))

    return max(scores)[2]


def format_seasons_used(seasons: pd.DataFrame) -> str:
    """화면에 '어느 시즌을 썼는지' 그대로 보여주기 위한 표기."""
    years = sorted(int(y) for y in seasons["collect_year"].unique())
    return ",".join(str(y) for y in years)


def annualize(seasons: pd.DataFrame, counting_cols: list[str]) -> pd.DataFrame:
    """진행 중인 시즌의 누적 스탯을 완료 시즌 기준으로 환산한다.

    비율 스탯은 건드리지 않는다. 표본이 작을 뿐 값 자체는 편향되지 않는다.
    """
    out = seasons.copy()
    completion = out.get("season_completion")
    if completion is None:
        return out

    scale = 1.0 / pd.to_numeric(completion, errors="coerce").replace(0, np.nan).fillna(1.0)
    for col in counting_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * scale
    return out


def _column(seasons: pd.DataFrame, name: str) -> pd.Series | None:
    """숫자로 바꾼 컬럼. 없으면 None.

    DataFrame.get(없는 이름)은 None을 주고 pd.to_numeric(None)은 Series가 아니라
    스칼라 NaN을 돌려준다. 그대로 쓰면 .fillna()에서 터진다.
    """
    if name not in seasons.columns:
        return None
    return pd.to_numeric(seasons[name], errors="coerce")


def _playing_time(seasons: pd.DataFrame) -> pd.Series | None:
    """비율 스탯을 가중할 표본 크기. 타자는 타수, 투수는 이닝."""
    for column in ("ab", "innings"):
        if column in seasons.columns:
            return pd.to_numeric(seasons[column], errors="coerce").fillna(0.0)
    return None


def _weighted_mean(values: pd.Series | None, weight: pd.Series | None) -> float:
    if values is None:
        return 0.0

    numbers = values.astype(float)
    valid = numbers.notna()
    if not valid.any():
        return 0.0

    if weight is None:
        return float(numbers[valid].mean())

    weights = weight[valid].clip(lower=0.0)
    if weights.sum() <= 0:
        return float(numbers[valid].mean())

    return float((numbers[valid] * weights).sum() / weights.sum())


def aggregate_seasons(
    seasons: pd.DataFrame,
    counting: dict[str, str],
    rate: dict[str, str],
) -> dict[str, float]:
    """선택된 시즌들을 3년 평균/합계 피처로 접는다.

    평균은 실제 시즌 수로 나눈다. 시즌이 2개뿐인데 3으로 나누면
    성적이 3분의 2로 축소된다.
    """
    n = len(seasons)
    if n == 0:
        return {}

    annualized = annualize(seasons, list(counting))
    out: dict[str, float] = {}

    for src, name in counting.items():
        values = _column(annualized, src)

        if name == "war":
            # 2013~2016 타자 WAR은 네이버가 제공하지 않아 결측이다. 결측을 0으로 세면
            # 기여도가 없었던 것처럼 되므로, 값이 있는 시즌만 평균 내고 3시즌으로 환산한다.
            valid = values.dropna() if values is not None else pd.Series(dtype=float)
            mean = float(valid.mean()) if len(valid) else 0.0
            out["war_3yr_avg"] = mean
            out["war_3yr_sum"] = mean * n
            out["war_seasons_valid"] = int(len(valid))
            continue

        total = 0.0 if values is None else float(values.fillna(0).sum())
        out[f"{name}_3yr_avg"] = total / n

    # 비율 스탯은 출전량으로 가중해서 평균 낸다.
    # 단순 평균이면 174타수짜리 부상 시즌의 OPS가 524타수 시즌과 같은 무게를 갖는다.
    # 홍창기 2023~2025 OPS는 단순 평균 .813, 타수 가중 .835로 갈린다.
    weight = _playing_time(annualized)
    for src, name in rate.items():
        out[f"{name}_3yr_avg"] = _weighted_mean(_column(annualized, src), weight)

    out["active_seasons"] = n
    return out


def last_season_values(seasons: pd.DataFrame, mapping: dict[str, str]) -> dict[str, float]:
    """가장 최근 시즌 단일값. 진행 중 시즌이면 누적은 환산된 값이 들어간다."""
    if seasons.empty:
        return {name: 0.0 for name in mapping.values()}

    latest = annualize(seasons, list(mapping)).sort_values("collect_year").iloc[-1]
    out = {}
    for src, name in mapping.items():
        value = pd.to_numeric(pd.Series([latest.get(src)]), errors="coerce").iloc[0]
        out[name] = 0.0 if pd.isna(value) else float(value)
    return out


# --------------------------------------------------------------------------
# 시장 수준 (타깃 누수 제거)
# --------------------------------------------------------------------------

def market_level_prior(
    fa_df: pd.DataFrame, fa_year: int, window: int = MARKET_WINDOW
) -> float:
    """직전 window개 연도 FA 계약의 연평균 중앙값.

    이전 구현은 groupby(fa_year)로 같은 해 계약의 중앙값을 넣었다.
    자기 자신이 포함된 중앙값이라 정답을 그대로 흘리는 값이었다.
    가장 이른 연도는 참고할 과거가 없어 NaN으로 둔다. 트리 계열이 알아서 처리한다.
    """
    prior = fa_df[
        (fa_df["fa_year"] < fa_year) & (fa_df["fa_year"] >= fa_year - window)
    ]["annual_avg_salary"]
    return float(prior.median()) if len(prior) else np.nan


# --------------------------------------------------------------------------
# 백분위 (학습 시점 분포 고정)
# --------------------------------------------------------------------------

def build_reference_dist(
    df: pd.DataFrame, cols: list[str], group_col: str
) -> dict[str, dict]:
    """학습 데이터의 값 분포를 저장해 둔다. 서빙 때 이 분포에 대고 순위를 매긴다."""
    reference: dict[str, dict] = {}
    for col in cols:
        reference[col] = {
            "all": df[col].dropna().to_numpy(),
            "by_group": {
                str(key): sub[col].dropna().to_numpy()
                for key, sub in df.groupby(group_col)
            },
        }
    return reference


def percentile_of(value: float, sample: np.ndarray) -> float:
    """sample 중 value 이하인 비율. 표본이 없으면 중앙값 취급."""
    if value is None or pd.isna(value) or sample is None or len(sample) == 0:
        return 0.5
    return float((sample <= value).mean())


def apply_reference_ranks(
    row: dict, reference: dict[str, dict], cols: list[str], group_value: str
) -> dict:
    out = dict(row)
    for col in cols:
        ref = reference.get(col)
        if ref is None:
            out[f"{col}_all_pct"] = 0.5
            out[f"{col}_pos_pct"] = 0.5
            continue
        value = row.get(col)
        out[f"{col}_all_pct"] = percentile_of(value, ref["all"])
        out[f"{col}_pos_pct"] = percentile_of(
            value, ref["by_group"].get(str(group_value), ref["all"])
        )
    return out


def add_percentile_ranks(
    df: pd.DataFrame, cols: list[str], group_col: str
) -> pd.DataFrame:
    """학습 데이터 자체의 상대 순위. 학습 시에만 쓴다."""
    out = df.copy()
    for col in cols:
        out[f"{col}_all_pct"] = out[col].rank(pct=True)
        out[f"{col}_pos_pct"] = out.groupby(group_col)[col].rank(pct=True)
    return out


# --------------------------------------------------------------------------
# 나이
# --------------------------------------------------------------------------

def resolve_age_at(
    player_name: str,
    target_year: int,
    fa_df: pd.DataFrame,
    birth_by_id: dict[int, int] | None = None,
    player_id: int | None = None,
) -> int | None:
    """생년이 알려진 선수만 나이를 돌려준다. 모르면 None.

    첫 시즌으로 생년을 추정해 봤지만 쓸 수 없었다. 데뷔 연령을 18~23 사이
    어느 값으로 잡아도 평균 오차가 5.5~8년이었다. 우리 데이터가 2013년부터라
    그 이전 데뷔 선수는 첫 시즌이 실제 데뷔가 아니고, 외국인 선수는 KBO 첫
    시즌이 20대 후반이다. isKoreanPlayer 필드는 전 레코드가 False라 국적 구분에도
    못 쓴다. 6년 오차면 prime_years_left가 뒤집히므로 결측으로 두는 편이 낫다.

    출처는 두 곳이다.
      1) data/fa_contracts_v4.csv의 birth_year (FA 계약 140건, 위키데이터로 교정함)
      2) data/player_birth_manual.csv (수기 보완분)
    """
    # 이름이 아니라 player_id로 찾는다. 같은 이름이 96쌍이라 이름으로 붙이면
    # 다른 사람의 생년이 들어온다.
    if birth_by_id and player_id is not None:
        birth = birth_by_id.get(int(player_id))
        if birth:
            return int(target_year - int(birth))

    known = fa_df[fa_df["player_name"] == player_name]
    if not known.empty and "birth_year" in known.columns:
        birth = known.sort_values("fa_year")["birth_year"].dropna()
        if len(birth):
            return int(target_year - int(birth.iloc[-1]))

    return None


# --------------------------------------------------------------------------
# 행 조립 (학습·서빙 공용)
# --------------------------------------------------------------------------

def build_hitter_row(
    seasons: pd.DataFrame,
    fa_year: int,
    age_at_fa: int,
    position_code: str,
    star: dict,
    market_level: float,
) -> dict | None:
    if seasons.empty:
        return None

    row = aggregate_seasons(seasons, HITTER_COUNTING, HITTER_RATE)
    row.update(last_season_values(
        seasons,
        {"war": "war_last_year", "ops": "ops_last_year", "wrc_plus": "wrc_plus_last_year"},
    ))

    star_score = int(star.get("star_score", 0))
    row.update({
        "fa_year": int(fa_year),
        "position": position_code,
        **{k: int(star.get(k, 0)) for k in EMPTY_STAR},
        **age_terms(age_at_fa),
        "market_level": market_level,
        "star_x_war": star_score * row.get("war_3yr_sum", 0.0),
        "star_x_ops": star_score * row.get("ops_3yr_avg", 0.0),
        "war_sum_sq": row.get("war_3yr_sum", 0.0) ** 2,
        "position_enc": POSITION_ENC.get(position_code, 6),
    })
    return row


def build_pitcher_row(
    seasons: pd.DataFrame,
    fa_year: int,
    age_at_fa: int,
    role: str,
    star: dict,
    market_level: float,
) -> dict | None:
    if seasons.empty:
        return None

    row = aggregate_seasons(seasons, PITCHER_COUNTING, PITCHER_RATE)
    row.update(last_season_values(
        seasons,
        {"war": "war_last_year", "era": "era_last_year", "whip": "whip_last_year"},
    ))

    innings = row.get("innings_3yr_avg", 0.0)
    era = row.get("era_3yr_avg", 0.0)
    wins = row.get("win_3yr_avg", 0.0)
    losses = row.get("lose_3yr_avg", 0.0)
    decisions = wins + losses
    star_score = int(star.get("star_score", 0))

    row.update({
        "fa_year": int(fa_year),
        "pitcher_role": role,
        **{k: int(star.get(k, 0)) for k in EMPTY_STAR},
        **age_terms(age_at_fa),
        "hit_per_inn": row.get("hit_allowed_3yr_avg", 0.0) / innings if innings else 0.0,
        "whip_era_ratio": row.get("whip_3yr_avg", 0.0) / era if era else 1.0,
        "win_rate": wins / decisions if decisions else 0.5,
        "star_x_war": star_score * row.get("war_3yr_sum", 0.0),
        "role_x_save": row.get("save_3yr_avg", 0.0) if role == "CL" else 0.0,
        "role_x_hold": row.get("hold_3yr_avg", 0.0) if role == "SU" else 0.0,
        "role_x_inn": innings if role == "SP" else 0.0,
        "market_level": market_level,
        "role_enc": PITCHER_ROLE_ENC.get(role, 2),
    })
    return row


def build_serving_row(
    seasons: pd.DataFrame,
    fa_year: int,
    age_at_fa: int,
    group_value: str,
    star: dict,
    market_level: float,
    reference: dict,
    is_hitter: bool,
) -> dict | None:
    """화면에서 예측 한 건을 낼 때 쓰는 입력 한 행.

    학습 때와 같은 build_*_row를 쓰고, 백분위만 학습 시점 분포에 대고 매긴다.
    이전 구현은 학습 데이터에 새 선수를 덧붙여 rank를 다시 계산해서,
    같은 선수라도 학습 데이터가 바뀌면 순위가 흔들렸다.
    """
    builder = build_hitter_row if is_hitter else build_pitcher_row
    row = builder(seasons, fa_year, age_at_fa, group_value, star, market_level)
    if row is None:
        return None

    cols = HITTER_PCT_COLS if is_hitter else PITCHER_PCT_COLS
    row = apply_reference_ranks(row, reference, cols, group_value)
    row["seasons_used"] = format_seasons_used(seasons)
    return row


def classify_pitcher_role(seasons: pd.DataFrame) -> str:
    """v3 시즌 스탯의 pitcher_role 중 최근 시즌 값. 없으면 집계로 판정."""
    if "pitcher_role" in seasons.columns:
        roles = seasons.sort_values("collect_year")["pitcher_role"].dropna()
        if len(roles):
            return str(roles.iloc[-1])

    games = pd.to_numeric(seasons.get("games"), errors="coerce").sum()
    if not games:
        return "SP"
    if pd.to_numeric(seasons.get("save"), errors="coerce").sum() / games >= 0.30:
        return "CL"
    if pd.to_numeric(seasons.get("hold"), errors="coerce").sum() / games >= 0.25:
        return "SU"
    return "SP"


# --------------------------------------------------------------------------
# 학습 데이터 파생 피처 (data_loader가 호출)
# --------------------------------------------------------------------------

def engineer_hitter_features(df: pd.DataFrame) -> pd.DataFrame:
    """이미 조립된 학습 테이블에 상대 순위를 붙인다.

    market_level·age_squared 등 나머지 파생은 build_hitter_row에서 이미 계산돼 있다.
    """
    out = df.copy()
    if "position" in out.columns:
        out["position"] = out["position"].fillna("OF")
    return add_percentile_ranks(out, HITTER_PCT_COLS, "position")


def engineer_pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pitcher_role" in out.columns:
        out["pitcher_role"] = out["pitcher_role"].fillna("SP")
    return add_percentile_ranks(out, PITCHER_PCT_COLS, "pitcher_role")
