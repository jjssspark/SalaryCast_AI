"""
FA 계약 + 시즌 스탯(v3) -> 학습 데이터 v8

v7까지는 학습 데이터를 노트북에서 만들고 서빙 피처는 src/features.py에서 따로 만들었다.
같은 계산이 두 벌로 존재해 한쪽만 고쳐지는 일이 반복됐다. 이제 양쪽 다
src.features의 build_hitter_row / build_pitcher_row를 쓴다.

v7 대비 달라진 것:
- market_level에서 타깃 누수 제거
- 최근 3시즌을 '존재하는' 3시즌으로 선택
- 시즌 수로 나눔 (항상 3으로 나누던 것)
- cs/gd 제외 (원본이 100% 결측)
- 투수 스탯 추가 (QS, 삼진, 볼넷, K/9, BB/9, K/BB, 경기당 이닝)

출력:
  data/hitter_training_v8.csv
  data/pitcher_training_v8.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (  # noqa: E402
    SEASON_WINDOW,
    build_hitter_row,
    build_pitcher_row,
    classify_pitcher_role,
    format_seasons_used,
    market_level_prior,
    resolve_player_id,
    select_recent_seasons,
    star_counts,
)

DATA_DIR = Path("data")


def load_awards(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "player_name", "year", "award"])
    return pd.read_csv(path)


def build(
    fa: pd.DataFrame,
    seasons: pd.DataFrame,
    awards: pd.DataFrame,
    is_pitcher: bool,
) -> tuple[pd.DataFrame, list[str]]:
    rows, skipped = [], []

    for _, contract in fa.iterrows():
        name = contract["player_name"]
        fa_year = int(contract["fa_year"])

        player_id = resolve_player_id(seasons, name, contract["team"], fa_year)
        if player_id is None:
            skipped.append(f"{name}({fa_year})")
            continue
        player_seasons = seasons[seasons["player_id"] == player_id]

        # FA 계약은 직전 시즌까지의 성적을 보고 맺는다.
        picked = select_recent_seasons(player_seasons, fa_year - 1, SEASON_WINDOW)
        if picked.empty:
            skipped.append(f"{name}({fa_year})")
            continue

        star = star_counts(awards, player_id, fa_year)
        market = market_level_prior(fa, fa_year)
        age = int(contract["age_at_fa"])

        if is_pitcher:
            row = build_pitcher_row(
                picked, fa_year, age, classify_pitcher_role(picked), star, market
            )
        else:
            row = build_hitter_row(
                picked, fa_year, age, str(contract["position"]), star, market
            )

        if row is None:
            skipped.append(f"{name}({fa_year})")
            continue

        row["player_name"] = name
        row["player_id"] = player_id
        row["seasons_used"] = format_seasons_used(picked)
        row["team"] = contract["team"]
        row["annual_avg_salary"] = float(contract["annual_avg_salary"])
        rows.append(row)

    if not rows:
        return pd.DataFrame(), skipped

    df = pd.DataFrame(rows)
    lead = ["player_name", "fa_year", "age_at_fa", "team", "annual_avg_salary"]
    ordered = lead + [c for c in df.columns if c not in lead]
    return df[ordered], skipped


def main() -> None:
    print("=" * 62)
    print("  학습 데이터 v8 조립")
    print("=" * 62)

    fa = pd.read_csv(DATA_DIR / "fa_contracts_v4.csv")
    h_seasons = pd.read_csv(DATA_DIR / "hitter_season_stats_2013_2026_v3.csv")
    p_seasons = pd.read_csv(DATA_DIR / "pitcher_season_stats_2013_2026_v3.csv")

    fa_pitchers = fa[fa["position"] == "P"]
    fa_hitters = fa[fa["position"] != "P"]
    print(f"\nFA 계약 {len(fa)}건 — 타자 {len(fa_hitters)} / 투수 {len(fa_pitchers)}")

    awards = load_awards(DATA_DIR / "star_features_v2.csv")
    hitters, h_skipped = build(fa_hitters, h_seasons, awards, False)
    pitchers, p_skipped = build(fa_pitchers, p_seasons, awards, True)

    h_out = DATA_DIR / "hitter_training_v8.csv"
    p_out = DATA_DIR / "pitcher_training_v8.csv"
    hitters.to_csv(h_out, index=False, encoding="utf-8-sig")
    pitchers.to_csv(p_out, index=False, encoding="utf-8-sig")

    print(f"\n저장: {h_out}  ({len(hitters)}행 / {hitters.shape[1]}컬럼)")
    print(f"저장: {p_out}  ({len(pitchers)}행 / {pitchers.shape[1]}컬럼)")

    if h_skipped:
        print(f"\n타자 제외 {len(h_skipped)}건 (해당 시점 시즌 기록 없음): {h_skipped}")
    if p_skipped:
        print(f"투수 제외 {len(p_skipped)}건: {p_skipped}")

    for label, df in [("타자", hitters), ("투수", pitchers)]:
        if df.empty:
            continue
        print(f"\n{label} 유효 시즌 수 분포")
        print(df["active_seasons"].value_counts().sort_index().to_string())
        print(f"{label} market_level 결측 {df['market_level'].isna().sum()}건 (최초 연도)")


if __name__ == "__main__":
    main()
