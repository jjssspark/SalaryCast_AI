"""
수동 보완이 필요한 항목을 뽑아 목록으로 낸다.

두 가지가 자동 수집으로 안 채워진다.

1. 선수 사진 — 네이버 profile에서 1,566명을 받았지만 현역 3명이 빠졌다.
2. 생년 — 네이버가 생년월일을 주지 않는다. 선수 상세 엔드포인트는 403이다.
   첫 시즌으로 추정해 봤지만 평균 오차가 6년이라 폐기했다.
   FA 계약 140건은 birth_year를 갖고 있으니 그 밖의 선수만 채우면 된다.

출력:
  output/reports/missing_photos.csv
  output/reports/missing_birth_year.csv
  data/player_birth_manual.csv   (이미 있으면 건드리지 않는다)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
REPORT_DIR = Path("output/reports")

# 생년을 채워야 할 우선순위. 낮을수록 먼저 필요하다.
PRIORITY_FUTURE_FA = 1   # 미래 FA 후보 — 예측 화면에 바로 뜬다
PRIORITY_ACTIVE = 2      # 2026 현역 — 검색으로 도달 가능


def missing_photos(master: pd.DataFrame, photos: pd.DataFrame) -> pd.DataFrame:
    """사진이 없는 선수 중 최근까지 뛴 사람만. 은퇴 선수는 화면에 뜰 일이 없다."""
    have = set(photos["player_id"])
    missing = master[~master["player_id"].isin(have)]
    active = missing[missing["latest_year"] >= 2025]

    cols = ["player_name", "player_id", "player_type", "team_latest",
            "latest_year", "season_count"]
    return active[cols].sort_values(
        ["latest_year", "season_count"], ascending=False
    ).reset_index(drop=True)


def missing_birth_year(
    master: pd.DataFrame,
    fa: pd.DataFrame,
    future: pd.DataFrame,
    games_2026: pd.Series,
) -> pd.DataFrame:
    known = set(fa.dropna(subset=["birth_year"])["player_name"])
    future_names = set(future["player_name"])

    need = master[
        (~master["player_name"].isin(known)) & (master["latest_year"] >= 2025)
    ].copy()

    need["games_2026"] = need["player_id"].map(games_2026).fillna(0).astype(int)
    need["priority"] = PRIORITY_ACTIVE
    need.loc[need["player_name"].isin(future_names), "priority"] = PRIORITY_FUTURE_FA
    need["birth_year"] = ""

    cols = ["priority", "player_name", "player_id", "player_type", "position",
            "team_latest", "latest_year", "games_2026", "birth_year"]
    return need[cols].sort_values(
        ["priority", "games_2026", "latest_year"], ascending=[True, False, False]
    ).reset_index(drop=True)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(DATA_DIR / "player_master.csv")
    photos = pd.read_csv(DATA_DIR / "player_photos.csv")
    fa = pd.read_csv(DATA_DIR / "fa_contracts_v4.csv")
    future = pd.read_csv(DATA_DIR / "future_fa_candidates.csv")

    seasons = pd.concat([
        pd.read_csv(DATA_DIR / "hitter_season_stats_2013_2026_v3.csv"),
        pd.read_csv(DATA_DIR / "pitcher_season_stats_2013_2026_v3.csv"),
    ])
    games_2026 = (
        seasons[seasons["collect_year"] == 2026]
        .groupby("player_id")["games"].max()
    )

    photo_gap = missing_photos(master, photos)
    birth_gap = missing_birth_year(master, fa, future, games_2026)

    photo_out = REPORT_DIR / "missing_photos.csv"
    birth_out = REPORT_DIR / "missing_birth_year.csv"
    photo_gap.to_csv(photo_out, index=False, encoding="utf-8-sig")
    birth_gap.to_csv(birth_out, index=False, encoding="utf-8-sig")

    template = DATA_DIR / "player_birth_manual.csv"
    if not template.exists():
        pd.DataFrame(columns=["player_name", "player_id", "birth_year"]).to_csv(
            template, index=False, encoding="utf-8-sig"
        )
        print(f"템플릿 생성: {template}")

    print(f"사진 없는 현역: {len(photo_gap)}명 -> {photo_out}")
    if len(photo_gap):
        print(photo_gap.to_string(index=False))

    print(f"\n생년 필요: {len(birth_gap)}명 -> {birth_out}")
    print(birth_gap["priority"].value_counts().sort_index().to_string())
    print("\n우선순위 1(미래 FA 후보) 전체")
    print(birth_gap[birth_gap.priority == PRIORITY_FUTURE_FA].to_string(index=False))


if __name__ == "__main__":
    main()
