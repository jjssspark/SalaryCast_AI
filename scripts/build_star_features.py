"""수상 이력을 선수(player_id)에 붙인다.

실행: .venv/bin/python scripts/build_star_features.py
입력: data/kbo_awards.csv, data/player_master.csv, data/*_season_stats_2010_2026_v4.csv
출력: data/star_features_v2.csv  (player_id, player_name, year, award)

수집(collect_star_features.py)과 나눠 둔다. 이름을 선수에 붙이는 규칙만 고칠 때
위키백과를 다시 치지 않기 위해서다.

이름으로 그냥 붙이면 두 가지가 섞인다.
  1. 동명이인 — KBO에 같은 이름이 96쌍 있다.
  2. 감독·코치 — 국가대표 명단 문서에서 선수와 같은 표에 들어 있고,
     은퇴 후 코치가 된 옛 선수는 이름만으로는 구분이 안 된다.
둘 다 '그 해 전후로 실제 KBO에서 뛰었는가'로 거른다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

# 국가대표 명단은 선수와 감독·코치가 같은 문서에 있다. 은퇴 후 코치가 된
# 옛 선수를 걸러내려면 '그 대회 무렵 실제로 뛰었는가'를 봐야 한다.
NATIONAL_TEAM_WINDOW = 2

# MVP·골든글러브는 선수만 실린 명단이라 코치 위험이 없다. 다만 우리 시즌
# 데이터가 2013년부터라 그 이전 수상은 활동 기록으로 확인할 수 없다.
# KBO 선수 경력이 20년을 넘는 일은 드무므로, 기록 시작 시점보다 이만큼
# 앞선 수상까지만 같은 선수로 본다.
CAREER_BACKREACH = 14


def load_seasons() -> pd.DataFrame:
    frames = []
    for name in ("hitter", "pitcher"):
        path = DATA_DIR / f"{name}_season_stats_2010_2026_v4.csv"
        frames.append(
            pd.read_csv(path)[["player_id", "player_name", "collect_year", "team", "games"]]
        )
    return pd.concat(frames, ignore_index=True)


def resolve(seasons: pd.DataFrame, name: str, year: int, team: str, award: str) -> int | None:
    """수상자 한 명을 player_id로. 확정 못 하면 None."""
    same_name = seasons[seasons["player_name"] == name]
    if same_name.empty:
        return None

    if award == "NT":
        active = same_name[
            same_name["collect_year"].between(
                year - NATIONAL_TEAM_WINDOW, year + NATIONAL_TEAM_WINDOW
            )
        ]
    else:
        first = int(same_name["collect_year"].min())
        latest = int(same_name["collect_year"].max())
        active = same_name if first - CAREER_BACKREACH <= year <= latest + 1 else same_name.iloc[0:0]

    if active.empty:
        return None

    if active["player_id"].nunique() == 1:
        return int(active["player_id"].iloc[0])

    # 동명이인이면 소속팀이 겹치는 쪽, 그다음 출전 경기가 많은 쪽.
    scores = []
    for player_id, sub in active.groupby("player_id"):
        matched = 1 if team and (sub["team"].astype(str).str[:2] == team[:2]).any() else 0
        games = pd.to_numeric(sub["games"], errors="coerce").fillna(0).sum()
        scores.append((matched, games, int(player_id)))
    return max(scores)[2]


def main() -> None:
    print("=" * 62)
    print("  수상 이력 -> 선수 연결")
    print("=" * 62)

    awards = pd.read_csv(DATA_DIR / "kbo_awards.csv")
    seasons = load_seasons()

    rows, dropped = [], []
    for award in awards.itertuples():
        player_id = resolve(
            seasons, award.player_name, int(award.year), str(award.team or ""), award.award
        )
        if player_id is None:
            dropped.append((award.player_name, int(award.year), award.award))
            continue
        rows.append({
            "player_id": player_id,
            "player_name": award.player_name,
            "year": int(award.year),
            "award": award.award,
        })

    table = pd.DataFrame(rows).drop_duplicates(subset=["player_id", "year", "award"])
    out = DATA_DIR / "star_features_v2.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n저장: {out}  ({len(table)}건 / 선수 {table['player_id'].nunique()}명)")
    print(table.groupby("award").size().to_string())

    print(f"\n연결 실패 {len(dropped)}건 — 2013년 이전 은퇴·코치·외국인 등")
    print(pd.Series([d[2] for d in dropped]).value_counts().to_string())

    # 미래 FA 후보가 실제로 채워졌는지 본다. 이게 이번 작업의 목적이다.
    future = pd.read_csv(DATA_DIR / "future_fa_candidates_v2.csv")
    counts = table.groupby(["player_name", "award"]).size().unstack(fill_value=0)

    print("\n미래 FA 후보 42명 수상 이력")
    for name in future["player_name"]:
        if name not in counts.index:
            print(f"  {name:6s} 없음")
            continue
        row = counts.loc[name]
        parts = [f"{key} {int(row[key])}" for key in ("MVP", "GG", "NT") if key in row and row[key]]
        print(f"  {name:6s} {' · '.join(parts) if parts else '없음'}")

    covered = sum(1 for name in future["player_name"] if name in counts.index)
    print(f"\n  수상 이력 있음 {covered}/42명")


if __name__ == "__main__":
    main()
