"""
네이버 스포츠 KBO 전 선수 재수집 (2010~2026)

기존 crawl_extend.py를 대체한다. 두 가지가 달라졌다.
1. API 파라미터가 sortField/sortDirection -> field/direction 으로 바뀌어 기존 스크립트는 400을 받는다.
2. pageSize를 키우면 정렬 한 번으로 해당 시즌 전 선수가 나온다. 기존 수집분은 시즌당 상위 100여 명뿐이었다.

응답의 profile 컬럼에 실제 선수 사진 URL이 들어 있다(playerImageUrl은 전량 404).
사진 테이블은 scripts/build_season_stats_v3.py에서 이 원시 파일을 읽어 만든다.

출력:
  data/naver_hitter_2010_2026_raw_v2.csv
  data/naver_pitcher_2010_2026_raw_v2.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
BASE_URL = "https://api-gw.sports.naver.com/statistics/categories/kbo/seasons/{year}/players"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://m.sports.naver.com/kbaseball/record/kbo",
}

# 2010년부터 받는다. FA 계약은 직전 3시즌으로 피처를 만들므로, 2010년 시즌이
# 있어야 2013년 FA를 담을 수 있다. 2009년까지도 API는 응답하지만 그해 FA
# 계약(2010년 적용)은 담을 수 없어 의미가 없다.
#
# WAR는 2017년부터만 온다. 그 이전 시즌은 전 선수 0으로 내려오고,
# build_season_stats_v3가 그런 해를 결측으로 바꾼다. 2010~2016년 FA 계약은
# 최상위 피처인 WAR 없이 학습에 들어간다.
YEARS = list(range(2010, 2027))
PAGE_SIZE = 500

# WAR 정렬만으로는 기록이 극소량인 선수가 빠질 수 있어 출전 기준 정렬을 함께 돈다.
HITTER_SORTS = ["hitterWar", "hitterGameCount", "hitterAb"]
PITCHER_SORTS = ["pitcherWar", "pitcherGameCount", "pitcherSave", "pitcherHold"]


def fetch_season(year: int, player_type: str, field: str) -> list[dict]:
    """한 시즌·한 정렬기준의 전 선수를 페이지 끝까지 가져온다."""
    rows: list[dict] = []
    page = 1
    while True:
        params = {
            "playerType": player_type,
            "gameType": "REGULAR_SEASON",
            "field": field,
            "direction": "DESC",
            "page": page,
            "pageSize": PAGE_SIZE,
        }
        try:
            resp = requests.get(
                BASE_URL.format(year=year), headers=HEADERS, params=params, timeout=20
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"    [오류] {year} {player_type} {field} page={page}: {exc}")
            break

        if not payload.get("success"):
            print(f"    [실패] {year} {player_type} {field}: {payload.get('message')}")
            break

        players = payload.get("result", {}).get("seasonPlayerStats", [])
        if not players:
            break

        for p in players:
            row = dict(p)
            row["collect_year"] = year
            row["sort_source"] = field
            rows.append(row)

        if len(players) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.3)

    return rows


def crawl(player_type: str, sorts: list[str]) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        year_rows: list[dict] = []
        for field in sorts:
            year_rows.extend(fetch_season(year, player_type, field))
            time.sleep(0.3)

        if not year_rows:
            print(f"  [{year}] 수집 0건")
            continue

        df = pd.DataFrame(year_rows).drop_duplicates(subset=["playerId", "collect_year"])
        print(f"  [{year}] {len(df):>4}명")
        frames.append(df)
        time.sleep(0.4)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    print("=" * 62)
    print("  네이버 KBO 전 선수 재수집 (2010~2026)")
    print("=" * 62)

    print("\n[1/2] 타자")
    hitters = crawl("HITTER", HITTER_SORTS)

    print("\n[2/2] 투수")
    pitchers = crawl("PITCHER", PITCHER_SORTS)

    hitter_out = DATA_DIR / "naver_hitter_2010_2026_raw_v2.csv"
    pitcher_out = DATA_DIR / "naver_pitcher_2010_2026_raw_v2.csv"

    hitters.to_csv(hitter_out, index=False, encoding="utf-8-sig")
    pitchers.to_csv(pitcher_out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {hitter_out}  ({len(hitters)}행, {hitters['playerName'].nunique()}명)")
    print(f"저장: {pitcher_out}  ({len(pitchers)}행, {pitchers['playerName'].nunique()}명)")
    print("\n선수 사진(player_photos.csv)은 scripts/build_season_stats_v3.py에서 만든다.")


if __name__ == "__main__":
    main()
