"""
네이버 스포츠 KBO API — 2013~2016 타자/투수 스탯 크롤링
기존 naver_hitter_2015_2026_raw_all.csv와 동일 포맷으로 저장 후 병합
"""

import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
BASE_URL = "https://api-gw.sports.naver.com/statistics/categories/kbo/seasons/{year}/players"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://m.sports.naver.com/kbaseball/record/kbo",
}

HITTER_SORTS = [
    "hitterOps", "hitterHr", "hitterRbi", "hitterHra",
    "hitterBb", "hitterWpa", "hitterSb", "hitterHit",
    "hitterGameCount", "hitterSlg", "hitterObp", "hitterWoba",
]

PITCHER_SORTS = [
    "pitcherEra", "pitcherSave", "pitcherHold", "pitcherWin",
    "pitcherStrikeout", "pitcherGameCount", "pitcherWhip",
]

TARGET_YEARS = list(range(2013, 2017))  # 2013~2016


def fetch_players(year: int, player_type: str, sort_field: str, page_size: int = 100) -> list:
    records = []
    page = 1
    while True:
        params = {
            "playerType": player_type,
            "sortField": sort_field,
            "sortDirection": "DESC",
            "gameType": "REGULAR_SEASON",
            "page": page,
            "pageSize": page_size,
        }
        try:
            resp = requests.get(
                BASE_URL.format(year=year), headers=HEADERS, params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [오류] {year} {player_type} {sort_field} page={page}: {e}")
            break

        result = data.get("result", {})
        players = result.get("seasonPlayerStats", result.get("players", []))
        if not players:
            break
        for p in players:
            row = p.copy()
            row["collect_year"] = year
            row["sort_source"] = sort_field
            records.append(row)

        total_cnt = result.get("totalCount", 0)
        fetched = page * page_size
        if not total_cnt or fetched >= total_cnt:
            break
        page += 1
        time.sleep(0.3)
    return records


def crawl_and_save(player_type: str, sort_list: list, out_file: Path):
    all_rows = []
    for year in TARGET_YEARS:
        print(f"\n  [{year}] {player_type} 수집 중...")
        year_rows = []
        for sort in sort_list:
            rows = fetch_players(year, player_type, sort)
            year_rows.extend(rows)
            print(f"    {sort}: {len(rows)}건")
            time.sleep(0.5)

        df_year = pd.DataFrame(year_rows)
        if df_year.empty:
            print(f"    [경고] {year} 데이터 없음")
            continue

        df_year = df_year.drop_duplicates(subset=["playerId", "collect_year"])
        print(f"    → {year}: {len(df_year)}명 (중복 제거 후)")
        all_rows.append(df_year)
        time.sleep(1.0)

    if not all_rows:
        print(f"[{player_type}] 수집된 데이터 없음")
        return

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_file}  ({len(result)}행)")


def merge_with_existing(new_file: Path, existing_file: Path, out_file: Path):
    if not existing_file.exists():
        print(f"기존 파일 없음: {existing_file}")
        return

    new_df = pd.read_csv(new_file, encoding="utf-8-sig")
    old_df = pd.read_csv(existing_file, encoding="utf-8-sig")

    combined = pd.concat([new_df, old_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["playerId", "collect_year", "sort_source"])
    combined = combined.sort_values(["collect_year", "sort_source"]).reset_index(drop=True)
    combined.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"병합 완료: {out_file}  총 {len(combined)}행")


if __name__ == "__main__":
    print("=" * 60)
    print("  네이버 KBO 스탯 크롤링 (2013~2016 확장)")
    print("=" * 60)

    # 타자
    hitter_new = DATA_DIR / "naver_hitter_2013_2016_raw.csv"
    print("\n[1/2] 타자 수집")
    crawl_and_save("HITTER", HITTER_SORTS, hitter_new)

    if hitter_new.exists():
        merge_with_existing(
            hitter_new,
            DATA_DIR / "naver_hitter_2015_2026_raw_all.csv",
            DATA_DIR / "naver_hitter_2013_2026_raw_all.csv",
        )

    # 투수
    pitcher_new = DATA_DIR / "naver_pitcher_2013_2016_raw.csv"
    print("\n[2/2] 투수 수집")
    crawl_and_save("PITCHER", PITCHER_SORTS, pitcher_new)

    if pitcher_new.exists():
        merge_with_existing(
            pitcher_new,
            DATA_DIR / "naver_pitcher_2015_2026_raw_all.csv",
            DATA_DIR / "naver_pitcher_2013_2026_raw_all.csv",
        )

    print("\n완료.")
