"""FA 2016·2017 계약 35건을 계약 테이블에 붙인다.

왜 필요한가. 계약 데이터가 2018년부터라 학습 표본이 140건뿐이었다. 앞선 두 해를
채우면 175건이 된다. 시즌 스탯이 2013년부터라 2016년 FA(직전 3시즌 = 2013~2015)가
넣을 수 있는 가장 이른 연도다.

출처: 위키백과 'KBO 리그 FA' 연도별 계약자 표
      (https://ko.wikipedia.org/wiki/KBO_리그_FA)
      정우람·오재원 계약 조건은 경향신문 2015-11-30 기사로 교차 확인했다.

담지 않은 것
  김현수 2016 볼티모어 (2년 700만달러), 황재균 2017 샌프란시스코 (1년 310만달러)
  — 해외 구단 계약이라 원화 총액이 없다. 타깃이 억 원이므로 넣을 수 없다.

담은 것
  이대호 2017 롯데 (4년 150억) — 해외에서 KBO로 돌아온 FA다. 기존 테이블도
  김현수 2018 LG(115억), 황재균 2018 KT(88억)를 같은 기준으로 담고 있다.
  직전 3시즌이 일본·미국이라 학습 표본에서는 자동으로 빠진다.

'2+1년' 같은 옵션 계약은 옵션을 포함한 총 연수로 적는다. 연평균은 총액 ÷ 연수다.
기존 140건과 같은 방식이다.

입력: data/fa_contracts_v5.csv, player_master.csv, player_birth_manual.csv
출력: data/fa_contracts_v6.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import resolve_player_id  # noqa: E402

DATA = Path(__file__).parent.parent / "data"

# (이름, fa_year, 포지션, 계약연수, 총액억, 계약구단)
CONTRACTS = [
    ("오재원", 2016, "2B", 4, 38.0, "두산"),
    ("고영민", 2016, "2B", 2, 5.0, "두산"),
    ("유한준", 2016, "OF", 4, 60.0, "KT"),
    ("손승락", 2016, "P", 4, 60.0, "롯데"),
    ("마정길", 2016, "P", 2, 6.2, "넥센"),
    ("이택근", 2016, "OF", 4, 35.0, "넥센"),
    ("정상호", 2016, "C", 4, 32.0, "LG"),
    ("윤길현", 2016, "P", 4, 38.0, "롯데"),
    ("정우람", 2016, "P", 4, 84.0, "한화"),
    ("박정권", 2016, "1B", 4, 30.0, "SK"),
    ("채병용", 2016, "P", 3, 10.5, "SK"),
    ("박재상", 2016, "OF", 2, 5.5, "SK"),
    ("이동현", 2016, "P", 3, 30.0, "LG"),
    ("김상현", 2016, "OF", 4, 17.0, "KT"),
    ("이범호", 2016, "3B", 4, 36.0, "KIA"),
    ("박석민", 2016, "3B", 4, 96.0, "NC"),
    ("이승엽", 2016, "1B", 2, 36.0, "삼성"),
    ("심수창", 2016, "P", 4, 13.0, "한화"),
    ("송승준", 2016, "P", 4, 40.0, "롯데"),
    ("김태균", 2016, "1B", 4, 84.0, "한화"),
    ("조인성", 2016, "C", 2, 10.0, "한화"),
    ("이원석", 2017, "3B", 4, 27.0, "삼성"),
    ("이현승", 2017, "P", 3, 27.0, "두산"),
    ("김재호", 2017, "SS", 4, 50.0, "두산"),
    ("김광현", 2017, "P", 4, 85.0, "SK"),
    ("우규민", 2017, "P", 4, 65.0, "삼성"),
    ("봉중근", 2017, "P", 2, 15.0, "LG"),
    ("정성훈", 2017, "3B", 1, 7.0, "LG"),
    ("조영훈", 2017, "1B", 2, 4.5, "NC"),
    ("이진영", 2017, "OF", 2, 15.0, "KT"),
    ("양현종", 2017, "P", 1, 22.5, "KIA"),
    ("나지완", 2017, "OF", 4, 40.0, "KIA"),
    ("최형우", 2017, "OF", 4, 100.0, "KIA"),
    ("차우찬", 2017, "P", 4, 95.0, "LG"),
    ("이대호", 2017, "1B", 4, 150.0, "롯데"),
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def main() -> None:
    fa = _read("fa_contracts_v5.csv")
    master = _read("player_master.csv")
    hitters = _read("hitter_season_stats_2013_2026_v3.csv")
    pitchers = _read("pitcher_season_stats_2013_2026_v3.csv")
    birth = _read("player_birth_manual.csv").dropna(subset=["birth_year"])
    birth_by_id = {int(r.player_id): int(r.birth_year) for r in birth.itertuples()}

    rows = []
    for name, fa_year, position, years, total, team in CONTRACTS:
        seasons = pitchers if position == "P" else hitters
        player_id = resolve_player_id(seasons, name, team, fa_year)
        birth_year = birth_by_id.get(player_id) if player_id else None
        rows.append({
            "player_name": name,
            "player_id": player_id,
            "fa_year": fa_year,
            "age_at_fa": fa_year - birth_year if birth_year else None,
            "position": position,
            "contract_years": years,
            "total_contract_amount": total,
            "annual_avg_salary": round(total / years, 2),
            "team": team,
            "birth_year": birth_year,
            "birth_year_source": "위키데이터" if birth_year else "",
        })

    added = pd.DataFrame(rows)
    added["player_id"] = added["player_id"].astype("Int64")
    added["birth_year"] = added["birth_year"].astype("Int64")
    added["age_at_fa"] = added["age_at_fa"].astype("Int64")

    out = pd.concat([added, fa], ignore_index=True).sort_values(
        ["fa_year", "annual_avg_salary"], ascending=[True, False]
    )
    out.to_csv(DATA / "fa_contracts_v6.csv", index=False, encoding="utf-8-sig")

    no_id = added[added["player_id"].isna()]
    no_birth = added[added["birth_year"].isna()]
    print(f"추가 {len(added)}건 → 총 {len(out)}건")
    if len(no_id):
        print(f"  ! player_id 미확정 {len(no_id)}건: {list(no_id['player_name'])}")
    if len(no_birth):
        print(f"  ! 생년 미확인 {len(no_birth)}건: {list(no_birth['player_name'])}")
    print(added[["player_name", "player_id", "fa_year", "age_at_fa",
                 "annual_avg_salary", "team"]].to_string(index=False))


if __name__ == "__main__":
    main()
