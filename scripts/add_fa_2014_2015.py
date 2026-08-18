"""FA 2014·2015 계약 35건을 계약 테이블에 붙인다.

왜. 시즌 스탯을 2010년까지 넓혀서(scripts/crawl_naver_v2.py) 직전 3시즌을 채울 수
있는 FA 연도가 2016년에서 2013년으로 내려갔다. 그중 금액을 확인할 수 있는
2014·2015년 두 해를 담는다.

출처: 위키백과 'KBO 리그 FA' 문서 원문(action=raw). 요약본은 쓰지 않았다.
      요약을 거치면 '롯데에서 두산으로 이적'이 '롯데 계약'으로 뭉개지고
      연도까지 어긋났다. 원문은 원소속과 계약 구단을 문장에 그대로 적어 둔다.

담지 않은 것
  2013년(2012시즌 후) FA — 위키백과 원문에 금액이 적힌 건이 김주찬 하나뿐이다.
  다른 출처를 봤지만 홍성흔·김주찬을 실제와 다른 연도에 놓고 있어 쓰지 않았다.
  총액 없이 이름만으로는 정답을 만들 수 없다.

  윤석민 2014 볼티모어, 오승환 2014 한신 — 해외 구단 계약이라 원화 총액이 없다.
  류현진 2013 LA다저스도 같다.

담은 것
  윤석민 2015 KIA (4년 90억) — 볼티모어에서 KBO로 돌아온 계약이다.
  기존 테이블의 이대호 2017·김현수 2018과 같은 기준이다.

'1+1년', '3+1년'은 옵션을 포함한 총 연수로 적는다. 기존 175건과 같은 방식이다.

주의: 타자 WAR는 2017년, 투수 WAR는 2014년부터만 네이버가 준다. 여기 담는
계약은 직전 3시즌이 그보다 이르므로 WAR 없이 학습에 들어간다.

입력: data/fa_contracts_v6.csv, player_master.csv, player_birth_manual.csv
출력: data/fa_contracts_v7.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import resolve_player_id  # noqa: E402

DATA = Path(__file__).parent.parent / "data"

# (이름, fa_year, 포지션, 계약연수, 총액억, 계약구단)
CONTRACTS = [
    ("이종욱", 2014, "OF", 4, 50.0, "NC"),
    ("손시헌", 2014, "SS", 4, 30.0, "NC"),
    ("최준석", 2014, "1B", 4, 35.0, "롯데"),
    ("정근우", 2014, "2B", 4, 70.0, "한화"),
    ("이대형", 2014, "OF", 4, 24.0, "KIA"),
    ("이병규", 2014, "OF", 3, 25.5, "LG"),
    ("권용관", 2014, "IF", 1, 1.0, "LG"),
    ("이용규", 2014, "OF", 4, 67.0, "한화"),
    ("장원삼", 2014, "P", 4, 60.0, "삼성"),
    ("박한이", 2014, "OF", 4, 28.0, "삼성"),
    ("강영식", 2014, "P", 4, 17.0, "롯데"),
    ("강민호", 2014, "C", 4, 75.0, "롯데"),
    ("이대수", 2014, "SS", 4, 20.0, "한화"),
    ("한상훈", 2014, "2B", 3, 13.0, "한화"),
    ("박정진", 2014, "P", 2, 8.0, "한화"),
    ("이성열", 2015, "OF", 2, 5.0, "한화"),
    ("최정", 2015, "3B", 4, 86.0, "SK"),
    ("조동화", 2015, "OF", 4, 22.0, "SK"),
    ("김강민", 2015, "OF", 4, 56.0, "SK"),
    ("나주환", 2015, "SS", 2, 5.5, "SK"),
    ("이재영", 2015, "P", 2, 4.5, "SK"),
    ("박경수", 2015, "2B", 4, 18.2, "KT"),
    ("박용택", 2015, "OF", 4, 50.0, "LG"),
    ("송은범", 2015, "P", 4, 34.0, "한화"),
    ("차일목", 2015, "C", 2, 4.5, "KIA"),
    ("권혁", 2015, "P", 4, 32.0, "한화"),
    ("배영수", 2015, "P", 3, 21.5, "한화"),
    ("조동찬", 2015, "3B", 4, 28.0, "삼성"),
    ("윤성환", 2015, "P", 4, 80.0, "삼성"),
    ("안지만", 2015, "P", 4, 65.0, "삼성"),
    ("장원준", 2015, "P", 4, 84.0, "두산"),
    ("김사율", 2015, "P", 4, 14.5, "KT"),
    ("박기혁", 2015, "SS", 4, 11.4, "KT"),
    ("김경언", 2015, "OF", 3, 8.5, "한화"),
    ("윤석민", 2015, "P", 4, 90.0, "KIA"),
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def main() -> None:
    fa = _read("fa_contracts_v6.csv")
    master = _read("player_master.csv")
    hitters = _read("hitter_season_stats_2010_2026_v4.csv")
    pitchers = _read("pitcher_season_stats_2010_2026_v4.csv")
    birth = _read("player_birth_manual.csv").dropna(subset=["birth_year"])
    birth_by_id = {int(r.player_id): int(r.birth_year) for r in birth.itertuples()}
    # 여러 번 FA를 한 선수는 뒤 계약에 생년이 이미 적혀 있다. 정근우·최준석처럼
    # 2014년에도 2018년에도 계약한 선수가 그렇다.
    known_birth = fa.dropna(subset=["player_id", "birth_year"])
    for row in known_birth.itertuples():
        birth_by_id.setdefault(int(row.player_id), int(row.birth_year))
    type_by_id = dict(zip(master["player_id"], master["player_type"]))

    rows, mismatched = [], []
    for name, fa_year, position, years, total, team in CONTRACTS:
        is_pitcher = position == "P"
        player_id = resolve_player_id(pitchers if is_pitcher else hitters, name, team, fa_year)

        # 내가 적은 포지션과 마스터의 선수 유형이 어긋나면 둘 중 하나가 틀렸다.
        if player_id is not None:
            known = type_by_id.get(player_id)
            if known and (known == "pitcher") != is_pitcher:
                mismatched.append((name, fa_year, position, known))

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
    for column in ("player_id", "birth_year", "age_at_fa"):
        added[column] = added[column].astype("Int64")

    out = pd.concat([added, fa], ignore_index=True).sort_values(
        ["fa_year", "annual_avg_salary"], ascending=[True, False]
    )
    out.to_csv(DATA / "fa_contracts_v7.csv", index=False, encoding="utf-8-sig")

    print(f"추가 {len(added)}건 → 총 {len(out)}건")
    if mismatched:
        print(f"  ! 포지션과 마스터 유형이 어긋남: {mismatched}")
    no_id = added[added["player_id"].isna()]
    if len(no_id):
        print(f"  ! player_id 미확정: {list(no_id['player_name'])}")
    no_birth = added[added["birth_year"].isna()]
    if len(no_birth):
        print(f"  ! 생년 미확인 {len(no_birth)}건: {list(no_birth['player_name'])}")


if __name__ == "__main__":
    main()
