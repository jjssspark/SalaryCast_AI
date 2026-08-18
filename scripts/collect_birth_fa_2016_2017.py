"""FA 2016·2017 계약자 35명의 생년을 채운다.

왜 따로 도는가. collect_player_birth.py는 latest_year >= 2025인 현역만 대상으로
한다. 화면에 뜰 사람만 채우면 됐기 때문이다. 그런데 2016·2017 FA 계약자는 대부분
은퇴해서 그 대상에 들지 않았고, age_at_fa가 비면 학습 표본에서 빠진다.

조회 방식과 동명이인 판정 규칙은 collect_player_birth.py 것을 그대로 쓴다.
결과는 player_birth_manual.csv에 덧붙인다(기존 행은 건드리지 않는다).

입력: data/fa_contracts_v6.csv, player_master.csv
출력: data/player_birth_manual.csv (갱신)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from collect_player_birth import (  # noqa: E402
    QUERY_KBO_TEAMS,
    QUERY_KOREAN,
    resolve,
    run_sparql,
)

DATA = Path(__file__).parent.parent / "data"
NEW_FA_YEARS = (2016, 2017)

# 우리 시즌 데이터가 시작하는 해. first_year가 이 값이면 실제 데뷔 연도가 아니라
# '데이터가 여기서부터라 여기서 처음 보인다'는 뜻이다. collect_player_birth.resolve는
# 데뷔 나이 17~35세로 후보를 거르는데, 2000년대 초 데뷔한 베테랑은 이 검사를
# 통과할 수 없다. 이승엽(1976년생)은 2013 - 1976 = 37세로 걸렸다.
WINDOW_START = 2013

# 위키데이터에 아예 항목이 없는 선수. KBO 공식 기록실에서 확인했다.
# player_id가 KBO 선수 ID와 같아 페이지를 직접 열어 대조했다.
#   https://www.koreabaseball.com/Record/Retire/Hitter.aspx?playerId=<id>
MANUAL_BIRTH = {
    71118: 1983,  # 이동현 1983-01-12
    77248: 1985,  # 오재원 1985-02-09
    74846: 1981,  # 박정권 1981-07-21
    71845: 1982,  # 채병용 1982-04-25
    71857: 1982,  # 박재상 1982-07-20
}


def main() -> None:
    fa = pd.read_csv(DATA / "fa_contracts_v6.csv", encoding="utf-8-sig")
    master = pd.read_csv(DATA / "player_master.csv", encoding="utf-8-sig")
    known = pd.read_csv(DATA / "player_birth_manual.csv", encoding="utf-8-sig")

    target = fa[fa["fa_year"].isin(NEW_FA_YEARS) & fa["birth_year"].isna()]
    target = target.merge(
        master[["player_id", "first_year", "team_latest"]], on="player_id", how="left"
    )
    print(f"대상 {len(target)}명")

    print("위키데이터 조회")
    reference = pd.concat(
        [run_sparql(QUERY_KOREAN, "국내"), run_sparql(QUERY_KBO_TEAMS, "KBO 구단")],
        ignore_index=True,
    ).drop_duplicates(["name", "birth_year"])
    print(f"  합계 {len(reference)}건 / 고유 이름 {reference['name'].nunique()}개")

    resolved, failed = [], []
    for row in target.itertuples():
        player_id = int(row.player_id)
        if player_id in MANUAL_BIRTH:
            resolved.append({
                "player_name": row.player_name, "player_id": player_id,
                "birth_year": MANUAL_BIRTH[player_id], "source": "kbo",
                "confidence": "KBO 기록실 수기 확인", "matched_title": "",
            })
            continue

        candidates = reference[reference["name"] == row.player_name]
        # 계약 구단으로 먼저 맞춰 본다. 은퇴 선수라 team_latest가 마지막 소속팀이고
        # 계약 당시 팀과 다를 수 있어 둘 다 시도한다.
        birth, reason = resolve(candidates, str(row.team), int(row.first_year))
        if not birth:
            birth, reason = resolve(candidates, str(row.team_latest), int(row.first_year))

        # 데뷔 연도를 모르는 베테랑은 나이 검사를 못 쓴다. 후보가 하나뿐일 때만
        # 받아들인다. 여럿이면 가릴 방법이 없으니 그대로 비워 둔다.
        if not birth and int(row.first_year) == WINDOW_START and len(candidates) == 1:
            birth, reason = int(candidates.iloc[0]["birth_year"]), "단일 후보(데뷔연도 불명)"

        record = {
            "player_name": row.player_name, "player_id": player_id,
            "birth_year": birth, "source": "wikidata" if birth else "",
            "confidence": reason, "matched_title": "",
        }
        (resolved if birth else failed).append(record)

    merged = pd.concat([known, pd.DataFrame(resolved)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["player_id"], keep="first")
    merged.to_csv(DATA / "player_birth_manual.csv", index=False, encoding="utf-8-sig")

    print(f"\n확보 {len(resolved)}명 / 미해결 {len(failed)}명 (총 {len(merged)}행)")
    if failed:
        print(pd.DataFrame(failed)[["player_name", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
