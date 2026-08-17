"""
선수 생년 수집 (위키데이터)

네이버 스포츠 API는 생년월일을 주지 않고 선수 상세 엔드포인트는 403이다.
첫 시즌으로 추정하는 방법은 평균 오차가 6년이라 폐기했다.

처음에는 한국어 위키백과 API로 선수마다 검색했는데 5명째에 429가 떴다.
제목을 추정해 50개씩 묶어도 마찬가지였다. 위키데이터 SPARQL은 한 번의 질의로
수천 명을 주고 소속팀까지 붙어 나와서 동명이인 구분에도 쓸 수 있다.

동명이인이 96쌍이라 이름만으로 붙이지 않는다.
  - 같은 이름이 하나뿐이면 그대로 채운다
  - 여럿이면 소속팀이 겹치는 후보만 쓴다
  - 그래도 못 좁히면 비워 둔다. 틀린 나이는 결측보다 나쁘다
데뷔 나이(첫 시즌 - 생년)가 17~35세를 벗어나면 다른 사람으로 보고 버린다.

출력:
  data/player_birth_manual.csv
  output/reports/birth_unresolved.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
REPORT_DIR = Path("output/reports")

SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "StoveLensAI/1.0 (KBO FA salary research)",
    "Accept": "application/sparql-results+json",
}

BASEBALL_PLAYER = "wd:Q10871364"
SOUTH_KOREA = "wd:Q884"

MIN_DEBUT_AGE, MAX_DEBUT_AGE = 17, 35

# 위키데이터 팀 표기를 우리 데이터의 축약 표기로 되돌린다.
TEAM_ALIASES = {
    "KIA": ["KIA 타이거즈", "기아 타이거즈", "해태 타이거즈", "Kia Tigers"],
    "LG": ["LG 트윈스", "LG Twins"],
    "두산": ["두산 베어스", "OB 베어스", "Doosan Bears"],
    "삼성": ["삼성 라이온즈", "Samsung Lions"],
    "롯데": ["롯데 자이언츠", "Lotte Giants"],
    "SSG": ["SSG 랜더스", "SK 와이번스", "SSG Landers", "SK Wyverns"],
    "SK": ["SK 와이번스", "SSG 랜더스"],
    "NC": ["NC 다이노스", "NC Dinos"],
    "KT": ["kt wiz", "KT 위즈", "KT Wiz"],
    "키움": ["키움 히어로즈", "넥센 히어로즈", "우리 히어로즈", "Kiwoom Heroes"],
    "넥센": ["넥센 히어로즈", "키움 히어로즈"],
    "한화": ["한화 이글스", "빙그레 이글스", "Hanwha Eagles"],
}

QUERY_KOREAN = f"""
SELECT ?name ?dob (GROUP_CONCAT(DISTINCT ?teamLabel; separator="|") AS ?teams) WHERE {{
  ?p wdt:P106 {BASEBALL_PLAYER} ;
     wdt:P27 {SOUTH_KOREA} ;
     wdt:P569 ?dob .
  OPTIONAL {{ ?p wdt:P54 ?team . ?team rdfs:label ?teamLabel . FILTER(LANG(?teamLabel) = "ko") }}
  ?p rdfs:label ?name . FILTER(LANG(?name) = "ko")
}}
GROUP BY ?name ?dob
"""

# 외국인 선수는 국적 조건에 안 걸린다. KBO 구단 소속 이력으로 따로 받는다.
QUERY_KBO_TEAMS = f"""
SELECT ?name ?dob (GROUP_CONCAT(DISTINCT ?teamLabel; separator="|") AS ?teams) WHERE {{
  ?team wdt:P118 ?league .
  ?league rdfs:label "KBO 리그"@ko .
  ?p wdt:P54 ?team ;
     wdt:P106 {BASEBALL_PLAYER} ;
     wdt:P569 ?dob .
  ?team rdfs:label ?teamLabel . FILTER(LANG(?teamLabel) = "ko")
  ?p rdfs:label ?name . FILTER(LANG(?name) = "ko")
}}
GROUP BY ?name ?dob
"""


def run_sparql(query: str, label: str) -> pd.DataFrame:
    try:
        response = requests.get(
            SPARQL, params={"query": query, "format": "json"},
            headers=HEADERS, timeout=180,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"  [{label}] 실패: {exc}")
        return pd.DataFrame(columns=["name", "birth_year", "teams"])

    rows = [
        {
            "name": item["name"]["value"],
            "birth_year": int(item["dob"]["value"][:4]),
            "teams": item.get("teams", {}).get("value", ""),
        }
        for item in response.json()["results"]["bindings"]
    ]
    print(f"  [{label}] {len(rows)}건")
    return pd.DataFrame(rows)


def team_matches(our_team: str, wikidata_teams: str) -> bool:
    if not wikidata_teams:
        return False
    return any(alias in wikidata_teams for alias in TEAM_ALIASES.get(our_team, []))


def resolve(
    candidates: pd.DataFrame, team: str, first_year: int
) -> tuple[int | None, str]:
    """(생년, 판정근거). 후보를 못 좁히면 None."""
    plausible = candidates[
        candidates["birth_year"].between(first_year - MAX_DEBUT_AGE, first_year - MIN_DEBUT_AGE)
    ]
    if plausible.empty:
        return None, "데뷔나이 불일치" if len(candidates) else "위키데이터 없음"

    if len(plausible) == 1:
        return int(plausible.iloc[0]["birth_year"]), "단일 후보"

    by_team = plausible[plausible["teams"].map(lambda t: team_matches(team, t))]
    if len(by_team) == 1:
        return int(by_team.iloc[0]["birth_year"]), "소속팀 일치"

    return None, f"후보 {len(plausible)}명 — 구분 불가"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("위키데이터 조회")
    reference = pd.concat(
        [run_sparql(QUERY_KOREAN, "국내"), run_sparql(QUERY_KBO_TEAMS, "KBO 구단")],
        ignore_index=True,
    ).drop_duplicates(["name", "birth_year"])
    print(f"  합계 {len(reference)}건 / 고유 이름 {reference['name'].nunique()}개")

    need = pd.read_csv(REPORT_DIR / "missing_birth_year.csv")
    master = pd.read_csv(DATA_DIR / "player_master.csv").set_index("player_id")
    need["first_year"] = need["player_id"].map(master["first_year"])

    resolved, unresolved = [], []
    for row in need.itertuples():
        candidates = reference[reference["name"] == row.player_name]
        birth, reason = resolve(candidates, str(row.team_latest), int(row.first_year))

        record = {
            "player_name": row.player_name,
            "player_id": row.player_id,
            "birth_year": birth,
            "source": "wikidata" if birth else "",
            "confidence": reason,
            "matched_title": "",
        }
        (resolved if birth else unresolved).append(record)

    out = DATA_DIR / "player_birth_manual.csv"
    pd.DataFrame(resolved).to_csv(out, index=False, encoding="utf-8-sig")
    pd.DataFrame(unresolved).to_csv(
        REPORT_DIR / "birth_unresolved.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n대상 {len(need)}명 -> 확보 {len(resolved)}명 / 미해결 {len(unresolved)}명")
    if unresolved:
        print("\n미해결 사유")
        print(pd.DataFrame(unresolved)["confidence"].value_counts().to_string())

    priority_one = need[need["priority"] == 1]["player_name"]
    got = set(pd.DataFrame(resolved)["player_name"]) if resolved else set()
    print(f"\n우선순위 1(미래 FA 후보) {len(priority_one)}명 중 "
          f"{sum(name in got for name in priority_one)}명 확보")


if __name__ == "__main__":
    main()
