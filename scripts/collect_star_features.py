"""KBO 수상 이력 수집 — MVP · 골든글러브 · 국가대표.

실행: .venv/bin/python scripts/collect_star_features.py
출력: data/kbo_awards.csv  (player_name, year, team, award, source)

기존 data/star_features_*.csv는 출처 없이 손으로 타이핑한 상수였다
(scripts/make_star_features.py). 검증해 보니 틀린 값이 있었다.
양의지·손아섭·강백호는 정규시즌 MVP 수상 이력이 없는데 mvp_count=1이고,
박병호는 2012·2013 두 번인데 1로 적혀 있었다. FA 계약자 140명분만 있어서
미래 FA 후보 42명 중 33명이 '수상 경력 없음'으로 처리되고 있기도 했다.

그래서 개별 입력 대신 연도별 수상자 명단 원본에서 만든다.

수집하지 않는 것:
  올스타 — 한국어 위키백과에 2015~2024 중 9개 연도만 있고 2014년 이전이 전무하다.
          있는 연도도 표에 감독·코치가 섞여 있다. 일부 시기만 세면 옛 선수가
          체계적으로 과소 집계되므로, 편향된 값을 넣느니 항목을 뺀다.
  포스트시즌 — 기존 데이터에서 140명 중 대부분이 1이라 사실상 상수였다.

위키데이터도 봤지만 KBO 골든글러브 수상자가 4명뿐이라 못 쓴다.

국가대표는 대회 문서가 아니라 '분류:{연도}년 ... 참가 선수'에서 읽는다.
대회 문서는 구조가 제각각이라 결과가 들쭉날쭉했다. 아시안게임 3개 대회는
문서에 명단이 아예 없어 0~5명만 나왔고, 올림픽·프리미어12는 반대로 감독·코치가
섞여 실제 엔트리보다 많이 잡혔다(2021 올림픽 31명, 실제 엔트리 24명). 분류는
대회마다 같은 이름 규칙을 쓰고 사람 단위로 붙어 있어 결과가 일정하다. 다만
참가국 선수가 전부 들어 있으므로 '분류:대한민국의 야구 선수'가 붙은 사람만 남긴다.
APBC 2017·2023만 참가 선수 분류가 없어 종전대로 문서에서 읽는다.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
API = "https://ko.wikipedia.org/w/api.php"
USER_AGENT = "SalaryCastAI/1.0 (KBO FA salary study; contact hsyoun585@gmail.com)"

# 국가대표 명단을 읽어 올 분류. 우리 시즌 데이터가 2013년부터라 2006년 이후만 본다.
NATIONAL_TEAM_CATEGORIES = [
    ("2026 WBC", "분류:2026년 월드 베이스볼 클래식 참가 선수"),
    ("2024 프리미어12", "분류:2024년 WBSC 프리미어 12 참가 선수"),
    ("2023 WBC", "분류:2023년 월드 베이스볼 클래식 참가 선수"),
    ("2022 아시안게임", "분류:2022년 아시안 게임 야구 참가 선수"),
    ("2021 올림픽", "분류:2020년 하계 올림픽 야구 참가 선수"),
    ("2019 프리미어12", "분류:2019년 WBSC 프리미어 12 참가 선수"),
    ("2018 아시안게임", "분류:2018년 아시안 게임 야구 참가 선수"),
    ("2017 WBC", "분류:2017년 월드 베이스볼 클래식 참가 선수"),
    ("2015 프리미어12", "분류:2015년 WBSC 프리미어 12 참가 선수"),
    ("2014 아시안게임", "분류:2014년 아시안 게임 야구 참가 선수"),
    ("2013 WBC", "분류:2013년 월드 베이스볼 클래식 참가 선수"),
    ("2009 WBC", "분류:2009년 월드 베이스볼 클래식 참가 선수"),
    ("2008 올림픽", "분류:2008년 하계 올림픽 야구 참가 선수"),
    ("2006 WBC", "분류:2006년 월드 베이스볼 클래식 참가 선수"),
]

# 참가 선수 분류가 없는 대회. 종전대로 대회 문서에서 읽는다.
NATIONAL_TEAM_PAGES = [
    ("2023 APBC", "2023년 아시아 프로야구 챔피언십"),
    ("2017 APBC", "2017년 아시아 프로야구 챔피언십"),
]

# 분류에는 참가국 선수가 전부 들어 있다. 한국 선수만 남기는 표식.
KOREAN_PLAYER_CATEGORY = "분류:대한민국의 야구 선수"

# [[문서명|표시이름]] 또는 [[이름]]
LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
YEAR_LINK = re.compile(r"\[\[(\d{4})년[^\]|]*\|(\d{4})\]\]|\[\[(\d{4})년[^\]]*\]\]")

# 선수가 아닌 이름을 걸러내는 말. 감독·코치 표가 같이 들어 있는 문서가 있다.
NOT_PLAYER = ("감독", "코치", "야구인", "위원", "해설")


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def query(**params) -> dict:
    """위키 API 한 번. 읽기 타임아웃이 간헐적으로 나서 두 번 더 시도한다.

    macOS 시스템 파이썬의 urllib은 루트 인증서를 못 찾아 SSL 검증에서 막힌다.
    requests는 certifi를 들고 다녀서 그냥 된다.
    """
    params.update({"action": "query", "format": "json"})
    for attempt in range(3):
        try:
            response = SESSION.get(API, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(3)
    raise AssertionError("unreachable")


def fetch_wikitext(title: str) -> str | None:
    """문서 원문(위키텍스트). 없으면 None."""
    result = query(
        prop="revisions", rvprop="content", rvslots="main",
        titles=title, redirects="1",
    )
    page = next(iter(result["query"]["pages"].values()))
    if "missing" in page:
        return None
    return page["revisions"][0]["slots"]["main"]["*"]


def category_members(category: str) -> list[str]:
    """분류에 속한 일반 문서 제목. 500건이 넘으면 이어 받는다."""
    titles: list[str] = []
    cont: dict = {}
    while True:
        result = query(
            list="categorymembers", cmtitle=category,
            cmlimit="500", cmnamespace=0, **cont,
        )
        titles += [page["title"] for page in result["query"]["categorymembers"]]
        if "continue" not in result:
            return titles
        cont = result["continue"]


def categories_of(titles: list[str]) -> dict[str, list[str]]:
    """문서 제목 -> 그 문서에 붙은 분류.

    cllimit 상한이 문서별이 아니라 응답 전체에 걸리므로, 한 번에 많이 물으면
    뒤쪽 문서의 분류가 잘려서 온다. 10개씩 끊고 continue도 따라간다.
    """
    found: dict[str, list[str]] = {title: [] for title in titles}
    for start in range(0, len(titles), 10):
        chunk = titles[start: start + 10]
        cont: dict = {}
        while True:
            result = query(prop="categories", cllimit="max", titles="|".join(chunk), **cont)
            for page in result["query"]["pages"].values():
                found.setdefault(page["title"], [])
                found[page["title"]] += [c["title"] for c in page.get("categories", [])]
            if "continue" not in result:
                break
            cont = result["continue"]
        time.sleep(0.3)
    return found


def roster_from_category(label: str, category: str) -> tuple[list[dict], str]:
    """분류에서 한국 선수만 골라 국가대표 기록으로 만든다."""
    members = category_members(category)
    if not members:
        return [], f"분류 없음: {category}"

    marks = categories_of(members)
    names = {
        # '김광현 (야구 선수)' -> '김광현'
        re.sub(r"\s*\([^)]*\)\s*", "", title).strip()
        for title in members
        if KOREAN_PLAYER_CATEGORY in marks.get(title, [])
    }
    rows = [
        {"player_name": name, "year": int(label[:4]), "team": "",
         "award": "NT", "source": label}
        for name in sorted(names)
    ]
    return rows, f"{category} ({len(members)}명 중 한국 {len(names)}명)"


def clean_name(raw: str) -> str:
    """'[[박건우 (1990년)|박건우]]' -> '박건우'."""
    name = raw.strip()

    link = LINK.search(name)
    if link:
        name = link.group(2) or link.group(1)

    name = re.sub(r"<[^>]+>", "", name)
    name = re.sub(r"'''?", "", name)
    name = re.sub(r"[\[\]]", "", name)
    name = re.sub(r"\s*\([^)]*\)\s*", "", name)
    return name.strip()


def strip_refs(text: str) -> str:
    return re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text, flags=re.S)


def parse_cell(cell: str) -> tuple[str, str] | None:
    """'[[홍창기]]<br/>(LG)' -> ('홍창기', 'LG')"""
    cell = strip_refs(cell).strip()
    if not cell:
        return None

    team = ""
    bracket = re.search(r"\(([^)]+)\)\s*$", cell)
    if bracket:
        team = clean_name(bracket.group(1))
        cell = cell[: bracket.start()]

    match = LINK.search(cell)
    name = clean_name(match.group(2) or match.group(1)) if match else clean_name(cell)
    if not name or len(name) > 12:
        return None
    return name, team


def row_year(line: str) -> int | None:
    """표 한 행의 첫 칸에서 연도를 읽는다.

    2006년부터는 시상식 문서로 링크가 걸려 있고([[2006년 ...|2006]]),
    2005년 이전은 그냥 |2005|| 형태다. 둘 다 받아야 1982년부터 잡힌다.
    """
    first = line.split("||")[0].lstrip("|").strip()

    linked = YEAR_LINK.search(first)
    if linked:
        value = linked.group(2) or linked.group(1) or linked.group(3)
    else:
        plain = re.fullmatch(r"(\d{4})", first)
        if not plain:
            return None
        value = plain.group(1)

    year = int(value)
    return year if 1982 <= year <= 2030 else None


def parse_golden_glove(text: str) -> list[dict]:
    """골든글러브 문서의 연도별 표. 한 행이 한 해, 칸마다 포지션별 수상자."""
    rows = []
    for line in text.split("\n"):
        if not line.lstrip().startswith("|") or line.lstrip().startswith("|-"):
            continue

        year = row_year(line)
        if year is None:
            continue

        for cell in line.split("||")[1:]:  # 첫 칸은 연도라 버린다
            parsed = parse_cell(cell)
            if parsed:
                rows.append({
                    "player_name": parsed[0], "year": year, "team": parsed[1],
                    "award": "GG", "source": "KBO 골든글러브",
                })
    return rows


ROWSPAN = re.compile(r'rowspan\s*=\s*"?(\d+)"?')

# 이름 칸이 rowspan으로 묶인 행에서는 첫 링크가 소속팀이라 팀명이 수상자로 잡힌다.
# (1999·2001~2003년이 전부 '삼성 라이온즈'로 들어왔다. 실제 수상자는 이승엽이다.)
TEAM_WORDS = (
    "타이거즈", "라이온즈", "베어스", "트윈스", "위즈", "랜더스", "자이언츠",
    "이글스", "다이노스", "히어로즈", "유니콘스", "드래곤즈", "청룡", "스타즈",
    "돌핀즈", "레이더스", "와이번스", "슈퍼스타즈", "빙그레", "쌍방울",
)


def parse_mvp(text: str) -> list[dict]:
    """MVP 문서는 연도가 헤더(!)로, 선수가 다음 줄(|)로 온다.

    2년 연속 수상자는 이름 칸을 rowspan으로 묶어 둔다. 박병호 2012~2013,
    이승엽 2002~2003이 그렇다. 뒤 연도 행에는 이름이 아예 없어서
    앞 행의 수상자를 물려받아야 한다.
    """
    rows = []
    lines = [line.strip() for line in strip_refs(text).split("\n")]
    carry_name, carry_left = None, 0

    for index, line in enumerate(lines):
        if not line.startswith("!"):
            continue
        head = YEAR_LINK.search(line)
        if not head:
            continue
        year = int(head.group(2) or head.group(1) or head.group(3))
        if not 1982 <= year <= 2030:
            continue

        found = None
        for follow in lines[index + 1: index + 6]:
            if follow.startswith("!") or follow.startswith("|-"):
                break
            if not follow.startswith("|") or 'align="left"' in follow:
                continue  # align="left"는 성적 설명 칸이다
            link = LINK.search(follow)
            if not link:
                continue
            if any(word in link.group(1) for word in TEAM_WORDS):
                continue
            name = clean_name(link.group(2) or link.group(1))
            if not name or len(name) > 12 or any(word in name for word in TEAM_WORDS):
                continue
            span = ROWSPAN.search(follow)
            found = (name, int(span.group(1)) if span else 1)
            break

        if found:
            name, span = found
            carry_name, carry_left = (name, span - 1) if span > 1 else (None, 0)
        elif carry_left > 0:
            name = carry_name
            carry_left -= 1
        else:
            continue

        rows.append({
            "player_name": name, "year": year, "team": "",
            "award": "MVP", "source": "KBO MVP",
        })
    return rows


def korea_section(text: str) -> str:
    """대회 문서에서 대한민국 대표팀 구획만 잘라낸다.

    WBC 문서는 참가국 16개 로스터가 한 문서에 다 들어 있다. 통째로 훑으면
    도미니카·일본 선수까지 한국 국가대표로 잡힌다(2023 WBC에서 99명이 나왔다).
    '=== ... 대한민국 ... ===' 헤더부터 다음 같은 급 헤더 직전까지만 본다.
    """
    headers = list(re.finditer(r"^(={2,4})\s*(.+?)\s*\1\s*$", text, re.M))
    for index, header in enumerate(headers):
        # 2006년 문서는 국가명 대신 IOC 코드를 쓴다 ({{국가대표팀|야구|KOR}}).
        if "대한민국" not in header.group(2) and "KOR" not in header.group(2):
            continue
        start = header.end()
        level = len(header.group(1))
        for follow in headers[index + 1:]:
            if len(follow.group(1)) <= level:
                return text[start:follow.start()]
        return text[start:]

    # 대한민국 전용 문서(틀:... 대한민국 선수 명단)는 나눌 구획이 없다.
    return text


def parse_roster(text: str, label: str) -> list[dict]:
    """국가대표 명단.

    '{{야구 국가대표팀 선수명단|이름=[[고영표]]|...}}' 틀이 있으면 그걸 쓴다.
    틀이 없는 옛 문서만 링크를 훑되, 감독·코치 줄은 뺀다.
    """
    year = int(label[:4])
    section = strip_refs(korea_section(text))

    def people(values) -> set[str]:
        return {
            name for name in (clean_name(value) for value in values)
            if re.fullmatch(r"[가-힣]{2,4}", name)
        }

    # 이름=[[정우영 (야구 선수)|정우영]] 처럼 값 안에 |가 또 있어서
    # 링크 통째로를 먼저 집고, 링크가 아닐 때만 다음 |까지 자른다.
    names = people(re.findall(r"\|\s*이름\s*=\s*(\[\[[^\]]+\]\]|[^|}\n]+)", section))

    # 둘러보기 상자 문서는 '| 이름 = 2019년 프리미어 12 대한민국 선수 명단'처럼
    # 문서 제목이 이름= 로 들어 있다. 사람 이름이 안 나오면 링크로 간다.
    if not names:
        links = []
        for line in section.split("\n"):
            if "감독" in line or "코치" in line:
                continue
            links += [
                match.group(2) or match.group(1)
                for match in LINK.finditer(line)
                if not any(word in match.group(1) for word in NOT_PLAYER)
            ]
        names = people(links)

    return [
        {"player_name": name, "year": year, "team": "", "award": "NT", "source": label}
        for name in sorted(names)
    ]


# 명단이 다른 문서에 있을 때의 두 가지 표기.
#   {{본문|틀:2019년 프리미어 12 대한민국 선수 명단}}   — 프리미어12·아시안게임
#   {{2013년 월드 베이스볼 클래식 대한민국 선수 명단}}   — 2013 WBC
ROSTER_POINTER = re.compile(
    r"\{\{\s*본문\s*\|\s*([^}|]+?)\s*\}\}|\{\{\s*(\d{4}년[^}|]*선수 명단)\s*\}\}"
)


def resolve_roster(label: str, title: str) -> tuple[list[dict], str]:
    """대회 문서에서 명단을 찾는다.

    대회마다 문서 구조가 다르다. WBC·올림픽은 본문에 명단이 직접 있고,
    프리미어12·아시안게임은 '{{본문|틀:2019년 프리미어 12 대한민국 선수 명단}}'처럼
    다른 문서로 넘긴다. 포인터가 있으면 한 번 따라간다.
    """
    text = fetch_wikitext(title)
    if text is None:
        return [], f"문서 없음: {title}"

    pointer = ROSTER_POINTER.search(korea_section(text))
    if pointer:
        target = (pointer.group(1) or pointer.group(2)).strip()
        linked = fetch_wikitext(target) or fetch_wikitext(f"틀:{target}")
        time.sleep(1)
        if linked is not None:
            found = parse_roster(linked, label)
            if found:
                return found, target

    return parse_roster(text, label), title


def main() -> None:
    print("=" * 62)
    print("  KBO 수상 이력 수집")
    print("=" * 62)

    rows: list[dict] = []

    for title, parser, label in [
        ("KBO 골든글러브", parse_golden_glove, "골든글러브"),
        ("KBO MVP", parse_mvp, "MVP"),
    ]:
        text = fetch_wikitext(title)
        if text is None:
            print(f"  [실패] 문서 없음: {title}")
            continue
        found = parser(text)
        rows.extend(found)
        years = sorted({row["year"] for row in found})
        print(f"  {label:8s} {len(found):4d}건  {years[0]}~{years[-1]}")
        time.sleep(1)

    print("\n  국가대표 — 분류")
    for label, category in NATIONAL_TEAM_CATEGORIES:
        found, where = roster_from_category(label, category)
        rows.extend(found)
        mark = "" if 20 <= len(found) <= 32 else "  <- 확인 필요"
        print(f"    {label:16s} {len(found):3d}명  ({where}){mark}")
        time.sleep(0.5)

    print("\n  국가대표 — 문서 (참가 선수 분류가 없는 대회)")
    for label, title in NATIONAL_TEAM_PAGES:
        found, where = resolve_roster(label, title)
        rows.extend(found)
        mark = "" if 20 <= len(found) <= 32 else "  <- 확인 필요"
        print(f"    {label:16s} {len(found):3d}명  ({where}){mark}")
        time.sleep(1)

    table = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["player_name", "year", "award"])
        .sort_values(["award", "year", "player_name"])
    )

    out = DATA_DIR / "kbo_awards.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}  ({len(table)}건 / 선수 {table['player_name'].nunique()}명)")
    print(table.groupby("award").size().to_string())


if __name__ == "__main__":
    main()
