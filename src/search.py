"""StoveLens AI — 선수 검색.

호출 위치: src/ui/pages.py (검색창)
데이터 파일을 직접 읽지 않는다. data/player_master.csv를 읽은 DataFrame을 주입받는다.

기존 화면은 미리 정해둔 목록에서 고르는 방식이라 검색 가능한 선수가 수십 명뿐이었다.
이제 시즌 기록이 있는 2,200여 명을 전부 찾을 수 있다.

이름은 player_id로 구분한다. KBO에는 같은 이름이 96쌍 있어서
'김현수'로 검색하면 2015~2026을 매년 뛴 선수와 1경기 뛴 선수가 함께 나온다.
둘을 합치면 안 되고, 소속팀과 활동 기간을 붙여 구분해서 보여준다.
"""

from __future__ import annotations

import pandas as pd

CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
CHOSUNG_SET = set(CHOSUNG)

MAX_RESULTS = 30

# 검색 결과 정렬 가중치. 낮을수록 위로 온다.
EXACT, PREFIX, CONTAINS, INITIALS = 0, 1, 2, 3


def to_chosung(text: str) -> str:
    """'홍창기' -> 'ㅎㅊㄱ'"""
    out = []
    for char in str(text):
        code = ord(char) - 0xAC00
        out.append(CHOSUNG[code // 588] if 0 <= code <= 11171 else char)
    return "".join(out)


def is_chosung_query(query: str) -> bool:
    """'ㅎㅊㄱ'처럼 초성만 입력했는지."""
    stripped = query.replace(" ", "")
    return bool(stripped) and all(char in CHOSUNG_SET for char in stripped)


def _match_rank(name: str, chosung: str, query: str, initials_mode: bool) -> int | None:
    if initials_mode:
        return INITIALS if query in chosung else None
    if name == query:
        return EXACT
    if name.startswith(query):
        return PREFIX
    if query in name:
        return CONTAINS
    return None


def search_players(
    master: pd.DataFrame,
    query: str,
    team: str | None = None,
    player_type: str | None = None,
    limit: int = MAX_RESULTS,
) -> pd.DataFrame:
    """이름·초성으로 선수를 찾는다.

    정확히 일치 -> 앞부분 일치 -> 부분 일치 -> 초성 일치 순으로 세우고,
    같은 등급 안에서는 시즌 수가 많고 최근까지 뛴 선수를 위로 올린다.
    한 경기 대타로 나온 동명이인이 주전보다 위에 오면 안 된다.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return master.iloc[0:0]

    pool = master
    if team:
        pool = pool[pool["team_latest"] == team]
    if player_type:
        pool = pool[pool["player_type"] == player_type]
    if pool.empty:
        return pool

    initials_mode = is_chosung_query(cleaned)
    ranks = [
        _match_rank(str(row.player_name), str(row.chosung), cleaned, initials_mode)
        for row in pool.itertuples()
    ]

    found = pool.assign(_rank=ranks)
    found = found[found["_rank"].notna()]
    if found.empty:
        return found.drop(columns=["_rank"])

    found = found.sort_values(
        ["_rank", "season_count", "latest_year"], ascending=[True, False, False]
    )
    return found.drop(columns=["_rank"]).head(limit)


def has_namesake(master: pd.DataFrame, player_name: str) -> bool:
    """같은 이름이 둘 이상인지. 화면에서 구분 정보를 붙일지 결정한다."""
    return int((master["player_name"] == player_name).sum()) > 1


def describe(row: pd.Series, master: pd.DataFrame | None = None) -> str:
    """검색 결과 한 줄 설명. 동명이인일 때만 활동 기간을 덧붙인다."""
    parts = [str(row.get("team_latest") or "무소속")]

    position = row.get("position")
    if position:
        parts.append(str(position))

    if master is not None and has_namesake(master, row["player_name"]):
        parts.append(f"{int(row['first_year'])}~{int(row['latest_year'])}")

    return " · ".join(parts)
