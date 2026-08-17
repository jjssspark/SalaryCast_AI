"""StoveLens AI — 홈 화면.

호출 위치: src/ui/pages.py route()
데이터 파일을 읽지 않는다. app/app.py가 만든 Context를 받는다.

이전 화면은 미리 정해둔 목록에서 선수를 고르는 방식이라 검색 가능한 선수가
수십 명뿐이었다. 이제 시즌 기록이 있는 1,800여 명을 이름·초성으로 찾는다.
"""

from __future__ import annotations

import streamlit as st

from src.search import describe, search_players
from src.serving import (
    Context,
    counters,
    headline_candidates,
    overall_median,
    resolve_by_name,
    ticker_rows,
)
from src.ui import components as ui
from src.ui.assets import FIELD_SVG, FLYING_BALL

SEARCH_LIMIT = 12


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _cached(context: Context, key: str, produce):
    """홈 타일은 후보 42명을 전부 예측해야 나온다. 검색어를 칠 때마다
    다시 계산하면 입력이 눈에 띄게 느려져서 세션에 담아둔다."""
    if key not in st.session_state:
        st.session_state[key] = produce(context)
    return st.session_state[key]


def render(context: Context) -> None:
    _html(ui.appbar())
    _html(ui.hero(FIELD_SVG, FLYING_BALL))

    query = st.text_input(
        "선수 검색",
        placeholder="선수 이름 검색",
        label_visibility="collapsed",
        key="home_query",
    )
    _html('<div class="searchhint">초성으로도 찾을 수 있음 — <b>ㅂㄷㅇ</b> → 박동원</div>')

    if query.strip():
        _render_results(context, query)
        return

    _render_landing(context)


def _render_results(context: Context, query: str) -> None:
    found = search_players(context.master, query, limit=SEARCH_LIMIT)
    rows = [
        {
            "player_id": int(row["player_id"]),
            "name": str(row["player_name"]),
            "team": str(row.get("team_latest") or ""),
            "meta": describe(row, context.master),
        }
        for _, row in found.iterrows()
    ]
    _html(ui.search_results(rows))


def _render_landing(context: Context) -> None:
    rows = _cached(context, "_ticker", ticker_rows)
    for row in rows:
        row["player_id"] = resolve_by_name(context, row["name"])
    _html(ui.ticker(rows))

    candidates = _cached(context, "_headline", headline_candidates)
    if candidates:
        _html(ui.section("다음 FA 주목 선수", "예상 연봉 순"))
        _html(ui.bento(candidates, overall_median(context, is_hitter=True)))

    _html(ui.board(counters(context)))
