"""StoveLens AI — 화면 라우터.

호출 위치: app/app.py main()
데이터 파일 없음. 주소창의 쿼리 파라미터만 읽는다.

    ?           -> 홈 (검색)
    ?p=<선수id> -> 그 선수의 분석 화면

Streamlit 위젯 대신 주소로 화면을 나눈다. 목업의 타일·티커·검색 결과가
전부 HTML 링크라 버튼 콜백을 걸 수 없고, 주소가 남아야 특정 선수 화면을
그대로 공유할 수 있다.
"""

from __future__ import annotations

import streamlit as st

from src.serving import Context
from src.ui import home, player


def route(context: Context) -> None:
    player_id = _requested_player()
    if player_id is None:
        home.render(context)
    else:
        player.render(context, player_id)


def _requested_player() -> int | None:
    raw = st.query_params.get("p")
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
