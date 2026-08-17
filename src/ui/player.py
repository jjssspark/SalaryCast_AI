"""StoveLens AI — 선수 한 명의 분석 화면.

호출 위치: src/ui/pages.py route()
데이터 파일을 읽지 않는다. app/app.py가 만든 Context를 받는다.

화면은 세 갈래로 갈린다.
  past   — 이미 FA 계약을 맺은 선수. 예측과 실제 계약을 나란히 놓는다.
  future — 앞으로 FA가 예정된 선수. 예상 연봉과 구단별 제시가를 낸다.
  active — 둘 다 아닌 현역. FA가 온다면 얼마일지를 참고값으로 낸다.
표본이 모자라면 숫자를 내지 않고 왜 못 내는지 적는다.

R²·RMSE·모델 이름 같은 말은 '분석 상세' 안에서만 쓴다.
"""

from __future__ import annotations

import streamlit as st

from src.explain import top_factors
from src.league import basis_text, compare, format_value
from src.predict import NotEnoughRecord
from src.serving import (
    Context,
    build_card,
    bundle_of,
    league_of,
    model_metrics,
    overall_median,
    position_median,
    salary_scale,
    season_table,
    stat_panel,
    team_offers,
    verdict,
)
from src.ui import components as ui

MODE_TITLE = {
    "past": "FA 직전 3시즌 성적",
    "future": "최근 3시즌 성적",
    "active": "최근 3시즌 성적",
}

NEED_LABEL = [(0.75, "보강 시급"), (0.5, "보강 검토"), (0.0, "여유")]


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _need_label(score: float) -> str:
    for threshold, label in NEED_LABEL:
        if score >= threshold:
            return label
    return "여유"


def render(context: Context, player_id: int) -> None:
    _html(ui.appbar(show_back=True))

    try:
        card = build_card(context, player_id)
    except NotEnoughRecord as error:
        _render_blocked(context, player_id, str(error))
        return

    _html(ui.player_hero(card))
    _html(f'<div class="cols">{_left_pane(context, card)}{_right_pane(context, card)}</div>')

    _html(ui.section("연봉을 결정한 것", "이 선수에게 크게 작용한 순서"))
    _html(ui.factor_list(top_factors(
        card.row,
        bundle_of(context, card.is_hitter),
        card.is_hitter,
        league_of(context, card.is_hitter),
    )))

    if card.mode != "past":
        _html(ui.section("구단별 예상 제시가", "포지션 필요도 · 우승 의지 · 재정 여력 반영"))
        _html(ui.offer_list(team_offers(context, card), _need_label))

    _html(ui.section("최근 3시즌 기록"))
    _html(ui.season_table(season_table(card), card.seasons_used))

    _html(ui.section("지표 설명"))
    _html(ui.glossary(card.is_hitter))

    _render_details(context, card)


def _render_blocked(context: Context, player_id: int, reason: str) -> None:
    found = context.master[context.master["player_id"] == player_id]
    name = str(found.iloc[0]["player_name"]) if not found.empty else "이 선수"

    _html(ui.section("예측할 수 없음"))
    _html(ui.notice(
        f"<b>{name}</b> — {reason}<br/>"
        "출전 기록이 적은 선수는 성적이 실력보다 우연에 크게 좌우돼, "
        "숫자를 내면 그럴듯해 보일 뿐 근거가 없습니다. 그래서 예측을 내지 않습니다."
    ))


def _left_pane(context: Context, card) -> str:
    median = overall_median(context, card.is_hitter)
    kind = "타자" if card.is_hitter else "투수"
    position = position_median(context, card)

    reference = f"역대 {kind} FA 연평균 중앙값은 <b>{median:.1f}억</b>"
    if position and position[1] >= 5 and card.is_hitter:
        reference += f", 그중 {card.position_label}는 <b>{position[0]:.1f}억</b>({position[1]}건)"
    reference += "."

    if card.mode == "past":
        return ui.versus_panel(card, verdict(card), reference)

    if card.mode == "active":
        subtitle = f"{card.fa_year}년에 FA가 온다고 가정 · 연평균 · 억 원"
    else:
        subtitle = f"{card.fa_year}년 FA 기준 · 연평균 · 억 원"

    return ui.money_panel(card, subtitle, reference, salary_scale(context))


def _right_pane(context: Context, card) -> str:
    league = league_of(context, card.is_hitter)
    stats = []

    for stat in stat_panel(card):
        value = float(stat["value"] or 0.0)
        stats.append({
            "key": stat["key"],
            "label": stat["label"],
            "fmt": stat["fmt"],
            "text": format_value(value, stat["fmt"]),
            "band": compare(league, stat["key"], value),
        })

    return ui.stat_board(stats, MODE_TITLE[card.mode], basis_text(league, card.is_hitter))


def _render_details(context: Context, card) -> None:
    """발표·보고서용. 여기 안에서만 모델 용어를 쓴다."""
    with st.expander("분석 상세 (발표용)"):
        st.markdown(
            f"**입력 시즌** {card.seasons_used}  \n"
            f"**FA 기준 연도** {card.fa_year}  \n"
            f"**분기** `{card.mode}`  \n"
            f"**나이** {card.age if card.age else '미상 — 나이 관련 피처를 결측으로 넣음'}  \n"
            f"**스타성** {card.star_score}점 "
            f"(MVP {card.mvp_count} · 골든글러브 {card.golden_glove_count} · "
            f"국가대표 {card.national_team})"
        )

        st.markdown(
            "| 대상 | 구성 | R² | RMSE | MAE | 표본 |\n|---|---|---|---|---|---|\n"
            + "\n".join(
                f"| {row['label']} | {row['method']} | {row['r2']:.3f} | "
                f"{row['rmse']:.2f}억 | {row['mae']:.2f}억 | {row['n']}명 |"
                for row in model_metrics(context)
            )
        )
        st.caption(
            "5-fold 교차검증을 5회 반복한 OOF 예측을 억 원 단위로 되돌려 계산한 값. "
            "타깃은 log1p(연평균 계약금). "
            "예상 범위는 해당 모델의 MAE만큼 위아래로 잡은 구간. "
            "요인 순서는 LightGBM 쪽 SHAP 기여도를 뜻이 같은 피처끼리 합산해 매김."
        )

        st.caption(
            "스타성은 연도별 수상자 명단에서 만든 값 "
            "(골든글러브 1982~2025 전량, MVP 39개 연도, 국제대회 10개). "
            "아시안게임 3개 대회는 한국어 위키백과에 야구 명단 문서가 없어 빠져 있음. "
            "올스타는 2014년 이전 자료가 없어 항목 자체를 뺐음."
        )
