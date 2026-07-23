"""StoveLens AI — 구단별 제시가 산정 + 렌더링.

호출 위치: src/ui/pages.py render_search()가 team_offers/render_team_bars/render_future_team_bars 호출.
데이터 파일: teams.csv, position_need.csv 기반 DataFrame(teams_df, pn_df)을 인자로 받음(직접 read 없음).
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: team_offers.py".
"""
import streamlit as st

from src.constants import POS_KR, TEAM_DATA

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


def team_offers(base, position, teams_df, pn_df):
    pn = pn_df[pn_df["position"] == position][["team_name", "need_score"]]
    m = teams_df.merge(pn, on="team_name", how="left")
    m["need_score"] = m["need_score"].fillna(0.5)
    m["offer"] = (base * (1 + m["need_score"] * 0.20 + m["win_now_score"] * 0.15 + m["cap_space_score"] * 0.10)).round(2)
    return m.sort_values("offer", ascending=False).reset_index(drop=True)


def render_team_bars(offers, base, player_pos=None):
    max_offer = offers["offer"].max()
    rows = ""
    for _, r in offers.iterrows():
        pct = r["offer"] / max_offer * 100
        color = "#C62828" if r["offer"] == max_offer else "#2c4a7e"
        abbr = r.get("team_abbr", r["team_name"][:2])
        rows += f"""
        <div class="team-row">
          <div class="team-name">{abbr}</div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
          </div>
          <div class="bar-val">{r['offer']:.1f}억</div>
        </div>"""
    st.markdown(f"""
    <p style="font-size:0.78rem;color:#888;margin:4px 0 10px 0;">
      기준 연봉 {base:.1f}억 · 포지션 필요도·구단 재정·우승 욕구 반영
    </p>{rows}""", unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.78rem;color:#888;margin:10px 0 4px 0;">구단별 제시가 산정 근거 ▼</p>', unsafe_allow_html=True)
    for _, r in offers.iterrows():
        tname = r["team_name"]
        toffer = r["offer"]
        low = round(toffer * 0.9, 1)
        high = round(toffer * 1.1, 1)
        tdata = next((v for k, v in TEAM_DATA.items() if k in tname), {})
        with st.expander(f"🏟️ {tname}  |  예상 제시가 {low:.1f}억 ~ {high:.1f}억"):
            _render_team_analysis(tdata, toffer, base, tname, player_pos)


def render_future_team_bars(predicted, player_pos=None):
    sorted_teams = sorted(TEAM_DATA.items(), key=lambda x: -x[1]["factor"])
    max_offer = predicted * sorted_teams[0][1]["factor"]
    rows = ""
    for team, tdata in sorted_teams:
        offer = round(predicted * tdata["factor"], 1)
        pct = offer / max_offer * 100
        color = "#C62828" if team == sorted_teams[0][0] else "#2c4a7e"
        low = round(offer * 0.9, 1)
        high = round(offer * 1.1, 1)
        rows += f"""
        <div class="team-row">
          <div class="team-name">{team}</div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
          </div>
          <div class="bar-val" style="width:90px;">{low}~{high}억</div>
        </div>"""
    st.markdown(f"""
    <p style="font-size:0.78rem;color:#888;margin:4px 0 10px 0;">
      기준 연봉 {predicted:.1f}억 · 구단별 선호도 보정 반영
    </p>{rows}""", unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.78rem;color:#888;margin:10px 0 4px 0;">구단별 제시가 산정 근거 ▼</p>', unsafe_allow_html=True)
    for team, tdata in sorted_teams:
        offer = round(predicted * tdata["factor"], 1)
        low = round(offer * 0.9, 1)
        high = round(offer * 1.1, 1)
        with st.expander(f"🏟️ {team}  |  예상 제시가 {low:.1f}억 ~ {high:.1f}억"):
            _render_team_analysis(tdata, offer, predicted, team, player_pos)


def _render_team_analysis(info, offer, base, team_name, player_pos=None):
    """구단 제시가 근거 텍스트 + 비교 바 차트."""
    try:
        reason = info.get("reason", "") or info.get("need", "")
        if reason:
            st.markdown(f"📋 {reason}")

        if player_pos:
            pos_kr = POS_KR.get(player_pos, player_pos)
            needs_list = info.get("needs", [])
            if any(pos_kr in n or n in pos_kr for n in needs_list):
                st.markdown(f"✅ 이 선수의 **{pos_kr}** 포지션은 해당 구단의 핵심 보강 포인트입니다. 경쟁 입찰 가능성 높음.")

        if info.get("win_now"):
            st.markdown("🔥 지금 당장 우승을 노리는 구단입니다. 즉전감 선수에게 공격적으로 접근합니다.")

        factor = info.get("factor", 1.0)
        st.markdown(f"**제시가 계산:** 기준 연봉 {base:.1f}억 × 구단 보정 계수 {factor:.2f} = **{offer:.1f}억**")
        st.caption("보정 계수: 1.0 기준 / 우승 의지·포지션 필요도·예산 규모 반영")

        if PLOTLY_OK:
            fig = go.Figure(go.Bar(
                x=[base, offer],
                y=["AI 예측 기준가", f"{team_name} 예상 제시가"],
                orientation="h",
                marker_color=["#3a3a5c", "#4f8ef7"],
                text=[f"{base:.1f}억", f"{offer:.1f}억"],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#1e1e30",
                font=dict(color="#f0f0f0"), showlegend=False,
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(ticksuffix="억", gridcolor="#3a3a5c"),
                height=110,
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
