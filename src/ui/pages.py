"""StoveLens AI — 페이지 단위 렌더링 (홈, 선수 검색, 상세 성적).

호출 위치: app/app.py main()이 render_home/render_search/show_detail_page 호출.
데이터 파일: 직접 read 없음 — st.session_state와 인자로 넘어온 DataFrame/모델을 사용,
show_detail_page 내부에서만 src.data_loader.load_season_stats()를 호출해 season_stats CSV를 읽음.
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: ui/".
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import POS_KR, TOOLTIPS
from src.data_loader import load_season_stats
from src.features import build_future_hitter_row, build_future_pitcher_row
from src.predict import predict_h, predict_p
from src.team_offers import render_future_team_bars, render_team_bars, team_offers
from src.ui.components import (
    _calc_pct,
    _html_table,
    _league_avg_pct,
    _parse_inning,
    _to_num,
    get_current_salary,
    get_player_fa_status,
    render_key_factors,
    show_player_photo,
)

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


def render_home(future_df):
    st.markdown("""
    <div class="hero">
      <h1>⚾ StoveLens AI</h1>
      <p>KBO FA 선수 예상 연봉 & 구단별 관심도 분석</p>
      <div class="hero-badge">2018~2026 FA 계약 데이터 기반 · AI 예측 모델</div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**사용 방법**")
        st.markdown("""
        <div class="how">
          <div class="how-num">1</div>
          <div class="how-txt">상단 탭에서 <strong>타자 찾기</strong> 또는 <strong>투수 찾기</strong>를 선택하세요</div>
        </div>
        <div class="how">
          <div class="how-num">2</div>
          <div class="how-txt">드롭다운에서 <strong>선수 이름</strong>을 선택하면 즉시 분석이 시작됩니다</div>
        </div>
        <div class="how">
          <div class="how-num">3</div>
          <div class="how-txt"><strong>예상 연봉</strong>과 <strong>구단별 관심도</strong>를 확인하세요
          <br><span style="font-size:0.8rem;color:#888;">실제 계약 선수는 예측 vs 실제 비교도 제공됩니다</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**서비스 안내**")
        st.markdown("""
        <p style="font-size:0.88rem;color:#555;line-height:1.8;margin:0;">
        • <strong>분석 대상</strong>: 2018~2026 KBO FA 계약 완료 선수<br>
        • <strong>예측 기반</strong>: 최근 3년 성적 + 나이 + 스타성 AI 분석<br>
        • <strong>구단별 관심도</strong>: 포지션 필요도 · 재정 · 우승 욕구 반영<br>
        • <strong>단위</strong>: 모든 연봉은 <strong>연평균(억 원)</strong> 기준
        </p>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**다음 FA 주목 선수 (2027~2030)**")
        st.markdown('<p style="font-size:0.8rem;color:#999;margin:0 0 12px 0;">실제 계약 전 · 예측 기능 준비 중</p>', unsafe_allow_html=True)
        highlights = future_df[future_df["fa_grade"].isin(["A", "B"])].head(12)
        for _, fr in highlights.iterrows():
            pos_kr = POS_KR.get(fr["position"], fr["position"])
            ptype = "타자" if fr["player_type"] == "hitter" else "투수"
            grade = fr.get("fa_grade", "")
            grade_txt = f'<span style="color:#C62828;font-weight:700">[{grade}급]</span> ' if pd.notna(grade) and grade else ""
            st.markdown(f"""
            <div class="fcard">
              <div>
                <div class="fcard-name">{fr['player_name']}</div>
                <div class="fcard-meta">{grade_txt}{fr['team_2026']} · {pos_kr} · {ptype}</div>
              </div>
              <div class="fcard-year">{int(fr['fa_year_expected'])}년 FA</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_search(is_hitter, df, teams_df, pn_df, future_df, hm, hx, hl, pm, px, pl, pr, ps, fa_df):
    player_type = "hitter" if is_hitter else "pitcher"
    df_sorted = df.sort_values("fa_year", ascending=False)
    unique_past = df_sorted.drop_duplicates("player_name")["player_name"].tolist()
    future_type = future_df[future_df["player_type"] == player_type].copy() if future_df is not None and len(future_df) > 0 else pd.DataFrame()
    future_names = [n for n in future_type["player_name"].tolist() if n not in unique_past] if len(future_type) > 0 else []
    if len(future_type) > 0 and future_names:
        ft_sorted = future_type[future_type["player_name"].isin(future_names)].sort_values("fa_year_expected")
        future_names = ft_sorted["player_name"].tolist()
    group_ended, group_active = [], []
    for name in unique_past:
        fa_st = get_player_fa_status(name, fa_df)
        (group_active if fa_st["status"] == "계약중" else group_ended).append(name)
    players = group_ended + group_active + future_names
    if is_hitter:
        default = next((n for n in ["최정", "이대호", "박건우"] if n in unique_past), players[0])
    else:
        default = next((n for n in ["양현종", "류현진", "원종현"] if n in unique_past), players[0])

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown(f'<p style="font-weight:700;font-size:0.95rem;color:#0d1b35;margin-bottom:4px;">{"타자" if is_hitter else "투수"} 선택</p>', unsafe_allow_html=True)
        selected = st.selectbox("선수 선택", players, index=players.index(default) if default in players else 0, label_visibility="collapsed")
        is_future_only = selected in future_names

        if is_future_only:
            fr = future_type[future_type["player_name"] == selected].iloc[0]
            h_stats_df, p_stats_df, h_star_df, p_star_df = load_season_stats()
            if is_hitter:
                future_pred_row = build_future_hitter_row(selected, fr, h_stats_df, h_star_df, df)
            else:
                future_pred_row = build_future_pitcher_row(selected, fr, p_stats_df, p_star_df, df)

            fa_yr_exp = int(fr["fa_year_expected"])
            star = int(future_pred_row.get("star_score", 0)) if future_pred_row is not None else 0
            star_str = "★" * min(star, 5) if star > 0 else ""
            if not is_hitter and future_pred_row is not None:
                pos = POS_KR.get(future_pred_row.get("pitcher_role", "SP"), "선발투수")
            else:
                pos = POS_KR.get(fr["position"], fr["position"])

            show_player_photo(selected, size=90)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;padding:8px 0 16px 0;">
              <div class="player-title">{selected}</div>
              <div style="margin:6px 0;">
                <span class="tag">{pos}</span>
                <span class="tag">{fr['team_2026']}</span>
                <span class="tag tag-red">FA 예정 {fa_yr_exp}년</span>
              </div>
              <div style="font-size:1.1rem;margin-top:6px;">{star_str}</div>
            </div>""", unsafe_allow_html=True)

            if future_pred_row is not None:
                st.markdown('<p style="font-size:0.85rem;font-weight:700;color:#444;margin:0 0 6px 0;">최근 3년 성적</p>', unsafe_allow_html=True)
                f_stats = ([
                    ("팀 기여도 (WAR)", f"{future_pred_row.get('war_3yr_sum', 0):.1f}"),
                    ("출루+장타 (OPS)", f"{future_pred_row.get('ops_3yr_avg', 0):.3f}"),
                    ("홈런 / 시즌",     f"{future_pred_row.get('hr_3yr_avg', 0):.1f}개"),
                    ("타점 / 시즌",     f"{future_pred_row.get('rbi_3yr_avg', 0):.1f}점"),
                    ("도루 / 시즌",     f"{future_pred_row.get('sb_3yr_avg', 0):.1f}개"),
                ] if is_hitter else [
                    ("팀 기여도 (WAR)", f"{future_pred_row.get('war_3yr_sum', 0):.1f}"),
                    ("방어율 (ERA)",     f"{future_pred_row.get('era_3yr_avg', 0):.2f}"),
                    ("이닝 / 시즌",     f"{future_pred_row.get('innings_3yr_avg', 0):.0f}이닝"),
                    ("세이브 / 시즌",   f"{future_pred_row.get('save_3yr_avg', 0):.1f}개"),
                    ("홀드 / 시즌",     f"{future_pred_row.get('hold_3yr_avg', 0):.1f}개"),
                ])
                for lbl, val in f_stats:
                    tip = TOOLTIPS.get(lbl, "")
                    title_attr = f' title="{tip}"' if tip else ""
                    st.markdown(
                        f'<div class="stat-row">'
                        f'<span class="stat-lbl"{title_attr}>{lbl}</span>'
                        f'<span class="stat-val">{val}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if st.button("📊 최근 3년 성적 자세히 보기 →", key=f"detail_btn_f_{selected}"):
                    st.session_state["page"] = "detail"
                    st.session_state["detail_player"] = selected
                    st.session_state["detail_type"] = player_type
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            row = df[df["player_name"] == selected].iloc[0]

            pos = POS_KR.get(row["position"], row["position"])
            if not is_hitter:
                pos = POS_KR.get(row.get("pitcher_role", "SP"), "선발투수")
            fa_yr = int(row["fa_year"])
            star  = int(row.get("star_score", 0))

            fa_status = get_player_fa_status(selected, fa_df)
            show_player_photo(selected, size=90)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            star_str = "★" * min(star, 5) if star > 0 else ""
            if fa_status["status"] == "계약중":
                badges_html = (
                    f'<span class="tag">{pos}</span>'
                    f'<span class="tag">{fa_yr}년 FA</span>'
                    f'<span class="tag">{fa_status["contract_years"]}년 계약 중</span>'
                    f'<span class="tag tag-red">계약 만료 {fa_status["contract_end_year"]}년</span>'
                )
            else:
                badges_html = (
                    f'<span class="tag">{pos}</span>'
                    f'<span class="tag">{fa_yr}년 FA 종료</span>'
                    f'<span class="tag tag-red">재FA 가능</span>'
                )
            st.markdown(f"""
            <div style="text-align:center;padding:8px 0 16px 0;">
              <div class="player-title">{selected}</div>
              <div style="margin:6px 0;">{badges_html}</div>
              <div style="font-size:1.1rem;margin-top:6px;">{star_str}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<p style="font-size:0.85rem;font-weight:700;color:#444;margin:0 0 6px 0;">최근 3년 성적</p>', unsafe_allow_html=True)
            stats = ([
                ("팀 기여도 (WAR)", f"{row.get('war_3yr_sum', 0):.1f}"),
                ("출루+장타 (OPS)", f"{row.get('ops_3yr_avg', 0):.3f}"),
                ("홈런 / 시즌",     f"{row.get('hr_3yr_avg', 0):.1f}개"),
                ("타점 / 시즌",     f"{row.get('rbi_3yr_avg', 0):.1f}점"),
                ("도루 / 시즌",     f"{row.get('sb_3yr_avg', 0):.1f}개"),
            ] if is_hitter else [
                ("팀 기여도 (WAR)", f"{row.get('war_3yr_sum', 0):.1f}"),
                ("방어율 (ERA)",     f"{row.get('era_3yr_avg', 0):.2f}"),
                ("이닝 / 시즌",     f"{row.get('innings_3yr_avg', 0):.0f}이닝"),
                ("세이브 / 시즌",   f"{row.get('save_3yr_avg', 0):.1f}개"),
                ("홀드 / 시즌",     f"{row.get('hold_3yr_avg', 0):.1f}개"),
            ])
            for lbl, val in stats:
                tip = TOOLTIPS.get(lbl, "")
                title_attr = f' title="{tip}"' if tip else ""
                st.markdown(
                    f'<div class="stat-row">'
                    f'<span class="stat-lbl"{title_attr}>{lbl}</span>'
                    f'<span class="stat-val">{val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if st.button("📊 최근 3년 성적 자세히 보기 →", key=f"detail_btn_{selected}"):
                st.session_state["page"] = "detail"
                st.session_state["detail_player"] = selected
                st.session_state["detail_type"] = player_type
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        if is_future_only:
            if future_pred_row is None:
                st.markdown("""
                <div class="card" style="text-align:center;padding:40px 24px;">
                  <div style="font-size:1.1rem;color:#666;">현재 스탯 데이터를 찾을 수 없습니다.</div>
                  <div style="font-size:0.88rem;color:#999;margin-top:8px;">선수명을 확인해주세요.</div>
                </div>""", unsafe_allow_html=True)
                return

            predicted = predict_h(future_pred_row, hx, hl, hm) if is_hitter else predict_p(future_pred_row, px, pl, pr, ps, pm)
            current_sal = get_current_salary(selected, fa_df)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            cur_sal_html = ""
            if current_sal is not None:
                cur_sal_html = (f'<div style="text-align:center;padding-bottom:10px;">'
                                f'<span style="font-size:0.82rem;color:#aaa;">현재 연봉 (2026)</span>'
                                f'<span style="font-size:1.1rem;font-weight:700;color:#88bbff;margin-left:10px;">약 {current_sal:.1f}억 원</span>'
                                f'</div>')
            st.markdown(f"""
            <div class="salary-box">
              {cur_sal_html}
              <div class="salary-label">AI 예상 FA 연봉</div>
              <div class="salary-num">{predicted:.1f}<span class="salary-unit"> 억 원</span></div>
            </div>
            <p style="font-size:0.78rem;color:#888;text-align:center;margin:4px 0 0 0;">
              {int(fr['fa_year_expected'])}년 FA 기준 · 최근 3년 성적 기반 예측
            </p>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            xgb_m_kf = hx if is_hitter else px
            feat_kf = hm["features"] if is_hitter else pm["features"]
            _kp = ("fh" if is_hitter else "fp") + "_" + selected
            render_key_factors(future_pred_row, xgb_m_kf, feat_kf, df, player_name=selected, key_prefix=_kp)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p style="font-weight:700;font-size:0.95rem;color:#0d1b35;margin:0 0 6px 0;">구단별 예상 제시가</p>', unsafe_allow_html=True)
            _fpos = future_pred_row.get("pitcher_role", "SP") if not is_hitter else fr["position"]
            render_future_team_bars(predicted, player_pos=_fpos)
            st.markdown("</div>", unsafe_allow_html=True)
            return

        predicted = predict_h(row, hx, hl, hm) if is_hitter else predict_p(row, px, pl, pr, ps, pm)
        actual    = float(row.get("annual_avg_salary", np.nan))
        has_actual = not np.isnan(actual)
        lookup_pos = row["position"] if is_hitter else row.get("pitcher_role", "SP")

        # 예상 연봉 카드
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="salary-box">
          <div class="salary-label">AI 예상 연봉 (연평균)</div>
          <div class="salary-num">{predicted:.1f}<span class="salary-unit"> 억 원</span></div>
        </div>""", unsafe_allow_html=True)

        if has_actual:
            diff = predicted - actual
            diff_pct = abs(diff) / actual * 100
            ratio = (actual - predicted) / predicted
            if ratio > 0.15:
                verdict, v_color, v_desc = "고평가", "#b71c1c", "실제 계약금이 AI 예측보다 상당히 높습니다"
            elif ratio < -0.15:
                verdict, v_color, v_desc = "저평가", "#1565c0", "실제 계약금이 AI 예측보다 낮습니다"
            else:
                verdict, v_color, v_desc = "적정", "#2e7d32", "예측과 실제 계약금이 유사합니다"
            st.markdown(f"""
            <div class="cmp-wrap">
              <div class="cmp-box">
                <div class="cmp-lbl">실제 계약 연봉</div>
                <div class="cmp-val" style="color:#1b5e20;">{actual:.1f}<span style="font-size:1rem;"> 억</span></div>
              </div>
              <div class="cmp-box">
                <div class="cmp-lbl">계약 평가</div>
                <div style="font-size:1.4rem;font-weight:900;color:{v_color};margin-top:4px;">{verdict}</div>
              </div>
            </div>
            <p style="font-size:0.78rem;color:#777;text-align:center;margin-top:4px;">{v_desc} · 오차 {diff_pct:.0f}%</p>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 핵심 요소 카드
        st.markdown('<div class="card">', unsafe_allow_html=True)
        xgb_m_kf = hx if is_hitter else px
        feat_kf = hm["features"] if is_hitter else pm["features"]
        _kp = ("ph" if is_hitter else "pp") + "_" + selected
        render_key_factors(row, xgb_m_kf, feat_kf, df, player_name=selected, key_prefix=_kp)
        st.markdown("</div>", unsafe_allow_html=True)

        # 구단별 관심도 카드
        offers = team_offers(predicted, lookup_pos, teams_df, pn_df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p style="font-weight:700;font-size:0.95rem;color:#0d1b35;margin:0 0 6px 0;">구단별 예상 제시가</p>', unsafe_allow_html=True)
        render_team_bars(offers, predicted, player_pos=lookup_pos)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("분석 상세 보기"):
            st.markdown("##### 모델 성능 지표")
            m1, m2, m3 = st.columns(3)
            if is_hitter:
                m1.metric("R²", "0.388")
                m2.metric("RMSE", "5.99억")
                m3.metric("학습 데이터", "80명")
                st.caption("모델: XGBoost + LightGBM 앙상블 | 타자 FA 2018~2026 기반")
            else:
                m1.metric("R²", "0.756")
                m2.metric("RMSE", "3.30억")
                m3.metric("학습 데이터", "43명")
                st.caption("모델: XGBoost + Random Forest 앙상블 | 투수 FA 2016~2026 기반")
            st.divider()
            st.markdown(f"**AI 예측 연봉:** {predicted:.2f}억 원")
            if has_actual:
                st.markdown(f"**실제 계약 연봉:** {actual:.2f}억 원")
                _d = predicted - actual
                st.markdown(f"**예측 오차:** {abs(_d):.2f}억 ({'과대 예측' if _d > 0 else '과소 예측'})")


def show_detail_page():
    player_name = st.session_state.get("detail_player", "")
    player_type = st.session_state.get("detail_type", "hitter")
    is_hitter = player_type == "hitter"

    if st.button("← 돌아가기", key="back_to_main"):
        st.session_state["page"] = "main"
        st.rerun()

    # 선수 사진 + 타이틀
    show_player_photo(player_name, size=80)
    st.title(f"📊 {player_name} — 최근 3년 성적 상세")

    # ── 데이터 로드 및 숫자 변환 ──────────────────────────────────────────────
    h_stats, p_stats, _, _ = load_season_stats()
    stat_df = h_stats if is_hitter else p_stats

    # pitcherInning "184 1/3" 문자열을 float으로 선처리
    if not is_hitter and "pitcherInning" in stat_df.columns:
        stat_df = stat_df.copy()
        stat_df["pitcherInning"] = stat_df["pitcherInning"].apply(_parse_inning)

    rows = stat_df[stat_df["playerName"] == player_name].sort_values("collect_year")
    if rows.empty:
        st.error(f"'{player_name}'의 시즌 데이터를 찾을 수 없습니다.")
        return

    recent_3 = rows.tail(3)
    all_years = rows["collect_year"].astype(int).tolist()

    # ── 섹션 1: 연도별 상세 스탯 테이블 ──────────────────────────────────────
    st.subheader("📋 연도별 상세 기록 (전체 커리어)")
    if is_hitter:
        raw_cols  = ["collect_year","teamName","hitterGameCount","hitterHra","hitterObp",
                     "hitterSlg","hitterOps","hitterWar","hitterHr","hitterRbi","hitterSb",
                     "hitterBb","hitterKk","hitterWoba"]
        kr_cols   = ["연도","팀","경기수","타율","출루율","장타율","OPS","WAR",
                     "홈런","타점","도루","볼넷","삼진","wOBA"]
        fmt_3dp   = {"타율","출루율","장타율","OPS","wOBA"}
        fmt_2dp   = set()
        fmt_1dp   = {"WAR"}
        fmt_int   = {"경기수","홈런","타점","도루","볼넷","삼진"}
    else:
        raw_cols  = ["collect_year","teamName","pitcherGameCount","pitcherInning",
                     "pitcherEra","pitcherWhip","pitcherWar","pitcherWin","pitcherLose",
                     "pitcherSave","pitcherHold","pitcherKk","pitcherBb"]
        kr_cols   = ["연도","팀","경기수","이닝","ERA","WHIP","WAR",
                     "승","패","세이브","홀드","삼진","볼넷"]
        fmt_3dp   = set()
        fmt_2dp   = {"ERA","WHIP"}
        fmt_1dp   = {"WAR","이닝"}
        fmt_int   = {"경기수","승","패","세이브","홀드","삼진","볼넷"}

    avail = [c for c in raw_cols if c in rows.columns]
    kr_map = dict(zip(raw_cols, kr_cols))
    disp = rows[avail].rename(columns=kr_map).copy()

    def _fmt(col_name, val):
        if pd.isna(val) or val is None:
            return "-"
        if col_name in fmt_3dp:
            return f"{float(val):.3f}"
        if col_name in fmt_2dp:
            return f"{float(val):.2f}"
        if col_name in fmt_1dp:
            return f"{float(val):.1f}"
        if col_name in fmt_int:
            return str(int(float(val)))
        return str(val)

    disp["연도"] = disp["연도"].astype(int)
    for col in disp.columns:
        if col not in ("연도", "팀"):
            disp[col] = disp[col].apply(lambda v: _fmt(col, v))

    st.markdown(_html_table(disp), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📖 스탯 용어 쉬운 설명 (처음 보시는 분들을 위해)"):
        if is_hitter:
            st.markdown("""
| 지표 | 한마디 설명 | 좋은 기준 |
|------|------------|---------|
| **타율** | 타석 중 안타를 친 비율. .300이면 3할 타자라 부름 | **.300 이상** |
| **출루율 (OBP)** | 아웃 당하지 않고 살아나간 비율 (안타+볼넷+몸맞공 포함) | **.350 이상** |
| **장타율 (SLG)** | 타석당 평균 베이스 진루 수. 홈런이 많을수록 높음 | **.450 이상** |
| **OPS** | 출루율+장타율. 타자의 전반적 공격력을 한 숫자로 요약 | **.850 이상** |
| **WAR** | 이 선수가 없었다면 팀이 잃었을 승리 수. 5.0이면 에이스급 | **3.0 이상** |
| **wOBA** | 각 타격 결과에 가중치를 달리 부여한 정밀 타격 지표 | **.360 이상** |
""")
        else:
            st.markdown("""
| 지표 | 한마디 설명 | 좋은 기준 |
|------|------------|---------|
| **ERA (방어율)** | 9이닝당 평균 자책점. 낮을수록 좋음 | **3.50 이하** |
| **WHIP** | 1이닝당 허용 출루 수 (피안타+볼넷). 낮을수록 안정적 | **1.20 이하** |
| **WAR** | 팀 기여도. 이 선수 없으면 팀이 잃었을 승리 수 | **2.0 이상** |
| **이닝** | 한 시즌 소화한 이닝. 선발 150이닝 이상이면 풀타임 에이스 | |
| **세이브** | 이기는 경기를 마무리하고 지킨 횟수. 마무리 투수 핵심 지표 | **20개 이상** |
| **홀드** | 중간에 리드를 지키고 다음 투수에게 넘긴 횟수. 셋업맨 지표 | **15개 이상** |
""")

    # ── 섹션 2: 연도별 추이 차트 ─────────────────────────────────────────────
    st.subheader("📈 연도별 성적 추이")
    if not PLOTLY_OK:
        st.warning("plotly 미설치. pip install plotly 후 재시작하세요.")
    else:
        _DK = dict(
            paper_bgcolor="#0f0f1a", plot_bgcolor="#1e1e30",
            font=dict(color="#f0f0f0", size=12),
            legend=dict(bgcolor="#1e1e30", bordercolor="#3a3a5c"),
            margin=dict(l=20, r=70, t=40, b=30),
            xaxis=dict(gridcolor="#3a3a5c", title="시즌", tickvals=all_years,
                       tickfont=dict(size=11)),
        )

        if is_hitter:
            # 차트 A: WAR (좌축) + OPS (우축)
            war_v = _to_num(rows["hitterWar"]).tolist() if "hitterWar" in rows.columns else []
            ops_v = _to_num(rows["hitterOps"]).tolist() if "hitterOps" in rows.columns else []
            if war_v:
                fa = go.Figure()
                fa.add_trace(go.Scatter(
                    x=all_years, y=war_v, name="WAR (팀 기여도)",
                    line=dict(color="#4f8ef7", width=2.5), mode="lines+markers+text",
                    text=[f"{v:.1f}" for v in war_v], textposition="top center",
                    textfont=dict(color="#4f8ef7", size=10),
                    marker=dict(size=7, color="#4f8ef7"),
                ))
                if ops_v:
                    fa.add_trace(go.Scatter(
                        x=all_years, y=ops_v, name="OPS",
                        line=dict(color="#f7c94f", width=2.5), mode="lines+markers+text",
                        text=[f"{v:.3f}" for v in ops_v], textposition="bottom center",
                        textfont=dict(color="#f7c94f", size=10),
                        marker=dict(size=7, color="#f7c94f"), yaxis="y2",
                    ))
                fa.update_layout(
                    title="WAR · OPS 연도별 추이",
                    yaxis=dict(title="WAR", gridcolor="#3a3a5c"),
                    yaxis2=dict(overlaying="y", side="right", title="OPS",
                                showgrid=False, tickformat=".3f"),
                    **_DK,
                )
                st.plotly_chart(fa, use_container_width=True,
                                key=f"ch_war_ops_{player_name}")

            # 차트 B: 홈런 · 타점 · 도루 (막대그래프)
            if "hitterHr" in rows.columns:
                fb = go.Figure()
                for colname, label, color in [
                    ("hitterHr",  "홈런", "#e53935"),
                    ("hitterRbi", "타점", "#ff9800"),
                    ("hitterSb",  "도루", "#43a047"),
                ]:
                    if colname in rows.columns:
                        vals = _to_num(rows[colname]).tolist()
                        fb.add_trace(go.Bar(
                            x=all_years, y=vals, name=label,
                            marker_color=color, opacity=0.85,
                            text=[str(int(v)) for v in vals],
                            textposition="outside",
                            textfont=dict(color=color, size=10),
                        ))
                fb.update_layout(
                    title="홈런 · 타점 · 도루 연도별 비교",
                    barmode="group",
                    yaxis=dict(gridcolor="#3a3a5c", title="개수"),
                    **_DK,
                )
                st.plotly_chart(fb, use_container_width=True,
                                key=f"ch_bat_{player_name}")

        else:
            # 투수 차트 A: WAR (좌축) + ERA (우축)
            war_v = _to_num(rows["pitcherWar"]).tolist() if "pitcherWar" in rows.columns else []
            era_v = _to_num(rows["pitcherEra"]).tolist() if "pitcherEra" in rows.columns else []
            if war_v:
                fa = go.Figure()
                fa.add_trace(go.Scatter(
                    x=all_years, y=war_v, name="WAR (팀 기여도)",
                    line=dict(color="#4f8ef7", width=2.5), mode="lines+markers+text",
                    text=[f"{v:.1f}" for v in war_v], textposition="top center",
                    textfont=dict(color="#4f8ef7", size=10),
                    marker=dict(size=7, color="#4f8ef7"),
                ))
                if era_v:
                    fa.add_trace(go.Scatter(
                        x=all_years, y=era_v, name="ERA (낮을수록 좋음)",
                        line=dict(color="#ef5350", width=2.5), mode="lines+markers+text",
                        text=[f"{v:.2f}" for v in era_v], textposition="bottom center",
                        textfont=dict(color="#ef5350", size=10),
                        marker=dict(size=7, color="#ef5350"), yaxis="y2",
                    ))
                fa.update_layout(
                    title="WAR · ERA 연도별 추이",
                    yaxis=dict(title="WAR", gridcolor="#3a3a5c"),
                    yaxis2=dict(overlaying="y", side="right",
                                title="ERA (낮을수록 좋음)", showgrid=False),
                    **_DK,
                )
                st.plotly_chart(fa, use_container_width=True,
                                key=f"ch_war_era_{player_name}")

            # 투수 차트 B: 이닝 (좌축) + 세이브/홀드 (우축)
            if "pitcherInning" in rows.columns:
                inn_v = _to_num(rows["pitcherInning"]).tolist()
                fb = go.Figure()
                fb.add_trace(go.Bar(
                    x=all_years, y=inn_v, name="소화 이닝",
                    marker_color="#4f8ef7", opacity=0.8,
                    text=[f"{v:.1f}" for v in inn_v], textposition="outside",
                    textfont=dict(color="#4f8ef7", size=10),
                ))
                for colname, label, color in [
                    ("pitcherSave", "세이브", "#fdd835"),
                    ("pitcherHold", "홀드",  "#66bb6a"),
                ]:
                    if colname in rows.columns:
                        sv = _to_num(rows[colname]).tolist()
                        fb.add_trace(go.Bar(
                            x=all_years, y=sv, name=label,
                            marker_color=color, opacity=0.85,
                            text=[str(int(v)) for v in sv], textposition="outside",
                            textfont=dict(color=color, size=10),
                        ))
                fb.update_layout(
                    title="이닝 · 세이브 · 홀드 연도별 비교",
                    barmode="group",
                    yaxis=dict(gridcolor="#3a3a5c"),
                    **_DK,
                )
                st.plotly_chart(fb, use_container_width=True,
                                key=f"ch_inn_{player_name}")

    # ── 섹션 3: 능력치 레이더 차트 ───────────────────────────────────────────
    st.subheader("🎯 리그 전체 선수 대비 능력치")
    st.caption(
        "0~100 점수. **파란 선**이 이 선수, **회색 점선**이 리그 평균입니다. "
        "파란 면적이 회색보다 크면 평균 이상."
    )

    if PLOTLY_OK:
        last3 = rows.tail(3)

        if is_hitter:
            metrics = [
                ("WAR\n(팀 기여도)", "hitterWar",  False),
                ("OPS\n(공격종합)", "hitterOps",  False),
                ("홈런",            "hitterHr",   False),
                ("출루율",          "hitterObp",  False),
                ("도루",            "hitterSb",   False),
            ]
        else:
            metrics = [
                ("WAR\n(팀 기여도)", "pitcherWar",   False),
                ("ERA\n(역순-낮을수록↑)", "pitcherEra", True),
                ("WHIP\n(역순)",     "pitcherWhip",  True),
                ("이닝",             "pitcherInning",False),
                ("세이브+홀드",      "_sh",          False),
            ]

        rlabels, pvals, lvals = [], [], []
        for label, col, inv in metrics:
            rlabels.append(label)
            if col == "_sh":
                sv = _to_num(rows["pitcherSave"]) if "pitcherSave" in rows.columns else pd.Series([0.0])
                hd = _to_num(rows["pitcherHold"]) if "pitcherHold" in rows.columns else pd.Series([0.0])
                player_val = float((sv + hd).tail(3).mean())
                sh_all = _to_num(stat_df.get("pitcherSave", pd.Series([0]))) + \
                         _to_num(stat_df.get("pitcherHold", pd.Series([0])))
                pvals.append(_calc_pct(sh_all, player_val, inv))
                lvals.append(_league_avg_pct(sh_all, inv))
            elif col in stat_df.columns:
                s_all = _to_num(stat_df[col])
                s_last3 = _to_num(last3[col]) if col in last3.columns else pd.Series([0.0])
                player_val = float(s_last3.mean()) if len(s_last3) > 0 else 0.0
                pvals.append(_calc_pct(s_all, player_val, inv))
                lvals.append(_league_avg_pct(s_all, inv))
            else:
                pvals.append(50.0)
                lvals.append(50.0)

        rfig = go.Figure()
        rfig.add_trace(go.Scatterpolar(
            r=pvals + [pvals[0]], theta=rlabels + [rlabels[0]],
            fill="toself", name=player_name,
            line_color="#4f8ef7", fillcolor="rgba(79,142,247,0.3)",
            mode="lines+markers",
            marker=dict(size=7, color="#4f8ef7"),
        ))
        rfig.add_trace(go.Scatterpolar(
            r=lvals + [lvals[0]], theta=rlabels + [rlabels[0]],
            fill="toself", name="리그 평균",
            line=dict(color="#aaaaaa", dash="dash", width=1.5),
            fillcolor="rgba(170,170,170,0.08)",
        ))
        rfig.update_layout(
            paper_bgcolor="#0f0f1a",
            polar=dict(
                bgcolor="#1e1e30",
                radialaxis=dict(color="#cccccc", gridcolor="#3a3a5c",
                                range=[0, 100], tickfont=dict(size=9)),
                angularaxis=dict(color="#e0e0e0", tickfont=dict(size=10)),
            ),
            font=dict(color="#f0f0f0"),
            legend=dict(bgcolor="#1e1e30", bordercolor="#3a3a5c"),
            margin=dict(l=60, r=60, t=60, b=60),
            height=420,
        )
        st.plotly_chart(rfig, use_container_width=True,
                        key=f"radar_{player_name}")

        # 지표별 점수 카드
        n = len(rlabels)
        cols_r = st.columns(n)
        for i, (label, pval, lval) in enumerate(zip(rlabels, pvals, lvals)):
            short = label.split("\n")[0]
            diff  = pval - lval
            if diff >= 10:
                color, arrow = "#4f8ef7", "▲"
                bg = "#0d1b35"
            elif diff >= -10:
                color, arrow = "#f7c94f", "●"
                bg = "#1a1a0f"
            else:
                color, arrow = "#ef5350", "▼"
                bg = "#2a0a0a"
            cols_r[i].markdown(
                f'<div style="text-align:center;background:{bg};border-radius:10px;'
                f'padding:10px 4px;border:1px solid #2a2a4a;">'
                f'<div style="font-size:0.72rem;color:#aaa;margin-bottom:4px;">{short}</div>'
                f'<div style="font-size:1.4rem;font-weight:900;color:{color};">{pval:.0f}</div>'
                f'<div style="font-size:0.65rem;color:{color};">{arrow} 평균 {lval:.0f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── 섹션 4: AI 종합 평가 ─────────────────────────────────────────────────
    st.subheader("💬 AI 종합 평가")
    lines = []
    if is_hitter:
        war_sum = _to_num(recent_3["hitterWar"]).sum() if "hitterWar" in recent_3.columns else 0.0
        ops_avg = _to_num(recent_3["hitterOps"]).mean() if "hitterOps" in recent_3.columns else 0.0
        hr_avg  = _to_num(recent_3["hitterHr"]).mean()  if "hitterHr"  in recent_3.columns else 0.0
        rbi_avg = _to_num(recent_3["hitterRbi"]).mean() if "hitterRbi" in recent_3.columns else 0.0
        sb_avg  = _to_num(recent_3["hitterSb"]).mean()  if "hitterSb"  in recent_3.columns else 0.0

        # WAR 평가
        if war_sum >= 12:
            lines.append(f"🏆 **WAR 합계 {war_sum:.1f}점** — 리그 최상위 기여도. FA 시장 최고 대우가 예상됩니다.")
        elif war_sum >= 7:
            lines.append(f"✅ **WAR 합계 {war_sum:.1f}점** — 팀의 핵심 선수급. 대형 계약 가능성이 높습니다.")
        elif war_sum >= 4:
            lines.append(f"📊 **WAR 합계 {war_sum:.1f}점** — 준주전급. 합리적인 중형 계약이 예상됩니다.")
        else:
            lines.append(f"⚠️ **WAR 합계 {war_sum:.1f}점** — 성적 기복 또는 부상 이력이 계약에 영향을 줄 수 있습니다.")

        # OPS 평가
        if ops_avg >= 0.900:
            lines.append(f"🔥 **OPS {ops_avg:.3f}** — .900 이상은 리그 최정상 타격 능력. 장기 대형 계약의 핵심 근거입니다.")
        elif ops_avg >= 0.800:
            lines.append(f"✅ **OPS {ops_avg:.3f}** — 안정적인 공격력. FA 시장에서 평균 이상 대우를 기대할 수 있습니다.")
        elif ops_avg >= 0.700:
            lines.append(f"📊 **OPS {ops_avg:.3f}** — 리그 평균 수준의 타격 능력입니다.")
        else:
            lines.append(f"⚠️ **OPS {ops_avg:.3f}** — 타격 능력이 평균 이하로, 연봉 산정에 불리하게 작용합니다.")

        # 추가 코멘트
        if hr_avg >= 25:
            lines.append(f"💥 **홈런 {hr_avg:.0f}개/시즌** — 장거리포 프리미엄 적용. 팬 흡입력과 스타성이 연봉을 끌어올립니다.")
        if sb_avg >= 25:
            lines.append(f"🏃 **도루 {sb_avg:.0f}개/시즌** — 최상급 스피드. 상위 타선 1·2번 자원으로 가치가 높습니다.")
        if rbi_avg >= 90:
            lines.append(f"🎯 **타점 {rbi_avg:.0f}점/시즌** — 클러치 능력 검증. 중심타선 역할에 충분한 성적입니다.")

    else:
        era_avg = _to_num(recent_3["pitcherEra"]).mean()  if "pitcherEra"  in recent_3.columns else 5.0
        war_sum = _to_num(recent_3["pitcherWar"]).sum()   if "pitcherWar"  in recent_3.columns else 0.0
        inn_avg = _to_num(recent_3["pitcherInning"]).mean()if "pitcherInning" in recent_3.columns else 0.0
        sv_avg  = _to_num(recent_3["pitcherSave"]).mean() if "pitcherSave" in recent_3.columns else 0.0
        hd_avg  = _to_num(recent_3["pitcherHold"]).mean() if "pitcherHold" in recent_3.columns else 0.0

        if era_avg < 3.0:
            lines.append(f"🏆 **ERA {era_avg:.2f}** — 3점대 미만은 리그 최정상급. FA 시장 최고 대우가 예상됩니다.")
        elif era_avg < 4.0:
            lines.append(f"✅ **ERA {era_avg:.2f}** — 안정적인 성적. 중형~대형 계약이 가능한 수준입니다.")
        elif era_avg < 5.0:
            lines.append(f"📊 **ERA {era_avg:.2f}** — 평균 수준. 구단이 보강 필요도와 예산에 따라 제시가를 결정할 것입니다.")
        else:
            lines.append(f"⚠️ **ERA {era_avg:.2f}** — 성적 부침이 있어 구단이 신중하게 접근할 가능성이 있습니다.")

        if war_sum >= 8:
            lines.append(f"🏆 **WAR 합계 {war_sum:.1f}점** — 에이스급 팀 기여도. 장기 계약의 충분한 근거가 됩니다.")
        elif war_sum >= 4:
            lines.append(f"✅ **WAR 합계 {war_sum:.1f}점** — 안정적인 팀 기여. 중형 계약이 예상됩니다.")

        if inn_avg >= 150:
            lines.append(f"💪 **{inn_avg:.0f}이닝/시즌** — 풀타임 에이스. 부상 없이 꾸준히 던지는 내구성이 가치를 높입니다.")
        if sv_avg >= 20:
            lines.append(f"🔒 **세이브 {sv_avg:.0f}개/시즌** — 검증된 마무리 투수. 마무리 자원은 FA 시장에서 희소가치가 높습니다.")
        elif hd_avg >= 20:
            lines.append(f"🛡️ **홀드 {hd_avg:.0f}개/시즌** — 핵심 셋업맨. 중간계투 필요 구단에서 높은 관심 예상됩니다.")

    if lines:
        for line in lines:
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:3px solid #4f8ef7;'
                f'border-radius:0 8px 8px 0;padding:10px 16px;margin-bottom:8px;'
                f'color:#e0e0e0;font-size:0.92rem;line-height:1.5;">{line}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("데이터 부족으로 평가를 생성하지 못했습니다.")
