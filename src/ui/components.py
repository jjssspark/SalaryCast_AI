"""StoveLens AI — 재사용 UI 컴포넌트 및 헬퍼 함수.

호출 위치: src/ui/pages.py(render_home, render_search, show_detail_page)가 이 모듈의
함수들을 호출.
데이터 파일: 직접 read 없음 — 호출부에서 넘긴 DataFrame/row를 가공. get_player_photo만
네이버 스포츠 공개 API(sports.naver.com)를 조회.
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: ui/".
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.constants import LOWER_IS_BETTER, POS_KR, STAT_INFO, STAT_LABEL_MAP, TEAM_LOGO

try:
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ── 강점 분석 ──────────────────────────────────────────────────────────────────
def hitter_strengths(row):
    out = []
    war = row.get("war_3yr_sum", 0)
    if war >= 12:  out.append(("·", f"리그 최상위 팀 기여도 — 3년 합산 WAR {war:.1f}"))
    elif war >= 6: out.append(("·", f"꾸준한 팀 기여 — 3년 합산 WAR {war:.1f}"))
    ops = row.get("ops_3yr_avg", 0)
    if ops >= 0.900:   out.append(("·", f"강력한 타격 능력 — OPS {ops:.3f}"))
    elif ops >= 0.800: out.append(("·", f"안정적인 타격 — OPS {ops:.3f}"))
    hr = row.get("hr_3yr_avg", 0)
    if hr >= 25: out.append(("·", f"장거리포 — 홈런 연평균 {hr:.0f}개"))
    elif hr >= 15: out.append(("·", f"중장거리 타자 — 홈런 연평균 {hr:.0f}개"))
    sb = row.get("sb_3yr_avg", 0)
    if sb >= 20: out.append(("·", f"스피드 스타 — 도루 연평균 {sb:.0f}개"))
    rbi = row.get("rbi_3yr_avg", 0)
    if rbi >= 80: out.append(("·", f"클러치 히터 — 타점 연평균 {rbi:.0f}점"))
    star = int(row.get("star_score", 0))
    if star >= 6:   out.append(("·", "국가대표·MVP급 스타 플레이어"))
    elif star >= 3: out.append(("·", "골든글러브·올스타 경력 보유"))
    return out[:4] or [("·", "꾸준한 리그 활동 이력")]


def pitcher_strengths(row):
    out = []
    role = POS_KR.get(row.get("pitcher_role", "SP"), "투수")
    war = row.get("war_3yr_sum", 0)
    if war >= 8:   out.append(("·", f"리그 최상위 팀 기여 ({role}) — WAR {war:.1f}"))
    elif war >= 4: out.append(("·", f"안정적인 팀 기여 ({role}) — WAR {war:.1f}"))
    era = row.get("era_3yr_avg", 99)
    if era <= 3.00:   out.append(("·", f"압도적 방어율 — ERA {era:.2f}"))
    elif era <= 4.00: out.append(("·", f"안정적 방어율 — ERA {era:.2f}"))
    innings = row.get("innings_3yr_avg", 0)
    if innings >= 150: out.append(("·", f"강철 체력 선발 — 연평균 {innings:.0f}이닝"))
    elif innings >= 60: out.append(("·", f"풀타임 활약 — 연평균 {innings:.0f}이닝"))
    saves = row.get("save_3yr_avg", 0)
    if saves >= 25:  out.append(("·", f"리그 최정상 마무리 — 연평균 {saves:.0f}세이브"))
    elif saves >= 12: out.append(("·", f"검증된 마무리 — 연평균 {saves:.0f}세이브"))
    holds = row.get("hold_3yr_avg", 0)
    if holds >= 20: out.append(("·", f"핵심 셋업맨 — 연평균 {holds:.0f}홀드"))
    star = int(row.get("star_score", 0))
    if star >= 4: out.append(("·", "국가대표·최다승급 스타 투수"))
    return out[:4] or [("·", "꾸준한 리그 활동 이력")]


# ── FA 상태 ────────────────────────────────────────────────────────────────────
def get_current_salary(player_name, fa_df):
    """fa_contracts에서 해당 선수의 가장 최근 FA 연평균 연봉 반환. 없으면 None."""
    try:
        rows = fa_df[fa_df["player_name"] == player_name]
        if not rows.empty:
            val = rows.sort_values("fa_year").iloc[-1]["annual_avg_salary"]
            if pd.notna(val) and float(val) > 0:
                salary = float(val)
                if salary > 1000:  # 원 단위면 억으로 변환
                    salary = salary / 1e8
                return round(salary, 1)
    except Exception:
        pass
    return None


def get_player_fa_status(player_name, fa_df, current_year=2026):
    """가장 최근 FA 계약 기준으로 현재 계약 상태 반환."""
    player_fa = fa_df[fa_df["player_name"] == player_name].sort_values("fa_year")
    if player_fa.empty:
        return {"status": "미래FA예정", "last_fa_year": None, "next_fa_year": None,
                "contract_years": None, "contract_end_year": None}
    latest = player_fa.iloc[-1]
    last_fa_year = int(latest["fa_year"])
    cy = latest.get("contract_years")
    if pd.isna(cy) or int(cy) == 0:
        tot = latest.get("total_contract_amount", 0)
        ann = latest.get("annual_avg_salary", 1)
        if pd.notna(tot) and pd.notna(ann) and float(ann) > 0:
            cy = max(1, round(float(tot) / float(ann)))
        else:
            cy = 1
    contract_years = int(cy)
    contract_end_year = last_fa_year + contract_years - 1
    status = "계약중" if current_year <= contract_end_year else "FA가능"
    return {
        "status": status,
        "last_fa_year": last_fa_year,
        "contract_years": contract_years,
        "contract_end_year": contract_end_year,
        "next_fa_year": last_fa_year + contract_years,
    }


# ── 핵심 요소 ──────────────────────────────────────────────────────────────────
def render_key_factors(row, xgb_model, meta_features, train_df, player_name="", key_prefix=""):
    try:
        importances = xgb_model.feature_importances_
    except Exception:
        return
    fi = sorted(zip(meta_features, importances), key=lambda x: -x[1])
    top5 = [(f, imp) for f, imp in fi if f in STAT_INFO or f in STAT_LABEL_MAP][:5]
    if not top5:
        return

    st.markdown('<p style="font-weight:700;font-size:0.95rem;margin:0 0 10px 0;">이 선수의 연봉을 결정한 핵심 요소</p>', unsafe_allow_html=True)

    for rank, (feat, _) in enumerate(top5, 1):
        if feat in STAT_INFO:
            info = STAT_INFO[feat]
            icon, label, unit, desc, good = info["icon"], info["label"], info["unit"], info["desc"], info["good"]
        elif feat in STAT_LABEL_MAP:
            label, desc = STAT_LABEL_MAP[feat]
            icon, unit = "📊", ""
            good = "low" if feat in LOWER_IS_BETTER else "high"
        else:
            continue

        val = row.get(feat) if hasattr(row, "get") else None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        # 값 포맷
        if feat in {"era_3yr_avg", "era_last_year", "whip_3yr_avg", "whip_last_year",
                    "ops_3yr_avg", "ops_last_year", "woba_3yr_avg"}:
            val_str = f"{val:.3f}"
        elif feat == "age_at_fa":
            val_str = f"{int(val)}"
        elif feat in {"innings_3yr_avg", "games_3yr_avg"}:
            val_str = f"{val:.0f}"
        elif feat in {"war_3yr_sum", "war_3yr_avg", "war_last_year"}:
            val_str = f"{val:.1f}"
        elif feat in {"hr_3yr_avg", "rbi_3yr_avg", "save_3yr_avg", "hold_3yr_avg", "sb_3yr_avg"}:
            val_str = f"{val:.1f}"
        elif feat == "wrc_plus_3yr_avg":
            val_str = f"{val:.0f}"
        elif feat == "star_score":
            val_str = f"{int(val)}"
        else:
            val_str = f"{val:.2f}"

        val_with_unit = f"{val_str}{unit}" if unit else val_str

        # 백분위 계산
        try:
            if feat in train_df.columns:
                col_data = train_df[feat].dropna()
                if SCIPY_OK:
                    raw_pct = scipy_stats.percentileofscore(col_data, val, kind="rank")
                    pct = int(100 - raw_pct) if good == "high" else int(raw_pct)
                else:
                    base = float((col_data < val).sum()) / max(len(col_data), 1)
                    pct = int((1 - base) * 100) if good == "high" else int(base * 100)
            else:
                pct = 50
        except Exception:
            pct = 50

        # expander 헤더
        exp_label = f"{icon}  {label}   {val_with_unit}   리그 상위 {pct}%"

        with st.expander(exp_label, expanded=False):
            # ① 스탯 설명
            st.markdown(f"**{label}란?** {desc}")

            # ② 해석 문장
            if good == "high":
                if pct <= 10:
                    verdict = f"리그 상위 {pct}% 수준으로 매우 뛰어납니다. 연봉을 크게 끌어올리는 요인입니다."
                elif pct <= 30:
                    verdict = f"리그 상위 {pct}% 수준으로 평균 이상입니다. 연봉에 긍정적으로 작용합니다."
                elif pct <= 60:
                    verdict = f"리그 평균 수준(상위 {pct}%)입니다. 연봉에 중립적으로 작용합니다."
                else:
                    verdict = f"리그 하위 {100-pct}% 수준으로, 연봉을 낮추는 방향으로 작용합니다."
            else:
                if pct >= 90:
                    verdict = "리그 최하위 수준입니다. 이 수치는 연봉 평가에서 매우 불리하게 작용합니다."
                elif pct >= 70:
                    verdict = "리그 평균 이하 수준입니다. 연봉에 다소 부정적으로 작용합니다."
                else:
                    verdict = f"리그 상위 {pct}% 수준으로 우수합니다. 연봉을 높이는 방향으로 작용합니다."

            st.markdown(f"**이 선수의 수치:** {val_with_unit} — {verdict}")

            # ③ 리그 분포 히스토그램
            if PLOTLY_OK and feat in train_df.columns:
                try:
                    col_data = train_df[feat].dropna()
                    hist_fig = go.Figure()
                    hist_fig.add_trace(go.Histogram(x=col_data, nbinsx=20,
                                                    marker_color="#3a3a5c", name="리그 전체"))
                    hist_fig.add_vline(x=val, line_color="#4f8ef7", line_width=2,
                                       annotation_text=f"{player_name} ({val_with_unit})" if player_name else val_with_unit,
                                       annotation_font_color="#4f8ef7")
                    hist_fig.update_layout(
                        paper_bgcolor="#0f0f1a", plot_bgcolor="#1e1e30",
                        font=dict(color="#f0f0f0"), showlegend=False,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis=dict(gridcolor="#3a3a5c"),
                        yaxis=dict(gridcolor="#3a3a5c", title="선수 수"),
                        height=200,
                    )
                    st.plotly_chart(hist_fig, use_container_width=True,
                                    key=f"hist_{key_prefix}_{feat}_{rank}")
                    st.caption(f"리그 전체 FA 선수 중 이 선수의 {label} 위치")
                except Exception:
                    pass


# ── 상세 스탯 (탭 카드용, 검색 화면 내 요약) ────────────────────────────────────
def render_detail_stats(player_name, is_hitter, h_stats_df, p_stats_df):
    """연도별 상세 스탯 테이블 + 추세 라인 + 레이더 차트."""
    try:
        stat_df = h_stats_df if is_hitter else p_stats_df
        rows = stat_df[stat_df["playerName"] == player_name].sort_values("collect_year")
        if rows.empty:
            st.caption("시즌 데이터를 찾을 수 없습니다.")
            return

        tab1, tab2 = st.tabs(["📋 연도별 상세 스탯", "📈 추세 그래프"])

        with tab1:
            if is_hitter:
                display = rows[["collect_year", "teamName", "hitterGameCount",
                                "hitterHra", "hitterObp", "hitterSlg", "hitterOps",
                                "hitterWar", "hitterHr", "hitterRbi", "hitterSb"]].copy()
                display.columns = ["연도", "팀", "경기", "타율", "출루율", "장타율",
                                   "OPS", "WAR", "홈런", "타점", "도루"]
                for col in ["타율", "출루율", "장타율", "OPS"]:
                    display[col] = display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                display["WAR"] = display["WAR"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "⚠️ 비활성")
            else:
                display = rows[["collect_year", "teamName", "pitcherGameCount",
                                "pitcherInning", "pitcherEra", "pitcherWhip",
                                "pitcherWar", "pitcherWin", "pitcherLose",
                                "pitcherSave", "pitcherHold"]].copy()
                display.columns = ["연도", "팀", "경기", "이닝", "ERA", "WHIP",
                                   "WAR", "승", "패", "세이브", "홀드"]
                for col in ["ERA", "WHIP"]:
                    display[col] = display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                display["WAR"] = display["WAR"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "⚠️ 비활성")
            display["연도"] = display["연도"].astype(int)
            st.dataframe(display, use_container_width=True, hide_index=True)

        with tab2:
            if not PLOTLY_OK:
                st.caption("plotly가 설치되지 않아 차트를 표시할 수 없습니다.")
                return
            years = rows["collect_year"].tolist()
            _DARK = dict(paper_bgcolor="#0f0f1a", plot_bgcolor="#1e1e30",
                         font=dict(color="#f0f0f0"), legend=dict(bgcolor="#1e1e30"),
                         margin=dict(l=20, r=60, t=40, b=20))

            if is_hitter:
                war_v = rows["hitterWar"].tolist()
                ops_v = rows["hitterOps"].tolist()
                hr_v  = rows["hitterHr"].tolist()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=years, y=war_v, name="WAR",
                                         line=dict(color="#4f8ef7", width=2), mode="lines+markers"))
                fig.add_trace(go.Scatter(x=years, y=ops_v, name="OPS",
                                         line=dict(color="#f4b400", width=2), mode="lines+markers",
                                         yaxis="y2"))
                fig.add_trace(go.Scatter(x=years, y=hr_v,  name="홈런",
                                         line=dict(color="#ff6b6b", width=2), mode="lines+markers",
                                         yaxis="y3"))
                fig.update_layout(title="연도별 주요 스탯 추세",
                                  xaxis=dict(tickvals=years, gridcolor="#3a3a5c"),
                                  yaxis=dict(title="WAR", gridcolor="#3a3a5c"),
                                  yaxis2=dict(overlaying="y", side="right", title="OPS", showgrid=False),
                                  yaxis3=dict(overlaying="y", side="right", showgrid=False,
                                              anchor="free", position=1.0, title="홈런"),
                                  **_DARK)
                # Radar
                all_war = stat_df["hitterWar"].dropna(); all_ops = stat_df["hitterOps"].dropna()
                all_hr  = stat_df["hitterHr"].dropna();  all_obp = stat_df["hitterObp"].dropna()
                all_sb  = stat_df["hitterSb"].dropna()
                last3   = rows.tail(3)
                def _pct(series, val):
                    s = series.dropna()
                    return float((s < val).sum()) / len(s) if len(s) > 0 else 0.5
                theta = ["WAR", "OPS", "홈런", "출루율", "도루"]
                p_vals = [_pct(all_war, last3["hitterWar"].mean()),
                          _pct(all_ops, last3["hitterOps"].mean()),
                          _pct(all_hr,  last3["hitterHr"].mean()),
                          _pct(all_obp, last3["hitterObp"].mean()),
                          _pct(all_sb,  last3["hitterSb"].mean())]
                l_vals = [0.5] * 5
            else:
                war_v = rows["pitcherWar"].tolist()
                era_v = rows["pitcherEra"].tolist()
                inn_v = rows["pitcherInning"].tolist()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=years, y=war_v, name="WAR",
                                         line=dict(color="#4f8ef7", width=2), mode="lines+markers"))
                fig.add_trace(go.Scatter(x=years, y=era_v, name="ERA",
                                         line=dict(color="#ff6b6b", width=2), mode="lines+markers",
                                         yaxis="y2"))
                fig.add_trace(go.Scatter(x=years, y=inn_v, name="이닝",
                                         line=dict(color="#4caf50", width=2), mode="lines+markers",
                                         yaxis="y3"))
                fig.update_layout(title="연도별 주요 스탯 추세",
                                  xaxis=dict(tickvals=years, gridcolor="#3a3a5c"),
                                  yaxis=dict(title="WAR", gridcolor="#3a3a5c"),
                                  yaxis2=dict(overlaying="y", side="right", title="ERA", showgrid=False),
                                  yaxis3=dict(overlaying="y", side="right", showgrid=False,
                                              anchor="free", position=1.0, title="이닝"),
                                  **_DARK)
                all_war  = stat_df["pitcherWar"].dropna()
                all_era  = stat_df["pitcherEra"].dropna()
                all_whip = stat_df["pitcherWhip"].dropna()
                all_inn  = stat_df["pitcherInning"].dropna()
                all_win  = stat_df["pitcherWin"].dropna()
                last3    = rows.tail(3)
                def _pct(series, val):
                    s = series.dropna()
                    return float((s < val).sum()) / len(s) if len(s) > 0 else 0.5
                theta = ["WAR", "ERA(역)", "WHIP(역)", "이닝", "승리"]
                p_vals = [_pct(all_war,  last3["pitcherWar"].mean()),
                          1 - _pct(all_era,  last3["pitcherEra"].mean()),
                          1 - _pct(all_whip, last3["pitcherWhip"].mean()),
                          _pct(all_inn,  last3["pitcherInning"].mean()),
                          _pct(all_win,  last3["pitcherWin"].mean())]
                l_vals = [0.5] * 5

            st.plotly_chart(fig, use_container_width=True)

            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=p_vals + [p_vals[0]], theta=theta + [theta[0]],
                fill="toself", name=player_name,
                line_color="#4f8ef7", fillcolor="rgba(79,142,247,0.2)"
            ))
            radar.add_trace(go.Scatterpolar(
                r=l_vals + [l_vals[0]], theta=theta + [theta[0]],
                fill="toself", name="리그 평균",
                line_color="#aaaaaa", fillcolor="rgba(170,170,170,0.15)"
            ))
            radar.update_layout(
                title="리그 평균 대비 현황",
                polar=dict(bgcolor="#1e1e30",
                           radialaxis=dict(color="#f0f0f0", gridcolor="#3a3a5c", range=[0, 1]),
                           angularaxis=dict(color="#f0f0f0")),
                legend=dict(bgcolor="#1e1e30"),
                paper_bgcolor="#0f0f1a", font=dict(color="#f0f0f0"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(radar, use_container_width=True)
    except Exception:
        st.caption("상세 스탯을 불러오는 중 오류가 발생했습니다.")


# ── 선수 사진 / 구단 로고 ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_player_photo(player_name: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://sports.naver.com/",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://sports.naver.com",
    }
    IMAGE_KEYS = ["playerImageUrl", "profileImageUrl", "imageUrl", "profileImg",
                  "playerImage", "image", "photo", "thumbnail"]

    # Try 1: 네이버 스포츠 KBO 선수 통계 API (시즌별, 타자+투수 모두)
    for year in [2026, 2025, 2024]:
        for ptype, sort in [("HITTER", "hitterWar"), ("PITCHER", "pitcherWar"),
                             ("PITCHER", "pitcherSave"), ("PITCHER", "pitcherHold")]:
            try:
                url = (
                    f"https://api-gw.sports.naver.com/statistics/categories/kbo"
                    f"/seasons/{year}/players?playerType={ptype}"
                    f"&sortField={sort}&gameType=REGULAR_SEASON&page=1&pageSize=200"
                )
                resp = requests.get(url, headers=headers, timeout=8)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                players = (data.get("result", {}).get("players")
                           or data.get("data", {}).get("players")
                           or data.get("players") or [])
                for p in players:
                    pname = p.get("playerName") or p.get("name") or p.get("playerNm") or ""
                    if pname == player_name:
                        for k in IMAGE_KEYS:
                            if p.get(k):
                                return p[k]
            except Exception:
                pass

    # Try 2: 네이버 스포츠 선수 상세 검색
    try:
        url = (f"https://api-gw.sports.naver.com/search/people"
               f"?keyword={requests.utils.quote(player_name)}&sport=kbo&from=0&size=5")
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for top_key in ["result", "data", ""]:
                block = data.get(top_key, data) if top_key else data
                for sub_key in ["players", "items", "people", "list"]:
                    items = block.get(sub_key, []) if isinstance(block, dict) else []
                    if items:
                        for img_key in IMAGE_KEYS:
                            if items[0].get(img_key):
                                return items[0][img_key]
    except Exception:
        pass

    # Try 3: 네이버 스포츠 KBO 팀 로스터에서 검색
    try:
        url = (f"https://api-gw.sports.naver.com/schedule/teams/kbo"
               f"/roster?query={requests.utils.quote(player_name)}")
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for img_key in IMAGE_KEYS:
                val = data.get("result", {}).get(img_key) or data.get(img_key)
                if val:
                    return val
    except Exception:
        pass

    return None


def show_player_photo(player_name: str, size: int = 100):
    """선수 사진을 HTML img 태그로 표시 (st.image 사용 안 함 - 빈 박스 방지)."""
    url = get_player_photo(player_name)
    if url:
        st.markdown(
            f'<div style="text-align:center;margin-bottom:10px;">'
            f'<img src="{url}" width="{size}" height="{size}" '
            f'style="border-radius:50%;object-fit:cover;border:2px solid #3a3a5c;"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return
    initials = player_name[:1] if player_name else "?"
    st.markdown(
        f'<div style="text-align:center;margin-bottom:10px;">'
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:linear-gradient(135deg,#1e3a5f,#4f8ef7);'
        f'display:inline-flex;align-items:center;justify-content:center;'
        f'font-size:{size//3}px;font-weight:bold;color:white;">'
        f'{initials}</div></div>',
        unsafe_allow_html=True,
    )


def show_team_logo(team_name: str, size: int = 40):
    url = TEAM_LOGO.get(team_name)
    if url:
        try:
            st.image(url, width=size)
            return
        except Exception:
            pass
    st.markdown(f"**{team_name}**")


# ── 상세 성적 페이지 헬퍼 ────────────────────────────────────────────────────────
def _parse_inning(val) -> float:
    """'184 1/3' 같은 이닝 문자열을 float으로 변환."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if " " in s:
        parts = s.split(" ", 1)
        try:
            whole = float(parts[0])
            if "/" in parts[1]:
                n, d = parts[1].split("/")
                return whole + float(n) / float(d)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return 0.0


def _to_num(series: pd.Series) -> pd.Series:
    """시리즈를 안전하게 float 변환 (문자열 포함 처리)."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _calc_pct(series: pd.Series, val: float, inv: bool = False) -> float:
    """val이 series 분포에서 몇 %ile인지 반환 (0~100). inv=True → 낮을수록 좋은 지표."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 50.0
    raw = float((s < val).sum()) / len(s)
    return round((1.0 - raw if inv else raw) * 100.0, 1)


def _league_avg_pct(series: pd.Series, inv: bool = False) -> float:
    """리그 평균값이 분포에서 몇 %ile인지 반환 (리그 평균 기준선 계산용)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 50.0
    avg = s.mean()
    raw = float((s < avg).sum()) / len(s)
    return round((1.0 - raw if inv else raw) * 100.0, 1)


def _html_table(df: pd.DataFrame) -> str:
    """다크 테마 HTML 테이블 생성 (st.dataframe 대신 사용)."""
    th_style = ("background:#1a1a2e;color:#aaccff;font-weight:700;"
                "padding:8px 12px;text-align:center;border-bottom:2px solid #3a3a5c;"
                "font-size:0.82rem;white-space:nowrap;")
    td_style = ("color:#f0f0f0;padding:7px 12px;text-align:center;"
                "border-bottom:1px solid #2a2a4a;font-size:0.85rem;")
    tr_even  = "background:#1a1a2e;"
    tr_odd   = "background:#0f0f1a;"

    head = "".join(f"<th style='{th_style}'>{c}</th>" for c in df.columns)
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = tr_even if i % 2 == 0 else tr_odd
        cells = "".join(f"<td style='{td_style}'>{v}</td>" for v in row)
        rows_html += f"<tr style='{bg}'>{cells}</tr>"

    return (
        f"<div style='overflow-x:auto;border-radius:10px;border:1px solid #2a2a4a;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )
