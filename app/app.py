"""
StoveLens AI — KBO FA 연봉 예측 서비스 (Streamlit)
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.data_loader import DataLoadError, load_data, load_models
from src.ui.pages import render_home, render_search, show_detail_page
from src.ui.styles import CSS


def main():
    st.set_page_config(
        page_title="StoveLens AI ⚾",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""
    <span class="field-dust" style="left:6%;animation-delay:0s;"></span>
    <span class="field-dust" style="left:22%;animation-delay:2s;"></span>
    <span class="field-dust" style="left:41%;animation-delay:4.5s;"></span>
    <span class="field-dust" style="left:63%;animation-delay:1.2s;"></span>
    <span class="field-dust" style="left:80%;animation-delay:3.4s;"></span>
    <span class="field-dust" style="left:93%;animation-delay:5.6s;"></span>
    """, unsafe_allow_html=True)

    # 스플래시 로딩 (최소 노출 시간 확보 — 데이터가 캐시돼 있으면 순간 로딩이라 깜빡이고 사라지는 것 방지)
    MIN_SPLASH_SECONDS = 1.6
    if "ready" not in st.session_state:
        splash = st.empty()
        with splash.container():
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:78vh;text-align:center;position:relative;">
              <div class="splash-ball">⚾</div>
              <div style="font-size:2.4rem;font-weight:900;color:#fff;margin:20px 0 6px;letter-spacing:-1px;">
                StoveLens <span style="color:#ff5252;">AI</span>
              </div>
              <div style="font-size:0.95rem;color:#9fb4cf;margin-bottom:26px;">KBO FA 연봉 예측 모델 로딩 중...</div>
              <div class="splash-bar"><div class="splash-bar-fill"></div></div>
            </div>""", unsafe_allow_html=True)
            _start = time.time()
            with st.spinner(""):
                try:
                    h_df, p_df, teams, pn, future, fa_df = load_data()
                    hm, hx, hl, pm, px, pl, pr, ps = load_models()
                except DataLoadError as e:
                    splash.empty()
                    st.error(f"⚠️ 서비스를 시작할 수 없습니다: {e}")
                    st.caption("data/, models/ 폴더에 필요한 파일이 있는지 확인해주세요.")
                    st.stop()
            _elapsed = time.time() - _start
            if _elapsed < MIN_SPLASH_SECONDS:
                time.sleep(MIN_SPLASH_SECONDS - _elapsed)
            st.session_state.update(dict(
                ready=True, h_df=h_df, p_df=p_df, teams=teams,
                pn=pn, future=future, fa_df=fa_df, hm=hm, hx=hx, hl=hl,
                pm=pm, px=px, pl=pl, pr=pr, ps=ps,
            ))
        splash.empty()
        st.rerun()

    # 상세 페이지 라우터
    if st.session_state.get("page") == "detail":
        show_detail_page()
        return

    s = st.session_state
    t_home, t_hit, t_pit = st.tabs(["홈", "타자 찾기", "투수 찾기"])

    with t_home:
        render_home(s.future)
    with t_hit:
        render_search(True,  s.h_df, s.teams, s.pn, s.future, s.hm, s.hx, s.hl, s.pm, s.px, s.pl, s.pr, s.ps, s.fa_df)
    with t_pit:
        render_search(False, s.p_df, s.teams, s.pn, s.future, s.hm, s.hx, s.hl, s.pm, s.px, s.pl, s.pr, s.ps, s.fa_df)


if __name__ == "__main__":
    main()
