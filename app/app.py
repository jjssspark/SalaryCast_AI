"""
StoveLens AI — KBO FA 연봉 예측 서비스 (Streamlit)
"""
import sys
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

    # 스플래시 로딩
    if "ready" not in st.session_state:
        splash = st.empty()
        with splash.container():
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:70vh;text-align:center;">
              <div style="font-size:5rem;animation:none;">⚾</div>
              <div style="font-size:2.2rem;font-weight:900;color:#0d1b35;margin:16px 0 8px;">
                StoveLens AI
              </div>
              <div style="font-size:1rem;color:#666;">KBO FA 연봉 예측 모델 로딩 중...</div>
            </div>""", unsafe_allow_html=True)
            with st.spinner(""):
                try:
                    h_df, p_df, teams, pn, future, fa_df = load_data()
                    hm, hx, hl, pm, px, pr = load_models()
                except DataLoadError as e:
                    splash.empty()
                    st.error(f"⚠️ 서비스를 시작할 수 없습니다: {e}")
                    st.caption("data/, models/ 폴더에 필요한 파일이 있는지 확인해주세요.")
                    st.stop()
            st.session_state.update(dict(
                ready=True, h_df=h_df, p_df=p_df, teams=teams,
                pn=pn, future=future, fa_df=fa_df, hm=hm, hx=hx, hl=hl, pm=pm, px=px, pr=pr,
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
        render_search(True,  s.h_df, s.teams, s.pn, s.future, s.hm, s.hx, s.hl, s.pm, s.px, s.pr, s.fa_df)
    with t_pit:
        render_search(False, s.p_df, s.teams, s.pn, s.future, s.hm, s.hx, s.hl, s.pm, s.px, s.pr, s.fa_df)


if __name__ == "__main__":
    main()
