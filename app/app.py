"""StoveLens AI — KBO FA 연평균 계약금 예측 서비스 (Streamlit).

실행: streamlit run app/app.py

여기서 하는 일은 셋뿐이다.
  1. 데이터·모델을 읽어 Context 하나로 묶는다
  2. 스타일과 SVG 심볼을 한 번 주입한다
  3. 라우터에 넘긴다

화면 구성은 src/ui/, 예측 조립은 src/serving.py에 있다.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402

from src.data_loader import (  # noqa: E402
    DataLoadError,
    load_awards,
    load_birth_lookup,
    load_data,
    load_models,
    load_search_index,
    load_season_stats,
)
from src.league import build_reference  # noqa: E402
from src.serving import Context  # noqa: E402
from src.ui import components as ui  # noqa: E402
from src.ui.assets import SVG_DEFS  # noqa: E402
from src.ui.pages import route  # noqa: E402
from src.ui.styles import CSS  # noqa: E402


@st.cache_resource(show_spinner=False)
def build_context() -> Context:
    teams, position_need, future, fa = load_data()
    hitter_seasons, pitcher_seasons = load_season_stats()
    master, photos = load_search_index()
    hitter_bundle, pitcher_bundle = load_models()

    # 사진은 이름이 아니라 player_id로 붙인다. 같은 이름이 96쌍이라
    # 이름으로 붙이면 다른 사람 얼굴이 나온다.
    photo_by_id = {
        int(row.player_id): str(row.photo_url)
        for row in photos.itertuples()
        if isinstance(row.photo_url, str) and row.photo_url
    }

    return Context(
        master=master,
        photos=photos,
        hitter_seasons=hitter_seasons,
        pitcher_seasons=pitcher_seasons,
        awards=load_awards(),
        fa=fa,
        future=future,
        birth=load_birth_lookup(),
        teams=teams,
        position_need=position_need,
        hitter_bundle=hitter_bundle,
        pitcher_bundle=pitcher_bundle,
        hitter_league=build_reference(hitter_seasons, is_hitter=True),
        pitcher_league=build_reference(pitcher_seasons, is_hitter=False),
        photo_by_id=photo_by_id,
    )


def main() -> None:
    st.set_page_config(
        page_title="StoveLens AI",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(SVG_DEFS, unsafe_allow_html=True)

    placeholder = st.empty()
    placeholder.markdown(ui.splash(), unsafe_allow_html=True)

    try:
        context = build_context()
    except DataLoadError as error:
        placeholder.empty()
        st.error(f"서비스를 시작할 수 없습니다: {error}")
        st.caption("data/, models/ 폴더에 필요한 파일이 있는지 확인해주세요.")
        st.stop()

    placeholder.empty()
    route(context)


if __name__ == "__main__":
    main()
