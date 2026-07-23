"""StoveLens AI — 앱 CSS.

호출 위치: app/app.py main()이 st.markdown(CSS, unsafe_allow_html=True)로 적용.
데이터 파일: 없음.
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: ui/" (스타일 상수를 UI 레이어로 분리).
"""

CSS = """
<style>
.stApp, .main, section.main, [data-testid="stAppViewContainer"] {
    background-color: #0f0f1a !important; color: #f0f0f0 !important;
}
.stApp * { color: #f0f0f0 !important; }
h1,h2,h3,h4,h5,h6 { color: #ffffff !important; }

[data-testid="stTabs"] button { color: #aaaaaa !important; background: transparent !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #ffffff !important; border-bottom: 2px solid #4f8ef7 !important; }

[data-testid="stSelectbox"] > div > div { background-color: #1e1e30 !important; border: 1px solid #3a3a5c !important; }
[data-testid="stSelectbox"] * { color: #ffffff !important; }
[data-testid="stSelectbox"] svg { fill: #ffffff !important; }
[data-baseweb="popover"] *, [data-baseweb="menu"] * { background-color: #1e1e30 !important; color: #ffffff !important; }
[data-baseweb="option"]:hover { background-color: #2a2a4a !important; }

.stTextInput input, [data-testid="stTextInput"] input { background-color: #1e1e30 !important; color: #ffffff !important; border: 1px solid #3a3a5c !important; }
label, [data-testid="stWidgetLabel"] * { color: #aaaaaa !important; }

.stButton > button { background-color: #4f8ef7 !important; color: #ffffff !important; border: none !important; }
.stButton > button:hover { background-color: #3a7ae0 !important; }

[data-testid="stExpander"] { background-color: #1e1e30 !important; border: 1px solid #3a3a5c !important; }
[data-testid="stExpander"] * { color: #f0f0f0 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div { color: #f0f0f0 !important; }
[data-testid="stMetricValue"] { color: #4f8ef7 !important; }
[data-testid="stDataFrame"] * { color: #f0f0f0 !important; background-color: #1e1e30 !important; }
#MainMenu, footer, header { visibility: hidden !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f1a; }
::-webkit-scrollbar-thumb { background: #3a3a5c; border-radius: 3px; }

.block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1100px; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #1e1e30; border-radius: 14px; padding: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; font-size: 1rem; font-weight: 700; color: #aaa; padding: 10px 28px; border: none; }
.stTabs [aria-selected="true"] { background: #C62828 !important; color: white !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

.hero { background: linear-gradient(160deg, #0d1b35 0%, #1a2f4f 100%); border-radius: 18px; padding: 48px 40px; text-align: center; margin-bottom: 28px; }
.hero h1 { font-size: 2.8rem; font-weight: 900; color: #fff !important; margin: 0; letter-spacing: -1px; }
.hero p  { font-size: 1rem; color: #b0c8e0 !important; margin-top: 10px; }
.hero-badge { display: inline-block; background: rgba(255,255,255,0.18); color: #ddeeff !important; border-radius: 50px; padding: 6px 20px; font-size: 0.85rem; margin-top: 16px; }

.card { background: #1a1a2e; border-radius: 14px; padding: 22px; border: 1px solid #2a2a4a; box-shadow: 0 1px 6px rgba(0,0,0,0.3); margin-bottom: 14px; }

.player-title { font-size: 1.9rem; font-weight: 900; color: #ffffff !important; margin-bottom: 6px; }
.tag { display: inline-block; background: #2a2a4a; color: #aaccff !important; border-radius: 50px; padding: 3px 14px; font-size: 0.82rem; font-weight: 700; margin-right: 6px; margin-bottom: 4px; }
.tag-red { background: #3a0a0a !important; color: #ff6b6b !important; }

.salary-box { text-align: center; padding: 24px 0; border-bottom: 1px solid #2a2a4a; margin-bottom: 18px; }
.salary-label { font-size: 0.88rem; color: #aaa !important; margin-bottom: 6px; }
.salary-num   { font-size: 3.2rem; font-weight: 900; color: #ff4444 !important; line-height: 1; }
.salary-unit  { font-size: 1.3rem; color: #ff4444 !important; font-weight: 600; }

.cmp-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }
.cmp-box  { text-align: center; background: #0f0f1a; border-radius: 12px; padding: 16px 8px; }
.cmp-lbl  { font-size: 0.78rem; color: #aaa !important; margin-bottom: 4px; }
.cmp-val  { font-size: 1.6rem; font-weight: 900; color: #ffffff !important; }

.strength { display: flex; align-items: flex-start; gap: 10px; background: #0f0f1a; border-radius: 8px; padding: 10px 14px; margin-bottom: 7px; font-size: 0.9rem; color: #e0e0e0 !important; font-weight: 500; line-height: 1.4; }

.team-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.team-name { width: 80px; font-size: 0.82rem; font-weight: 700; color: #cccccc !important; }
.bar-bg { flex: 1; background: #2a2a4a; border-radius: 50px; height: 22px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 50px; }
.bar-val { width: 58px; text-align: right; font-size: 0.85rem; font-weight: 800; color: #aaccff !important; }

.fcard { background: #1a1a2e; border-radius: 10px; padding: 12px 16px; border-left: 3px solid #C62828; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
.fcard-name { font-weight: 800; font-size: 0.98rem; color: #ffffff !important; }
.fcard-meta { font-size: 0.8rem; color: #aaa !important; margin-top: 2px; }
.fcard-year { background: #C62828; color: white !important; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 700; white-space: nowrap; }

.stat-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #2a2a4a; }
.stat-row:last-child { border-bottom: none; }
.stat-lbl { font-size: 0.88rem; color: #aaa !important; }
.stat-lbl[title] { cursor: help; text-decoration: underline dotted #555; }
.stat-val { font-weight: 800; color: #ffffff !important; font-size: 0.9rem; }

.how { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 18px; }
.how-num { background: #C62828; color: white !important; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 0.9rem; flex-shrink: 0; }
.how-txt { font-size: 0.9rem; color: #cccccc !important; line-height: 1.5; margin-top: 3px; }

.element-container:empty { display: none !important; min-height: 0 !important; }
[data-testid="column"] { overflow: hidden; }
</style>
"""
