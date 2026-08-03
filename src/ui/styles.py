"""StoveLens AI — 앱 CSS.

호출 위치: app/app.py main()이 st.markdown(CSS, unsafe_allow_html=True)로 적용.
데이터 파일: 없음.
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: ui/" (스타일 상수를 UI 레이어로 분리).
"""

CSS = """
<style>
.stApp, .main, section.main, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1400px 700px at 12% -10%, rgba(198,40,40,0.12) 0%, rgba(198,40,40,0) 45%),
        radial-gradient(1200px 800px at 100% 0%, rgba(79,142,247,0.10) 0%, rgba(79,142,247,0) 50%),
        radial-gradient(1600px 600px at 50% 120%, rgba(46,125,50,0.12) 0%, rgba(46,125,50,0) 55%),
        #0f0f1a !important;
    color: #f0f0f0 !important;
}
[data-testid="stAppViewContainer"] { position: relative; }
[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='130' height='130'%3E%3Cpath d='M6 65 Q 32 40 65 65 T 124 65' stroke='%23ff5252' stroke-width='1.1' fill='none' stroke-dasharray='2.5 5' opacity='0.05'/%3E%3C/svg%3E");
    background-repeat: repeat; background-size: 130px 130px;
}
[data-testid="stAppViewContainer"] .block-container { position: relative; z-index: 1; }
.field-dust { position: fixed; width: 3px; height: 3px; border-radius: 50%; background: #fff; box-shadow: 0 0 6px 1px rgba(255,255,255,0.5); opacity: 0; pointer-events: none; z-index: 0; animation: dustDrift 9s ease-in-out infinite; }
@keyframes dustDrift { 0% { opacity: 0; transform: translateY(0); } 15% { opacity: .5; } 85% { opacity: .35; } 100% { opacity: 0; transform: translateY(-140px); } }
.stApp * { color: #f0f0f0 !important; }
h1,h2,h3,h4,h5,h6 { color: #ffffff !important; }

[data-testid="stTabs"] button { color: #aaaaaa !important; background: transparent !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #ffffff !important; border-bottom: 2px solid #4f8ef7 !important; }

[data-testid="stSelectbox"] > div > div { background-color: #1e1e30 !important; border: 1px solid #3a3a5c !important; transition: border-color .2s ease, box-shadow .2s ease; }
[data-testid="stSelectbox"]:focus-within > div > div { border-color: #ff5252 !important; box-shadow: 0 0 0 3px rgba(255,82,82,0.22) !important; }
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

@keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes floatBall { 0%, 100% { transform: translateY(0) rotate(-4deg); } 50% { transform: translateY(-8px) rotate(4deg); } }
@keyframes shine { 0% { left: -60%; } 55% { left: 120%; } 100% { left: 120%; } }

.hero {
    position: relative; overflow: hidden;
    background:
        radial-gradient(120% 140% at 15% -10%, rgba(198,40,40,0.42) 0%, rgba(198,40,40,0) 48%),
        radial-gradient(100% 120% at 100% 0%, rgba(79,142,247,0.34) 0%, rgba(79,142,247,0) 52%),
        radial-gradient(90% 75% at 50% 118%, rgba(46,125,50,0.40) 0%, rgba(46,125,50,0) 62%),
        linear-gradient(160deg, #0d1b35 0%, #1a2f4f 100%);
    border-radius: 20px; padding: 52px 40px 36px; text-align: center; margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    animation: fadeInUp .7s ease both;
}
.hero::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image:
        repeating-linear-gradient(100deg, rgba(255,255,255,0.045) 0px, rgba(255,255,255,0.045) 1px, transparent 1px, transparent 46px),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='96' height='96'%3E%3Cpath d='M6 48 Q 26 28 46 48 T 90 48' stroke='%23ff5252' stroke-width='1.6' fill='none' stroke-dasharray='3 4' opacity='0.32'/%3E%3C/svg%3E");
    background-repeat: repeat, repeat;
    background-size: auto, 96px 96px;
}
.hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='460' height='460'%3E%3Cpath d='M230 30 L430 230 L230 430 L30 230 Z' fill='none' stroke='%23ffffff' stroke-width='2.5' opacity='0.13'/%3E%3Cpath d='M230 30 L430 230 L230 430 L30 230 Z' fill='none' stroke='%23ffffff' stroke-width='1' opacity='0.08' transform='scale(0.7) translate(98 98)'/%3E%3Ccircle cx='230' cy='230' r='6' fill='%23ffffff' opacity='0.24'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: bottom -60px right -60px;
    background-size: 460px 460px;
}
.hero-spark { position: absolute; width: 6px; height: 6px; border-radius: 50%; background: #fff; box-shadow: 0 0 8px 2px rgba(255,255,255,0.6); opacity: 0; pointer-events: none; animation: sparkFloat 4.2s ease-in-out infinite; }
@keyframes sparkFloat { 0%, 100% { opacity: 0; transform: translateY(0); } 50% { opacity: .85; transform: translateY(-22px); } }
.hero-ball { font-size: 2.6rem; display: inline-block; animation: floatBall 3.2s ease-in-out infinite; filter: drop-shadow(0 6px 10px rgba(0,0,0,0.45)); position: relative; }
.hero h1 { font-size: 2.9rem; font-weight: 900; color: #fff !important; margin: 6px 0 0; letter-spacing: -1.5px; position: relative; }
.hero h1 .accent { color: #ff5252 !important; }
.hero p  { font-size: 1.05rem; color: #b0c8e0 !important; margin-top: 12px; position: relative; }
.hero-badge {
    position: relative; overflow: hidden; display: inline-block;
    background: rgba(255,255,255,0.12); color: #ddeeff !important;
    border: 1px solid rgba(255,255,255,0.16); border-radius: 50px;
    padding: 7px 22px; font-size: 0.85rem; margin-top: 18px;
}
.hero-badge::after {
    content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
    animation: shine 3.6s ease-in-out infinite;
}

.player-hero {
    position: relative; overflow: hidden; text-align: center;
    padding: 30px 20px 22px; border-radius: 18px; margin-bottom: 14px;
    background:
        radial-gradient(120% 100% at 50% 0%, rgba(79,142,247,0.22) 0%, rgba(79,142,247,0) 65%),
        linear-gradient(180deg, #14142a 0%, #101020 100%);
    border: 1px solid rgba(255,255,255,0.07);
    animation: fadeInUp .6s ease both;
}
.player-hero::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: repeating-linear-gradient(100deg, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 40px);
}
.player-photo-ring {
    position: relative; display: inline-flex; margin-bottom: 14px; border-radius: 50%;
    box-shadow: 0 0 0 3px #0f0f1a, 0 0 0 5px rgba(79,142,247,0.5), 0 8px 26px rgba(0,0,0,0.55);
    animation: ringPulse 2.6s ease-in-out infinite;
}
@keyframes ringPulse {
    0%, 100% { box-shadow: 0 0 0 3px #0f0f1a, 0 0 0 5px rgba(79,142,247,0.5), 0 8px 26px rgba(0,0,0,0.55); }
    50% { box-shadow: 0 0 0 3px #0f0f1a, 0 0 0 8px rgba(255,82,82,0.45), 0 8px 30px rgba(0,0,0,0.6); }
}

.trust-bar {
    display: flex; justify-content: center; gap: 0; margin-top: 32px; position: relative; flex-wrap: wrap;
    background: #05060c; border: 1px solid #2a3550; border-radius: 12px;
    padding: 16px 8px; box-shadow: inset 0 2px 10px rgba(0,0,0,0.6), 0 4px 14px rgba(0,0,0,0.35);
}
.trust-item { padding: 0 28px; text-align: center; position: relative; }
.trust-item + .trust-item { border-left: 1px dashed rgba(120,200,140,0.25); }
.trust-num {
    display: block; font-family: 'Courier New', ui-monospace, monospace; font-size: 1.7rem; font-weight: 900;
    color: #7CFC9C !important; letter-spacing: 1px;
    text-shadow: 0 0 10px rgba(124,252,156,0.65), 0 0 22px rgba(124,252,156,0.35);
}
.trust-lbl { display: block; font-size: 0.68rem; color: #7a8bab !important; margin-top: 5px; letter-spacing: 0.5px; }

.section-title { font-size: 1rem; font-weight: 800; color: #fff !important; margin: 0 0 16px 0; display: flex; align-items: center; gap: 9px; }
.section-title::before { content: ""; width: 4px; height: 16px; background: #C62828; border-radius: 2px; display: inline-block; flex-shrink: 0; }

.card {
    background: #1a1a2e; border-radius: 14px; padding: 22px; border: 1px solid #2a2a4a;
    box-shadow: 0 1px 6px rgba(0,0,0,0.3); margin-bottom: 14px;
    animation: fadeInUp .6s ease both;
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.4); border-color: #3a5a8f; }

.player-title { font-size: 1.9rem; font-weight: 900; color: #ffffff !important; margin-bottom: 6px; }
.tag { display: inline-block; background: #2a2a4a; color: #aaccff !important; border-radius: 50px; padding: 3px 14px; font-size: 0.82rem; font-weight: 700; margin-right: 6px; margin-bottom: 4px; }
.tag-red { background: #3a0a0a !important; color: #ff6b6b !important; }

.salary-box { position: relative; text-align: center; padding: 24px 0; border-bottom: 1px solid #2a2a4a; margin-bottom: 18px; }
.salary-box::before { content: ""; position: absolute; top: 50%; left: 50%; width: 220px; height: 220px; background: radial-gradient(circle, rgba(255,68,68,0.18) 0%, transparent 70%); transform: translate(-50%, -50%); pointer-events: none; animation: glowBreathe 2.8s ease-in-out infinite; }
.salary-label { font-size: 0.88rem; color: #aaa !important; margin-bottom: 6px; position: relative; }
.salary-num   { font-size: 3.2rem; font-weight: 900; color: #ff4444 !important; line-height: 1; position: relative; animation: fadeInUp .5s ease both, numGlow 2.8s ease-in-out infinite; }
.salary-unit  { font-size: 1.3rem; color: #ff4444 !important; font-weight: 600; position: relative; }
@keyframes glowBreathe { 0%, 100% { opacity: .7; transform: translate(-50%, -50%) scale(1); } 50% { opacity: 1; transform: translate(-50%, -50%) scale(1.12); } }
@keyframes numGlow { 0%, 100% { text-shadow: 0 0 20px rgba(255,68,68,0.35); } 50% { text-shadow: 0 0 34px rgba(255,68,68,0.65); } }

.cmp-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }
.cmp-box  { text-align: center; background: #0f0f1a; border-radius: 12px; padding: 16px 8px; }
.cmp-lbl  { font-size: 0.78rem; color: #aaa !important; margin-bottom: 4px; }
.cmp-val  { font-size: 1.6rem; font-weight: 900; color: #ffffff !important; }

.strength { display: flex; align-items: flex-start; gap: 10px; background: #0f0f1a; border-radius: 8px; padding: 10px 14px; margin-bottom: 7px; font-size: 0.9rem; color: #e0e0e0 !important; font-weight: 500; line-height: 1.4; }

.team-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.team-name { width: 92px; font-size: 0.82rem; font-weight: 700; color: #cccccc !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-bg { flex: 1; background: #2a2a4a; border-radius: 50px; height: 22px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 50px; background-image: linear-gradient(90deg, rgba(255,255,255,0.12), transparent); width: 0; animation: barGrow 1s cubic-bezier(.2,.7,.3,1) both; }
.bar-fill-lead { position: relative; overflow: hidden; }
.bar-fill-lead::after {
    content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.55), transparent);
    animation: shine 2.6s ease-in-out infinite;
}
.bar-val { width: 58px; text-align: right; font-size: 0.85rem; font-weight: 800; color: #aaccff !important; }

@keyframes barGrow { from { width: 0; } }

.fcard { position: relative; background: #1a1a2e; border-radius: 10px; padding: 12px 16px; border-left: 3px solid #C62828; margin-bottom: 8px; display: flex; align-items: center; gap: 12px; transition: transform .18s ease, background .18s ease, border-color .18s ease; }
.fcard:hover { transform: translateX(5px); background: #20203a; border-left-color: #ff5252; }
.fcard-rank { flex-shrink: 0; width: 26px; height: 26px; border-radius: 50%; background: #0f0f1a; border: 1px solid #3a3a5c; color: #ffb020 !important; font-weight: 900; font-size: 0.8rem; display: flex; align-items: center; justify-content: center; }
.fcard-name { font-weight: 800; font-size: 0.98rem; color: #ffffff !important; }
.fcard-meta { font-size: 0.8rem; color: #aaa !important; margin-top: 2px; }
.fcard-year { background: #C62828; color: white !important; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 700; white-space: nowrap; }

.fcard-popup {
    position: absolute; top: 50%; right: calc(100% + 16px); width: 320px;
    background: #16162a; border: 1px solid #3a3a5c; border-radius: 16px; padding: 22px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.65);
    opacity: 0; visibility: hidden; pointer-events: none; z-index: 80;
    transform: translateY(-50%) translateX(8px);
    transition: opacity .2s ease, transform .2s ease, visibility .2s;
}
.fcard:hover .fcard-popup { opacity: 1; visibility: visible; transform: translateY(-50%) translateX(0); }
.fcard-popup-photo { width: 104px; height: 104px; border-radius: 50%; overflow: hidden; margin: 0 auto 14px; border: 3px solid #3a3a5c; box-shadow: 0 6px 18px rgba(0,0,0,0.4); }
.fcard-popup-photo img { width: 100%; height: 100%; object-fit: cover; }
.fcard-popup-name { text-align: center; font-weight: 900; font-size: 1.25rem; color: #fff !important; margin-bottom: 12px; }
.fcard-popup-row { font-size: 0.88rem; color: #ccc !important; padding: 7px 0; border-top: 1px solid #262640; display: flex; gap: 7px; align-items: center; line-height: 1.5; }
.fcard-popup-row:first-of-type { border-top: none; }

.stat-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #2a2a4a; transition: background .15s ease, padding-left .15s ease; border-radius: 6px; }
.stat-row:last-child { border-bottom: none; }
.stat-row:hover { background: #20203a; padding-left: 6px; }
.stat-lbl { font-size: 0.88rem; color: #aaa !important; }
.stat-lbl[title] { cursor: help; text-decoration: underline dotted #555; }
.stat-val { font-weight: 800; color: #ffffff !important; font-size: 0.9rem; }

.splash-ball { font-size: 4.6rem; display: inline-block; animation: floatBall 2.4s ease-in-out infinite; filter: drop-shadow(0 12px 20px rgba(0,0,0,0.5)); }
.splash-bar { width: 200px; height: 4px; background: #1e1e30; border-radius: 4px; overflow: hidden; position: relative; }
.splash-bar-fill { position: absolute; top: 0; left: -40%; width: 40%; height: 100%; background: linear-gradient(90deg, transparent, #4f8ef7, #ff5252, transparent); animation: splashBar 1.3s ease-in-out infinite; }
@keyframes splashBar { 0% { left: -40%; } 100% { left: 100%; } }

.how { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 18px; }
.how-num { background: #C62828; color: white !important; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 0.9rem; flex-shrink: 0; }
.how-txt { font-size: 0.9rem; color: #cccccc !important; line-height: 1.5; margin-top: 3px; }

.element-container:empty { display: none !important; min-height: 0 !important; }
[data-testid="column"] { overflow: visible; }
</style>
"""
