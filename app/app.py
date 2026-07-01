"""
StoveLens AI — KBO FA 연봉 예측 서비스 (Streamlit)
"""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import requests
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

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"

# ── 상수 ──────────────────────────────────────────────────────────────────────
POS_KR = {
    "C": "포수", "1B": "1루수", "2B": "2루수", "3B": "3루수",
    "SS": "유격수", "OF": "외야수", "IF": "내야수", "DH": "지명타자",
    "SP": "선발투수", "SU": "셋업맨", "CL": "마무리",
}
TEAM_EMOJI = {
    "KIA 타이거즈": "🐯", "삼성 라이온즈": "🦁", "LG 트윈스": "⚡",
    "두산 베어스": "🐻", "KT 위즈": "🧙", "SSG 랜더스": "🚀",
    "롯데 자이언츠": "🦅", "한화 이글스": "🔥", "NC 다이노스": "🦕",
    "키움 히어로즈": "🦸",
}
POS_CODE = {"포수": "C", "외야수": "OF", "내야수": "IF", "1루수": "1B", "2루수": "2B", "3루수": "3B", "유격수": "SS"}
TEAM_DATA = {
    "LG":  {"rank":1,  "budget":"상",  "win_now":True,  "needs":["외야수","우완불펜"],        "factor":1.08,
             "reason":"2025년 1위 팀. 즉시 우승에 기여할 FA 선수에게 공격적 투자 기조. 잠실 대형 구장 + 풍부한 모기업 지원으로 고액 계약 여력 충분."},
    "KIA": {"rank":2,  "budget":"상",  "win_now":True,  "needs":["불펜","좌타자"],            "factor":1.06,
             "reason":"2025년 2위. 우승 경험 있는 강팀. 광주 KIA 챔피언스필드 배경의 강력한 팬덤과 모기업 지원. 보강 포인트 해당 포지션엔 적극적."},
    "두산": {"rank":3,  "budget":"중상", "win_now":True,  "needs":["선발","포수"],             "factor":1.04,
             "reason":"전통 강호로 우승 경쟁 중. 잠실 홈 이점으로 대형 FA에 관심. 단 최근 3년간 지출이 많아 선별적 투자 가능성."},
    "삼성": {"rank":4,  "budget":"중상", "win_now":False, "needs":["에이스","내야수"],          "factor":1.00,
             "reason":"삼성전자 모기업 지원은 크지만 최근 성적 부진으로 재건 과정. 장기 계약보다 실용적인 계약 선호 경향."},
    "SSG": {"rank":5,  "budget":"중",  "win_now":False, "needs":["외야 뎁스","중간계투"],      "factor":0.97,
             "reason":"신세계 대기업 구단이지만 최근 성적 하락으로 구단 운영 방향 재정비 중. 핵심 포지션 한정 투자 기조."},
    "한화": {"rank":6,  "budget":"상",  "win_now":True,  "needs":["경험 있는 선수","클로저"],   "factor":1.05,
             "reason":"최근 대형 FA에 적극 투자하는 기조로 전환. 대전 연고 팬덤과 모기업 지원으로 오버페이 사례 다수. 우승 의지 강함."},
    "NC":  {"rank":7,  "budget":"중",  "win_now":False, "needs":["전력 재건"],                "factor":0.94,
             "reason":"전력 재건 중인 구단. 대형 FA보다 중형 계약 + 육성 전략. 해당 선수의 포지션이 급하지 않으면 제시가 보수적."},
    "KT":  {"rank":8,  "budget":"중",  "win_now":False, "needs":["균형 보강"],                "factor":0.95,
             "reason":"KT위즈는 고른 전력 보강 전략. 특정 포지션 절박함 낮아 협상력에서 우위 가져가는 편. 중형 계약 선호."},
    "롯데": {"rank":9,  "budget":"중",  "win_now":False, "needs":["투수력","포수"],             "factor":0.92,
             "reason":"사직야구장 대형 팬덤 보유하지만 구단 재정 여건상 최상위 FA에는 한계. 해당 선수 포지션 보강 급하면 프리미엄 지급 가능."},
    "키움": {"rank":10, "budget":"하",  "win_now":False, "needs":["재건","선발"],               "factor":0.85,
             "reason":"전통적으로 FA 시장에 소극적. 육성 위주 운영 철학. 고척돔 특성상 투수 유리 환경. 외부 FA 대형 계약은 이례적."},
}
TEAM_LOGO = {
    "LG":  "https://lgtwins.com/assets/images/common/logo.png",
    "두산": "https://www.doosanbears.com/common/img/logo.png",
    "KIA": "https://tigers.co.kr/assets/images/common/logo_tigers.png",
    "삼성": "https://www.samsunglions.com/common/img/lions_logo.png",
    "SSG": "https://www.ssglanders.com/assets/images/logo.png",
    "한화": "https://www.hanwhaeagles.co.kr/images/common/logo_pc.png",
    "NC":  "https://www.ncdinos.com/assets/images/common/logo_white.png",
    "KT":  "https://www.ktwiz.co.kr/images/logo.png",
    "롯데": "https://www.giantsclub.com/images/common/logo.png",
    "키움": "https://heroesbaseball.co.kr/images/common/logo.png",
}
TOOLTIPS = {
    "팀 기여도 (WAR)": "이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 중요한 선수.",
    "출루+장타 (OPS)": "출루율 + 장타율. 타자의 전반적인 공격력을 나타내는 지표.",
    "방어율 (ERA)": "9이닝당 자책점 평균. 낮을수록 좋은 투수.",
    "이닝 / 시즌": "투수가 한 시즌 동안 소화한 이닝 수. 많을수록 신뢰받는 선발.",
    "세이브 / 시즌": "승리를 지킨 횟수. 마무리 투수의 핵심 지표.",
    "홀드 / 시즌": "이어받은 리드를 지키고 다음 투수에게 넘긴 횟수. 셋업맨 지표.",
    "홈런 / 시즌": "한 시즌 평균 홈런 수. 장타력을 나타내는 지표.",
    "타점 / 시즌": "한 시즌 평균 타점. 득점 기회에서의 클러치 능력.",
    "도루 / 시즌": "한 시즌 평균 도루 수. 주루 능력과 스피드 지표.",
}
STAT_LABEL_MAP = {
    "war_3yr_avg":       ("팀 기여도 (WAR)",       "이 선수가 없었다면 팀이 잃었을 승리 수"),
    "war_3yr_sum":       ("3년 누적 기여도",        "3년간 팀에 기여한 총 승리 수"),
    "ops_3yr_avg":       ("공격 종합 지수 (OPS)",   "출루율+장타율. 높을수록 강한 타자"),
    "ops_last_year":     ("작년 공격 지수",         "직전 시즌 OPS"),
    "wrc_plus_3yr_avg":  ("조정 득점 기여 (wRC+)",  "100이 평균. 높을수록 평균 이상 타자"),
    "hr_3yr_avg":        ("홈런 / 시즌",            "3년 평균 홈런 수"),
    "rbi_3yr_avg":       ("타점 / 시즌",            "3년 평균 타점"),
    "sb_3yr_avg":        ("도루 / 시즌",            "3년 평균 도루 수"),
    "woba_3yr_avg":      ("가중 출루율 (wOBA)",     "출루·장타를 가중 계산한 종합 지표"),
    "war_last_year":     ("작년 기여도",            "직전 시즌 WAR"),
    "age_at_fa":         ("FA 시점 나이",           "나이가 많을수록 계약 규모 감소 경향"),
    "games_3yr_avg":     ("평균 출장 경기",         "3년 평균 출장 수. 부상 없이 꾸준할수록 높음"),
    "era_3yr_avg":       ("방어율 (ERA)",            "9이닝당 자책점. 낮을수록 좋음"),
    "era_last_year":     ("작년 방어율",            "직전 시즌 ERA"),
    "whip_3yr_avg":      ("WHIP",                   "이닝당 출루 허용 수. 낮을수록 좋음"),
    "innings_3yr_avg":   ("평균 이닝 소화",         "3년 평균 이닝. 높을수록 꾸준한 투수"),
    "hold_3yr_avg":      ("홀드 / 시즌",            "중간계투 셋업맨의 핵심 지표"),
    "save_3yr_avg":      ("세이브 / 시즌",          "마무리 투수의 핵심 지표"),
    "win_3yr_avg":       ("승리 / 시즌",            "3년 평균 승리 수"),
    "star_score":        ("스타성 지수",            "MVP·골든글러브·국가대표 경력 반영"),
    "star_x_war":        ("스타 × 기여도",          "스타성과 WAR의 복합 지표"),
    "war_sum_sq":        ("기여도 가속 지표",        "WAR 합산의 제곱. 고기여 선수를 강조"),
    "market_level":      ("시장 연봉 수준",         "해당 FA 시즌의 평균 연봉 수준"),
    "prime_years_left":  ("전성기 잔여",            "35세 기준 남은 전성기 년수"),
}
LOWER_IS_BETTER = {"era_3yr_avg", "era_last_year", "whip_3yr_avg", "whip_last_year", "age_at_fa"}
STAT_INFO = {
    "war_3yr_avg":      {"icon":"🏆","label":"팀 기여도(WAR) 3년 평균","unit":"",     "desc":"이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 없어서는 안 될 핵심 선수입니다.","good":"high"},
    "war_3yr_sum":      {"icon":"🏆","label":"팀 기여도(WAR) 3년 합계","unit":"",     "desc":"3년치 WAR을 합산한 누적 기여도. 꾸준함과 임팩트를 동시에 반영합니다.","good":"high"},
    "ops_3yr_avg":      {"icon":"⚡","label":"OPS(공격 종합지수) 3년 평균","unit":"", "desc":"출루율+장타율. 0.900 이상이면 리그 최상위 타자, 0.700 미만이면 평균 이하입니다.","good":"high"},
    "ops_last_year":    {"icon":"📈","label":"직전 시즌 OPS","unit":"",               "desc":"FA 직전 마지막 시즌의 공격 종합 지수. 최근 타격 상태를 직접 반영합니다.","good":"high"},
    "war_last_year":    {"icon":"📈","label":"직전 시즌 기여도(WAR)","unit":"",       "desc":"FA 직전 마지막 시즌의 팀 기여도. 선수의 현재 컨디션을 가장 직접적으로 반영합니다.","good":"high"},
    "hr_3yr_avg":       {"icon":"💥","label":"홈런 3년 평균","unit":"개",             "desc":"장타력을 나타내는 핵심 지표. 팬들이 가장 좋아하는 스타성 스탯이기도 합니다.","good":"high"},
    "rbi_3yr_avg":      {"icon":"🎯","label":"타점 3년 평균","unit":"점",             "desc":"득점 기회에서 팀에 기여한 점수. 클러치 능력과 중심타선 역할을 나타냅니다.","good":"high"},
    "woba_3yr_avg":     {"icon":"📊","label":"wOBA(가중출루율) 3년 평균","unit":"",   "desc":"단타·2루타·홈런 등 타격 결과에 가중치를 달리 부여한 종합 타격 지표입니다.","good":"high"},
    "wrc_plus_3yr_avg": {"icon":"🌟","label":"wRC+(조정 득점 창출) 3년 평균","unit":"","desc":"100이 리그 평균. 120이면 평균보다 20% 더 뛰어난 타자란 뜻입니다.","good":"high"},
    "sb_3yr_avg":       {"icon":"🏃","label":"도루 3년 평균","unit":"개",             "desc":"발빠름과 주루 능력. 외야수·내야수 고속형 선수의 핵심 매력 포인트입니다.","good":"high"},
    "age_at_fa":        {"icon":"🗓️","label":"FA 신청 나이","unit":"세",              "desc":"어릴수록 장기 계약 가치가 높습니다. 30세 이하면 장기 계약, 35세 이상이면 단기 계약이 일반적입니다.","good":"low"},
    "games_3yr_avg":    {"icon":"⚾","label":"평균 경기 수(3년)","unit":"경기",       "desc":"1시즌 144경기 중 평균 출전 수. 높을수록 부상 없이 꾸준히 뛴 내구성을 의미합니다.","good":"high"},
    "era_3yr_avg":      {"icon":"🎯","label":"평균자책점(ERA) 3년 평균","unit":"",    "desc":"9이닝당 실점 평균. 낮을수록 좋습니다. 3.00 미만이면 에이스급, 5.00 이상이면 불안정합니다.","good":"low"},
    "era_last_year":    {"icon":"📉","label":"직전 시즌 방어율(ERA)","unit":"",        "desc":"FA 직전 마지막 시즌 ERA. 최근 투구 상태를 가장 직접적으로 반영합니다.","good":"low"},
    "whip_3yr_avg":     {"icon":"🚫","label":"WHIP(이닝당 출루 허용) 3년 평균","unit":"","desc":"이닝당 출루 허용 수. 1.00 미만이면 최상위 제구력, 1.40 이상이면 위험 수준입니다.","good":"low"},
    "innings_3yr_avg":  {"icon":"⏱️","label":"평균 이닝(3년)","unit":"이닝",          "desc":"선발이면 높을수록 에이스. 마무리·중간계투는 상대적으로 낮지만 역할이 다릅니다.","good":"high"},
    "hold_3yr_avg":     {"icon":"🛡️","label":"홀드 3년 평균","unit":"개",              "desc":"중간계투(셋업)의 핵심 지표. 리드 상황에서 경기를 안정적으로 넘기는 능력입니다.","good":"high"},
    "save_3yr_avg":     {"icon":"🔒","label":"세이브 3년 평균","unit":"개",            "desc":"마무리 투수의 핵심 지표. 20세이브 이상이면 리그 정상급 마무리입니다.","good":"high"},
    "star_score":       {"icon":"⭐","label":"스타성 지수","unit":"점",                "desc":"MVP·골든글러브·국가대표 경력을 점수화한 지표. 스타성이 높을수록 비성적 프리미엄이 붙습니다.","good":"high"},
}

# ── CSS ───────────────────────────────────────────────────────────────────────
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

# ── 피처 엔지니어링 ────────────────────────────────────────────────────────────
def engineer_hitter_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_level"] = df.groupby("fa_year")["annual_avg_salary"].transform("median")
    df["age_squared"] = df["age_at_fa"] ** 2
    df["prime_years_left"] = (35 - df["age_at_fa"]).clip(0, 10)
    df["star_x_war"] = df["star_score"] * df["war_3yr_sum"]
    df["star_x_ops"] = df["star_score"] * df["ops_3yr_avg"]
    df["war_sum_sq"] = df["war_3yr_sum"] ** 2
    pos_map = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5, "IF": 6, "DH": 7}
    df["position_enc"] = df["position"].map(pos_map).fillna(6).astype(int)
    for col in ["war_3yr_avg", "war_3yr_sum", "ops_3yr_avg", "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg"]:
        df[f"{col}_all_pct"] = df[col].rank(pct=True)
        df[f"{col}_pos_pct"] = df.groupby("position")[col].rank(pct=True)
    return df


def engineer_pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_level"] = df.groupby("fa_year")["annual_avg_salary"].transform("median")
    df["age_squared"] = df["age_at_fa"] ** 2
    df["prime_left"] = (35 - df["age_at_fa"]).clip(0, 10)
    df["war_per_game"] = df["war_3yr_sum"] / (df["games_3yr_avg"] * 3 + 1)
    df["star_x_war"] = df["star_score"] * df["war_3yr_sum"]
    df["role_SP"] = (df["pitcher_role"] == "SP").astype(int)
    df["role_SU"] = (df["pitcher_role"] == "SU").astype(int)
    df["role_CL"] = (df["pitcher_role"] == "CL").astype(int)
    return df


# ── 데이터 / 모델 ──────────────────────────────────────────────────────────────
def get_future_fa_candidates(h_season_df, p_season_df):
    candidates = []
    for df, player_type in [(h_season_df, "hitter"), (p_season_df, "pitcher")]:
        year_col = "collect_year" if "collect_year" in df.columns else "year"
        team_col = "teamName" if "teamName" in df.columns else "team"

        debut = df.groupby("playerName")[year_col].min().reset_index()
        debut.columns = ["player_name", "debut_year"]
        debut["fa_year_expected"] = debut["debut_year"] + 10
        fa = debut[debut["fa_year_expected"].between(2027, 2030)].copy()
        fa["player_type"] = player_type

        season_cnt = df.groupby("playerName")[year_col].count()
        fa["_cnt"] = fa["player_name"].map(season_cnt).fillna(0)

        latest = (df.sort_values(year_col)
                    .groupby("playerName").last()
                    .reset_index()[["playerName", team_col]])
        latest.columns = ["player_name", "team_2026"]
        fa = fa.merge(latest, on="player_name", how="left")
        candidates.append(fa)

    combined = pd.concat(candidates, ignore_index=True)
    combined = (combined.sort_values("_cnt", ascending=False)
                        .drop_duplicates("player_name", keep="first")
                        .drop(columns="_cnt")
                        .reset_index(drop=True))
    combined["position"] = "외야수"
    combined["fa_grade"] = "B"
    return combined


@st.cache_data
def load_data():
    h = pd.read_csv(DATA / "hitter_training_v7.csv", encoding="utf-8-sig")
    p = pd.read_csv(DATA / "pitcher_training_v7.csv", encoding="utf-8-sig")
    teams = pd.read_csv(DATA / "teams.csv", encoding="utf-8-sig")
    pn = pd.read_csv(DATA / "position_need.csv", encoding="utf-8-sig")
    fa = pd.read_csv(DATA / "fa_contracts_v3.csv", encoding="utf-8-sig")
    h_stats = pd.read_csv(DATA / "hitter_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
    p_stats = pd.read_csv(DATA / "pitcher_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
    future = get_future_fa_candidates(h_stats, p_stats)
    h = engineer_hitter_features(h)
    p = engineer_pitcher_features(p)
    return h, p, teams, pn, future, fa


@st.cache_resource
def load_models():
    hm = joblib.load(MODELS / "hitter_model_meta.pkl")
    hx = joblib.load(MODELS / "hitter_xgb_model.pkl")
    hl = joblib.load(MODELS / "hitter_lgb_model.pkl")
    pm = joblib.load(MODELS / "pitcher_model_meta.pkl")
    px = joblib.load(MODELS / "pitcher_xgb_model.pkl")
    pr = joblib.load(MODELS / "pitcher_rf_model.pkl")
    return hm, hx, hl, pm, px, pr


@st.cache_data
def load_season_stats():
    h_stats = pd.read_csv(DATA / "hitter_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
    p_stats = pd.read_csv(DATA / "pitcher_season_stats_2015_2026_v2.csv", encoding="utf-8-sig")
    h_star  = pd.read_csv(DATA / "star_features_hitter.csv", encoding="utf-8-sig")
    p_star  = pd.read_csv(DATA / "star_features_pitcher.csv", encoding="utf-8-sig")
    return h_stats, p_stats, h_star, p_star


# ── 예측 ──────────────────────────────────────────────────────────────────────
def predict_h(row, xgb_m, lgb_m, meta):
    X = row[meta["features"]].values.reshape(1, -1)
    lp = meta["xgb_weight"] * xgb_m.predict(X)[0] + meta["lgb_weight"] * lgb_m.predict(X)[0]
    return float(np.expm1(lp))


def predict_p(row, xgb_m, rf_m, meta):
    X = row[meta["features"]].values.reshape(1, -1)
    lp = meta["xgb_weight"] * xgb_m.predict(X)[0] + meta["rf_weight"] * rf_m.predict(X)[0]
    return float(np.expm1(lp))


# ── 구단별 제시가 ──────────────────────────────────────────────────────────────
def team_offers(base, position, teams_df, pn_df):
    pn = pn_df[pn_df["position"] == position][["team_name", "need_score"]]
    m = teams_df.merge(pn, on="team_name", how="left")
    m["need_score"] = m["need_score"].fillna(0.5)
    m["offer"] = (base * (1 + m["need_score"] * 0.20 + m["win_now_score"] * 0.15 + m["cap_space_score"] * 0.10)).round(2)
    return m.sort_values("offer", ascending=False).reset_index(drop=True)


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


# ── 미래 FA 피처 빌더 ───────────────────────────────────────────────────────────
def build_future_hitter_row(player_name, fr, h_stats_df, h_star_df, h_df):
    rows = h_stats_df[h_stats_df["playerName"] == player_name].sort_values("collect_year", ascending=False)
    if rows.empty:
        return None
    year_col_s = "collect_year" if "collect_year" in rows.columns else "year"
    base_yr = int(rows[year_col_s].max())
    target_yrs = [base_yr - i for i in range(3)]
    recent3 = rows[rows[year_col_s].isin(target_yrs)]
    if recent3.empty:
        recent3 = rows.head(3)
    n_yrs = 3
    last   = rows.iloc[0]
    avg3   = recent3.sum(numeric_only=True).fillna(0) / n_yrs
    active_seasons_h = len(recent3)

    existing = h_df[h_df["player_name"] == player_name]
    if not existing.empty:
        old = existing.sort_values("fa_year").iloc[-1]
        age_at_fa = int(old["age_at_fa"]) + (int(fr["fa_year_expected"]) - int(old["fa_year"]))
    else:
        age_at_fa = 32

    star_row = h_star_df[h_star_df["player_name"] == player_name]
    star = star_row.iloc[-1] if not star_row.empty else pd.Series({
        "mvp_count": 0, "golden_glove_count": 0, "allstar_count": 0,
        "national_team": 0, "postseason_experience": 0, "star_score": 0,
    })

    pos_code   = POS_CODE.get(fr.get("position", "외야수"), "OF")
    enc_map    = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5, "IF": 6, "DH": 7}
    war_sum    = float(recent3["hitterWar"].fillna(0).sum())
    war_avg    = war_sum / n_yrs
    ops_avg    = float(avg3.get("hitterOps", 0))
    hr_avg     = float(avg3.get("hitterHr", 0))
    rbi_avg    = float(avg3.get("hitterRbi", 0))
    woba_avg   = float(avg3.get("hitterWoba", 0))
    star_score = int(star.get("star_score", 0))

    recent_market = h_df[h_df["fa_year"] == h_df["fa_year"].max()]["annual_avg_salary"].median()
    if pd.isna(recent_market):
        recent_market = float(h_df["annual_avg_salary"].median())

    ref = h_df[["position", "war_3yr_avg", "war_3yr_sum", "ops_3yr_avg",
                 "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg"]].copy()
    ref["position"] = ref["position"].fillna("OF")
    new_val = {"position": pos_code, "war_3yr_avg": war_avg, "war_3yr_sum": war_sum,
               "ops_3yr_avg": ops_avg, "hr_3yr_avg": hr_avg, "rbi_3yr_avg": rbi_avg, "woba_3yr_avg": woba_avg}
    ref = pd.concat([ref, pd.DataFrame([new_val])], ignore_index=True)
    pct = {}
    for col in ["war_3yr_avg", "war_3yr_sum", "ops_3yr_avg", "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg"]:
        pct[f"{col}_all_pct"] = float(ref[col].rank(pct=True).iloc[-1])
        pct[f"{col}_pos_pct"] = float(ref.groupby("position")[col].rank(pct=True).iloc[-1])

    return pd.Series({
        "fa_year": int(fr["fa_year_expected"]),
        "age_at_fa": age_at_fa,
        "games_3yr_avg":     float(avg3.get("hitterGameCount", 0)),
        "ab_3yr_avg":        float(avg3.get("hitterAb", 0)),
        "avg_3yr_avg":       float(avg3.get("hitterHra", 0)),
        "obp_3yr_avg":       float(avg3.get("hitterObp", 0)),
        "slg_3yr_avg":       float(avg3.get("hitterSlg", 0)),
        "ops_3yr_avg":       ops_avg,
        "isop_3yr_avg":      float(avg3.get("hitterIsop", 0)),
        "babip_3yr_avg":     float(avg3.get("hitterBabip", 0)),
        "woba_3yr_avg":      woba_avg,
        "wrc_plus_3yr_avg":  float(avg3.get("hitterWrcPlus", 0)),
        "hr_3yr_avg":        hr_avg,
        "rbi_3yr_avg":       rbi_avg,
        "run_3yr_avg":       float(avg3.get("hitterRun", 0)),
        "hit_3yr_avg":       float(avg3.get("hitterHit", 0)),
        "double_3yr_avg":    float(avg3.get("hitterH2", 0)),
        "triple_3yr_avg":    float(avg3.get("hitterH3", 0)),
        "bb_3yr_avg":        float(avg3.get("hitterBb", 0)),
        "hp_3yr_avg":        float(avg3.get("hitterHp", 0)),
        "kk_3yr_avg":        float(avg3.get("hitterKk", 0)),
        "sb_3yr_avg":        float(avg3.get("hitterSb", 0)),
        "cs_3yr_avg":        float(avg3.get("hitterCs", 0)),
        "gd_3yr_avg":        float(avg3.get("hitterGd", 0)),
        "wpa_3yr_avg":       float(avg3.get("hitterWpa", 0)),
        "war_3yr_avg":       war_avg,
        "war_3yr_sum":       war_sum,
        "war_last_year":     float(last["hitterWar"]) if not pd.isna(last["hitterWar"]) else 0.0,
        "ops_last_year":     float(last["hitterOps"]) if not pd.isna(last["hitterOps"]) else 0.0,
        "wrc_plus_last_year": float(last["hitterWrcPlus"]) if not pd.isna(last["hitterWrcPlus"]) else 0.0,
        "mvp_count":          int(star.get("mvp_count", 0)),
        "golden_glove_count": int(star.get("golden_glove_count", 0)),
        "allstar_count":      int(star.get("allstar_count", 0)),
        "national_team":      int(star.get("national_team", 0)),
        "postseason_experience": int(star.get("postseason_experience", 0)),
        "star_score":         star_score,
        **pct,
        "market_level":     float(recent_market),
        "age_squared":      age_at_fa ** 2,
        "prime_years_left": max(0, min(10, 35 - age_at_fa)),
        "star_x_war":       star_score * war_sum,
        "star_x_ops":       star_score * ops_avg,
        "war_sum_sq":       war_sum ** 2,
        "position_enc":     enc_map.get(pos_code, 5),
        "active_seasons":   active_seasons_h,
    })


def build_future_pitcher_row(player_name, fr, p_stats_df, p_star_df, p_df):
    rows = p_stats_df[p_stats_df["playerName"] == player_name].sort_values("collect_year", ascending=False)
    if rows.empty:
        return None
    year_col_s = "collect_year" if "collect_year" in rows.columns else "year"
    base_yr = int(rows[year_col_s].max())
    target_yrs = [base_yr - i for i in range(3)]
    recent3 = rows[rows[year_col_s].isin(target_yrs)]
    if recent3.empty:
        recent3 = rows.head(3)
    n_yrs = 3
    last   = rows.iloc[0]
    avg3   = recent3.sum(numeric_only=True).fillna(0) / n_yrs
    active_seasons_p = len(recent3)

    save_avg = float(avg3.get("pitcherSave", 0))
    hold_avg = float(avg3.get("pitcherHold", 0))
    role = "CL" if save_avg >= 10 else ("SU" if hold_avg >= 10 else "SP")

    existing = p_df[p_df["player_name"] == player_name]
    if not existing.empty:
        old = existing.sort_values("fa_year").iloc[-1]
        age_at_fa = int(old["age_at_fa"]) + (int(fr["fa_year_expected"]) - int(old["fa_year"]))
    else:
        age_at_fa = 32

    star_row = p_star_df[p_star_df["player_name"] == player_name]
    star = star_row.iloc[-1] if not star_row.empty else pd.Series({
        "mvp_count": 0, "golden_glove_count": 0, "allstar_count": 0,
        "national_team": 0, "postseason_experience": 0, "star_score": 0,
    })

    star_score   = int(star.get("star_score", 0))
    war_sum      = float(recent3["pitcherWar"].fillna(0).sum())
    war_avg      = war_sum / n_yrs
    games_avg    = float(avg3.get("pitcherGameCount", 0))

    recent_market = p_df[p_df["fa_year"] == p_df["fa_year"].max()]["annual_avg_salary"].median()
    if pd.isna(recent_market):
        recent_market = float(p_df["annual_avg_salary"].median())

    return pd.Series({
        "age_at_fa":           age_at_fa,
        "games_3yr_avg":       games_avg,
        "innings_3yr_avg":     float(avg3.get("pitcherInning", 0)),
        "era_3yr_avg":         float(avg3.get("pitcherEra", 0)),
        "whip_3yr_avg":        float(avg3.get("pitcherWhip", 0)),
        "win_3yr_avg":         float(avg3.get("pitcherWin", 0)),
        "lose_3yr_avg":        float(avg3.get("pitcherLose", 0)),
        "save_3yr_avg":        save_avg,
        "hold_3yr_avg":        hold_avg,
        "hit_allowed_3yr_avg": float(avg3.get("pitcherHit", 0)),
        "war_3yr_avg":         war_avg,
        "war_3yr_sum":         war_sum,
        "war_last_year":       float(last["pitcherWar"]) if not pd.isna(last["pitcherWar"]) else 0.0,
        "era_last_year":       float(last["pitcherEra"]) if not pd.isna(last["pitcherEra"]) else 0.0,
        "whip_last_year":      float(last["pitcherWhip"]) if not pd.isna(last["pitcherWhip"]) else 0.0,
        "mvp_count":           int(star.get("mvp_count", 0)),
        "golden_glove_count":  int(star.get("golden_glove_count", 0)),
        "allstar_count":       int(star.get("allstar_count", 0)),
        "national_team":       int(star.get("national_team", 0)),
        "postseason_experience": int(star.get("postseason_experience", 0)),
        "star_score":          star_score,
        "market_level":        float(recent_market),
        "age_squared":         age_at_fa ** 2,
        "prime_left":          max(0, min(10, 35 - age_at_fa)),
        "war_per_game":        war_sum / (games_avg * 3 + 1),
        "star_x_war":          star_score * war_sum,
        "role_SP":             1 if role == "SP" else 0,
        "role_SU":             1 if role == "SU" else 0,
        "role_CL":             1 if role == "CL" else 0,
        "pitcher_role":        role,
        "active_seasons":      active_seasons_p,
    })


# ── UI 컴포넌트 ────────────────────────────────────────────────────────────────
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
        # TEAM_DATA lookup by short name contained in full team name
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


def _render_team_analysis(info, offer, base, team_name, player_pos=None):
    """구단 제시가 근거 텍스트 + 비교 바 차트."""
    try:
        # ① 팀 상황 요약
        reason = info.get("reason", "") or info.get("need", "")
        if reason:
            st.markdown(f"📋 {reason}")

        # ② 포지션 needs 매칭
        if player_pos:
            pos_kr = POS_KR.get(player_pos, player_pos)
            needs_list = info.get("needs", [])
            if any(pos_kr in n or n in pos_kr for n in needs_list):
                st.markdown(f"✅ 이 선수의 **{pos_kr}** 포지션은 해당 구단의 핵심 보강 포인트입니다. 경쟁 입찰 가능성 높음.")

        # ③ 우승 의지
        if info.get("win_now"):
            st.markdown("🔥 지금 당장 우승을 노리는 구단입니다. 즉전감 선수에게 공격적으로 접근합니다.")

        # ④ 제시가 계산 근거
        factor = info.get("factor", 1.0)
        st.markdown(f"**제시가 계산:** 기준 연봉 {base:.1f}억 × 구단 보정 계수 {factor:.2f} = **{offer:.1f}억**")
        st.caption("보정 계수: 1.0 기준 / 우승 의지·포지션 필요도·예산 규모 반영")

        # ⑤ 가로 바 차트
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


def render_search(is_hitter, df, teams_df, pn_df, future_df, hm, hx, hl, pm, px, pr, fa_df):
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
            age   = int(row["age_at_fa"]) + (2026 - int(row["fa_year"]))
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

            predicted = predict_h(future_pred_row, hx, hl, hm) if is_hitter else predict_p(future_pred_row, px, pr, pm)
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

        predicted = predict_h(row, hx, hl, hm) if is_hitter else predict_p(row, px, pr, pm)
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


# ── 상세 성적 페이지 ───────────────────────────────────────────────────────────
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


# ── 메인 ──────────────────────────────────────────────────────────────────────
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
                h_df, p_df, teams, pn, future, fa_df = load_data()
                hm, hx, hl, pm, px, pr = load_models()
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
