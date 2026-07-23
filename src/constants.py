"""StoveLens AI — 앱 전역 상수 (팀 데이터, 포지션 매핑, 툴팁, 스탯 라벨)."""

POS_KR = {
    "C": "포수", "1B": "1루수", "2B": "2루수", "3B": "3루수",
    "SS": "유격수", "OF": "외야수", "IF": "내야수", "DH": "지명타자",
    "SP": "선발투수", "SU": "셋업맨", "CL": "마무리",
}

POS_CODE = {"포수": "C", "외야수": "OF", "내야수": "IF", "1루수": "1B", "2루수": "2B", "3루수": "3B", "유격수": "SS"}

TEAM_EMOJI = {
    "KIA 타이거즈": "🐯", "삼성 라이온즈": "🦁", "LG 트윈스": "⚡",
    "두산 베어스": "🐻", "KT 위즈": "🧙", "SSG 랜더스": "🚀",
    "롯데 자이언츠": "🦅", "한화 이글스": "🔥", "NC 다이노스": "🦕",
    "키움 히어로즈": "🦸",
}

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
