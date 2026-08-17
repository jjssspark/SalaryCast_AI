"""StoveLens AI — 이 선수의 연봉을 무엇이 밀어올리고 무엇이 깎았는지.

호출 위치: src/ui/player.py ('연봉을 결정한 것' 패널)
데이터 파일을 읽지 않는다. data_loader가 준 모델 번들과 한 행짜리 입력만 받는다.

모델 전체의 피처 중요도를 쓰면 누구를 검색하든 같은 순서가 나온다.
선수마다 다른 근거를 내야 하므로 SHAP으로 이 한 건의 기여도를 구한다.
번들에 들어 있는 트리 모델 쪽에만 건다. 트리라야 정확한 기여도가 나오고,
같이 블렌드된 Ridge는 선형이라 순서를 크게 바꾸지 않는다.

피처는 57개지만 화면에는 4개만 보여준다. war_3yr_avg / war_3yr_sum /
war_last_year / star_x_war는 사람 입장에서 전부 '팀 기여도' 하나라
같은 뜻의 피처끼리 묶어 합산한 뒤 순위를 매긴다.
"""

from __future__ import annotations

import numpy as np

from src.predict import to_frame

# 화면에 보일 요인 그룹.
#   members — 이 그룹으로 합산할 모델 피처 이름
#   stat    — 리그 비교에 쓸 시즌 스탯 컬럼 (없으면 학습 표본 기준으로 대체)
#   source  — 값을 가져올 모델 입력 컬럼
#   tip     — 물음표 툴팁. None이면 툴팁 없음
HITTER_GROUPS = [
    {
        "key": "war", "label": "팀 기여도", "stat": "war",
        "source": "war_3yr_avg", "fmt": "{:.1f}", "unit": "/시즌",
        "members": ["war_3yr_avg", "war_3yr_sum", "war_last_year", "war_sum_sq",
                    "war_seasons_valid", "star_x_war",
                    "war_3yr_avg_all_pct", "war_3yr_avg_pos_pct",
                    "war_3yr_sum_all_pct", "war_3yr_sum_pos_pct"],
        "tip": "이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 중요한 선수.",
        "note": "WAR · 최근 3시즌 평균",
    },
    {
        "key": "bat", "label": "공격 생산력", "stat": "ops",
        "source": "ops_3yr_avg", "fmt": "{:.3f}", "unit": "OPS",
        "members": ["ops_3yr_avg", "ops_last_year", "woba_3yr_avg",
                    "wrc_plus_3yr_avg", "wrc_plus_last_year", "avg_3yr_avg",
                    "obp_3yr_avg", "slg_3yr_avg", "isop_3yr_avg", "babip_3yr_avg",
                    "star_x_ops", "hit_3yr_avg", "run_3yr_avg", "double_3yr_avg",
                    "triple_3yr_avg", "bb_3yr_avg", "hp_3yr_avg", "kk_3yr_avg",
                    "ops_3yr_avg_all_pct", "ops_3yr_avg_pos_pct",
                    "woba_3yr_avg_all_pct", "woba_3yr_avg_pos_pct",
                    "wrc_plus_3yr_avg_all_pct", "wrc_plus_3yr_avg_pos_pct"],
        "tip": "출루율 + 장타율. 타자의 전반적인 공격력을 나타내는 지표.",
        "note": "OPS · 최근 3시즌 평균",
    },
    {
        "key": "power", "label": "장타력", "stat": "hr",
        "source": "hr_3yr_avg", "fmt": "{:.1f}", "unit": "홈런/시즌",
        "members": ["hr_3yr_avg", "hr_3yr_avg_all_pct", "hr_3yr_avg_pos_pct"],
        "tip": "시즌당 홈런 수. 장타력을 가장 직관적으로 보여주는 지표.",
        "note": "최근 3시즌 평균",
    },
    {
        "key": "rbi", "label": "해결 능력", "stat": "rbi",
        "source": "rbi_3yr_avg", "fmt": "{:.1f}", "unit": "타점/시즌",
        "members": ["rbi_3yr_avg", "rbi_3yr_avg_all_pct", "rbi_3yr_avg_pos_pct"],
        "tip": "자신의 타격으로 홈에 불러들인 주자 수. 득점권 해결 능력을 보여줌.",
        "note": "최근 3시즌 평균",
    },
    {
        "key": "speed", "label": "주루", "stat": "sb",
        "source": "sb_3yr_avg", "fmt": "{:.1f}", "unit": "도루/시즌",
        "members": ["sb_3yr_avg"],
        "tip": "시즌당 도루 수. 발과 주루 판단을 함께 보여주는 지표.",
        "note": "최근 3시즌 평균",
    },
    {
        "key": "durability", "label": "출장 내구성", "stat": None,
        "source": "games_3yr_avg", "fmt": "{:.0f}", "unit": "경기/시즌",
        "members": ["games_3yr_avg", "ab_3yr_avg", "active_seasons", "wpa_3yr_avg"],
        "tip": None,
        "note": "부상 없이 꾸준히 뛰었는지",
    },
]

PITCHER_GROUPS = [
    {
        "key": "war", "label": "팀 기여도", "stat": "war",
        "source": "war_3yr_avg", "fmt": "{:.1f}", "unit": "/시즌",
        "members": ["war_3yr_avg", "war_3yr_sum", "war_last_year",
                    "war_seasons_valid", "star_x_war",
                    "war_3yr_sum_all_pct", "war_3yr_sum_pos_pct"],
        "tip": "이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 중요한 선수.",
        "note": "WAR · 최근 3시즌 평균",
    },
    {
        "key": "prevent", "label": "실점 억제", "stat": "era",
        "source": "era_3yr_avg", "fmt": "{:.2f}", "unit": "방어율",
        "members": ["era_3yr_avg", "era_last_year", "whip_3yr_avg",
                    "whip_last_year", "whip_era_ratio", "hit_per_inn",
                    "hit_allowed_3yr_avg",
                    "era_3yr_avg_all_pct", "era_3yr_avg_pos_pct"],
        "tip": "9이닝당 자책점 평균. 낮을수록 좋은 투수.",
        "note": "ERA · 최근 3시즌 평균 · 낮을수록 좋음",
    },
    {
        "key": "innings", "label": "이닝 소화력", "stat": "innings",
        "source": "innings_3yr_avg", "fmt": "{:.0f}", "unit": "이닝/시즌",
        "members": ["innings_3yr_avg", "role_x_inn", "ip_per_game_3yr_avg",
                    "qs_3yr_avg", "games_3yr_avg", "active_seasons", "wpa_3yr_avg",
                    "innings_3yr_avg_all_pct", "innings_3yr_avg_pos_pct"],
        "tip": "한 시즌 소화한 이닝. 많을수록 팀이 길게 맡긴 투수.",
        "note": "최근 3시즌 평균",
    },
    {
        "key": "strikeout", "label": "탈삼진", "stat": "so",
        "source": "so_3yr_avg", "fmt": "{:.0f}", "unit": "삼진/시즌",
        "members": ["so_3yr_avg", "k9_3yr_avg", "k_bb_3yr_avg"],
        "tip": "수비 도움 없이 스스로 아웃을 잡아내는 능력.",
        "note": "최근 3시즌 평균",
    },
    {
        "key": "control", "label": "제구", "stat": None,
        "source": "bb9_3yr_avg", "fmt": "{:.2f}", "unit": "볼넷/9이닝",
        "members": ["bb_3yr_avg", "bb9_3yr_avg"],
        "tip": "9이닝당 볼넷. 낮을수록 안정적인 제구.",
        "note": "낮을수록 좋음",
    },
    {
        "key": "closer", "label": "마무리 역할", "stat": "save",
        "source": "save_3yr_avg", "fmt": "{:.0f}", "unit": "세이브/시즌",
        "members": ["save_3yr_avg", "role_x_save"],
        "tip": "승리를 지키고 경기를 끝낸 횟수. 마무리 투수의 핵심 지표.",
        "note": "최근 3시즌 평균",
    },
    {
        "key": "setup", "label": "셋업 역할", "stat": "hold",
        "source": "hold_3yr_avg", "fmt": "{:.0f}", "unit": "홀드/시즌",
        "members": ["hold_3yr_avg", "role_x_hold"],
        "tip": "이어받은 리드를 지키고 다음 투수에게 넘긴 횟수.",
        "note": "최근 3시즌 평균",
    },
    {
        "key": "win", "label": "승리 기여", "stat": None,
        "source": "win_3yr_avg", "fmt": "{:.1f}", "unit": "승/시즌",
        "members": ["win_3yr_avg", "lose_3yr_avg", "win_rate"],
        "tip": None,
        "note": "최근 3시즌 평균",
    },
]

# 타자·투수 공통. 시즌 스탯이 아니라 조건이라 리그 비교 막대를 다르게 그린다.
COMMON_GROUPS = [
    {
        "key": "age", "label": "나이", "stat": None,
        "source": "age_at_fa", "fmt": "{:.0f}", "unit": "세",
        "members": ["age_at_fa", "age_squared", "prime_years_left"],
        "tip": None,
        "note": "FA 시점 나이 · 낮을수록 유리",
    },
    {
        "key": "star", "label": "스타성", "stat": None,
        "source": "star_score", "fmt": "{:.0f}", "unit": "점",
        "members": ["star_score", "mvp_count", "golden_glove_count", "national_team"],
        "tip": None,
        "note": "MVP 5점 · 골든글러브 3점 · 국가대표 2점",
    },
    {
        "key": "position", "label": "포지션 프리미엄", "stat": None,
        "source": None, "fmt": None, "unit": "",
        "members": ["position_enc", "role_enc"],
        "tip": None,
        "note": "포지션 희소성",
    },
    {
        "key": "market", "label": "시장 상황", "stat": None,
        "source": "market_level", "fmt": "{:.1f}", "unit": "억",
        "members": ["market_level", "fa_year"],
        "tip": None,
        "note": "직전 3년 FA 계약 연평균 중앙값",
    },
]


def _groups(is_hitter: bool) -> list[dict]:
    return (HITTER_GROUPS if is_hitter else PITCHER_GROUPS) + COMMON_GROUPS


# SHAP TreeExplainer를 걸 수 있는 모델. 번들 구성은 학습 때 성능으로 정해지므로
# 특정 모델 이름을 박아두면 블렌드가 바뀌는 순간 근거가 통째로 사라진다.
# (타자 블렌드가 LightGBM+Ridge에서 XGBoost+Ridge로 바뀌자 실제로 빈 화면이 됐다.)
TREE_MODELS = ("LightGBM", "XGBoost", "RandomForest")


def _contributions(row: dict, bundle: dict) -> dict[str, float] | None:
    """피처별 SHAP 기여도. 구할 수 없으면 None."""
    model = next(
        (bundle["models"][name] for name in TREE_MODELS if name in bundle["models"]),
        None,
    )
    if model is None:
        return None

    try:
        import shap
    except ImportError:
        return None

    features = bundle["meta"]["features"]
    frame = to_frame(row, features).astype(float)

    try:
        values = shap.TreeExplainer(model).shap_values(frame)
    except Exception:
        # SHAP은 모델 내부 구조에 의존한다. 버전이 어긋나면 근거만 빠지고
        # 예측 자체는 살아 있어야 하므로 여기서 멈추지 않는다.
        return None

    return dict(zip(features, np.asarray(values).reshape(len(features)).astype(float)))


def top_factors(
    row: dict,
    bundle: dict,
    is_hitter: bool,
    league: dict,
    limit: int = 4,
) -> list[dict]:
    """이 선수의 연봉을 움직인 요인 상위 limit개.

    impact가 양수면 올린 요인, 음수면 깎은 요인이다.
    """
    from src.league import compare, format_value

    contributions = _contributions(row, bundle)
    if contributions is None:
        return []

    scored = []
    for group in _groups(is_hitter):
        impact = sum(contributions.get(name, 0.0) for name in group["members"])
        if impact == 0.0:
            continue

        source = group.get("source")
        value = row.get(source) if source else None
        if source and (value is None or _is_nan(value)):
            continue

        entry = {
            "label": group["label"],
            "note": group["note"],
            "tip": group["tip"],
            "impact": float(impact),
            "up": impact > 0,
            "value_text": _format_value(group, value, row),
            "unit": group["unit"],
        }

        band = compare(league, group["stat"], value) if group["stat"] else None
        if band:
            entry.update({
                "bar": band["bar"],
                "avg_at": band["avg_at"],
                "right": (
                    f"리그 평균 <b>{format_value(band['avg'], group['fmt'])}</b>"
                    f" · {band['rank_text']}"
                ),
            })
        else:
            entry.update(_fallback_band(group, row, bundle["reference"]))

        scored.append(entry)

    scored.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return scored[:limit]


def _is_nan(value) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _format_value(group: dict, value, row: dict) -> str:
    from src.league import format_value

    if group["key"] == "position":
        return str(row.get("position") or row.get("pitcher_role") or "-")
    if value is None or group["fmt"] is None:
        return "-"
    try:
        return format_value(value, group["fmt"])
    except (TypeError, ValueError):
        return str(value)


def _fallback_band(group: dict, row: dict, reference: dict) -> dict:
    """리그 시즌 스탯으로 견줄 수 없는 항목. FA 계약자 표본 안에서의 위치로 그린다."""
    source = group.get("source")
    sample = (reference.get(source) or {}).get("all") if source else None
    value = row.get(source) if source else None

    if sample is None or len(sample) == 0 or value is None or _is_nan(value):
        return {"bar": 0.5, "avg_at": 0.5, "right": group["note"]}

    pct = float((np.asarray(sample) <= float(value)).mean())

    if group["key"] == "age":
        return {
            "bar": 1.0 - pct,  # 나이는 낮을수록 유리하다
            "avg_at": 0.5,
            "right": f"FA 계약자 평균 <b>{float(np.mean(sample)):.1f}세</b>",
        }

    return {
        "bar": pct,
        "avg_at": 0.5,
        "right": f"FA 계약자 중 상위 {max(1, min(99, round((1 - pct) * 100)))}%",
    }
