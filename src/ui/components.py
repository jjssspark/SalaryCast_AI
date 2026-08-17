"""StoveLens AI — 화면 조각을 HTML 문자열로 만든다.

호출 위치: src/ui/home.py, src/ui/player.py
데이터 파일 없음. serving.Card와 league.compare() 결과만 받는다.

Streamlit 위젯으로는 목업(output/mockups/design_v3.html)의 전광판·벤토·티커를
만들 수 없어서 HTML을 직접 쓴다. 여기 함수는 전부 문자열만 돌려주고
st.markdown 호출은 하지 않는다. 화면 조립과 값 계산을 섞지 않기 위해서다.

선수 이름·팀명은 데이터에서 온 문자열이라 그대로 넣지 않고 escape한다.
"""

from __future__ import annotations

from html import escape

from src.league import format_value
from src.ui.assets import BALL, BAT, STITCH, team_color

# 화면에 노출하는 지표 설명. 모델 용어는 쓰지 않는다.
GLOSSARY_HITTER = [
    ("팀 기여도", "WAR", "이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 중요한 선수."),
    ("공격 종합", "OPS", "출루율 + 장타율. 타자의 전반적인 공격력을 나타내는 지표."),
    ("출루율", "OBP", "타석에서 아웃 당하지 않은 비율. 높을수록 공격 기회를 많이 만드는 타자."),
    ("장타율", "SLG", "타석당 평균 베이스 진루 수. 높을수록 장타력이 강한 타자."),
    ("조정 득점 생산", "wRC+", "100이 리그 평균. 높을수록 평균보다 뛰어난 타자."),
    ("가중 출루율", "wOBA", "단타·2루타·홈런에 다른 가중치를 준 종합 타격 지표."),
]

GLOSSARY_PITCHER = [
    ("팀 기여도", "WAR", "이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 중요한 선수."),
    ("방어율", "ERA", "9이닝당 자책점 평균. 낮을수록 좋은 투수."),
    ("이닝당 출루 허용", "WHIP", "이닝당 출루 허용 수. 낮을수록 안정적인 투수."),
    ("퀄리티스타트", "QS", "선발로 6이닝 이상을 3자책 이하로 막은 경기 수."),
    ("9이닝당 탈삼진", "K/9", "9이닝 기준 탈삼진 수. 높을수록 스스로 아웃을 잡는 투수."),
    ("9이닝당 볼넷", "BB/9", "9이닝 기준 볼넷 수. 낮을수록 안정적인 제구."),
]

STAT_TIPS = {
    "war": "이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 중요한 선수.",
    "ops": "출루율 + 장타율. 타자의 전반적인 공격력을 나타내는 지표.",
    "hr": "시즌당 홈런 수. 장타력을 가장 직관적으로 보여주는 지표.",
    "rbi": "자신의 타격으로 홈에 불러들인 주자 수. 득점권 해결 능력을 보여줌.",
    "era": "9이닝당 자책점 평균. 낮을수록 좋은 투수.",
    "whip": "이닝당 출루 허용 수. 낮을수록 안정적인 투수.",
    "innings": "한 시즌 소화한 이닝. 많을수록 팀이 길게 맡긴 투수.",
    "save": "승리를 지키고 경기를 끝낸 횟수. 마무리 투수의 핵심 지표.",
    "hold": "이어받은 리드를 지키고 다음 투수에게 넘긴 횟수. 셋업맨의 지표.",
}

MODE_CHIP = {
    "past": ("hot", "{year} FA 완료"),
    "extension": ("hot", "{year} 비FA 다년계약"),
    "future": ("soon", "{year} FA 예정"),
    "active": ("", "{year} 기준 예상"),
}


def _pct(value: float) -> str:
    """0~1을 CSS 폭으로. 0%면 막대가 안 보여서 최소 폭을 준다."""
    return f"{max(3.0, min(100.0, float(value) * 100)):.0f}%"


def _tip(text: str | None) -> str:
    if not text:
        return ""
    return f'<span class="tip" data-tip="{escape(text)}">?</span>'


def stars(score: int) -> str:
    """스타성 점수를 별로. MVP 5점·골든글러브 3점·국가대표 2점 합산 기준.

    골든글러브 한 번(3점)이면 별 하나, 열 번 받은 양의지(32점)면 다섯 개다.
    """
    for threshold, filled in ((20, 5), (12, 4), (6, 3), (3, 2), (1, 1)):
        if score >= threshold:
            return "★" * filled + "☆" * (5 - filled)
    return "☆☆☆☆☆"


def career_line(card) -> str:
    """MVP·골든글러브·국가대표를 사람이 읽는 문장으로."""
    parts = []
    if card.mvp_count:
        parts.append(f"MVP {card.mvp_count}회")
    if card.golden_glove_count:
        parts.append(f"골든글러브 {card.golden_glove_count}회")
    if card.national_team:
        parts.append(f"국가대표 {card.national_team}회")
    return " · ".join(parts) if parts else "주요 수상 없음"


# --------------------------------------------------------------------------
# 공통 뼈대
# --------------------------------------------------------------------------

def appbar(show_back: bool = False) -> str:
    back = '<a class="backlink" href="?" target="_self">← 처음으로</a>' if show_back else ""
    return (
        f'<div class="appbar">'
        f'<a class="brand" href="?" target="_self">{BALL}<b>StoveLens <i>AI</i></b></a>'
        f"{back}</div>"
    )


def section(title: str, note: str | None = None) -> str:
    tail = f'<span class="note">{escape(note)}</span>' if note else ""
    return (
        f'<div class="sect"><span class="plate"></span>'
        f"<h2>{escape(title)}</h2>{tail}{STITCH}</div>"
    )


def notice(text: str) -> str:
    return f'<div class="notice">{text}</div>'


def splash() -> str:
    return (
        f'<div class="splash">{BALL}'
        f'<div class="nm">StoveLens <i>AI</i></div>'
        f'<div class="msg">KBO FA 연봉 예측 모델을 불러오는 중</div></div>'
    )


# --------------------------------------------------------------------------
# 홈
# --------------------------------------------------------------------------

def hero(field_svg: str, flying_ball: str) -> str:
    return (
        f'<div class="hero">{field_svg}{flying_ball}'
        f'<div class="kicker"><span class="lamp"></span>KBO FREE AGENT VALUATION</div>'
        f"<h1><span><i>다음 FA는</i></span>"
        f"<span><i>얼마짜리인가<em>.</em></i></span></h1></div>"
    )


def ticker(rows: list[dict]) -> str:
    if not rows:
        return ""

    def one(row: dict) -> str:
        inner = (
            f'<span class="yy">{row["year"]}</span>'
            f'<span class="nn">{escape(row["name"])}</span>'
            f'<span class="tt">{escape(row["team"])} · {escape(row["position"])}</span>'
            f'<span class="plate"></span>'
        )
        if row.get("player_id"):
            return f'<a class="it" href="?p={row["player_id"]}" target="_self">{inner}</a>'
        return f'<span class="it">{inner}</span>'

    items = "".join(one(row) for row in rows)
    return (
        '<div class="ticker">'
        '<div class="lbl"><span class="lamp" style="background:#fff;box-shadow:0 0 8px #fff"></span>NEXT FA</div>'
        f'<div class="win"><div class="rail">'
        f'<div class="grp">{items}</div>'
        f'<div class="grp" aria-hidden="true">{items}</div>'
        f"</div></div></div>"
    )


def bento(candidates: list[dict], median: float, median_label: str = "타자") -> str:
    """상위 후보 타일. 첫 칸은 크게, 나머지는 작게."""
    if not candidates:
        return ""

    head, rest = candidates[0], candidates[1:5]
    relation = "높음" if head["predicted"] >= median else "낮음"
    age = f" · {head['age']}세" if head.get("age") else ""

    tiles = (
        f'<a class="tile big" style="--tc:{team_color(head["team"])}" '
        f'href="?p={head["player_id"]}" target="_self">{BAT}'
        f'<div class="yy">{head["fa_year"]} FA 예정</div>'
        f'<div class="nn">{escape(head["name"])}</div>'
        f'<div class="tt">{escape(head["team"])} · {escape(head["position"])}{age}</div>'
        f'<div class="pred"><div class="k">AI 예상 연봉</div>'
        f'<div class="v">{head["predicted"]:.1f}<small>억 / 년</small></div>'
        f'<div class="ref">역대 {escape(median_label)} FA 연평균 중앙값 '
        f"<b>{median:.1f}억</b>보다 {relation}</div></div></a>"
    )

    for item in rest:
        tiles += (
            f'<a class="tile md" style="--tc:{team_color(item["team"])}" '
            f'href="?p={item["player_id"]}" target="_self">'
            f'<div class="yy">{item["fa_year"]}</div>'
            f'<div class="nn">{escape(item["name"])}</div>'
            f'<div class="tt">{escape(item["team"])} · {escape(item["position"])}</div>'
            f'<span class="go">분석 보기 →</span></a>'
        )

    return f'<div class="bento">{tiles}</div>'


def board(cells: list[tuple[str, int]]) -> str:
    body = "".join(
        f'<div class="cell"><div class="cl">{escape(label)}</div>'
        f'<div class="cv">{value:,}</div></div>'
        for label, value in cells
    )
    return f'<div class="board">{body}</div>'


def search_results(rows: list[dict]) -> str:
    if not rows:
        return '<div class="empty">검색 결과가 없습니다. 이름 일부나 초성으로 다시 찾아보세요.</div>'

    lines = "".join(
        f'<a class="rline" style="--tc:{team_color(row["team"])}" '
        f'href="?p={row["player_id"]}" target="_self">'
        f'<span class="nm">{escape(row["name"])}</span>'
        f'<span class="meta">{escape(row["meta"])}</span>'
        f'<span class="go">분석 보기 →</span></a>'
        for row in rows
    )
    return f'<div class="results">{lines}</div>'


# --------------------------------------------------------------------------
# 선수 화면
# --------------------------------------------------------------------------

def player_hero(card) -> str:
    chip_class, chip_text = MODE_CHIP[card.mode]

    if card.photo_url:
        photo = f'<img src="{escape(str(card.photo_url))}" alt="{escape(card.name)}"/>'
    else:
        photo = escape(card.name[:1])

    age_text = f"FA 시점 {card.age}세" if card.age else "나이 정보 없음"
    detail = f"{age_text} &nbsp;·&nbsp; {escape(career_line(card))}"

    return (
        f'<div class="phero" style="--tc:{team_color(card.team)}">'
        f'<span class="wm">{card.fa_year}</span>'
        f'<div class="slab"><div class="ph">{photo}</div></div>'
        f'<div class="body"><div class="chips">'
        f'<span class="chip team">{escape(card.team)}</span>'
        f'<span class="chip">{escape(card.position_label)}</span>'
        f'<span class="chip {chip_class}">{escape(chip_text.format(year=card.fa_year))}</span>'
        f"</div><h2>{escape(card.name)}</h2>"
        f'<div class="sub"><span class="stars">{stars(card.star_score)}</span>'
        f"&nbsp;&nbsp; {detail}</div></div></div>"
    )


def money_panel(card, subtitle: str, reference_html: str, scale_max: float) -> str:
    """예상 연봉 + 예상 범위 막대. 막대의 오른쪽 끝은 역대 FA 최고 연평균이다."""
    span = max(scale_max, card.high) or 1.0
    left = card.low / span * 100
    right = 100 - (card.high / span * 100)
    dot = card.predicted / span * 100

    return (
        f'<div class="pane"><div class="pane-t"><b>AI 예상 연봉</b>'
        f"<s>{escape(subtitle)}</s></div>"
        f'<div class="money"><span class="n">{card.predicted:.1f}</span>'
        f'<span class="u">억 / 년</span></div>'
        f'<div class="money-ref">{reference_html}</div>'
        f'<div class="gauge"><div class="lbl">예상 범위</div><div class="track">'
        f'<div class="band" style="left:{left:.1f}%;right:{max(0.0, right):.1f}%"></div>'
        f'<div class="dot" style="left:{dot:.1f}%"></div></div>'
        f'<div class="ends"><span>{card.low:.1f}억</span>'
        f'<span>{card.high:.1f}억</span></div></div></div>'
    )


def versus_panel(card, result: dict, reference_html: str) -> str:
    return (
        f'<div class="pane"><div class="pane-t"><b>예측 vs 실제</b><s>연평균 · 억 원</s></div>'
        f'<div class="versus">'
        f'<div class="vs-box ai"><div class="k">AI 예측</div>'
        f'<div class="v">{card.predicted:.1f}</div></div>'
        f'<div class="vs-mid"></div>'
        f'<div class="vs-box real"><div class="k">실제 계약</div>'
        f'<div class="v">{card.actual:.1f}</div></div></div>'
        f'<div class="verdict {result["tone"]}">'
        f'<div class="big">{escape(result["title"])}</div>'
        f'<div class="txt">{escape(result["text"])}<br/>{reference_html}</div>'
        f"</div></div>"
    )


def stat_board(stats: list[dict], title: str, basis: str) -> str:
    """전광판 스탯 4칸. stats 각 항목은 label/text/band/fmt를 갖는다."""
    cells = ""
    for index, stat in enumerate(stats):
        band = stat.get("band")
        if band:
            comparison = (
                f'<div class="scale">'
                f'<i style="width:{_pct(band["bar"])};animation-delay:{index * 0.1:.1f}s"></i>'
                f'<span class="avg" style="left:{_pct(band["avg_at"])}"></span></div>'
                f'<div class="cmp">'
                f'<span>리그 평균 {format_value(band["avg"], stat.get("fmt"))}</span>'
                f'<span class="rank">{band["rank_text"]}</span></div>'
            )
        else:
            comparison = '<div class="cmp"><span>비교 기준 없음</span></div>'

        cells += (
            f'<div class="sb"><div class="k">{escape(stat["label"])}'
            f'{_tip(STAT_TIPS.get(stat.get("key")))}</div>'
            f'<div class="v">{stat["text"]}</div>{comparison}</div>'
        )

    return (
        f'<div class="pane"><div class="pane-t"><b>{escape(title)}</b>'
        f"<s>3시즌을 시즌당 평균 낸 값</s></div>"
        f'<div class="sboard">{cells}</div>'
        f'<div class="basis">{basis}</div></div>'
    )


def factor_list(factors: list[dict]) -> str:
    if not factors:
        return notice(
            "이 선수의 예측 근거를 계산하지 못했습니다. "
            "위의 예상 연봉은 정상적으로 나온 값입니다."
        )

    body = ""
    for index, factor in enumerate(factors):
        direction = "" if factor["up"] else " down"
        unit = f'<small>{escape(factor["unit"])}</small>' if factor.get("unit") else ""
        body += (
            f'<div class="f{direction}"><div class="fhead"><span class="base"></span>'
            f'<span class="fnm">{escape(factor["label"])}{_tip(factor.get("tip"))}</span>'
            f'<span class="fval">{factor["value_text"]}{unit}</span></div>'
            f'<div class="scale">'
            f'<i style="width:{_pct(factor["bar"])};animation-delay:{index * 0.1:.1f}s"></i>'
            f'<span class="avg" style="left:{_pct(factor["avg_at"])}"></span></div>'
            f'<div class="fsub"><span>{escape(factor["note"])}</span>'
            f'<span>{factor["right"]}</span></div></div>'
        )
    return f'<div class="pane"><div class="factors">{body}</div></div>'


def offer_list(offers, need_label) -> str:
    """구단별 제시가. offers는 team_abbr/team_name/low/high/offer/need_score 컬럼을 갖는다."""
    top = float(offers["offer"].max()) or 1.0
    rows = ""
    for index, offer in enumerate(offers.itertuples()):
        rows += (
            f'<div class="of" style="--tc:{team_color(offer.team_name)}">'
            f'<span class="tm">{escape(str(offer.team_abbr))}</span>'
            f'<span class="track"><i style="width:{offer.offer / top * 100:.0f}%;'
            f'animation-delay:{index * 0.06:.2f}s"></i></span>'
            f'<span class="need">{escape(need_label(offer.need_score))}</span>'
            f'<span class="amt">{offer.low:.1f} ~ {offer.high:.1f}억</span></div>'
        )
    return f'<div class="offers">{rows}</div>'


def season_table(table, seasons_used: str) -> str:
    """최근 3시즌 원본 기록 표.

    st.dataframe을 쓰면 이 화면에서 혼자 밝은 회색으로 떠서 HTML로 직접 그린다.
    """
    head = "".join(f"<th>{escape(str(column))}</th>" for column in table.columns)
    body = ""
    for row in table.itertuples(index=False):
        cells = ""
        for value in row:
            text = f"{value:.3f}".lstrip("0") if isinstance(value, float) and value < 1 \
                else f"{value:g}" if isinstance(value, float) else str(value)
            cells += f"<td>{escape(text)}</td>"
        body += f"<tr>{cells}</tr>"

    return (
        f'<div class="pane stable"><div class="pane-t"><b>최근 3시즌 기록</b>'
        f"<s>{escape(seasons_used)} 시즌</s></div>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def glossary(is_hitter: bool) -> str:
    entries = GLOSSARY_HITTER if is_hitter else GLOSSARY_PITCHER
    body = "".join(
        f'<div class="gl"><dt>{escape(name)} <em>{escape(code)}</em></dt>'
        f"<dd>{escape(desc)}</dd></div>"
        for name, code, desc in entries
    )
    return f'<dl class="glossary">{body}</dl>'
