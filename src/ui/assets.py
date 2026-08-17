"""StoveLens AI — 화면에서 쓰는 SVG 심볼과 구단 색.

호출 위치: app/app.py(심볼 1회 주입), src/ui/home.py, src/ui/player.py,
          src/ui/components.py(구단 색 조회)
데이터 파일 없음.

야구공·배트·실밥은 이미지 파일 대신 SVG 심볼로 한 번만 넣고 <use>로 재사용한다.
Streamlit은 정적 파일을 따로 서빙해야 해서 외부 이미지를 쓰면 배포에서 깨진다.
"""

from __future__ import annotations

# 구단 대표색. 목업 output/mockups/design_v3.html의 .t-* 클래스와 같은 값이다.
TEAM_COLORS = {
    "LG": "#C30452",
    "두산": "#1A477B",
    "KIA": "#EA0029",
    "SSG": "#CE0E2D",
    "SK": "#CE0E2D",
    "삼성": "#3E8EE8",
    "롯데": "#4E86DE",
    "NC": "#5A8BD0",
    "키움": "#B23A52",
    "넥센": "#B23A52",
    "한화": "#FC4E00",
    "KT": "#7A8794",
    "kt": "#7A8794",
}

DEFAULT_TEAM_COLOR = "#5D7466"


def team_color(team: str | None) -> str:
    """'KIA', 'KIA 타이거즈' 어느 쪽이 와도 맞는 색을 준다."""
    if not team:
        return DEFAULT_TEAM_COLOR
    text = str(team).strip()
    for key, color in TEAM_COLORS.items():
        if text.startswith(key) or key in text:
            return color
    return DEFAULT_TEAM_COLOR


SVG_DEFS = """
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <symbol id="sym-stitch" viewBox="0 0 240 12" preserveAspectRatio="none">
    <path d="M0 6 H240" stroke="rgba(255,255,255,.1)" stroke-width="1"/>
    <g stroke="#FF4438" stroke-width="1.6" stroke-linecap="round">
      <path d="M6 2 L14 6 M6 10 L14 6 M26 2 L34 6 M26 10 L34 6 M46 2 L54 6 M46 10 L54 6
               M66 2 L74 6 M66 10 L74 6 M86 2 L94 6 M86 10 L94 6 M106 2 L114 6 M106 10 L114 6
               M126 2 L134 6 M126 10 L134 6 M146 2 L154 6 M146 10 L154 6 M166 2 L174 6 M166 10 L174 6
               M186 2 L194 6 M186 10 L194 6 M206 2 L214 6 M206 10 L214 6 M226 2 L234 6 M226 10 L234 6"/>
    </g>
  </symbol>

  <symbol id="sym-ball" viewBox="0 0 44 44">
    <defs>
      <radialGradient id="sl-ball-grad" cx="34%" cy="28%" r="72%">
        <stop offset="0%" stop-color="#FFFFFF"/>
        <stop offset="62%" stop-color="#F3F3EF"/>
        <stop offset="100%" stop-color="#C7CAC4"/>
      </radialGradient>
    </defs>
    <circle cx="22" cy="22" r="21" fill="url(#sl-ball-grad)"/>
    <g fill="none" stroke="#D8352A" stroke-width="1.5" stroke-linecap="round">
      <path d="M9.5 4.6 C15.5 12.5 15.5 31.5 9.5 39.4"/>
      <path d="M34.5 4.6 C28.5 12.5 28.5 31.5 34.5 39.4"/>
    </g>
    <g stroke="#D8352A" stroke-width="1.4" stroke-linecap="round">
      <path d="M11.6 9.2 L7.4 10.6 M12.9 14.6 L8.4 15.4 M13.4 20.2 L8.7 20.5
               M13.4 25.4 L8.7 25.2 M12.9 30.6 L8.4 30.0 M11.6 35.4 L7.4 34.4"/>
      <path d="M32.4 9.2 L36.6 10.6 M31.1 14.6 L35.6 15.4 M30.6 20.2 L35.3 20.5
               M30.6 25.4 L35.3 25.2 M31.1 30.6 L35.6 30.0 M32.4 35.4 L36.6 34.4"/>
    </g>
  </symbol>

  <symbol id="sym-bat" viewBox="0 0 240 60">
    <g fill="#EADCC4">
      <rect x="6" y="26" width="66" height="9" rx="4.5"/>
      <rect x="2" y="23" width="12" height="15" rx="4"/>
      <path d="M70 25 C120 20 190 12 232 24 C238 26 238 34 232 36 C190 48 120 40 70 35 Z"/>
    </g>
  </symbol>
</svg>
"""

# 홈 히어로 뒤에 깔리는 다이아몬드.
FIELD_SVG = """
<svg class="field" viewBox="0 0 400 400" aria-hidden="true">
  <path class="dirt" d="M200 336 L54 190 L200 44 L346 190 Z" fill="#B07A4A" fill-opacity=".07"/>
  <circle class="dirt" cx="200" cy="190" r="30" fill="#B07A4A" fill-opacity=".1"/>
  <g fill="none" stroke="rgba(255,255,255,.15)" stroke-width="1.5">
    <path class="ln" d="M200 330 L60 190 L200 50 L340 190 Z"/>
    <path class="ln d1" d="M200 330 L4 134 M200 330 L396 134"/>
    <path class="ln d2" d="M10 140 A252 252 0 0 1 390 140"/>
    <circle class="ln d3" cx="200" cy="190" r="26"/>
  </g>
  <g class="late" fill="rgba(255,255,255,.22)">
    <rect x="192" y="322" width="16" height="16" transform="rotate(45 200 330)"/>
    <rect x="52" y="182" width="16" height="16" transform="rotate(45 60 190)"/>
    <rect x="192" y="42" width="16" height="16" transform="rotate(45 200 50)"/>
    <rect x="332" y="182" width="16" height="16" transform="rotate(45 340 190)"/>
  </g>
</svg>
"""

FLYING_BALL = (
    '<div class="arc" aria-hidden="true"><span class="fly"><span class="bob">'
    '<svg class="rot"><use href="#sym-ball"/></svg></span></span></div>'
)

STITCH = '<svg class="stitch" aria-hidden="true"><use href="#sym-stitch"/></svg>'
BALL = '<svg class="logoball" aria-hidden="true"><use href="#sym-ball"/></svg>'
BAT = '<svg class="bat" viewBox="0 0 240 60" aria-hidden="true"><use href="#sym-bat"/></svg>'
