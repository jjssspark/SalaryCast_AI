"""FA 계약을 붙일 때 동명이인의 계약이 딸려오지 않는가.

투수 김현수의 카드에 타자 김현수의 16.67억 계약이 붙어 "고평가 계약"으로 표시된 적이
있다. resolve_player_id는 타자와 투수로 갈린 시즌 테이블 안에서 동명이인을 가리는데,
그 안에서 이름이 유일해지면 소속팀을 대조하기 전에 반환하기 때문이었다.

_past_contract가 쓰는 것은 fa·master·시즌 테이블 넷뿐이라 SimpleNamespace로 세운다.
Context 전체를 만들면 모델까지 로드해야 해서 테스트가 무거워진다.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.serving import _past_contract

DATA = Path("data")

SOURCES = {
    "fa": "fa_contracts_v4.csv",
    "master": "player_master.csv",
    "hitter_seasons": "hitter_season_stats_2013_2026_v3.csv",
    "pitcher_seasons": "pitcher_season_stats_2013_2026_v3.csv",
}


@pytest.fixture(scope="module")
def context():
    for name in SOURCES.values():
        if not (DATA / name).exists():
            pytest.skip(f"{name}이 없어 건너뛴다")

    return SimpleNamespace(**{
        attr: pd.read_csv(DATA / name, encoding="utf-8-sig")
        for attr, name in SOURCES.items()
    })


def _ids(context, name: str) -> dict[str, int]:
    """이름이 같은 타자·투수의 player_id."""
    rows = context.master[context.master["player_name"] == name]
    return {str(row.player_type): int(row.player_id) for row in rows.itertuples()}


def test_pitcher_does_not_inherit_the_hitters_contract(context):
    """타자 김현수의 FA 계약이 투수 김현수에게 붙으면 안 된다."""
    ids = _ids(context, "김현수")
    assert "pitcher" in ids and "hitter" in ids, "김현수 동명이인 표본이 사라졌다"

    assert _past_contract(context, ids["pitcher"], "김현수") is None

    signed = _past_contract(context, ids["hitter"], "김현수")
    assert signed is not None, "실제 계약 당사자가 계약을 못 찾았다"
    assert signed["position"] != "P"


def test_hitter_does_not_inherit_the_pitchers_contract(context):
    """투수 최원준의 FA 계약이 타자 최원준에게 붙으면 안 된다.

    이쪽이 더 위험하다. 예측 11.2억 대 계약 12.0억이라 '적정 계약'으로 그럴듯하게
    표시돼 틀린 줄 모르고 지나간다.
    """
    ids = _ids(context, "최원준")
    assert "pitcher" in ids and "hitter" in ids, "최원준 동명이인 표본이 사라졌다"

    assert _past_contract(context, ids["hitter"], "최원준") is None

    signed = _past_contract(context, ids["pitcher"], "최원준")
    assert signed is not None, "실제 계약 당사자가 계약을 못 찾았다"
    assert signed["position"] == "P"


def test_every_matched_contract_agrees_with_the_player_type(context):
    """전수 점검 — 붙은 계약의 포지션과 선수 유형이 어긋나는 카드가 없어야 한다."""
    signed_names = set(context.fa["player_name"])
    mismatched = []

    for row in context.master.itertuples():
        if row.player_name not in signed_names:
            continue
        contract = _past_contract(context, int(row.player_id), row.player_name)
        if contract is None:
            continue
        if (contract["position"] == "P") != (str(row.player_type) == "pitcher"):
            mismatched.append((row.player_name, int(row.player_id), contract["position"]))

    assert not mismatched, f"유형이 어긋난 계약 매칭: {mismatched}"
