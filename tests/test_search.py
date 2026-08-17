"""검색 인덱스 — 설계 스펙 §8 "임의 선수 20명 표본에 대해 이름·초성 검색이 모두 성공".

표본을 손으로 적지 않고 실제 마스터에서 고정 간격으로 뽑는다.
통과하는 선수만 골라 적는 일을 막기 위해서다.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.search import has_namesake, is_chosung_query, search_players, to_chosung

MASTER_PATH = Path("data/player_master.csv")
SAMPLE_SIZE = 20


@pytest.fixture(scope="module")
def master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        pytest.skip(f"{MASTER_PATH} 없음")
    return pd.read_csv(MASTER_PATH, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def sample(master: pd.DataFrame) -> pd.DataFrame:
    """마스터 전체에서 고르게 20명. 앞쪽(가나다순)에만 몰리지 않게 한다."""
    step = max(1, len(master) // SAMPLE_SIZE)
    return master.iloc[::step].head(SAMPLE_SIZE)


def test_chosung_extraction_handles_complex_syllables():
    assert to_chosung("홍창기") == "ㅎㅊㄱ"
    assert to_chosung("최정") == "ㅊㅈ"


def test_chosung_query_is_detected_only_for_pure_initials():
    assert is_chosung_query("ㅎㅊㄱ")
    assert not is_chosung_query("홍창기")


def test_every_sampled_player_is_findable_by_full_name(master, sample):
    missed = [
        row.player_name
        for row in sample.itertuples()
        if row.player_name not in set(search_players(master, row.player_name)["player_name"])
    ]
    assert not missed, f"이름으로 못 찾은 선수: {missed}"


def test_every_sampled_player_is_findable_by_initials(master, sample):
    missed = [
        row.player_name
        for row in sample.itertuples()
        if row.player_name
        not in set(search_players(master, to_chosung(row.player_name))["player_name"])
    ]
    assert not missed, f"초성으로 못 찾은 선수: {missed}"


def test_partial_name_matches(master):
    found = search_players(master, "창기")
    assert not found.empty
    assert all("창기" in name for name in found["player_name"])


def test_blank_query_returns_nothing(master):
    assert search_players(master, "   ").empty


def test_team_filter_narrows_results(master):
    team = master["team_latest"].dropna().iloc[0]
    found = search_players(master, to_chosung(master["player_name"].iloc[0]), team=team)
    assert set(found["team_latest"]) <= {team}


def test_namesakes_are_all_returned_not_collapsed(master):
    """같은 이름이 여러 명이면 전부 나와야 한다. 하나로 접으면 다른 사람이 조회된다."""
    counts = master["player_name"].value_counts()
    duplicated = counts[counts > 1]
    if duplicated.empty:
        pytest.skip("동명이인이 없다")

    name = str(duplicated.index[0])
    assert has_namesake(master, name)
    assert len(search_players(master, name)) >= 2


def test_results_put_the_regular_above_the_one_game_namesake(master):
    """동명이인 중 시즌 수가 많은 쪽이 위에 온다."""
    counts = master["player_name"].value_counts()
    spread = [
        name for name in counts[counts > 1].index
        if master.loc[master["player_name"] == name, "season_count"].nunique() > 1
    ]
    if not spread:
        pytest.skip("시즌 수가 갈리는 동명이인이 없다")

    seasons = search_players(master, str(spread[0]))["season_count"].tolist()
    assert seasons == sorted(seasons, reverse=True)
