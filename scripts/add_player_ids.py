"""FA 계약·FA 예정 목록에 player_id를 박아 넣는다.

왜 필요한가. 두 파일은 선수를 이름으로만 적어 두었는데 KBO에는 같은 이름이
많다. 마스터에 박건우는 4명, 김민수는 3명이다. 화면 코드가 이름으로 붙이면
롯데 박건우가 'NC 외야수, 2028년 FA 예정' 카드를 받는다. 실제로 그랬다.

계약 쪽은 src.features.resolve_player_id가 출전 경기 수와 소속팀으로 골라내
9건 모두 맞히고 있었지만, 시즌 데이터가 한 줄만 바뀌어도 결과가 달라진다.
한 번 확정해 파일에 적어두고 그 뒤로는 조회만 한다.

입력: data/fa_contracts_v4.csv, data/future_fa_candidates.csv
출력: data/fa_contracts_v5.csv, data/future_fa_candidates_v2.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import resolve_player_id  # noqa: E402

DATA = Path(__file__).parent.parent / "data"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def resolve_contracts(fa: pd.DataFrame, hitters: pd.DataFrame, pitchers: pd.DataFrame) -> pd.Series:
    ids = []
    for row in fa.itertuples():
        seasons = pitchers if row.position == "P" else hitters
        ids.append(resolve_player_id(seasons, row.player_name, row.team, int(row.fa_year)))
    return pd.Series(ids, index=fa.index, dtype="Int64")


def resolve_future(future: pd.DataFrame, master: pd.DataFrame) -> pd.Series:
    """이름 + 선수 유형으로 좁히고, 그래도 여럿이면 2026년 소속팀으로 가른다."""
    ids = []
    for row in future.itertuples():
        same = master[
            (master["player_name"] == row.player_name)
            & (master["player_type"] == row.player_type)
        ]
        if len(same) > 1:
            same = same[same["team_latest"] == row.team_2026]
        if len(same) != 1:
            raise ValueError(f"player_id를 확정할 수 없다: {row.player_name} ({row.team_2026})")
        ids.append(int(same.iloc[0]["player_id"]))
    return pd.Series(ids, index=future.index, dtype="Int64")


def main() -> None:
    master = _read("player_master.csv")
    hitters = _read("hitter_season_stats_2013_2026_v3.csv")
    pitchers = _read("pitcher_season_stats_2013_2026_v3.csv")

    fa = _read("fa_contracts_v4.csv")
    fa.insert(1, "player_id", resolve_contracts(fa, hitters, pitchers))
    # 시즌 스탯에 아예 없는 선수는 확정할 수 없다. 투수 김상수(2021 SK)가 그렇다.
    # 네이버 크롤 원본에 없어서 학습 표본에서도 이미 빠져 있다. 비워 두고 알린다.
    unresolved = fa[fa["player_id"].isna()]
    for row in unresolved.itertuples():
        print(f"  ! player_id 미확정 — {row.player_name} {row.fa_year} {row.team} (시즌 기록 없음)")
    fa.to_csv(DATA / "fa_contracts_v5.csv", index=False, encoding="utf-8-sig")

    future = _read("future_fa_candidates.csv")
    future.insert(1, "player_id", resolve_future(future, master))
    future.to_csv(DATA / "future_fa_candidates_v2.csv", index=False, encoding="utf-8-sig")

    print(f"fa_contracts_v5.csv        {len(fa)}건")
    print(f"future_fa_candidates_v2.csv {len(future)}명")


if __name__ == "__main__":
    main()
