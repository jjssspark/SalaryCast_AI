"""아직 FA 예정 목록에 없는 현역의 FA 자격 연도를 추정한다.

왜 필요한가. 조사해서 적어 둔 FA 예정 선수는 42명뿐이다. 8시즌 넘게 뛴 현역
191명 중 84명이 계약·예정·다년계약 어디에도 없어, 화면이 근거 없이
'내년에 FA가 온다고 가정'한 값을 보여줬다.

KBO FA 자격은 1군 등록일수 145일을 한 시즌으로 쳐서 고졸 8시즌, 대졸 7시즌이다
(출처: 나무위키 'KBO FA 제도'). 우리에게는 등록일수가 없고 시즌별 출전 기록만
있는데, 이 기록에는 145일을 못 채운 시즌도 한 줄로 잡힌다. 그래서 규정 시즌 수를
그대로 쓰면 자격 연도를 실제보다 이르게 잡는다.

조사해 둔 42명에 규칙을 맞춰 본 결과가 이렇다.

    데뷔 + 8  →  정확 30/42, ±1년 42/42
    데뷔 + 9  →  정확 32/42, ±1년 40/42
    데뷔 + 10 →  정확 29/42, ±1년 38/42

+9를 쓴다. 규정 8시즌에 '등록일수 미달 시즌' 몫으로 한 시즌을 얹은 값이고,
평균 오차도 0에 가장 가깝다(+0.00년).

이 값은 추정이다. 조사한 값과 섞지 않으려고 파일을 따로 둔다.
화면은 이 파일에서 온 연도에 '추정' 표시를 붙인다.

입력: data/player_master.csv, fa_contracts_v6.csv,
      future_fa_candidates_v2.csv, non_fa_extensions.csv
출력: data/fa_eligibility_estimated.csv
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent.parent / "data"

# 규정 8시즌 + 등록일수 미달 시즌 몫 1. 위 도크스트링의 실측 근거 참고.
SEASONS_TO_ELIGIBLE = 9

# 최근 두 시즌 안에 1군 기록이 있어야 현역으로 본다. 은퇴·방출 선수까지
# 'FA 예정'으로 잡으면 목록이 사실과 멀어진다.
ACTIVE_SINCE = 2025


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def main() -> None:
    master = _read("player_master.csv")
    fa = _read("fa_contracts_v6.csv")
    future = _read("future_fa_candidates_v2.csv")
    extensions = _read("non_fa_extensions.csv")

    known_ids = set(fa["player_id"].dropna().astype(int)) | set(future["player_id"].astype(int))
    known_pairs = set(zip(extensions["player_name"], extensions["team"]))

    active = master[master["latest_year"] >= ACTIVE_SINCE]
    target = active[
        ~active["player_id"].isin(known_ids)
        & ~active.apply(lambda r: (r["player_name"], r["team_latest"]) in known_pairs, axis=1)
    ].copy()

    # 데뷔 기준 자격 연도와 '다음 시즌' 중 늦은 쪽. 이미 자격 시즌을 채운
    # 선수는 언제든 나올 수 있으므로 가장 이른 시점인 다음 해로 본다.
    #
    # 하한을 하나 더 둔다. 2025년까지만 뛴 선수는 latest_year + 1이 2026,
    # 즉 이미 지나간 연도가 된다. 우리가 아는 마지막 시즌의 다음 해보다
    # 이르게 잡지 않는다.
    next_fa_year = int(master["latest_year"].max()) + 1
    target["fa_year_expected"] = target.apply(
        lambda r: max(
            int(r["first_year"]) + SEASONS_TO_ELIGIBLE,
            int(r["latest_year"]) + 1,
            next_fa_year,
        ),
        axis=1,
    )
    target["basis"] = target.apply(
        lambda r: f"데뷔 {int(r['first_year'])} · 1군 {int(r['season_count'])}시즌", axis=1
    )

    out = target[[
        "player_id", "player_name", "player_type", "position",
        "team_latest", "first_year", "season_count", "fa_year_expected", "basis",
    ]].sort_values(["fa_year_expected", "player_name"])
    out.to_csv(DATA / "fa_eligibility_estimated.csv", index=False, encoding="utf-8-sig")

    print(f"추정 대상 {len(out)}명 (현역 {len(active)}명 중)")
    print(out["fa_year_expected"].value_counts().sort_index().to_string())
    veterans = out[out["season_count"] >= 8]
    print(f"\n8시즌 이상인데 목록에 없던 선수: {len(veterans)}명 → 이제 추정 연도를 받는다")


if __name__ == "__main__":
    main()
