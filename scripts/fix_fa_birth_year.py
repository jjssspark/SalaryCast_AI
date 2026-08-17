"""
FA 계약 데이터의 생년 교정 (v3 -> v4)

fa_contracts_v3.csv의 age_at_fa는 fa_year - birth_year로 만들어졌는데(140건 중 139건 일치)
그 birth_year가 상당수 틀렸다. 위키데이터와 대조하니 절반 가까이 어긋났고,
실제값을 아는 선수로 확인해 보면 위키데이터 쪽이 맞았다.

  허경민  v3 1993 / 실제 1990 (2025 FA 기준 32세 -> 35세)
  정수빈  v3 1992 / 실제 1990
  오재일  v3 1988 / 실제 1986
  우규민  v3 1983 / 실제 1985

나이는 상위 피처라 학습 데이터 전체에 영향을 준다.

위키데이터 값이라고 무조건 덮지 않는다. 동명이인을 못 좁힌 경우와
데뷔 나이가 말이 안 되는 경우는 원본을 그대로 둔다.

출력:
  data/fa_contracts_v4.csv
  output/reports/birth_year_corrections.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_player_birth import (  # noqa: E402
    QUERY_KBO_TEAMS,
    QUERY_KOREAN,
    resolve,
    run_sparql,
)

DATA_DIR = Path("data")
REPORT_DIR = Path("output/reports")


def first_season_lookup() -> pd.Series:
    seasons = pd.concat([
        pd.read_csv(DATA_DIR / "hitter_season_stats_2013_2026_v3.csv"),
        pd.read_csv(DATA_DIR / "pitcher_season_stats_2013_2026_v3.csv"),
    ])
    return seasons.groupby("player_name")["collect_year"].min()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("위키데이터 조회")
    reference = pd.concat(
        [run_sparql(QUERY_KOREAN, "국내"), run_sparql(QUERY_KBO_TEAMS, "KBO 구단")],
        ignore_index=True,
    ).drop_duplicates(["name", "birth_year"])

    fa = pd.read_csv(DATA_DIR / "fa_contracts_v3.csv")
    first = first_season_lookup()

    fa["birth_year_source"] = "원본"
    changes = []

    for index, row in fa.iterrows():
        first_year = first.get(row["player_name"])
        if pd.isna(first_year):
            continue

        found, reason = resolve(
            reference[reference["name"] == row["player_name"]],
            str(row["team"]), int(first_year),
        )
        if found is None or found == row["birth_year"]:
            continue

        old_age = int(row["age_at_fa"])
        new_age = int(row["fa_year"]) - found
        fa.at[index, "birth_year"] = found
        fa.at[index, "age_at_fa"] = new_age
        fa.at[index, "birth_year_source"] = "wikidata"

        changes.append({
            "선수": row["player_name"],
            "FA연도": int(row["fa_year"]),
            "기존생년": int(row["birth_year"]),
            "교정생년": found,
            "기존나이": old_age,
            "교정나이": new_age,
            "근거": reason,
        })

    out = DATA_DIR / "fa_contracts_v4.csv"
    fa.to_csv(out, index=False, encoding="utf-8-sig")

    table = pd.DataFrame(changes)
    lines = [
        "# FA 계약 생년 교정 (v3 -> v4)",
        "",
        "`fa_contracts_v3.csv`의 `age_at_fa`는 `fa_year - birth_year`로 만들어졌는데",
        "그 `birth_year`가 틀린 건이 많았음. 위키데이터와 대조해 교정함.",
        "동명이인을 못 좁힌 건과 데뷔 나이가 맞지 않는 건은 원본을 유지함.",
        "",
        f"- 전체 {len(fa)}건",
        f"- 교정 {len(changes)}건",
        f"- 유지 {len(fa) - len(changes)}건",
        "",
    ]
    if changes:
        # pandas.to_markdown은 tabulate를 요구한다. 표 하나 때문에 의존성을 늘리지 않는다.
        header = list(table.columns)
        lines += [
            "## 교정 내역", "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        lines += [
            "| " + " | ".join(str(row[col]) for col in header) + " |"
            for _, row in table.iterrows()
        ]

    (REPORT_DIR / "birth_year_corrections.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"\n교정 {len(changes)}건 / 유지 {len(fa) - len(changes)}건")
    print(f"저장: {out}")
    print(f"보고서: {REPORT_DIR / 'birth_year_corrections.md'}")

    if changes:
        shift = (table["교정나이"] - table["기존나이"]).abs()
        print(f"\n나이 변화 폭: 평균 {shift.mean():.1f}년 / 최대 {shift.max()}년")
        print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
