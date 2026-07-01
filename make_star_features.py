"""
스타성 변수 생성
  - star_features_hitter.csv  → hitter_training_v5.csv
  - star_features_pitcher.csv → pitcher_training_v5.csv

star_score = MVP×5 + 골든글러브×3 + 국가대표×2 + 올스타×1 + 포스트시즌×1
pitcher_role: save_3yr_avg>=10→CL / hold_3yr_avg>=10→SU / else→SP (데이터에서 자동 계산)
"""

import pandas as pd

# ── 타자 스타성 ─────────────────────────────────────────────
# (player_name, fa_year) → FA 계약 시점까지 누적 수상 기록
HITTER_STAR_DATA = {
    ("민병헌",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=5, national_team=1, postseason_experience=1),
    ("황재균",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=5, national_team=1, postseason_experience=1),
    ("정근우",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=6, national_team=1, postseason_experience=1),
    ("손아섭",  2018): dict(mvp_count=0, golden_glove_count=2, allstar_count=7, national_team=1, postseason_experience=1),
    ("강민호",  2018): dict(mvp_count=1, golden_glove_count=4, allstar_count=8, national_team=1, postseason_experience=1),
    ("최준석",  2018): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=0, postseason_experience=1),
    ("김주찬",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=5, national_team=1, postseason_experience=1),
    ("이대형",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=4, national_team=1, postseason_experience=0),
    ("이종욱",  2018): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("지석훈",  2018): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("손시헌",  2018): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("정의윤",  2018): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("채태인",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=4, national_team=1, postseason_experience=1),
    ("김상수",  2019): dict(mvp_count=0, golden_glove_count=1, allstar_count=5, national_team=1, postseason_experience=1),
    ("김민성",  2019): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("이용규",  2019): dict(mvp_count=0, golden_glove_count=2, allstar_count=7, national_team=1, postseason_experience=1),
    ("송광민",  2019): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("모창민",  2019): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("박경수",  2019): dict(mvp_count=0, golden_glove_count=1, allstar_count=5, national_team=1, postseason_experience=1),
    ("박용택",  2019): dict(mvp_count=0, golden_glove_count=3, allstar_count=10, national_team=1, postseason_experience=1),
    ("최정",    2019): dict(mvp_count=0, golden_glove_count=5, allstar_count=8,  national_team=1, postseason_experience=1),
    ("이재원",  2019): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("양의지",  2019): dict(mvp_count=1, golden_glove_count=3, allstar_count=6, national_team=1, postseason_experience=1),
    ("안치홍",  2020): dict(mvp_count=0, golden_glove_count=1, allstar_count=5, national_team=1, postseason_experience=1),
    ("이성열",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=0, postseason_experience=1),
    ("김태균",  2020): dict(mvp_count=1, golden_glove_count=3, allstar_count=9, national_team=1, postseason_experience=1),
    ("전준우",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=4, national_team=1, postseason_experience=1),
    ("김선빈",  2020): dict(mvp_count=0, golden_glove_count=2, allstar_count=5, national_team=1, postseason_experience=1),
    ("유한준",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=0, postseason_experience=1),
    ("김강민",  2020): dict(mvp_count=0, golden_glove_count=2, allstar_count=6, national_team=1, postseason_experience=1),
    ("오지환",  2020): dict(mvp_count=0, golden_glove_count=2, allstar_count=5, national_team=1, postseason_experience=1),
    ("오재원",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=0, postseason_experience=1),
    ("박석민",  2020): dict(mvp_count=0, golden_glove_count=2, allstar_count=7, national_team=1, postseason_experience=1),
    ("김성현",  2021): dict(mvp_count=0, golden_glove_count=1, allstar_count=4, national_team=1, postseason_experience=1),
    ("김용의",  2021): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("허경민",  2021): dict(mvp_count=0, golden_glove_count=1, allstar_count=4, national_team=1, postseason_experience=1),
    ("최주환",  2021): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=0, postseason_experience=1),
    ("오재일",  2021): dict(mvp_count=0, golden_glove_count=1, allstar_count=4, national_team=1, postseason_experience=1),
    ("최형우",  2021): dict(mvp_count=1, golden_glove_count=6, allstar_count=9, national_team=1, postseason_experience=1),
    ("정수빈",  2021): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("김재호",  2021): dict(mvp_count=0, golden_glove_count=2, allstar_count=6, national_team=1, postseason_experience=1),
    ("이대호",  2021): dict(mvp_count=1, golden_glove_count=7, allstar_count=10, national_team=1, postseason_experience=1),
    ("이원석",  2021): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("박해민",  2022): dict(mvp_count=0, golden_glove_count=2, allstar_count=5, national_team=1, postseason_experience=1),
    ("손아섭",  2022): dict(mvp_count=1, golden_glove_count=3, allstar_count=9, national_team=1, postseason_experience=1),
    ("정훈",    2022): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=0, postseason_experience=1),
    ("나성범",  2022): dict(mvp_count=0, golden_glove_count=2, allstar_count=6, national_team=1, postseason_experience=1),
    ("박병호",  2022): dict(mvp_count=1, golden_glove_count=5, allstar_count=8, national_team=1, postseason_experience=1),
    ("김현수",  2022): dict(mvp_count=0, golden_glove_count=4, allstar_count=9, national_team=1, postseason_experience=1),
    ("강민호",  2022): dict(mvp_count=1, golden_glove_count=6, allstar_count=10, national_team=1, postseason_experience=1),
    ("황재균",  2022): dict(mvp_count=0, golden_glove_count=1, allstar_count=7, national_team=1, postseason_experience=1),
    ("박건우",  2022): dict(mvp_count=0, golden_glove_count=2, allstar_count=5, national_team=1, postseason_experience=1),
    ("김재환",  2022): dict(mvp_count=1, golden_glove_count=2, allstar_count=7, national_team=1, postseason_experience=1),
    ("장성우",  2022): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("최재훈",  2022): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("이명기",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("오태곤",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("김상수",  2023): dict(mvp_count=0, golden_glove_count=1, allstar_count=6, national_team=1, postseason_experience=1),
    ("박세혁",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("박민우",  2023): dict(mvp_count=0, golden_glove_count=1, allstar_count=5, national_team=1, postseason_experience=1),
    ("노진혁",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("양의지",  2023): dict(mvp_count=1, golden_glove_count=5, allstar_count=8, national_team=1, postseason_experience=1),
    ("채은성",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("박동원",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("유강남",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("양석환",  2024): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("김선빈",  2024): dict(mvp_count=0, golden_glove_count=3, allstar_count=7, national_team=1, postseason_experience=1),
    ("오지환",  2024): dict(mvp_count=0, golden_glove_count=3, allstar_count=7, national_team=1, postseason_experience=1),
    ("김성욱",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("서건창",  2025): dict(mvp_count=1, golden_glove_count=3, allstar_count=7, national_team=1, postseason_experience=1),
    ("류지혁",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("심우준",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("허경민",  2025): dict(mvp_count=0, golden_glove_count=2, allstar_count=5, national_team=1, postseason_experience=1),
    ("최정",    2025): dict(mvp_count=0, golden_glove_count=8, allstar_count=12, national_team=1, postseason_experience=1),
    ("강민호",  2026): dict(mvp_count=1, golden_glove_count=8, allstar_count=12, national_team=1, postseason_experience=1),
    ("강백호",  2026): dict(mvp_count=1, golden_glove_count=2, allstar_count=5,  national_team=1, postseason_experience=1),
    ("김현수",  2026): dict(mvp_count=0, golden_glove_count=5, allstar_count=11, national_team=1, postseason_experience=1),
    ("박찬호",  2026): dict(mvp_count=0, golden_glove_count=0, allstar_count=2,  national_team=0, postseason_experience=1),
    ("조수행",  2026): dict(mvp_count=0, golden_glove_count=0, allstar_count=1,  national_team=0, postseason_experience=0),
    ("최형우",  2026): dict(mvp_count=1, golden_glove_count=8, allstar_count=12, national_team=1, postseason_experience=1),
}

# ── 투수 스타성 ─────────────────────────────────────────────
# pitcher_role은 pitcher_training_v4.csv의 save/hold 데이터로 자동 계산
PITCHER_STAR_DATA = {
    ("안영명",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=3, national_team=1, postseason_experience=1),
    ("박정진",  2018): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("김승회",  2018): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("권오준",  2018): dict(mvp_count=0, golden_glove_count=1, allstar_count=3, national_team=1, postseason_experience=1),
    ("윤성환",  2019): dict(mvp_count=0, golden_glove_count=3, allstar_count=6, national_team=1, postseason_experience=1),
    ("이보근",  2019): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("금민철",  2019): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=0),
    ("정우람",  2020): dict(mvp_count=0, golden_glove_count=4, allstar_count=7, national_team=1, postseason_experience=1),
    ("윤규진",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("고효준",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("송은범",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("진해수",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("오주원",  2020): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("우규민",  2021): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("차우찬",  2021): dict(mvp_count=0, golden_glove_count=2, allstar_count=5, national_team=1, postseason_experience=1),
    ("유희관",  2021): dict(mvp_count=0, golden_glove_count=2, allstar_count=5, national_team=0, postseason_experience=1),
    ("이용찬",  2021): dict(mvp_count=0, golden_glove_count=2, allstar_count=4, national_team=1, postseason_experience=1),
    ("백정현",  2022): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("한현희",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=1, postseason_experience=1),
    ("이재학",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("김진성",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("장시환",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("원종현",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=1, postseason_experience=1),
    ("이태양",  2023): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("임창민",  2024): dict(mvp_count=0, golden_glove_count=1, allstar_count=3, national_team=1, postseason_experience=1),
    ("주권",    2024): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("임찬규",  2024): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("오승환",  2024): dict(mvp_count=0, golden_glove_count=7, allstar_count=12, national_team=1, postseason_experience=1),
    ("홍건희",  2024): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("김강률",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("노경은",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("우규민",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=3, national_team=1, postseason_experience=1),
    ("임기영",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("장현식",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("최원태",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("김원중",  2025): dict(mvp_count=0, golden_glove_count=1, allstar_count=4, national_team=1, postseason_experience=1),
    ("엄상백",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("구승민",  2025): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
    ("이준영",  2026): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("김광현",  2026): dict(mvp_count=0, golden_glove_count=5, allstar_count=10, national_team=1, postseason_experience=1),
    ("양현종",  2026): dict(mvp_count=0, golden_glove_count=4, allstar_count=10, national_team=1, postseason_experience=1),
    ("최원준",  2026): dict(mvp_count=0, golden_glove_count=0, allstar_count=1, national_team=0, postseason_experience=1),
    ("이영하",  2026): dict(mvp_count=0, golden_glove_count=0, allstar_count=2, national_team=0, postseason_experience=1),
}


def _calc_star_score(row: dict) -> int:
    return (
        row["mvp_count"]               * 5
        + row["golden_glove_count"]    * 3
        + row["national_team"]         * 2
        + row["allstar_count"]         * 1
        + row["postseason_experience"] * 1
    )


def _classify_pitcher_role(save_avg: float, hold_avg: float) -> str:
    if save_avg >= 10:
        return "CL"
    if hold_avg >= 10:
        return "SU"
    return "SP"


def build_hitter_star() -> pd.DataFrame:
    rows = []
    for (name, year), vals in HITTER_STAR_DATA.items():
        rows.append({"player_name": name, "fa_year": year,
                     **vals, "star_score": _calc_star_score(vals)})
    return pd.DataFrame(rows)


def build_pitcher_star(pitcher_v4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, year), vals in PITCHER_STAR_DATA.items():
        match = pitcher_v4[
            (pitcher_v4["player_name"] == name) & (pitcher_v4["fa_year"] == year)
        ]
        if match.empty:
            role = "SP"
        else:
            role = _classify_pitcher_role(
                match["save_3yr_avg"].iloc[0],
                match["hold_3yr_avg"].iloc[0],
            )
        rows.append({"player_name": name, "fa_year": year,
                     **vals, "star_score": _calc_star_score(vals),
                     "pitcher_role": role})
    return pd.DataFrame(rows)


def main() -> None:
    # ── 타자 ──
    hitter_star = build_hitter_star()
    hitter_star.to_csv("data/star_features_hitter.csv", index=False, encoding="utf-8-sig")
    print(f"star_features_hitter.csv 저장 완료 ({len(hitter_star)}명)")

    hitter_v4 = pd.read_csv("data/hitter_training_v4.csv")
    hitter_v5 = hitter_v4.merge(hitter_star, on=["player_name", "fa_year"], how="left")
    star_cols = ["mvp_count", "golden_glove_count", "allstar_count",
                 "national_team", "postseason_experience", "star_score"]
    missing_h = hitter_v5[hitter_v5["star_score"].isna()]["player_name"].tolist()
    if missing_h:
        print(f"⚠️  타자 스타성 누락: {missing_h}")
        hitter_v5[star_cols] = hitter_v5[star_cols].fillna(0)
    hitter_v5.to_csv("data/hitter_training_v5.csv", index=False, encoding="utf-8-sig")
    print(f"hitter_training_v5.csv 저장 완료 — {len(hitter_v5)}명, {len(hitter_v5.columns)}컬럼")

    # ── 투수 ──
    pitcher_v4 = pd.read_csv("data/pitcher_training_v4.csv")
    pitcher_star = build_pitcher_star(pitcher_v4)
    pitcher_star.to_csv("data/star_features_pitcher.csv", index=False, encoding="utf-8-sig")
    print(f"\nstar_features_pitcher.csv 저장 완료 ({len(pitcher_star)}명)")
    print(pitcher_star[["player_name", "fa_year", "golden_glove_count",
                         "allstar_count", "national_team", "star_score",
                         "pitcher_role"]].to_string(index=False))

    pitcher_v5 = pitcher_v4.merge(pitcher_star, on=["player_name", "fa_year"], how="left")
    missing_p = pitcher_v5[pitcher_v5["star_score"].isna()]["player_name"].tolist()
    if missing_p:
        print(f"⚠️  투수 스타성 누락: {missing_p}")
        pitcher_v5[star_cols + ["pitcher_role"]] = pitcher_v5[star_cols + ["pitcher_role"]].fillna({"pitcher_role": "SP", **{c: 0 for c in star_cols}})
    pitcher_v5.to_csv("data/pitcher_training_v5.csv", index=False, encoding="utf-8-sig")
    print(f"\npitcher_training_v5.csv 저장 완료 — {len(pitcher_v5)}명, {len(pitcher_v5.columns)}컬럼")


if __name__ == "__main__":
    main()
