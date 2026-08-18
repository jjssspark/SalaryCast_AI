"""
원시 크롤 데이터(v2) -> 시즌 스탯 정제(v3)

기존 *_season_stats_2015_2026_v2.csv를 대체한다. 달라진 점:
1. 수집 범위가 시즌당 상위 100여 명에서 전 선수로 늘었다.
2. 투수 컬럼을 대폭 살렸다. 기존 정제본은 17컬럼뿐이라 삼진/볼넷/QS를 전부 버렸다.
3. position을 profile JSON에서 뽑는다. 기존에는 전 선수를 "외야수"로 하드코딩했다.
4. 2026처럼 진행 중인 시즌을 구분할 수 있게 season_completion을 남긴다.

이닝은 "187 1/3" 형태 문자열이라 실수로 바꿔서 저장한다.

출력:
  data/hitter_season_stats_2010_2026_v4.csv
  data/pitcher_season_stats_2010_2026_v4.csv
  data/player_master.csv
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

HITTER_RENAME = {
    "playerId": "player_id",
    "playerName": "player_name",
    "teamShortName": "team",
    "isQualified": "is_qualified",
    "hitterGameCount": "games",
    "hitterAb": "ab",
    "hitterHra": "avg",
    "hitterObp": "obp",
    "hitterSlg": "slg",
    "hitterOps": "ops",
    "hitterIsop": "isop",
    "hitterBabip": "babip",
    "hitterWoba": "woba",
    "hitterWrcPlus": "wrc_plus",
    "hitterHr": "hr",
    "hitterRbi": "rbi",
    "hitterRun": "run",
    "hitterHit": "hit",
    "hitterH2": "h2",
    "hitterH3": "h3",
    "hitterBb": "bb",
    "hitterHp": "hbp",
    "hitterKk": "so",
    "hitterSb": "sb",
    "hitterWpa": "wpa",
    "hitterWar": "war",
}

PITCHER_RENAME = {
    "playerId": "player_id",
    "playerName": "player_name",
    "teamShortName": "team",
    "isQualified": "is_qualified",
    "pitcherGameCount": "games",
    "pitcherEra": "era",
    "pitcherWhip": "whip",
    "pitcherWin": "win",
    "pitcherLose": "lose",
    "pitcherSave": "save",
    "pitcherHold": "hold",
    "pitcherQs": "qs",
    "pitcherKk": "so",
    "pitcherBb": "bb",
    "pitcherHp": "hbp",
    "pitcherHit": "hit",
    "pitcherHr": "hr",
    "pitcherR": "r",
    "pitcherEr": "er",
    "pitcherInningKk": "k9",
    "pitcherInningBb": "bb9",
    "pitcherKkBbRate": "k_bb",
    "pitcherPaKkRate": "pa_k_rate",
    "pitcherPaBbRate": "pa_bb_rate",
    "pitcherWpa": "wpa",
    "pitcherWar": "war",
}


def parse_innings(value) -> float:
    """'187 1/3' -> 187.333, '156' -> 156.0"""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan

    whole, _, frac = text.partition(" ")
    total = float(whole) if whole else 0.0
    if frac == "1/3":
        total += 1 / 3
    elif frac == "2/3":
        total += 2 / 3
    return round(total, 3)


def extract_position(profile_raw) -> str | None:
    if not isinstance(profile_raw, str) or not profile_raw.strip():
        return None
    try:
        return json.loads(profile_raw).get("position") or None
    except json.JSONDecodeError:
        return None


def season_completion_ratio(raw_hitters: pd.DataFrame) -> pd.Series:
    """
    시즌별 진행률. 완료 시즌은 1.0, 진행 중인 시즌만 1 미만이 된다.

    시즌 길이를 144로 고정하면 안 된다. KBO 정규시즌은 2013~2014년 128경기,
    2015년부터 144경기라 2013~2014가 89%로 잘못 찍힌다.
    완료된 시즌은 누군가 전 경기를 뛰므로 그 해 최다 출전 경기 수가 곧 시즌 길이다.
    진행 중인 시즌만 직전 시즌 길이를 기준으로 비율을 낸다.
    """
    season_max = raw_hitters.groupby("collect_year")["hitterGameCount"].max()
    ratio = pd.Series(1.0, index=season_max.index)

    latest = season_max.index.max()
    reference = season_max.drop(index=latest).iloc[-1]
    if season_max[latest] < reference:
        ratio[latest] = season_max[latest] / reference

    return ratio


def blank_out_unprovided_war(df: pd.DataFrame) -> pd.DataFrame:
    """WAR을 아예 제공하지 않은 시즌은 0이 아니라 결측으로 둔다.

    네이버는 2013~2016년 타자 WAR을 주지 않아 그 4개 시즌이 전원 0으로 들어온다.
    0으로 두면 '기여도 0인 선수'가 되어 FA 2018~2019년 계약의 war_3yr_sum이
    실제보다 낮게 잡힌다. 한 시즌의 WAR이 전원 0이면 미제공으로 판단한다.
    """
    out = df.copy()
    unprovided = out.groupby("collect_year")["war"].transform(lambda s: (s == 0).all())
    if unprovided.any():
        years = sorted(out.loc[unprovided, "collect_year"].unique())
        print(f"    WAR 미제공 시즌 -> 결측 처리: {years}")
    out.loc[unprovided, "war"] = np.nan
    return out


def add_season_completion(df: pd.DataFrame, completion: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["season_completion"] = df["collect_year"].map(completion).round(3)
    return df


def classify_role(row) -> str:
    """
    KBO FA 투수는 연간 5~10명 수준이라 역할별 모델 분리는 표본이 부족하다.
    피처 하나로 넣기 위해 SP/SU/CL/RP 네 갈래로 나눈다.
    pitcherStart가 API에서 100% 결측이라 QS와 경기당 이닝으로 선발을 판정한다.
    """
    games = row["games"] or 0
    if games and row["save"] / games >= 0.30:
        return "CL"
    if games and row["hold"] / games >= 0.25:
        return "SU"
    if row["qs"] > 0 or (row["ip_per_game"] or 0) >= 3.0:
        return "SP"
    return "RP"


def build_hitters(raw: pd.DataFrame, completion: pd.Series) -> pd.DataFrame:
    df = raw.copy()
    df["position"] = df["profile"].map(extract_position)
    df = df.rename(columns=HITTER_RENAME)

    keep = ["collect_year", "player_id", "player_name", "team", "position", "is_qualified"]
    keep += [v for v in HITTER_RENAME.values() if v not in keep]
    df = df[keep].drop_duplicates(subset=["player_id", "collect_year"])
    df = blank_out_unprovided_war(df)

    return (
        add_season_completion(df, completion)
        .sort_values(["player_name", "collect_year"])
        .reset_index(drop=True)
    )


def build_pitchers(raw: pd.DataFrame, completion: pd.Series) -> pd.DataFrame:
    df = raw.copy()
    df["position"] = df["profile"].map(extract_position)
    df["innings"] = df["pitcherInning"].map(parse_innings)
    df = df.rename(columns=PITCHER_RENAME)

    keep = [
        "collect_year", "player_id", "player_name", "team",
        "position", "is_qualified", "innings",
    ]
    keep += [v for v in PITCHER_RENAME.values() if v not in keep]
    df = df[keep].drop_duplicates(subset=["player_id", "collect_year"])

    df["ip_per_game"] = (df["innings"] / df["games"].replace(0, np.nan)).round(3)
    df["pitcher_role"] = df.apply(classify_role, axis=1)
    df = blank_out_unprovided_war(df)

    return (
        add_season_completion(df, completion)
        .sort_values(["player_name", "collect_year"])
        .reset_index(drop=True)
    )


def extract_photo_url(profile_raw) -> str | None:
    """profile JSON의 image. playerImageUrl 컬럼은 전량 404라 쓰지 않는다."""
    if not isinstance(profile_raw, str) or not profile_raw.strip():
        return None
    try:
        return json.loads(profile_raw).get("image") or None
    except json.JSONDecodeError:
        return None


def build_photo_table(*raw_frames: pd.DataFrame) -> pd.DataFrame:
    """선수별 사진 한 건. 같은 선수는 가장 최근 시즌 사진을 쓴다.

    키는 player_id다. 이름으로 묶으면 동명이인 중 한 명만 사진을 갖게 되고
    나머지는 사진 없음으로 남는다. 박건우·양현종·김현수가 여기 걸렸다.
    """
    parts = []
    for df in raw_frames:
        if df.empty or "profile" not in df.columns:
            continue
        sub = df[["playerId", "playerName", "collect_year", "profile"]].copy()
        sub["photo_url"] = sub["profile"].map(extract_photo_url)
        parts.append(sub.drop(columns=["profile"]))

    if not parts:
        return pd.DataFrame(
            columns=["player_id", "player_name", "photo_url", "source", "collected_at"]
        )

    merged = pd.concat(parts, ignore_index=True).dropna(subset=["photo_url"])
    merged = merged.sort_values("collect_year", ascending=False).drop_duplicates("playerId")

    return pd.DataFrame({
        "player_id": merged["playerId"],
        "player_name": merged["playerName"],
        "photo_url": merged["photo_url"],
        "source": "naver_api",
        "collected_at": date.today().isoformat(),
    }).sort_values("player_name").reset_index(drop=True)


CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"


def to_chosung(text: str) -> str:
    """'홍창기' -> 'ㅎㅊㄱ'. 초성만 입력해도 검색되게 하려고 미리 만들어 둔다."""
    out = []
    for char in str(text):
        code = ord(char) - 0xAC00
        out.append(CHOSUNG[code // 588] if 0 <= code <= 11171 else char)
    return "".join(out)


def build_master(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
    """검색 인덱스용 선수 마스터.

    키는 player_name이 아니라 player_id다. KBO에는 같은 이름이 96쌍 있어
    이름으로 묶으면 김현수 두 명이 한 명이 된다.
    """
    def last_valid(series: pd.Series):
        """가장 최근 시즌의 값이 비어 있으면 그 이전 시즌 값으로 내려간다."""
        valid = series.dropna()
        return valid.iloc[-1] if len(valid) else None

    parts = []
    for df, ptype, volume_col in [(hitters, "hitter", "ab"), (pitchers, "pitcher", "innings")]:
        agg = (
            df.sort_values("collect_year")
            .groupby("player_id")
            .agg(
                player_name=("player_name", "last"),
                position=("position", last_valid),
                team_latest=("team", last_valid),
                first_year=("collect_year", "min"),
                latest_year=("collect_year", "max"),
                season_count=("collect_year", "nunique"),
                volume=(volume_col, "sum"),
            )
            .reset_index()
        )
        agg["player_type"] = ptype
        parts.append(agg)

    # 같은 선수가 타자·투수 양쪽에 잡히면 실제로 더 많이 뛴 쪽을 그 선수의 유형으로
    # 본다. 예전에는 시즌 수로 갈랐는데, 투수도 타석에 서면 타자 테이블에 같은 시즌
    # 수로 잡혀 동점이 됐다. 동점이면 정렬 순서대로 타자가 이겨서 손승락·박영현·
    # 김서현 같은 투수 67명이 타자로 분류됐고, 화면에서 타자 모델로 예측됐다.
    #
    # 타수와 이닝은 단위가 다르지만 섞일 일이 없다. 진짜 투수는 타수가 0에 가깝고
    # (지명타자제) 야수가 등판해도 한두 이닝이다.
    merged = pd.concat(parts, ignore_index=True)

    # 네이버가 포지션을 알려준 선수는 그 값이 출전량보다 낫다. 투수로 전향했거나
    # 야수로 전향한 선수는 양쪽 기록이 다 적어서 출전량만으로는 뒤집힌다.
    # 김성민(SK 14타수 / SSG 0.7이닝)이 그런 경우다.
    naver = merged.dropna(subset=["position"])
    declared = (
        naver[naver["position"].isin(["투수", "야수", "내야수", "외야수", "포수", "유격수"])]
        .assign(declared=lambda d: d["position"].eq("투수").map({True: "pitcher", False: "hitter"}))
        .drop_duplicates("player_id")
        .set_index("player_id")["declared"]
    )
    merged["declared"] = merged["player_id"].map(declared)
    merged["type_match"] = merged["declared"].isna() | merged["declared"].eq(merged["player_type"])

    merged = merged.sort_values(
        ["player_id", "type_match", "volume", "season_count", "latest_year"],
        ascending=[True, False, False, False, False],
    ).drop_duplicates("player_id")

    # 네이버 profile의 position은 절반 가까이 비어 있다. 대체 엔드포인트는 403이라
    # 소속 테이블로 최소한의 구분만 채운다. FA 계약 140건과 미래 FA 후보 42명은
    # 각자 CSV에 position이 다 있으므로 학습과 예측 화면에는 영향이 없다.
    fallback = merged["player_type"].map({"pitcher": "투수", "hitter": "야수"})
    merged["position"] = merged["position"].fillna(fallback)
    merged["position_source"] = np.where(
        merged["position"].eq(fallback), "추정", "네이버"
    )

    merged["chosung"] = merged["player_name"].map(to_chosung)

    cols = [
        "player_id", "player_name", "chosung", "player_type",
        "position", "position_source", "team_latest",
        "first_year", "latest_year", "season_count",
    ]
    return merged[cols].sort_values(
        ["player_name", "latest_year"], ascending=[True, False]
    ).reset_index(drop=True)


def main() -> None:
    print("=" * 62)
    print("  시즌 스탯 정제 v4 (2010~2026)")
    print("=" * 62)

    raw_h = pd.read_csv(DATA_DIR / "naver_hitter_2010_2026_raw_v2.csv")
    raw_p = pd.read_csv(DATA_DIR / "naver_pitcher_2010_2026_raw_v2.csv")

    completion = season_completion_ratio(raw_h)

    print("\n시즌 진행률")
    for year, ratio in completion.items():
        flag = "  <- 진행 중" if ratio < 0.99 else ""
        print(f"  {year}  {ratio * 100:5.1f}%{flag}")

    hitters = build_hitters(raw_h, completion)
    pitchers = build_pitchers(raw_p, completion)
    master = build_master(hitters, pitchers)

    photos = build_photo_table(raw_h, raw_p)

    h_out = DATA_DIR / "hitter_season_stats_2010_2026_v4.csv"
    p_out = DATA_DIR / "pitcher_season_stats_2010_2026_v4.csv"
    m_out = DATA_DIR / "player_master.csv"
    photo_out = DATA_DIR / "player_photos.csv"

    hitters.to_csv(h_out, index=False, encoding="utf-8-sig")
    pitchers.to_csv(p_out, index=False, encoding="utf-8-sig")
    master.to_csv(m_out, index=False, encoding="utf-8-sig")
    photos.to_csv(photo_out, index=False, encoding="utf-8-sig")

    print(f"\n저장: {h_out}  ({len(hitters)}행, {hitters.player_name.nunique()}명)")
    print(f"저장: {p_out}  ({len(pitchers)}행, {pitchers.player_name.nunique()}명)")
    print(f"저장: {m_out}  ({len(master)}명)")
    print(f"저장: {photo_out}  ({len(photos)}명, 마스터 대비 "
          f"{photos.player_id.isin(master.player_id).sum()}명 매칭)")

    print("\n포지션 분포 (마스터)")
    print(master["position"].value_counts(dropna=False).to_string())

    print("\n투수 역할 분포 (2026)")
    print(pitchers[pitchers.collect_year == 2026]["pitcher_role"].value_counts().to_string())


if __name__ == "__main__":
    main()
