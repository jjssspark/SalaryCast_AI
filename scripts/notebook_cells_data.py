"""01·02 노트북 셀 내용 — 데이터 확인과 전처리.

셀 본문이 길어 gen_notebooks.py에서 떼어냈다. 한 파일 800줄 규약을 넘겼다.
"""

from __future__ import annotations

from scripts.notebook_common import SETUP, code, md


def notebook_01() -> list[dict]:
    C = [md("""
# 01. 데이터 확인 — StoveLens AI (SalaryCast_AI)

KBO FA 선수의 연평균 계약금을 예측하기 위해 모은 원천 데이터를 점검한다.

확인할 것
1. 네이버 스포츠에서 받은 시즌 기록이 **연도별로 빠짐없이** 들어왔는가
2. 정제한 시즌 스탯에 결측이 어디에 있는가
3. 정답이 될 FA 계약 데이터가 몇 건이고 어떤 분포인가

> 이 노트북은 데이터를 **읽기만** 한다. 수집은 `scripts/crawl_naver_v2.py`,
> 정제는 `scripts/build_season_stats_v3.py`가 한다.
""")]

    C.append(md("## 0. 환경 설정"))
    C.append(code(SETUP))

    C.append(md("""
## 1. 원천 데이터 로드

`_raw_all`은 처음 수집한 것, `_raw_v2`는 다시 수집한 것이다.
두 파일을 같이 두는 이유는 다음 절에서 비교하기 위해서다.

이 비교는 2013~2026 구간으로 한다. 지금 쓰는 시즌 스탯은 나중에 2010년까지
더 넓혀 받은 `_2010_2026_v4`이고, 아래에서 로드하는 것도 그쪽이다.
"""))
    C.append(code("""
raw_old = pd.read_csv(DATA / "naver_hitter_2013_2026_raw_all.csv")
raw_new = pd.read_csv(DATA / "naver_hitter_2013_2026_raw_v2.csv")
seasons_h = pd.read_csv(DATA / "hitter_season_stats_2010_2026_v4.csv")
seasons_p = pd.read_csv(DATA / "pitcher_season_stats_2010_2026_v4.csv")
fa = pd.read_csv(DATA / "fa_contracts_v7.csv")

for label, frame in [("타자 원본(구)", raw_old), ("타자 원본(신)", raw_new),
                     ("타자 시즌스탯 v4", seasons_h), ("투수 시즌스탯 v4", seasons_p),
                     ("FA 계약 v7", fa)]:
    print(f"{label:16s} {frame.shape[0]:5d}행 {frame.shape[1]:3d}컬럼")
"""))

    C.append(md("""
## 2. 연도별 수집 인원 — 처음 수집은 70%가 비어 있었다

처음 수집은 정렬 기준별 상위 N명만 가져오는 구조에 `pageSize=100`이 걸려 있었다.
부상이나 백업으로 출전이 적은 선수는 **어느 정렬 기준에서도 상위권에 못 들어**
통째로 빠졌다. 홍창기의 2025 시즌이 없어서 발견했다 (TROUBLESHOOTING TS-001).
"""))
    C.append(code("""
old_by_year = raw_old.groupby("year")["playerId"].nunique()
new_by_year = raw_new.groupby("year")["playerId"].nunique()
compare = pd.DataFrame({"기존 수집": old_by_year, "재수집": new_by_year}).fillna(0).astype(int)
compare["누락률"] = (1 - compare["기존 수집"] / compare["재수집"]).round(3)
display(compare.tail(8))

ax = compare[["기존 수집", "재수집"]].plot(
    kind="bar", figsize=(11, 4), color=["#c9ccd1", "#1f6feb"], width=0.78)
ax.set_title("연도별 수집된 타자 수 — 기존 vs 재수집")
ax.set_xlabel("연도"); ax.set_ylabel("선수 수")
ax.legend(title=None)
plt.tight_layout()
plt.savefig(CHARTS / "01_collection_coverage.png", bbox_inches="tight")
plt.show()

print(f"재수집으로 늘어난 타자 시즌 기록: {len(raw_old)} -> {len(raw_new)}행")
"""))

    C.append(md("""
## 3. 결측 확인

정제된 시즌 스탯에서 값이 빈 컬럼을 본다.
비율 스탯(`babip`, `wrc_plus` 등)은 타석이 아주 적은 시즌에 계산이 안 돼 비는 경우가 있다.
"""))
    C.append(code("""
def missing_report(frame, label):
    ratio = (frame.isna().mean() * 100).sort_values(ascending=False)
    ratio = ratio[ratio > 0]
    print(f"[{label}] 결측 있는 컬럼 {len(ratio)}개 / 전체 {frame.shape[1]}개")
    return ratio

miss_h = missing_report(seasons_h, "타자 시즌스탯")
miss_p = missing_report(seasons_p, "투수 시즌스탯")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, ratio, title in [(axes[0], miss_h, "타자"), (axes[1], miss_p, "투수")]:
    if len(ratio):
        ratio.head(10).plot(kind="barh", ax=ax, color="#e05c5c")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "결측 없음", ha="center", va="center")
    ax.set_title(f"{title} 결측 비율 상위 (%)")
plt.tight_layout()
plt.savefig(CHARTS / "01_missing.png", bbox_inches="tight")
plt.show()
"""))

    C.append(md("""
## 4. 정답(타깃) 데이터 — FA 계약 210건

`annual_avg_salary`가 예측 타깃이다. 단위는 **억 원**이고,
총액이 아니라 계약 기간으로 나눈 연평균이다.
"""))
    C.append(code("""
print(f"FA 계약 {len(fa)}건 / {fa.fa_year.min()}~{fa.fa_year.max()}년")
# position은 코드다 — P 투수, C 포수, 1B/2B/3B/SS 내야, OF 외야, IF 내야 유틸
print(f"타자 {(fa.position != 'P').sum()}건 · 투수 {(fa.position == 'P').sum()}건")
display(fa.annual_avg_salary.describe().round(2).to_frame("연평균 계약금(억)"))
display(fa.nlargest(5, "annual_avg_salary")[
    ["player_name", "fa_year", "position", "contract_years",
     "total_contract_amount", "annual_avg_salary"]])
"""))
    C.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

axes[0].hist(fa.annual_avg_salary, bins=25, color="#1f6feb", edgecolor="white")
axes[0].set_title("연평균 계약금 분포")
axes[0].set_xlabel("억 원"); axes[0].set_ylabel("선수 수")

by_year = fa.groupby("fa_year")["annual_avg_salary"].agg(["mean", "count"])
axes[1].bar(by_year.index, by_year["mean"], color="#2ea043")
axes[1].set_title("FA 연도별 평균 계약금")
axes[1].set_xlabel("FA 연도"); axes[1].set_ylabel("억 원")

plt.tight_layout()
plt.savefig(CHARTS / "01_target_distribution.png", bbox_inches="tight")
plt.show()

print(f"왜도(skew) {fa.annual_avg_salary.skew():.2f} — 오른쪽으로 길게 늘어진 분포다.")
print("소수의 대형 계약이 평균을 끌어올린다. 03에서 로그 변환을 검토한다.")
"""))

    C.append(md("""
## 정리

- 처음 수집은 **연도당 선수의 상당수가 빠져 있었다.** 재수집으로 복구했다 (TS-001)
- 결측은 비율 스탯에 몰려 있고, 타석이 극히 적은 시즌에서 생긴다
- 타깃은 210건이고 **오른쪽으로 크게 치우친 분포**다 → 로그 변환 검토 대상

다음: `02_preprocessing.ipynb` — 이 데이터로 학습셋 v10을 만든다.
"""))
    return C


def notebook_02() -> list[dict]:
    C = [md("""
# 02. 전처리 — 학습셋 v10 만들기

01에서 확인한 데이터를 모델이 먹을 수 있는 형태로 바꾼다.

이 노트북에서 다루는 것
1. **생년 교정** — FA 원본의 생년이 틀려 나이 피처가 오염돼 있었다
2. **3년 집계 피처** — FA 직전 3시즌을 하나의 행으로
3. **스타성 피처** — MVP·골든글러브·국가대표 수상 이력
4. **범주형 인코딩** — 투수 역할(선발/셋업/마무리)을 숫자로
5. **리그 대비 백분위** — 같은 스탯도 그 해 리그 수준에 따라 값어치가 다르다
6. **데이터 누수 컬럼 제외**
""")]

    C.append(md("## 0. 환경 설정"))
    C.append(code(SETUP))

    C.append(md("""
## 1. 생년 교정 (TS-002)

FA 계약 원본(`fa_contracts_v3`)의 `age_at_fa`가 실제 생년과 안 맞았다.
위키데이터와 네이버 프로필로 대조해 고친 것이 `fa_contracts_v4`다.
나이는 연봉과 강하게 얽히는 피처라 그대로 두면 학습 전체가 틀어진다.
"""))
    C.append(code("""
fa_old = pd.read_csv(DATA / "fa_contracts_v3.csv")
fa_new = pd.read_csv(DATA / "fa_contracts_v4.csv")

merged = fa_old.merge(fa_new, on=["player_name", "fa_year"], suffixes=("_old", "_new"))
merged["나이차"] = merged["age_at_fa_new"] - merged["age_at_fa_old"]
changed = merged[merged["나이차"] != 0]

print(f"대조 {len(merged)}건 중 나이가 바뀐 계약 {len(changed)}건")
if len(changed):
    print(f"최대 차이 {changed['나이차'].abs().max():.0f}세")
    display(changed.reindex(changed["나이차"].abs().sort_values(ascending=False).index)[
        ["player_name", "fa_year", "age_at_fa_old", "age_at_fa_new", "나이차"]].head())

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = range(int(changed["나이차"].min()), int(changed["나이차"].max()) + 2)
    ax.hist(changed["나이차"], bins=bins, color="#e0a05c", edgecolor="white")
    ax.set_title("생년 교정으로 달라진 FA 시점 나이")
    ax.set_xlabel("교정 후 - 교정 전 (세)"); ax.set_ylabel("계약 건수")
    plt.tight_layout()
    plt.savefig(CHARTS / "02_age_correction.png", bbox_inches="tight")
    plt.show()
"""))

    C.append(md("""
## 2. 학습셋 v10 로드

3년 집계는 `scripts/build_training_v8.py`가 만든다.
FA 직전 3시즌을 평균·합으로 접고, 부상 등으로 시즌이 비면 있는 시즌만 쓴다.
여기서는 만들어진 결과를 확인한다.
"""))
    C.append(code("""
hitters = pd.read_csv(DATA / "hitter_training_v10.csv")
pitchers = pd.read_csv(DATA / "pitcher_training_v10.csv")

print(f"타자 {hitters.shape[0]}명 / {hitters.shape[1]}컬럼")
print(f"투수 {pitchers.shape[0]}명 / {pitchers.shape[1]}컬럼")
display(hitters[["player_name", "fa_year", "age_at_fa", "war_3yr_sum",
                 "ops_3yr_avg", "star_score", "annual_avg_salary"]].head())
"""))

    C.append(md("""
## 3. 스타성 피처

`national_team`, `mvp_count`, `golden_glove_count`를 합쳐 `star_score`를 만든다.
이 값들은 예전에 **출처 없이 손으로 적혀 있었고 상당수가 사실과 달랐다** (TS-004).
지금은 위키백과의 연도별 수상자 명단과 참가 선수 분류에서 자동으로 만든다 (TS-010).

> 아래 '대회별 등재 인원'은 실제 엔트리 수가 아니다.
> `kbo_awards.csv`는 (선수, 연도, 수상종류)로 중복을 지운다.
> 한 해에 두 대회를 뛴 선수는 한 번만 남는다 — 국가대표 **경력 연도** 수를
> 세는 것이 목적이라 같은 해를 두 번 세면 안 되기 때문이다.
> 그래서 2023 APBC는 엔트리 26명이지만 5명이 같은 해 WBC와 겹쳐 21로 보인다.
"""))
    C.append(code("""
awards = pd.read_csv(DATA / "kbo_awards.csv")
print("수상 기록", len(awards), "건")
display(awards.groupby("award").size().to_frame("건수"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

nt = awards[awards.award == "NT"].groupby("source").size().sort_values()
nt.plot(kind="barh", ax=axes[0], color="#1f6feb")
axes[0].set_title("국가대표 — 대회별 등재 인원"); axes[0].set_xlabel("명")

star = pd.concat([hitters["star_score"], pitchers["star_score"]])
axes[1].hist(star, bins=range(0, int(star.max()) + 2), color="#8957e5", edgecolor="white")
axes[1].set_title("FA 선수 star_score 분포"); axes[1].set_xlabel("점수"); axes[1].set_ylabel("명")

plt.tight_layout()
plt.savefig(CHARTS / "02_star_features.png", bbox_inches="tight")
plt.show()

print(f"수상 이력이 전혀 없는 FA 선수 {(star == 0).sum()}명 / {len(star)}명")
"""))

    C.append(md("""
## 4. 범주형 인코딩 — 투수 역할

투수는 선발(SP)·셋업(SU)·마무리(CL)의 가치 기준이 다르다.
세이브가 많은 마무리와 이닝이 많은 선발을 같은 잣대로 보면 안 된다.

역할별로 **모델을 3개 만들지 않고 피처 하나로 넣는다.**
KBO FA 투수는 표본이 72명뿐이라 3분할하면 한 덩어리가 20명 안팎이 되어
모델이 패턴을 못 잡는다.
"""))
    C.append(code("""
from sklearn.preprocessing import LabelEncoder

print(pitchers["pitcher_role"].value_counts().to_string())

# 방법 1) 라벨 인코딩 — 값 하나로 압축된다. 트리 모델은 이걸로 충분하다.
encoder = LabelEncoder()
role_encoded = encoder.fit_transform(pitchers["pitcher_role"])
print("\\n라벨 인코딩:", dict(zip(encoder.classes_,
                              encoder.transform(encoder.classes_).tolist())))

# 방법 2) 원핫 인코딩 — 순서가 없는 범주라 선형 모델에는 이쪽이 맞다.
onehot = pd.get_dummies(pitchers["pitcher_role"], prefix="role", drop_first=True)
display(onehot.head())

print("라벨 인코딩은 없는 순서(0 < 1 < 2)를 만든다.")
print("트리 모델은 분기로 나누므로 상관없지만, 선형 모델에는 원핫이 맞다.")
"""))
    C.append(code("""
fig, ax = plt.subplots(figsize=(7, 3.4))
order = [r for r in ["SP", "SU", "CL"] if r in set(pitchers["pitcher_role"])]
sns.boxplot(data=pitchers, x="pitcher_role", y="annual_avg_salary",
            order=order, ax=ax, color="#9ec5f7")
ax.set_title("투수 역할별 연평균 계약금")
ax.set_xlabel("역할 (SP 선발 / SU 셋업 / CL 마무리)"); ax.set_ylabel("억 원")
plt.tight_layout()
plt.savefig(CHARTS / "02_pitcher_role.png", bbox_inches="tight")
plt.show()
"""))

    C.append(md("""
## 5. 리그 대비 백분위

OPS .850은 타고투저 해에는 평범하고 투고타저 해에는 최상위다.
그래서 원래 스탯과 함께 **그 해 리그에서 몇 %인지**를 같이 넣는다.
전체 기준과 포지션 기준 두 가지를 만든다.
"""))
    C.append(code("""
from src.features import HITTER_PCT_COLS, PITCHER_PCT_COLS, add_percentile_ranks

print("타자 백분위 대상:", HITTER_PCT_COLS)
print("투수 백분위 대상:", PITCHER_PCT_COLS)

hitters_eng = add_percentile_ranks(hitters.copy(), HITTER_PCT_COLS, "position")
pitchers_eng = add_percentile_ranks(pitchers.copy(), PITCHER_PCT_COLS, "pitcher_role")

added = [c for c in hitters_eng.columns if c not in hitters.columns]
print(f"\\n타자에 늘어난 컬럼 {len(added)}개")
display(hitters_eng[["player_name", "war_3yr_sum",
                     "war_3yr_sum_all_pct", "war_3yr_sum_pos_pct"]].head())
"""))

    C.append(md("""
## 6. 데이터 누수 컬럼 제외

정답을 간접적으로 흘리는 컬럼은 반드시 빼야 한다.
`total_contract_amount`(총액)와 `contract_years`(계약 연수)는
나누면 바로 타깃이 나온다. 넣으면 지표가 치솟지만 아무 쓸모가 없다.
실제 서비스에서는 **계약 전에** 예측해야 하므로 이 값들을 알 수 없다.
"""))
    C.append(code("""
EXCLUDED = {
    "annual_avg_salary",       # 타깃 그 자체
    "total_contract_amount",   # 타깃 x 계약연수
    "contract_years",          # 타깃과 총액을 잇는 값
    "player_name", "player_id", "team", "position", "pitcher_role", "seasons_used",
}

def feature_columns(frame):
    return [c for c in frame.columns
            if c not in EXCLUDED and pd.api.types.is_numeric_dtype(frame[c])]

h_cols = feature_columns(hitters_eng)
p_cols = feature_columns(pitchers_eng)
print(f"타자 모델 입력 피처 {len(h_cols)}개")
print(f"투수 모델 입력 피처 {len(p_cols)}개")
print("\\n제외한 컬럼:", sorted(EXCLUDED & set(hitters_eng.columns)))
"""))

    C.append(md("""
## 정리

| 단계 | 한 일 |
|---|---|
| 생년 교정 | FA 원본 생년 오류를 고쳐 나이 피처 오염 제거 (TS-002) |
| 3년 집계 | FA 직전 3시즌을 한 행으로 |
| 스타성 | 위키백과에서 자동 수집한 수상·국가대표 이력 (TS-004·TS-010) |
| 인코딩 | 투수 역할을 라벨/원핫으로 — 별도 모델 3개 대신 피처 하나 |
| 백분위 | 같은 스탯도 그 해 리그 기준으로 다시 |
| 누수 제거 | 총액·계약연수 제외 |

다음: `03_eda.ipynb` — 만들어진 피처와 타깃의 관계를 본다.
"""))
    return C


