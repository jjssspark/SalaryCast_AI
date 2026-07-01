# SalaryCast AI (StoveLens AI) — 프로젝트 컨텍스트

## 프로젝트 개요

프로젝트명: **SalaryCast_AI**
모델명: **StoveLens AI**
목표: KBO FA 선수의 최근 성적 데이터 기반으로 예상 연평균 계약금 예측 + 구단별 적정 제시가 추천

핵심 기능:
1. KBO FA 선수 연평균 계약금 예측
2. 타자/투수 분리 모델링
3. 스타성·포지션 프리미엄·투수 역할 분류 반영
4. Streamlit 기반 시연 서비스

예측 타깃: `annual_avg_salary` (단위: 억 원)
평가 지표: R², MSE, RMSE

---

## 제출/발표 일정 ⚠️

| 일정 | 내용 |
|------|------|
| **6/24(수) 09:00** | 개인별 발표 + 시연 (팀별 4명) |
| 발표 후 이메일 제출 | ahnhg2000@gmail.com |

제출 파일 (zip으로 묶어서):
- `홍길동_머신러닝프로젝트.hwp`
- `홍길동_머신러닝프로젝트.ipynb`
- `홍길동_머신러닝프로젝트.py`

> 발표/시연용 파일명은 반드시 **영문 또는 영문+숫자** 조합 사용

시연 방법 (3가지 중 택1):
- 방법1: Streamlit Cloud — https://share.streamlit.io
- 방법2: ngrok
- 방법3: 로컬 IP — `streamlit run app.py` 후 `192.168.x.x:8501`

보고서 양식: 교재 폴더 내 `머신러닝 산출물_20250312(홍길동v0.1).hwp` 참조

---

## 일별 작업 계획 (수~일)

### ✅ 수요일 (완료)
- 프로젝트 세팅 (폴더, GitHub, requirements.txt)
- 네이버 KBO 타자/투수 데이터 수집 (2015~2026)
- FA 계약 데이터 140건 구축 (`fa_contracts_v3.csv`)
- 학습 데이터셋 v4 생성 (타자 80명/36컬럼, 투수 43명/22컬럼)

### ✅ 목요일 (완료)
- 데이터 누수 발견 및 제거 (`total_contract_amount`, `contract_years` 등)
- 타자/투수 모델 v2 학습
- 타자 R² 0.388 / 투수 R² 0.756 확인
- fa_year가 투수 Feature Importance 1위 → 금요일 제거 실험 필요

### 🔲 금요일 (오늘)
1. **fa_year 제거 실험** — 타자/투수 각각 제거 전/후 R²·RMSE 비교표 생성
2. **스타성 변수 추가** — `star_features.csv` 생성 → `hitter_training_v5.csv`
3. **pitcher_role 추가** — SP/SU/CL 분류 → `pitcher_training_v5.csv`
4. **최종 모델 선정** — R², RMSE 기준 + 발표 설명 가능성
5. **모델 저장** — `models/hitter_model.pkl`, `pitcher_model.pkl` 등

### 🔲 토요일
1. **구단 데이터 정리** — `teams.csv`, `position_need.csv`
2. **구단별 제시가 보정 공식 설계** — 예측 연봉 × 구단 보정 계수
3. **Streamlit 기본 화면** — 선수 선택 → 예측 연봉 → 구단별 제시가 출력
4. **시각화** — 실제값 vs 예측값, Feature Importance, 구단별 제시가 막대그래프

### 🔲 일요일 (제출 마감)
1. **보고서 작성** — `.hwp` (양식 참조)
2. **제출 파일 정리** — `홍길동_머신러닝프로젝트.hwp / .ipynb / .py` → zip
3. **Streamlit 실행 테스트** — 에러 없이 예측 동작 확인
4. **GitHub 정리** — requirements.txt 갱신, README.md, 최종 push
5. **이메일 제출** — ahnhg2000@gmail.com

---

## 현재 진행 상태 (6/18 기준)

- [x] 프로젝트 세팅 (폴더, GitHub, requirements.txt)
- [x] 타자/투수 원천 데이터 수집 (2015~2026)
- [x] 시즌 스탯 정제 (v2)
- [x] FA 계약 데이터 구축 — 140건 (`fa_contracts_v3.csv`)
- [x] 학습 데이터셋 생성 — 타자 80명/36컬럼, 투수 43명/22컬럼 (v4)
- [x] 데이터 누수 발견 및 제거 (아래 참고)
- [x] 타자/투수 모델 v2 학습 완료
- [ ] fa_year 제거 실험
- [ ] 스타성 변수 추가 (hitter_training_v5)
- [ ] pitcher_role (SP/SU/CL) 피처 추가 (pitcher_training_v5)
- [ ] 최종 모델 선정 및 저장
- [ ] 구단별 2차 보정 로직
- [ ] Streamlit 앱 구현
- [ ] 보고서 작성 및 제출 파일 정리

---

## 현재 모델 성능 (v2 기준)

### 타자
| 모델 | R² | RMSE |
|------|-----|------|
| Linear Regression | 0.384 | 6.01 |
| Random Forest | 0.388 | 5.99 |

Feature Importance 상위: `war_3yr_sum`, `run_3yr_avg`, `double_3yr_avg`, `ops_last_year`, `woba_3yr_avg`, `war_last_year`, `age_at_fa`

해석: 성적만으로 38% 설명 → 스타성·비성적 요소 추가 필요

### 투수
| 모델 | R² | RMSE |
|------|-----|------|
| Linear Regression | 0.756 | 3.30 |
| Random Forest | 0.570 | 4.37 |

Feature Importance 상위: `fa_year`(1위⚠️), `innings_3yr_avg`, `hold_3yr_avg`, `era_3yr_avg`, `war_3yr_sum`

해석: fa_year가 1위 → 연봉 인플레이션 효과. 순수 성적 기반 예측을 위해 제거 실험 필요.

---

## 데이터 누수 주의 ⚠️

아래 컬럼은 정답(annual_avg_salary)을 간접적으로 노출하므로 **모델 입력에서 반드시 제외**:
- `annual_avg_salary` (타깃 자체)
- `total_contract_amount`
- `contract_years`
- `player_name`
- `position`
- `team`

---

## Streamlit 서비스 스펙 ⚠️ (핵심 설계)

### 입력
- 선수 이름 텍스트 입력 (직접 검색)
- 소속팀 드롭다운 필터 (2026년 기준 소속팀)

### 검색 후 동작: 두 가지 모드 자동 분기

#### 모드 A — 과거 FA 완료 선수 (27년 이전에 FA 계약 이미 존재)
```
선수 검색
→ fa_contracts_v3.csv에서 가장 최근 FA 계약 데이터 조회
→ 해당 FA 시점 스탯으로 모델 예측
→ 출력:
   ① 예측 연봉 vs 실제 계약 연봉 비교
   ② 예측 근거 (기여한 스탯 Top5 등)
   ③ 해당 선수의 최근 3년 스탯 요약
```

#### 모드 B — 미래 FA 예정 선수 (27~30년도 FA 예정)
```
선수 검색
→ 현재(26년 기준) 스탯으로 모델 예측
→ 출력:
   ① 예측 연봉 (실제치 없으므로 비교 없음)
   ② 구단별 적정 제시가 (10개 구단)
   ③ 예측 근거 (기여한 스탯 Top5 등)
   ④ 해당 선수의 최근 3년 스탯 요약
```

### 판단 기준
- `fa_contracts_v3.csv`에 해당 선수의 FA 기록이 있으면 → 모드 A
- 없으면 (미래 FA 예정) → 모드 B

### 필요한 추가 데이터
- 27~30년도 FA 예정 선수 목록 (`future_fa_candidates.csv`)
  - 컬럼: player_name, team_2026, fa_year_expected, position, player_type

---

## 모델 구조

### 1차 예측 (XGBoost Regressor)
```
선수 입력 데이터 → XGBoost → annual_avg_salary 예측
```

### 2차 보정 (구단별 제시가)
```
예측 연봉
+ 구단별 포지션 필요도 (position_need_score)
+ 샐러리캡 잔여 여유 (salary_cap_space)
+ 윈나우 지수 (win_now_score)
→ 구단별 적정 제시가
```

---

## 데이터 구조

### player_type 구분
- `hitter` — 타자
- `pitcher` — 투수 (별도 모델로 학습)

### 핵심 공통 피처
```
player_name, fa_year, age_at_fa, age_squared, position,
war_3yr_avg, war_3yr_sum, games_3yr_avg, injury_days_3yr
```

### 타자 전용 피처
```
ops_3yr, wrc_plus_3yr, hr_3yr_avg, rbi_3yr_avg,
obp_3yr, slg_3yr, avg_3yr, sb_3yr_avg,
strikeout_rate_3yr, walk_rate_3yr
```

### 투수 전용 피처
```
pitcher_role (SP / SU / CL)  ← 카테고리 피처로 추가 예정
fip_3yr, era_3yr, whip_3yr,
innings_3yr_avg, strikeouts_3yr_avg, walks_3yr_avg,
saves_3yr_avg,    ← CL에 중요
holds_3yr_avg,    ← SU에 중요
starter_ratio     ← SP/RP 연속값
```

---

## 투수 역할(pitcher_role) 설계 결정사항 ⚠️

> **별도 모델 3개(SP/SU/CL) 아님 → `pitcher_role` 피처 하나로 통합**

이유: KBO FA 투수 데이터는 연간 5~10명 수준. 3분할하면 서브셋이 너무 작아 모델이 패턴을 못 잡음.
해결책: `pitcher_role = SP / SU / CL` 카테고리 피처 추가 → XGBoost가 역할별 중요 스탯 자동 학습.

임시 분류 규칙:
- `save_3yr_avg >= 10` → CL (마무리)
- `hold_3yr_avg >= 10` → SU (셋업)
- 그 외 → SP (선발)

---

## 스타성 변수 추가 계획 (hitter_training_v5)

추가할 컬럼:
```
mvp_count, golden_glove_count, allstar_count,
national_team, postseason_experience, star_score
```

star_score 산정 기준:
- MVP: +5
- 골든글러브: +3
- 국가대표: +2
- 올스타: +1
- 포스트시즌 경험: +1

---

## 폴더 구조

```
SalaryCast_AI/
├── data/
│   ├── naver_pitcher_2015_2026_raw_all.csv   # 투수 원시 데이터
│   ├── naver_hitter_2015_2026_raw_all.csv    # 타자 원시 데이터
│   ├── pitcher_season_stats_2015_2026_v2.csv
│   ├── hitter_season_stats_2015_2026_v2.csv
│   ├── fa_contracts_v3.csv                   # FA 계약 140건 (Y값)
│   ├── pitcher_training_v4.csv               # 투수 43명, 22컬럼
│   ├── hitter_training_v4.csv                # 타자 80명, 36컬럼
│   ├── pitcher_training_v5.csv               # pitcher_role 추가 예정
│   ├── hitter_training_v5.csv                # 스타성 추가 예정
│   ├── star_features.csv                     # 스타성 데이터 (생성 예정)
│   ├── teams.csv                             # 구단 데이터
│   └── position_need.csv                     # 포지션 필요도 (생성 예정)
├── notebooks/
│   ├── data_collect_test.ipynb
│   ├── fa_contract_collect.ipynb
│   ├── make_training_dataset.ipynb
│   ├── model_train.ipynb                     # 실험 기록용 (수정 X)
│   └── model_train_v2.ipynb                  # 현재 최신
├── models/
│   ├── hitter_model.pkl                      # 저장 예정
│   ├── pitcher_model.pkl                     # 저장 예정
│   ├── hitter_features.pkl                   # 저장 예정
│   └── pitcher_features.pkl                  # 저장 예정
├── app/
│   └── app.py                                # Streamlit 앱
├── crawl_pitcher_all.py                      # 투수 재크롤링 스크립트
└── CLAUDE.md
```

---

## 데이터 수집 방법

**네이버 스포츠 API**
```
GET https://api-gw.sports.naver.com/statistics/categories/kbo/seasons/{year}/players
params:
  playerType: PITCHER or HITTER
  sortField: pitcherWar / pitcherSave / pitcherHold 등
  gameType: REGULAR_SEASON
  page, pageSize
```

투수 수집 시 `pitcherWar`만으로 정렬하면 계투·마무리 누락 가능.
→ `crawl_pitcher_all.py`: 7개 sort 기준 반복 수집 후 dedup.

---

## 알려진 이슈

### matplotlib 한글 폰트 경고
```
WARNING: Font family 'NanumGothic' not found.
UserWarning: Glyph ... missing from font(s) DejaVu Sans.
```
해결: 노트북 최상단에 아래 코드 추가
```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔고딕 설치 후
# !apt-get install -y fonts-nanum  (Colab/Ubuntu)
# !pip install koreanize-matplotlib  (간편 설치)
import koreanize_matplotlib  # pip install koreanize-matplotlib
```
또는 영문 라벨로 대체하는 것도 방법.

---

## Streamlit UI/UX 설계 원칙 ⚠️

### 핵심 원칙: 보고서용 ≠ 서비스용
사용자는 야구 팬 / 구단 관계자. MSE, MAE, R², Feature Importance 같은 용어 절대 노출 금지.
디자인은 심플하고 간지나게. 기능은 최소화, 직관성 최대화.

---

### 전체 화면 흐름
```
① 메인 화면
   └─ 선수 이름 검색창 (+ 팀 필터 드롭다운, 선택사항)
         ↓ 검색
② 선수 카드 자동 분기
   ├─ 모드 A: FA 완료 선수 (fa_contracts_v3.csv에 기록 있음)
   └─ 모드 B: 미래 FA 예정 선수 (future_fa_candidates.csv에 있음)
```

---

### 모드 A — FA 완료 선수 화면

```
┌──────────────────────────────────────────────────────┐
│  홍길동   LG 트윈스 · 외야수 · FA 2024년              │
└──────────────────────────────────────────────────────┘

[ 연봉 비교 ]
  실제 계약   ████████████████  연평균 25억
  AI 예측     ████████████░░░░  연평균 22억

  → "실제 계약이 예측보다 3억 높습니다. 다소 고평가된 계약입니다."
    or "예측과 거의 일치합니다. 적정 계약으로 평가됩니다."
    or "실제 계약이 예측보다 낮습니다. 저평가된 계약입니다."

[ 이 선수의 연봉을 결정한 핵심 요소 ]
  🏆  팀 기여도(WAR) 3년 합계     14.2   (리그 상위 5%)
  📈  출루율 3년 평균              .405   (리그 상위 10%)
  💪  홈런 3년 평균                22개   (리그 상위 15%)
  (각 스탯 위에 마우스 올리면 툴팁: 간략 설명 표시)

[ 최근 3년 주요 스탯 요약 ]
  표 형태로 깔끔하게

[ 분석 상세 보기 ▼ ] ← 토글 (기본 접힘)
  교수님 보고서용: R², RMSE, Feature Importance 그래프 등
```

---

### 모드 B — 미래 FA 예정 선수 화면

```
┌──────────────────────────────────────────────────────┐
│  홍창기   LG 트윈스 · 외야수 · FA 예정 2027년         │
└──────────────────────────────────────────────────────┘

[ AI 예상 연봉 ]
  💰  약 25억 원 / 년

[ 이 선수의 연봉을 결정한 핵심 요소 ]
  🏆  팀 기여도(WAR) 3년 합계     14.2
  📈  출루율 3년 평균              .405
  🏃  도루 3년 평균                24개

[ 구단별 예상 제시가 ]
  LG     ████████░░  22억 ~ 27억   🔥 잔류 가능성 높음
  두산   ██████░░░░  18억 ~ 23억
  NC     █████░░░░░  15억 ~ 20억
  ...

[ 최근 3년 주요 스탯 요약 ]

[ 분석 상세 보기 ▼ ] ← 토글 (기본 접힘)
```

---

### 스탯 툴팁 목록 (마우스 오버 시 표시)
| 스탯 | 툴팁 설명 |
|------|---------|
| WAR | 이 선수가 없었다면 팀이 잃었을 승리 수. 높을수록 팀에 중요한 선수. |
| OPS | 출루율 + 장타율. 타자의 전반적인 공격력을 나타내는 지표. |
| ERA | 9이닝당 자책점 평균. 낮을수록 좋은 투수. |
| FIP | 수비 영향을 제외한 순수 투수 능력치. ERA보다 실력을 더 잘 반영. |
| WHIP | 이닝당 출루 허용 수. 낮을수록 안정적인 투수. |
| wRC+ | 100이 리그 평균. 높을수록 평균보다 뛰어난 타자. |
| 출루율(OBP) | 타석에서 아웃 당하지 않은 비율. 높을수록 공격 기회를 많이 만드는 타자. |
| 장타율(SLG) | 타석당 평균 베이스 진루 수. 높을수록 장타력이 강한 타자. |

---

### 금지 표현 (사용자 화면)
- ❌ R², RMSE, MSE, MAE, 잔차, 정규화
- ❌ XGBoost, RandomForest, LinearRegression
- ❌ Feature Importance
- ✅ 대신: "AI가 예측한", "핵심 요소", "예상 연봉", "적정 계약 / 고평가 / 저평가"

### 기술 지표 노출 위치
`분석 상세 보기 ▼` 토글 안에만 표시 (발표·보고서용)

---

## 수업 커리큘럼 반영 요구사항 ⚠️

수업에서 배운 아래 기법들을 프로젝트 코드에 반드시 사용할 것.
보고서/발표에서 "이 기법을 왜 썼는가"를 설명할 수 있어야 함.

### 회귀 모델 (노트북1)
| 기법 | 프로젝트 적용 위치 |
|------|-------------------|
| 선형회귀 (LinearRegression) | 타자/투수 베이스라인 모델 |
| 다중선형회귀 | 복수 피처 사용 → 다중선형회귀 형태 |
| 다항회귀 (PolynomialFeatures) | age_at_fa의 비선형 효과 반영 (`age_squared` 이미 포함) |
| 경사하강법 | 모델 최적화 원리 설명용 (보고서) |

### 분류/앙상블 (노트북1)
| 기법 | 프로젝트 적용 위치 |
|------|-------------------|
| 랜덤포레스트 (RandomForestRegressor) | 타자/투수 메인 모델 후보 |
| 의사결정나무 (DecisionTree) | 피처 중요도 직관적 설명용 |
| KNN | 비교 실험용 (선택) |

### 성능 지표 (노트북1)
모든 모델 평가 시 아래 4개 지표를 함께 출력할 것:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
```

### 인코딩 (노트북2)
| 기법 | 프로젝트 적용 위치 |
|------|-------------------|
| 라벨인코딩 (LabelEncoder) | pitcher_role (SP/SU/CL) 숫자 변환 |
| 원핫인코딩 (pd.get_dummies) | pitcher_role을 모델 입력으로 변환 |
| sklearn OneHotEncoder | 대안 방법 (비교 설명 가능) |

```python
# 수업 방식 그대로 사용 예시
import pandas as pd
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['pitcher_role_encoded'] = le.fit_transform(df['pitcher_role'])

# 또는 원핫
df = pd.get_dummies(df, columns=['pitcher_role'], drop_first=True)
```

### XGBoost (노트북2 — 자전거 수요량 예측 예제 참고)
```python
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
```

### 모델 저장 (노트북2 — joblib)
```python
import joblib
joblib.dump(model, 'models/hitter_model.pkl')
joblib.dump(feature_cols, 'models/hitter_features.pkl')
# 불러올 때
model = joblib.load('models/hitter_model.pkl')
```

### 시각화 필수 항목 (노트북1+2)
보고서 및 Streamlit에 반드시 포함:
1. 실제값 vs 예측값 산점도
2. Feature Importance 막대그래프
3. 모델별 R²/RMSE/MAE 비교 표 or 그래프
4. 구단별 적정 제시가 막대그래프

```python
# 한글 폰트 (반드시 노트북 최상단에)
import koreanize_matplotlib  # pip install koreanize-matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 개발 원칙

1. `model_train.ipynb`는 실험 기록으로 두고, 최종 분석은 `model_train_v2.ipynb` 또는 `clean_model_train.ipynb`에서 진행
2. 데이터 누수 방지: `total_contract_amount`, `contract_years`는 모델 입력 제외
3. 예측 타깃: `annual_avg_salary` 단독
4. 타자/투수 모델 반드시 분리
5. 모든 모델 평가는 R², MSE, RMSE 출력
6. Feature Importance 저장 및 시각화
7. 데이터 파일은 버전 suffix 관리 (`_v2`, `_v3` 등). 기존 파일 덮어쓰기 금지
8. 모델 저장: `models/` 폴더, 파일명에 날짜 포함
9. 인코딩: 모든 CSV는 `utf-8-sig` (한글 깨짐 방지)
10. 연봉 단위: **억 원**. 예) 10억 = 10.0

---

## 보고서 구조 (일요일 목표)

1. 개요 및 현황
2. 추진 배경 및 목적
3. 데이터 수집
4. 데이터 전처리
5. EDA
6. 모델 정의
7. 모델 학습
8. 모델 평가
9. Feature Importance 분석
10. Streamlit 서비스 화면
11. 한계점 및 개선 방향

반드시 포함: 데이터 출처, FA 140건 구축, 타자/투수 분리, 데이터 누수 발견·제거, R²/MSE/RMSE, 최종 모델 선정 이유
