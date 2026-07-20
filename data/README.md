# 데이터 구조

## 정식 사용 파일 (canonical)

| 파일 | 용도 | 생성 스크립트 |
|------|------|---------------|
| `hitter_training_v5.csv`, `pitcher_training_v5.csv` | 모델 학습 입력 (스타성 피처 포함) | `scripts/make_star_features.py` |
| `hitter_training_v7.csv`, `pitcher_training_v7.csv` | Streamlit 앱 표시/추론용 (v5 + 추가 파생 컬럼, 피처 일부는 앱에서 재계산) | 수동 보강 (재현 스크립트 소실, 아래 "알려진 한계" 참고) |
| `hitter_season_stats_2015_2026_v2.csv`, `pitcher_season_stats_2015_2026_v2.csv` | 연도별 시즌 스탯 (앱의 미래 FA 예측·상세 스탯 탭에 사용) | 원본 생성 스크립트 소실 (아래 참고) |
| `fa_contracts_v3.csv` | FA 계약 실적 (Y값, 140건) | 수집 노트북 (`notebooks/`) |
| `star_features_hitter.csv`, `star_features_pitcher.csv` | MVP·골든글러브·국가대표 등 스타성 원천 데이터 | `scripts/make_star_features.py` |
| `teams.csv`, `position_need.csv` | 구단별 제시가 보정용 메타데이터 | 수동 작성 |
| `future_fa_candidates.csv` | 27~30년 FA 예정 선수 목록 | 앱 실행 시 `get_future_fa_candidates()`로 동적 생성(캐시) |
| `naver_hitter_2013_2026_raw_all.csv`, `naver_pitcher_2013_2026_raw_all.csv` | 네이버 스포츠 API 원천 수집 데이터 (가장 넓은 범위) | `scripts/crawl_extend.py` |

## 모델이 실제로 쓰는 피처

`models/*_model_meta.pkl`의 `features` 리스트 기준으로, 학습에 실제 사용되는 피처는 v5 원본 컬럼 + `app/app.py`의
`engineer_hitter_features()` / `engineer_pitcher_features()`가 로드 시점에 계산하는 파생 피처(`market_level`,
`age_squared`, `prime_years_left`, `star_x_war`, `war_sum_sq`, 포지션 백분위 등)다.
v7에 있는 `war_trend`, `ops_trend`, `woba_trend`, `war_std`, `ops_std`, `war_peak`, `ops_peak`, `hr_peak`, `ops_sq`, `is_prime`
컬럼은 실험 단계에서 추가됐으나 현재 모델 피처 리스트에는 포함되지 않는 미사용 컬럼이다.

## data/archive/

과거 버전(v2~v4, v6)과 후속 파일로 대체된 원천 데이터(`_2013_2016`, `_2015_2026` 원본, non-v2 시즌 스탯)를 보관.
재현성 참고용으로만 남기고 신규 코드에서 참조하지 않는다.

## 알려진 한계 (재현성 갭)

- `hitter_training_v5.csv → v7.csv`, `naver_*_raw_all.csv → *_season_stats_*_v2.csv` 변환 스크립트는
  원 저장소에 보존되어 있지 않다. 원본 데이터를 처음부터 다시 수집·가공해야 하는 경우 해당 변환 로직을
  새로 작성해야 한다. (Day2 이후 데이터 파이프라인 스크립트화 권장)
