# 데이터 구조

## 앱과 학습이 실제로 읽는 파일

| 파일 | 용도 | 생성 스크립트 |
|------|------|---------------|
| `fa_contracts_v7.csv` | FA 계약 실적 (Y값, 210건 · 2014~2026) | `add_player_ids.py` → `add_fa_2016_2017.py` → `add_fa_2014_2015.py` |
| `future_fa_candidates_v2.csv` | 조사해서 확인한 FA 예정 선수 42명 | 수동 조사 + `scripts/add_player_ids.py` (player_id 부여) |
| `fa_eligibility_estimated.csv` | 위 목록에 없는 현역의 FA 자격 연도 추정값 | `scripts/estimate_fa_eligibility.py` |
| `non_fa_extensions.csv` | 비FA 다년계약. FA 계약과 섞지 않는다 | 수동 조사 |
| `hitter_season_stats_2010_2026_v4.csv`, `pitcher_season_stats_2010_2026_v4.csv` | 연도별 시즌 스탯 | `scripts/build_season_stats_v3.py` |
| `player_master.csv` | 선수 마스터 (검색·유형·소속) | `scripts/build_season_stats_v3.py` |
| `player_photos.csv`, `player_photos_manual.csv` | 선수 사진. 수기가 API 위에 덮인다 | `scripts/build_season_stats_v3.py` / 수동 |
| `player_birth_manual.csv` | 생년. 나이 피처의 근거 | `scripts/collect_player_birth.py`, `scripts/collect_birth_old_fa.py` |
| `star_features_v2.csv` | MVP·골든글러브·국가대표 연도별 수상 | `scripts/collect_star_features.py` |
| `hitter_training_v10.csv`, `pitcher_training_v10.csv` | 모델 학습 입력 (타자 136명 / 투수 72명) | `scripts/build_training_v8.py` |
| `teams.csv`, `position_need.csv` | 구단별 제시가 보정용 메타데이터 | 수동 작성 |
| `naver_hitter_2010_2026_raw_v2.csv`, `naver_pitcher_2010_2026_raw_v2.csv` | 네이버 스포츠 API 원천 수집 (2010~2026) | `scripts/crawl_naver_v2.py` |

## 조사값과 추정값을 파일로 나눠 둔 이유

`future_fa_candidates_v2.csv`는 실제로 확인한 42명이고, `fa_eligibility_estimated.csv`는
데뷔 연도와 뛴 시즌 수로 계산한 추정값이다. 한 파일에 섞으면 어느 쪽이 사실인지
구분할 수 없어진다. 화면도 추정값으로 만든 카드에는 "FA 예상(추정)"을 붙인다.

추정 규칙과 그 근거는 `scripts/estimate_fa_eligibility.py` 도크스트링에 있다.

## player_id를 파일에 박아 둔 이유

KBO에는 같은 이름이 많다. 마스터에 박건우가 4명, 김민수가 3명이다. 이름으로 붙이면
롯데 박건우가 "NC 외야수 · 2028년 FA 예정" 카드를 받는다. 실제로 그랬다 (TS-012).

계약과 FA 예정 목록에는 `player_id` 컬럼이 있고 코드는 그 값으로만 맞춘다.
한 건(투수 김상수 2021 SK)만 비어 있는데, 네이버 크롤 원본에 그 선수가 없다.

## 버전 접미사

같은 이름으로 덮어쓰지 않는다. 앞선 버전은 그때 발행한 리포트가 그대로 재현되도록
남긴다. 학습 데이터의 번호는 피처 구성이 아니라 입력 계약 수를 따른다.

| 파일 | 계약 | 범위 |
|---|---|---|
| `*_training_v8.csv` | 140건 | 2018~2026 |
| `*_training_v9.csv` | 175건 | 2016~2026 |
| `*_training_v10.csv` | 210건 | 2014~2026 |

피처 구성은 셋 다 같다.

## WAR가 없는 구간

네이버는 타자 WAR를 2017년부터, 투수 WAR를 2014년부터만 준다. 그 이전 시즌은
전 선수 0으로 내려오고 `build_season_stats_v3.py`가 그런 해를 결측으로 바꾼다.

FA 계약은 직전 3시즌으로 피처를 만들므로, 2014~2019년 타자 계약과 2014~2016년
투수 계약은 최상위 피처인 WAR 없이 학습에 들어간다. 트리 모델이 결측을 그대로
받으므로 학습은 되지만, 그 구간의 예측은 나머지 피처에 기댄다.

## data/archive/

후속 파일로 대체된 과거 버전을 보관한다. 재현성 참고용으로만 남기고 신규 코드에서
참조하지 않는다.

## 알려진 한계

- `hitter_training_v5 → v7` 변환 스크립트는 저장소에 없다. v8 이후 파이프라인
  (`build_season_stats_v3.py` → `build_training_v8.py`)으로 대체됐으므로 v5·v7은
  더 이상 쓰지 않는다.
- 마스터의 포지션은 1854명 중 423명만 네이버가 준 값이고 나머지는 추정이다.
