# ⚾ SalaryCast AI — KBO FA 연봉 예측 모델

KBO 프리에이전트(FA) 선수의 계약 연봉을 머신러닝으로 예측하는 프로젝트입니다.  
2018~2026년 네이버 스포츠 KBO 시즌 통계를 수집하고, FA 계약 데이터와 결합하여 학습 데이터를 구성합니다.

---

## 📁 프로젝트 구조

```
SalaryCast_AI/
├── data/
│   ├── fa_contract_urls.csv                 # FA 계약 기사 URL 목록
│   ├── fa_contracts.csv                     # FA 계약 정보 (선수명, 연도, 금액 등)
│   ├── hitter_season_stats_2018_2026.csv    # 타자 시즌 통계
│   ├── pitcher_season_stats_2018_2026.csv   # 투수 시즌 통계
│   ├── hitter_training.csv                  # 타자 학습 데이터셋
│   ├── pitcher_training.csv                 # 투수 학습 데이터셋
│   ├── naver_hitter_2018_2026_raw_all.csv   # 네이버 원본 타자 데이터
│   ├── naver_pitcher_2018_2026_raw_all.csv  # 네이버 원본 투수 데이터
│   └── teams.csv                            # 팀 정보
└── notebooks/
    ├── data_collect_test.ipynb              # 네이버 KBO API 크롤링 탐색
    ├── fa_contract_collect.ipynb            # FA 계약 데이터 수집
    └── make_training_dataset.ipynb          # 학습 데이터셋 생성
```

---

## ⚙️ 주요 기능

### 1. 데이터 수집
- 네이버 스포츠 API에서 KBO 선수 시즌 통계 수집 (2018~2026)
- 타자 / 투수 구분 수집, 국내 선수 필터링
- FA 계약 뉴스 기사 파싱을 통한 계약 금액 추출

### 2. 학습 데이터 생성
- FA 계약 연도 기준 **직전 3개년 시즌 통계** 집계
- 타자 / 투수 포지션별 학습 데이터셋 분리 생성
- 이닝 수 등 비표준 포맷 전처리 포함

### 3. 연봉 예측 (예정)
- scikit-learn 기반 회귀 모델로 FA 계약 연봉 예측
- 타자 / 투수 별도 모델 학습

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3 |
| 데이터 처리 | pandas, numpy |
| 머신러닝 | scikit-learn |
| 시각화 | matplotlib |
| 데이터 수집 | requests, BeautifulSoup |

---

## 🚀 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/jjssspark/SalaryCast_AI.git
cd SalaryCast_AI

# 의존성 설치
pip install -r requirements.txt
```

노트북 실행 순서:

```
1. data_collect_test.ipynb       — 네이버 KBO 선수 통계 수집
2. fa_contract_collect.ipynb     — FA 계약 데이터 수집
3. make_training_dataset.ipynb   — 학습 데이터셋 생성
```

---

## 📊 데이터 파이프라인

```
네이버 KBO API (2018~2026)
        ↓
시즌 통계 수집 (타자 / 투수)
        ↓
FA 계약 데이터 매핑
        ↓
직전 3개년 통계 집계 및 전처리
        ↓
타자 / 투수 학습 데이터셋 생성
        ↓
연봉 예측 모델 (예정)
```
