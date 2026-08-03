# ⚾ StoveLens AI (SalaryCast AI) — KBO FA 연봉 예측 서비스

KBO 프리에이전트(FA) 선수의 최근 성적 데이터를 기반으로 예상 연평균 계약금을 예측하고,
구단별 적정 제시가를 추천하는 머신러닝 프로젝트입니다.
2015~2026년 네이버 스포츠 KBO 시즌 통계를 수집하고, FA 계약 데이터와 결합하여 학습 데이터를 구성한 뒤,
XGBoost 기반 예측 모델과 Streamlit 시연 서비스로 이어집니다.

[![CI](https://github.com/jjssspark/SalaryCast_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/jjssspark/SalaryCast_AI/actions/workflows/ci.yml)

---

## 🎬 데모

![데모](docs/images/demo.gif)

선수 검색 → AI 예상 연봉 → 실제 계약 비교 → 구단별 제시가까지 이어지는 핵심 플로우입니다.
고화질 원본은 [docs/video/salarycast_demo.mp4](docs/video/salarycast_demo.mp4)에서 다운로드해 볼 수 있습니다.

---

## 📸 스크린샷

| 홈 화면 | FA 카드 호버 프로필 |
|---|---|
| ![홈 화면](docs/images/01-home.jpg) | ![호버 팝업](docs/images/02-fa-hover-popup.jpg) |

| 타자 상세 (실제 vs 예측 비교) | 구단별 예상 제시가 |
|---|---|
| ![타자 상세](docs/images/03-hitter-detail.jpg) | ![구단별 제시가](docs/images/04-team-offers.jpg) |

| 투수 상세 (고평가 판정) | 예측 근거 상세 |
|---|---|
| ![투수 상세](docs/images/05-pitcher-detail.jpg) | ![예측 근거](docs/images/06-key-factor-detail.jpg) |

> 🔗 배포 링크: **[stovelens-ai.streamlit.app](https://stovelens-ai.streamlit.app/)**

---

## 📁 프로젝트 구조

```
SalaryCast_AI/
├── app/
│   └── app.py                 # Streamlit 엔트리포인트 (얇은 라우팅만 담당)
├── src/                        # 앱 핵심 로직 모듈
│   ├── constants.py            # 팀 데이터·툴팁 등 상수
│   ├── data_loader.py          # 데이터/모델 로딩 (+ 실패 시 에러 핸들링)
│   ├── features.py             # 피처 엔지니어링
│   ├── predict.py              # 연봉 예측 (XGBoost/LightGBM 앙상블)
│   ├── team_offers.py          # 구단별 제시가 보정 로직
│   └── ui/                     # 화면 렌더링 (홈/검색/상세 페이지, 스타일)
├── tests/                       # pytest 단위 테스트
├── data/                        # 원본·정제·학습용 CSV
├── models/                      # 학습된 모델(.pkl)
├── notebooks/                    # 데이터 수집·전처리·모델 학습 노트북
└── scripts/                      # 크롤링·유틸 스크립트
```

---

## ⚙️ 주요 기능

### 1. 데이터 수집
- 네이버 스포츠 API에서 KBO 선수 시즌 통계 수집 (2015~2026)
- 타자 / 투수 구분 수집, 국내 선수 필터링
- FA 계약 뉴스 기사 파싱을 통한 계약 금액 추출 (140건)

### 2. 학습 데이터 생성
- FA 계약 연도 기준 **직전 3개년 시즌 통계** 집계
- 타자 / 투수 포지션별 학습 데이터셋 분리 생성
- 스타성 지표(MVP·골든글러브·국가대표 등), 투수 역할(SP/SU/CL) 피처 추가

### 3. 연봉 예측 모델
- 타자 / 투수 분리 학습 (선형회귀 → 랜덤포레스트 → XGBoost/LightGBM 앙상블)
- 교차검증·하이퍼파라미터 튜닝, SHAP 기반 피처 중요도 분석
- 평가지표: R², MSE, RMSE

### 4. Streamlit 시연 서비스
- 선수 검색 → AI 예상 연봉 + 예측 근거(핵심 요소) 표시
- 과거 FA 완료 선수: 실제 계약 vs 예측 비교
- 미래 FA 예정 선수: 구단별 적정 제시가 추천

---

## 📈 모델 성능

표본이 적어(타자 80명·투수 43명) 단일 train/test 분할 대신 5-fold 교차검증의 OOF(Out-of-Fold) 예측으로 평가했습니다. 연봉은 log 변환 후 학습해 큰 값 쪽으로 쏠리는 것을 완화했습니다.

**타자 — 최종 모델: XGBoost 0.05 + LightGBM 0.95 앙상블**

| 지표 | 값 |
|---|---|
| R² (5-fold CV, OOF, log-scale) | 0.507 |
| RMSE | 5.48억 원 |

**투수 — 최종 모델: XGBoost + LightGBM + RandomForest 스태킹 → Ridge**

| 지표 | 값 |
|---|---|
| R² (5-fold CV, OOF, log-scale) | 0.410 |
| RMSE | 3.66억 원 |

타자 모델은 `war_last_year`·`double_3yr_avg`·`prime_years_left`, 투수 모델은 `war_3yr_sum_all_pct`·`market_level`·`innings_3yr_avg`가 상위 피처로 나타났습니다. 상세 비교 실험(선형회귀/랜덤포레스트/XGBoost 단일모델 성능, fa_year 포함·제외 비교)은 `notebooks/clean_model_train.ipynb`에 남아 있습니다.

---

## 🛠️ 기술 스택

| 분류 | 기술 | 왜 골랐는가 |
|------|------|------|
| 언어 | Python 3.13 | 데이터 수집·모델링·서비스를 한 언어로 끊김 없이 연결 |
| 데이터 처리 | pandas, numpy | 시즌 스탯·FA 계약 데이터의 표 형태 가공에 표준 |
| 머신러닝 | scikit-learn, XGBoost, LightGBM, SHAP | 선형회귀 베이스라인 → 트리 앙상블로 단계적 고도화, SHAP으로 예측 근거 설명 |
| 웹 서비스 | Streamlit, Plotly | 파이썬 모델 코드와 분리된 API 서버 없이 바로 시연 서비스로 연결 ([ADR-06](docs/ADR.md) 참고) |
| 데이터 수집 | requests, BeautifulSoup | 네이버 스포츠 API 직접 호출 + FA 계약 기사 파싱에 충분히 가벼움 |
| 테스트 / CI | pytest, ruff, GitHub Actions | 커밋마다 회귀 확인, 별도 인프라 없이 GitHub에서 바로 동작 |

---

## 🚀 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/jjssspark/SalaryCast_AI.git
cd SalaryCast_AI

# 가상환경 생성 및 의존성 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Streamlit 앱 실행

```bash
streamlit run app/app.py
```

### 테스트 실행

```bash
pytest tests/ -v
```

### 데이터 파이프라인 재현 (노트북 실행 순서)

```
1. data_collect_test.ipynb       — 네이버 KBO 선수 통계 수집
2. fa_contract_collect.ipynb     — FA 계약 데이터 수집
3. make_training_dataset.ipynb   — 학습 데이터셋 생성
4. clean_model_train.ipynb       — 모델 학습·평가·SHAP 분석
```

---

## 📊 아키텍처 & 데이터 파이프라인

```mermaid
flowchart TD
    subgraph external["외부 의존"]
        naver[("네이버 스포츠 API<br/>KBO 시즌 통계 2015~2026")]
    end

    subgraph offline["오프라인 파이프라인 (notebooks/)"]
        collect["시즌 통계 수집<br/>타자 / 투수 분리"]
        fa["FA 계약 데이터 매핑<br/>140건"]
        agg["직전 3개년 통계 집계<br/>+ 스타성 · 투수역할 피처"]
        train["모델 학습<br/>선형회귀 → 랜덤포레스트 → XGBoost/LightGBM"]
    end

    subgraph models["산출물 (models/, data/)"]
        pkl[("hitter_model.pkl<br/>pitcher_model.pkl")]
        csv[("hitter_training_v5.csv<br/>pitcher_training_v5.csv")]
    end

    subgraph runtime["런타임 (app/, src/)"]
        app["app.py — 라우팅"]
        predict["predict.py — 연봉 예측"]
        offers["team_offers.py — 구단별 제시가 보정"]
        ui["src/ui/ — 화면 렌더링"]
    end

    user(["사용자 (브라우저)"])

    naver --> collect --> fa --> agg --> train
    train --> pkl
    agg --> csv
    pkl --> predict
    csv --> predict
    app --> ui
    ui --> predict --> offers --> ui
    user <--> app
```

- **개발 환경**: `notebooks/`에서 노트북 실행 순서(①~④)대로 재현하며 로컬 CSV·pkl 파일을 직접 다룬다.
- **배포 환경**: Streamlit 앱은 `models/`, `data/`에 이미 커밋된 학습 결과물(.pkl/.csv)만 읽는다 — 배포 서버에서 재학습하지 않는다. 네이버 API는 오프라인 수집 단계에서만 호출되고, 런타임에는 선수 사진 조회(`get_player_photo`) 시 다시 호출된다.

---

## 🧭 트러블슈팅 & 설계 결정

**트러블슈팅 3줄 요약** ([전체 기록 →](docs/TROUBLESHOOTING.md))
- Streamlit의 `st.markdown` 호출은 매번 독립된 DOM 엘리먼트로 렌더링돼, 여러 호출로 나눠 그린 카드가 빈 박스로 깨지는 게 근본 원인이었다.
- 선수 사진이 안 뜨는 문제는 "네트워크 차단"이라는 첫 가설이 틀렸고, 실제로는 API 파라미터명 변경·응답 키 변경·`pageSize` 부족·CDN Referer 차단 네 가지가 겹친 문제였다.
- 코드를 고쳐도 화면이 그대로였던 건 Streamlit `runOnSave` 미설정과, 그 알림 배너를 가리던 커스텀 CSS가 겹친 환경 문제였다.

**ADR 3줄 요약** ([전체 기록 →](docs/ADR.md))
- 투수 역할(SP/SU/CL)은 표본이 너무 적어 모델 3개로 쪼개는 대신 `pitcher_role` 단일 피처로 통합했다.
- 타자/투수 모델을 완전히 분리 학습하고, 총액·계약기간 등 데이터 누수 컬럼을 학습 피처에서 제외했다.
- 1차 연봉 예측(XGBoost/LightGBM)과 2차 구단별 제시가 보정(도메인 규칙)을 분리해서 설계했다.

**프로젝트 회고 및 진행 기록**: [Notion →](https://app.notion.com/p/SalaryCast_AI-00af6f1e619a82c4b882016d34088dd1?source=copy_link)
