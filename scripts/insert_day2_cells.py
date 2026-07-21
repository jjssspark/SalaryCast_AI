"""notebooks/clean_model_train.ipynb 에 Day2 실험 셀 삽입 (결론 셀 앞)."""

import json
from pathlib import Path

NB_PATH = Path("notebooks/clean_model_train.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

new_cells = [
    md(
        "## 7. Day2 — 모델 성능 고도화 (교차검증 · 하이퍼파라미터 튜닝 · 앙상블 · SHAP)\n\n"
        "**목표**: 타자 R² 0.38 수준을 교차검증 기반 튜닝과 피처 개선으로 끌어올린다.\n\n"
        "**적용 기법**\n"
        "- 5-fold 교차검증으로 baseline 재측정 (단일 test split의 우연성 제거)\n"
        "- XGBoost / LightGBM 하이퍼파라미터 튜닝 (RandomizedSearchCV 40회 탐색 결과 반영)\n"
        "- 타자: 포지션 내 상대 순위(percentile rank), 스타성 교호작용, 연봉 인플레이션(market_level) 피처 추가\n"
        "- 투수: 역할별(SP/SU/CL) 교호작용 · 상대 순위, 비율 스탯(K/9, whip/era 등) 추가\n"
        "- 타자: XGB+LGB 가중 평균 앙상블 (가중치 그리드서치)\n"
        "- 투수: XGB+LGB+RF → Ridge 메타모델 스태킹\n"
        "- 타깃은 log1p 변환 후 학습, 예측 시 expm1로 역변환\n"
    ),
    code(
        "import sys\n"
        "sys.path.append(str(Path('..') / 'scripts'))\n"
        "from train_final_model import train_hitter, train_pitcher\n"
        "\n"
        "h_r2, h_rmse, h_feats = train_hitter()\n"
        "p_r2, p_rmse, p_feats = train_pitcher()\n"
    ),
    md(
        "### 7-1. Before / After 비교표\n\n"
        "> **참고**: v4 baseline은 단일 train/test split(타자 test 16명, 투수 test 9명) 기준이라 "
        "우연성이 크다 (특히 투수는 test 9명뿐). v5는 5-fold 교차검증(OOF)으로 전체 표본을 검증에 사용해 "
        "더 정직한 성능을 측정한다. 투수 R²이 낮아진 것은 실제 성능 저하가 아니라 **측정 방식이 보수적으로 "
        "바뀐 것**이며, fold별 표준편차(±0.2~0.25)가 커 표본 부족(43명)에 따른 불확실성이 크다는 점도 "
        "함께 확인됐다.\n"
    ),
    code(
        "comparison = pd.DataFrame({\n"
        "    '구분': ['타자', '타자', '투수', '투수'],\n"
        "    '버전': [\n"
        "        'v4 baseline (단일 split, 16명)',\n"
        "        'v5 튜닝+앙상블 (5-fold CV, 80명 전체)',\n"
        "        'v4 baseline (단일 split, 9명)',\n"
        "        'v5 튜닝+앙상블 (5-fold CV, 43명 전체)',\n"
        "    ],\n"
        "    '모델': ['RandomForest', 'XGB+LGB 가중앙상블(log)', 'LinearRegression', 'XGB+LGB+RF Stacking(log)'],\n"
        "    'R²': [0.388, round(h_r2, 3), 0.756, round(p_r2, 3)],\n"
        "    'RMSE(억)': [5.99, round(h_rmse, 2), 3.30, round(p_rmse, 2)],\n"
        "})\n"
        "display(comparison)\n"
    ),
    md(
        "### 7-2. SHAP Summary Plot\n\n"
        "`scripts/generate_shap_plots.py` 실행 결과 (`output/day2/`에 저장됨)\n\n"
        "**타자**\n\n"
        "![타자 SHAP summary](../output/day2/hitter_shap_summary.png)\n\n"
        "![타자 Feature Importance](../output/day2/hitter_feature_importance.png)\n\n"
        "**투수**\n\n"
        "![투수 SHAP summary](../output/day2/pitcher_shap_summary.png)\n\n"
        "![투수 Feature Importance](../output/day2/pitcher_feature_importance.png)\n"
    ),
    md(
        "### 7-3. Day2 결론\n\n"
        "- 타자: R² 0.388 → **0.507** (5-fold CV 기준, 목표 0.5 달성). "
        "포지션 내 상대 순위 피처와 XGB+LGB 앙상블이 가장 크게 기여.\n"
        "- 투수: 단일 split 기준 0.756이었던 수치가 5-fold CV 기준 **0.410**으로 측정됨. "
        "실제 성능 저하라기보다 평가 방식이 정직해진 결과이며, 표본(43명) 대비 피처 수(39개)가 많아 "
        "과적합 위험이 있다는 것이 이번 Day2 작업에서 새로 확인된 한계점.\n"
        "- 다음 개선 여지: 투수 피처 수를 줄이거나(feature selection) 정규화를 강화해 표본 대비 "
        "피처 비율을 낮추는 실험이 필요.\n"
    ),
]

# 결론(마지막) 셀 바로 앞에 삽입
insert_at = len(nb["cells"]) - 1
nb["cells"][insert_at:insert_at] = new_cells

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"셀 {len(new_cells)}개 삽입 완료 (index {insert_at} 앞)")
