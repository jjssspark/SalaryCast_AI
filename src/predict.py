"""StoveLens AI — 앙상블 연봉 예측.

호출 위치: src/ui/pages.py render_search()가 predict_h/predict_p 호출.
데이터 파일: 없음 — models/*.pkl에서 로드된 모델 객체(src.data_loader.load_models 반환값)를 인자로 받음.
사용자 지시: Notion Day3 체크리스트 "src/ 디렉터리 설계: predict.py".
"""
import numpy as np


def predict_h(row, xgb_m, lgb_m, meta):
    X = row[meta["features"]].values.reshape(1, -1)
    lp = meta["xgb_weight"] * xgb_m.predict(X)[0] + meta["lgb_weight"] * lgb_m.predict(X)[0]
    return float(np.expm1(lp))


def predict_p(row, xgb_m, rf_m, meta):
    X = row[meta["features"]].values.reshape(1, -1)
    lp = meta["xgb_weight"] * xgb_m.predict(X)[0] + meta["rf_weight"] * rf_m.predict(X)[0]
    return float(np.expm1(lp))
