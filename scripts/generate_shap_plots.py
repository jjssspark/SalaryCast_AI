"""
Day2 — SHAP summary plot 및 Feature Importance 시각화 생성
scripts/train_final_model.py 로 학습된 models/*.pkl 을 로드해 output/day2/ 에 저장
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
OUT_DIR = Path("output/day2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

from train_final_model import build_hitter_features, build_pitcher_features  # noqa: E402


def hitter_shap():
    df = pd.read_csv(DATA_DIR / "hitter_training_v5.csv")
    X, y, _ = build_hitter_features(df)
    xgb = pickle.load(open(MODELS_DIR / "hitter_xgb_model.pkl", "rb"))

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title("타자 모델 — SHAP Summary (XGBoost)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hitter_shap_summary.png", dpi=150)
    plt.close()
    print("저장: output/day2/hitter_shap_summary.png")

    fi = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
    plt.figure(figsize=(8, 6))
    fi[::-1].plot(kind="barh", color="steelblue")
    plt.title("타자 모델 — Feature Importance Top 15 (XGBoost)", fontsize=13)
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hitter_feature_importance.png", dpi=150)
    plt.close()
    print("저장: output/day2/hitter_feature_importance.png")


def pitcher_shap():
    df = pd.read_csv(DATA_DIR / "pitcher_training_v5.csv")
    X, y, _ = build_pitcher_features(df)
    xgb = pickle.load(open(MODELS_DIR / "pitcher_xgb_model.pkl", "rb"))

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title("투수 모델 — SHAP Summary (XGBoost)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pitcher_shap_summary.png", dpi=150)
    plt.close()
    print("저장: output/day2/pitcher_shap_summary.png")

    fi = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
    plt.figure(figsize=(8, 6))
    fi[::-1].plot(kind="barh", color="salmon")
    plt.title("투수 모델 — Feature Importance Top 15 (XGBoost)", fontsize=13)
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pitcher_feature_importance.png", dpi=150)
    plt.close()
    print("저장: output/day2/pitcher_feature_importance.png")


if __name__ == "__main__":
    hitter_shap()
    pitcher_shap()
