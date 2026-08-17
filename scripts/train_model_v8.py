"""
학습 데이터 v8 -> 모델 재학습 + 누수 제거 전후 비교

train_final_model.py는 v5를 읽고 피처 로직을 자체 보유해서 src/features.py와 갈라져 있었다.
이 스크립트는 src.features만 쓴다.

평가를 억 원 단위로 되돌려서 낸다. 로그 공간 R²는 실제 오차 감각과 다르다.
누수를 걷어내면 지표가 떨어질 수 있고, 떨어지면 떨어진 대로 적는다.

출력:
  models/{hitter,pitcher}_v8_*.pkl
  models/reference_dist_{hitter,pitcher}_v8.pkl
  output/reports/leakage_ablation.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (  # noqa: E402
    HITTER_PCT_COLS,
    PITCHER_PCT_COLS,
    add_percentile_ranks,
    build_reference_dist,
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("output/reports")

# 정답을 직접 담고 있거나 식별자에 불과해 모델 입력에서 빼야 하는 컬럼
EXCLUDED = {
    "annual_avg_salary", "total_contract_amount", "contract_years",
    "player_name", "player_id", "team", "position", "pitcher_role", "seasons_used",
}

SEED = 42


def make_models(seed: int = SEED) -> dict:
    """표본이 100건 미만이라 n_jobs=1이 오히려 빠르다. 병렬화 오버헤드가 계산량보다 크다."""
    return {
        "XGBoost": XGBRegressor(
            n_estimators=600, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
            random_state=seed, verbosity=0, n_jobs=1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=5,
            random_state=seed, verbose=-1, n_jobs=1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=6, min_samples_leaf=2,
            random_state=seed, n_jobs=1,
        ),
        # market_level은 가장 이른 FA 연도에 참고할 과거가 없어 NaN이다.
        # 트리 계열은 NaN을 그대로 받지만 Ridge는 못 받아 중앙값으로 채운다.
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        ),
    }


def oof_predict(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> np.ndarray:
    """반복 K-fold의 평균 OOF 예측. 표본이 작아 한 번의 분할로는 흔들린다."""
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    accumulated = np.zeros(len(y))

    for train_idx, valid_idx in cv.split(X):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        accumulated[valid_idx] += fold_model.predict(X.iloc[valid_idx])

    return accumulated / n_repeats


def score_in_billions(y_log: pd.Series, pred_log: np.ndarray) -> dict:
    """로그 타깃을 억 원으로 되돌려 평가한다. 사용자가 보는 단위가 억 원이다."""
    actual = np.expm1(y_log)
    predicted = np.clip(np.expm1(pred_log), 0, None)
    return {
        "r2": r2_score(actual, predicted),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
    }


def prepare(df: pd.DataFrame, pct_cols: list[str], group_col: str):
    engineered = add_percentile_ranks(df.copy(), pct_cols, group_col)
    y = np.log1p(engineered["annual_avg_salary"])
    feature_cols = [
        c for c in engineered.columns
        if c not in EXCLUDED and pd.api.types.is_numeric_dtype(engineered[c])
    ]
    X = engineered[feature_cols].astype(float)
    return X, y, feature_cols, engineered


def evaluate_all(X: pd.DataFrame, y: pd.Series) -> dict:
    results = {}
    for name, model in make_models().items():
        pred = oof_predict(model, X, y)
        results[name] = {"pred": pred, **score_in_billions(y, pred)}
        m = results[name]
        print(f"    {name:14s} R² {m['r2']:6.3f}   RMSE {m['rmse']:5.2f}억   MAE {m['mae']:5.2f}억")
    return results


def best_blend(results: dict, y: pd.Series):
    """두 모델 가중 평균 중 가장 좋은 조합. 단일 모델보다 나을 때만 채택한다."""
    names = list(results)
    best = (None, None, 1.0, {"r2": -np.inf})

    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for weight in np.arange(0.0, 1.01, 0.05):
                blended = weight * results[first]["pred"] + (1 - weight) * results[second]["pred"]
                metrics = score_in_billions(y, blended)
                if metrics["r2"] > best[3]["r2"]:
                    best = (first, second, float(weight), metrics)
    return best


def train_group(label: str, csv_name: str, pct_cols: list[str], group_col: str) -> dict:
    print("=" * 62)
    print(f"  {label} 모델 학습 (v8)")
    print("=" * 62)

    df = pd.read_csv(DATA_DIR / csv_name)
    print(f"데이터 {len(df)}명 / 원본 {df.shape[1]}컬럼")

    X, y, feature_cols, engineered = prepare(df, pct_cols, group_col)
    print(f"모델 입력 피처 {len(feature_cols)}개\n  단일 모델 OOF 성능:")
    results = evaluate_all(X, y)

    first, second, weight, blend_metrics = best_blend(results, y)
    single_best = max(results.items(), key=lambda kv: kv[1]["r2"])
    print(f"\n    최적 블렌드   {first} {weight:.2f} + {second} {1 - weight:.2f}"
          f"   R² {blend_metrics['r2']:.3f}  RMSE {blend_metrics['rmse']:.2f}억")

    use_blend = blend_metrics["r2"] > single_best[1]["r2"] + 1e-6
    chosen = {
        "method": f"blend:{first}+{second}" if use_blend else single_best[0],
        "weight": weight if use_blend else 1.0,
        **({k: v for k, v in blend_metrics.items()} if use_blend else {
            k: v for k, v in single_best[1].items() if k != "pred"
        }),
    }
    print(f"    선정: {chosen['method']}  R² {chosen['r2']:.3f}  RMSE {chosen['rmse']:.2f}억")

    prefix = MODEL_DIR / f"{label}_v8"
    members = sorted({first, second}) if use_blend else [single_best[0]]
    for name in members:
        model = make_models()[name]
        model.fit(X, y)
        joblib.dump(model, f"{prefix}_{name.lower()}.pkl")

    reference = build_reference_dist(engineered, pct_cols, group_col)
    joblib.dump(reference, MODEL_DIR / f"reference_dist_{label}_v8.pkl")

    meta = {
        "features": feature_cols,
        "method": chosen["method"],
        "blend_first": first if use_blend else single_best[0],
        "blend_second": second if use_blend else None,
        "weight": chosen["weight"],
        "members": members,
        "r2": chosen["r2"],
        "rmse_억": chosen["rmse"],
        "mae_억": chosen["mae"],
        "mse": chosen["mse"],
        "n_samples": len(df),
        "target": "log1p(annual_avg_salary)",
        "pct_cols": pct_cols,
        "group_col": group_col,
    }
    joblib.dump(meta, f"{prefix}_meta.pkl")
    joblib.dump(feature_cols, f"{prefix}_features.pkl")
    print(f"    저장: {prefix}_*.pkl\n")

    return {"label": label, "metrics": chosen, "X": X, "y": y, "df": engineered}


def leakage_ablation(entry: dict) -> dict:
    """market_level을 v7 방식(같은 해 타깃 중앙값)으로 되돌리면 얼마나 부풀려지는지."""
    X_clean, y, df = entry["X"], entry["y"], entry["df"]

    X_leaky = X_clean.copy()
    X_leaky["market_level"] = (
        df.groupby("fa_year")["annual_avg_salary"].transform("median").to_numpy()
    )

    model = make_models()["XGBoost"]
    return {
        "clean": score_in_billions(y, oof_predict(model, X_clean, y)),
        "leaky": score_in_billions(y, oof_predict(model, X_leaky, y)),
    }


def write_report(entries: list[dict], ablations: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 누수 제거 전후 비교 (v8)",
        "",
        "`market_level`은 v7까지 `groupby(fa_year)['annual_avg_salary'].median()`이었음.",
        "같은 해 계약의 중앙값이라 자기 자신이 포함된 값이고, 정답을 직접 흘림.",
        "v8은 직전 3개 연도 계약만 보고 계산함.",
        "",
        "지표는 전부 **억 원 단위로 되돌린 뒤** 계산한 OOF 값임 (5-fold x 5회 반복, XGBoost 고정).",
        "",
        "| 대상 | 방식 | R2 | RMSE(억) | MAE(억) |",
        "|---|---|---|---|---|",
    ]
    for label, result in ablations.items():
        for mode, korean in [("leaky", "v7 방식(누수)"), ("clean", "v8 방식(제거)")]:
            m = result[mode]
            lines.append(
                f"| {label} | {korean} | {m['r2']:.3f} | {m['rmse']:.2f} | {m['mae']:.2f} |"
            )

    lines += ["", "## 최종 선정 모델", "",
              "| 대상 | 방식 | R2 | RMSE(억) | MAE(억) | 표본 |", "|---|---|---|---|---|---|"]
    for entry in entries:
        m = entry["metrics"]
        lines.append(
            f"| {entry['label']} | {m['method']} | {m['r2']:.3f} | "
            f"{m['rmse']:.2f} | {m['mae']:.2f} | {len(entry['df'])}명 |"
        )

    (REPORT_DIR / "leakage_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"보고서: {REPORT_DIR / 'leakage_ablation.md'}")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    entries = [
        train_group("hitter", "hitter_training_v8.csv", HITTER_PCT_COLS, "position"),
        train_group("pitcher", "pitcher_training_v8.csv", PITCHER_PCT_COLS, "pitcher_role"),
    ]

    print("=" * 62)
    print("  누수 제거 전후 비교")
    print("=" * 62)
    ablations = {}
    for entry in entries:
        result = leakage_ablation(entry)
        ablations[entry["label"]] = result
        print(f"  {entry['label']}: 누수 R² {result['leaky']['r2']:.3f} "
              f"-> 제거 R² {result['clean']['r2']:.3f}")

    write_report(entries, ablations)


if __name__ == "__main__":
    main()
