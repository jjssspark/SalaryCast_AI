"""
fa_year 제거 실험
- 타자/투수 각각 fa_year 포함 vs 제거 후 R², RMSE 비교
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

TARGET = "annual_avg_salary"
LEAKAGE_COLS = ["player_name", "position", "team",
                "contract_years", "total_contract_amount"]


def prepare(df, exclude_fa_year=False):
    drop = LEAKAGE_COLS + [TARGET]
    if exclude_fa_year:
        drop.append("fa_year")

    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df[TARGET]

    # 전체가 NaN인 컬럼 제거
    X = X.dropna(axis=1, how="all")

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 남은 결측치 → 중앙값으로 대체
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def evaluate(X, y, label):
    results = {}
    for name, model in [
        ("Linear Regression", LinearRegression()),
        ("Random Forest",     RandomForestRegressor(n_estimators=100, random_state=42)),
    ]:
        r2_scores  = cross_val_score(model, X, y, cv=5, scoring="r2")
        mse_scores = cross_val_score(model, X, y, cv=5,
                                     scoring="neg_mean_squared_error")
        r2   = r2_scores.mean()
        rmse = np.sqrt(-mse_scores.mean())
        results[name] = {"R²": round(r2, 3), "RMSE": round(rmse, 2)}
        print(f"  [{label}] {name:20s}  R²={r2:.3f}  RMSE={rmse:.2f}")
    return results


def run_experiment(csv_path, player_type):
    print(f"\n{'='*58}")
    print(f"  {player_type} 실험")
    print(f"{'='*58}")
    df = pd.read_csv(csv_path)

    print("\n▶ fa_year 포함")
    X_with, y = prepare(df, exclude_fa_year=False)
    res_with  = evaluate(X_with, y, "포함")

    print("\n▶ fa_year 제거")
    X_without, y = prepare(df, exclude_fa_year=True)
    res_without  = evaluate(X_without, y, "제거")

    print(f"\n{'─'*58}")
    print(f"  비교 결과 ({player_type})")
    print(f"{'─'*58}")
    print(f"  {'모델':<22} {'조건':<8} {'R²':>6}  {'RMSE':>6}")
    print(f"  {'-'*46}")
    for model in res_with:
        w  = res_with[model]
        wo = res_without[model]
        diff_r2   = wo["R²"]   - w["R²"]
        diff_rmse = wo["RMSE"] - w["RMSE"]
        sign_r2   = "+" if diff_r2   >= 0 else ""
        sign_rmse = "+" if diff_rmse >= 0 else ""
        print(f"  {model:<22} {'포함':<8} {w['R²']:>6}  {w['RMSE']:>6}")
        print(f"  {'':<22} {'제거':<8} {wo['R²']:>6}  {wo['RMSE']:>6}"
              f"   (ΔR²={sign_r2}{diff_r2:.3f}, ΔRMSE={sign_rmse}{diff_rmse:.2f})")
        print()

    return res_with, res_without


if __name__ == "__main__":
    run_experiment("data/hitter_training_v4.csv", "타자")
    run_experiment("data/pitcher_training_v4.csv", "투수")

    print("\n" + "="*58)
    print("  판단 기준")
    print("="*58)
    print("  R² 높을수록 좋음 / RMSE 낮을수록 좋음")
    print("  ΔR² 양수 & ΔRMSE 음수 → fa_year 제거가 더 유리")
    print("  ΔR² 음수 & ΔRMSE 양수 → fa_year 포함이 더 유리")
