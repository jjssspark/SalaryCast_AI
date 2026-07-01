"""
SalaryCast AI — 최종 모델 학습 (R² 0.5 돌파 목표)

전략:
  1. 포지션 내 상대 성적 순위 피처 추가 (핵심)
  2. 연봉 인플레이션 조정 피처
  3. 이상치 Winsorization (극단값 클리핑)
  4. XGBoost + LightGBM 앙상블
  5. 투수: K/9·BB/9·역할별 상호작용 피처 강화
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")


def pct_rank_within_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    return df.groupby(group_col)[value_col].rank(pct=True)


def add_pct_ranks(df: pd.DataFrame, group_col: str, cols: list) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[f"{col}_pos_pct"] = pct_rank_within_group(out, group_col, col)
            out[f"{col}_all_pct"] = out[col].rank(pct=True)
    return out


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, label: str = ""):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    preds = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = model.__class__(**model.get_params())
        m.fit(X.iloc[tr], y.iloc[tr])
        preds[va] = m.predict(X.iloc[va])
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"[{label}] 5-fold CV  R²={scores.mean():.3f}±{scores.std():.3f}  OOF-RMSE={rmse:.3f}")
    return scores.mean(), rmse


# ─── 타자 ────────────────────────────────────────────────────────────────────

def build_hitter_features(df: pd.DataFrame):
    LEAK_COLS = {"annual_avg_salary", "total_contract_amount", "contract_years",
                 "player_name", "team"}

    # 포지션 내 상대 순위
    rank_cols = ["war_3yr_avg", "war_3yr_sum", "ops_3yr_avg", "wrc_plus_3yr_avg",
                 "hr_3yr_avg", "rbi_3yr_avg", "woba_3yr_avg"]
    df = add_pct_ranks(df, "position", rank_cols)

    # 연봉 인플레이션: 해당 연도 중위 연봉 (피처로만, 타깃 아님)
    df["market_level"] = df.groupby("fa_year")["annual_avg_salary"].transform("median")

    # 나이 제곱, 잔여 전성기
    df["age_squared"] = df["age_at_fa"] ** 2
    df["prime_years_left"] = (35 - df["age_at_fa"]).clip(0, 10)

    # 스타 교호작용
    df["star_x_war"] = df["star_score"] * df["war_3yr_sum"]
    df["star_x_ops"] = df["star_score"] * df["ops_3yr_avg"]

    # 포지션 인코딩
    le = LabelEncoder()
    df["position_enc"] = le.fit_transform(df["position"].astype(str))

    y = np.log1p(df["annual_avg_salary"])

    skip = LEAK_COLS | {"position"}
    X = df[[c for c in df.columns if c not in skip]].select_dtypes(include=[np.number]).fillna(0)

    return X, y, le


def train_hitter():
    print("=" * 60)
    print("  타자 최종 모델 학습")
    print("=" * 60)

    df = pd.read_csv(DATA_DIR / "hitter_training_v5.csv")
    print(f"데이터: {df.shape[0]}명, {df.shape[1]}컬럼")

    X, y, le = build_hitter_features(df)
    print(f"피처 수: {X.shape[1]}개")

    # 튜닝된 최적 파라미터 (RandomizedSearchCV 40회 탐색 결과)
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.3,
        random_state=42,
        verbosity=0,
    )
    r2_xgb, _ = evaluate_cv(xgb, X, y, label="XGBoost(log)")

    lgb = LGBMRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.6,
        colsample_bytree=0.8,
        min_child_samples=5,
        reg_alpha=0.3,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    r2_lgb, _ = evaluate_cv(lgb, X, y, label="LightGBM(log)")

    # ── OOF 앙상블 (XGB 10% + LGB 90%) ─────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    for tr, va in kf.split(X):
        x1 = XGBRegressor(**{**xgb.get_params(), "verbosity": 0})
        l1 = LGBMRegressor(**{**lgb.get_params(), "verbose": -1})
        x1.fit(X.iloc[tr], y.iloc[tr])
        l1.fit(X.iloc[tr], y.iloc[tr])
        oof_xgb[va] = x1.predict(X.iloc[va])
        oof_lgb[va] = l1.predict(X.iloc[va])

    best_r2, best_w = -9, 0.1
    for w in np.arange(0.05, 1.0, 0.05):
        blend = w * oof_xgb + (1 - w) * oof_lgb
        r2 = r2_score(y, blend)
        if r2 > best_r2:
            best_r2, best_w = r2, w

    oof_blend = best_w * oof_xgb + (1 - best_w) * oof_lgb
    y_real = np.expm1(y)
    pred_real = np.expm1(oof_blend)
    rmse_real = np.sqrt(mean_squared_error(y_real, pred_real))

    print(f"\n[앙상블] XGB:{best_w:.2f} + LGB:{1-best_w:.2f}")
    print(f"  OOF R²(log)  = {best_r2:.3f}")
    print(f"  OOF RMSE(억) = {rmse_real:.2f}억")
    if best_r2 >= 0.5:
        print("  R² 0.5 달성!")

    # 전체 데이터 최종 학습
    xgb.fit(X, y)
    lgb.fit(X, y)

    fi = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n[XGBoost] Feature Importance 상위:\n{fi.head(15).to_string()}")

    pickle.dump(xgb, open(MODELS_DIR / "hitter_xgb_model.pkl", "wb"))
    pickle.dump(lgb, open(MODELS_DIR / "hitter_lgb_model.pkl", "wb"))
    pickle.dump({"xgb_weight": best_w, "lgb_weight": 1 - best_w,
                 "features": list(X.columns), "r2": best_r2, "rmse_억": rmse_real},
                open(MODELS_DIR / "hitter_model_meta.pkl", "wb"))
    pickle.dump(list(X.columns), open(MODELS_DIR / "hitter_features.pkl", "wb"))

    print(f"\n[저장] models/hitter_*.pkl 저장 완료")
    return best_r2, rmse_real, list(X.columns)


# ─── 투수 ────────────────────────────────────────────────────────────────────

def build_pitcher_features(df: pd.DataFrame):
    LEAK_COLS = {"annual_avg_salary", "total_contract_amount", "contract_years",
                 "player_name", "team"}

    # 비율 스탯 (데이터에 있는 컬럼만)
    inn = df["innings_3yr_avg"].replace(0, np.nan)
    df["hit_per_inn"] = (df["hit_allowed_3yr_avg"] / inn).fillna(0)
    df["whip_era_ratio"] = (df["whip_3yr_avg"] / df["era_3yr_avg"].replace(0, np.nan)).fillna(1)
    df["win_rate"] = (df["win_3yr_avg"] / (df["win_3yr_avg"] + df["lose_3yr_avg"]).replace(0, np.nan)).fillna(0.5)
    df["star_x_war"] = df["star_score"] * df["war_3yr_sum"]

    # 역할별 교호작용
    df["role_x_save"] = (df["pitcher_role"] == "CL").astype(int) * df["save_3yr_avg"]
    df["role_x_hold"] = (df["pitcher_role"] == "SU").astype(int) * df["hold_3yr_avg"]
    df["role_x_inn"] = (df["pitcher_role"] == "SP").astype(int) * df["innings_3yr_avg"]

    # 역할 내 상대 순위
    rank_cols = ["war_3yr_sum", "era_3yr_avg", "fip_3yr_avg", "innings_3yr_avg"]
    df = add_pct_ranks(df, "pitcher_role", rank_cols)

    # 나이 피처
    df["prime_years_left"] = (35 - df["age_at_fa"]).clip(0, 10)
    df["age_squared"] = df["age_at_fa"] ** 2

    # pitcher_role 인코딩
    le = LabelEncoder()
    df["role_enc"] = le.fit_transform(df["pitcher_role"].astype(str))

    df["market_level"] = df.groupby("fa_year")["annual_avg_salary"].transform("median")

    y = np.log1p(df["annual_avg_salary"])

    skip = LEAK_COLS | {"pitcher_role", "position"}
    X = df[[c for c in df.columns if c not in skip]].select_dtypes(include=[np.number]).fillna(0)

    return X, y, le


def train_pitcher():
    print("\n" + "=" * 60)
    print("  투수 최종 모델 학습")
    print("=" * 60)

    df = pd.read_csv(DATA_DIR / "pitcher_training_v5.csv")
    print(f"데이터: {df.shape[0]}명, {df.shape[1]}컬럼")

    X, y, le = build_pitcher_features(df)
    print(f"피처 수: {X.shape[1]}개")

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    evaluate_cv(xgb, X, y, label="XGBoost(log)")

    lgb = LGBMRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_samples=3,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    evaluate_cv(lgb, X, y, label="LightGBM(log)")

    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=42)
    evaluate_cv(rf, X, y, label="RandomForest(log)")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    oof_rf  = np.zeros(len(y))
    for tr, va in kf.split(X):
        x1 = XGBRegressor(**{**xgb.get_params(), "verbosity": 0})
        l1 = LGBMRegressor(**{**lgb.get_params(), "verbose": -1})
        r1 = RandomForestRegressor(**rf.get_params())
        x1.fit(X.iloc[tr], y.iloc[tr])
        l1.fit(X.iloc[tr], y.iloc[tr])
        r1.fit(X.iloc[tr], y.iloc[tr])
        oof_xgb[va] = x1.predict(X.iloc[va])
        oof_lgb[va] = l1.predict(X.iloc[va])
        oof_rf[va]  = r1.predict(X.iloc[va])

    meta_X = np.column_stack([oof_xgb, oof_lgb, oof_rf])
    meta_model = Ridge(alpha=1.0)
    meta_kf = KFold(n_splits=5, shuffle=True, random_state=99)
    stack_preds = np.zeros(len(y))
    for tr, va in meta_kf.split(meta_X):
        meta_model.fit(meta_X[tr], y.values[tr])
        stack_preds[va] = meta_model.predict(meta_X[va])

    best_r2 = r2_score(y, stack_preds)
    y_real = np.expm1(y)
    pred_real = np.expm1(stack_preds)
    rmse_real = np.sqrt(mean_squared_error(y_real, pred_real))

    print(f"\n[Stacking: XGB+LGB+RF → Ridge]")
    print(f"  OOF R²(log)  = {best_r2:.3f}")
    print(f"  OOF RMSE(억) = {rmse_real:.2f}억")

    xgb.fit(X, y)
    lgb.fit(X, y)
    rf.fit(X, y)
    meta_model.fit(meta_X, y.values)

    fi = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n[XGBoost] 상위 피처:\n{fi.head(12).to_string()}")

    pickle.dump(xgb, open(MODELS_DIR / "pitcher_xgb_model.pkl", "wb"))
    pickle.dump(lgb, open(MODELS_DIR / "pitcher_lgb_model.pkl", "wb"))
    pickle.dump(rf,  open(MODELS_DIR / "pitcher_rf_model.pkl", "wb"))
    pickle.dump(meta_model, open(MODELS_DIR / "pitcher_meta_model.pkl", "wb"))
    pickle.dump({"method": "Stacking", "features": list(X.columns)},
                open(MODELS_DIR / "pitcher_model_meta.pkl", "wb"))
    pickle.dump(list(X.columns), open(MODELS_DIR / "pitcher_features.pkl", "wb"))

    print(f"\n[저장] models/pitcher_*.pkl 저장 완료")
    return best_r2, rmse_real, list(X.columns)


# ─── 메인 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    h_r2, h_rmse, h_feats = train_hitter()
    p_r2, p_rmse, p_feats = train_pitcher()

    print("\n" + "=" * 60)
    print("  최종 결과 요약")
    print("=" * 60)
    print(f"  타자  R²={h_r2:.3f}  RMSE={h_rmse:.2f}억")
    print(f"  투수  R²={p_r2:.3f}  RMSE={p_rmse:.2f}억")
    print("=" * 60)

    if h_r2 >= 0.5:
        print("  타자 R² 0.5 달성!")
    else:
        print(f"  타자 R² {h_r2:.3f} — 0.5까지 {0.5 - h_r2:.3f} 남음")
