"""v8 예측 정확도 검증. 읽기 전용 — 모델도 데이터도 고치지 않는다.

meta에 적힌 R²는 OOF로 잰 값이라 학습 성능은 아니다. 다만 블렌드 조합과 가중치를
그 OOF 점수로 골랐기 때문에, 고른 값 자체는 그 점수 쪽으로 휜다. 여기서는
고르는 과정까지 폴드 안에 넣어(중첩 CV) 그 휨이 얼마인지 잰다.

같이 재는 것:
  - 아무것도 안 배운 베이스라인 대비 얼마나 나은가
  - 앱이 과거 FA 선수에게 보여주는 값이 인샘플인가 (모드 A 화면의 착시)
  - 연봉대별로 어느 쪽으로 치우치는가
  - 학습 백분위와 서빙 백분위가 같은 값을 내는가

실행: .venv/bin/python scripts/verify_accuracy_v8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.model_selection import RepeatedKFold  # noqa: E402

import train_model_v8 as T  # noqa: E402
from src.features import (  # noqa: E402
    HITTER_PCT_COLS,
    PITCHER_PCT_COLS,
    apply_reference_ranks,
    build_reference_dist,
)

GROUPS = [
    ("hitter", "타자", "hitter_training_v8.csv", HITTER_PCT_COLS, "position"),
    ("pitcher", "투수", "pitcher_training_v8.csv", PITCHER_PCT_COLS, "pitcher_role"),
]

# 중첩 CV 비용을 감당할 만큼만. 바깥 10폴드 × 안쪽 4모델 × 15핏.
OUTER_SPLITS, OUTER_REPEATS, INNER_REPEATS = 5, 2, 3

SALARY_BANDS = [(0, 5), (5, 10), (10, 20), (20, 999)]


def load_group(csv_name: str, pct_cols: list[str], group_col: str):
    df = pd.read_csv(T.DATA_DIR / csv_name, encoding="utf-8-sig")
    X, y, feature_cols, engineered = T.prepare(df, pct_cols, group_col)
    return df, X, y, feature_cols, engineered


def fit_choice(X, y, choice):
    """선택된 구성을 주어진 학습 데이터에 맞춰 예측 함수로 만든다."""
    if choice["blend"]:
        first = T.make_models()[choice["first"]].fit(X, y)
        second = T.make_models()[choice["second"]].fit(X, y)
        w = choice["weight"]
        return lambda Z: w * first.predict(Z) + (1 - w) * second.predict(Z)

    model = T.make_models()[choice["first"]].fit(X, y)
    return lambda Z: model.predict(Z)


def select_config(X, y, inner_repeats: int) -> dict:
    """학습 스크립트와 같은 방식으로 모델·블렌드를 고른다. 주어진 데이터 안에서만."""
    inner = {}
    for name, model in T.make_models().items():
        pred = T.oof_predict(model, X, y, n_repeats=inner_repeats)
        inner[name] = {"pred": pred, **T.score_in_billions(y, pred)}

    first, second, weight, blend_metrics = T.best_blend(inner, y)
    single = max(inner.items(), key=lambda kv: kv[1]["r2"])
    use_blend = blend_metrics["r2"] > single[1]["r2"] + 1e-6

    if use_blend:
        return {"blend": True, "first": first, "second": second, "weight": weight}
    return {"blend": False, "first": single[0], "second": None, "weight": 1.0}


def nested_cv(X, y) -> tuple[np.ndarray, list[str]]:
    """고르는 과정까지 폴드 안에서. 바깥 폴드는 선택에 전혀 관여하지 않는다."""
    cv = RepeatedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=7)
    total = np.zeros(len(y))
    seen = np.zeros(len(y))
    picked = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        X_tr = X.iloc[train_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)

        choice = select_config(X_tr, y_tr, INNER_REPEATS)
        label = (
            f"{choice['first']}+{choice['second']} {choice['weight']:.2f}"
            if choice["blend"] else choice["first"]
        )
        picked.append(label)
        print(f"    폴드 {fold:2d}/{OUTER_SPLITS * OUTER_REPEATS}  선택: {label}")

        predict = fit_choice(X_tr, y_tr, choice)
        total[test_idx] += predict(X.iloc[test_idx])
        seen[test_idx] += 1

    return total / seen, picked


def baseline_scores(X, y) -> dict:
    """아무것도 안 배운 예측. 여기를 못 넘으면 모델이 한 일이 없다."""
    cv = RepeatedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=7)
    const = np.zeros(len(y))
    war_only = np.zeros(len(y))
    seen = np.zeros(len(y))

    war_col = "war_3yr_sum" if "war_3yr_sum" in X.columns else X.columns[0]

    for train_idx, test_idx in cv.split(X):
        y_tr = y.iloc[train_idx]
        const[test_idx] += float(y_tr.median())

        # WAR 3년 합 하나만 쓰는 단순 회귀
        coef = np.polyfit(X.iloc[train_idx][war_col].fillna(0), y_tr, 1)
        war_only[test_idx] += np.polyval(coef, X.iloc[test_idx][war_col].fillna(0))
        seen[test_idx] += 1

    return {
        "중앙값 고정": T.score_in_billions(y, const / seen),
        f"{war_col} 단일회귀": T.score_in_billions(y, war_only / seen),
    }


def deployed_in_sample(label: str, X: pd.DataFrame) -> np.ndarray:
    """배포된 모델이 학습 데이터 자신에게 내는 예측 (= 앱이 과거 FA 선수에게 보여주는 값)."""
    meta = joblib.load(T.MODEL_DIR / f"{label}_v8_meta.pkl")
    models = {
        name: joblib.load(T.MODEL_DIR / f"{label}_v8_{name.lower()}.pkl")
        for name in meta["members"]
    }
    Z = X[meta["features"]].astype(float)

    if meta["method"].startswith("blend:"):
        w = float(meta["weight"])
        return w * models[meta["blend_first"]].predict(Z) + (1 - w) * models[
            meta["blend_second"]
        ].predict(Z)
    return models[meta["method"]].predict(Z)


def serving_percentile_gap(engineered, pct_cols, group_col) -> pd.DataFrame:
    """학습 백분위(rank)와 서빙 백분위(참조분포 대비 비율)가 얼마나 벌어지는가."""
    reference = build_reference_dist(engineered, pct_cols, group_col)
    rows = []

    for _, row in engineered.iterrows():
        served = apply_reference_ranks(
            row.to_dict(), reference, pct_cols, str(row[group_col])
        )
        for col in pct_cols:
            for suffix in ("_all_pct", "_pos_pct"):
                key = f"{col}{suffix}"
                rows.append({
                    "컬럼": key,
                    "학습": float(row[key]),
                    "서빙": float(served[key]),
                    "차이": abs(float(row[key]) - float(served[key])),
                })

    return pd.DataFrame(rows)


def error_table(names, actual, predicted) -> pd.DataFrame:
    return pd.DataFrame({
        "선수": names,
        "실제": actual,
        "예측": predicted,
        "오차": predicted - actual,
        "절대오차": np.abs(predicted - actual),
    })


def band_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for low, high in SALARY_BANDS:
        part = table[(table["실제"] >= low) & (table["실제"] < high)]
        if part.empty:
            continue
        rows.append({
            "연봉대": f"{low}~{high if high < 999 else ''}억",
            "인원": len(part),
            "평균 실제": part["실제"].mean(),
            "평균 예측": part["예측"].mean(),
            "평균 오차": part["오차"].mean(),
            "MAE": part["절대오차"].mean(),
        })
    return pd.DataFrame(rows)


def fmt(metrics: dict) -> str:
    return (f"R² {metrics['r2']:6.3f}   RMSE {metrics['rmse']:5.2f}억   "
            f"MAE {metrics['mae']:5.2f}억")


def to_md(table: pd.DataFrame) -> str:
    """tabulate 없이 마크다운 표. 의존성 하나 늘리려고 이걸 못 쓸 이유가 없다."""
    def cell(v):
        return f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else str(v)

    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body])


def hit_rates(table: pd.DataFrame) -> dict:
    """실제 대비 몇 % 안에 들어오는가. R²보다 '맞았다'에 가까운 감각."""
    ratio = table["절대오차"] / table["실제"].clip(lower=0.1)
    spearman = table["실제"].corr(table["예측"], method="spearman")
    return {
        "±20% 이내": float((ratio <= 0.2).mean()),
        "±30% 이내": float((ratio <= 0.3).mean()),
        "±50% 이내": float((ratio <= 0.5).mean()),
        "순위상관": float(spearman),
    }


def main() -> None:
    report: list[str] = ["# v8 예측 정확도 검증", ""]

    for label, korean, csv_name, pct_cols, group_col in GROUPS:
        print("=" * 70)
        print(f"  {korean} ({label})")
        print("=" * 70)

        df, X, y, feature_cols, engineered = load_group(csv_name, pct_cols, group_col)
        actual = np.expm1(y).to_numpy()
        meta = joblib.load(T.MODEL_DIR / f"{label}_v8_meta.pkl")

        print(f"표본 {len(df)}명 / 피처 {len(feature_cols)}개"
              f"  (피처가 표본의 {len(feature_cols) / len(df):.0%})")

        print("\n[1] 배포 모델이 자기 학습 데이터에 내는 값 (인샘플)")
        in_sample = deployed_in_sample(label, engineered)
        in_metrics = T.score_in_billions(y, in_sample)
        print(f"    {fmt(in_metrics)}")

        print("\n[2] meta에 적힌 OOF 성능")
        print(f"    meta 기록:  R² {meta['r2']:6.3f}   RMSE {meta['rmse_억']:5.2f}억"
              f"   MAE {meta['mae_억']:5.2f}억")

        print("\n[3] 중첩 CV — 모델 선택까지 폴드 안에서")
        nested_pred, picked = nested_cv(X, y)
        nested_metrics = T.score_in_billions(y, nested_pred)
        print(f"    {fmt(nested_metrics)}")
        counts = pd.Series(picked).value_counts()
        print(f"    폴드별 선택 분포: {counts.to_dict()}")

        print("\n[4] 베이스라인")
        baselines = baseline_scores(X, y)
        for name, metrics in baselines.items():
            print(f"    {name:22s} {fmt(metrics)}")

        print("\n[5] 연봉대별 편향 (중첩 CV 예측 기준)")
        table = error_table(df["player_name"], actual, np.clip(np.expm1(nested_pred), 0, None))
        bands = band_summary(table)
        print(bands.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

        rates = hit_rates(table)
        print("\n[5-1] 맞춘 비율 (중첩 CV)")
        print("    " + "   ".join(f"{k} {v:.0%}" if k != "순위상관" else f"{k} {v:.3f}"
                                  for k, v in rates.items()))

        print("\n[6] 오차가 큰 5명")
        worst = table.nlargest(5, "절대오차")
        print(worst.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

        print("\n[7] 학습 백분위 vs 서빙 백분위")
        gap = serving_percentile_gap(engineered, pct_cols, group_col)
        print(f"    최대 차이 {gap['차이'].max():.4f}   평균 차이 {gap['차이'].mean():.4f}"
              f"   0.05 넘는 건수 {(gap['차이'] > 0.05).sum()} / {len(gap)}")

        report += [
            f"## {korean}",
            "",
            f"- 표본 {len(df)}명, 피처 {len(feature_cols)}개",
            "",
            "| 측정 | R² | RMSE | MAE |",
            "|---|---|---|---|",
            f"| 인샘플 (앱이 과거 FA에 보여주는 값) | {in_metrics['r2']:.3f} | "
            f"{in_metrics['rmse']:.2f} | {in_metrics['mae']:.2f} |",
            f"| meta 기록 OOF (선택 편향 포함) | {meta['r2']:.3f} | "
            f"{meta['rmse_억']:.2f} | {meta['mae_억']:.2f} |",
            f"| 중첩 CV (선택까지 폴드 안) | {nested_metrics['r2']:.3f} | "
            f"{nested_metrics['rmse']:.2f} | {nested_metrics['mae']:.2f} |",
        ]
        for name, metrics in baselines.items():
            report.append(f"| 베이스라인 — {name} | {metrics['r2']:.3f} | "
                          f"{metrics['rmse']:.2f} | {metrics['mae']:.2f} |")

        report += [
            "",
            "### 맞춘 비율 (중첩 CV)",
            "",
            "  ·  ".join(f"{k} **{v:.0%}**" if k != "순위상관" else f"{k} **{v:.3f}**"
                         for k, v in rates.items()),
            "",
            "### 연봉대별 편향 (중첩 CV)",
            "",
            to_md(bands),
            "",
            "### 오차가 큰 5명",
            "",
            to_md(worst),
            "",
            "### 학습/서빙 백분위 차이",
            "",
            f"최대 {gap['차이'].max():.4f} · 평균 {gap['차이'].mean():.4f} · "
            f"0.05 초과 {(gap['차이'] > 0.05).sum()}건 / {len(gap)}건",
            "",
        ]
        print()

    out = ROOT / "output" / "reports" / "accuracy_v8.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"리포트 저장: {out}")


if __name__ == "__main__":
    main()
