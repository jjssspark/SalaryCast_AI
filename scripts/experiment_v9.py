"""v9 후보 실험 — 인플레이션 보정과 피처 축소. 읽기 전용.

왜 이걸 하는가:
시간 순서로 검증하면(과거로만 학습해 그해를 예측) 거의 모든 해를 낮게 부른다.
타자 -1.4 ~ -3.2억, 투수 -3.0 ~ -4.9억. 연봉이 해마다 오르는데 모델이 못 따라간다.

원인은 트리 모델이 fa_year를 외삽하지 못하는 것이다. 학습이 2021년까지면
2022년 입력을 2021년과 똑같이 취급한다. Ridge는 외삽하지만 블렌드에서 비중이 작다.

그래서 타깃을 '그해 시장 수준 대비 비율'로 바꾼다. 추세는 모델이 아니라
시장 지표가 담당하고, 모델은 '이 선수가 시장 대비 얼마나 비싼가'만 배운다.

같이 보는 것: 투수는 46명에 47피처라 폴드 안에서 상위 K개만 남겨 재본다.

평가는 시간 순서를 1순위로 본다. 무작위 K-fold는 참고로만 둔다.
앱이 하는 일이 미래 FA 예측이라 시간 순서가 실제 용도에 가깝다.

실행: .venv/bin/python scripts/experiment_v9.py
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

from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402

import train_model_v8 as T  # noqa: E402
from src.features import HITTER_PCT_COLS, PITCHER_PCT_COLS  # noqa: E402

GROUPS = [
    ("hitter", "타자", "hitter_training_v8.csv", HITTER_PCT_COLS, "position"),
    ("pitcher", "투수", "pitcher_training_v8.csv", PITCHER_PCT_COLS, "pitcher_role"),
]

TOP_K = 15
MIN_TRAIN = 25


def market_levels(engineered: pd.DataFrame) -> pd.Series:
    """각 계약의 '그해 시장 수준'. 나눗셈의 분모라 NaN이면 안 된다.

    기본은 직전 3년 FA 연평균 중앙값(market_level 피처와 같은 값)이다.
    가장 이른 연도는 참고할 과거가 없어 NaN이라, 같은 해 계약의 중앙값을
    자기 자신만 빼고 계산해 채운다. 자기 연봉이 분모에 들어가면 정답이 샌다.
    """
    level = engineered["market_level"].astype(float).copy()
    salary = engineered["annual_avg_salary"].astype(float)
    year = engineered["fa_year"].astype(int)

    for idx in level[level.isna()].index:
        peers = salary[(year == year[idx]) & (salary.index != idx)]
        level[idx] = float(peers.median())

    return level


def top_k_features(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    """폴드 안에서만 고른다. 전체로 고르면 테스트 정보가 새어 들어온다."""
    probe = LGBMRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        min_child_samples=5, random_state=T.SEED, verbose=-1, n_jobs=1,
    ).fit(X, y)
    order = np.argsort(probe.feature_importances_)[::-1]
    return [X.columns[i] for i in order[:k]]


def deployed_config(label: str) -> dict:
    meta = joblib.load(T.MODEL_DIR / f"{label}_v8_meta.pkl")
    if meta["method"].startswith("blend:"):
        return {"blend": True, "first": meta["blend_first"],
                "second": meta["blend_second"], "weight": float(meta["weight"])}
    return {"blend": False, "first": meta["method"], "second": None, "weight": 1.0}


def fit_predict(config: dict, X_tr, y_tr, X_te) -> np.ndarray:
    if config["blend"]:
        a = T.make_models()[config["first"]].fit(X_tr, y_tr)
        b = T.make_models()[config["second"]].fit(X_tr, y_tr)
        w = config["weight"]
        return w * a.predict(X_te) + (1 - w) * b.predict(X_te)
    return T.make_models()[config["first"]].fit(X_tr, y_tr).predict(X_te)


def _one_pass(config, X, y_log, level, train_idx, test_idx, ratio: bool, k: int | None):
    """로그 공간 예측 한 벌."""
    columns = list(X.columns)
    if ratio and "market_level" in columns:
        columns.remove("market_level")  # 분모로 쓰므로 피처에서 뺀다

    X_tr, X_te = X.iloc[train_idx][columns], X.iloc[test_idx][columns]

    # level은 과거 계약에서 나온 값이라 테스트 시점에도 알 수 있다.
    target = y_log - np.log1p(level) if ratio else y_log
    y_tr = target.iloc[train_idx]

    if k is not None:
        picked = top_k_features(X_tr, y_tr, k)
        X_tr, X_te = X_tr[picked], X_te[picked]

    pred = fit_predict(config, X_tr, y_tr, X_te)
    if ratio:
        pred = pred + np.log1p(level.iloc[test_idx].to_numpy())
    return pred


def run_variant(config, X, y_log, level, train_idx, test_idx, mode: str, k: int | None):
    """한 폴드. 억 원 단위 예측값을 돌려준다.

    mode: base(현행) · ratio(시장 대비 비율) · mix(둘을 로그 공간에서 절반씩)
    """
    args = (config, X, y_log, level, train_idx, test_idx)

    if mode == "mix":
        base = _one_pass(*args, False, k)
        adjusted = _one_pass(*args, True, k)
        pred = 0.5 * base + 0.5 * adjusted
    else:
        pred = _one_pass(*args, mode == "ratio", k)

    return np.clip(np.expm1(pred), 0, None)


def _score(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "r2": float(1 - np.sum((actual - pred) ** 2)
                    / np.sum((actual - actual.mean()) ** 2)),
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "bias": float(np.mean(pred - actual)),
    }


def evaluate(label, X, y_log, level, years, config, mode: str, k: int | None) -> dict:
    actual = np.expm1(y_log).to_numpy()

    # 시간 순서 — 과거로만 학습해 그해를 예측
    time_pred = np.full(len(y_log), np.nan)
    per_year = []
    for cutoff in sorted(years.unique()):
        train_idx = np.where(years < cutoff)[0]
        test_idx = np.where(years == cutoff)[0]
        if len(train_idx) < MIN_TRAIN or len(test_idx) == 0:
            continue
        got = run_variant(config, X, y_log, level, train_idx, test_idx, mode, k)
        time_pred[test_idx] = got
        per_year.append({"연도": int(cutoff), "대상": len(test_idx),
                         "MAE": float(np.mean(np.abs(got - actual[test_idx]))),
                         "평균오차": float(np.mean(got - actual[test_idx]))})

    seen = ~np.isnan(time_pred)
    time_metrics = {**_score(actual[seen], time_pred[seen]), "n": int(seen.sum())}

    # 무작위 K-fold — 참고
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
    total = np.zeros(len(y_log))
    count = np.zeros(len(y_log))
    for train_idx, test_idx in cv.split(X):
        total[test_idx] += run_variant(config, X, y_log, level, train_idx, test_idx, mode, k)
        count[test_idx] += 1

    return {"time": time_metrics, "random": _score(actual, total / count),
            "per_year": per_year}


def main() -> None:
    report = ["# v9 실험 — 인플레이션 보정과 피처 축소", "",
              "평가 1순위는 시간 순서 검증이다. 앱이 하는 일이 미래 FA 예측이라",
              "무작위 K-fold보다 실제 용도에 가깝다.", ""]

    for label, korean, csv_name, pct_cols, group_col in GROUPS:
        df = pd.read_csv(T.DATA_DIR / csv_name, encoding="utf-8-sig")
        X, y_log, feature_cols, engineered = T.prepare(df, pct_cols, group_col)
        level = market_levels(engineered)
        years = engineered["fa_year"].astype(int)
        config = deployed_config(label)

        print("=" * 74)
        print(f"  {korean} — 표본 {len(df)}명 / 피처 {len(feature_cols)}개")
        print("=" * 74)

        variants = [
            ("현행 (v8)", "base", None),
            ("인플레 보정", "ratio", None),
            ("보정 절반 혼합", "mix", None),
            (f"피처 상위 {TOP_K}", "base", TOP_K),
            (f"보정 + 상위 {TOP_K}", "ratio", TOP_K),
            (f"혼합 + 상위 {TOP_K}", "mix", TOP_K),
        ]

        rows = []
        details = {}
        for name, mode, k in variants:
            result = evaluate(label, X, y_log, level, years, config, mode, k)
            details[name] = result
            t, r = result["time"], result["random"]
            rows.append({
                "구성": name,
                "시간 R²": t["r2"], "시간 MAE": t["mae"], "시간 편향": t["bias"],
                "무작위 R²": r["r2"], "무작위 MAE": r["mae"],
            })
            print(f"  {name:18s} 시간 R² {t['r2']:6.3f}  MAE {t['mae']:5.2f}억  "
                  f"편향 {t['bias']:+5.2f}억   |  무작위 R² {r['r2']:6.3f}  MAE {r['mae']:5.2f}억")

        table = pd.DataFrame(rows)
        best = table.loc[table["시간 MAE"].idxmin(), "구성"]
        print(f"\n  시간 순서 MAE 최소: {best}   (대상 {details[best]['time']['n']}명)")
        print("\n  연도별 (그 구성):")
        print(pd.DataFrame(details[best]["per_year"]).to_string(
            index=False, float_format=lambda v: f"{v:.2f}"))
        print()

        report += [
            f"## {korean}",
            "",
            f"표본 {len(df)}명 · 피처 {len(feature_cols)}개 · 구성 고정 "
            f"{config['first']}{'+' + config['second'] if config['blend'] else ''}",
            "",
            "| 구성 | 시간 R² | 시간 MAE | 시간 편향 | 무작위 R² | 무작위 MAE |",
            "|---|---|---|---|---|---|",
        ]
        for row in rows:
            report.append(
                f"| {row['구성']} | {row['시간 R²']:.3f} | {row['시간 MAE']:.2f}억 | "
                f"{row['시간 편향']:+.2f}억 | {row['무작위 R²']:.3f} | {row['무작위 MAE']:.2f}억 |"
            )
        report += ["", f"시간 순서 MAE 최소: **{best}**", "",
                   "연도별 (그 구성):", "",
                   "| 연도 | 대상 | MAE | 평균오차 |", "|---|---|---|---|"]
        for entry in details[best]["per_year"]:
            report.append(f"| {entry['연도']} | {entry['대상']} | {entry['MAE']:.2f}억 | "
                          f"{entry['평균오차']:+.2f}억 |")
        report.append("")

    out = ROOT / "output" / "reports" / "experiment_v9.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"리포트 저장: {out}")


if __name__ == "__main__":
    main()
