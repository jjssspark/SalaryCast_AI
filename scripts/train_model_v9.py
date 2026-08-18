"""v9 학습 — 인플레이션 보정을 절반 섞고, 투수는 피처를 줄인다.

v8과 무엇이 다른가:

1. 예측을 두 벌 만들어 로그 공간에서 절반씩 섞는다.
   - base : v8과 같다. log1p(연평균 계약금)을 맞힌다.
   - ratio: log1p(연봉) - log1p(그해 시장 수준)을 맞힌다.

   왜: 시간 순서로 검증하면 v8은 거의 모든 해를 낮게 불렀다(타자 -1.5억,
   투수 -3.7억). 트리 모델이 fa_year를 외삽하지 못해서다. 학습이 2021년까지면
   2022년 입력을 2021년과 똑같이 취급한다. ratio 쪽은 추세를 모델이 아니라
   시장 지표가 담당하므로 이 문제를 피한다. 둘을 섞은 것이 각각보다 나았다.

2. 투수는 폴드 안 선택에서 상위 15개 피처만 남긴 쪽이 나았다.
   46명에 47피처라 그대로 쓰면 과적합한다.

실험 근거는 output/reports/experiment_v9.md에 있다. 채택 기준은 시간 순서 MAE다.

실행: .venv/bin/python scripts/train_model_v9.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_v9 as E  # noqa: E402
import train_model_v8 as T  # noqa: E402
from src.features import (  # noqa: E402
    HITTER_PCT_COLS,
    PITCHER_PCT_COLS,
    build_reference_dist,
)

# base와 ratio를 섞는 비율(base 쪽 가중치)과 남길 피처 수. 둘 다 그룹마다 다르다.
# 계약을 175건으로 늘린 뒤 시간 순서 검증으로 다시 골랐다
# (scripts/experiment_topk.py, output/reports/experiment_topk.md).
#
# 타자는 ratio만 쓰는 게 좋고(mix 0.0) 투수는 base를 3/4 쓰는 게 좋다. 방향이
# 반대인 이유가 데이터에 있다. ratio는 연봉을 그해 시장 수준으로 나눈 값을
# 맞히는데, 그 분모는 직전 3년 계약의 중앙값이다. 타자는 연간 계약이 6~15건이라
# 중앙값이 안정적이지만 투수는 1~9건이고 2022년은 한 건뿐이다. 못 믿을 분모로
# 나눈 값을 많이 쓰면 안 된다.
#
# 피처 수는 투수만 자른다. v9 초기에는 15개였는데 너무 과했다. 25개가 최적이고
# 그 주변(20~47)이 모두 15개보다 낫다. 타자는 자르면 전 구간에서 나빠진다.
GROUPS = [
    {"label": "hitter", "korean": "타자", "csv": "hitter_training_v10.csv",
     "pct_cols": HITTER_PCT_COLS, "group_col": "position",
     "top_k": None, "mix_weight": 0.0},
    {"label": "pitcher", "korean": "투수", "csv": "pitcher_training_v10.csv",
     "pct_cols": PITCHER_PCT_COLS, "group_col": "pitcher_role",
     "top_k": 25, "mix_weight": 0.75},
]


def choose(X: pd.DataFrame, y: pd.Series) -> dict:
    """v8과 같은 절차로 단일/블렌드를 고른다."""
    results = T.evaluate_all(X, y)
    first, second, weight, blend_metrics = T.best_blend(results, y)
    single = max(results.items(), key=lambda kv: kv[1]["r2"])
    use_blend = blend_metrics["r2"] > single[1]["r2"] + 1e-6

    if use_blend:
        oof = weight * results[first]["pred"] + (1 - weight) * results[second]["pred"]
        return {"method": f"blend:{first}+{second}", "blend_first": first,
                "blend_second": second, "weight": float(weight),
                "members": sorted({first, second}), "oof": oof}

    name = single[0]
    return {"method": name, "blend_first": name, "blend_second": None,
            "weight": 1.0, "members": [name], "oof": results[name]["pred"]}


def _fit_part(spec: dict, X, y_log, level, train_idx, test_idx, part: str) -> np.ndarray:
    """한 폴드에서 part 하나를 학습해 로그 예측을 낸다. 선택도 폴드 안에서."""
    columns = [c for c in X.columns if not (part == "ratio" and c == "market_level")]
    target = y_log - np.log1p(level) if part == "ratio" else y_log
    y_tr = target.iloc[train_idx]

    if spec["top_k"]:
        columns = E.top_k_features(X.iloc[train_idx][columns], y_tr, spec["top_k"])

    X_tr, X_te = X.iloc[train_idx][columns], X.iloc[test_idx][columns]
    with contextlib.redirect_stdout(io.StringIO()):
        chosen = choose(X_tr, y_tr)

    fitted = {name: T.make_models()[name].fit(X_tr, y_tr) for name in chosen["members"]}
    if chosen["blend_second"]:
        w = chosen["weight"]
        pred = (w * fitted[chosen["blend_first"]].predict(X_te)
                + (1 - w) * fitted[chosen["blend_second"]].predict(X_te))
    else:
        pred = fitted[chosen["blend_first"]].predict(X_te)

    return pred + np.log1p(level.iloc[test_idx].to_numpy()) if part == "ratio" else pred


def time_metrics_full(spec: dict, X, y_log, level, years) -> dict:
    """시간 순서 검증. 모델 선택과 피처 선택까지 폴드 안에서 다시 한다.

    E.evaluate는 v8 구성을 고정해 쓰므로 실험 비교용으로는 맞지만, 배포될
    모델의 성능으로 적기에는 어긋난다. 화면의 오차 범위가 이 값이라 정확히 잰다.
    """
    actual = np.expm1(y_log).to_numpy()
    predicted = np.full(len(y_log), np.nan)

    for cutoff in sorted(years.unique()):
        train_idx = np.where(years < cutoff)[0]
        test_idx = np.where(years == cutoff)[0]
        if len(train_idx) < E.MIN_TRAIN or len(test_idx) == 0:
            continue
        args = (spec, X, y_log, level, train_idx, test_idx)
        weight = spec["mix_weight"]
        mixed = (weight * _fit_part(*args, "base")
                 + (1 - weight) * _fit_part(*args, "ratio"))
        predicted[test_idx] = np.clip(np.expm1(mixed), 0, None)

    seen = ~np.isnan(predicted)
    return {**E._score(actual[seen], predicted[seen]), "n": int(seen.sum())}


def train_group(spec: dict) -> dict:
    label, korean = spec["label"], spec["korean"]
    print("=" * 66)
    print(f"  {korean} 모델 학습 (v9)")
    print("=" * 66)

    df = pd.read_csv(T.DATA_DIR / spec["csv"], encoding="utf-8-sig")
    X, y_log, feature_cols, engineered = T.prepare(df, spec["pct_cols"], spec["group_col"])
    level = E.market_levels(engineered)
    print(f"데이터 {len(df)}명 / 피처 {len(feature_cols)}개")

    parts: dict[str, dict] = {}
    members: list[str] = []
    oof_log: dict[str, np.ndarray] = {}

    for part in ("base", "ratio"):
        # ratio는 market_level을 분모로 쓰므로 피처에서 뺀다.
        columns = [c for c in feature_cols if not (part == "ratio" and c == "market_level")]
        target = y_log - np.log1p(level) if part == "ratio" else y_log

        if spec["top_k"]:
            columns = E.top_k_features(X[columns], target, spec["top_k"])

        print(f"\n  [{part}] 피처 {len(columns)}개")
        chosen = choose(X[columns], target)
        print(f"    선정: {chosen['method']}")

        # base 멤버는 접두사 없이 저장한다. explain.py가 트리 모델을
        # 'LightGBM' 같은 이름으로 찾기 때문이다.
        prefix = "" if part == "base" else "ratio_"
        for name in chosen["members"]:
            model = T.make_models()[name]
            model.fit(X[columns], target)
            key = f"{prefix}{name}"
            joblib.dump(model, T.MODEL_DIR / f"{label}_v9_{key.lower()}.pkl")
            members.append(key)

        oof = chosen.pop("oof")
        oof_log[part] = oof + np.log1p(level) if part == "ratio" else oof
        parts[part] = {**chosen, "features": columns, "prefix": prefix}

    mix_weight = spec["mix_weight"]
    mix_oof = mix_weight * oof_log["base"] + (1 - mix_weight) * oof_log["ratio"]
    random_metrics = T.score_in_billions(y_log, mix_oof)
    print(f"\n  혼합 무작위 OOF   R² {random_metrics['r2']:.3f}  "
          f"RMSE {random_metrics['rmse']:.2f}억  MAE {random_metrics['mae']:.2f}억")

    # 시간 순서 — 실제 용도(미래 FA 예측)에 가까운 숫자
    time_metrics = time_metrics_full(
        spec, X, y_log, level, engineered["fa_year"].astype(int))
    print(f"  혼합 시간 순서    R² {time_metrics['r2']:.3f}  "
          f"MAE {time_metrics['mae']:.2f}억  편향 {time_metrics['bias']:+.2f}억"
          f"  (대상 {time_metrics['n']}명)")

    reference = build_reference_dist(engineered, spec["pct_cols"], spec["group_col"])
    joblib.dump(reference, T.MODEL_DIR / f"reference_dist_{label}_v9.pkl")

    # 예측을 실제로 끌고 가는 쪽. 화면의 '핵심 요소'는 이 부분의 모델에서 뽑는다.
    # 타자는 mix_weight가 0이라 base가 예측에 한 푼도 기여하지 않는다. 그 모델로
    # 근거를 설명하면 예측과 무관한 이야기를 보여주게 된다.
    explain_part = "base" if mix_weight >= 0.5 else "ratio"

    meta = {
        "features": parts[explain_part]["features"],  # explain.py가 쓰는 기준 피처
        "explain_part": explain_part,
        "members": members,
        "target_mode": "mix",
        "target": (f"{mix_weight}*log1p(salary) + {1 - mix_weight}*"
                   "(log1p(salary/market_level) + log1p(market_level))"),
        "mix_weight": mix_weight,
        "parts": parts,
        "n_samples": len(df),
        # 화면의 오차 범위는 시간 순서 MAE를 쓴다. 미래 FA 예측이 실제 용도라
        # 무작위 K-fold보다 이쪽이 정직하다.
        "mae_억": time_metrics["mae"],
        "r2": random_metrics["r2"],
        "rmse_억": random_metrics["rmse"],
        "random_mae_억": random_metrics["mae"],
        "time_r2": time_metrics["r2"],
        "time_mae_억": time_metrics["mae"],
        "time_bias_억": time_metrics["bias"],
        "time_n": time_metrics["n"],
    }
    joblib.dump(meta, T.MODEL_DIR / f"{label}_v9_meta.pkl")
    print(f"  저장: {label}_v9_*.pkl ({len(members)}개 모델)")
    return {"label": label, "korean": korean, "meta": meta}


def main() -> None:
    entries = [train_group(spec) for spec in GROUPS]

    lines = ["# v9 모델 학습 결과", "",
             "v8 대비 바뀐 것: 인플레이션 보정 예측을 절반 섞고, 투수는 피처를 15개로 줄였다.",
             "화면의 오차 범위(mae_억)는 시간 순서 MAE를 쓴다.", "",
             "| 그룹 | 표본 | 시간 R² | 시간 MAE | 시간 편향 | 무작위 R² | 무작위 MAE |",
             "|---|---|---|---|---|---|---|"]
    for entry in entries:
        m = entry["meta"]
        lines.append(
            f"| {entry['korean']} | {m['n_samples']}명 | {m['time_r2']:.3f} | "
            f"{m['time_mae_억']:.2f}억 | {m['time_bias_억']:+.2f}억 | "
            f"{m['r2']:.3f} | {m['random_mae_억']:.2f}억 |"
        )
    lines += ["", "## 구성", ""]
    for entry in entries:
        m = entry["meta"]
        lines.append(f"- **{entry['korean']}**")
        for part, spec in m["parts"].items():
            lines.append(f"  - {part}: {spec['method']} · 피처 {len(spec['features'])}개")

    out = ROOT / "output" / "reports" / "model_v9.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n리포트 저장: {out}")


if __name__ == "__main__":
    main()
