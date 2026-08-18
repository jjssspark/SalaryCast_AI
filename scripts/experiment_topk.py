"""피처 수를 줄이면 나아지는가.

v9에서 투수는 47개 중 상위 15개만 남겨 R²가 0.143 -> 0.378로 올랐다. 표본
46명에 피처 47개는 너무 많았기 때문이다. 그런데 타자는 그 처방을 안 했다.
지금 타자는 113명에 피처 55개다. 비율로 보면 투수와 크게 다르지 않다.

피처 수(TOP_K)와 섞는 비율(MIX_WEIGHT)을 같이 본다. 둘은 독립이 아니다 —
피처를 줄이면 base 쪽이 안정되므로 섞는 비율의 최적점도 움직인다.

폴드마다 base·ratio 로그 예측을 한 번만 만들고 섞기만 바꾸므로, 실제 학습은
TOP_K 후보 수만큼만 돈다.

평가는 train_model_v9와 같은 시간 순서 검증이다.

주의: 채택 기준과 보고 수치가 같은 지표라 고른 값은 그쪽으로 휜다.
후보를 적게 두고, 이긴 폭이 작으면 바꾸지 않는다.

출력: output/reports/experiment_topk.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_v9 as E  # noqa: E402
import train_model_v8 as T  # noqa: E402
import train_model_v9 as V9  # noqa: E402

MIX_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_K_CANDIDATES = {
    "hitter": [None, 35, 25, 15],
    # 처음에는 10·15·20만 봤는데 격자 끝인 20이 이겼다. 끝에서 이기면 더 밀어봐야
    # 한다. 25까지 좋아지다가 30·전체에서 다시 나빠졌다.
    "pitcher": [None, 30, 25, 20, 15, 10],
}


def fold_parts(spec: dict, df: pd.DataFrame):
    X, y_log, _, engineered = T.prepare(df, spec["pct_cols"], spec["group_col"])
    level = E.market_levels(engineered)
    years = df["fa_year"].reset_index(drop=True)

    base = np.full(len(y_log), np.nan)
    ratio = np.full(len(y_log), np.nan)
    for cutoff in sorted(years.unique()):
        train_idx = np.where(years < cutoff)[0]
        test_idx = np.where(years == cutoff)[0]
        if len(train_idx) < E.MIN_TRAIN or len(test_idx) == 0:
            continue
        args = (spec, X, y_log, level, train_idx, test_idx)
        base[test_idx] = V9._fit_part(*args, "base")
        ratio[test_idx] = V9._fit_part(*args, "ratio")
    return np.expm1(y_log).to_numpy(), base, ratio


def main() -> None:
    lines = ["# 피처 수 · 섞는 비율 동시 탐색", "",
             "시간 순서 검증. 폴드 안에서 모델과 피처를 다시 고른다.",
             "TOP_K는 학습 폴드에서 상관 상위 몇 개를 남길지, MIX_WEIGHT는 base 쪽 가중치다.", ""]

    for spec in V9.GROUPS:
        label = spec["label"]
        df = pd.read_csv(T.DATA_DIR / spec["csv"], encoding="utf-8-sig")
        current_k, current_mix = spec["top_k"], spec["mix_weight"]
        results = []

        for top_k in TOP_K_CANDIDATES[label]:
            trial = {**spec, "top_k": top_k}
            actual, base, ratio = fold_parts(trial, df)
            seen = ~np.isnan(base) & ~np.isnan(ratio)
            for weight in MIX_CANDIDATES:
                predicted = np.clip(np.expm1(weight * base[seen] + (1 - weight) * ratio[seen]), 0, None)
                score = E._score(actual[seen], predicted)
                score["bias"] = float(np.mean(predicted - actual[seen]))
                results.append((top_k, weight, score, int(seen.sum())))
            print(f"  {label} TOP_K={top_k} 완료", flush=True)

        results.sort(key=lambda r: r[2]["mae"])
        best_k, best_w, best_s, n = results[0]
        now = next((r for r in results
                    if r[0] == current_k and r[1] == current_mix), results[0])

        print(f"\n{spec['korean']} — 평가 대상 {n}명")
        print(f"  현재  TOP_K={current_k} MIX={current_mix}  "
              f"R² {now[2]['r2']:.3f}  MAE {now[2]['mae']:.2f}억  편향 {now[2]['bias']:+.2f}억")
        print(f"  최선  TOP_K={best_k} MIX={best_w}  "
              f"R² {best_s['r2']:.3f}  MAE {best_s['mae']:.2f}억  편향 {best_s['bias']:+.2f}억")
        print("  상위 6개")
        for top_k, weight, score, _ in results[:6]:
            print(f"    TOP_K={str(top_k):>4} MIX={weight:.2f}  "
                  f"R² {score['r2']:6.3f}  MAE {score['mae']:.2f}억  편향 {score['bias']:+.2f}억")

        lines += [f"## {spec['korean']} (평가 대상 {n}명)", "",
                  "| TOP_K | MIX_WEIGHT | R² | RMSE | MAE | 편향 |", "|---|---|---|---|---|---|"]
        lines += [
            f"| {top_k} | {weight:.2f} | {score['r2']:.3f} | {score['rmse']:.2f}억 | "
            f"{score['mae']:.2f}억 | {score['bias']:+.2f}억 |"
            for top_k, weight, score, _ in results
        ]
        lines += [""]

    out = ROOT / "output" / "reports" / "experiment_topk.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n리포트 저장: {out}")


if __name__ == "__main__":
    main()
