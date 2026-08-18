"""base와 ratio를 섞는 비율을 다시 고른다.

왜. 계약을 175건으로 늘린 뒤 시간 순서 편향이 타자 -1.33억 -> -2.29억으로
커졌다. 편향이 음수라는 건 실제보다 낮게 부른다는 뜻이다. 2016·2017 계약이
지금보다 싸므로, 연봉 자체를 맞히는 base 쪽이 그 수준으로 끌려간 것으로 본다.

ratio는 연봉을 그해 시장 수준으로 나눈 비율을 맞히므로 시세 변동을 모델이 아니라
시장 지표가 담당한다. 섞는 비율(MIX_WEIGHT, base 쪽 가중치)을 낮추면 이 끌림이
줄어야 한다. v9는 0.5를 근거 없이 고정한 값이었다.

평가는 train_model_v9와 같은 시간 순서 검증이다. 과거로만 학습해 그 해를
예측하고, 모델·피처 선택까지 폴드 안에서 다시 한다.

주의: 채택 기준과 보고 수치가 같은 지표다. 여기서 고른 값은 그 지표 쪽으로
휜다. 그래서 후보를 5개로만 두고 그중 최선을 고른다.

출력: output/reports/experiment_mix_weight.md
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

CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def fold_parts(spec, df):
    """폴드마다 base·ratio 예측을 로그 공간으로 한 번만 만들어 둔다.

    섞는 비율만 바꿔 보는 것이므로 학습을 후보 수만큼 반복할 이유가 없다.
    """
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
    lines = ["# base·ratio 섞는 비율 재선정", "",
             "MIX_WEIGHT는 base 쪽 가중치다. 0이면 ratio만, 1이면 base만 쓴다.",
             "시간 순서 검증(과거로만 학습해 그 해를 예측, 폴드 안에서 모델·피처 재선택).", ""]
    chosen = {}

    for spec in V9.GROUPS:
        df = pd.read_csv(T.DATA_DIR / spec["csv"], encoding="utf-8-sig")
        actual, base, ratio = fold_parts(spec, df)
        seen = ~np.isnan(base) & ~np.isnan(ratio)

        rows = []
        for weight in CANDIDATES:
            mixed = weight * base[seen] + (1 - weight) * ratio[seen]
            predicted = np.clip(np.expm1(mixed), 0, None)
            score = E._score(actual[seen], predicted)
            score["bias"] = float(np.mean(predicted - actual[seen]))
            rows.append((weight, score))

        best = min(rows, key=lambda r: r[1]["mae"])[0]
        chosen[spec["label"]] = best

        print(f"\n{spec['korean']} — 평가 대상 {int(seen.sum())}명")
        for weight, score in rows:
            mark = "  <- 선정" if weight == best else ""
            print(f"  MIX {weight:.2f}   R² {score['r2']:6.3f}  "
                  f"MAE {score['mae']:.2f}억  편향 {score['bias']:+.2f}억{mark}")

        lines += [f"## {spec['korean']} (평가 대상 {int(seen.sum())}명)", "",
                  "| MIX_WEIGHT | R² | RMSE | MAE | 편향 |", "|---|---|---|---|---|"]
        lines += [
            f"| {w:.2f}{' (선정)' if w == best else ''} | {s['r2']:.3f} | "
            f"{s['rmse']:.2f}억 | {s['mae']:.2f}억 | {s['bias']:+.2f}억 |"
            for w, s in rows
        ]
        lines += [""]

    lines += ["## 선정", "", "| 그룹 | MIX_WEIGHT |", "|---|---|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in chosen.items()]

    out = ROOT / "output" / "reports" / "experiment_mix_weight.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n선정: {chosen}")
    print(f"리포트 저장: {out}")


if __name__ == "__main__":
    main()
