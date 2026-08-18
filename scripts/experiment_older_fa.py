"""계약 데이터를 과거로 넓힌 것이 예측을 낫게 하는가.

왜 따로 재는가. 계약을 175건으로 늘리고 나서 시간 순서 R²가 타자 0.536 → 0.424로
떨어졌다. 그런데 이 두 숫자는 서로 비교할 수 없다. 시간 순서 검증은 '그 해 이전
데이터로 그 해를 예측'하는데, 2016·2017이 생기면서 예전에는 학습 표본이 모자라
평가하지 못했던 이른 연도까지 평가 대상에 들어왔기 때문이다. 평가 대상이 바뀌면
점수도 바뀐다.

그래서 평가 대상을 고정하고 학습 표본만 바꿔서 잰다.

  기존: 2018년 이후 계약으로만 학습
  확장: 2014년 이후 계약으로 학습
  평가: 두 경우 모두 2018년 이후 계약만 — 같은 사람, 같은 해

학습·예측 절차는 train_model_v9.time_metrics_full과 같다. 폴드 안에서 모델과
피처를 다시 고르는 것까지 그대로다.

출력: output/reports/experiment_older_fa.md
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

BASELINE_FROM = 2018  # 확장 전 계약 테이블이 담고 있던 가장 이른 FA 연도
EXPANDED_FROM = 2014  # 지금 담고 있는 가장 이른 FA 연도


def time_predictions(spec: dict, df: pd.DataFrame, train_from: int) -> np.ndarray:
    """시간 순서 예측. train_from 이후 계약만 학습에 쓴다. 평가 대상은 전체."""
    X, y_log, _, engineered = T.prepare(df, spec["pct_cols"], spec["group_col"])
    level = E.market_levels(engineered)
    years = df["fa_year"].reset_index(drop=True)

    predicted = np.full(len(y_log), np.nan)
    for cutoff in sorted(years.unique()):
        train_idx = np.where((years < cutoff) & (years >= train_from))[0]
        test_idx = np.where(years == cutoff)[0]
        if len(train_idx) < E.MIN_TRAIN or len(test_idx) == 0:
            continue
        args = (spec, X, y_log, level, train_idx, test_idx)
        # 여기서 재는 것은 학습 표본을 넓힌 효과다. 섞는 비율은 지금 쓰는 값으로
        # 두 조건 모두 고정한다.
        weight = spec["mix_weight"]
        mixed = (weight * V9._fit_part(*args, "base")
                 + (1 - weight) * V9._fit_part(*args, "ratio"))
        predicted[test_idx] = np.clip(np.expm1(mixed), 0, None)
    return predicted


def main() -> None:
    lines = ["# 계약을 과거로 넓힌 효과", "",
             "평가 대상을 2018년 이후 계약으로 고정하고 학습 표본만 바꿔 비교했다.",
             "시간 순서 검증(과거로만 학습해 그 해를 예측), 폴드 안에서 모델·피처 재선택.", ""]

    for spec in V9.GROUPS:
        df = pd.read_csv(T.DATA_DIR / spec["csv"], encoding="utf-8-sig")
        actual = df["annual_avg_salary"].to_numpy()
        years = df["fa_year"].to_numpy()

        old = time_predictions(spec, df, BASELINE_FROM)
        new = time_predictions(spec, df, EXPANDED_FROM)

        # 두 쪽 다 예측을 낸 2018년 이후 계약만 본다.
        seen = ~np.isnan(old) & ~np.isnan(new) & (years >= BASELINE_FROM)
        old_score = E._score(actual[seen], old[seen])
        new_score = E._score(actual[seen], new[seen])

        print(f"\n{spec['korean']} — 평가 대상 {int(seen.sum())}명 (2018년 이후)")
        print(f"  2018~ 학습   R² {old_score['r2']:.3f}  MAE {old_score['mae']:.2f}억")
        print(f"  2014~ 학습   R² {new_score['r2']:.3f}  MAE {new_score['mae']:.2f}억")

        lines += [
            f"## {spec['korean']} (평가 대상 {int(seen.sum())}명)", "",
            "| 학습 표본 | R² | RMSE | MAE |",
            "|---|---|---|---|",
            f"| 2018년 이후만 (기존) | {old_score['r2']:.3f} | {old_score['rmse']:.2f}억 | {old_score['mae']:.2f}억 |",
            f"| 2014년 이후 (확장) | {new_score['r2']:.3f} | {new_score['rmse']:.2f}억 | {new_score['mae']:.2f}억 |",
            "",
        ]

    out = ROOT / "output" / "reports" / "experiment_older_fa.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n리포트 저장: {out}")


if __name__ == "__main__":
    main()
