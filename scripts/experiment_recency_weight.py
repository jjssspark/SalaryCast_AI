"""오래된 계약의 비중을 낮추면 나아지는가.

계약을 2014년까지 넓혔더니 전체 시간 순서 성능은 올랐지만, 평가를 2018년 이후로
고정한 통제 실험에서는 타자가 오히려 나빠졌다(R² 0.587 -> 0.504,
output/reports/experiment_older_fa.md). 옛 계약이 이른 연도 학습에는 도움이 되고
최근 연도 예측에는 방해라는 뜻이다.

그렇다면 옛 계약을 버리지 말고 비중만 낮추면 둘 다 챙길 수 있다. 학습 표본마다
가중치를 준다.

    w = 0.5 ** ((예측할 해 - 계약 연도) / 반감기)

반감기 4년이면 4년 전 계약은 절반, 8년 전은 1/4로 센다. 반감기가 무한대면
지금과 같다.

가중치는 최종 적합에만 건다. 모델 선택은 가중치 없이 해서 가중치 효과만 본다.

평가는 train_model_v9와 같은 시간 순서 검증이다.

주의: 채택 기준과 보고 수치가 같은 지표다. 이긴 폭이 작으면 바꾸지 않는다.

출력: output/reports/experiment_recency_weight.md
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

# 타자는 격자 끝인 2년에서 이겨서 1년까지 더 밀어봤다. 2~3년이 봉우리고
# 양끝에서 다시 나빠진다. 투수는 어떤 값도 "없음"보다 나쁘고 짧을수록
# 나빠지는 단조라 더 밀어볼 이유가 없었다.
HALF_LIVES = {
    "hitter": [None, 10, 6, 4, 3, 2, 1.5, 1],
    "pitcher": [None, 10, 6, 4, 2],
}


def run(spec: dict, df: pd.DataFrame, half_life: float | None) -> tuple[dict, int]:
    X, y_log, _, engineered = T.prepare(df, spec["pct_cols"], spec["group_col"])
    level = E.market_levels(engineered)
    years = engineered["fa_year"].astype(int).reset_index(drop=True)

    actual = np.expm1(y_log).to_numpy()
    predicted = np.full(len(y_log), np.nan)
    mix = spec["mix_weight"]

    for cutoff in sorted(years.unique()):
        train_idx = np.where(years < cutoff)[0]
        test_idx = np.where(years == cutoff)[0]
        if len(train_idx) < E.MIN_TRAIN or len(test_idx) == 0:
            continue

        weight = V9.recency_weight(half_life, years.iloc[train_idx], cutoff)

        args = (spec, X, y_log, level, train_idx, test_idx)
        mixed = (mix * V9._fit_part(*args, "base", sample_weight=weight)
                 + (1 - mix) * V9._fit_part(*args, "ratio", sample_weight=weight))
        predicted[test_idx] = np.clip(np.expm1(mixed), 0, None)

    seen = ~np.isnan(predicted)
    score = E._score(actual[seen], predicted[seen])
    score["bias"] = float(np.mean(predicted[seen] - actual[seen]))
    return score, int(seen.sum())


def main() -> None:
    lines = ["# 오래된 계약 비중 낮추기", "",
             "학습 표본에 `0.5 ** ((예측할 해 - 계약 연도) / 반감기)` 가중치를 준다.",
             "반감기 없음이 지금 설정이다. 시간 순서 검증, 그룹별 설정은 그대로 둔다.", ""]

    for spec in V9.GROUPS:
        df = pd.read_csv(T.DATA_DIR / spec["csv"], encoding="utf-8-sig")
        rows = []
        for half_life in HALF_LIVES[spec["label"]]:
            score, n = run(spec, df, half_life)
            rows.append((half_life, score, n))
            print(f"  {spec['label']} 반감기={half_life}  R² {score['r2']:.3f}  "
                  f"MAE {score['mae']:.2f}억  편향 {score['bias']:+.2f}억", flush=True)

        n = rows[0][2]
        best = min(rows, key=lambda r: r[1]["mae"])
        print(f"\n{spec['korean']} — 평가 대상 {n}명  최선 반감기={best[0]} "
              f"MAE {best[1]['mae']:.2f}억\n", flush=True)

        lines += [f"## {spec['korean']} (평가 대상 {n}명)", "",
                  "| 반감기 | R² | RMSE | MAE | 편향 |", "|---|---|---|---|---|"]
        lines += [
            f"| {'없음' if hl is None else f'{hl}년'} | {s['r2']:.3f} | {s['rmse']:.2f}억 | "
            f"{s['mae']:.2f}억 | {s['bias']:+.2f}억 |"
            for hl, s, _ in rows
        ]
        lines += [""]

    out = ROOT / "output" / "reports" / "experiment_recency_weight.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트 저장: {out}")


if __name__ == "__main__":
    main()
