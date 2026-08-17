# 누수 제거 전후 비교 (v8)

`market_level`은 v7까지 `groupby(fa_year)['annual_avg_salary'].median()`이었음.
같은 해 계약의 중앙값이라 자기 자신이 포함된 값이고, 정답을 직접 흘림.
v8은 직전 3개 연도 계약만 보고 계산함.

지표는 전부 **억 원 단위로 되돌린 뒤** 계산한 OOF 값임 (5-fold x 5회 반복, XGBoost 고정).

| 대상 | 방식 | R2 | RMSE(억) | MAE(억) |
|---|---|---|---|---|
| hitter | v7 방식(누수) | 0.573 | 4.79 | 3.14 |
| hitter | v8 방식(제거) | 0.559 | 4.86 | 3.17 |
| pitcher | v7 방식(누수) | 0.571 | 3.04 | 2.35 |
| pitcher | v8 방식(제거) | 0.572 | 3.04 | 2.33 |

## 최종 선정 모델

| 대상 | 방식 | R2 | RMSE(억) | MAE(억) | 표본 |
|---|---|---|---|---|---|
| hitter | blend:LightGBM+Ridge | 0.618 | 4.53 | 3.01 | 93명 |
| pitcher | blend:XGBoost+Ridge | 0.645 | 2.77 | 2.07 | 46명 |
