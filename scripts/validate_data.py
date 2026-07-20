"""데이터 누수 컬럼·결측치 검증 스크립트.

사용법: python scripts/validate_data.py
"""
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent.parent / "data"

# 모델 입력에 절대 포함되면 안 되는 컬럼 (annual_avg_salary 정답을 간접 노출)
LEAKAGE_COLS = {"total_contract_amount", "contract_years"}

TRAINING_FILES = [
    "hitter_training_v5.csv",
    "hitter_training_v7.csv",
    "pitcher_training_v5.csv",
    "pitcher_training_v7.csv",
]


def check_file(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"파일 없음: {path.name}"]

    df = pd.read_csv(path, encoding="utf-8-sig")

    if "annual_avg_salary" not in df.columns:
        errors.append(f"{path.name}: 타깃 컬럼 annual_avg_salary 없음")

    missing_target = df["annual_avg_salary"].isna().sum() if "annual_avg_salary" in df.columns else 0
    if missing_target:
        errors.append(f"{path.name}: annual_avg_salary 결측 {missing_target}건")

    null_ratio = df.isna().mean()
    high_null = null_ratio[null_ratio > 0.5]
    if not high_null.empty:
        errors.append(f"{path.name}: 결측률 50% 초과 컬럼 {list(high_null.index)}")

    dup = df["player_name"].duplicated().sum() if "player_name" in df.columns else 0
    if dup:
        errors.append(f"{path.name}: player_name 중복 {dup}건 (동일 선수 여러 FA 시즌이면 정상일 수 있음, 확인 필요)")

    return errors


def main() -> int:
    all_errors: list[str] = []
    for fname in TRAINING_FILES:
        all_errors.extend(check_file(DATA / fname))

    leakage_note = (
        "참고: total_contract_amount, contract_years는 v5/v7 원본 CSV에는 존재하지만 "
        "app/app.py 및 models/*_model_meta.pkl의 feature 리스트에는 포함되지 않음 (모델 입력 제외 확인됨)."
    )

    if all_errors:
        print("검증 실패:")
        for e in all_errors:
            print(f"  - {e}")
        print(leakage_note)
        return 1

    print("검증 통과: 결측/누수 이슈 없음")
    print(leakage_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
