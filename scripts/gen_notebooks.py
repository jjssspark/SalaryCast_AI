"""현재 파이프라인(학습셋 v10 · 모델 v9) 기준 노트북 생성.

notebooks/에 있던 3개는 7월 v4/v5 시절 기록이고 fa_contracts_v3를 읽는다.
그 뒤로 수집 누락 복구(TS-001)·생년 교정(TS-002)·스타성 재수집(TS-004)·
국가대표 보강(TS-010)이 들어가면서 README 수치와 노트북이 어긋났다.
구버전은 notebooks/archive/로 옮기고 현재 파이프라인을 다시 만든다.

기존 scripts/gen_clean_notebook.py와 같은 방식(셀을 파이썬에서 조립)이다.
노트북을 직접 편집하면 출력·실행 카운트가 섞여 diff가 지저분해진다.

셀 본문은 길어서 따로 뒀다.
  scripts/notebook_common.py       셀 헬퍼와 공통 첫 셀
  scripts/notebook_cells_data.py   01 데이터 확인 · 02 전처리
  scripts/notebook_cells_model.py  03 EDA · 04 모델 학습

노트북은 데이터를 읽기만 한다. models/와 data/에는 쓰지 않는다.

실행:
  .venv/bin/python -m scripts.gen_notebooks
  .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""

from __future__ import annotations

from scripts.notebook_cells_data import notebook_01, notebook_02
from scripts.notebook_cells_model import notebook_03, notebook_04
from scripts.notebook_common import OUT_DIR, write


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("노트북 생성")
    write("01_data_check.ipynb", notebook_01())
    write("02_preprocessing.ipynb", notebook_02())
    write("03_eda.ipynb", notebook_03())
    write("04_model_train.ipynb", notebook_04())
    print("\n실행해서 출력을 채우려면:")
    print("  .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb")


if __name__ == "__main__":
    main()
