"""노트북 생성에 쓰는 공용 조각 — 셀 헬퍼와 모든 노트북의 첫 셀.

원본 설명은 gen_notebooks.py에 있다.

(원래 설명)
현재 파이프라인(학습셋 v10 · 모델 v9) 기준 노트북 생성.

notebooks/에 있던 3개는 7월 v4/v5 시절 기록이고 fa_contracts_v3를 읽는다.
그 뒤로 수집 누락 복구(TS-001)·생년 교정(TS-002)·스타성 재수집(TS-004)·
국가대표 보강(TS-010)이 들어가면서 README 수치와 노트북이 어긋났다.
구버전은 notebooks/archive/로 옮기고 현재 파이프라인을 다시 만든다.

기존 scripts/gen_clean_notebook.py와 같은 방식(셀을 파이썬에서 조립)이다.
노트북을 직접 편집하면 출력·실행 카운트가 섞여 diff가 지저분해진다.

읽기 전용이다. models/와 data/에는 쓰지 않는다.

출력:
  notebooks/01_data_check.ipynb
  notebooks/02_preprocessing.ipynb
  notebooks/03_eda.ipynb
  notebooks/04_model_train.ipynb

실행: .venv/bin/python -m scripts.gen_notebooks
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("notebooks")


def md(text: str) -> dict:
    lines = text.strip().split("\n")
    return {
        "cell_type": "markdown", "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


def code(text: str) -> dict:
    lines = text.strip().split("\n")
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


def write(name: str, cells: list[dict]) -> None:
    # nbformat 4.5부터 셀마다 id가 필요하다. 없으면 실행할 때마다 경고가 뜬다.
    for index, cell in enumerate(cells):
        cell["id"] = f"{name.split('_')[0]}-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = OUT_DIR / name
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {path}  ({len(cells)}셀)")


# 모든 노트북 첫 셀. 프로젝트 루트에서 열든 notebooks/에서 열든 경로가 맞게 한다.
SETUP = """
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

DATA = ROOT / "data"
CHARTS = ROOT / "output" / "charts" / "notebook"
CHARTS.mkdir(parents=True, exist_ok=True)

# 순서 주의: seaborn의 set_style이 font.family를 sans-serif로 되돌린다.
# 폰트를 먼저 잡고 스타일을 나중에 걸면 한글이 전부 네모로 깨진다.
sns.set_style("whitegrid")

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("NanumGothic", "AppleGothic", "Malgun Gothic", "Noto Sans CJK KR"):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    else:
        print("경고: 한글 폰트를 못 찾았다. 라벨이 네모로 나올 수 있다.")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

print("루트:", ROOT)
print("폰트:", plt.rcParams["font.family"])
"""


