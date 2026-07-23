import pandas as pd

from src.team_offers import team_offers


def _teams_df():
    return pd.DataFrame({
        "team_name": ["A팀", "B팀"],
        "win_now_score": [1.0, 0.0],
        "cap_space_score": [0.5, 0.5],
    })


def _pn_df():
    return pd.DataFrame({
        "position": ["OF", "OF"],
        "team_name": ["A팀", "B팀"],
        "need_score": [1.0, 0.0],
    })


def test_team_offers_computes_offer_and_sorts_descending():
    result = team_offers(10.0, "OF", _teams_df(), _pn_df())

    a_offer = 10.0 * (1 + 1.0 * 0.20 + 1.0 * 0.15 + 0.5 * 0.10)
    b_offer = 10.0 * (1 + 0.0 * 0.20 + 0.0 * 0.15 + 0.5 * 0.10)

    assert result.iloc[0]["team_name"] == "A팀"
    assert result.iloc[0]["offer"] == round(a_offer, 2)
    assert result.iloc[1]["offer"] == round(b_offer, 2)
    assert result.iloc[0]["offer"] > result.iloc[1]["offer"]


def test_team_offers_defaults_missing_need_score_to_half():
    pn_df = pd.DataFrame({
        "position": ["SS"],  # no matching rows for "OF" position lookup below
        "team_name": ["A팀"],
        "need_score": [1.0],
    })
    result = team_offers(10.0, "OF", _teams_df(), pn_df)

    a_row = result[result["team_name"] == "A팀"].iloc[0]
    expected = 10.0 * (1 + 0.5 * 0.20 + 1.0 * 0.15 + 0.5 * 0.10)
    assert a_row["offer"] == round(expected, 2)
