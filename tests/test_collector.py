import pandas as pd
import pytest

from collector.build_dataset import build_windows, export_splits, prepare
from collector.clean import compile_awards, dedupe_traded_players

OTHER_COLUMNS = [
    "GS", "PLAYER_AGE", "FGA", "FG3A", "FTA", "OREB", "FG_PCT", "FG3_PCT", "FT_PCT",
    "MIN_PER_GAME", "PTS_PER_GAME", "REB_PER_GAME", "DREB_PER_GAME", "AST_PER_GAME",
    "STL_PER_GAME", "BLK_PER_GAME", "TOV_PER_GAME", "PF_PER_GAME", "AWARD_COUNT",
]

def seasons(player_years: dict) -> pd.DataFrame:
    rows = []
    for player_id, years in player_years.items():
        for y in years:
            row = {"PLAYER_ID": player_id, "SEASON_ID": f"{y}-{str(y + 1)[2:]}", "GP": y - 1990}
            row.update({c:1.0 for c in OTHER_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows);

def test_windows_have_no_temporal_leak():
    w = build_windows(seasons({1: range(2005, 2013)}))
    assert len(w) == 5      # 8 seasons -> 5 windows of 3 + 1

    target = w["id_TARGET_YEAR"]

    assert (w["x_s1_GP"] + 1990 == target - 3).all()
    assert (w["x_s2_GP"] + 1990 == target - 2).all()
    assert (w["x_s3_GP"] + 1990 == target - 1).all()
    assert (w["y_GP"] + 1990 == target).all()

def test_windows_skip_missing_seasons():
    w = build_windows(seasons({1: [2005, 2006, 2007, 2009, 2010, 2011, 2012]}))
    assert w["id_TARGET_YEAR"].tolist() == [2012]

def test_splits_have_no_common_target_year(tmp_path):
    w = build_windows(seasons({1: range(2000, 2026), 2: range(2010, 2026)}))
    export_splits(w, val_from=2020, test_from=2023, out_dir=tmp_path)
 
    years = {name: set(pd.read_csv(tmp_path / f"{name}.csv")["id_TARGET_YEAR"])
             for name in ["train", "val", "test"]}
 
    assert years["train"].isdisjoint(years["val"])
    assert years["train"].isdisjoint(years["test"])
    assert years["val"].isdisjoint(years["test"])
    # Et dans le bon ordre : tout le train avant la val, toute la val avant le test.
    assert max(years["train"]) < min(years["val"])
    assert max(years["val"]) < min(years["test"])
    # Aucune fenêtre perdue ou dupliquée.
    assert sum(len(pd.read_csv(tmp_path / f"{n}.csv")) for n in years) == len(w)

def test_prepare_rejects_duplicate_player_season():
    df = seasons({1: [2010, 2011]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # 2010-11 en double
    with pytest.raises(ValueError):
        prepare(df)

def test_awards_one_row_even_if_name_spelled_differently():
    awards_raw = pd.DataFrame({
        "PERSON_ID": [2405, 2405],
        "FIRST_NAME": ["Amare", "Amar'e"],
        "LAST_NAME": ["Stoudemire", "Stoudemire"],
        "DESCRIPTION": ["NBA All-Star", "All-NBA"],
        "ALL_NBA_TEAM_NUMBER": [None, "1"],
        "SEASON": ["2004-05", "2004-05"],
    })
    summary = compile_awards(awards_raw)
    assert len(summary) == 1
    assert summary.loc[0, "NBA All-Star"] == 1
    assert summary.loc[0, "All_NBA_Team--1"] == 1
 
 
def test_traded_player_keeps_only_tot_row():
    df = pd.DataFrame({
        "PLAYER_ID": [7, 7, 7, 8],
        "SEASON_ID": ["2015-16"] * 4,
        "TEAM_ABBREVIATION": ["BOS", "LAL", "TOT", "MIA"],
    })
    out = dedupe_traded_players(df)
    assert out["TEAM_ABBREVIATION"].tolist() == ["TOT", "MIA"]
