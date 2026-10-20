import numpy as np
import pandas as pd
 
KEYS = ["PLAYER_ID", "SEASON_ID"]
 
STAT_COLUMNS = ["PLAYER_AGE", "GP", "GS", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A",
                "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
                "BLK", "TOV", "PF", "PTS"]
 

PER_GAME_TOTALS = ["MIN", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
 
AWARDS_UNUSED = ["FIRST_NAME", "LAST_NAME", "TEAM", "MONTH", "WEEK", "CONFERENCE",
                 "TYPE", "SUBTYPE1", "SUBTYPE2", "SUBTYPE3"]
 
FIRST_SEASON_PREFIX = "2"   
MIN_SEASONS = 4             
 
 
def compile_awards(awards_raw: pd.DataFrame) -> pd.DataFrame:
    a = awards_raw.drop(columns=[c for c in AWARDS_UNUSED if c in awards_raw.columns])
    a = a.dropna(subset=["PERSON_ID"])
    a["PERSON_ID"] = a["PERSON_ID"].astype(int)
 
    team = a["ALL_NBA_TEAM_NUMBER"]
    is_all_nba = ~(team.isna() | team.isin(["", "(null)"]))
    a.loc[is_all_nba, "DESCRIPTION"] = "All_NBA_Team--" + team[is_all_nba].astype(str)
    a = a.drop(columns=["ALL_NBA_TEAM_NUMBER"])
 
    summary = a.pivot_table(index=["PERSON_ID", "SEASON"], columns="DESCRIPTION",
                            aggfunc="size", fill_value=0).reset_index()
    summary.columns.name = None
    return summary.rename(columns={"PERSON_ID": "PLAYER_ID", "SEASON": "SEASON_ID"})
 
 
def dedupe_traded_players(df: pd.DataFrame) -> pd.DataFrame:
    has_tot = df.groupby(KEYS)["TEAM_ABBREVIATION"].transform(lambda s: "TOT" in s.values)
    return df[~has_tot | (df["TEAM_ABBREVIATION"] == "TOT")]
 
 
def make_season_table(career_raw: pd.DataFrame, awards_raw: pd.DataFrame) -> pd.DataFrame:
    awards = compile_awards(awards_raw)
    award_columns = [c for c in awards.columns if c not in KEYS]
 
    df = career_raw.merge(awards, on=KEYS, how="left")
    if len(df) != len(career_raw): 
        raise ValueError("Le merge stats + récompenses a créé des doublons")
    df[award_columns] = df[award_columns].fillna(0)
 
    df = df.drop(columns=["LEAGUE_ID", "TEAM_ID"])
    df = df[df["SEASON_ID"].str.startswith(FIRST_SEASON_PREFIX)]
 
    df = dedupe_traded_players(df).drop(columns=["TEAM_ABBREVIATION"])
    if df.duplicated(KEYS).any():
        raise ValueError("Plusieurs lignes pour un même (joueur, saison) après dédoublonnage")
 
    n_seasons = df.groupby("PLAYER_ID")["SEASON_ID"].transform("count")
    df = df[n_seasons >= MIN_SEASONS].copy()
 
    for col in STAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    gp = df["GP"].replace(0, np.nan)
    for col in PER_GAME_TOTALS:
        df[f"{col}_PER_GAME"] = df[col] / gp
    df = df.drop(columns=PER_GAME_TOTALS)
 
    df["AWARD_COUNT"] = df[award_columns].sum(axis=1)
    df["HAS_AWARD"] = df["AWARD_COUNT"] > 0
 
    return df.reset_index(drop=True)
