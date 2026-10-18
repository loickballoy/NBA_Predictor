from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "data" / "processed"

TOTALS_TO_PER_GAME = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB"]

FEATURES = [
    "GP", "GS",
    "MIN_PER_GAME", "PTS_PER_GAME", "REB_PER_GAME", "OREB_PER_GAME", "DREB_PER_GAME",
    "AST_PER_GAME", "STL_PER_GAME", "BLK_PER_GAME", "TOV_PER_GAME", "PF_PER_GAME",
    "FGA_PER_GAME", "FG_PCT", "FG3A_PER_GAME", "FG3_PCT", "FTA_PER_GAME", "FT_PCT",
    "AWARD_COUNT",
]

LAST_SEASON_ONLY = ["PLAYER_AGE"]

TARGETS = [
    "GP", "MIN_PER_GAME", "PTS_PER_GAME", "REB_PER_GAME", "AST_PER_GAME",
    "STL_PER_GAME", "BLK_PER_GAME", "TOV_PER_GAME",
]

KEYS = ["PLAYER_ID", "START_YEAR"]

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    for col in TOTALS_TO_PER_GAME + ["GP", "GS", "PLAYER_AGE"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    gp = d["GP"].replace(0, np.nan)
    for col in TOTALS_TO_PER_GAME:
        per_game = f"{col}_PER_GAME"
        if per_game not in d.columns and col in d.columns:
            d[per_game] = d[col] / gp

    if "AWARD_COUNT" not in d.columns:
        d["AWARD_COUNT"] = 0

    d["START_YEAR"] = d["SEASON_ID"].str[:4].astype(int)

    for col in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    missing = [c for c in FEATURES + LAST_SEASON_ONLY + TARGETS if c not in d.columns]
    if missing:
        raise KeyError(f"Colonnes absentes du df : {missing}")

    if d.duplicated(KEYS).any():
        raise ValueError("Plusieurs lignes pour un même joueur, saison")

    return d

def _season(d: pd.DataFrame, offset: int, cols: list, prefix: str) -> pd.DataFrame:
    s = d[KEYS + cols].copy()
    s["START_YEAR"] -= offset
    return s.rename(columns={c: prefix + c for c in cols})

def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    d = prepare(df)

    w = _season(d, 0, FEATURES, "x_s1_")
    w = w.merge(_season(d, 1, FEATURES, "x_s2_"), on=KEYS, how="inner")
    w = w.merge(_season(d, 2, FEATURES + LAST_SEASON_ONLY, "x_s3_"), on=KEYS, how="inner")
    w = w.merge(_season(d, 3, TARGETS, "y_"), on=KEYS, how="inner")

    w["TARGET_YEAR"] = w["START_YEAR"] + 3
    w = w.rename(columns={"PLAYER_ID": "id_PLAYER_ID", "TARGET_YEAR": "id_TARGET_YEAR"})
    w = w.drop(columns=["START_YEAR"])

    before = len(w)
    w = w.dropna()
    if len(w) < before:
        print(f" {before - len(w)} fenêtres retirées")

    id_cols = [c for c in w.columns if c.startswith("id_")]
    x_cols = [c for c in w.columns if c.startswith("x_")]
    y_cols = [c for c in w.columns if c.startswith("y_")]

    return w[id_cols + x_cols + y_cols].reset_index(drop=True)

def export_splits(w: pd.DataFrame, val_from: int = 2020, test_from: int = 2023, out_dir: Path = OUT_DIR) -> None:
    year = w["id_TARGET_YEAR"]
    splits = {
        "train": w[year < val_from],
        "val": w[(year >= val_from) & (year < test_from)],
        "test": w[year >= test_from]
    }

    empty = [name for name, part in splits.items() if part.empty()]
    if empty:
        raise ValueError(f"Empty Splits: {empty}, ajust val_from and test_from")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False, float_format="%.6g")
        print(f"{name:5s} : {len(part):6d} lignes -> {path}")
    n_x = sum(c.startswith("x_") for c in w.columns)
    n_y = sum(c.startswith("y_") for c in w.columns)
    print(f"{n_x} features, {n_y} cibles")