from pathlib import Path
import pandas as pd
import time
import random

from nba_api.stats.endpoints import playercareerstats, playerawards
from nba_api.stats.static import players

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_DELAY = 2.0
MAX_DELAY = 60.0
REQUEST_PAUSE = 0.6

def fetch_raw(player_id: int, fetch_fn, kind: str,retries=3):
    out_dir = RAW_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True) 
    
    out = out_dir / f"{player_id}.parquet"
    if out.exists():
        print(f"Data for player {player_id} already exists.")
        return
    for attempt in range(retries):
        try:
            data = fetch_fn(player_id)
            break
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"[{kind}] player {player_id} : failed after {retries} tries"
                ) from e
            delay = min(BASE_DELAY * 2 ** attempt, MAX_DELAY)
            delay += random.uniform(0, delay / 2)
            print(f"[{kind}] player {player_id} : try {attempt + 1} / {retries} ")
            time.sleep(delay)
    data.to_parquet(out)
    print(f"Fetched data for player {player_id}.")
    time.sleep(REQUEST_PAUSE)

def fetch_player_career_stats(player_id: int):
    return playercareerstats.PlayerCareerStats(player_id=player_id, timeout=30).get_data_frames()[0]

def fetch_player_awards(player_id: int):
    return playerawards.PlayerAwards(player_id=player_id, timeout=30).get_data_frames()[0]

def fetch_all():
    all_players = players.get_players()
    failures = []
    for player in all_players:
        player_id = player['id']
        try:
            fetch_raw(player_id, fetch_player_career_stats, "career_stats")
        except RuntimeError as e:
            print(e)
            failures.append((player_id, "career_stats"))
        try:
            fetch_raw(player_id, fetch_player_awards, "awards")
        except RuntimeError as e:
            print(e)
            failures.append((player_id, "awards"))
    return failures


def load_all_career_stats():
    return pd.concat([pd.read_parquet(f) for f in RAW_DIR.glob("career_stats/*.parquet")], ignore_index=True)

def load_all_awards():
    return pd.concat([pd.read_parquet(f) for f in RAW_DIR.glob("awards/*.parquet")], ignore_index=True)

