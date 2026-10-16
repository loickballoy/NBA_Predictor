from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import playercareerstats, playerawards, playerdashboardbyclutch
from nba_api.stats.static import players

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_raw(player_id: int, fetch_fn, kind: str,retries=3):
    out = RAW_DIR / kind /f"{player_id}.parquet"
    if out.exists():
        print(f"Data for player {player_id} already exists.")
        return
    for attempt in range(retries):
        try:
            data = fetch_fn(player_id)
            data.to_parquet(out)
            print(f"Fetched data for player {player_id}.")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for player {player_id}: {e}")

def fetch_player_career_stats(player_id: int):
    return playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]

def fetch_player_awards(player_id: int):
    return playerawards.PlayerAwards(player_id=player_id).get_data_frames()[0]

def fetch_all():
    all_players = players.get_players()
    for player in all_players:
        player_id = player['id']
        fetch_raw(player_id, fetch_player_career_stats, "career_stats")
        fetch_raw(player_id, fetch_player_awards, "awards")

def load_all_career_stats():
    return pd.concat([pd.read_parquet(f) for f in RAW_DIR.glob("career_stats/*.parquet")], ignore_index=True)

def load_all_awards():
    return pd.concat([pd.read_parquet(f) for f in RAW_DIR.glob("awards/*.parquet")], ignore_index=True)

if __name__ == "__main__":
    fetch_all()
    df = load_all()
    print(df)