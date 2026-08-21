from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog
import pandas as pd

team =teams.find_team_by_abbreviation('SAS');
team_id = team['id']

tgl = teamgamelog.TeamGameLog(team_id=team_id, season_type_all_star='Playoffs')
df_regular_season = tgl.team_game_log.get_data_frame()

player = players.find_players_by_full_name('bilal')

for p in player:
    print(p['full_name'], p['id'])

