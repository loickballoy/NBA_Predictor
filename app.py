import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

NBA_model = joblib.load('My_NBA_predictor.joblib')

def predict_nba_winner(team_home, team_away, game_date):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'team_code_home': [team_home],
        'team_code_away': [team_away],
        'game_date': [game_date]
    })

    input_data['game_date'] = pd.to_datetime(input_data['game_date'])
    input_data['team_code_home'] = input_data['team_home'].astype('category').cat.codes
    input_data['team_code_away'] = input_data['team_away'].astype('category').cat.codes
    input_data['day_code'] = input_data['game_date'].dt.dayofweek

    predictors = ['team_code_home', 'team_code_away', 'day_code']
    
    # Make prediction
    prediction = NBA_model.predict(input_data[predictors])
    
    # Return the predicted winner
    return 'Home Team Wins' if prediction[0] == 1 else 'Away Team Wins'

predict_nba_winner("DEN", "MIA", "2024-12-25")
    
