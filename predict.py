import joblib
import numpy as np

# Load trained models and encoders
clf_outcome = joblib.load("match_outcome_model.pkl")
clf_score = joblib.load("match_score_model.pkl")
le_home = joblib.load("home_team_encoder.pkl")
le_away = joblib.load("away_team_encoder.pkl")

# Get user input
home_team = input("Enter Home Team: ")
away_team = input("Enter Away Team: ")

# Encode teams
if home_team not in le_home.classes_ or away_team not in le_away.classes_:
    print("Error: Team names not found in dataset.")
    exit()

home_encoded = le_home.transform([home_team])[0]
away_encoded = le_away.transform([away_team])[0]

# Predict outcome
outcome_pred = clf_outcome.predict([[home_encoded, away_encoded]])[0]
outcome_label = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[outcome_pred]

# Predict score
score_pred = clf_score.predict([[home_encoded, away_encoded]])[0]

# Output results
print(f"Predicted Outcome: {outcome_label}")
print(f"Predicted Score: {score_pred[0]} - {score_pred[1]}")
