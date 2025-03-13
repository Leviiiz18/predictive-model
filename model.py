import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset (Replace with your actual dataset path)
df = pd.read_csv("D:\\cric\\PremierLeague1.csv") 

# Ensure dataset has required columns
required_columns = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Dataset must contain 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', and 'FTR' columns.")

# Encode team names
le_home = LabelEncoder()
le_away = LabelEncoder()
df["HomeTeam"] = le_home.fit_transform(df["HomeTeam"])
df["AwayTeam"] = le_away.fit_transform(df["AwayTeam"])

# Define features and labels
X = df[["HomeTeam", "AwayTeam"]]
y_outcome = df["FTR"]  # Match outcome (H = Home win, D = Draw, A = Away win)
y_score = df[["FTHG", "FTAG"]]  # Score (Home goals, Away goals)

# Train-test split
X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(X, y_outcome, test_size=0.2, random_state=42)
X_train_score, X_test_score, y_score_train, y_score_test = train_test_split(X, y_score, test_size=0.2, random_state=42)

# Train RandomForest models
clf_outcome = RandomForestClassifier(n_estimators=100, random_state=42)
clf_outcome.fit(X_train, y_outcome_train)

clf_score = RandomForestClassifier(n_estimators=100, random_state=42)
clf_score.fit(X_train_score, y_score_train)

# Save models and label encoders
joblib.dump(clf_outcome, "match_outcome_model.pkl")
joblib.dump(clf_score, "match_score_model.pkl")
joblib.dump(le_home, "home_team_encoder.pkl")
joblib.dump(le_away, "away_team_encoder.pkl")

print("Model training complete. Models saved.")
