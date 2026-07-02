"""
feature_engineering.py

This script implements Step 4 and 5 of the Machine Learning Pipeline.
It:
1. Selects the most predictive features for match outcome prediction.
2. Fits and saves a StandardScaler to normalize the continuous numerical features.
3. Offers a transformation pipeline that can be reused for inference/predictions.

Author: Fresh Graduate / Junior ML Engineer
Date: 2026-06-26
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Definitive list of features to be utilized in our models
FEATURE_COLUMNS = [
    'rank_diff',
    'titles_diff',
    'hist_win_pct_diff',
    'goals_scored_diff',
    'goals_conceded_diff',
    'squad_value_diff',
    'avg_rating_diff',
    'star_player_diff',
    'attack_rating_diff',
    'midfield_rating_diff',
    'defense_rating_diff',
    'manager_experience_diff',
    'manager_win_rate_diff',
    'manager_trophies_diff'
]

def load_training_data():
    """
    Loads the merged match dataset from the processed folder.
    """
    filepath = 'data/processed/match_training_data.csv'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed training data not found at {filepath}. Please run data_preprocessing.py first.")
    return pd.read_csv(filepath)

def engineer_and_scale_features(df, is_training=True, scaler_path='models/scaler.joblib'):
    """
    Extracts relevant features and applies StandardScaler.
    Saves the fitted scaler if is_training=True.
    Loads existing scaler if is_training=False.
    """
    X = df[FEATURE_COLUMNS].copy()
    
    # Handle any potential infinite or NaN values that might occur
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        # Save scaler for online inference
        joblib.dump(scaler, scaler_path)
        print(f"StandardScaler fitted and saved successfully to '{scaler_path}'")
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. You must fit a scaler during training first.")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        
    return X_scaled, scaler

def prepare_match_features_inference(team_a_stats, team_b_stats, scaler_path='models/scaler.joblib'):
    """
    Given statistics of two teams, computes the relative difference features
    and scales them using the saved StandardScaler.
    Returns a 2D numpy array suitable for model prediction.
    """
    # Calculate differences (Team A - Team B)
    diff_stats = {
        'rank_diff': team_a_stats['fifa_ranking'] - team_b_stats['fifa_ranking'],
        'titles_diff': team_a_stats['world_cup_titles'] - team_b_stats['world_cup_titles'],
        'hist_win_pct_diff': team_a_stats['historical_win_percentage'] - team_b_stats['historical_win_percentage'],
        'goals_scored_diff': team_a_stats['goals_scored_per_match'] - team_b_stats['goals_scored_per_match'],
        'goals_conceded_diff': team_a_stats['goals_conceded_per_match'] - team_b_stats['goals_conceded_per_match'],
        'squad_value_diff': team_a_stats['squad_value_euro_million'] - team_b_stats['squad_value_euro_million'],
        'avg_rating_diff': team_a_stats['avg_player_rating'] - team_b_stats['avg_player_rating'],
        'star_player_diff': team_a_stats['max_player_rating'] - team_b_stats['max_player_rating'],
        'attack_rating_diff': team_a_stats['attack_rating'] - team_b_stats['attack_rating'],
        'midfield_rating_diff': team_a_stats['midfield_rating'] - team_b_stats['midfield_rating'],
        'defense_rating_diff': team_a_stats['defense_rating'] - team_b_stats['defense_rating'],
        'manager_experience_diff': team_a_stats['manager_experience_years'] - team_b_stats['manager_experience_years'],
        'manager_win_rate_diff': team_a_stats['manager_win_rate'] - team_b_stats['manager_win_rate'],
        'manager_trophies_diff': team_a_stats['manager_trophies_won'] - team_b_stats['manager_trophies_won']
    }
    
    # Convert to DataFrame matching the training feature columns exactly
    df_diff = pd.DataFrame([diff_stats])[FEATURE_COLUMNS]
    
    # Load scaler and transform
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(df_diff)
    
    return X_scaled

if __name__ == "__main__":
    df = load_training_data()
    X_scaled, scaler = engineer_and_scale_features(df, is_training=True)
    print(f"Successfully processed {len(X_scaled)} records with {len(FEATURE_COLUMNS)} features.")
