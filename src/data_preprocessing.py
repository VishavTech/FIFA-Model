"""
data_preprocessing.py

This script is responsible for the Data Collection & Preprocessing phase of the Machine Learning pipeline.
Since this is a predictive project for learning purposes, the script:
1. Automatically creates the directory structure (data/raw and data/processed).
2. Generates highly realistic historical and current datasets for 48 World Cup 2026 teams, including:
   - Team Historical Stats (FIFA ranking, World Cup titles, historical win percentage, goals).
   - Player Statistics (Age, position, overall ratings, goals, caps, market value).
   - Manager Statistics (Experience, matches managed, win percentage, major trophies).
   - Historical Match Results (1,000 matches simulated with realistic strength-based outcomes).
3. Cleans the data, handles potential missing values, and aggregates player stats to the team level.
4. Merges all datasets into a unified team profile dataset.
5. Builds a symmetrized training dataset for the machine learning models.

Author: Fresh Graduate / Junior ML Engineer
Date: 2026-06-26
"""

import os
import pandas as pd
import numpy as np

def create_directory_structure():
    """
    Creates the project directories for data storage.
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'outputs'
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def generate_raw_datasets():
    """
    Generates realistic synthetic datasets for FIFA World Cup 2026.
    Includes 48 teams spanning all regional confederations.
    """
    # 48 Qualified Teams for FIFA World Cup 2026
    teams_data = [
        # Team, FIFA Rank, WC Titles, Historical Win %, Avg Goals Scored, Avg Goals Conceded, Base Squad Value (M Euros)
        ("Argentina", 1, 3, 0.72, 2.1, 0.8, 1050),
        ("France", 2, 2, 0.70, 2.2, 0.9, 1200),
        ("Spain", 3, 1, 0.68, 2.0, 0.8, 1000),
        ("England", 4, 1, 0.65, 2.1, 0.9, 1300),
        ("Brazil", 5, 5, 0.73, 2.3, 1.0, 1100),
        ("Belgium", 6, 0, 0.62, 1.9, 1.1, 650),
        ("Portugal", 7, 0, 0.64, 2.0, 1.0, 980),
        ("Netherlands", 8, 0, 0.61, 1.8, 1.1, 750),
        ("Italy", 9, 4, 0.58, 1.6, 0.9, 700),
        ("Colombia", 10, 0, 0.59, 1.7, 1.0, 320),
        ("Croatia", 11, 0, 0.57, 1.5, 1.2, 420),
        ("Germany", 12, 4, 0.60, 1.9, 1.2, 850),
        ("Morocco", 13, 0, 0.58, 1.4, 0.8, 380),
        ("Uruguay", 14, 2, 0.56, 1.6, 1.1, 480),
        ("USA", 15, 0, 0.52, 1.5, 1.2, 350),
        ("Japan", 16, 0, 0.55, 1.7, 1.1, 280),
        ("Senegal", 17, 0, 0.54, 1.4, 1.0, 290),
        ("Switzerland", 18, 0, 0.50, 1.4, 1.2, 260),
        ("Denmark", 19, 0, 0.51, 1.5, 1.1, 310),
        ("Iran", 20, 0, 0.49, 1.3, 0.9, 80),
        ("Mexico", 21, 0, 0.48, 1.3, 1.3, 220),
        ("South Korea", 22, 0, 0.50, 1.4, 1.1, 210),
        ("Australia", 23, 0, 0.47, 1.3, 1.2, 120),
        ("Sweden", 24, 0, 0.49, 1.4, 1.2, 240),
        ("Ukraine", 25, 0, 0.46, 1.3, 1.2, 200),
        ("Turkey", 26, 0, 0.45, 1.4, 1.3, 230),
        ("Austria", 27, 0, 0.46, 1.3, 1.2, 190),
        ("Ecuador", 28, 0, 0.44, 1.2, 1.1, 210),
        ("Poland", 29, 0, 0.43, 1.2, 1.3, 170),
        ("Nigeria", 30, 0, 0.46, 1.3, 1.1, 250),
        ("Egypt", 31, 0, 0.45, 1.2, 1.0, 140),
        ("Tunisia", 32, 0, 0.42, 1.1, 1.1, 95),
        ("Algeria", 33, 0, 0.44, 1.2, 1.0, 130),
        ("Cameroon", 34, 0, 0.41, 1.1, 1.2, 110),
        ("Canada", 35, 0, 0.45, 1.3, 1.3, 180),
        ("Costa Rica", 36, 0, 0.40, 1.1, 1.3, 60),
        ("Panama", 37, 0, 0.38, 1.0, 1.4, 45),
        ("Jamaica", 38, 0, 0.39, 1.1, 1.4, 85),
        ("Saudi Arabia", 39, 0, 0.40, 1.1, 1.2, 70),
        ("Qatar", 40, 0, 0.38, 1.1, 1.3, 50),
        ("Iraq", 41, 0, 0.37, 1.0, 1.2, 35),
        ("Uzbekistan", 42, 0, 0.36, 1.0, 1.1, 40),
        ("Ivory Coast", 43, 0, 0.45, 1.3, 1.1, 260),
        ("Ghana", 44, 0, 0.42, 1.2, 1.2, 150),
        ("Chile", 45, 0, 0.43, 1.2, 1.3, 110),
        ("Paraguay", 46, 0, 0.40, 1.0, 1.2, 95),
        ("Peru", 47, 0, 0.41, 1.0, 1.2, 80),
        ("New Zealand", 48, 0, 0.35, 0.9, 1.4, 30)
    ]
    
    # 1. Create Team Historical Stats Dataframe
    df_teams = pd.DataFrame(teams_data, columns=[
        'team', 'fifa_ranking', 'world_cup_titles', 'historical_win_percentage',
        'goals_scored_per_match', 'goals_conceded_per_match', 'squad_value_euro_million'
    ])
    df_teams.to_csv('data/raw/historical_team_stats.csv', index=False)
    print("Generated data/raw/historical_team_stats.csv")

    # 2. Generate Player Statistics
    # For each team, we generate 15 key players (3 Forwards, 5 Midfielders, 5 Defenders, 2 Goalkeepers)
    players_list = []
    np.random.seed(42) # Set seed for reproducibility
    
    for idx, row in df_teams.iterrows():
        team_name = row['team']
        base_rating = 85 - (row['fifa_ranking'] * 0.28) # Higher rank = higher rating
        base_rating = max(65, min(92, base_rating)) # Clamp ratings
        
        # Generator specs per team
        positions = ['FWD']*3 + ['MID']*5 + ['DEF']*5 + ['GK']*2
        for i, pos in enumerate(positions):
            player_name = f"{team_name[:3].upper()}_Player_{i+1}"
            age = int(np.random.normal(26.5, 3.8))
            age = max(18, min(39, age))
            
            # Overall rating has some random variance around the team's base rating
            rating = int(np.random.normal(base_rating, 2.5))
            rating = max(60, min(95, rating))
            
            # Caps correlate with age
            caps = int(max(0, (age - 20) * np.random.uniform(2, 7)))
            
            # Goals and Assists based on position
            if pos == 'FWD':
                goals = int(np.random.exponential(12)) if rating > 75 else int(np.random.exponential(5))
                assists = int(np.random.normal(4, 2))
            elif pos == 'MID':
                goals = int(np.random.exponential(4))
                assists = int(np.random.exponential(7)) if rating > 75 else int(np.random.exponential(3))
            elif pos == 'DEF':
                goals = int(np.random.exponential(1))
                assists = int(np.random.exponential(2))
            else: # GK
                goals = 0
                assists = 0
            
            goals = max(0, goals)
            assists = max(0, assists)
            
            # Market value depends heavily on rating and age (young + highly rated = expensive)
            age_factor = max(0.1, 1 - (max(0, age - 25) * 0.08)) # Peak value around 23-25
            val = (10 ** ((rating - 55) / 10)) * age_factor * np.random.uniform(0.8, 1.2)
            val = round(max(0.5, val), 2) # minimum 500k value
            
            players_list.append({
                'player_name': player_name,
                'team': team_name,
                'age': age,
                'position': pos,
                'overall_rating': rating,
                'goals_last_season': goals,
                'assists_last_season': assists,
                'minutes_played': int(np.random.normal(2200, 500)),
                'international_caps': caps,
                'market_value_euro_million': val
            })
            
    df_players = pd.DataFrame(players_list)
    df_players.to_csv('data/raw/player_stats.csv', index=False)
    print("Generated data/raw/player_stats.csv")

    # 3. Generate Manager Statistics
    managers_list = []
    for idx, row in df_teams.iterrows():
        team_name = row['team']
        # Stronger teams tend to have more experienced managers with higher win %
        rank_factor = (50 - row['fifa_ranking']) / 50 # 1.0 for rank 1 down to 0.0 for rank 48
        exp = int(np.random.normal(12 + rank_factor * 8, 4))
        exp = max(2, exp)
        
        matches = int(exp * np.random.normal(15, 3))
        win_pct = row['historical_win_percentage'] + np.random.normal(0, 0.03)
        win_pct = round(max(0.35, min(0.78, win_pct)), 2)
        
        trophies = int(max(0, np.random.normal(rank_factor * 5, 2)))
        
        managers_list.append({
            'manager_name': f"Coach_{team_name[:3].upper()}",
            'team': team_name,
            'years_of_experience': exp,
            'matches_managed': matches,
            'win_percentage': win_pct,
            'major_trophies': trophies,
            'intl_experience': 1 if np.random.random() > 0.3 else 0
        })
        
    df_managers = pd.DataFrame(managers_list)
    df_managers.to_csv('data/raw/manager_stats.csv', index=False)
    print("Generated data/raw/manager_stats.csv")

    # 4. Generate Historical Matches Data (for model training)
    # We will simulate 1,200 international matches between 2018 and 2026.
    matches_list = []
    team_names = df_teams['team'].tolist()
    team_lookup = df_teams.set_index('team').to_dict('index')
    
    match_id = 1
    for year in range(2018, 2027):
        # Number of matches in that year
        num_matches = 140
        for _ in range(num_matches):
            # Pick two random teams
            t1, t2 = np.random.choice(team_names, size=2, replace=False)
            
            # Calculate their relative strength to simulate a realistic score
            t1_stats = team_lookup[t1]
            t2_stats = team_lookup[t2]
            
            # Strength calculation based on FIFA ranking, historical goals, and squad value
            # Note: lower ranking is better!
            strength_t1 = (50 - t1_stats['fifa_ranking']) * 0.15 + (t1_stats['squad_value_euro_million'] ** 0.5) * 0.05 + t1_stats['goals_scored_per_match'] * 0.5
            strength_t2 = (50 - t2_stats['fifa_ranking']) * 0.15 + (t2_stats['squad_value_euro_million'] ** 0.5) * 0.05 + t2_stats['goals_scored_per_match'] * 0.5
            
            # Poisson lambda parameters for scoring goals
            lambda_t1 = max(0.5, 1.3 + (strength_t1 - strength_t2) * 0.25)
            lambda_t2 = max(0.5, 1.3 + (strength_t2 - strength_t1) * 0.25)
            
            # Generate random goals
            t1_goals = np.random.poisson(lambda_t1)
            t2_goals = np.random.poisson(lambda_t2)
            
            # Limit crazy scores
            t1_goals = min(7, t1_goals)
            t2_goals = min(7, t2_goals)
            
            matches_list.append({
                'match_id': match_id,
                'team_a': t1,
                'team_b': t2,
                'team_a_goals': t1_goals,
                'team_b_goals': t2_goals,
                'year': year,
                'tournament': np.random.choice(['World Cup', 'Friendly', 'Qualifiers', 'Continental Cup'], p=[0.1, 0.3, 0.4, 0.2])
            })
            match_id += 1
            
    df_matches = pd.DataFrame(matches_list)
    df_matches.to_csv('data/raw/historical_match_results.csv', index=False)
    print(f"Generated data/raw/historical_match_results.csv with {len(df_matches)} matches.")


def preprocess_and_merge_data():
    """
    Cleans datasets, aggregates players and manager data to the team level, 
    and merges everything into a master processed dataset.
    """
    print("\n--- Preprocessing & Merging ---")
    
    # Load raw datasets
    df_teams = pd.read_csv('data/raw/historical_team_stats.csv')
    df_players = pd.read_csv('data/raw/player_stats.csv')
    df_managers = pd.read_csv('data/raw/manager_stats.csv')
    df_matches = pd.read_csv('data/raw/historical_match_results.csv')

    # 1. Clean missing values (though synthetic, shows professional ML code)
    df_teams = df_teams.dropna()
    df_players = df_players.dropna()
    df_managers = df_managers.dropna()
    df_matches = df_matches.dropna()

    # 2. Aggregate player statistics per team
    print("Aggregating player stats per team...")
    
    # Compute overall aggregates
    player_agg = df_players.groupby('team').agg(
        avg_player_rating=('overall_rating', 'mean'),
        max_player_rating=('overall_rating', 'max'), # Represents "Star Player" rating (e.g. Messi, Mbappe)
        avg_player_age=('age', 'mean'),
        player_total_market_value=('market_value_euro_million', 'sum'),
        total_intl_caps=('international_caps', 'sum')
    ).reset_index()
    
    # Compute position-specific aggregates (Attack, Midfield, Defense)
    pos_agg = df_players.groupby(['team', 'position'])['overall_rating'].mean().unstack().reset_index()
    pos_agg = pos_agg.rename(columns={
        'FWD': 'attack_rating',
        'MID': 'midfield_rating',
        'DEF': 'defense_rating',
        'GK': 'goalkeeper_rating'
    })
    
    # Merge aggregations
    team_players_processed = pd.merge(player_agg, pos_agg, on='team')

    # 3. Rename manager stats to avoid columns clash
    manager_processed = df_managers.rename(columns={
        'years_of_experience': 'manager_experience_years',
        'matches_managed': 'manager_matches_managed',
        'win_percentage': 'manager_win_rate',
        'major_trophies': 'manager_trophies_won',
        'intl_experience': 'manager_intl_experience'
    }).drop(columns=['manager_name'])

    # 4. Merge all elements into a single Master Team Profile
    print("Merging Team, Player Aggregates and Manager Profiles...")
    master_team_profile = pd.merge(df_teams, team_players_processed, on='team', how='inner')
    master_team_profile = pd.merge(master_team_profile, manager_processed, on='team', how='inner')
    
    # Double-check features
    master_team_profile.to_csv('data/processed/team_stats_processed.csv', index=False)
    print(f"Saved master profile for {len(master_team_profile)} teams to 'data/processed/team_stats_processed.csv'")

    # 5. Build Symmetrized Match Dataset for Model Training
    # For every match between Team A and Team B:
    # Row 1: Team A - Team B differences -> Label: 1 if A won, 0 if Draw/B won
    # Row 2: Team B - Team A differences -> Label: 1 if B won, 0 if Draw/A won
    # This prevents the machine learning model from developing home/away bias.
    print("Generating symmetrized training dataset from historical matches...")
    
    # Build dictionary lookup for team profiles
    profiles_dict = master_team_profile.set_index('team').to_dict('index')
    
    training_rows = []
    
    for idx, match in df_matches.iterrows():
        tA = match['team_a']
        tB = match['team_b']
        goalsA = match['team_a_goals']
        goalsB = match['team_b_goals']
        
        # Guard in case team is not in master profile
        if tA not in profiles_dict or tB not in profiles_dict:
            continue
            
        statsA = profiles_dict[tA]
        statsB = profiles_dict[tB]
        
        # --- Side A vs Side B ---
        row_A = {
            'match_id': match['match_id'],
            'team': tA,
            'opponent': tB,
            # Target
            'match_won': 1 if goalsA > goalsB else 0,
            # Core team differences (A - B)
            'rank_diff': statsA['fifa_ranking'] - statsB['fifa_ranking'], # Note: negative rank diff is GOOD (e.g. Rank 1 vs Rank 10 = -9)
            'titles_diff': statsA['world_cup_titles'] - statsB['world_cup_titles'],
            'hist_win_pct_diff': statsA['historical_win_percentage'] - statsB['historical_win_percentage'],
            'goals_scored_diff': statsA['goals_scored_per_match'] - statsB['goals_scored_per_match'],
            'goals_conceded_diff': statsA['goals_conceded_per_match'] - statsB['goals_conceded_per_match'],
            'squad_value_diff': statsA['squad_value_euro_million'] - statsB['squad_value_euro_million'],
            # Player-level aggregated differences
            'avg_rating_diff': statsA['avg_player_rating'] - statsB['avg_player_rating'],
            'star_player_diff': statsA['max_player_rating'] - statsB['max_player_rating'],
            'avg_age_diff': statsA['avg_player_age'] - statsB['avg_player_age'],
            'attack_rating_diff': statsA['attack_rating'] - statsB['attack_rating'],
            'midfield_rating_diff': statsA['midfield_rating'] - statsB['midfield_rating'],
            'defense_rating_diff': statsA['defense_rating'] - statsB['defense_rating'],
            # Manager differences
            'manager_experience_diff': statsA['manager_experience_years'] - statsB['manager_experience_years'],
            'manager_win_rate_diff': statsA['manager_win_rate'] - statsB['manager_win_rate'],
            'manager_trophies_diff': statsA['manager_trophies_won'] - statsB['manager_trophies_won']
        }
        
        # --- Side B vs Side A ---
        row_B = {
            'match_id': match['match_id'],
            'team': tB,
            'opponent': tA,
            # Target
            'match_won': 1 if goalsB > goalsA else 0,
            # Core team differences (B - A)
            'rank_diff': statsB['fifa_ranking'] - statsA['fifa_ranking'],
            'titles_diff': statsB['world_cup_titles'] - statsA['world_cup_titles'],
            'hist_win_pct_diff': statsB['historical_win_percentage'] - statsA['historical_win_percentage'],
            'goals_scored_diff': statsB['goals_scored_per_match'] - statsA['goals_scored_per_match'],
            'goals_conceded_diff': statsB['goals_conceded_per_match'] - statsA['goals_conceded_per_match'],
            'squad_value_diff': statsB['squad_value_euro_million'] - statsA['squad_value_euro_million'],
            # Player-level aggregated differences
            'avg_rating_diff': statsB['avg_player_rating'] - statsA['avg_player_rating'],
            'star_player_diff': statsB['max_player_rating'] - statsA['max_player_rating'],
            'avg_age_diff': statsB['avg_player_age'] - statsA['avg_player_age'],
            'attack_rating_diff': statsB['attack_rating'] - statsA['attack_rating'],
            'midfield_rating_diff': statsB['midfield_rating'] - statsA['midfield_rating'],
            'defense_rating_diff': statsB['defense_rating'] - statsA['defense_rating'],
            # Manager differences
            'manager_experience_diff': statsB['manager_experience_years'] - statsA['manager_experience_years'],
            'manager_win_rate_diff': statsB['manager_win_rate'] - statsA['manager_win_rate'],
            'manager_trophies_diff': statsB['manager_trophies_won'] - statsA['manager_trophies_won']
        }
        
        training_rows.append(row_A)
        training_rows.append(row_B)

    df_training = pd.DataFrame(training_rows)
    df_training.to_csv('data/processed/match_training_data.csv', index=False)
    print(f"Saved symmetrized match training dataset with {len(df_training)} records to 'data/processed/match_training_data.csv'")

if __name__ == "__main__":
    create_directory_structure()
    generate_raw_datasets()
    preprocess_and_merge_data()
