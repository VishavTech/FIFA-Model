"""
predict.py

This script implements Step 8 of the Machine Learning Pipeline.
It:
1. Loads the master processed team profiles.
2. Organizes the 48 qualified countries into 12 realistic balanced groups (Groups A to L) using seeding pots based on FIFA ranking.
3. Implements a complete Monte Carlo Simulation of the FIFA World Cup 2026.
4. For each tournament trial (e.g., 1,000 trials):
   - Simulates the Group Stage matches using win/draw/loss probabilities predicted by the ML model.
   - Computes group standings and ranks them based on points, goal difference, and FIFA rank.
   - Advances 32 teams: top 2 from each of the 12 groups (24 teams) and the 8 best 3rd-place teams.
   - Simulates the Knockout Stage (Round of 32, 16, Quarterfinals, Semifinals, and Final) to crown a Champion.
5. Calculates the aggregated probabilities of every team winning the World Cup.
6. Saves the simulation results in a JSON file for the dashboard.

Author: Fresh Graduate / Junior ML Engineer
Date: 2026-06-26
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

# Import feature extraction
from src.feature_engineering import FEATURE_COLUMNS, prepare_match_features_inference

def load_team_profiles():
    """
    Loads the master processed team profile dataset.
    """
    filepath = 'data/processed/team_stats_processed.csv'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed team stats not found. Please run data_preprocessing.py first.")
    return pd.read_csv(filepath)

def load_trained_model():
    """
    Loads the best trained classifier model.
    """
    filepath = 'models/best_model.joblib'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model not found at {filepath}. Please run train_model.py first.")
    return joblib.load(filepath)

def form_groups(df_teams):
    """
    Simulates the FIFA official pot-drawing system.
    Splits the 48 teams into 4 seeding pots of 12 teams based on their FIFA ranking.
    Dynamically creates 12 balanced groups (Group A to L).
    """
    # Sort teams by ranking
    df_sorted = df_teams.sort_values('fifa_ranking').reset_index(drop=True)
    
    # 4 Pots of 12 teams
    pot1 = df_sorted.iloc[0:12].to_dict('records')
    pot2 = df_sorted.iloc[12:24].to_dict('records')
    pot3 = df_sorted.iloc[24:36].to_dict('records')
    pot4 = df_sorted.iloc[36:48].to_dict('records')
    
    groups = {chr(65 + i): [] for i in range(12)} # Groups A to L
    
    # Fill groups systematically to ensure balance (snake/shifted pattern)
    for i in range(12):
        group_char = chr(65 + i)
        # 1 team from Pot 1
        groups[group_char].append(pot1[i])
        # 1 team from Pot 2 (shifted to mix)
        groups[group_char].append(pot2[(i + 3) % 12])
        # 1 team from Pot 3 (shifted to mix)
        groups[group_char].append(pot3[(i + 6) % 12])
        # 1 team from Pot 4 (shifted to mix)
        groups[group_char].append(pot4[(i + 9) % 12])
        
    return groups

def predict_match_probabilities(team_a, team_b, model, scaler_path='models/scaler.joblib'):
    """
    Runs the ML model to predict win, draw, and loss probabilities.
    Standardizes difference features and outputs: P(A wins), P(Draw), P(B wins).
    """
    # Calculate difference and scale
    X_diff = prepare_match_features_inference(team_a, team_b, scaler_path)
    
    # Get probability of Team A winning from model
    prob_a_won = model.predict_proba(X_diff)[0][1] # P(A wins)
    
    # Map raw model probability smoothly into Win, Draw, Loss probabilities.
    # We assign a historical constant draw rate of 26% for international group stages.
    # The remaining 74% is divided based on the model's relative strength probability.
    p_draw = 0.26
    p_win_a = 0.74 * prob_a_won
    p_win_b = 0.74 * (1.0 - prob_a_won)
    
    # Normalize just in case of float tolerances
    total_p = p_win_a + p_draw + p_win_b
    p_win_a /= total_p
    p_draw /= total_p
    p_win_b /= total_p
    
    return p_win_a, p_draw, p_win_b

def simulate_group_stage(groups, model):
    """
    Simulates round-robin group play (3 matches per team in each of the 12 groups).
    Returns final standings and points for each group.
    """
    standings = {}
    
    for group_name, teams in groups.items():
        # Initialize standings table for this group
        group_standings = {t['team']: {'points': 0, 'goals_scored': 0, 'goals_conceded': 0, 'goal_diff': 0, 'fifa_ranking': t['fifa_ranking']} for t in teams}
        
        # Play round robin (6 matches in a group of 4)
        n = len(teams)
        for i in range(n):
            for j in range(i + 1, n):
                t1 = teams[i]
                t2 = teams[j]
                
                # Predict probabilities
                pw1, pdraw, pw2 = predict_match_probabilities(t1, t2, model)
                
                # Roll a dice for the outcome
                outcome = np.random.choice(['win1', 'draw', 'win2'], p=[pw1, pdraw, pw2])
                
                # Simulate realistic goals based on relative ratings
                lambda1 = max(0.5, 1.2 + (t1['avg_player_rating'] - t2['avg_player_rating']) * 0.1)
                lambda2 = max(0.5, 1.2 + (t2['avg_player_rating'] - t1['avg_player_rating']) * 0.1)
                
                if outcome == 'win1':
                    g1 = int(np.random.poisson(lambda1)) + 1
                    g2 = int(np.random.poisson(lambda2))
                    if g2 >= g1: g2 = g1 - 1
                    # Update standings
                    group_standings[t1['team']]['points'] += 3
                elif outcome == 'win2':
                    g1 = int(np.random.poisson(lambda1))
                    g2 = int(np.random.poisson(lambda2)) + 1
                    if g1 >= g2: g1 = g2 - 1
                    # Update standings
                    group_standings[t2['team']]['points'] += 3
                else: # Draw
                    g1 = int(np.random.poisson(lambda1))
                    g2 = g1 # Must be equal
                    # Update standings
                    group_standings[t1['team']]['points'] += 1
                    group_standings[t2['team']]['points'] += 1
                    
                # Update goals
                group_standings[t1['team']]['goals_scored'] += g1
                group_standings[t1['team']]['goals_conceded'] += g2
                group_standings[t1['team']]['goal_diff'] += (g1 - g2)
                
                group_standings[t2['team']]['goals_scored'] += g2
                group_standings[t2['team']]['goals_conceded'] += g1
                group_standings[t2['team']]['goal_diff'] += (g2 - g1)
                
        # Sort group standings: Points, Goal Diff, Goals Scored, FIFA Ranking (lower ranking is better)
        sorted_standings = sorted(
            group_standings.items(),
            key=lambda x: (x[1]['points'], x[1]['goal_diff'], x[1]['goals_scored'], -x[1]['fifa_ranking']),
            reverse=True
        )
        standings[group_name] = sorted_standings
        
    return standings

def advance_teams_to_knockout(group_standings, df_teams):
    """
    Selects 32 advancing teams under the new 48-team FIFA World Cup format:
    - Top 2 teams from each of the 12 groups (24 teams)
    - 8 Best 3rd-placed teams (8 teams)
    """
    advancing_teams_names = []
    third_placed_teams = []
    
    # 1. Take top 2 from every group
    for group_name, standings in group_standings.items():
        advancing_teams_names.append(standings[0][0]) # 1st place
        advancing_teams_names.append(standings[1][0]) # 2nd place
        
        # Collect 3rd placed team metadata for ranking
        t3_name, t3_stats = standings[2]
        third_placed_teams.append({
            'team': t3_name,
            'points': t3_stats['points'],
            'goal_diff': t3_stats['goal_diff'],
            'goals_scored': t3_stats['goals_scored'],
            'fifa_ranking': t3_stats['fifa_ranking']
        })
        
    # 2. Sort third-placed teams: Points, Goal Diff, Goals Scored, FIFA Rank
    third_placed_sorted = sorted(
        third_placed_teams,
        key=lambda x: (x['points'], x['goal_diff'], x['goals_scored'], -x['fifa_ranking']),
        reverse=True
    )
    
    # Take the top 8
    for i in range(8):
        advancing_teams_names.append(third_placed_sorted[i]['team'])
        
    # Load complete team stats objects for these advancing team names
    profiles_lookup = df_teams.set_index('team').to_dict('index')
    advancing_profiles = []
    for name in advancing_teams_names:
        stats = profiles_lookup[name]
        stats['team'] = name
        advancing_profiles.append(stats)
        
    return advancing_profiles

def simulate_knockout_match(team_a, team_b, model):
    """
    Simulates a knockout match. Must yield a winner.
    If drawn, resolves based on model win ratio bias (simulating extra time/penalties).
    """
    pw1, pdraw, pw2 = predict_match_probabilities(team_a, team_b, model)
    
    # Roll dice for normal 90 mins
    outcome = np.random.choice(['win1', 'draw', 'win2'], p=[pw1, pdraw, pw2])
    
    if outcome == 'win1':
        return team_a
    elif outcome == 'win2':
        return team_b
    else:
        # Resolve draw (extra-time/penalties)
        # Weighted slightly towards the stronger team (proportional to pw1 and pw2)
        p_resolve_a = pw1 / (pw1 + pw2)
        p_resolve_b = pw2 / (pw1 + pw2)
        winner = np.random.choice([team_a, team_b], p=[p_resolve_a, p_resolve_b])
        return winner

def simulate_knockout_stage(advancing_teams, model):
    """
    Simulates the knockout bracket from Round of 32 to Final.
    """
    # Round of 32
    # Simple brackets: Seed 1 vs Seed 32, Seed 2 vs Seed 31...
    # (To be structurally clean and robust, we sort by FIFA ranking)
    sorted_bracket = sorted(advancing_teams, key=lambda x: x['fifa_ranking'])
    
    current_round = sorted_bracket
    round_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    
    for round_name in round_names:
        next_round = []
        n = len(current_round)
        # Pair them: index i plays index n-1-i (classic seed pairing)
        for i in range(n // 2):
            t1 = current_round[i]
            t2 = current_round[n - 1 - i]
            winner = simulate_knockout_match(t1, t2, model)
            next_round.append(winner)
        current_round = next_round
        
    # Final winner is crowned
    return current_round[0]

def run_monte_carlo_simulation(num_simulations=1000):
    """
    Runs num_simulations tournament runs and aggregates winning frequencies.
    """
    print(f"\n--- Running Monte Carlo Simulation ({num_simulations} Runs) ---")
    
    df_teams = load_team_profiles()
    model = load_trained_model()
    
    # Form Pots and Groups once
    groups = form_groups(df_teams)
    
    # Champion tracker dictionary
    champions_count = {t['team']: 0 for t in df_teams.to_dict('records')}
    
    for run in range(1, num_simulations + 1):
        if run % 200 == 0:
            print(f" -> Completed {run}/{num_simulations} tournament simulations...")
            
        # 1. Group Stage
        group_standings = simulate_group_stage(groups, model)
        
        # 2. Extract Advancing
        advancing = advance_teams_to_knockout(group_standings, df_teams)
        
        # 3. Simulate bracket and return champion team dict
        champion = simulate_knockout_stage(advancing, model)
        
        # 4. Increment count
        champions_count[champion['team']] += 1
        
    # Calculate probabilities
    results = []
    for team_name, count in champions_count.items():
        prob = count / num_simulations
        results.append({
            'team': team_name,
            'simulation_wins': count,
            'winning_probability': round(prob, 4)
        })
        
    # Sort descending
    results_sorted = sorted(results, key=lambda x: x['winning_probability'], reverse=True)
    
    # Save simulation results to outputs/
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/simulation_results.json', 'w') as f:
        json.dump(results_sorted, f, indent=4)
        
    print("Simulation completed successfully. Results written to 'outputs/simulation_results.json'")
    
    # Print Top 5
    print("\n--- Predicted Top 5 FIFA 2026 World Cup Winners ---")
    for idx, item in enumerate(results_sorted[:5]):
        print(f"{idx+1}. {item['team']} — {item['winning_probability'] * 100:.2f}% probability")
    print(f"Predicted Champion: {results_sorted[0]['team']}")
    
    return results_sorted

if __name__ == "__main__":
    run_monte_carlo_simulation()
