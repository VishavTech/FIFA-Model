"""
app.py

This is the main web dashboard application, written in pure Python using Flask.
In accordance with the prompt requirements, the frontend is simple, minimal,
and acts strictly as an interface to:
1. Load the trained machine learning model.
2. View performance benchmarks and static evaluation plots.
3. Show the aggregated Monte Carlo tournament winning probabilities.
4. Run interactive head-to-head match predictions.

This app is lightweight, fully responsive, and has no JavaScript/React framework overhead.

Author: Fresh Graduate / Junior ML Engineer
Date: 2026-06-26
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
import joblib

# Import custom prediction modules
from src.predict import predict_match_probabilities, form_groups, simulate_group_stage, advance_teams_to_knockout, simulate_knockout_stage

app = Flask(__name__, template_folder='templates')
PORT = 3000

# Helper to load processed team statistics
def load_processed_teams():
    filepath = 'data/processed/team_stats_processed.csv'
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

# Route: Serve the primary dashboard web interface
@app.get('/')
def index():
    return render_template('index.html')

# Route: Serve generated evaluation figures directly from outputs/
@app.get('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory('outputs', filename)

# API Route: Fetch model benchmarking results
@app.get('/api/model_comparison')
def get_model_comparison():
    filepath = 'outputs/model_comparison.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Model comparison stats not found. Please run main.py."}), 404

# API Route: Fetch best model info
@app.get('/api/best_model')
def get_best_model():
    filepath = 'models/best_model_info.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    # Fallback to model_comparison if best_model_info doesn't exist
    comp_path = 'outputs/model_comparison.json'
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comp_data = json.load(f)
        if comp_data:
            # Find model with best accuracy
            best_name = max(comp_data.keys(), key=lambda k: comp_data[k].get('accuracy', 0))
            return jsonify({
                "model_name": best_name,
                "metrics": comp_data[best_name]
            })
    return jsonify({"error": "Best model info not found. Please run main.py."}), 404

# API Route: Fetch Monte Carlo tournament winning probabilities
@app.get('/api/simulation_results')
def get_simulation_results():
    filepath = 'outputs/simulation_results.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Simulation results not found. Please run main.py."}), 404

# API Route: Get alphabetical list of participating teams
@app.get('/api/teams')
def get_teams():
    df = load_processed_teams()
    if df is not None:
        teams = sorted(df['team'].tolist())
        return jsonify(teams)
    return jsonify([])

# API Route: Interactive head-to-head match predictor
@app.get('/api/predict_match')
def predict_matchup():
    team_a = request.args.get('team_a')
    team_b = request.args.get('team_b')
    
    if not team_a or not team_b:
        return jsonify({"error": "Both team_a and team_b query parameters are required."}), 400
        
    df_teams = load_processed_teams()
    if df_teams is None:
        return jsonify({"error": "Processed team profiles not found. Please run main.py."}), 500
        
    # Find matching team stats
    stats_a = df_teams[df_teams['team'] == team_a].to_dict('records')
    stats_b = df_teams[df_teams['team'] == team_b].to_dict('records')
    
    if not stats_a or not stats_b:
        return jsonify({"error": "One or both teams not found in participating profiles."}), 404
        
    stats_a = stats_a[0]
    stats_b = stats_b[0]
    
    # Load trained model
    if not os.path.exists('models/best_model.joblib'):
        return jsonify({"error": "Trained ML model not found. Please run training first."}), 500
    model = joblib.load('models/best_model.joblib')
    
    # Run prediction probabilities
    pw1, pdraw, pw2 = predict_match_probabilities(stats_a, stats_b, model)
    
    return jsonify({
        'team_a': team_a,
        'team_b': team_b,
        'p_win_a': round(float(pw1), 4),
        'p_draw': round(float(pdraw), 4),
        'p_win_b': round(float(pw2), 4),
        'team_a_stats': stats_a,
        'team_b_stats': stats_b
    })

# API Route: Run a new Monte Carlo simulation of 1,000 matches on-demand
@app.get('/api/run_simulation')
def trigger_simulation():
    df_teams = load_processed_teams()
    if df_teams is None:
        return jsonify({"error": "Processed team profiles not found."}), 500
        
    if not os.path.exists('models/best_model.joblib'):
        return jsonify({"error": "Trained ML model not found."}), 500
    model = joblib.load('models/best_model.joblib')
    
    # Perform 1,000 tournament simulations
    num_simulations = 1000
    groups = form_groups(df_teams)
    champions_count = {t['team']: 0 for t in df_teams.to_dict('records')}
    
    for _ in range(num_simulations):
        # Group stage
        group_standings = simulate_group_stage(groups, model)
        # Advance top 32
        advancing = advance_teams_to_knockout(group_standings, df_teams)
        # Play knockout bracket
        champion = simulate_knockout_stage(advancing, model)
        champions_count[champion['team']] += 1
        
    # Compile results
    results = []
    for team_name, count in champions_count.items():
        prob = count / num_simulations
        results.append({
            'team': team_name,
            'simulation_wins': count,
            'winning_probability': round(prob, 4)
        })
        
    results_sorted = sorted(results, key=lambda x: x['winning_probability'], reverse=True)
    
    # Cache to outputs folder
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/simulation_results.json', 'w') as f:
        json.dump(results_sorted, f, indent=4)
        
    return jsonify({
        "status": "success",
        "message": f"Successfully completed {num_simulations} simulated tournaments.",
        "top_winner": results_sorted[0]['team']
    })

if __name__ == "__main__":
    # Bind to port 3000 and allow connections from outside the container
    print(f"Booting Flask web app on http://0.0.0.0:{PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False)
