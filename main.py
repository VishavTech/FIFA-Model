"""
main.py

This is the primary orchestrator script for the "FIFA World Cup 2026 Winner Prediction" project.
It runs the entire Machine Learning Pipeline sequentially:
1. Data Generation and Cleaning (data_preprocessing.py)
2. Feature Selection and Standardization (feature_engineering.py)
3. Model Training and Benchmarking (train_model.py)
4. Deep Performance Evaluation and Plotting (evaluation.py)
5. Tournament Monte Carlo Prediction (predict.py)

Running this file prepares all serialized models, validation curves, and simulation stand-by profiles for the Web UI.

Author: Fresh Graduate / Junior ML Engineer
Date: 2026-06-26
"""

import os
import sys
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FIFA Predictor API is running"}

def run_pipeline():
    print("==============================================================")
    print("⚽ FIFA WORLD CUP 2026 MACHINE LEARNING PIPELINE STARTING ⚽")
    print("==============================================================\n")

    # Step 1 & 2 & 3: Run data generator, cleaner, and consolidator
    print("🚀 [Step 1-3] Running Data Preprocessing and Cleansing...")
    from src.data_preprocessing import create_directory_structure, generate_raw_datasets, preprocess_and_merge_data
    create_directory_structure()
    generate_raw_datasets()
    preprocess_and_merge_data()

    # Step 4: Run Feature Engineering
    print("\n⚡ [Step 4] Extracting and Normalizing Features...")
    from src.feature_engineering import load_training_data, engineer_and_scale_features
    df_train = load_training_data()
    X_scaled, _ = engineer_and_scale_features(df_train, is_training=True)
    print(f"Features ready. Shape of training matrix: {X_scaled.shape}")

    # Step 5 & 6: Train Models and Compare
    print("\n🧠 [Step 5-6] Training Classifiers and Performing Model Selection...")
    from src.train_model import run_model_training_and_comparison
    run_model_training_and_comparison()

    # Step 7: Model Evaluation
    print("\n📊 [Step 7] Generating Evaluation Graphics & Performance Report...")
    from src.evaluation import generate_evaluation_artifacts
    generate_evaluation_artifacts()

    # Step 8: World Cup Tournament Simulation
    print("\n🔮 [Step 8] Executing World Cup 2026 Monte Carlo Simulations (1,000 Runs)...")
    from src.predict import run_monte_carlo_simulation
    run_monte_carlo_simulation(num_simulations=1000)

    print("\n==============================================================")
    print("✅ MACHINE LEARNING PIPELINE SUCCESSFULLY RUN & CONCLUDED!")
    print("All serialized models, plots, and data files are cached.")
    print("==============================================================")

if __name__ == "__main__":
    run_pipeline()
