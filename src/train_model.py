"""
train_model.py

This script implements Step 6 of the Machine Learning Pipeline.
It:
1. Loads the symmetrized training dataset.
2. Scales features using the module from feature_engineering.py.
3. Splits data into train and test sets (80% / 20%).
4. Trains multiple baseline and ensemble classifiers:
   - Logistic Regression (Linear baseline)
   - Decision Tree Classifier (Non-linear baseline)
   - Random Forest Classifier (Robust bagging ensemble)
   - Gradient Boosting Classifier (Powerful boosting ensemble)
5. Compares model performance across metrics: Accuracy, Precision, Recall, and F1 Score.
6. Chooses and saves the best-performing model as a serialized joblib file.
7. Saves the comparison metrics in a JSON file for web dashboard display.

Author: Fresh Graduate / Junior ML Engineer
Date: 2026-06-26
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Import feature lists and processors
from src.feature_engineering import load_training_data, engineer_and_scale_features

def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    """
    Fits the model, calculates test metrics, and runs 5-fold cross-validation.
    Returns a dictionary of metrics.
    """
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Perform 5-fold cross-validation on full dataset for robustness
    cv_scores = cross_val_score(model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'cv_mean_accuracy': round(float(cv_mean), 4),
        'cv_std': round(float(cv_std), 4)
    }

def run_model_training_and_comparison():
    """
    The main execution pipeline for training, comparing, and saving models.
    """
    print("--- Starting Model Training & Comparison ---")
    
    # Load dataset
    df = load_training_data()
    y = df['match_won'].values
    
    # Engineer and scale features
    X_scaled, _ = engineer_and_scale_features(df, is_training=True)
    
    # Split into train and test sets (Stratified to maintain class distributions)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Define models to compare
    models_to_evaluate = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, random_state=42)
    }
    
    comparison_results = {}
    fitted_models = {}
    
    # Train and evaluate each model
    for name, model in models_to_evaluate.items():
        print(f"Evaluating model: {name}...")
        metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test)
        comparison_results[name] = metrics
        fitted_models[name] = model
        print(f" -> Accuracy: {metrics['accuracy']} | CV Mean Accuracy: {metrics['cv_mean_accuracy']} | F1-Score: {metrics['f1_score']}")
        
    # Write comparison results to outputs/ for web dashboard retrieval
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/model_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print("Saved model comparison results to 'outputs/model_comparison.json'")
    
    # Choose the best model based on Testing Accuracy
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]['accuracy'])
    best_model = fitted_models[best_model_name]
    
    print(f"\n🏆 Best Performing Model: {best_model_name} (Accuracy: {comparison_results[best_model_name]['accuracy']})")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/best_model.joblib'
    joblib.dump(best_model, model_save_path)
    print(f"Saved best model successfully to '{model_save_path}'")
    
    # Also save metadata about the best model
    best_model_info = {
        'model_name': best_model_name,
        'metrics': comparison_results[best_model_name]
    }
    with open('models/best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=4)

if __name__ == "__main__":
    run_model_training_and_comparison()
