"""
evaluation.py

This script implements Step 7 of the Machine Learning Pipeline.
It:
1. Loads the best-performing model and standard features.
2. Generates evaluation figures:
   - Confusion Matrix (how many True Positives, False Positives, etc.)
   - Receiver Operating Characteristic (ROC) Curve & Area Under Curve (AUC)
   - Feature Importance Plot (which factors are most influential in predicting winners)
3. Configures Matplotlib to run in headless 'Agg' mode to prevent rendering errors.
4. Saves all static figures to the 'outputs/' folder for display on the dashboard.

Author: Fresh Graduate / Junior ML Engineer
Date: 2026-06-26
"""

import os
import json
import numpy as np
import pandas as pd

# Set Matplotlib backend to 'Agg' for headless, non-interactive environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib

# Import feature lists and processors
from src.feature_engineering import load_training_data, engineer_and_scale_features, FEATURE_COLUMNS

def generate_evaluation_artifacts():
    print("\n--- Generating Model Evaluation Plots & Reports ---")
    
    # 1. Load data
    df = load_training_data()
    y = df['match_won'].values
    X_scaled, _ = engineer_and_scale_features(df, is_training=False)
    
    # Split the exact same test set
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 2. Load model
    if not os.path.exists('models/best_model.joblib'):
        raise FileNotFoundError("Best model not found. Please train models first using train_model.py.")
        
    model = joblib.load('models/best_model.joblib')
    
    with open('models/best_model_info.json', 'r') as f:
        model_info = json.load(f)
    best_model_name = model_info['model_name']
    
    # 3. Generate Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Create text report
    report = classification_report(y_test, y_pred)
    with open('outputs/classification_report.txt', 'w') as f:
        f.write(report)
    print("Saved classification report to 'outputs/classification_report.txt'")

    # --- Plot 1: Confusion Matrix ---
    print("Generating Confusion Matrix Plot...")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Draw/Loss', 'Win'],
           yticklabels=['Draw/Loss', 'Win'],
           title=f'Confusion Matrix\n({best_model_name})',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    
    # --- Plot 2: ROC Curve ---
    if y_prob is not None:
        print("Generating ROC Curve Plot...")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic\n({best_model_name})')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig('outputs/roc_curve.png', dpi=150)
        plt.close()

    # --- Plot 3: Feature Importance ---
    # Supported by Random Forest / Gradient Boosting / Decision Tree
    # For Logistic Regression, we can use absolute coefficients as proxy!
    print("Generating Feature Importance Plot...")
    importance = []
    features_ordered = FEATURE_COLUMNS
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        
    if len(importance) > 0:
        # Sort feature importances in descending order
        indices = np.argsort(importance)
        
        plt.figure(figsize=(8, 6))
        plt.title(f"Feature Importance Map\n({best_model_name})", fontsize=12, fontweight='bold')
        
        # Display horizontal bars
        bars = plt.barh(range(len(indices)), [importance[i] for i in indices], color='#3b82f6', align='center')
        
        # Style grid and spines
        plt.yticks(range(len(indices)), [features_ordered[i] for i in indices])
        plt.xlabel('Relative Importance Value')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Hide top/right border lines for clean aesthetic
        for spine in ['top', 'right']:
            plt.gca().spines[spine].set_visible(False)
            
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=150)
        plt.close()
        print("Feature importance plot successfully created.")
    else:
        print("Feature importances could not be calculated for this model class.")

if __name__ == "__main__":
    generate_evaluation_artifacts()
