# FIFA World Cup 2026 Winner Prediction using Machine Learning

An end-to-end Machine Learning pipeline and interactive web dashboard designed in **pure Python** to predict the champion of the **FIFA World Cup 2026**. This project aggregates team, player, and manager-level metrics, engineers relative strength difference features, trains several standard classification models, and executes a full **Monte Carlo Tournament Simulation** (1,000 runs) to identify winning probabilities.

This repository is structured as a **beginner-to-intermediate level portfolio showcase** for entry-level Machine Learning Engineer and Data Scientist candidates, emphasizing strong modular programming, mathematical rigor, and model interpretability over unnecessary web development complexity.

---

## 🚀 Key Features & Highlights

1. **Realistic Synthetic Data Generation**: Simulates 48 national squads (reflecting the official 2026 expansion), 720 individual players, and 48 manager profiles with high-fidelity attributes (FIFA Rank, overall rating, squad value, caps, coach win-rates).
2. **Symmetric Difference Feature Engineering**: Leverages "data symmetrization" (adding both $Team A - Team B$ and $Team B - Team A$ difference matrices) to avoid home/away bias. Standardizes continuous inputs using `StandardScaler`.
3. **Model Selection & Benchmarking**: Compares four distinct models: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Gradient Boosting Classifier using standard validation metrics (Accuracy, Precision, Recall, F1 Score) and 5-Fold Cross-Validation.
4. **Monte Carlo Bracket Simulation**: Models the actual 2026 48-team bracket format (12 Groups of 4, Round of 32 knockout qualification, wild card 3rd-place allocations) and runs 1,000 trials to calculate the exact winning probability of every team.
5. **Interactive Web Interface**: A clean, single-page Flask web application to browse model charts, look up tournament simulation leaderboards, and run head-to-head match predictions.

---

## 📂 Project Directory Structure

```text
FIFA_World_Cup_Prediction/
│
├── data/
│   ├── raw/                    # Original uncleaned datasets
│   │   ├── historical_team_stats.csv   # Base FIFA rankings & titles
│   │   ├── player_stats.csv            # Detailed player records
│   │   ├── manager_stats.csv           # Coach history & win rate
│   │   └── historical_match_results.csv# 1,200 international matches (2018-2026)
│   │
│   └── processed/              # Consolidated and clean data
│       ├── team_stats_processed.csv    # Final aggregated team profiles
│       └── match_training_data.csv     # Difference-engineered match rows
│
├── src/                        # Modular pipeline components
│   ├── data_preprocessing.py   # Handles synthetic generation, cleaning, and merging
│   ├── feature_engineering.py  # Calculates differences and applies standard scaling
│   ├── train_model.py          # Trains classifiers and records cross-validation
│   ├── predict.py              # Houses bracket drawing and Monte Carlo simulations
│   └── evaluation.py           # Plots Confusion Matrix, ROC curves, and importance
│
├── models/                     # Serialized binary artifacts
│   ├── scaler.joblib           # Fitted StandardScaler object
│   ├── best_model.joblib       # Best trained classifier (Random Forest)
│   └── best_model_info.json    # Metadata describing best model parameters
│
├── outputs/                    # Output logs, metrics, and figures
│   ├── model_comparison.json   # Benchmark statistics of all 4 models
│   ├── simulation_results.json # Leaderboard of 1,000 Monte Carlo runs
│   ├── confusion_matrix.png    # Validation Confusion Matrix image
│   ├── roc_curve.png           # Validation ROC-AUC curve image
│   └── feature_importance.png  # Variable importance plot
│
├── templates/                  # Frontend view templates
│   └── index.html              # Tailwind-styled Flask HTML dashboard
│
├── app.py                      # Flask backend and predictive REST APIs
├── main.py                     # Entry point to execute the full ML pipeline
├── requirements.txt            # Python dependencies
└── start_app.js                # Shell-bootstrap launcher for hosting servers
```

---

## 📊 Machine Learning Pipeline & Implementation

### 1. Data Collection & Preprocessing (`src/data_preprocessing.py`)
- We model **48 qualified countries** across all regional confederations.
- Generates **720 player profiles** (15 per squad) spanning different positions (GK, DEF, MID, FWD), with attributes representing market value, international caps, age, and skill overall ratings.
- Aggregates individual player metrics to the team level, extracting:
  - `avg_player_rating` and `max_player_rating` (Star player indicator)
  - Squad positional ratings: `attack_rating`, `midfield_rating`, `defense_rating`
  - Total market values and squad size metrics.
- Integrates coach records (`years_of_experience`, `win_percentage`, `major_trophies`).
- Generates **1,200 historical matches** (2018–2026) with goal distributions drawn from a Poisson process influenced by the relative strengths of the opposing squads.

### 2. Feature Selection & Engineering (`src/feature_engineering.py`)
Matches are represented by differences ($Team_A - Team_B$). Negative values indicate Team B holds the advantage.
The final model consumes **14 primary differences**:
- `rank_diff`, `titles_diff`, `hist_win_pct_diff`
- `goals_scored_diff`, `goals_conceded_diff`, `squad_value_diff`
- `avg_rating_diff`, `star_player_diff`, `attack_rating_diff`, `midfield_rating_diff`, `defense_rating_diff`
- `manager_experience_diff`, `manager_win_rate_diff`, `manager_trophies_diff`

Features are standardized using a `StandardScaler` fitted on the training set to prevent features with larger scales (like `squad_value_diff`) from distorting gradient updates:
$$z = \frac{x - \mu}{\sigma}$$

### 3. Training & Model Selection (`src/train_model.py`)
Data is divided into an **80/20 train/test holdout split** (stratified on the binary target `match_won`).
Four classifier architectures are evaluated:
1. **Logistic Regression**: Serves as a high-interpretability linear baseline.
2. **Decision Tree Classifier**: Examines non-linear, hierarchical feature cutoffs.
3. **Random Forest Classifier**: Bagging ensemble that mitigates variance.
4. **Gradient Boosting Classifier**: Boosting ensemble that reduces bias sequentially.

The models are compared on test set metrics (Accuracy, Precision, Recall, F1) and **5-Fold Cross-Validation Accuracy** to verify generalizability. The best-performing model (typically Random Forest or Gradient Boosting) is serialized to `models/best_model.joblib`.

### 4. Comprehensive Evaluation (`src/evaluation.py`)
Generates and caches three static figures:
- **Confusion Matrix**: Maps True Positives, False Positives, True Negatives, and False Negatives.
- **ROC Curve**: Plots True Positive Rate (Sensitivity) vs. False Positive Rate (1 - Specificity) at various classification thresholds.
- **Feature Importance**: Scores each variable's relative split-criterion contribution.

---

## 🔮 Tournament Simulation Engine (`src/predict.py`)

Under the **48-team 2026 World Cup format**, the tournament is simulated as follows:
1. **FIFA Group Draws**: Teams are distributed into 4 pots of 12 teams based on their FIFA ranking and split into **12 Groups of 4 (Groups A to L)**.
2. **Group Stage**: Each team plays 3 matches within their group. Match outcomes are resolved using the model's win/draw/loss probabilities:
   - Let $p$ be the model's winning probability for Team A.
   - We assign a realistic international draw rate $P_{draw} = 0.26$.
   - The remaining probability is allocated: $P_{win A} = 0.74 \times p$, $P_{win B} = 0.74 \times (1 - p)$.
   - standings are sorted by: Points (3 for win, 1 for draw), Goal Difference, Goals Scored, and FIFA Rank.
3. **Wildcard Allocation**: The **top 2 from each of the 12 groups (24 teams)** plus the **8 best 3rd-placed teams** (ranked by points, goal diff, goals) advance to the **Round of 32**.
4. **Knockout Stage**: Matches are played in a single-elimination bracket (Round of 32 $\rightarrow$ Round of 16 $\rightarrow$ Quarterfinals $\rightarrow$ Semifinals $\rightarrow$ Final). Draws are resolved using a weighted probability coin toss biased towards the stronger team.
5. **Monte Carlo Aggregation**: The simulator runs this entire tournament **1,000 times**. The overall winning probability of each country is its win frequency divided by 1,000.

---

## 🛠️ Installation & Setup Guide

Ensure you have **Python 3.8+** installed.

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/FIFA_World_Cup_Prediction.git
cd FIFA_World_Cup_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Machine Learning Pipeline
This script generates the raw data, preprocesses it, trains all 4 models, generates evaluation plots, and runs the 1,000 tournament simulations:
```bash
python main.py
```

### 4. Launch the Web Dashboard
```bash
python app.py
```
Open [http://localhost:3000](http://localhost:3000) in your web browser to view the interactive portfolio.

---

## 💡 Future Enhancements (For Interview Topics)
If asked by a recruiter how to scale or improve this project, you can discuss:
1. **Feature Additions**: Scraping real-time player injury statistics, team travel distances, altitude maps of host stadiums, and historical head-to-head records.
2. **Advanced Modeling**: Implementing XGBoost, LightGBM, or deep Neural Networks with Hyperparameter tuning using GridSearch or Optuna.
3. **Goal-Level Simulation**: Transitioning from binary win/loss classification to a double-Poisson or negative-binomial model predicting actual goal counts.
4. **Hype & Form tracking**: Utilizing Sentiment Analysis on sports news and Twitter data to capture a "Team Morale" coefficient.
