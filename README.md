# LoL Championship Player Performance Prediction

This project analyzes player performance data from the **2024 League of Legends World Championship** to predict win rates based on in-game statistics. By modeling the 'Win rate' as the target variable, the project aims to uncover which gameplay strategies correlate with higher success rates.

## Dataset
[**2024 League of Legends World Championship**](https://www.kaggle.com/datasets/anmatngu/2024-lol-championship-player-stats-and-swiss-stage)

The dataset includes detailed player statistics, such as:
- **KDA (Kill/Death/Assist) Ratio**
- **Vision Score per Minute (VSPM)**
- **Wards Placed per Minute (WPM)**
- **Gold Differential at 15 Minutes (GD@15)**
- ... etc.

Each row represents a player’s cumulative performance metrics throughout the championship, offering insights into individual impact and strategic effectiveness.

## `Notebook/notebook.ipynb`

### 1. Exploratory Data Analysis (EDA)
- **Data Cleaning**: Removed unnecessary columns and refined specific numerical features.
- **Feature Analysis**: Visualized numerical and categorical feature distributions using histograms, bar plots, and correlation heatmaps.
- **Feature Importance**: Assessed feature relevance with correlation analysis and `mutual_info_regression`.
- **Data Preparation**: Split dataset using k-fold cross-validation and encode categorical features.

### 2. Model Training and Tuning
Trained and tuned several models with parameter optimization:
- **Linear Regression**: parameter `alpha`.
- **Decision Tree**: parameters `max_depth` and `min_samples_leaf`.
- **Random Forest**: parameters `n_estimators`, `max_depth`, and `min_samples_leaf`.
- **XGBoost**: parameters `eta`, `max_depth`, and `min_child_weight`.

### 3. Model Selection
Evaluated models using RMSE as the metric. The best model’s name and parameter settings are stored in `best_model_params.json`.
- The RMSE of the test dataset:
```
0.1450606188503998
```
